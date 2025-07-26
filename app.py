"""
Aplicación web con Flask. Agente de IA experto basado en un documento PDF.
Implementa tono adaptativo, calificación de satisfacción y dashboard analítico con login.
"""

from dotenv import load_dotenv
load_dotenv()
import os
import markdown
import fitz
import json
from datetime import datetime
from flask import Flask, request, render_template, session, redirect, url_for, flash
from flask_session import Session
import google.generativeai as genai
import pandas as pd
import plotly.express as px

def extract_pdf_text(pdf_path):
    """
    Lee un archivo PDF y devuelve todo su texto como una sola cadena.
    """
    if not os.path.exists(pdf_path):
        return "Error: No se encontró el archivo PDF en la ruta especificada."
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        return f"Error al leer el archivo PDF: {e}"

# --- 1. Configuración de la aplicación Flask ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "una-clave-secreta-muy-segura-y-dificil-de-adivinar"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# --- 2. Configuración del modelo Gemini y carga de datos ---
COMPANY_DATA_PDF = "empresa_data.pdf"
COMPANY_KNOWLEDGE = extract_pdf_text(COMPANY_DATA_PDF)
# Cargar credenciales de administrador desde variables de entorno
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password')

# --- LÍNEA DE DEPURACIÓN TEMPORAL ---
# Esta línea imprimirá en tu terminal las credenciales que el programa está leyendo.
print(f"--- DEBUG: Usuario leído='{ADMIN_USERNAME}', Contraseña leída='{ADMIN_PASSWORD}' ---")


try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró la variable de entorno GEMINI_API_KEY.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Error al configurar Gemini: {e}")
    model = None

MAX_HISTORY = 5

# --- RUTAS DE LA APLICACIÓN ---

@app.route("/")
def home():
    """
    Renderiza la página de inicio y limpia el historial para una nueva conversación.
    """
    session["history"] = []
    return render_template("index.html", history=session.get("history", []))

# --- RUTAS DE AUTENTICACIÓN Y DASHBOARD ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Muestra el formulario de login y procesa las credenciales.
    """
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            flash('Inicio de sesión exitoso.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Credenciales incorrectas. Por favor, intenta de nuevo.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """
    Cierra la sesión del administrador.
    """
    session.pop('logged_in', None)
    flash('Has cerrado la sesión.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    """
    Muestra el dashboard solo si el administrador ha iniciado sesión.
    """
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        with open("chat_logs.jsonl", "r", encoding="utf-8") as f:
            log_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        return render_template('dashboard.html', error="No se ha generado ningún log todavía.")

    if not log_data:
        return render_template('dashboard.html', error="El archivo de logs está vacío.")
    
    df = pd.DataFrame(log_data)

    avg_rating = df['rating'].mean()
    rating_counts = df['rating'].value_counts().reset_index()
    rating_counts.columns = ['rating', 'count']
    fig_ratings = px.pie(rating_counts, values='count', names='rating', title='Distribución de Calificaciones')
    graph_ratings_html = fig_ratings.to_html(full_html=False)

    sentiments = [turn['sentiment'] for conv in df['conversation'] for turn in conv]
    sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_sentiments = px.bar(sentiment_counts, x='sentiment', y='count', title='Frecuencia de Sentimientos', color='sentiment')
    graph_sentiments_html = fig_sentiments.to_html(full_html=False)

    total_conversations = len(df)
    df['turn_count'] = df['conversation'].apply(len)
    avg_turns = df['turn_count'].mean()

    return render_template('dashboard.html',
                           total_conversations=total_conversations,
                           avg_rating=f"{avg_rating:.2f}",
                           avg_turns=f"{avg_turns:.2f}",
                           graph_ratings_html=graph_ratings_html,
                           graph_sentiments_html=graph_sentiments_html)

# --- RUTAS DE CHAT ---

@app.route('/rate', methods=['POST'])
def rate():
    """
    Recibe la calificación, guarda la conversación completa y finaliza la sesión.
    """
    rating = request.form.get('rating')
    history = session.get('history', [])

    if history and rating:
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session.sid,
                "rating": int(rating),
                "conversation": history
            }
            with open("chat_logs.jsonl", "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            flash("¡Muchas gracias por tu calificación!", "success")
        except Exception as e:
            print(f"Error al escribir el log de calificación: {e}")
            flash("Ocurrió un error al guardar tu calificación.", "danger")
    
    session.clear()
    return redirect(url_for('home'))

@app.route("/predict", methods=["POST"])
def predict():
    """
    Procesa la entrada del usuario, obtiene una respuesta del modelo y la devuelve.
    """
    if not model or "Error" in COMPANY_KNOWLEDGE:
        error_msg = "El modelo de IA no está configurado o no se pudo leer el PDF. Revisa la consola."
        return render_template("index.html", error=error_msg, history=session.get("history", []))

    prompt = request.form.get("prompt")
    if not prompt:
        return render_template("index.html", error="Por favor, ingresa un texto válido.", history=session.get("history", []))

    history = session.get("history", [])
    
    sentiment = "neutro"
    try:
        sentiment_prompt = f'Analiza el sentimiento del siguiente texto. Responde únicamente con una de estas tres palabras: positivo, negativo, neutro. Texto: "{prompt}"'
        sentiment_response = model.generate_content(sentiment_prompt)
        detected_sentiment = sentiment_response.text.strip().lower()
        if detected_sentiment in ['positivo', 'negativo', 'neutro']:
            sentiment = detected_sentiment
    except Exception as e:
        print(f"Error al analizar el sentimiento: {e}")

    chat_history_context = ""
    for item in history[-MAX_HISTORY:]:
        chat_history_context += f"Usuario: {item['prompt']}\nModelo: {item['response_raw']}\n"
    
    specialized_prompt = f"""
    Eres un asistente de IA conversacional y experto para la empresa "INGE LEAN S.A.S.".
    
    **Análisis de la Conversación Actual:**
    - Sentimiento del último mensaje del usuario: **{sentiment}**
    - Historial de la conversación:
    {chat_history_context}

    **TUS INSTRUCCIONES DE COMPORTAMIENTO:**
    1.  **ADAPTA TU TONO:** Ajusta tu estilo de respuesta según el sentimiento del usuario.
        - Si el sentimiento es **negativo**, tu tono debe ser especialmente empático, comprensivo y servicial. Pide disculpas si la situación lo amerita.
        - Si el sentimiento es **positivo**, responde de manera más amigable y entusiasta.
        - Si el sentimiento es **neutro**, mantén un tono profesional, claro y directo.
    2.  **USA EL CONTEXTO:** Utiliza el historial de la conversación para entender preguntas de seguimiento y dar respuestas coherentes.
    3.  **RESPETA TU ÚNICA FUENTE DE VERDAD:** Tu respuesta final DEBE basarse exclusivamente en el documento de conocimiento proporcionado a continuación. Si la respuesta no se encuentra en el texto, responde cortésmente: "Lo siento, no tengo información sobre ese tema". No inventes información bajo ninguna circunstancia.

    --- INICIO DEL DOCUMENTO DE CONOCIMIENTO ---
    {COMPANY_KNOWLEDGE}
    --- FIN DEL DOCUMENTO DE CONOCIMIENTO ---

    Ahora, responde a la siguiente pregunta del usuario aplicando estrictamente tus instrucciones de comportamiento:
    Usuario: {prompt}
    """

    try:
        response = model.generate_content(specialized_prompt)
        response_text = response.text
        response_html = markdown.markdown(response_text)

        history.append({
            "prompt": prompt,
            "response_raw": response_text,
            "response_html": response_html,
            "sentiment": sentiment 
        })
        session["history"] = history
        return render_template("index.html", history=history)

    except Exception as e:
        error_message = f"Ocurrió un error al contactar al modelo de IA: {e}"
        return render_template("index.html", error=error_message, history=history)

if __name__ == "__main__":
    app.run(debug=True)
