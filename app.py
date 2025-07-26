"""
Aplicación web con Flask. Agente de IA experto basado en un documento PDF.
Implementa registro de usuario, dashboard, login y envío de correo.
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
from flask_mail import Mail, Message
import google.generativeai as genai
import pandas as pd
import plotly.express as px

def extract_pdf_text(pdf_path):
    """
    Lee un archivo PDF y devuelve una tupla (texto, error).
    """
    if not os.path.exists(pdf_path):
        return None, f"Error: No se encontró el archivo PDF en {pdf_path}"
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text, None
    except Exception as e:
        return None, f"Error al leer el archivo PDF: {e}"

# --- 1. Configuración de la aplicación Flask ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "una-clave-secreta-muy-segura-y-dificil-de-adivinar"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# --- Configuración de Flask-Mail ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
mail = Mail(app)

# --- 2. Configuración del modelo y carga de datos ---
COMPANY_KNOWLEDGE, knowledge_error = extract_pdf_text("empresa_data.pdf")

if knowledge_error: print(knowledge_error)

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password')

try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: raise ValueError("No se encontró la variable de entorno GEMINI_API_KEY.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Error al configurar Gemini: {e}")
    model = None

MAX_HISTORY = 5

# --- RUTAS DE LA APLICACIÓN ---

@app.route("/", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_data = {
            "full_name": request.form.get('full_name'), "cedula": request.form.get('cedula'),
            "age": request.form.get('age'), "email": request.form.get('email'),
            "phone": request.form.get('phone'), "city": request.form.get('city'),
            "is_client": request.form.get('is_client') == 'yes'
        }
        if not all(user_data.values()):
            flash("Todos los campos son obligatorios.", "danger")
            return render_template("register.html")
        if not user_data["cedula"].isdigit() or not user_data["phone"].isdigit():
            flash("La cédula y el teléfono solo deben contener números.", "danger")
            return render_template("register.html")
        session['user_data'] = user_data
        session["history"] = []
        return redirect(url_for('chat'))
    return render_template("register.html")

@app.route("/chat")
def chat():
    if 'user_data' not in session:
        return redirect(url_for('register'))
    return render_template("index.html", history=session.get("history", []))

# --- RUTAS DE AUTENTICACIÓN Y DASHBOARD ---
@app.route('/login', methods=['GET', 'POST'])
def login():
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
    session.pop('logged_in', None)
    flash('Has cerrado la sesión.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
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
    if 'user_info' in df.columns:
        df_users = pd.json_normalize(df['user_info'])
        df = pd.concat([df.drop('user_info', axis=1), df_users], axis=1)
    avg_rating = df['rating'].mean()
    rating_counts = df['rating'].value_counts().reset_index()
    rating_counts.columns = ['rating', 'count']
    fig_ratings = px.pie(rating_counts, values='count', names='rating', title='Distribución de Calificaciones')
    graph_ratings_html = fig_ratings.to_html(full_html=False)
    sentiments = [turn.get('sentiment', 'desconocido') for conv in df['conversation'] for turn in conv]
    sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_sentiments = px.bar(sentiment_counts, x='sentiment', y='count', title='Frecuencia de Sentimientos', color='sentiment')
    graph_sentiments_html = fig_sentiments.to_html(full_html=False)
    total_conversations = len(df)
    df['turn_count'] = df['conversation'].apply(len)
    avg_turns = df['turn_count'].mean()
    conversations_data = df.to_dict(orient='records')
    return render_template('dashboard.html',
                           total_conversations=total_conversations,
                           avg_rating=f"{avg_rating:.2f}",
                           avg_turns=f"{avg_turns:.2f}",
                           graph_ratings_html=graph_ratings_html,
                           graph_sentiments_html=graph_sentiments_html,
                           conversations_data=conversations_data)

# --- RUTAS DE CHAT ---
@app.route('/rate', methods=['POST'])
def rate():
    rating = request.form.get('rating')
    history = session.get('history', [])
    user_data = session.get('user_data', {})
    if history and rating and user_data:
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(), "session_id": session.sid,
                "user_info": user_data, "rating": int(rating), "conversation": history
            }
            with open("chat_logs.jsonl", "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            try:
                email_html = "<h3>Gracias por usar nuestro asistente.</h3>"
                email_html += f"<p><b>Nombre:</b> {user_data.get('full_name')}</p>"
                email_html += f"<p><b>Calificación:</b> {'⭐' * int(rating)}</p>"
                email_html += "<h4>Transcripción de la Conversación:</h4><hr>"
                for item in history:
                    email_html += f"<p><b>Tú:</b> {item['prompt']}</p>"
                    email_html += f"<div><b>Asistente:</b> {item['response_html']}</div><br>"
                
                msg = Message(
                    subject="Transcripción de tu chat con el Asistente INGE LEAN",
                    # --- CORRECCIÓN: Se añade el remitente del correo ---
                    sender=app.config['MAIL_USERNAME'],
                    recipients=[user_data.get('email')], 
                    html=email_html
                )
                mail.send(msg)
                flash("¡Gracias por tu calificación! Te hemos enviado una copia del chat a tu correo.", "success")
            except Exception as e:
                print(f"Error al enviar el correo: {e}")
                flash("¡Gracias por tu calificación! No pudimos enviar la copia del chat.", "warning")
        except Exception as e:
            print(f"Error al escribir el log de calificación: {e}")
            flash("Ocurrió un error al guardar tu calificación.", "danger")
    session.clear()
    return redirect(url_for('register'))

@app.route("/predict", methods=["POST"])
def predict():
    if 'user_data' not in session:
        return redirect(url_for('register'))
    if not COMPANY_KNOWLEDGE:
        return render_template("index.html", error="La base de conocimiento no está disponible.", history=session.get("history", []))
    prompt = request.form.get("prompt")
    if not prompt:
        return render_template("index.html", error="Por favor, ingresa un texto válido.", history=session.get("history", []))
    history = session.get("history", [])
    sentiment = "neutro"
    try:
        sentiment_prompt = f'Analiza el sentimiento. Responde solo con: positivo, negativo, neutro. Texto: "{prompt}"'
        sentiment_response = model.generate_content(sentiment_prompt)
        detected_sentiment = sentiment_response.text.strip().lower()
        if detected_sentiment in ['positivo', 'negativo', 'neutro']:
            sentiment = detected_sentiment
    except Exception as e:
        print(f"Error al analizar sentimiento: {e}")
    chat_history_context = "".join(f"Usuario: {item['prompt']}\nModelo: {item['response_raw']}\n" for item in history[-MAX_HISTORY:])
    
    specialized_prompt = f"""
    Eres un asistente de IA conversacional y experto para la empresa "INGE LEAN S.A.S.".
    
    **Análisis de la Conversación Actual:**
    - Sentimiento del último mensaje del usuario: **{sentiment}**
    - Historial de la conversación:
    {chat_history_context}

    **TUS INSTRUCCIONES DE COMPORTAMIENTO:**
    1.  **ADAPTA TU TONO:** Ajusta tu estilo de respuesta según el sentimiento del usuario.
        - Si el sentimiento es **negativo**, tu tono debe ser especialmente empático, comprensivo y servicial.
        - Si el sentimiento es **positivo**, responde de manera más amigable y entusiasta.
        - Si el sentimiento es **neutro**, mantén un tono profesional, claro y directo.
    2.  **SÉ PROACTIVO Y ÚTIL:** Si la respuesta directa a la pregunta del usuario no se encuentra en el texto, no te limites a decir "no tengo la información". Intenta ofrecer información relacionada que sí esté en el documento y que pueda ser de ayuda. Por ejemplo, si preguntan por una ciudad donde no tienes servicio, puedes mencionar las ciudades donde sí lo tienes.
    3.  **RESPETA TU ÚNICA FUENTE DE VERDAD:** Tu respuesta final DEBE basarse exclusivamente en el documento de conocimiento. Nunca inventes información. Si no tienes ninguna información relacionada que ofrecer, entonces sí responde cortésmente que no tienes información sobre ese tema.

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
        history.append({ "prompt": prompt, "response_raw": response_text, "response_html": response_html, "sentiment": sentiment })
        session["history"] = history
        return render_template("index.html", history=history)
    except Exception as e:
        return render_template("index.html", error=f"Error al contactar al modelo: {e}", history=history)

if __name__ == "__main__":
    app.run(debug=True)
