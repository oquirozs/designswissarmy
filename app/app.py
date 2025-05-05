from flask import Flask, render_template, request, send_file, redirect, url_for, session, g, jsonify
from flask_babel import Babel, Locale, gettext as _
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import io
import qrcode
from gtts import gTTS
import cv2
#import base64
import requests
from sklearn.cluster import KMeans
import torch
from torchvision import transforms
import warnings
import paypalrestsdk
from config import Config
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config.from_object(Config)

# Configuración de Babel para internacionalización
babel = Babel(app)

def get_locale():
    # Verificar si el idioma está en la sesión
    if 'language' in session:
        return session['language']
    return request.accept_languages.best_match(app.config['LANGUAGES'])

#@babel.localeselector
babel.init_app(app, locale_selector=get_locale)
#def get_locale():
    # Verificar si el idioma está en la sesión
#    if 'language' in session:
#        return session['language']
#    return request.accept_languages.best_match(app.config['LANGUAGES'])

@app.before_request
def before_request():
    g.locale = get_locale()

@app.route('/set_language/<language>')
def set_language(language):
    if language in app.config['LANGUAGES']:
        session['language'] = language
    return redirect(request.referrer or url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

# Configuración de PayPal
paypalrestsdk.configure({
    "mode": app.config['PAYPAL_MODE'],
    "client_id": app.config['PAYPAL_CLIENT_ID'],
    "client_secret": app.config['PAYPAL_CLIENT_SECRET']
})

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['REMOVE_BG_API_KEY'] = 'asYWaQdf2hsQ3p8DDFTFZ53f'  # Reemplaza con tu API key

# Asegurar que la carpeta de uploads existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ruta para donaciones con PayPal
@app.route('/donaciones')
def donaciones():
    return render_template('donaciones.html')

@app.route('/create_payment', methods=['POST'])
def create_payment():
    amount = request.form.get('amount')

    payment = paypalrestsdk.Payment({
        "intent": "sale",
        "payer": {
            "payment_method": "paypal"
        },
        "transactions": [{
            "amount": {
                "total": amount,
                "currency": "USD"
            },
            "description": _("Donación al Portal Multifuncional")
        }],
        "redirect_urls": {
            "return_url": url_for('execute_payment', _external=True),
            "cancel_url": url_for('donaciones', _external=True)
        }
    })

    if payment.create():
        for link in payment.links:
            if link.rel == "approval_url":
                return jsonify({'redirect_url': link.href})
    else:
        return jsonify({'error': payment.error}), 400

@app.route('/execute_payment')
def execute_payment():
    payment_id = request.args.get('paymentId')
    payer_id = request.args.get('PayerID')

    payment = paypalrestsdk.Payment.find(payment_id)

    if payment.execute({"payer_id": payer_id}):
        return render_template('donaciones.html', success=_("¡Gracias por tu donación!"))
    else:
        return render_template('donaciones.html', error=_("Hubo un problema con tu donación."))


# Cargar modelo REAL ESRGAN (simplificado - en producción cargarías un modelo preentrenado)
# Nota: En un caso real, descargarías los pesos preentrenados de REAL ESRGAN
try:
    # Modelo RealESRGAN (x4 plus)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4

    # Cargar pesos preentrenados (esto descargará los modelos la primera vez)
    model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
    os.makedirs('weights', exist_ok=True)

    if not os.path.exists(model_path):
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        torch.hub.download_url_to_file(url, model_path)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=400,  # Tile size, 0 para no usar tiles
        tile_pad=10,
        pre_pad=0,
        half=False  # No usar float16
    )
except Exception as e:
    print(f"Error al cargar Real-ESRGAN: {e}")
    upsampler = None

# Mejoramiento de imágenes con REAL ESRGAN
@app.route('/mejora-imagen', methods=['GET', 'POST'])
def mejora_imagen():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Leer la imagen con OpenCV
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

                if upsampler:
                    # Procesar con Real-ESRGAN
                    output, _ = upsampler.enhance(img, outscale=4)
                else:
                    # Fallback si no hay modelo
                    output = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2),
                               interpolation=cv2.INTER_CUBIC)

                # Guardar la imagen mejorada
                enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], 'enhanced_' + filename)
                cv2.imwrite(enhanced_path, output)

                return render_template('mejora_imagen.html',
                                     original_img=filename,
                                     enhanced_img='enhanced_' + filename)

            except Exception as e:
                print(f"Error al procesar imagen: {e}")
                return render_template('mejora_imagen.html', error=_("Error al procesar la imagen"))

    return render_template('mejora_imagen.html')

# Extractor de paleta de colores con K-Means
@app.route('/paleta-colores', methods=['GET', 'POST'])
def paleta_colores():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Procesamiento para extraer colores dominantes con K-Means
                img = Image.open(filepath)
                img = img.convert('RGB')

                # Redimensionar para hacer el procesamiento más rápido
                img_small = img.resize((100, 100))
                pixels = np.array(img_small).reshape(-1, 3)

                # Usar K-Means para encontrar colores dominantes
                kmeans = KMeans(n_clusters=5, random_state=42)
                kmeans.fit(pixels)

                # Obtener los colores dominantes (centroides de los clusters)
                colors = kmeans.cluster_centers_.astype(int)

                # Ordenar colores por frecuencia
                unique, counts = np.unique(kmeans.labels_, return_counts=True)
                sorted_colors = [color for _, color in sorted(zip(counts, colors), reverse=True)]

                return render_template('paleta_colores.html',
                                     image=filename,
                                     colors=sorted_colors)

            except Exception as e:
                print(f"Error al extraer colores: {e}")
                return render_template('paleta_colores.html', error="Error al extraer colores")

    return render_template('paleta_colores.html')

# Eliminar fondo con Remove.bg API
@app.route('/remove-bg', methods=['GET', 'POST'])
def remove_bg():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Llamar a la API de Remove.bg
                response = requests.post(
                    'https://api.remove.bg/v1.0/removebg',
                    files={'image_file': open(filepath, 'rb')},
                    data={'size': 'auto'},
                    headers={'X-Api-Key': app.config['REMOVE_BG_API_KEY']},
                )

                if response.status_code == requests.codes.ok:
                    # Guardar la imagen sin fondo
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'nobg_' + filename)
                    with open(result_path, 'wb') as out:
                        out.write(response.content)

                    return render_template('remove_bg.html',
                                         original_img=filename,
                                         result_img='nobg_' + filename)
                else:
                    error_msg = f"Error API: {response.status_code} {response.text}"
                    print(error_msg)
                    return render_template('remove_bg.html', error=error_msg)

            except Exception as e:
                print(f"Error al eliminar fondo: {e}")
                return render_template('remove_bg.html', error="Error al eliminar el fondo")

    return render_template('remove_bg.html')

# (Las rutas text_to_speech y generador_qr permanecen iguales)
@app.route('/text-to-speech', methods=['GET', 'POST'])
def text_to_speech():
    if request.method == 'POST':
        text = request.form.get('text', '')
        lang = request.form.get('lang', 'es')

        if text:
            tts = gTTS(text=text, lang=lang, slow=False)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'speech.mp3')
            tts.save(audio_path)

            return render_template('text_to_speech.html',
                                 audio_file='speech.mp3',
                                 text=text)

    return render_template('text_to_speech.html')

@app.route('/generador-qr', methods=['GET', 'POST'])
def generador_qr():
    if request.method == 'POST':
        data = request.form.get('data', '')

        if data:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")

            # Guardar QR
            qr_path = os.path.join(app.config['UPLOAD_FOLDER'], 'qr_code.png')
            img.save(qr_path)

            return render_template('generador_qr.html',
                                 qr_image='qr_code.png',
                                 data=data)

    return render_template('generador_qr.html')

if __name__ == '__main__':
    app.run(debug=True)