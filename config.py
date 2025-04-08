import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Configuración general
    SECRET_KEY = os.getenv('SECRET_KEY', 'secret-key-default')
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Remove.bg
    REMOVE_BG_API_KEY = os.getenv('REMOVE_BG_API_KEY')
    
    # PayPal
    PAYPAL_MODE = os.getenv('PAYPAL_MODE', 'sandbox')  # 'sandbox' o 'live'
    PAYPAL_CLIENT_ID = os.getenv('PAYPAL_CLIENT_ID')
    PAYPAL_CLIENT_SECRET = os.getenv('PAYPAL_CLIENT_SECRET')
    
    # Babel
    LANGUAGES = ['en', 'es']
    BABEL_DEFAULT_LOCALE = 'es'