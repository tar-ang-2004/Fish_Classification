"""
Flask Application Configuration
==============================

Configuration settings for the Fish Classification Flask app.
You can modify these settings based on your environment.
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    # Model settings
    MODEL_PATH = 'fish_classifier_model.pth'
    MODEL_INFO_PATH = 'model_info.json'
    
    # Application settings
    APP_NAME = 'Fish Classification App'
    APP_VERSION = '1.0.0'
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5000
    
    @staticmethod
    def init_app(app):
        """Initialize app with configuration"""
        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    HOST = '127.0.0.1'  # Localhost only for development

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'prod-secret-key-must-be-set'
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Log to stderr in production
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
