"""
Configuration file for ASDP (AI Survey Data Processor) Application
Ministry of Statistics and Programme Implementation (MoSPI)
"""

import os
from datetime import datetime

class Config:
    """Application configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Data Processing Configuration
    SUPPORTED_FILE_TYPES = {'csv', 'xlsx', 'xls'}
    
    # Missing Value Imputation
    IMPUTATION_METHODS = {
        'mean': 'Mean Imputation',
        'median': 'Median Imputation', 
        'knn': 'K-Nearest Neighbors (KNN)'
    }
    KNN_NEIGHBORS = 5
    
    # Outlier Detection
    OUTLIER_DETECTION_METHODS = {
        'iqr': 'Interquartile Range (IQR)',
        'zscore': 'Z-Score',
        'isolation_forest': 'Isolation Forest (AI-based)'
    }
    OUTLIER_HANDLING_METHODS = {
        'winsorize': 'Winsorization',
        'remove': 'Remove Outliers'
    }
    IQR_THRESHOLD = 1.5
    ZSCORE_THRESHOLD = 3.0
    ISOLATION_FOREST_CONTAMINATION = 0.1
    
    # Statistical Analysis
    CONFIDENCE_LEVEL = 0.95
    Z_SCORE_95 = 1.96
    
    # Visualization Settings
    MAX_VISUALIZATION_COLUMNS = 5
    PLOT_HEIGHT = 400
    PLOT_WIDTH = 600
    
    # Report Generation
    REPORT_TITLE = "ASDP (AI Survey Data Processor) Report"
    REPORT_SUBTITLE = "Ministry of Statistics and Programme Implementation (MoSPI)"
    PDF_PAGE_SIZE = 'A4'
    
    # Application Metadata
    APP_NAME = "ASDP (AI Survey Data Processor)"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Automated Data Preparation, Estimation and Report Writing"
    DEVELOPED_FOR = "Ministry of Statistics and Programme Implementation (MoSPI)"
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @staticmethod
    def get_app_info():
        """Get application information"""
        return {
            'name': Config.APP_NAME,
            'version': Config.APP_VERSION,
            'description': Config.APP_DESCRIPTION,
            'developed_for': Config.DEVELOPED_FOR,
            'startup_time': datetime.now().isoformat()
        }
    
    @staticmethod
    def get_processing_options():
        """Get available processing options for the frontend"""
        return {
            'imputation_methods': Config.IMPUTATION_METHODS,
            'outlier_detection_methods': Config.OUTLIER_DETECTION_METHODS,
            'outlier_handling_methods': Config.OUTLIER_HANDLING_METHODS,
            'supported_file_types': list(Config.SUPPORTED_FILE_TYPES),
            'max_file_size_mb': Config.MAX_CONTENT_LENGTH // (1024 * 1024)
        }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'change-this-in-production'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
