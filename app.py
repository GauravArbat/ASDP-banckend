import os
import json
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import io
import base64
from datetime import datetime
import zipfile
from werkzeug.utils import secure_filename
import tempfile
import warnings
import math
from sqlalchemy import text
warnings.filterwarnings('ignore')

app = Flask(__name__)
# Configure via environment when available (for Render/Docker)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
try:
    app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
except Exception:
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app, 
     origins=["https://asdp-frontend.vercel.app"],
     supports_credentials=True)

# Database and authentication setup
# Prefer DATABASE_URL/SQLALCHEMY_DATABASE_URI from environment for portability
app.config['SQLALCHEMY_DATABASE_URI'] = (
    os.environ.get('DATABASE_URL')
    or os.environ.get('SQLALCHEMY_DATABASE_URI')
    or 'sqlite:///app.db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['AVATAR_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'avatars')
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Configure login manager to handle unauthorized access
@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({'error': 'Authentication required'}), 401

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AVATAR_FOLDER'], exist_ok=True)

# Auth models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'admin' or 'user'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    profile_image = db.Column(db.String(512))  # relative path like /avatars/filename.png

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(1024), nullable=False)
    rows = db.Column(db.Integer)
    columns = db.Column(db.Integer)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    owner = db.relationship('User', backref='datasets')


class ProcessingRun(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    config = db.Column(db.JSON)
    cleaning_log = db.Column(db.JSON)
    estimates = db.Column(db.JSON)
    plots_count = db.Column(db.Integer)
    success = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    dataset = db.relationship('Dataset')
    user = db.relationship('User')


class ReportRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    format = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    dataset = db.relationship('Dataset')
    user = db.relationship('User')


@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None


def admin_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        if getattr(current_user, 'role', 'user') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return view_func(*args, **kwargs)
    return wrapped


def migrate_database():
    """Migrate database schema to latest version"""
    try:
        # Check and update user table
        result = db.session.execute(db.text("PRAGMA table_info(user)"))
        cols = [row[1] for row in result]
        if 'profile_image' not in cols:
            db.session.execute(db.text("ALTER TABLE user ADD COLUMN profile_image TEXT"))
            print('[MIGRATION] Added profile_image column to user table')
        
        # Check and update report_record table
        result = db.session.execute(db.text("PRAGMA table_info(report_record)"))
        cols = [row[1] for row in result]
        if 'dataset_id' not in cols:
            db.session.execute(db.text("ALTER TABLE report_record ADD COLUMN dataset_id INTEGER"))
            print('[MIGRATION] Added dataset_id column to report_record table')
        if 'user_id' not in cols:
            db.session.execute(db.text("ALTER TABLE report_record ADD COLUMN user_id INTEGER"))
            print('[MIGRATION] Added user_id column to report_record table')
        if 'format' not in cols:
            db.session.execute(db.text("ALTER TABLE report_record ADD COLUMN format VARCHAR(10)"))
            print('[MIGRATION] Added format column to report_record table')
        if 'created_at' not in cols:
            db.session.execute(db.text("ALTER TABLE report_record ADD COLUMN created_at DATETIME"))
            print('[MIGRATION] Added created_at column to report_record table')
        
        # Check and update processing_run table
        result = db.session.execute(db.text("PRAGMA table_info(processing_run)"))
        cols = [row[1] for row in result]
        if 'dataset_id' not in cols:
            db.session.execute(db.text("ALTER TABLE processing_run ADD COLUMN dataset_id INTEGER"))
            print('[MIGRATION] Added dataset_id column to processing_run table')
        if 'user_id' not in cols:
            db.session.execute(db.text("ALTER TABLE processing_run ADD COLUMN user_id INTEGER"))
            print('[MIGRATION] Added user_id column to processing_run table')
        if 'config' not in cols:
            db.session.execute(db.text("ALTER TABLE processing_run ADD COLUMN config TEXT"))
            print('[MIGRATION] Added config column to processing_run table')
        if 'cleaning_log' not in cols:
            db.session.execute(db.text("ALTER TABLE processing_run ADD COLUMN cleaning_log TEXT"))
            print('[MIGRATION] Added cleaning_log column to processing_run table')
        if 'estimates' not in cols:
            db.session.execute(db.text("ALTER TABLE processing_run ADD COLUMN estimates TEXT"))
            print('[MIGRATION] Added estimates column to processing_run table')
        if 'plots_count' not in cols:
            db.session.execute(db.text("ALTER TABLE processing_run ADD COLUMN plots_count INTEGER"))
            print('[MIGRATION] Added plots_count column to processing_run table')
        if 'success' not in cols:
            db.session.execute(db.text("ALTER TABLE processing_run ADD COLUMN success BOOLEAN"))
            print('[MIGRATION] Added success column to processing_run table')
        if 'created_at' not in cols:
            db.session.execute(db.text("ALTER TABLE processing_run ADD COLUMN created_at DATETIME"))
            print('[MIGRATION] Added created_at column to processing_run table')
        
        # Check and update dataset table
        result = db.session.execute(db.text("PRAGMA table_info(dataset)"))
        cols = [row[1] for row in result]
        if 'owner_id' not in cols:
            db.session.execute(db.text("ALTER TABLE dataset ADD COLUMN owner_id INTEGER"))
            print('[MIGRATION] Added owner_id column to dataset table')
        if 'uploaded_at' not in cols:
            db.session.execute(db.text("ALTER TABLE dataset ADD COLUMN uploaded_at DATETIME"))
            print('[MIGRATION] Added uploaded_at column to dataset table')
        
        db.session.commit()
        print('[MIGRATION] Database migration completed successfully')
        
    except Exception as e:
        print(f'[MIGRATION] Error during migration: {e}')
        db.session.rollback()
        raise

class DataProcessor:
    def __init__(self):
        self.data = None
        self.cleaned_data = None
        self.weights = None
        self.cleaning_log = []
        self.estimates = {}
        
    def load_data(self, file_path):
        """Load data from CSV or Excel file"""
        try:
            import pandas as pd  # Lazy import
            if file_path.endswith('.csv'):
                # Try fast path first, then fallbacks for tricky files
                try:
                    self.data = pd.read_csv(file_path)
                except Exception:
                    try:
                        # Auto-detect separator and use python engine
                        self.data = pd.read_csv(file_path, engine='python', sep=None)
                    except Exception:
                        # Encoding/line issues fallback
                        self.data = pd.read_csv(
                            file_path,
                            engine='python',
                            sep=None,
                            encoding='latin1',
                            on_bad_lines='skip'
                        )
            elif file_path.endswith(('.xlsx', '.xls')):
                try:
                    self.data = pd.read_excel(file_path)
                except ImportError as ie:
                    raise ImportError("Excel reading requires openpyxl. Install with: pip install openpyxl") from ie
            else:
                raise ValueError("Unsupported file format")
            
            # Attempt to coerce numeric-like object columns (e.g., values with commas or currency symbols)
            try:
                object_columns = self.data.select_dtypes(include=['object']).columns
                converted_columns = []
                for col in object_columns:
                    original_series = self.data[col]
                    # Remove common thousands separators and currency symbols then coerce
                    cleaned = (
                        original_series.astype(str)
                        .str.replace(r"[\s,₹$]", "", regex=True)
                        .str.replace(r"[^0-9eE+\-.]", "", regex=True)
                    )
                    numeric_series = pd.to_numeric(cleaned, errors='coerce')
                    # If majority became numeric, accept conversion
                    if numeric_series.notna().mean() >= 0.8:
                        self.data[col] = numeric_series
                        converted_columns.append(col)
                if converted_columns:
                    self.cleaning_log.append(
                        f"Auto-converted numeric-like columns: {', '.join(converted_columns)}"
                    )
            except Exception:
                # Non-fatal; proceed without coercion
                pass

            # Check if data is empty or has no columns
            if self.data is None or len(self.data) == 0 or len(self.data.columns) == 0:
                self.cleaning_log.append("Error: No data or columns found in file")
                return False
                
            self.cleaning_log.append(f"Data loaded successfully: {len(self.data)} rows, {len(self.data.columns)} columns")
            return True
        except Exception as e:
            self.cleaning_log.append(f"Error loading data: {str(e)}")
            return False
    
    def detect_missing_values(self):
        """Detect and report missing values as a list of dicts (no pandas dependency)."""
        total_rows = len(self.data)
        missing_summary = self.data.isnull().sum()
        missing_percentage = (missing_summary / total_rows) * 100
        results = []
        for column_name, miss_count in missing_summary.items():
            if miss_count > 0:
                results.append({
                    'Column': column_name,
                    'Missing_Count': int(miss_count),
                    'Missing_Percentage': float(missing_percentage[column_name])
                })
        return results
    
    def impute_missing_values(self, method='mean', columns=None):
        """Impute missing values using specified method"""
        # Determine numeric columns
        if columns is None:
            numeric_columns = self.data.select_dtypes(include=['number']).columns
        else:
            numeric_columns = [col for col in columns if col in self.data.columns and self.data[col].dtype in ['int64', 'float64']]

        # Fast path without sklearn for mean/median
        if method in ('mean', 'median'):
            if method == 'mean':
                self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mean())
            else:
                self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())
            self.cleaning_log.append(f"Imputed missing values using {method} method for {len(numeric_columns)} columns")
            return

        # KNN requires scikit-learn
        if method == 'knn':
            try:
                from sklearn.impute import KNNImputer  # type: ignore
            except Exception as import_error:
                # Fallback to mean imputation if scikit-learn is unavailable
                self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mean())
                self.cleaning_log.append("scikit-learn not installed; KNN imputation unavailable. Fell back to mean imputation.")
                return
            imputer = KNNImputer(n_neighbors=5)
            self.data[numeric_columns] = imputer.fit_transform(self.data[numeric_columns])
            self.cleaning_log.append(f"Imputed missing values using {method} method for {len(numeric_columns)} columns")
            return

        raise ValueError("Method must be 'mean', 'median', or 'knn'")
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """Detect outliers using specified method"""
        outliers_report = {}
        numeric_columns = self.data.select_dtypes(include=['number']).columns
        
        for column in numeric_columns:
            if method == 'iqr':
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
            elif method == 'zscore':
                # Compute z-scores using pandas to avoid external dependencies and index misalignment
                series = self.data[column]
                mean = series.mean()
                std = series.std(ddof=0)
                if std == 0 or (std != std):  # handle zero or NaN std
                    outliers = self.data.iloc[0:0]
                else:
                    z_scores = ((series - mean) / std).abs()
                    outliers = self.data[z_scores > threshold]
            elif method == 'isolation_forest':
                try:
                    from sklearn.ensemble import IsolationForest
                except Exception as import_error:
                    # Fallback to IQR if scikit-learn is unavailable
                    self.cleaning_log.append("scikit-learn not installed; Isolation Forest unavailable. Fell back to IQR method.")
                    Q1 = self.data[column].quantile(0.25)
                    Q3 = self.data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
                    outliers_report[column] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(self.data)) * 100,
                        'indices': outliers.index.tolist()
                    }
                    continue
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = self.data[iso_forest.fit_predict(self.data[[column]]) == -1]
            
            outliers_report[column] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.data)) * 100,
                'indices': outliers.index.tolist()
            }
        
        return outliers_report
    
    def handle_outliers(self, method='winsorize', columns=None, percentile=5):
        """Handle outliers using specified method"""
        if columns is None:
            numeric_columns = self.data.select_dtypes(include=['number']).columns
        else:
            numeric_columns = [col for col in columns if col in self.data.columns and self.data[col].dtype in ['int64', 'float64']]
        
        for column in numeric_columns:
            if method == 'winsorize':
                lower = self.data[column].quantile(percentile / 100.0)
                upper = self.data[column].quantile(1 - percentile / 100.0)
                self.data[column] = self.data[column].clip(lower, upper)
            elif method == 'remove':
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
        
        self.cleaning_log.append(f"Handled outliers using {method} method for {len(numeric_columns)} columns")
    
    def apply_weights(self, weight_column):
        """Apply survey weights"""
        if weight_column in self.data.columns:
            self.weights = self.data[weight_column]
            self.cleaning_log.append(f"Applied weights from column: {weight_column}")
            return True
        else:
            self.cleaning_log.append(f"Weight column {weight_column} not found")
            return False
    
    def calculate_estimates(self, columns=None):
        """Calculate weighted and unweighted estimates"""
        if columns is None:
            numeric_columns = self.data.select_dtypes(include=['number']).columns
        else:
            numeric_columns = [col for col in columns if col in self.data.columns and self.data[col].dtype in ['int64', 'float64']]
        
        estimates = {}
        for column in numeric_columns:
            # Unweighted estimates
            unweighted_mean = self.data[column].mean()
            unweighted_std = self.data[column].std()
            unweighted_se = unweighted_std / math.sqrt(len(self.data))
            
            estimates[column] = {
                'unweighted': {
                    'mean': unweighted_mean,
                    'std': unweighted_std,
                    'se': unweighted_se,
                    'ci_95_lower': unweighted_mean - 1.96 * unweighted_se,
                    'ci_95_upper': unweighted_mean + 1.96 * unweighted_se
                }
            }
            
            # Weighted estimates
            if self.weights is not None:
                weight_sum = self.weights.sum()
                if weight_sum == 0:
                    weighted_mean = float('nan')
                    weighted_std = float('nan')
                    weighted_se = float('nan')
                else:
                    weighted_mean = (self.data[column] * self.weights).sum() / weight_sum
                    weighted_variance = (((self.data[column] - weighted_mean) ** 2) * self.weights).sum() / weight_sum
                    weighted_std = math.sqrt(weighted_variance)
                    weighted_se = weighted_std / math.sqrt(len(self.data))
                
                estimates[column]['weighted'] = {
                    'mean': weighted_mean,
                    'std': weighted_std,
                    'se': weighted_se,
                    'ci_95_lower': weighted_mean - 1.96 * weighted_se,
                    'ci_95_upper': weighted_mean + 1.96 * weighted_se
                }
        
        self.estimates = estimates
        self.cleaning_log.append(f"Calculated estimates for {len(numeric_columns)} columns")
        return estimates
    
    def generate_visualizations(self):
        """Generate data visualizations"""
        plots = {}
        # Lazy import plotly when needed
        try:
            import plotly.express as px
        except Exception as import_error:
            # If plotly is not installed, return empty plots with a hint
            self.cleaning_log.append("Plotly not installed; skipping visualizations.")
            return plots
        
        # Distribution plots for numeric columns
        numeric_columns = self.data.select_dtypes(include=['number']).columns[:5]  # Limit to first 5 columns
        
        for column in numeric_columns:
            fig = px.histogram(self.data, x=column, title=f'Distribution of {column}')
            plots[f'dist_{column}'] = fig.to_html(full_html=False)
        
        # Correlation heatmap
        if len(numeric_columns) > 1:
            corr_matrix = self.data[numeric_columns].corr()
            fig = px.imshow(corr_matrix, title='Correlation Matrix')
            plots['correlation'] = fig.to_html(full_html=False)
        
        # Missing values plot
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            fig = px.bar(x=missing_data.index, y=missing_data.values, title='Missing Values by Column')
            plots['missing'] = fig.to_html(full_html=False)
        
        return plots
    
    def generate_report(self, format='pdf'):
        """Generate comprehensive report"""
        if format == 'pdf':
            return self._generate_pdf_report()
        else:
            return self._generate_html_report()
    
    def _generate_pdf_report(self):
        """Generate PDF report"""
        # Lazy import reportlab only when generating PDF
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
        except Exception as import_error:
            raise ImportError("Missing reportlab for PDF reports. Install with: pip install reportlab or request HTML report instead.") from import_error
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("ASDP (AI Survey Data Processor) Report", title_style))
        story.append(Spacer(1, 12))
        
        # Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Paragraph(f"Data Processing completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Total records processed: {len(self.data)}", styles['Normal']))
        story.append(Paragraph(f"Total variables: {len(self.data.columns)}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Cleaning Log
        story.append(Paragraph("Data Cleaning Log", styles['Heading2']))
        for log_entry in self.cleaning_log:
            story.append(Paragraph(f"• {log_entry}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Estimates Table
        if self.estimates:
            story.append(Paragraph("Statistical Estimates", styles['Heading2']))
            estimate_data = [['Variable', 'Mean', 'Std Dev', 'Standard Error', '95% CI Lower', '95% CI Upper']]
            
            for var, est in self.estimates.items():
                if 'weighted' in est:
                    row = [
                        var,
                        f"{est['weighted']['mean']:.4f}",
                        f"{est['weighted']['std']:.4f}",
                        f"{est['weighted']['se']:.4f}",
                        f"{est['weighted']['ci_95_lower']:.4f}",
                        f"{est['weighted']['ci_95_upper']:.4f}"
                    ]
                else:
                    row = [
                        var,
                        f"{est['unweighted']['mean']:.4f}",
                        f"{est['unweighted']['std']:.4f}",
                        f"{est['unweighted']['se']:.4f}",
                        f"{est['unweighted']['ci_95_lower']:.4f}",
                        f"{est['unweighted']['ci_95_upper']:.4f}"
                    ]
                estimate_data.append(row)
            
            table = Table(estimate_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _generate_html_report(self):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Survey Data Processing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 20px 0; }}
                .log-entry {{ margin: 5px 0; padding: 5px; background-color: #f8f9fa; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ASDP (AI Survey Data Processor) Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>Total records processed: {len(self.data)}</p>
                <p>Total variables: {len(self.data.columns)}</p>
            </div>
            
            <div class="section">
                <h2>Data Cleaning Log</h2>
                {''.join([f'<div class="log-entry">• {entry}</div>' for entry in self.cleaning_log])}
            </div>
            
            <div class="section">
                <h2>Statistical Estimates</h2>
                <table>
                    <tr>
                        <th>Variable</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Standard Error</th>
                        <th>95% CI Lower</th>
                        <th>95% CI Upper</th>
                    </tr>
        """
        
        if self.estimates:
            for var, est in self.estimates.items():
                if 'weighted' in est:
                    html_content += f"""
                    <tr>
                        <td>{var}</td>
                        <td>{est['weighted']['mean']:.4f}</td>
                        <td>{est['weighted']['std']:.4f}</td>
                        <td>{est['weighted']['se']:.4f}</td>
                        <td>{est['weighted']['ci_95_lower']:.4f}</td>
                        <td>{est['weighted']['ci_95_upper']:.4f}</td>
                    </tr>
                    """
                else:
                    html_content += f"""
                    <tr>
                        <td>{var}</td>
                        <td>{est['unweighted']['mean']:.4f}</td>
                        <td>{est['unweighted']['std']:.4f}</td>
                        <td>{est['unweighted']['se']:.4f}</td>
                        <td>{est['unweighted']['ci_95_lower']:.4f}</td>
                        <td>{est['unweighted']['ci_95_upper']:.4f}</td>
                    </tr>
                    """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html_content

# Global processor instance
processor = DataProcessor()

# API Authentication endpoints
@app.route('/api/auth/me', methods=['GET'])
def api_auth_me():
    if current_user.is_authenticated:
        return jsonify({
            'is_authenticated': True,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'role': current_user.role,
                'profile_image': current_user.profile_image
            }
        })
    return jsonify({'is_authenticated': False, 'user': None})

@app.route('/api/auth/login', methods=['POST'])
def api_auth_login():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid credentials'}), 401

    login_user(user)
    return jsonify({
        'success': True,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'profile_image': user.profile_image
        }
    })

@app.route('/api/auth/register', methods=['POST'])
def api_auth_register():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip() or None
    password = data.get('password') or ''
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400

    user = User(username=username, email=email, role='user')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    
    return jsonify({
        'success': True,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'profile_image': user.profile_image
        }
    })

@app.route('/api/auth/logout', methods=['POST'])
def api_auth_logout():
    if current_user.is_authenticated:
        logout_user()
    return jsonify({'success': True})

@app.route('/api/auth/profile', methods=['GET', 'POST'])
def api_auth_profile():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if request.method == 'GET':
        return jsonify({
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'role': current_user.role,
                'profile_image': current_user.profile_image
            }
        })
    
    # POST - update profile
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    if 'email' in data:
        current_user.email = data['email'].strip() or None
    
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/auth/avatar', methods=['POST'])
def api_auth_avatar():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'avatar' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(f"{current_user.id}_{file.filename}")
        filepath = os.path.join(app.config['AVATAR_FOLDER'], filename)
        file.save(filepath)
        
        # Update user profile
        current_user.profile_image = f"/avatars/{filename}"
        db.session.commit()
        
        return jsonify({'success': True, 'profile_image': current_user.profile_image})

@app.route('/api/auth/avatars/<path:filename>')
def api_auth_avatars(filename):
    return send_from_directory(app.config['AVATAR_FOLDER'], filename)

@app.route('/')
def index():
    try:
        # Basic health check - ensure database is accessible
        db_status = "healthy"
        try:
            # Simple database connectivity test
            db.session.execute(text("SELECT 1"))
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        return jsonify({
            'message': 'Welcome to the ASDP (AI Survey Data Processor) API',
            'version': '1.0.0',
            'status': 'running',
            'database': db_status,
            'timestamp': datetime.utcnow().isoformat(),
            'endpoints': {
                'health': '/health',
                'auth': '/api/auth/*',
                'data': '/upload, /clean, /report, /download_data',
                'admin': '/admin, /admin/summary',
                'profile': '/profile'
            }
        })
    except Exception as e:
        return jsonify({
            'error': 'Service error',
            'message': str(e),
            'status': 'error',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/health')
def health_check():
    try:
        # Check database connectivity
        db_status = "healthy"
        try:
            db.session.execute(text("SELECT 1"))
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Check upload directory
        upload_status = "healthy"
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['AVATAR_FOLDER'], exist_ok=True)
        except Exception as e:
            upload_status = f"error: {str(e)}"
        
        return jsonify({
            'status': 'healthy' if db_status == "healthy" and upload_status == "healthy" else 'degraded',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {
                'database': db_status,
                'upload_directory': upload_status
            },
            'version': '1.0.0',
            'service': 'ASDP Backend API'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/test-auth')
def test_auth():
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'role': current_user.role
            }
        })
    else:
        return jsonify({'authenticated': False})


# Authentication pages and handlers
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return jsonify({'message': 'Login page not available via this endpoint. Use /api/auth/login for JSON.'})

    # Support form submit or JSON
    data = request.json if request.is_json else request.form
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    if not username or not password:
        if request.is_json:
            return jsonify({'error': 'Username and password required'}), 400
        return jsonify({'message': 'Username and password required'}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        if request.is_json:
            return jsonify({'error': 'Invalid credentials'}), 401
        return jsonify({'message': 'Invalid credentials'}), 401

    login_user(user)
    if request.is_json:
        return jsonify({'success': True, 'user': {'username': user.username, 'role': user.role}})
    return jsonify({'message': 'Login successful. Use /api/auth/me to check authentication.'})


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return jsonify({'message': 'Register page not available via this endpoint. Use /api/auth/register for JSON.'})

    data = request.json if request.is_json else request.form
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip() or None
    password = data.get('password') or ''
    confirm = data.get('confirm') or ''

    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400
    if password != confirm:
        return jsonify({'message': 'Passwords do not match'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists'}), 400

    user = User(username=username, email=email, role='user')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify({'message': 'Registration successful. Use /api/auth/me to check authentication.'})


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if current_user.is_authenticated:
        logout_user()
    if request.method == 'POST' and request.is_json:
        return jsonify({'success': True})
    return jsonify({'message': 'Logout successful.'})


# Deprecated duplicate JSON-only login/logout removed (handled above by GET/POST routes)


@app.route('/admin/summary')
@login_required
@admin_required
def admin_summary():
    users_count = User.query.count()
    datasets_count = Dataset.query.count()
    latest_datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).limit(10).all()
    payload = {
        'users': users_count,
        'datasets': datasets_count,
        'latest': [
            {
                'id': d.id,
                'filename': d.filename,
                'rows': d.rows,
                'columns': d.columns,
                'owner': (d.owner.username if d.owner else None),
                'uploaded_at': d.uploaded_at.isoformat()
            }
            for d in latest_datasets
        ]
    }
    return jsonify(payload)


@app.route('/admin')
@login_required
@admin_required
def admin_page():
    try:
        users_count = User.query.count()
        datasets_count = Dataset.query.count()
        runs_count = ProcessingRun.query.count()
        reports_count = ReportRecord.query.count()
        latest_datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).limit(10).all()
        latest_runs = ProcessingRun.query.order_by(ProcessingRun.created_at.desc()).limit(10).all()
        all_users = User.query.order_by(User.created_at.desc()).all()
        
        return jsonify({
            'users': users_count,
            'datasets': datasets_count,
            'runs': runs_count,
            'reports': reports_count,
            'latest_datasets': [
                {
                    'id': d.id,
                    'filename': d.filename,
                    'rows': d.rows,
                    'columns': d.columns,
                    'owner': (d.owner.username if d.owner else None),
                    'uploaded_at': d.uploaded_at.isoformat()
                }
                for d in latest_datasets
            ],
            'recent_runs': [
                {
                    'id': r.id,
                    'dataset_id': r.dataset_id,
                    'user_id': r.user_id,
                    'config': r.config,
                    'cleaning_log': r.cleaning_log,
                    'estimates': r.estimates,
                    'plots_count': r.plots_count,
                    'success': r.success,
                    'created_at': r.created_at.isoformat()
                }
                for r in latest_runs
            ],
            'all_users': [
                {
                    'id': u.id,
                    'username': u.username,
                    'email': u.email,
                    'role': u.role,
                    'created_at': u.created_at.isoformat()
                }
                for u in all_users
            ]
        })
    except Exception as e:
        print(f"Admin dashboard error: {e}")
        return jsonify({'error': f'Failed to load admin dashboard: {str(e)}'}), 500

@app.route('/admin/user/<int:user_id>/role', methods=['POST'])
@login_required
@admin_required
def update_user_role(user_id):
    data = request.json
    role = data.get('role')
    if role not in ['user', 'admin']:
        return jsonify({'error': 'Invalid role'}), 400
    target = User.query.get(user_id)
    if not target:
        return jsonify({'error': 'User not found'}), 404
    target.role = role
    db.session.commit()
    if request.is_json:
        return jsonify({'success': True})
    # Redirect back to admin dashboard after update
    return jsonify({'message': 'Role updated successfully.'})

@app.route('/api/admin/dashboard')
@login_required
@admin_required
def api_admin_dashboard():
    try:
        users_count = User.query.count()
        datasets_count = Dataset.query.count()
        runs_count = ProcessingRun.query.count()
        reports_count = ReportRecord.query.count()
        latest_datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).limit(10).all()
        latest_runs = ProcessingRun.query.order_by(ProcessingRun.created_at.desc()).limit(10).all()
        all_users = User.query.order_by(User.created_at.desc()).all()
        
        return jsonify({
            'users': users_count,
            'datasets': datasets_count,
            'runs': runs_count,
            'reports': reports_count,
            'latest_datasets': [
                {
                    'id': d.id,
                    'filename': d.filename,
                    'rows': d.rows,
                    'columns': d.columns,
                    'owner': (d.owner.username if d.owner else None),
                    'uploaded_at': d.uploaded_at.isoformat()
                }
                for d in latest_datasets
            ],
            'recent_runs': [
                {
                    'id': r.id,
                    'dataset_id': r.dataset_id,
                    'user_id': r.user_id,
                    'config': r.config,
                    'cleaning_log': r.cleaning_log,
                    'estimates': r.estimates,
                    'plots_count': r.plots_count,
                    'success': r.success,
                    'created_at': r.created_at.isoformat()
                }
                for r in latest_runs
            ],
            'all_users': [
                {
                    'id': u.id,
                    'username': u.username,
                    'email': u.email,
                    'role': u.role,
                    'created_at': u.created_at.isoformat()
                }
                for u in all_users
            ]
        })
    except Exception as e:
        print(f"API Admin dashboard error: {e}")
        return jsonify({'error': f'Failed to load admin dashboard: {str(e)}'}), 500


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'GET':
        return jsonify({
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'role': current_user.role,
                'profile_image': current_user.profile_image
            }
        })
    # POST: update username/email and optionally password
    data = request.form
    new_username = (data.get('username') or '').strip()
    new_email = (data.get('email') or '').strip() or None
    new_password = data.get('password') or ''
    # Handle avatar upload
    file = request.files.get('avatar')
    if file and file.filename:
        from werkzeug.utils import secure_filename
        fname = secure_filename(f"{current_user.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        save_path = os.path.join(app.config['AVATAR_FOLDER'], fname)
        file.save(save_path)
        current_user.profile_image = f"/avatars/{fname}"

    if new_username and new_username != current_user.username:
        if User.query.filter(User.username == new_username, User.id != current_user.id).first():
            return jsonify({'error': 'Username already taken'}), 400
        current_user.username = new_username
    if new_email and new_email != current_user.email:
        if User.query.filter(User.email == new_email, User.id != current_user.id).first():
            return jsonify({'error': 'Email already in use'}), 400
        current_user.email = new_email
    if new_password:
        current_user.set_password(new_password)
    db.session.commit()
    return jsonify({'message': 'Profile updated successfully.'})


@app.route('/avatars/<path:filename>')
def serve_avatar(filename):
    return send_from_directory(app.config['AVATAR_FOLDER'], filename)


@app.route('/admin/user/<int:user_id>/role', methods=['POST'])
@login_required
@admin_required
def admin_update_role(user_id: int):
    target = User.query.get_or_404(user_id)
    payload_role = None
    try:
        payload_role = request.form.get('role') if request.form else None
        if not payload_role and request.is_json:
            payload_role = (request.get_json(silent=True) or {}).get('role')
    except Exception:
        payload_role = None
    role = (payload_role or '').strip()
    if role not in ('admin', 'user'):
        return jsonify({'error': 'Invalid role'}), 400
    target.role = role
    db.session.commit()
    if request.is_json:
        return jsonify({'success': True})
    # Redirect back to admin dashboard after update
    return jsonify({'message': 'Role updated successfully.'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load data
        if processor.load_data(filepath):
            # Get initial data summary (guard against unexpected errors)
            try:
                # Track dataset in DB (if DB is initialized)
                ds = None
                try:
                    rows_count = len(processor.data)
                    cols_count = len(processor.data.columns)
                    ds = Dataset(
                        filename=filename,
                        filepath=filepath,
                        rows=rows_count,
                        columns=cols_count,
                        owner_id=(current_user.id if hasattr(current_user, 'id') and current_user.is_authenticated else None)
                    )
                    db.session.add(ds)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    print(f"Warning: Failed to save dataset to database: {e}")
                    # Continue without database tracking
                    ds = None
                
                summary = {
                    'rows': len(processor.data),
                    'columns': len(processor.data.columns),
                    'column_names': processor.data.columns.tolist(),
                    'data_types': processor.data.dtypes.astype(str).to_dict(),
                    'missing_values': processor.detect_missing_values()
                }
                
                response_data = {'success': True, 'summary': summary}
                if ds:
                    response_data['dataset'] = {'id': ds.id}
                
                return jsonify(response_data)
            except Exception as e:
                return jsonify({'error': f'Failed to summarize data: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Failed to load data'}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/clean', methods=['POST'])
def clean_data():
    data = request.json
    cleaning_config = data.get('config', {})
    
    # Check if data is available
    if processor.data is None or len(processor.data) == 0 or len(processor.data.columns) == 0:
        return jsonify({'error': 'No data available for cleaning. Please upload a file first.'}), 400
    
    try:
        # Missing value imputation
        if 'imputation' in cleaning_config:
            method = cleaning_config['imputation'].get('method', 'mean')
            columns = cleaning_config['imputation'].get('columns', None)
            processor.impute_missing_values(method=method, columns=columns)
        
        # Outlier detection and handling
        if 'outliers' in cleaning_config:
            detection_method = cleaning_config['outliers'].get('detection_method', 'iqr')
            handling_method = cleaning_config['outliers'].get('handling_method', 'winsorize')
            columns = cleaning_config['outliers'].get('columns', None)
            
            outliers_report = processor.detect_outliers(method=detection_method)
            processor.handle_outliers(method=handling_method, columns=columns)
        
        # Apply weights
        if 'weights' in cleaning_config:
            weight_column = cleaning_config['weights'].get('column', None)
            if weight_column:
                processor.apply_weights(weight_column)
        
        # Calculate estimates
        estimate_columns = cleaning_config.get('estimate_columns', None)
        estimates = processor.calculate_estimates(columns=estimate_columns)
        
        # Generate visualizations
        plots = processor.generate_visualizations()

        # Persist processing run details
        try:
            # Attach to the most recent dataset from this session if available
            ds = Dataset.query.order_by(Dataset.uploaded_at.desc()).first()
            run = ProcessingRun(
                dataset_id=(ds.id if ds else None),
                user_id=(current_user.id if hasattr(current_user, 'id') and current_user.is_authenticated else None),
                config=cleaning_config,
                cleaning_log=processor.cleaning_log,
                estimates=estimates,
                plots_count=len(plots),
                success=True
            )
            db.session.add(run)
            db.session.commit()
        except Exception:
            db.session.rollback()
            pass

        return jsonify({
            'success': True,
            'cleaning_log': processor.cleaning_log,
            'estimates': estimates,
            'plots': plots
        })
    
    except Exception as e:
        # Log the error for debugging
        error_msg = f"Error cleaning data: {str(e)}"
        print(f"Clean endpoint error: {error_msg}")
        return jsonify({'error': error_msg}), 400

@app.route('/report', methods=['POST'])
def generate_report():
    data = request.json
    report_format = data.get('format', 'pdf')
    
    try:
        report_content = processor.generate_report(format=report_format)
        
        if report_format == 'pdf':
            # Ensure a valid PDF bytes response
            pdf_bytes = report_content.getvalue() if hasattr(report_content, 'getvalue') else bytes(report_content)
            return send_file(
                io.BytesIO(pdf_bytes),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'survey_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
            )
        else:
            return jsonify({'html_content': report_content})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    finally:
        # Log the report generation attempt
        try:
            ds = Dataset.query.order_by(Dataset.uploaded_at.desc()).first()
            rec = ReportRecord(
                dataset_id=(ds.id if ds else None),
                user_id=(current_user.id if hasattr(current_user, 'id') and current_user.is_authenticated else None),
                format=report_format
            )
            db.session.add(rec)
            db.session.commit()
        except Exception:
            db.session.rollback()
            pass

@app.route('/download_data', methods=['POST'])
def download_processed_data():
    try:
        if processor.data is not None:
            # Stream CSV from memory to avoid leaving temp files on disk
            csv_buffer = io.StringIO()
            processor.data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            return send_file(
                io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'processed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        else:
            return jsonify({'error': 'No data available'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

# Initialize database
def init_database():
    """Initialize database and create admin user"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with app.app_context():
                db.create_all()
                print("✅ Database initialized successfully")
                
                # Create admin user if none exists
                try:
                    admin_users = User.query.filter_by(role='admin').count()
                    if admin_users == 0:
                        # Create default admin user
                        admin_username = "admin"
                        admin_email = "admin@asdp.gov.in"
                        admin_password = "admin123"  # Change this in production!
                        
                        # Check if admin user already exists
                        existing_user = User.query.filter_by(username=admin_username).first()
                        if not existing_user:
                            admin_user = User(
                                username=admin_username,
                                email=admin_email,
                                role='admin',
                                created_at=datetime.utcnow()
                            )
                            admin_user.set_password(admin_password)
                            db.session.add(admin_user)
                            db.session.commit()
                            print("✅ Admin user created successfully!")
                            print(f"Username: {admin_username}")
                            print(f"Password: {admin_password}")
                            print("⚠️  IMPORTANT: Change the password after first login!")
                        else:
                            print("✅ Admin user already exists")
                    else:
                        print(f"✅ Found {admin_users} admin user(s)")
                except Exception as e:
                    print(f"⚠️  Warning: Could not create admin user: {str(e)}")
                    # Don't fail the entire startup for admin user issues
                
                # Success - break out of retry loop
                break
                
        except Exception as e:
            retry_count += 1
            print(f"❌ Error initializing database (attempt {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count < max_retries:
                print(f"🔄 Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                print(f"❌ Failed to initialize database after {max_retries} attempts")
                print("⚠️  Application will continue but may not function properly")
                # In production, we might want to exit here
                # For now, let's continue and see if the app can run

# Initialize database when app starts
print("🚀 Starting ASDP Backend...")
print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
print(f"🗄️  Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
print(f"🔑 Secret key configured: {'Yes' if app.config['SECRET_KEY'] != 'change-me' else 'No'}")

init_database()

print("✅ ASDP Backend initialization completed!")
print("🌐 Application is ready to serve requests")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)