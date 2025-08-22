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
warnings.filterwarnings('ignore')

app = Flask(__name__)
# Configure via environment when available (for Render/Docker)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
try:
    app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
except Exception:
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# CORS for SPA development (React at :5173)
try:
    # CORS(
    #     app,
    #     supports_credentials=True,
    #     resources={
    #         r"/*": {
    #             "origins": [
    #                 "http://localhost:5173",
    #                 "http://127.0.0.1:5173",
    #                 "https://asdp-frontend.vercel.app",
    #             ]
    #         }
    #     },
    # )
    CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": r".*"}}
)
except Exception:
    # Fallback to permissive if flask_cors older version
    CORS(app)

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


class ReportRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer, db.ForeignKey('processing_run.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    report_data = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function


# Data processing classes
class DataProcessor:
    def __init__(self):
        self.cleaning_log = []
        self.estimates = {}
        self.plots_count = 0

    def clean_data(self, df):
        """Clean the dataset and log changes"""
        self.cleaning_log = []
        original_shape = df.shape
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            self.cleaning_log.append(f"Removed {initial_rows - len(df)} duplicate rows")

        # Handle missing values
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                if count < len(df) * 0.5:  # Less than 50% missing
                    if df[col].dtype in ['object', 'string']:
                        df[col] = df[col].fillna('Unknown')
                        self.cleaning_log.append(f"Filled {count} missing values in '{col}' with 'Unknown'")
                    else:
                        df[col] = df[col].fillna(df[col].median())
                        self.cleaning_log.append(f"Filled {count} missing values in '{col}' with median")
                else:
                    df = df.dropna(subset=[col])
                    self.cleaning_log.append(f"Dropped rows with missing values in '{col}' ({count} rows)")

        # Remove columns with all null values
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            df = df.drop(columns=null_columns)
            self.cleaning_log.append(f"Removed columns with all null values: {', '.join(null_columns)}")

        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        final_shape = df.shape
        if final_shape != original_shape:
            self.cleaning_log.append(f"Final dataset shape: {final_shape[0]} rows, {final_shape[1]} columns")
        
        return df

    def generate_estimates(self, df):
        """Generate statistical estimates for the dataset"""
        self.estimates = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Add column-specific statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            self.estimates['numeric_stats'] = {}
            for col in numeric_cols:
                self.estimates['numeric_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return self.estimates

    def generate_plots(self, df):
        """Generate plots for the dataset"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            plots = []
            self.plots_count = 0
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Missing values heatmap
            if df.isnull().sum().sum() > 0:
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
                plt.title('Missing Values Heatmap')
                plt.tight_layout()
                
                # Save plot to base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                plots.append({
                    'title': 'Missing Values Heatmap',
                    'data': img_base64,
                    'type': 'heatmap'
                })
                plt.close()
                self.plots_count += 1
            
            # 2. Numeric columns distribution
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = math.ceil(len(numeric_cols) / n_cols)
                
                if n_rows > 0:
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                    if n_rows == 1:
                        axes = [axes] if n_cols == 1 else axes
                    else:
                        axes = axes.flatten()
                    
                    for i, col in enumerate(numeric_cols):
                        if i < len(axes):
                            sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                            axes[i].set_title(f'Distribution of {col}')
                            axes[i].set_xlabel(col)
                    
                    # Hide empty subplots
                    for i in range(len(numeric_cols), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    
                    # Save plot to base64
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    plots.append({
                        'title': 'Numeric Columns Distribution',
                        'data': img_base64,
                        'type': 'distribution'
                    })
                    plt.close()
                    self.plots_count += 1
            
            # 3. Correlation matrix for numeric columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                
                # Save plot to base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                plots.append({
                    'title': 'Correlation Matrix',
                    'data': img_base64,
                    'type': 'correlation'
                })
                plt.close()
                self.plots_count += 1
            
            # 4. Categorical columns analysis
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                n_cols = min(2, len(categorical_cols))
                n_rows = math.ceil(len(categorical_cols) / n_cols)
                
                if n_rows > 0:
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                    if n_rows == 1:
                        axes = [axes] if n_cols == 1 else axes
                    else:
                        axes = axes.flatten()
                    
                    for i, col in enumerate(categorical_cols):
                        if i < len(axes):
                            value_counts = df[col].value_counts().head(10)
                            value_counts.plot(kind='bar', ax=axes[i])
                            axes[i].set_title(f'Top 10 Values in {col}')
                            axes[i].set_xlabel(col)
                            axes[i].tick_params(axis='x', rotation=45)
                    
                    # Hide empty subplots
                    for i in range(len(categorical_cols), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    
                    # Save plot to base64
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    plots.append({
                        'title': 'Categorical Columns Analysis',
                        'data': img_base64,
                        'type': 'categorical'
                    })
                    plt.close()
                    self.plots_count += 1
            
            return plots
            
        except ImportError as e:
            return [{'error': f'Required libraries not available: {str(e)}'}]
        except Exception as e:
            return [{'error': f'Error generating plots: {str(e)}'}]

    def create_report(self, df, config):
        """Create a comprehensive report"""
        # Clean data
        cleaned_df = self.clean_data(df)
        
        # Generate estimates
        estimates = self.generate_estimates(cleaned_df)
        
        # Generate plots
        plots = self.generate_plots(cleaned_df)
        
        # Create report data
        report_data = {
            'config': config,
            'cleaning_log': self.cleaning_log,
            'estimates': estimates,
            'plots': plots,
            'plots_count': self.plots_count,
            'original_shape': df.shape,
            'cleaned_shape': cleaned_df.shape,
            'timestamp': datetime.now().isoformat()
        }
        
        return report_data, cleaned_df


# Global processor instance
processor = DataProcessor()


# API Routes
@app.route('/')
def index():
    return jsonify({
        'message': 'ASDP API is running',
        'version': '1.0.0',
        'endpoints': {
            'auth': {
                'login': '/api/auth/login',
                'register': '/api/auth/register',
                'logout': '/api/auth/logout',
                'profile': '/api/auth/profile',
                'me': '/api/auth/me'
            },
            'admin': {
                'dashboard': '/api/admin/dashboard',
                'update_role': '/api/admin/user/<id>/role'
            },
            'data': {
                'upload': '/api/data/upload',
                'clean': '/api/data/clean',
                'report': '/api/data/report',
                'download': '/api/data/download'
            }
        }
    })


# Authentication API endpoints
@app.route('/api/auth/me', methods=['GET'])
def whoami():
    try:
        if current_user.is_authenticated:
            return jsonify({
                'is_authenticated': True,
                'user': {
                    'id': current_user.id,
                    'username': current_user.username,
                    'email': current_user.email,
                    'role': current_user.role,
                    'profile_image': current_user.profile_image,
                }
            })
        return jsonify({'is_authenticated': False})
    except Exception as e:
        return jsonify({'is_authenticated': False, 'error': str(e)}), 200


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON data required'}), 400

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
def register():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON data required'}), 400
    
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip() or None
    password = data.get('password') or ''
    confirm = data.get('confirm') or ''

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    if password != confirm:
        return jsonify({'error': 'Passwords do not match'}), 400
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
            'role': user.role
        }
    })


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    if current_user.is_authenticated:
        logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/auth/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'GET':
        return jsonify({
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'role': current_user.role,
                'profile_image': current_user.profile_image,
            }
        })
    
    # POST: update profile
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON data required'}), 400
    
    new_username = (data.get('username') or '').strip()
    new_email = (data.get('email') or '').strip() or None
    new_password = data.get('password') or ''

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
    
    return jsonify({
        'success': True, 
        'user': {
            'id': current_user.id,
            'username': current_user.username,
            'email': current_user.email,
            'role': current_user.role,
            'profile_image': current_user.profile_image,
        }
    })


@app.route('/api/auth/avatar', methods=['POST'])
@login_required
def upload_avatar():
    if 'avatar' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        fname = secure_filename(f"{current_user.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        save_path = os.path.join(app.config['AVATAR_FOLDER'], fname)
        file.save(save_path)
        current_user.profile_image = f"/api/auth/avatars/{fname}"
        db.session.commit()
        
        return jsonify({
            'success': True,
            'profile_image': current_user.profile_image
        })


@app.route('/api/auth/avatars/<path:filename>')
def serve_avatar(filename):
    return send_from_directory(app.config['AVATAR_FOLDER'], filename)


# Admin API endpoints
@app.route('/api/admin/dashboard', methods=['GET'])
@login_required
@admin_required
def admin_dashboard():
    try:
        # Get basic counts with error handling
        try:
            users_count = User.query.count()
        except Exception as e:
            print(f"Error counting users: {str(e)}")
            users_count = 0
            
        try:
            datasets_count = Dataset.query.count()
        except Exception as e:
            print(f"Error counting datasets: {str(e)}")
            datasets_count = 0
            
        try:
            runs_count = ProcessingRun.query.count()
        except Exception as e:
            print(f"Error counting runs: {str(e)}")
            runs_count = 0
            
        # Handle reports count carefully - the table might have schema issues
        try:
            reports_count = ReportRecord.query.count()
        except Exception as e:
            print(f"Error counting reports: {str(e)}")
            # If ReportRecord table has issues, try to count from processing runs
            try:
                reports_count = ProcessingRun.query.filter(ProcessingRun.plots_count.isnot(None)).count()
            except:
                reports_count = 0
        
        # Get latest data with error handling
        try:
            latest_datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).limit(10).all()
        except Exception as e:
            print(f"Error getting latest datasets: {str(e)}")
            latest_datasets = []
            
        try:
            latest_runs = ProcessingRun.query.order_by(ProcessingRun.created_at.desc()).limit(10).all()
        except Exception as e:
            print(f"Error getting latest runs: {str(e)}")
            latest_runs = []
            
        try:
            all_users = User.query.order_by(User.created_at.desc()).all()
        except Exception as e:
            print(f"Error getting users: {str(e)}")
            all_users = []
        
        return jsonify({
            'stats': {
                'users': users_count,
                'datasets': datasets_count,
                'runs': runs_count,
                'reports': reports_count
            },
            'latest_datasets': [
                {
                    'id': ds.id,
                    'filename': ds.filename,
                    'rows': ds.rows,
                    'columns': ds.columns,
                    'uploaded_at': ds.uploaded_at.isoformat(),
                    'owner': ds.owner.username if ds.owner else None
                } for ds in latest_datasets
            ],
            'latest_runs': [
                {
                    'id': run.id,
                    'dataset_id': run.dataset_id,
                    'user_id': run.user_id,
                    'success': run.success,
                    'plots_count': run.plots_count or 0,
                    'created_at': run.created_at.isoformat()
                } for run in latest_runs
            ],
            'users': [
                {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'created_at': user.created_at.isoformat()
                } for user in all_users
            ]
        })
    except Exception as e:
        print(f"Admin dashboard error: {str(e)}")
        return jsonify({'error': f'Database error: {str(e)}'}), 500


@app.route('/api/admin/user/<int:user_id>/role', methods=['POST'])
@login_required
@admin_required
def admin_update_role(user_id: int):
    try:
        target = User.query.get_or_404(user_id)
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        role = (data.get('role') or '').strip()
        if role not in ('admin', 'user'):
            return jsonify({'error': 'Invalid role'}), 400
        
        # Prevent admin from removing their own admin role
        if target.id == current_user.id and role == 'user':
            return jsonify({'error': 'Cannot remove your own admin role'}), 400
        
        old_role = target.role
        target.role = role
        db.session.commit()
        
        print(f"Admin {current_user.username} updated user {target.username} role from {old_role} to {role}")
        
        return jsonify({
            'success': True,
            'user': {
                'id': target.id,
                'username': target.username,
                'role': target.role
            },
            'message': f'Role updated from {old_role} to {role}'
        })
    except Exception as e:
        print(f"Role update error: {str(e)}")
        return jsonify({'error': f'Database error: {str(e)}'}), 500


# Data processing API endpoints
@app.route('/api/data/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            import pandas as pd
            
            # Read the file
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                return jsonify({'error': 'Unsupported file format. Please upload CSV or Excel files.'}), 400
            
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(filepath)
            
            # Create dataset record
            dataset = Dataset(
                filename=filename,
                filepath=filepath,
                rows=len(df),
                columns=len(df.columns),
                owner_id=current_user.id
            )
            db.session.add(dataset)
            db.session.commit()
            
            # Create a summary for the frontend
            summary = {
                'rows': dataset.rows,
                'columns': dataset.columns,
                'column_names': list(df.columns),
                'data_types': {col: str(df[col].dtype) for col in df.columns},
                'missing_values': [col for col in df.columns if df[col].isnull().sum() > 0]
            }
            
            return jsonify({
                'success': True,
                'dataset': {
                    'id': dataset.id,
                    'filename': dataset.filename,
                    'rows': dataset.rows,
                    'columns': dataset.columns,
                    'uploaded_at': dataset.uploaded_at.isoformat()
                },
                'summary': summary
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/api/data/clean', methods=['POST'])
@login_required
def clean_data():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON data required'}), 400
    
    dataset_id = data.get('dataset_id')
    if not dataset_id:
        return jsonify({'error': 'Dataset ID required'}), 400
    
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user owns the dataset or is admin
    if dataset.owner_id != current_user.id and current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        import pandas as pd
        
        # Read the dataset
        df = pd.read_csv(dataset.filepath)
        
        # Clean the data
        cleaned_df = processor.clean_data(df)
        
        # Save cleaned data
        cleaned_filename = f"cleaned_{os.path.basename(dataset.filepath)}"
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
        cleaned_df.to_csv(cleaned_filepath, index=False)
        
        # Create processing run record
        run = ProcessingRun(
            dataset_id=dataset.id,
            user_id=current_user.id,
            config={'action': 'clean'},
            cleaning_log=processor.cleaning_log,
            success=True
        )
        db.session.add(run)
        db.session.commit()

        return jsonify({
            'success': True,
            'cleaning_log': processor.cleaning_log,
            'original_shape': df.shape,
            'cleaned_shape': cleaned_df.shape,
            'run_id': run.id
        })
    
    except Exception as e:
        return jsonify({'error': f'Error cleaning data: {str(e)}'}), 500


@app.route('/api/data/report', methods=['POST'])
@login_required
def generate_report():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON data required'}), 400
    
    dataset_id = data.get('dataset_id')
    config = data.get('config', {})
    
    if not dataset_id:
        return jsonify({'error': 'Dataset ID required'}), 400
    
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user owns the dataset or is admin
    if dataset.owner_id != current_user.id and current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        import pandas as pd
        
        # Read the dataset
        df = pd.read_csv(dataset.filepath)
        
        # Generate report
        report_data, cleaned_df = processor.create_report(df, config)
        
        # Save cleaned data if needed
        if config.get('save_cleaned', False):
            cleaned_filename = f"cleaned_{os.path.basename(dataset.filepath)}"
            cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
            cleaned_df.to_csv(cleaned_filepath, index=False)
        
        # Create processing run record
        run = ProcessingRun(
            dataset_id=dataset.id,
            user_id=current_user.id,
            config=config,
            cleaning_log=processor.cleaning_log,
            estimates=processor.estimates,
            plots_count=processor.plots_count,
            success=True
        )
        db.session.add(run)
        db.session.commit()
        
        # Create report record
        report = ReportRecord(
            run_id=run.id,
            user_id=current_user.id,
            report_data=report_data
        )
        db.session.add(report)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'report': report_data,
            'run_id': run.id,
            'report_id': report.id
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500


@app.route('/api/data/download', methods=['POST'])
@login_required
def download_data():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON data required'}), 400
    
    dataset_id = data.get('dataset_id')
    format_type = data.get('format', 'csv')
    
    if not dataset_id:
        return jsonify({'error': 'Dataset ID required'}), 400
    
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user owns the dataset or is admin
    if dataset.owner_id != current_user.id and current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        import pandas as pd
        
        # Read the dataset
        df = pd.read_csv(dataset.filepath)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format_type}') as tmp_file:
            if format_type == 'csv':
                df.to_csv(tmp_file.name, index=False)
            elif format_type == 'excel':
                df.to_excel(tmp_file.name, index=False)
            elif format_type == 'json':
                df.to_json(tmp_file.name, orient='records', indent=2)
            else:
                return jsonify({'error': 'Unsupported format'}), 400
            
            # Read file content and encode
            with open(tmp_file.name, 'rb') as f:
                file_content = f.read()
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            # Encode to base64
            file_base64 = base64.b64encode(file_content).decode('utf-8')
            
            return jsonify({
                'success': True,
                'filename': f"{dataset.filename.split('.')[0]}.{format_type}",
                'data': file_base64,
                'size': len(file_content)
            })
    
    except Exception as e:
        return jsonify({'error': f'Error downloading data: {str(e)}'}), 500


# Legacy route handlers for backward compatibility
@app.route('/me', methods=['GET'])
def legacy_me():
    return whoami()

@app.route('/login', methods=['GET', 'POST'])
def legacy_login():
    if request.method == 'GET':
        return jsonify({'error': 'Use POST method'}), 405
    return login()

@app.route('/register', methods=['GET', 'POST'])
def legacy_register():
    if request.method == 'GET':
        return jsonify({'error': 'Use POST method'}), 405
    return register()

@app.route('/logout', methods=['GET', 'POST'])
def legacy_logout():
    return logout()

@app.route('/profile', methods=['GET', 'POST'])
def legacy_profile():
    return profile()

@app.route('/admin', methods=['GET'])
def legacy_admin():
    return admin_dashboard()

@app.route('/admin/summary', methods=['GET'])
def legacy_admin_summary():
    return admin_dashboard()

@app.route('/admin/user/<int:user_id>/role', methods=['POST'])
def legacy_admin_update_role(user_id):
    return admin_update_role(user_id)

@app.route('/upload', methods=['POST'])
def legacy_upload():
    return upload_file()

@app.route('/clean', methods=['POST'])
def legacy_clean():
    return clean_data()

@app.route('/report', methods=['POST'])
def legacy_report():
    return generate_report()

@app.route('/download_data', methods=['POST'])
def legacy_download():
    return download_data()


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
