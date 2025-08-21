# Project Structure

## ASDP (AI Survey Data Processor) Application
### Ministry of Statistics and Programme Implementation (MoSPI)

```
SIH/
├── 📁 Core Application Files
│   ├── app.py                 # Main Flask application with DataProcessor class
│   ├── config.py              # Configuration settings and parameters
│   ├── run.py                 # Application startup script
│   └── requirements.txt       # Python dependencies
│
├── 📁 Web Interface
│   └── templates/
│       └── index.html         # Main web interface (HTML/CSS/JavaScript)
│
├── 📁 Data & Testing
│   ├── sample_data.csv        # Sample survey dataset for testing
│   └── test_app.py            # Unit tests for the application
│
├── 📁 Installation & Setup
│   ├── setup.py               # Python package setup script
│   ├── run.bat                # Windows batch file for easy startup
│   ├── run.sh                 # Linux/Mac shell script for easy startup
│   └── .gitignore             # Git ignore patterns
│
├── 📁 Documentation
│   ├── README.md              # Comprehensive project documentation
│   └── PROJECT_STRUCTURE.md   # This file - project structure overview
│
└── 📁 Generated Directories (created at runtime)
    └── uploads/               # Directory for uploaded files (auto-created)
```

## File Descriptions

### Core Application Files

#### `app.py`
- **Main Flask application** with all routes and API endpoints
- **DataProcessor class** containing all data processing logic
- **Features**: File upload, data cleaning, statistical analysis, report generation
- **Key methods**: 
  - `load_data()` - Load CSV/Excel files
  - `impute_missing_values()` - Handle missing data
  - `detect_outliers()` - Identify outliers
  - `apply_weights()` - Apply survey weights
  - `calculate_estimates()` - Generate statistics
  - `generate_report()` - Create PDF/HTML reports

#### `config.py`
- **Configuration management** for all application settings
- **Processing parameters** for imputation, outlier detection, etc.
- **Environment-specific configs** (development, production, testing)
- **Application metadata** and version information

#### `run.py`
- **Application startup script** with browser auto-launch
- **User-friendly console output** with status messages
- **Error handling** and graceful shutdown

### Web Interface

#### `templates/index.html`
- **Modern, responsive web interface** using Bootstrap 5
- **Drag-and-drop file upload** with real-time validation
- **Interactive configuration panels** for processing options
- **Real-time progress tracking** and results display
- **Interactive visualizations** using Plotly.js
- **Professional styling** with custom CSS

### Data & Testing

#### `sample_data.csv`
- **Sample survey dataset** with realistic variables
- **Includes**: respondent_id, age, income, education_level, household_size, employment_status, health_score, weight
- **Used for**: Testing, demonstration, and user training

#### `test_app.py`
- **Comprehensive unit tests** for all DataProcessor methods
- **Test coverage**: Data loading, cleaning, analysis, reporting
- **Automated testing** with detailed output
- **Quality assurance** for production deployment

### Installation & Setup

#### `setup.py`
- **Python package configuration** for distribution
- **Dependency management** and metadata
- **Console script entry points** for easy command-line access
- **Professional packaging** standards

#### `run.bat` & `run.sh`
- **Cross-platform startup scripts** for easy deployment
- **Automatic dependency installation** and verification
- **User-friendly error messages** and guidance
- **One-click application launch**

### Documentation

#### `README.md`
- **Comprehensive project documentation** with usage examples
- **Installation instructions** for different platforms
- **Feature descriptions** and technical specifications
- **API documentation** and configuration options
- **Benefits and use cases** for MoSPI

## Key Features by File

### Data Processing (`app.py`)
- ✅ CSV/Excel file upload and parsing
- ✅ Missing value imputation (mean, median, KNN)
- ✅ Outlier detection (IQR, Z-score, Isolation Forest)
- ✅ Survey weight application
- ✅ Statistical estimation with confidence intervals
- ✅ Data visualization generation

### Web Interface (`templates/index.html`)
- ✅ Modern, responsive design
- ✅ Drag-and-drop file upload
- ✅ Real-time data analysis
- ✅ Interactive configuration panels
- ✅ Progress tracking and results display
- ✅ Download options (PDF, HTML, CSV)

### Testing (`test_app.py`)
- ✅ Unit tests for all core functions
- ✅ Data validation and error handling
- ✅ Statistical accuracy verification
- ✅ Report generation testing
- ✅ Performance and reliability checks

### Configuration (`config.py`)
- ✅ Environment-specific settings
- ✅ Processing parameter management
- ✅ Security and performance tuning
- ✅ Application metadata and versioning

## Technology Stack

### Backend
- **Flask** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning (KNN, Isolation Forest)
- **SciPy** - Statistical functions
- **ReportLab** - PDF generation

### Frontend
- **Bootstrap 5** - Responsive UI framework
- **Plotly.js** - Interactive visualizations
- **Vanilla JavaScript** - Dynamic functionality
- **HTML5/CSS3** - Modern web standards

### Development Tools
- **Python 3.8+** - Programming language
- **pip** - Package management
- **unittest** - Testing framework
- **Git** - Version control

## Deployment Options

### Local Development
```bash
python run.py
```

### Production Deployment
```bash
pip install -r requirements.txt
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Security Considerations

- ✅ File upload validation and sanitization
- ✅ Secure file handling with temporary files
- ✅ Input validation and error handling
- ✅ CORS configuration for web security
- ✅ Environment variable configuration

## Performance Optimizations

- ✅ Efficient data processing with pandas
- ✅ Memory management for large datasets
- ✅ Asynchronous file operations
- ✅ Optimized visualization rendering
- ✅ Caching for repeated operations

This project structure provides a complete, production-ready solution for ASDP (AI Survey Data Processor), specifically designed for the Ministry of Statistics and Programme Implementation (MoSPI) requirements.
