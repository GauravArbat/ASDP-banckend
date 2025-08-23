FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas openpyxl plotly reportlab && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads/avatars && \
    mkdir -p instance

# Set environment variables
ENV SECRET_KEY=change-me
ENV UPLOAD_FOLDER=uploads
ENV SQLALCHEMY_DATABASE_URI=sqlite:///app.db
ENV FLASK_ENV=production

# Render provides PORT env var; default to 8000 for local
ENV PORT=8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the application
CMD ["sh", "-lc", "gunicorn -w 2 -k gthread -b 0.0.0.0:${PORT} --timeout 120 app:app"]




