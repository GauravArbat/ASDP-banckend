FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas openpyxl plotly reportlab && \
    pip install -r requirements.txt

COPY . .

# Create upload directories
RUN mkdir -p uploads/avatars

# Set environment variables
ENV SECRET_KEY=change-me
ENV UPLOAD_FOLDER=uploads
ENV SQLALCHEMY_DATABASE_URI=sqlite:///app.db

# Render provides PORT env var; default to 8000 for local
ENV PORT=8000
EXPOSE 8000

# Start the application
CMD ["sh", "-lc", "gunicorn -w 2 -k gthread -b 0.0.0.0:${PORT} app:app"]




