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

RUN mkdir -p uploads/avatars

ENV SECRET_KEY=change-me

# Render provides PORT env var; default to 8000 for local
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-lc", "gunicorn -w 2 -k gthread -b 0.0.0.0:${PORT} app:app"]




