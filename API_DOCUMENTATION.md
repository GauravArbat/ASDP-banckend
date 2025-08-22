# ASDP Backend API Documentation

This is a clean, API-only backend for the ASDP (Automated Statistical Data Processing) application.

## Base URL
- Development: `http://localhost:5000`
- Production: Configure via environment variables

## Authentication

All endpoints except `/api/auth/login` and `/api/auth/register` require authentication.

### Login
**POST** `/api/auth/login`
```json
{
  "username": "your_username",
  "password": "your_password"
}
```

### Register
**POST** `/api/auth/register`
```json
{
  "username": "new_username",
  "email": "user@example.com",
  "password": "password",
  "confirm": "password"
}
```

### Logout
**POST** `/api/auth/logout`

### Get Current User
**GET** `/api/auth/me`

### Update Profile
**POST** `/api/auth/profile`
```json
{
  "username": "new_username",
  "email": "new_email@example.com",
  "password": "new_password"
}
```

### Upload Avatar
**POST** `/api/auth/avatar`
- Content-Type: `multipart/form-data`
- Field: `avatar` (image file)

## Admin Endpoints

### Dashboard
**GET** `/api/admin/dashboard`
- Requires admin role

### Update User Role
**POST** `/api/admin/user/{user_id}/role`
```json
{
  "role": "admin" // or "user"
}
```

## Data Processing Endpoints

### Upload Dataset
**POST** `/api/data/upload`
- Content-Type: `multipart/form-data`
- Field: `file` (CSV or Excel file)

### Clean Data
**POST** `/api/data/clean`
```json
{
  "dataset_id": 1
}
```

### Generate Report
**POST** `/api/data/report`
```json
{
  "dataset_id": 1,
  "config": {
    "save_cleaned": true
  }
}
```

### Download Data
**POST** `/api/data/download`
```json
{
  "dataset_id": 1,
  "format": "csv" // or "excel", "json"
}
```

## Response Format

All API responses follow this format:

### Success Response
```json
{
  "success": true,
  "data": {...}
}
```

### Error Response
```json
{
  "error": "Error message"
}
```

## Status Codes

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `500` - Internal Server Error

## CORS Configuration

The API is configured to accept requests from:
- `http://localhost:5173` (React development server)
- `http://127.0.0.1:5173`

## Environment Variables

- `SECRET_KEY` - Flask secret key
- `DATABASE_URL` - Database connection string
- `UPLOAD_FOLDER` - File upload directory
- `MAX_CONTENT_LENGTH` - Maximum file size (default: 16MB)

## Running the Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The server will start on `http://localhost:5000`
