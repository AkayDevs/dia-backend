# DIA Backend

Document Intelligence Analysis Backend Service built with FastAPI.

## Features

- User Authentication and Authorization
- Document Upload and Management
- Document Analysis (Table Detection, Text Extraction, etc.)
- RESTful API
- SQLite Database (easily upgradeable to PostgreSQL)
- JWT-based Authentication
- CORS Support
- Environment Variable Configuration

## Prerequisites

- Python 3.9+
- Poetry (Python package manager)
- pyenv (Python version manager)

## Setup

1. Set up Python environment:
```bash
pyenv install 3.9.0
pyenv local 3.9.0
```

2. Install dependencies:
```bash
poetry install
```

3. Create and configure `.env` file:
```bash
cp .env.example .env
# Edit .env with your configurations
```

4. Initialize the database:
```bash
poetry run alembic upgrade head
```

5. Run the development server:
```bash
poetry run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`
API documentation will be available at `http://localhost:8000/docs`

## Project Structure

```
dia-backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── routes/
│   ├── core/
│   ├── db/
│   ├── ml/
│   ├── schemas/
│   └── services/
├── tests/
│   ├── unit/
│   └── integration/
├── .env
├── poetry.lock
└── pyproject.toml
```

## Development

1. Run tests:
```bash
poetry run pytest
```

2. Format code:
```bash
poetry run black .
poetry run isort .
```

3. Run linting:
```bash
poetry run flake8
poetry run mypy .
```

## API Documentation

Once the server is running, you can access:
- Swagger UI documentation at `/docs`
- ReDoc documentation at `/redoc`

## Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS configuration
- Rate limiting
- Input validation
- Secure file uploads 