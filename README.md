# Document Intelligence Analysis Backend

A FastAPI-based backend service for document analysis, including features like table detection, text extraction, summarization, and template conversion.

## Features

- Document Upload & Management
- Text Extraction
- Table Detection
- Text Summarization
- Template Conversion
- User Authentication & Authorization
- Secure File Handling
- API Documentation (via Swagger UI)

## Tech Stack

- Python 3.9+
- FastAPI
- SQLAlchemy (ORM)
- PyTorch (ML Models)
- Poetry (Dependency Management)
- SQLite (Database)

## Prerequisites

- Python 3.9 or higher
- Poetry for dependency management
- pyenv (recommended for Python version management)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd dia-backend
```

2. Install Python 3.9+ using pyenv:
```bash
pyenv install 3.9.0
pyenv local 3.9.0
```

3. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

4. Install dependencies:
```bash
poetry install
```

5. Create environment file:
```bash
cp .env.example .env
# Edit .env with your configurations
```

6. Create necessary directories:
```bash
mkdir -p uploads app/ml/models
```

## Development

1. Activate the virtual environment:
```bash
poetry shell
```

2. Run the development server:
```bash
python app/main.py
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
│   │   ├── config.py
│   │   └── security.py
│   ├── db/
│   │   ├── session.py
│   │   └── models/
│   ├── ml/
│   │   └── models/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── tests/
│   ├── unit/
│   └── integration/
├── uploads/
├── .env
├── .env.example
├── pyproject.toml
└── README.md
```

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Run tests using pytest:
```bash
poetry run pytest
```

## Security

- JWT-based authentication
- File validation and sanitization
- CORS protection
- Rate limiting
- Input validation
- Secure password hashing

## License

[Your License Here]
