# Docker Setup for DIA Backend

This guide explains how to containerize and run the DIA (Document Intelligence Analysis) backend using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, but recommended)

## Quick Start with Docker Compose

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd dia-backend
   ```

2. Create environment variables file:
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

4. The API will be available at `http://localhost:8000` and the Swagger documentation at `http://localhost:8000/docs`.

## Manual Docker Setup

If you prefer to run Docker commands manually:

1. Build the Docker image:
   ```bash
   docker build -t dia-backend .
   ```

2. Run the container:
   ```bash
   docker run -d \
     --name dia-backend \
     -p 8000:8000 \
     -v $(pwd)/uploads:/app/uploads \
     -v $(pwd)/logs:/app/logs \
     -v $(pwd)/temp:/app/temp \
     -e ENV=production \
     -e SECRET_KEY=your-secret-key \
     -e FIRST_SUPERUSER=admin@example.com \
     -e FIRST_SUPERUSER_PASSWORD=admin123 \
     -e BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:8000 \
     dia-backend
   ```

## Volume Mounts

The Docker setup includes the following volume mounts:

- `./uploads:/app/uploads`: For storing uploaded documents
- `./logs:/app/logs`: For application logs
- `./temp:/app/temp`: For temporary files

These volumes ensure that your data persists even if the container is restarted or rebuilt.

## Environment Variables

You can configure the application using environment variables. The most important ones are:

- `ENV`: Set to `production` for deployment
- `SECRET_KEY`: Used for JWT token generation
- `FIRST_SUPERUSER` and `FIRST_SUPERUSER_PASSWORD`: Admin user credentials
- `BACKEND_CORS_ORIGINS`: Comma-separated list of allowed origins

## Managing Dynamic Content

The Dockerfile is configured to handle dynamic content in the following ways:

1. **Uploaded Files**: The `uploads` directory is mounted as a volume, so all uploaded files are stored outside the container for persistence.

2. **Logs**: The `logs` directory is mounted as a volume for persistent logging.

3. **Temporary Files**: The `temp` directory is mounted for temporary file operations.

4. **Database**: For production use, consider using an external database instead of the SQLite file.

## Docker Image Optimization

The Dockerfile uses multi-stage builds to reduce the final image size:

1. The `builder` stage installs all build dependencies and compiles any required packages.
2. The final stage only includes runtime dependencies, reducing the image size.

## Security Considerations

- The application runs as a non-root user (`dia`) for improved security.
- Sensitive data can be provided via environment variables.
- Use secrets management for production deployments.

## Troubleshooting

If you encounter issues:

1. Check the logs:
   ```bash
   docker-compose logs -f
   ```

2. Verify that all required directories exist and have proper permissions.

3. Ensure environment variables are correctly set in your `.env` file or docker-compose.yml. 