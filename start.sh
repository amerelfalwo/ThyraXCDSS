#!/bin/bash

# Start Redis server in the background
echo "Starting Redis..."
redis-server --daemonize yes

# Start RabbitMQ server in the background
echo "Starting RabbitMQ..."
rabbitmq-server -detached

# Wait a few seconds for RabbitMQ to initialize
sleep 5

# Start Celery Worker in the background
echo "Starting Celery Worker..."
celery -A app.core.celery_app worker --loglevel=info &

# Start Celery Beat in the background (if you have periodic tasks)
echo "Starting Celery Beat..."
celery -A app.core.celery_app beat --loglevel=info &

# Start the FastAPI application on port 7860 (Hugging Face default)
echo "Starting FastAPI server..."
exec uvicorn main:app --host 0.0.0.0 --port 7860
