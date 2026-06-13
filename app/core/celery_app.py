"""
Celery configuration for asynchronous task offloading.
"""
from celery import Celery

# Broker: RabbitMQ, Backend: Redis
# Explicitly matching the docker-compose.yml configuration
CELERY_BROKER_URL = "amqp://customuser:custompassword@rabbitmq:5672//"
CELERY_RESULT_BACKEND = "redis://redis:6379/0"

celery_app = Celery(
    "thyrax_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.worker.tasks"]
)

# Optional but recommended default configurations
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    redis_socket_keepalive=True,
    redis_retry_on_timeout=True,
)

from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "weekly-hallucination-eval": {
        "task": "app.worker.tasks.evaluate_weekly_hallucinations",
        "schedule": crontab(hour=0, minute=0, day_of_week="sunday"),
    }
}
