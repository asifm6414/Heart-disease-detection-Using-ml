import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = int(os.getenv("GUNICORN_WORKERS", max(2, multiprocessing.cpu_count() // 2)))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
worker_class = "gthread"
worker_connections = 1000
timeout = 30
keepalive = 5
preload_app = True
accesslog = "-"
errorlog = "-"
