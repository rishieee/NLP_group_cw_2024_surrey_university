# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 4  # Number of worker processes
timeout = 120
