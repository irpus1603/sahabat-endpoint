# Gunicorn configuration for video streaming
import multiprocessing

# Server socket
bind = "0.0.0.0:8082"
backlog = 2048

# Worker processes - Use threads to avoid CoreML fork() crash
workers = 16
worker_class = "gthread"
threads = multiprocessing.cpu_count() * 2
worker_connections = 1000
timeout = 300  # 5 minutes timeout for video streaming
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "socaedge_gunicorn"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = None
# certfile = None

# For video streaming - important settings
worker_tmp_dir = None  # Use default temp dir for macOS compatibility
preload_app = False  # Disable preloading to avoid CoreML fork issues

# Fix CoreML fork() crash on macOS
def on_starting(server):
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

def worker_int(worker):
    # Handle worker interruption gracefully
    import os
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Graceful timeout for long-running requests
graceful_timeout = 30

# Keep workers alive during long requests
worker_keep_alive = True