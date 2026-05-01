import logging
import os
from datetime import datetime

# Create log file name
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create logs directory ONLY
logs_dir = os.path.join(os.getcwd(), "logs")
try:
    os.makedirs(logs_dir, exist_ok=True)
except Exception:
    # If logs directory creation fails, use /tmp instead
    logs_dir = "/tmp"
    os.makedirs(logs_dir, exist_ok=True)

# Final log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__ == "__main__":
#     logging.info("Logging has started.")
#     print(f"Log file created at: {LOG_FILE_PATH}")
