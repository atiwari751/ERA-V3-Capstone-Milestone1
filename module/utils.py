import datetime
import sys
import logging
import os
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp-server")

def log(stage: str, msg: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{stage}] {msg}")

def mcp_log(level: str, message: str) -> None:
    """Log a message to stderr to avoid interfering with JSON communication"""
    sys.stderr.write(f"{level.upper()}: {message}\n")
    sys.stderr.flush()
    
    # Also log to the logger
    log_level = getattr(logging, level.upper() if level.upper() in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') else 'INFO')
    logger.log(log_level, message)

def load_prompt(filename: str) -> str:
    """Load a prompt from a file."""
    path = os.path.join(os.path.dirname(__file__), 'prompts', filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Could not find prompt file at {path}")
        return ""
