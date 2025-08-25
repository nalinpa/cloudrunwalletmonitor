import logging
import os
import sys

def setup_logging():
    """Setup logging for Cloud Functions without external dependencies"""
    
    log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
    
    # Cloud Functions automatically handle log routing
    logging.basicConfig(
        level=log_level,
        format='{"severity": "%(levelname)s", "message": "%(message)s", "timestamp": "%(asctime)s"}',
        stream=sys.stdout,
        force=True
    )
    
    # Suppress noisy logs
    logging.getLogger('werkzeug').setLevel(logging.WARNING)