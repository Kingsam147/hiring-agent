"""
Configuration settings for the hiring agent application.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Global development mode flag
DEVELOPMENT_MODE = True
