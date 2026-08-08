"""
config.py - Central Configuration File
=======================================
All project settings are stored here so you only need to change one file
when switching between development, testing, or production environments.

Think of this as the "control panel" for the entire project.
"""

import os
import logging

# ─────────────────────────────────────────────
#  PROJECT PATHS
# ─────────────────────────────────────────────
# Base directory = the folder where this file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sub-folders created at install time
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")   # user-uploaded images
MODELS_DIR  = os.path.join(BASE_DIR, "models")    # cached HF model weights
LOGS_DIR    = os.path.join(BASE_DIR, "logs")      # application log files

# Ensure all directories exist when the app starts
for directory in [UPLOADS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ─────────────────────────────────────────────
#  DATABASE CONFIGURATION
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "3306")),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASSWORD", "samiksha2006"),  # ← change this
    "database": os.getenv("DB_NAME",     "emotion_recognition_db"),
}

# SQLAlchemy connection string
# Format: mysql+pymysql://user:password@host:port/database
DATABASE_URL = (
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ─────────────────────────────────────────────
#  HUGGING FACE MODEL
# ─────────────────────────────────────────────
# "trpakov/vit-face-expression" is a Vision Transformer fine-tuned on facial
# expression datasets. It outputs probabilities for 7 emotions.
MODEL_NAME = "trpakov/vit-face-expression"

# Where to cache model weights locally (avoids re-downloading every run)
MODEL_CACHE_DIR = MODELS_DIR

# The 7 emotion labels this model recognises (index matches model output)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ─────────────────────────────────────────────
#  IMAGE PROCESSING
# ─────────────────────────────────────────────
# Resize every uploaded image to this size before feeding it to the model
IMAGE_SIZE = (224, 224)   # (width, height) in pixels

# OpenCV Haar Cascade file shipped with the cv2 package
FACE_CASCADE_PATH = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

# Minimum pixel area a detected face must have to be accepted
MIN_FACE_SIZE = (30, 30)

# How many recent predictions to display in the history table
HISTORY_LIMIT = 20

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_FILE  = os.path.join(LOGS_DIR, "app.log")
LOG_LEVEL = logging.DEBUG  # change to INFO or WARNING in production

# Log format: timestamp | level | module | message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────
APP_TITLE       = "🧠 AI Image Emotion Recognition System"
APP_DESCRIPTION = (
    "Upload a facial photo and let AI detect the emotion — "
    "powered by Hugging Face Transformers & OpenCV."
)
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]
