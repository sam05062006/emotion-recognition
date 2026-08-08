# 🧠 AI Image Emotion Recognition System

A production-ready, beginner-friendly system that detects human emotions
from facial images using Hugging Face Transformers, OpenCV, MySQL, and Streamlit.

---

## 📸 What It Does

1. User uploads a photo through the browser.
2. OpenCV reads and preprocesses the image.
3. A Haar Cascade locates the face and crops it.
4. A Vision Transformer (ViT) model classifies the emotion.
5. Results (emotion + confidence) are displayed and stored in MySQL.

---

## 🗂️ Project Structure

```
EmotionRecognition/
│
├── app.py               ← Streamlit UI + AppController (entry point)
├── config.py            ← All settings: paths, DB credentials, model name
├── database.py          ← SQLAlchemy ORM model (EmotionResult table)
├── image_processor.py   ← OpenCV pipeline (ImageProcessor class)
├── emotion_detector.py  ← Hugging Face inference (EmotionDetector class)
├── face_detector.py     ← Haar cascade face detection (FaceDetector class)
├── mysql_operations.py  ← MySQL CRUD operations (DatabaseManager class)
│
├── models/              ← Hugging Face model weights cached here
├── uploads/             ← User-uploaded images saved here
├── logs/                ← Application log files
│
├── requirements.txt
├── schema.sql           ← MySQL CREATE TABLE script
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourname/EmotionRecognition.git
cd EmotionRecognition
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up MySQL

Start MySQL and run the schema script:

```bash
mysql -u root -p < schema.sql
```

This creates the `emotion_recognition_db` database and the `emotion_results` table.

### 5. Configure database credentials

Open `config.py` and update the `DB_CONFIG` section:

```python
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "YOUR_PASSWORD_HERE",   # ← change this
    "database": "emotion_recognition_db",
}
```

### 6. Run the application

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 🔬 Technologies Explained

### OpenCV (`cv2`)

| Function | What it does |
|---|---|
| `cv2.imread(path)` | Loads an image from disk as a BGR NumPy array |
| `cv2.resize(img, (w, h))` | Scales image to target dimensions |
| `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | Swaps R and B channels |
| `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` | Converts to Hue-Saturation-Value |
| `cv2.CascadeClassifier.detectMultiScale()` | Finds faces using Haar features |
| `cv2.rectangle()` | Draws a bounding box |
| `cv2.imwrite(path, img)` | Saves image to disk |

### Hugging Face Transformers (three APIs)

#### 1. `pipeline()` — simplest

```python
from transformers import pipeline

pipe = pipeline("image-classification", model="trpakov/vit-face-expression")
results = pipe(pil_image)   # → [{"label": "happy", "score": 0.92}, …]
```

#### 2. `AutoImageProcessor` + `AutoModelForImageClassification` — mid-level

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

processor = AutoImageProcessor.from_pretrained(model_name)
model     = AutoModelForImageClassification.from_pretrained(model_name)

inputs    = processor(images=pil_image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

probs = torch.softmax(outputs.logits, dim=-1)
```

#### 3. Raw `AutoModel` — lowest level (logits)

Same as above but you work directly with `outputs.logits` (unnormalised scores).
Use this when fine-tuning or extracting embeddings.

### MySQL + SQLAlchemy

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine  = create_engine("mysql+pymysql://user:pass@localhost/emotion_recognition_db")
Session = sessionmaker(bind=engine)

with Session() as session:
    session.add(EmotionResult(image_name="test.jpg", predicted_emotion="happy", ...))
    session.commit()
```

---

## 🗄️ Database Schema

```sql
CREATE TABLE emotion_results (
    id                INT PRIMARY KEY AUTO_INCREMENT,
    image_name        VARCHAR(255) NOT NULL,
    predicted_emotion VARCHAR(50)  NOT NULL,
    confidence        FLOAT        NOT NULL,
    all_scores        TEXT,               -- JSON string
    image_width       INT,
    image_height      INT,
    detection_time    FLOAT,
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🧠 Supported Emotions

| Emotion | Emoji |
|---|---|
| Happy | 😊 |
| Sad | 😢 |
| Angry | 😠 |
| Fear | 😨 |
| Surprise | 😲 |
| Neutral | 😐 |
| Disgust | 🤢 |

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)              │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Upload   │  │   History    │  │   Stats      │  │
│  │   Tab     │  │    Tab       │  │    Tab       │  │
│  └─────┬─────┘  └──────────────┘  └──────────────┘  │
│        │                                             │
│  ┌─────▼──────────────────────────────────────────┐  │
│  │              AppController                      │  │
│  └─────┬──────────┬──────────┬───────────────┬────┘  │
└────────┼──────────┼──────────┼───────────────┼────────┘
         │          │          │               │
    ┌────▼───┐ ┌────▼───┐ ┌───▼────┐   ┌──────▼──────┐
    │Image   │ │ Face   │ │Emotion │   │  Database   │
    │Proc.   │ │Detect. │ │Detect. │   │  Manager    │
    │(cv2)   │ │(Haar)  │ │(HF ViT)│   │(SQLAlchemy) │
    └────────┘ └────────┘ └───┬────┘   └──────┬──────┘
                               │               │
                          ┌────▼────┐    ┌─────▼─────┐
                          │  ViT    │    │  MySQL DB  │
                          │  Model  │    │            │
                          └─────────┘    └────────────┘
```

---

## 🚀 Future Improvements

- Real-time webcam emotion detection using `cv2.VideoCapture(0)`.
- Batch processing: upload a ZIP of images and analyse all at once.
- MTCNN or RetinaFace for more accurate face detection at angles.
- Model fine-tuning on a custom dataset using PyTorch Trainer API.
- REST API using FastAPI for integration with other services.
- Docker container for one-command deployment.
- User authentication for multi-user history tracking.

---

## 📄 Licence

MIT — free to use, modify, and distribute.
