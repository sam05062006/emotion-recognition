"""
app.py - Streamlit Frontend & AppController
============================================
The application entry point.  Run with:

    streamlit run app.py

AppController orchestrates all other classes:
  - ImageProcessor   → OpenCV preprocessing
  - FaceDetector     → Haar-cascade face detection
  - EmotionDetector  → Hugging Face Transformer inference
  - DatabaseManager  → MySQL CRUD operations
"""

import io
import json
import logging
import os
import time
import uuid
from datetime import date, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    APP_TITLE, APP_DESCRIPTION, ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB, HISTORY_LIMIT, UPLOADS_DIR,
    LOG_FILE, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL, EMOTION_LABELS,
)
from database import EmotionResult
from emotion_detector import EmotionDetector
from face_detector import FaceDetector
from image_processor import ImageProcessor
from mysql_operations import DatabaseManager

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  EMOJI MAP  (for UI flavour)
# ─────────────────────────────────────────────
EMOTION_EMOJI = {
    "happy":    "😊",
    "sad":      "😢",
    "angry":    "😠",
    "fear":     "😨",
    "surprise": "😲",
    "neutral":  "😐",
    "disgust":  "🤢",
}

EMOTION_COLOR = {
    "happy":    "#FFD700",
    "sad":      "#4169E1",
    "angry":    "#DC143C",
    "fear":     "#8B008B",
    "surprise": "#FF8C00",
    "neutral":  "#808080",
    "disgust":  "#228B22",
}


# ─────────────────────────────────────────────
#  APP CONTROLLER CLASS
# ─────────────────────────────────────────────
class AppController:
    """
    Coordinates the entire application lifecycle.

    Responsibilities:
      - Initialise all service classes.
      - Handle the image upload → detect → store pipeline.
      - Render Streamlit pages.
    """

    def __init__(self):
        self.image_processor = ImageProcessor()
        self.face_detector   = FaceDetector()
        self.emotion_detector = EmotionDetector()
        self.db_manager      = DatabaseManager()
        self._db_connected   = False
        logger.info("AppController initialised.")

    # ──────────────────────────────────────────
    #  INITIALISATION
    # ──────────────────────────────────────────
    def initialize(self) -> None:
        """Connect to database and load the ML model (with Streamlit caching)."""
        if not self._db_connected:
            self._db_connected = self.db_manager.connect()
            if not self._db_connected:
                st.warning(
                    "⚠️ MySQL connection failed. Predictions will still work "
                    "but won't be saved. Check your DB settings in config.py."
                )

    # ──────────────────────────────────────────
    #  CORE DETECTION PIPELINE
    # ──────────────────────────────────────────
    def process_image(self, uploaded_file) -> dict:
        """
        Full pipeline: upload → preprocess → detect face → detect emotion → store.

        Args:
            uploaded_file: Streamlit UploadedFile object.

        Returns:
            result dict from EmotionDetector.detect() or {} on failure.
        """
        try:
            # 1. Load image from the uploaded bytes
            pil_image = Image.open(uploaded_file).convert("RGB")
            orig_w, orig_h = pil_image.size

            # 2. Convert to OpenCV BGR for preprocessing
            bgr_image = self.image_processor.load_from_pil(pil_image)

            # 3. Run the preprocessing pipeline (for UI display)
            _, steps = self.image_processor.preprocess_for_model(bgr_image)

            # 4. Detect face
            face_bbox = self.face_detector.get_largest_face(bgr_image)
            face_pil  = pil_image   # default: use full image

            if face_bbox:
                x, y, w, h = face_bbox
                # Crop the face region (with padding) from the BGR image
                face_bgr = self.image_processor.crop_region(bgr_image, x, y, w, h)
                # Annotate the original image with a green bounding box
                annotated_bgr = self.image_processor.draw_bounding_box(
                    bgr_image, x, y, w, h, label="Face"
                )
                # Convert cropped face to PIL for the model
                face_rgb = self.image_processor.bgr_to_rgb(face_bgr)
                face_pil = Image.fromarray(face_rgb)
            else:
                annotated_bgr = bgr_image

            # 5. Save the uploaded image to disk
            unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            save_path   = os.path.join(UPLOADS_DIR, unique_name)
            bgr_image_resized = self.image_processor.resize_image(bgr_image)
            cv2.imwrite(save_path, bgr_image_resized)

            # 6. Run emotion detection
            result = self.emotion_detector.detect(
                pil_image  = face_pil,
                image_name = uploaded_file.name,
                image_size = (orig_w, orig_h),
            )

            if result is None:
                return {}

            # 7. Attach UI extras
            result["original_bgr"]  = bgr_image
            result["annotated_bgr"] = annotated_bgr
            result["face_pil"]      = face_pil
            result["face_bbox"]     = face_bbox
            result["steps"]         = steps

            # 8. Save to MySQL
            if self._db_connected:
                record_id = self.db_manager.save_result(result)
                result["db_id"] = record_id

            return result

        except Exception as exc:
            logger.error("process_image error: %s", exc)
            return {}

    # ──────────────────────────────────────────
    #  STREAMLIT PAGE RENDERING
    # ──────────────────────────────────────────
    def render_upload_tab(self) -> None:
        """Render the main image-upload and prediction tab."""
        st.header("📤 Upload & Analyse")
        st.markdown(APP_DESCRIPTION)

        # ── File uploader ──────────────────────────────────────────────
        uploaded_file = st.file_uploader(
            label    = f"Choose an image ({', '.join(ALLOWED_EXTENSIONS)})",
            type     = ALLOWED_EXTENSIONS,
            help     = f"Maximum file size: {MAX_UPLOAD_SIZE_MB} MB",
        )

        if uploaded_file is None:
            st.info("👆 Upload a facial image to get started.")
            return

        # ── Image preview ─────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, use_container_width=True)

        # ── Run detection on button press ─────────────────────────────
        if st.button("🔍 Detect Emotion", type="primary", use_container_width=True):
            with st.spinner("Analysing image…"):
                result = self.process_image(uploaded_file)

            if not result:
                st.error("Detection failed. Check logs for details.")
                return

            # ── Face preview ───────────────────────────────────────────
            with col2:
                st.subheader("Detected Face")
                if result.get("face_bbox"):
                    # Show the annotated image (green bounding box)
                    annotated_rgb = cv2.cvtColor(
                        result["annotated_bgr"], cv2.COLOR_BGR2RGB
                    )
                    st.image(annotated_rgb, use_container_width=True)
                else:
                    st.image(
                        result["face_pil"], use_container_width=True,
                        caption="No face detected — using full image"
                    )

            # ── Main prediction result ─────────────────────────────────
            emotion   = result["predicted_emotion"]
            confidence= result["confidence"]
            emoji     = EMOTION_EMOJI.get(emotion, "🤔")
            color     = EMOTION_COLOR.get(emotion, "#888888")

            st.markdown("---")
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {color}22, {color}44);
                    border-left: 6px solid {color};
                    border-radius: 12px;
                    padding: 24px 32px;
                    margin: 16px 0;
                ">
                    <h1 style="margin:0; font-size:3rem;">{emoji} {emotion.upper()}</h1>
                    <h3 style="margin:8px 0 0; color:{color};">
                        Confidence: {confidence:.1%}
                    </h3>
                    <p style="margin:4px 0 0; color:#888; font-size:0.9rem;">
                        Detected in {result['detection_time']:.3f}s
                        {"· Face detected ✓" if result.get('face_bbox') else "· No face found"}
                        {"· Saved to DB (id=" + str(result.get('db_id')) + ")" if result.get('db_id') else ""}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── All emotion probabilities ──────────────────────────────
            self._render_probability_chart(result["all_scores"])

            # ── Three-approach comparison ──────────────────────────────
            self._render_approach_comparison(result)

            # ── Preprocessing steps ────────────────────────────────────
            self._render_preprocessing_steps(result.get("steps", {}))

    def _render_probability_chart(self, all_scores: dict) -> None:
        """Render a horizontal bar chart of all emotion probabilities."""
        st.subheader("📊 All Emotion Probabilities")

        if not all_scores:
            st.warning("No scores available.")
            return

        df = pd.DataFrame(
            list(all_scores.items()), columns=["Emotion", "Probability"]
        ).sort_values("Probability", ascending=False)

        df["Emoji"]      = df["Emotion"].map(EMOTION_EMOJI)
        df["Label"]      = df["Emoji"] + " " + df["Emotion"].str.capitalize()
        df["Percentage"] = (df["Probability"] * 100).round(2)

        # Streamlit bar chart
        chart_data = df.set_index("Label")["Percentage"]
        st.bar_chart(chart_data, height=300)

        # Numeric table
        display_df = df[["Label", "Percentage"]].copy()
        display_df.columns = ["Emotion", "Confidence (%)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    def _render_approach_comparison(self, result: dict) -> None:
        """Show output from all three Hugging Face approaches side by side."""
        st.subheader("🔬 Hugging Face API Comparison")
        st.markdown(
            "The same model, the same image — three different APIs. "
            "Results should be identical (minor float differences are normal)."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**① pipeline()**")
            st.caption(
                "Simplest API. One call does everything: preprocessing, "
                "inference, and output formatting."
            )
            pipeline_res = result.get("pipeline_results", [])
            if pipeline_res:
                for r in pipeline_res[:3]:
                    st.write(f"`{r['label']}`: {r['score']:.4f}")

        with col2:
            st.markdown("**② AutoImageProcessor + AutoModel**")
            st.caption(
                "Mid-level API. You control preprocessing and call the "
                "model manually — returns post-softmax probabilities."
            )
            proc_res = result.get("processor_results", [])
            if proc_res:
                for r in proc_res[:3]:
                    st.write(f"`{r['label']}`: {r['score']:.4f}")

        with col3:
            st.markdown("**③ Raw AutoModel (logits)**")
            st.caption(
                "Lowest level. Returns raw logits (unnormalised scores) — "
                "ideal for fine-tuning or extracting embeddings."
            )
            am_res = result.get("automodel_results", {})
            if am_res:
                logits = am_res.get("logits", {})
                sorted_logits = sorted(logits.items(), key=lambda x: x[1], reverse=True)
                for label, val in sorted_logits[:3]:
                    st.write(f"`{label}`: {val:.4f} (logit)")

    def _render_preprocessing_steps(self, steps: dict) -> None:
        """Display intermediate images from the OpenCV preprocessing pipeline."""
        if not steps:
            return

        st.subheader("🔧 OpenCV Preprocessing Pipeline")
        st.markdown(
            "Each step below transforms the image in a specific way before "
            "it reaches the neural network."
        )

        step_info = {
            "original_bgr": ("Original (BGR)", "As loaded by cv2.imread() — note colours may look wrong because OpenCV uses BGR, not RGB."),
            "resized":       ("Resized 224×224", "cv2.resize() shrinks or stretches the image to the model's required input dimensions."),
            "rgb":           ("BGR → RGB", "cv2.cvtColor(BGR2RGB) swaps the R and B channels. This is what the model actually receives."),
            "hsv":           ("RGB → HSV", "Hue-Saturation-Value colour space. Useful for colour-based operations and skin detection."),
        }

        cols = st.columns(len(step_info))
        for col, (key, (title, caption)) in zip(cols, step_info.items()):
            if key in steps:
                with col:
                    img = steps[key]
                    # Convert to uint8 if it's a float normalised array
                    if img.dtype != np.uint8:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    # OpenCV images are BGR; Streamlit expects RGB
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        if key in ("original_bgr", "resized"):
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img, caption=title, use_container_width=True)
                    st.caption(caption)

    def render_history_tab(self) -> None:
        """Render the prediction history tab with filtering and export."""
        st.header("📋 Prediction History")

        if not self._db_connected:
            st.warning("Database not connected — history unavailable.")
            return

        total = self.db_manager.get_total_count()
        st.metric("Total Predictions", total)

        # ── Filters ───────────────────────────────────────────────────
        with st.expander("🔍 Filter & Search", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                emotion_filter = st.selectbox(
                    "Filter by emotion",
                    ["All"] + EMOTION_LABELS,
                )
            with col2:
                start_date = st.date_input(
                    "From date", value=date.today() - timedelta(days=30)
                )
            with col3:
                end_date = st.date_input("To date", value=date.today())

        # ── Fetch records ─────────────────────────────────────────────
        if emotion_filter != "All":
            records = self.db_manager.get_by_emotion(emotion_filter)
        else:
            records = self.db_manager.get_by_date_range(start_date, end_date)
            if not records:
                records = self.db_manager.get_history()

        if not records:
            st.info("No predictions found for the selected filters.")
            return

        # ── Display table ─────────────────────────────────────────────
        df = pd.DataFrame(records)
        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].apply(lambda x: f"{x:.1%}")
        if "detection_time" in df.columns:
            df["detection_time"] = df["detection_time"].apply(
                lambda x: f"{x:.3f}s" if x else "—"
            )

        display_cols = [
            "id", "image_name", "predicted_emotion",
            "confidence", "image_width", "image_height",
            "detection_time", "created_at",
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True, hide_index=True)

        # ── Row-level actions ─────────────────────────────────────────
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            delete_id = st.number_input(
                "Delete record by ID", min_value=1, step=1, value=1
            )
            if st.button("🗑️ Delete", type="secondary"):
                if self.db_manager.delete_result(int(delete_id)):
                    st.success(f"Record {delete_id} deleted.")
                    st.rerun()
                else:
                    st.error(f"Record {delete_id} not found.")

        with col2:
            # ── CSV export ─────────────────────────────────────────────
            export_df = self.db_manager.export_to_dataframe(records)
            csv_bytes  = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label     = "⬇️ Export to CSV",
                data      = csv_bytes,
                file_name = "emotion_history.csv",
                mime      = "text/csv",
            )

    def render_stats_tab(self) -> None:
        """Render an overall statistics page."""
        st.header("📈 Emotion Statistics")

        if not self._db_connected:
            st.warning("Database not connected — statistics unavailable.")
            return

        stats = self.db_manager.get_emotion_stats()

        if not stats:
            st.info("No predictions yet. Upload an image to get started!")
            return

        # ── Bar chart ─────────────────────────────────────────────────
        stats_df = pd.DataFrame(
            list(stats.items()), columns=["Emotion", "Count"]
        ).sort_values("Count", ascending=False)

        stats_df["Emoji"] = stats_df["Emotion"].map(EMOTION_EMOJI)
        stats_df["Label"] = stats_df["Emoji"] + " " + stats_df["Emotion"].str.capitalize()

        st.bar_chart(stats_df.set_index("Label")["Count"], height=350)

        # ── Metric cards ──────────────────────────────────────────────
        cols = st.columns(min(len(stats), 4))
        for col, (emotion, count) in zip(cols, stats.items()):
            emoji = EMOTION_EMOJI.get(emotion, "🔵")
            col.metric(
                label = f"{emoji} {emotion.capitalize()}",
                value = count,
            )

    def render_about_tab(self) -> None:
        """Render project documentation / how-it-works tab."""
        st.header("ℹ️ About This Project")
        st.markdown("""
## 🧠 AI Image Emotion Recognition System

This project demonstrates a complete computer-vision + NLP pipeline:

### Technologies Used

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Web UI |
| Image I/O | OpenCV (cv2) | Read, resize, convert, draw |
| Face Detection | Haar Cascades | Locate faces in the image |
| Deep Learning | Hugging Face Transformers | ViT emotion classifier |
| Database | MySQL + SQLAlchemy | Store & query results |
| Language | Python 3 + OOP | Clean, modular code |

### How It Works

```
User uploads image
       ↓
OpenCV reads & preprocesses (BGR→RGB, resize to 224×224)
       ↓
Haar Cascade detects face → crop to face region
       ↓
Hugging Face ViT model classifies emotion (7 classes)
       ↓
Results displayed + stored in MySQL
```

### Three Hugging Face APIs

**1. `pipeline()`** — highest-level, one-liner:
```python
pipe = pipeline("image-classification", model="trpakov/vit-face-expression")
result = pipe(pil_image)
```

**2. `AutoImageProcessor` + `AutoModelForImageClassification`** — mid-level:
```python
processor = AutoImageProcessor.from_pretrained(model_name)
model     = AutoModelForImageClassification.from_pretrained(model_name)
inputs    = processor(images=pil_image, return_tensors="pt")
outputs   = model(**inputs)
probs     = softmax(outputs.logits, dim=-1)
```

**3. Raw `AutoModel`** — lowest-level (returns raw logits):
```python
# Same as above, but use outputs.logits directly for fine-tuning / embeddings
```

### OpenCV Colour Spaces

| Space | Channels | Best for |
|---|---|---|
| BGR | Blue, Green, Red | OpenCV default |
| RGB | Red, Green, Blue | PyTorch / Pillow |
| HSV | Hue, Saturation, Value | Colour detection, skin masks |
| Grayscale | Intensity only | Face detection (faster) |
""")


# ─────────────────────────────────────────────
#  STREAMLIT APP ENTRY POINT
# ─────────────────────────────────────────────
def main():
    # ── Page config ────────────────────────────────────────────────────
    st.set_page_config(
        page_title = "Emotion Recogniser",
        page_icon  = "🧠",
        layout     = "wide",
        initial_sidebar_state = "collapsed",
    )

    st.title(APP_TITLE)

    # ── Singleton controller (cached across re-runs) ───────────────────
    # st.session_state ensures we don't recreate the controller on every
    # Streamlit re-render, which would reload the model each time.
    if "controller" not in st.session_state:
        with st.spinner("Loading model and connecting to database…"):
            ctrl = AppController()
            ctrl.initialize()
            st.session_state.controller = ctrl

    controller: AppController = st.session_state.controller

    # ── Tab navigation ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Detect", "📋 History", "📈 Statistics", "ℹ️ About"
    ])

    with tab1:
        controller.render_upload_tab()

    with tab2:
        controller.render_history_tab()

    with tab3:
        controller.render_stats_tab()

    with tab4:
        controller.render_about_tab()


if __name__ == "__main__":
    main()
