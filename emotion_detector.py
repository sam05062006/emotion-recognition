"""
emotion_detector.py - EmotionDetector Class
============================================
Performs emotion classification using Hugging Face Transformers.

────────────────────────────────────────────
THREE APPROACHES — WHY THREE?
────────────────────────────────────────────

1. pipeline()
   ──────────
   The simplest, highest-level API.  pipeline("image-classification", …)
   wraps all the complexity into one call: it loads the model, preprocesses
   the image, runs inference, and formats the output.

   Use when: you just want predictions as fast as possible.

2. AutoTokenizer / AutoProcessor
   ────────────────────────────
   The mid-level API.  AutoTokenizer (for text) / AutoImageProcessor (for
   images) converts raw inputs into the exact tensor format the model needs.
   You control the preprocessing explicitly.

   Use when: you need custom pre/post-processing or batching.

3. AutoModel
   ──────────
   The lowest-level API.  Gives you the raw model outputs (logits — unnormalised
   scores).  You must apply softmax yourself to turn them into probabilities.

   Use when: you need model internals (attention maps, embeddings, fine-tuning).

────────────────────────────────────────────
WHAT IS A VISION TRANSFORMER (ViT)?
────────────────────────────────────────────
The model we use (trpakov/vit-face-expression) is a Vision Transformer —
the same attention mechanism used in text LLMs, but applied to image patches.
The image is split into 16×16 pixel patches, each patch is embedded as a
vector, and a Transformer encoder learns which patches "attend to" each other.
The final classification head maps the [CLS] token embedding to emotion labels.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import MODEL_NAME, MODEL_CACHE_DIR, EMOTION_LABELS

logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Classifies the emotion in a facial image using a pre-trained ViT model.

    Lazy-loads the model on first use to avoid slow startup times.
    All three Hugging Face APIs (pipeline, AutoProcessor, AutoModel) are
    demonstrated so you can compare them side-by-side.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name  = model_name
        self._pipeline   = None   # Approach 1: pipeline object
        self._processor  = None   # Approach 2: AutoImageProcessor
        self._model      = None   # Approach 3: AutoModelForImageClassification
        self._loaded     = False
        logger.info("EmotionDetector created. Model: %s", model_name)

    # ──────────────────────────────────────────
    #  LAZY MODEL LOADING
    # ──────────────────────────────────────────
    def load_model(self) -> bool:
        """
        Download (first time) or load from cache all three HF components.

        Returns True on success, False on failure.
        """
        if self._loaded:
            return True

        try:
            logger.info("Loading Hugging Face model '%s' …", self.model_name)

            # ── Approach 1: pipeline ──────────────────────────────────────
            # pipeline() is the highest-level API.  It automatically:
            #   - downloads the model weights and config
            #   - selects the right preprocessing for image-classification
            #   - returns human-readable results
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
    task="image-classification",
    model=self.model_name,
    top_k=None,
)
            logger.info("✓ pipeline() loaded.")

            # ── Approach 2: AutoImageProcessor ───────────────────────────
            # AutoImageProcessor (formerly AutoFeatureExtractor) handles:
            #   - resizing to the model's expected input size
            #   - normalising pixel values with the model's mean & std
            #   - converting to a PyTorch tensor
            from transformers import AutoImageProcessor
            self._processor = AutoImageProcessor.from_pretrained(
    self.model_name
)
            logger.info("✓ AutoImageProcessor loaded.")

            # ── Approach 3: AutoModelForImageClassification ───────────────
            # This is the raw neural-network model.  It returns "logits"
            # (unnormalised scores for each class) that you must post-process.
            from transformers import AutoModelForImageClassification
            self._model = AutoModelForImageClassification.from_pretrained(
    self.model_name
)
            self._model.eval()   # set to inference mode (disables dropout etc.)
            logger.info("✓ AutoModelForImageClassification loaded.")

            self._loaded = True
            logger.info("All three HF components loaded successfully.")
            return True

        except Exception as exc:
            logger.error("Model loading failed: %s", exc)
            return False

    # ──────────────────────────────────────────
    #  APPROACH 1 — pipeline()
    # ──────────────────────────────────────────
    def predict_with_pipeline(
        self, pil_image: Image.Image
    ) -> Optional[List[Dict]]:
        """
        Predict emotion using the high-level pipeline() API.

        Args:
            pil_image: a PIL.Image.Image in RGB mode.

        Returns:
            List of dicts like [{"label": "happy", "score": 0.92}, …]
            sorted by score descending, or None on error.

        WHEN TO USE:
          Use pipeline() when you want clean, simple code and don't need
          to customise preprocessing or batch multiple images manually.
        """
        if not self._loaded and not self.load_model():
            return None

        try:
            results = self._pipeline(pil_image)
            # Normalise label to lowercase
            for r in results:
                r["label"] = r["label"].lower()
            logger.debug("pipeline() results: %s", results)
            return results
        except Exception as exc:
            logger.error("pipeline() prediction failed: %s", exc)
            return None

    # ──────────────────────────────────────────
    #  APPROACH 2 — AutoImageProcessor + model
    # ──────────────────────────────────────────
    def predict_with_processor(
        self, pil_image: Image.Image
    ) -> Optional[List[Dict]]:
        """
        Predict emotion using AutoImageProcessor + AutoModelForImageClassification.

        This is the "middle road" — more control than pipeline(), simpler
        than raw tensors.

        Steps:
          1. Processor converts PIL image → dict of tensors
          2. Model receives the tensors → returns logits
          3. We apply softmax to turn logits into probabilities

        WHEN TO USE:
          Use this when you need custom batching, want to inspect the
          preprocessed tensors, or are building a production inference server.
        """
        if not self._loaded and not self.load_model():
            return None

        try:
            import torch

            # Step 1: preprocess
            # return_tensors="pt" → return PyTorch tensors
            inputs = self._processor(images=pil_image, return_tensors="pt")

            # Step 2: forward pass (no gradient tracking needed for inference)
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Step 3: logits → probabilities via softmax
            # outputs.logits shape: (batch_size=1, num_classes=7)
            probabilities = torch.nn.functional.softmax(
                outputs.logits, dim=-1
            )[0].tolist()   # [0] removes the batch dimension

            # Map probabilities to label names from the model's config
            id2label = self._model.config.id2label   # {0: "angry", 1: "disgust", …}
            results = [
                {"label": id2label[i].lower(), "score": prob}
                for i, prob in enumerate(probabilities)
            ]
            results.sort(key=lambda x: x["score"], reverse=True)

            logger.debug("AutoProcessor results: %s", results)
            return results

        except Exception as exc:
            logger.error("AutoProcessor prediction failed: %s", exc)
            return None

    # ──────────────────────────────────────────
    #  APPROACH 3 — Raw AutoModel tensors
    # ──────────────────────────────────────────
    def predict_with_automodel(
        self, pil_image: Image.Image
    ) -> Optional[Dict]:
        """
        Predict emotion using AutoModel at the lowest level.

        This approach gives you access to:
          - raw logits (for custom loss functions during fine-tuning)
          - hidden states (for embedding extraction)
          - attention weights (for visualisation / interpretability)

        WHEN TO USE:
          Use AutoModel when you're fine-tuning the model, extracting
          features, or need full control over every tensor operation.

        Returns a dict with keys: "scores", "logits", "top_label"
        """
        if not self._loaded and not self.load_model():
            return None

        try:
            import torch

            # Preprocess using the same processor as Approach 2
            inputs = self._processor(images=pil_image, return_tensors="pt")

            with torch.no_grad():
                outputs = self._model(**inputs)

            logits       = outputs.logits[0]                         # raw scores
            probabilities = torch.softmax(logits, dim=-1).tolist()  # 0-1 range
            id2label     = self._model.config.id2label

            scores = {
                id2label[i].lower(): round(prob, 6)
                for i, prob in enumerate(probabilities)
            }
            top_label = max(scores, key=scores.get)

            result = {
                "scores":    scores,
                "logits":    {id2label[i].lower(): float(logits[i]) for i in range(len(logits))},
                "top_label": top_label,
            }
            logger.debug("AutoModel results: top=%s scores=%s", top_label, scores)
            return result

        except Exception as exc:
            logger.error("AutoModel prediction failed: %s", exc)
            return None

    # ──────────────────────────────────────────
    #  UNIFIED DETECTION METHOD
    # ──────────────────────────────────────────
    def detect(
        self,
        pil_image: Image.Image,
        image_name: str = "unknown",
        image_size: Tuple[int, int] = (0, 0),
    ) -> Optional[Dict]:
        """
        Run emotion detection and return a structured result dict.

        Uses the pipeline() approach as the primary output (simple & reliable),
        while also running all three approaches so they can be displayed
        side-by-side in the UI.

        Args:
            pil_image:  PIL Image in RGB mode (already cropped to face if available)
            image_name: original filename (for database storage)
            image_size: (width, height) of the original uploaded image

        Returns:
            dict with keys:
              image_name, predicted_emotion, confidence, all_scores,
              image_width, image_height, detection_time,
              pipeline_results, processor_results, automodel_results
        """
        if not self._loaded and not self.load_model():
            logger.error("Model not loaded; aborting detection.")
            return None

        start_time = time.time()

        # ── Run all three approaches ──────────────────────────────────────
        pipeline_results  = self.predict_with_pipeline(pil_image)
        processor_results = self.predict_with_processor(pil_image)
        automodel_results = self.predict_with_automodel(pil_image)

        detection_time = round(time.time() - start_time, 4)

        # ── Build canonical output from pipeline results ──────────────────
        if not pipeline_results:
            logger.error("Primary pipeline returned no results.")
            return None

        top = pipeline_results[0]  # highest-confidence prediction
        all_scores = {r["label"]: round(r["score"], 6) for r in pipeline_results}

        result = {
            # ── fields stored in MySQL ──
            "image_name":        image_name,
            "predicted_emotion": top["label"],
            "confidence":        round(top["score"], 6),
            "all_scores":        all_scores,
            "image_width":       image_size[0],
            "image_height":      image_size[1],
            "detection_time":    detection_time,

            # ── extra fields for UI display ──
            "pipeline_results":  pipeline_results,
            "processor_results": processor_results,
            "automodel_results": automodel_results,
        }

        logger.info(
            "Detected emotion: '%s' (%.1f%%) in %.3fs",
            top["label"], top["score"] * 100, detection_time,
        )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded
