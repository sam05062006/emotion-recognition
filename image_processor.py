"""
image_processor.py - ImageProcessor Class
==========================================
All OpenCV image-processing logic lives here.

WHAT IS OPENCV?
  OpenCV (Open Source Computer Vision Library) is a powerful library for
  reading, modifying, and analysing images and video.  In Python we import
  it as `cv2`.

KEY CONCEPT: COLOUR SPACES
  A colour space is just a way of representing colour.
  - BGR  – OpenCV's default (Blue-Green-Red).  Note: NOT RGB!
  - RGB  – Red-Green-Blue.  Used by most other tools (Pillow, Matplotlib).
  - HSV  – Hue-Saturation-Value.  Great for colour-based operations
           because "what colour is it?" (hue) is a single channel.
"""

import logging
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config import IMAGE_SIZE, UPLOADS_DIR

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles all image loading, preprocessing, and saving operations.

    Methods follow the natural pipeline order:
      load → resize → convert colour → normalize → save
    """

    def __init__(self, target_size: Tuple[int, int] = IMAGE_SIZE):
        """
        Args:
            target_size: (width, height) to resize images to.
                         Default from config.py is (224, 224).
        """
        self.target_size = target_size
        logger.info("ImageProcessor initialised. Target size: %s", target_size)

    # ──────────────────────────────────────────
    #  LOADING
    # ──────────────────────────────────────────
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Read an image from disk into a NumPy array using OpenCV.

        cv2.imread() returns a NumPy array with shape (height, width, 3)
        where the last dimension is [Blue, Green, Red] (BGR order).

        Returns None if the file doesn't exist or can't be decoded.
        """
        if not os.path.exists(image_path):
            logger.error("Image not found: %s", image_path)
            return None

        # cv2.IMREAD_COLOR = load as 3-channel BGR (ignores alpha channel)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("cv2.imread failed (unsupported format?): %s", image_path)
            return None

        h, w = image.shape[:2]
        logger.debug("Loaded image '%s' – size %dx%d px.", image_path, w, h)
        return image

    def load_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert a Pillow Image object to an OpenCV BGR NumPy array.

        Streamlit's file_uploader returns a Pillow Image, so we need this
        bridge between the two libraries.

        Pillow uses RGB order → cv2.cvtColor converts to BGR.
        """
        # Convert to RGB first (in case it's RGBA/palette mode)
        pil_rgb = pil_image.convert("RGB")
        # np.array turns the PIL Image into a (H, W, 3) uint8 array in RGB order
        rgb_array = np.array(pil_rgb)
        # cv2.COLOR_RGB2BGR swaps the R and B channels
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array

    # ──────────────────────────────────────────
    #  RESIZING
    # ──────────────────────────────────────────
    def resize_image(
        self,
        image: np.ndarray,
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize an image to the target dimensions.

        WHY RESIZE?
          Neural networks expect a fixed-size input (e.g. 224×224).
          Resizing ensures every image has the same dimensions before
          it enters the model.

        cv2.resize(src, (width, height))  ← note: OpenCV takes (W, H) not (H, W)
        cv2.INTER_AREA is best for shrinking (reduces moiré artifacts).
        cv2.INTER_LINEAR is the default and works well for enlarging.
        """
        target = size or self.target_size
        resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        logger.debug("Resized image to %s.", target)
        return resized

    # ──────────────────────────────────────────
    #  COLOUR-SPACE CONVERSIONS
    # ──────────────────────────────────────────
    def bgr_to_rgb(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Convert BGR → RGB.

        WHY?
          OpenCV loads images in BGR order (a historical quirk).
          The Hugging Face model and Pillow both expect RGB, so we must swap
          the channels before further processing.

        cv2.cvtColor is the recommended way — it handles all edge cases
        (different bit depths, channel counts) automatically.
        """
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    def rgb_to_bgr(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB → BGR (useful when saving with OpenCV after PIL processing)."""
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    def bgr_to_hsv(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Convert BGR → HSV (Hue-Saturation-Value).

        WHY USE HSV?
          In BGR/RGB, a single real-world colour (e.g. "red") spans a large
          range of (R,G,B) values depending on lighting.
          In HSV, hue is a single number (0-179 in OpenCV), so colour-based
          operations like "find skin tones" become much simpler.

          H = 0..179  (colour wheel angle)
          S = 0..255  (how vivid/saturated)
          V = 0..255  (brightness)
        """
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    def bgr_to_gray(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Convert BGR → grayscale (single channel).
        Used internally for face detection (Haar cascades work on grayscale).
        """
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # ──────────────────────────────────────────
    #  NORMALISATION
    # ──────────────────────────────────────────
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Scale pixel values from [0, 255] to [0.0, 1.0].

        WHY NORMALISE?
          Neural networks train faster and more stably when inputs are in
          a small numeric range.  Dividing by 255 ensures no single pixel
          value dominates the calculation.

          Before: [[0, 128, 255], ...]   dtype=uint8
          After:  [[0.0, 0.502, 1.0], ...]  dtype=float32
        """
        # astype converts the array dtype; dividing by 255.0 does the scaling
        return image.astype(np.float32) / 255.0

    # ──────────────────────────────────────────
    #  ANNOTATION
    # ──────────────────────────────────────────
    def draw_bounding_box(
        self,
        image: np.ndarray,
        x: int, y: int, w: int, h: int,
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw a rectangle and optional label on a copy of the image.

        cv2.rectangle(img, top_left, bottom_right, colour_BGR, line_thickness)
        cv2.putText(img, text, origin, font, scale, colour, thickness)

        NOTE: This modifies a copy so the original is not changed.
        """
        annotated = image.copy()

        # Draw the bounding rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        if label:
            # Choose a font size that scales with the bounding box width
            font_scale = max(0.4, w / 200)
            cv2.putText(
                annotated, label,
                (x, y - 8),                  # 8 pixels above the top edge
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,                 # anti-aliased = smoother text
            )

        return annotated

    # ──────────────────────────────────────────
    #  CROPPING
    # ──────────────────────────────────────────
    def crop_region(
        self,
        image: np.ndarray,
        x: int, y: int, w: int, h: int,
        padding: int = 20,
    ) -> np.ndarray:
        """
        Crop a rectangular region from an image with optional padding.

        NumPy array slicing: image[y1:y2, x1:x2]
        (rows first because arrays are row-major: shape = (H, W, C))

        The padding enlarges the crop slightly to include context around the
        face, which often improves emotion-recognition accuracy.
        """
        img_h, img_w = image.shape[:2]

        # Clamp coordinates to stay within image boundaries
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        cropped = image[y1:y2, x1:x2]
        logger.debug("Cropped region (%d,%d) – (%d,%d).", x1, y1, x2, y2)
        return cropped

    # ──────────────────────────────────────────
    #  SAVING
    # ──────────────────────────────────────────
    def save_image(self, image: np.ndarray, filename: str) -> Optional[str]:
        """
        Save a NumPy image array to disk using cv2.imwrite().

        cv2.imwrite() accepts BGR images and automatically encodes based on
        the file extension (.jpg → JPEG, .png → PNG, etc.).

        Returns the absolute path where the file was saved, or None on error.
        """
        try:
            save_path = os.path.join(UPLOADS_DIR, filename)
            success = cv2.imwrite(save_path, image)
            if not success:
                logger.error("cv2.imwrite failed for '%s'.", save_path)
                return None
            logger.info("Image saved to '%s'.", save_path)
            return save_path
        except Exception as exc:
            logger.error("save_image error: %s", exc)
            return None

    # ──────────────────────────────────────────
    #  FULL PIPELINE (convenience method)
    # ──────────────────────────────────────────
    def preprocess_for_model(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Run the complete preprocessing pipeline needed before emotion detection:
          1. Resize to model input size
          2. Convert BGR → RGB
          3. Normalise to [0, 1]

        Also returns a dict of intermediate results for the UI to display,
        showing each transformation step.

        Returns:
            processed_image: float32 RGB array ready for the model
            steps: dict of labelled intermediate images for visualisation
        """
        steps = {}

        # Step 1 – original (BGR, as loaded by OpenCV)
        steps["original_bgr"] = image.copy()

        # Step 2 – resize
        resized = self.resize_image(image)
        steps["resized"] = resized.copy()

        # Step 3 – BGR → RGB
        rgb = self.bgr_to_rgb(resized)
        steps["rgb"] = rgb.copy()

        # Step 4 – RGB → HSV (for display only; not fed to model)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        steps["hsv"] = hsv.copy()

        # Step 5 – normalise
        normalised = self.normalize_image(rgb)
        steps["normalised"] = normalised

        logger.debug("Preprocessing pipeline complete.")
        return normalised, steps

    def get_image_info(self, image: np.ndarray) -> dict:
        """Return basic metadata about an image array."""
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        return {
            "width":    w,
            "height":   h,
            "channels": channels,
            "dtype":    str(image.dtype),
            "size_bytes": image.nbytes,
        }
