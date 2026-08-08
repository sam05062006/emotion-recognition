"""
face_detector.py - FaceDetector Class
======================================
Uses OpenCV's Haar Cascade classifier to locate faces in an image.

WHAT IS A HAAR CASCADE?
  A Haar Cascade is a machine-learning-based object detector.
  It was trained by Viola & Jones (2001) to find patterns (called "features")
  that are statistically more common in face images than in background images.

  The detector scans the image at many scales and positions, checking at each
  window whether the local features match a face.

  OpenCV ships the pre-trained XML file with the package, so there's nothing
  extra to download.

WHY USE IT INSTEAD OF A DEEP LEARNING DETECTOR?
  - Zero extra dependencies (just cv2).
  - Very fast — runs in milliseconds on a CPU.
  - Good enough for frontal, well-lit faces.
  - A deep-learning detector (MTCNN, RetinaFace) is more accurate for
    rotated or partially occluded faces, but requires more setup.
"""

import logging
import os
import urllib.request
from typing import List, Tuple, Optional

import cv2
import numpy as np

from config import FACE_CASCADE_PATH, MIN_FACE_SIZE

logger = logging.getLogger(__name__)

# URL to download the Haar cascade XML if it's not already present
HAAR_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)


class FaceDetector:
    """
    Detects human faces in images using OpenCV's Haar Cascade classifier.

    Attributes:
        cascade: the loaded cv2.CascadeClassifier object.
        scale_factor: how much the image is shrunk at each scale step.
        min_neighbours: how many overlapping detections a region needs to
                        be considered a real face (higher = fewer false positives).
        min_face_size: minimum (width, height) in pixels for a face to count.
    """

    def __init__(
        self,
        cascade_path: str = FACE_CASCADE_PATH,
        scale_factor: float = 1.1,
        min_neighbours: int = 5,
        min_face_size: Tuple[int, int] = MIN_FACE_SIZE,
    ):
        self.cascade_path   = cascade_path
        self.scale_factor   = scale_factor
        self.min_neighbours = min_neighbours
        self.min_face_size  = min_face_size
        self.cascade: Optional[cv2.CascadeClassifier] = None
        self._load_cascade()

    # ──────────────────────────────────────────
    #  LOADING THE CASCADE
    # ──────────────────────────────────────────
    def _load_cascade(self) -> None:
        """
        Load the Haar cascade XML.

        OpenCV ships the cascade file inside its Python package.
        We first try to find it there, then fall back to FACE_CASCADE_PATH.
        """
        # Preferred: use the one bundled with the installed cv2 package
        bundled_path = os.path.join(
            os.path.dirname(cv2.__file__),
            "data",
            "haarcascade_frontalface_default.xml",
        )

        if os.path.exists(bundled_path):
            path_to_use = bundled_path
        elif os.path.exists(self.cascade_path):
            path_to_use = self.cascade_path
        else:
            logger.warning(
                "Haar cascade not found locally. Downloading from GitHub…"
            )
            path_to_use = self._download_cascade()

        self.cascade = cv2.CascadeClassifier(path_to_use)

        if self.cascade.empty():
            logger.error("CascadeClassifier failed to load from '%s'.", path_to_use)
            self.cascade = None
        else:
            logger.info("Haar cascade loaded from '%s'.", path_to_use)

    def _download_cascade(self) -> str:
        """Download the cascade XML file and save it to FACE_CASCADE_PATH."""
        try:
            os.makedirs(os.path.dirname(self.cascade_path) or ".", exist_ok=True)
            urllib.request.urlretrieve(HAAR_CASCADE_URL, self.cascade_path)
            logger.info("Cascade downloaded to '%s'.", self.cascade_path)
            return self.cascade_path
        except Exception as exc:
            logger.error("Failed to download cascade: %s", exc)
            return self.cascade_path   # may still fail; handled in _load_cascade

    # ──────────────────────────────────────────
    #  DETECTION
    # ──────────────────────────────────────────
    def detect_faces(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces in a BGR image.

        HOW IT WORKS:
          1. Convert to grayscale — Haar features are intensity-based,
             so colour information is irrelevant and grayscale is faster.
          2. Apply histogram equalisation — improves detection under
             varying lighting conditions.
          3. Call detectMultiScale — slides the detection window across
             the image at multiple scales.

        Args:
            image: BGR image as a NumPy array (shape H×W×3).

        Returns:
            A list of (x, y, w, h) tuples — one per detected face.
            (x, y) is the top-left corner; w and h are width and height.
        """
        if self.cascade is None:
            logger.error("Cascade not loaded. Cannot detect faces.")
            return []

        if image is None or image.size == 0:
            logger.warning("detect_faces received an empty image.")
            return []

        # Step 1: grayscale — CascadeClassifier only works on single-channel images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: histogram equalisation — spreads out pixel intensities so
        #         features are more visible in dark or overexposed images
        equalized = cv2.equalizeHist(gray)

        # Step 3: detectMultiScale
        #   scaleFactor  – each step shrinks image by this factor (1.1 = 10%)
        #   minNeighbors – how many detections per window before it's accepted
        #                  (higher = stricter = fewer false positives)
        #   minSize      – ignore windows smaller than this
        faces = self.cascade.detectMultiScale(
            equalized,
            scaleFactor  = self.scale_factor,
            minNeighbors = self.min_neighbours,
            minSize      = self.min_face_size,
            flags        = cv2.CASCADE_SCALE_IMAGE,
        )

        # detectMultiScale returns an empty tuple () when no faces are found,
        # not an empty list, so we standardise the output.
        if len(faces) == 0:
            logger.debug("No faces detected in image.")
            return []

        face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        logger.info("Detected %d face(s).", len(face_list))
        return face_list

    def get_largest_face(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Return the bounding box of the largest detected face.

        When multiple faces are present we pick the largest one (by area)
        because it's usually closest to the camera — and thus most likely
        the subject of the photo.
        """
        faces = self.detect_faces(image)
        if not faces:
            return None

        # Sort by area (w * h) descending; take the first (largest) face
        largest = max(faces, key=lambda f: f[2] * f[3])
        logger.debug("Largest face: x=%d y=%d w=%d h=%d", *largest)
        return largest

    def is_face_detected(self, image: np.ndarray) -> bool:
        """Convenience method: return True if at least one face is found."""
        return len(self.detect_faces(image)) > 0

    def annotate_faces(
        self,
        image: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes around all detected faces.

        Returns a new array (does NOT modify the original image).
        """
        annotated = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        return annotated
