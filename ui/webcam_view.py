import sys
from collections import deque, Counter

import cv2
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap


# ---------- Platform-specific tuning ----------
def _detect_platform_config():
    if sys.platform.startswith("win"):
        return {"backend": cv2.CAP_DSHOW, "max_probe_index": 5, "preview_interval_ms": 30}
    elif sys.platform == "darwin":
        return {"backend": cv2.CAP_AVFOUNDATION, "max_probe_index": 4, "preview_interval_ms": 33}
    else:
        return {"backend": cv2.CAP_V4L2, "max_probe_index": 5, "preview_interval_ms": 30}


_PLATFORM_CONFIG = _detect_platform_config()
CAMERA_BACKEND = _PLATFORM_CONFIG["backend"]
MAX_PROBE_INDEX = _PLATFORM_CONFIG["max_probe_index"]
PREVIEW_INTERVAL_MS = _PLATFORM_CONFIG["preview_interval_ms"]
PREDICT_EVERY_N_FRAMES = 24        # throttle classifier calls vs. preview fps
PREDICTION_HISTORY_LEN = 7         # rolling window per tracked face, for smoothing
CONFIDENCE_THRESHOLD = 0.55        # below this, show "Uncertain" instead of a label
FACE_CROP_PADDING = 0.15           # extra margin around each Haar box, as a fraction of w/h
MATCH_IOU_THRESHOLD = 0.3          # min overlap to treat a detection as "the same person"
MAX_MISSED_FRAMES = 24             # frames a tracked face can go undetected before being dropped

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

BOX_COLOR = (166, 227, 161)            # Catppuccin Mocha green, BGR for cv2 drawing
UNCERTAIN_BOX_COLOR = (137, 180, 250)  # Mocha blue, used when below confidence threshold
TEXT_BG_COLOR = (30, 30, 46)           # Mocha crust


def _iou(box_a, box_b):
    """Intersection-over-union of two (x, y, w, h) boxes, used to match a
    detection in this frame to a tracked face from the previous frame."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax + aw, bx + bw)
    inter_y1 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class TrackedFace:
    """One person currently being tracked across frames. Keeps its own
    smoothed prediction history so multiple people in frame don't share
    or overwrite each other's results."""

    def __init__(self, bbox):
        self.bbox = bbox
        self.history = deque(maxlen=PREDICTION_HISTORY_LEN)
        self.smoothed = None  # (label, confidence, is_confident)
        self.missed_frames = 0

    def record_prediction(self, result):
        self.history.append(result)
        labels = [label for label, _ in self.history]
        majority_label = Counter(labels).most_common(1)[0][0]
        confidences = [c for label, c in self.history if label == majority_label]
        avg_confidence = sum(confidences) / len(confidences)
        self.smoothed = (majority_label, avg_confidence, avg_confidence >= CONFIDENCE_THRESHOLD)


class WebcamView(QWidget):
    """Live webcam preview with device selection, multi-face detection, and
    a bounding box + smoothed prediction overlay for every person in frame.

    on_predict: callable(rgb_face_crop) -> (label: str, confidence: float)
    Called periodically (throttled) per tracked face, synchronously, on the
    UI thread.
    """

    def __init__(self, on_predict, parent=None):
        super().__init__(parent)
        self.on_predict = on_predict
        self.capture = None
        self.frame_count = 0
        self.tracked_faces = []  # list[TrackedFace]

        self._build_ui()
        self._populate_devices()
        self._start_camera(self.device_combo.currentData())

    # ---------- UI ----------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        self.camera_label = QLabel("Camera:")
        controls.addWidget(self.camera_label)

        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        controls.addWidget(self.device_combo, stretch=1)
        self.controls_row = controls
        layout.addLayout(controls)

        self.preview_label = QLabel("No camera")
        self.preview_label.setObjectName("webcamPreview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(360)
        layout.addWidget(self.preview_label, stretch=1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame)

    # ---------- Device enumeration ----------

    def _populate_devices(self):
        self.device_combo.blockSignals(True)
        self.device_combo.clear()

        found_any = False
        for index in range(MAX_PROBE_INDEX):
            cap = cv2.VideoCapture(index, CAMERA_BACKEND)
            if cap.isOpened():
                label = "Default Camera" if index == 0 else f"Camera {index}"
                self.device_combo.addItem(label, userData=index)
                found_any = True
            cap.release()

        self.device_combo.blockSignals(False)

        if not found_any:
            self.device_combo.addItem("No camera found", userData=None)

        show_selector = self.device_combo.count() > 1
        self.camera_label.setVisible(show_selector)
        self.device_combo.setVisible(show_selector)

    # ---------- Camera lifecycle ----------

    def _start_camera(self, index):
        self._stop_camera()

        if index is None:
            self.preview_label.setText("No camera available")
            return

        self.capture = cv2.VideoCapture(index, CAMERA_BACKEND)
        if not self.capture.isOpened():
            self.preview_label.setText(f"Could not open camera {index}")
            self.capture = None
            return

        self.frame_count = 0
        self.tracked_faces = []
        self.timer.start(PREVIEW_INTERVAL_MS)

    def _stop_camera(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _on_device_changed(self, _ui_index):
        selected = self.device_combo.currentData()
        self._start_camera(selected)

    # ---------- Frame loop ----------

    def _on_frame(self):
        if self.capture is None:
            return

        ok, frame_bgr = self.capture.read()
        if not ok:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detections = self._detect_faces(frame_bgr)
        self._update_tracked_faces(detections)

        self.frame_count += 1
        if self.frame_count % PREDICT_EVERY_N_FRAMES == 0:
            for face in self.tracked_faces:
                self._run_classifier(frame_rgb, face)

        self._draw_overlay(frame_rgb)
        self._show_frame(frame_rgb)

    def _detect_faces(self, frame_bgr):
        """Returns a list of (x, y, w, h) boxes, one per detected face —
        no longer just the single largest."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return [tuple(f) for f in faces]

    def _update_tracked_faces(self, detections):
        """Match this frame's detections against faces tracked from previous
        frames (by bounding-box overlap), so each person keeps their own
        prediction history instead of it resetting or bleeding into someone
        else's every frame."""
        unmatched_detections = list(detections)
        still_tracked = []

        for face in self.tracked_faces:
            best_match = None
            best_iou = MATCH_IOU_THRESHOLD
            for detection in unmatched_detections:
                score = _iou(face.bbox, detection)
                if score > best_iou:
                    best_iou = score
                    best_match = detection

            if best_match is not None:
                face.bbox = best_match
                face.missed_frames = 0
                unmatched_detections.remove(best_match)
                still_tracked.append(face)
            else:
                face.missed_frames += 1
                if face.missed_frames <= MAX_MISSED_FRAMES:
                    still_tracked.append(face)
                # else: dropped — person has left the frame

        # Any detections left over are new people who just entered frame
        for detection in unmatched_detections:
            still_tracked.append(TrackedFace(detection))

        self.tracked_faces = still_tracked

    def _padded_crop(self, frame_rgb, bbox):
        """Expand the Haar box by FACE_CROP_PADDING before cropping, so the
        classifier sees a framing closer to typical portrait/face-dataset
        crops rather than a tight Haar rectangle."""
        x, y, w, h = bbox
        frame_h, frame_w = frame_rgb.shape[:2]

        pad_w = int(w * FACE_CROP_PADDING)
        pad_h = int(h * FACE_CROP_PADDING)

        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(frame_w, x + w + pad_w)
        y1 = min(frame_h, y + h + pad_h)

        return frame_rgb[y0:y1, x0:x1]

    def _run_classifier(self, frame_rgb, face: TrackedFace):
        crop = self._padded_crop(frame_rgb, face.bbox)
        if crop.size == 0:
            return

        result = self.on_predict(crop)
        if result is None:
            return

        face.record_prediction(result)

    def _draw_overlay(self, frame_rgb):
        for face in self.tracked_faces:
            if face.missed_frames > 0:
                continue

            x, y, w, h = face.bbox

            if face.smoothed is not None:
                label, confidence, is_confident = face.smoothed
                box_color = BOX_COLOR if is_confident else UNCERTAIN_BOX_COLOR
                text = f"{label} ({confidence:.0%})" if is_confident else "Uncertain"
            else:
                box_color = UNCERTAIN_BOX_COLOR
                text = "..."

            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), box_color, 2)

            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(y - 10, text_h + 10)

            cv2.rectangle(
                frame_rgb,
                (x, label_y - text_h - baseline - 4),
                (x + text_w + 8, label_y + baseline - 4),
                box_color, -1
            )
            cv2.putText(
                frame_rgb, text, (x + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_BG_COLOR, 2
            )

    def _show_frame(self, frame_rgb):
        h, w, ch = frame_rgb.shape
        qimage = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(pixmap)

    # ---------- Cleanup ----------

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        if self.capture is not None:
            self.timer.start(PREVIEW_INTERVAL_MS)
        super().showEvent(event)