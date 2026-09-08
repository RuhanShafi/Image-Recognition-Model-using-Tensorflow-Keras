import sys
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
PREDICT_EVERY_N_FRAMES = 15  # throttle classifier calls vs. preview fps

# Face detector: bundled with opencv-python, no extra download/dependency
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

BOX_COLOR = (166, 227, 161)   # Catppuccin Mocha green, in BGR for cv2 drawing
TEXT_COLOR = (205, 214, 244)  # Catppuccin Mocha text


class WebcamView(QWidget):
    """Live webcam preview with device selection, face detection, and a
    bounding box overlay showing the classifier's live label + confidence.

    on_predict: callable(rgb_face_crop) -> (label: str, confidence: float)
    Called periodically (throttled), synchronously, on the UI thread.
    """

    def __init__(self, on_predict, parent=None):
        super().__init__(parent)
        self.on_predict = on_predict
        self.capture = None
        self.frame_count = 0

        self.last_bbox = None          # (x, y, w, h) in frame coordinates
        self.last_prediction = None    # (label, confidence)

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
        self.last_bbox = None
        self.last_prediction = None
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

        # Detect every frame — Haar cascade is cheap enough for this at
        # webcam resolution, unlike the classifier itself.
        self._detect_face(frame_bgr)

        self.frame_count += 1
        if self.last_bbox is not None and self.frame_count % PREDICT_EVERY_N_FRAMES == 0:
            self._run_classifier(frame_rgb)

        self._draw_overlay(frame_rgb)
        self._show_frame(frame_rgb)

    def _detect_face(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) == 0:
            self.last_bbox = None
            self.last_prediction = None
            return

        # Largest detected face — avoids the box jumping between people
        # if more than one face is in frame.
        self.last_bbox = max(faces, key=lambda f: f[2] * f[3])

    def _run_classifier(self, frame_rgb):
        x, y, w, h = self.last_bbox
        crop = frame_rgb[y:y + h, x:x + w]
        if crop.size == 0:
            return

        result = self.on_predict(crop)
        if result is not None:
            self.last_prediction = result

    def _draw_overlay(self, frame_rgb):
        if self.last_bbox is None:
            return

        x, y, w, h = self.last_bbox
        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), BOX_COLOR, 2)

        if self.last_prediction is not None:
            label, confidence = self.last_prediction
            text = f"{label} ({confidence:.0%})"

            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(y - 10, text_h + 10)

            cv2.rectangle(
                frame_rgb,
                (x, label_y - text_h - baseline - 4),
                (x + text_w + 8, label_y + baseline - 4),
                BOX_COLOR, -1
            )
            cv2.putText(
                frame_rgb, text, (x + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 46), 2  # Mocha crust, for contrast on the green fill
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