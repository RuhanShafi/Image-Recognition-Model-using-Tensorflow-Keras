from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel,
    QDockWidget, QPushButton, QSplitter
)
from PySide6.QtCore import Qt

from ui.theme import STYLESHEET, MOCHA
from ui.webcam_view import WebcamView
from ui.pdf_overlay import PDFOverlay


class MainWindow(QMainWindow):
    def __init__(self, predictor=None, pdf_path=None):
        super().__init__()
        self.predictor = predictor
        self.pdf_path = pdf_path
        self.setWindowTitle("Face Classifier")
        self.resize(1200, 700)
        self.setStyleSheet(STYLESHEET)

        self._build_central()
        self._build_toolbar_dock()

    def _build_central(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # --- Title ---
        title = QLabel("Face Classifier")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # --- Content area: webcam + PDF side-by-side, PDF pane hidden by default ---
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setObjectName("viewPanel")

        self.webcam_view = WebcamView(on_predict=self._run_prediction)
        self.splitter.addWidget(self.webcam_view)

        self.pdf_overlay = PDFOverlay(
            pdf_path=self.pdf_path,
            on_close=self._hide_pdf
        )
        self.pdf_overlay.setVisible(False)
        self.splitter.addWidget(self.pdf_overlay)

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter, stretch=1)
        self.setCentralWidget(central)

    def _build_toolbar_dock(self):
        dock = QDockWidget("Controls", self)
        dock.setObjectName("controlsDock")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        dock_content = QWidget()
        dock_layout = QVBoxLayout(dock_content)
        dock_layout.setContentsMargins(12, 12, 12, 12)
        dock_layout.setSpacing(10)
        dock_layout.setAlignment(Qt.AlignTop)

        self.webcam_btn = QPushButton("Webcam")
        self.static_btn = QPushButton("Static Image")
        self.pdf_toggle_btn = QPushButton("Toggle Model Info")
        self.pdf_toggle_btn.clicked.connect(self._toggle_pdf)

        for btn in (self.webcam_btn, self.static_btn, self.pdf_toggle_btn):
            dock_layout.addWidget(btn)

        dock.setWidget(dock_content)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _toggle_pdf(self):
        self.pdf_overlay.setVisible(not self.pdf_overlay.isVisible())

    def _hide_pdf(self):
        self.pdf_overlay.setVisible(False)

    def _run_prediction(self, image):
        if self.predictor is None:
            return None
        label, confidence = self.predictor.predict(image)
        return label, confidence


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())