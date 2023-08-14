import sys

import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from infer_main import cgr_detect


class SmokeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.timer = QTimer(self)
        # Load the UI file
        loadUi("un.ui", self)
        self.filepath=None
        self.video_capture=None
        self.init_slots()
        self.cgr_conf=0.4


        # Connect the button click event to a function
        # self.detect_button.clicked.connect(self.detect_smoke)

    def init_slots(self):
        self.timer.timeout.connect(self.update_frame)
        self.loadvideo.clicked.connect(self.load_file)
        self.start_process.clicked.connect(self.start)
        self.stop_process.clicked.connect(self.stop)


    def detect_smoke(self):
        # Implement your smoke detection logic here
        # Update UI components accordingly
        pass


    def load_file(self):
        self.timer.stop()
        self.filepath=self.open_file_dialog()
        if self.filepath is not None:
            self.video_capture = cv2.VideoCapture(self.filepath)
            self.choose_frame()

    def start(self):
        if self.filepath is not None:
            self.timer.start(33)

    def stop(self):
        self.timer.stop()

    def open_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4);;All Files (*)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                selected_file = selected_files[0]
                print("Selected File:", selected_file)
                return selected_file

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self.video_capture.release()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)


    def choose_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self.video_capture.release()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmokeDetectionApp()
    window.show()
    sys.exit(app.exec_())
