import argparse
import sys

import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from PyQt5 import QtCore

from infer_main import cgr_detect


class SmokeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.timer = QTimer(self)
        # Load the UI file
        loadUi("un.ui", self)
        self.filepath=None
        self.video_capture=None
        self.label.setScaledContents(True)
        parser = argparse.ArgumentParser()
        parser.add_argument('--cgr_conf', type=float,
                            default=0.4, help='香烟检测阈值')
        parser.add_argument('--skeleton', type=bool,
                            default=False, help='是否画出骨架')
        parser.add_argument('--cig_box', type=bool,
                            default=False, help='是否画出香烟框')
        parser.add_argument('--threshold', type=int,
                            default=50, help='连续检测阈值（不建议改动）')
        self.opt = parser.parse_args()
        self.init_ui()

    def init_ui(self):

        self.timer.timeout.connect(self.update_frame)
        self.loadvideo.clicked.connect(self.load_file)
        self.loadcam.clicked.connect(self.load_dir)
        self.start_process.clicked.connect(self.start)
        self.stop_process.clicked.connect(self.stop)
        self.replay.clicked.connect(self.replayer)

        self.skeleton.setChecked(self.opt.skeleton)
        self.cig.setChecked(self.opt.cig_box)
        self.skeleton.stateChanged.connect(self.box_change)
        self.cig.stateChanged.connect(self.box_change)

        self.cgr_Slider.setMinimum(25)
        self.cgr_Slider.setMaximum(100)
        self.cgr_Slider.setSingleStep(1)
        self.cgr_Slider.setValue(int(self.opt.cgr_conf*100))
        self.cgr_label.setText(str(self.opt.cgr_conf))
        self.cgr_Slider.valueChanged.connect(self.value_change)

        self.detect_Slider.setMinimum(0)
        self.detect_Slider.setMaximum(100)
        self.detect_Slider.setSingleStep(1)
        self.detect_Slider.setValue(self.opt.threshold)
        self.detect_label.setText(str(self.opt.threshold))
        self.detect_Slider.valueChanged.connect(self.value_change)

        self.position.sliderMoved.connect(self.set_position)

    def box_change(self):
        self.opt.skeleton=self.skeleton.isChecked()
        self.opt.cig_box= self.cig.isChecked()

    def value_change(self):
        self.opt.cgr_conf=self.cgr_Slider.value()/100
        self.opt.threshold=self.detect_Slider.value()
        self.cgr_label.setText(str(self.cgr_Slider.value()/100))
        self.detect_label.setText(str(self.detect_Slider.value()))


    def load_file(self):
        self.timer.stop()
        self.filepath=self.open_file_dialog()
        if self.filepath is not None:
            self.video_capture = cv2.VideoCapture(self.filepath)
            self.position.setMaximum(int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.choose_frame()

    def load_dir(self):
        self.timer.stop()
        self.filepath=self.caminput.text()
        if self.filepath is not None:
            if self.filepath=="0":
                self.video_capture = cv2.VideoCapture(0)
                self.video_capture.set(3, 1920)
                self.video_capture.set(4, 1080)
            else:
                self.video_capture = cv2.VideoCapture(self.filepath)
            self.position.setMaximum(int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.choose_frame()

    def start(self):
        if self.filepath is not None:
            self.timer.start(33)

    def stop(self):
        self.timer.stop()

    def replayer(self):
        if self.filepath == "0":
            self.video_capture = cv2.VideoCapture(0)
            self.video_capture.set(3, 1920)
            self.video_capture.set(4, 1080)
        else:
            self.video_capture = cv2.VideoCapture(self.filepath)
        self.choose_frame()
        self.start()

    def set_position(self,positions):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, positions)

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
        frame,fps,tim=cgr_detect(frame,self.opt)
        self.fps.setText(f"FPS：{round(fps,1)}")
        self.process_time.setText(f"帧处理时间：{round(tim,3)} s")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
        self.position.setValue(int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)))

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    window = SmokeDetectionApp()
    window.show()
    sys.exit(app.exec_())
