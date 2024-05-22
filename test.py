from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
import sys
import cv2

class ImageViewer(QWidget):
    def __init__(self, frame):
        super().__init__()
        self.initUI(frame)

    def initUI(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(q_img)

        label = QLabel(self)
        label.setPixmap(pixmap)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        self.setLayout(vbox)
        self.setWindowTitle('yolov8')
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    frame = cv2.imread('image.png')  # Replace with your frame
    viewer = ImageViewer(frame)
    sys.exit(app.exec_())
