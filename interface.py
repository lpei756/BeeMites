import sys
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# 全局加载训练过的模型
model = tf.keras.models.load_model('models/mite_mobilenet_model.h5')


def predict_image_class(image_path):
    # 加载和预处理图像
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # 缩放
    img_array = tf.expand_dims(img_array, 0)  # 转换为批量大小为1的数组

    # 使用模型进行预测
    predictions = model.predict(img_array)
    predicted_class = 'bee' if tf.argmax(predictions[0]) == 0 else 'varroa mite'

    return predicted_class


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'BeeMitesTensorFlow - Picture Recognition'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        # 创建小部件
        self.button = QPushButton('Load Image', self)
        self.button.clicked.connect(self.load_image)

        self.label = QLabel('Prediction will appear here.', self)
        self.label.setAlignment(Qt.AlignCenter)

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # 显示窗口
        self.show()

    def load_image(self):
        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.xpm *.jpg);;All Files (*)",
                                                    options=options)
        if image_path:
            prediction = predict_image_class(image_path)
            self.label.setText(f"Prediction: {prediction}")
            QMessageBox.information(self, "Prediction", f"The image is predicted as: {prediction}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
