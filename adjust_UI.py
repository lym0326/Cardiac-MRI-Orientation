
import os
import sys
import time
# import SimpleITK as sitk
import SimpleITK
from SimpleITK import GetImageFromArray, WriteImage, GetArrayFromImage
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox

import model
from PyQt5.QtCore import QThread, pyqtSignal
# 添加predict的多线程
class PredictionThread(QThread):
    finished_signal = pyqtSignal(object)

    def __init__(self,input_data):
        super().__init__()
        self.input_data = input_data

    def run(self):
        time.sleep(1)
        modul = model.utilss()
        result = modul.predict(self.input_data)
        self.finished_signal.emit(result)
# UI界面（使用qt5 designer)
class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setUpParams()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1123, 824)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("QMainWindow {\n"
"background-color: rgb(244, 243, 255)\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.show = QtWidgets.QLabel(self.centralwidget)
        self.show.setGeometry(QtCore.QRect(200, 260, 511, 551))
        self.show.setAutoFillBackground(False)
        self.show.setStyleSheet("QLabel {\n"
"background-color:white\n"
"}")
        self.show.setObjectName("show")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(-10, 0, 211, 841))
        self.frame.setAutoFillBackground(False)
        self.frame.setStyleSheet("QFrame {\n"
"background-color: rgb(230, 224, 255)\n"
"}\n"
"")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.layoutWidget = QtWidgets.QWidget(self.frame)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 90, 160, 30))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.open = QtWidgets.QPushButton(self.layoutWidget)
        self.open.setStyleSheet("QPushButton {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QPushButton::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")

        self.open.setObjectName("open")
        self.open.clicked.connect(lambda: self.openfiles())
        self.horizontalLayout.addWidget(self.open)

        self.options = QtWidgets.QComboBox(self.layoutWidget)
        self.options.setAutoFillBackground(False)
        self.options.setStyleSheet("QComboBox {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QComboBox::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")
        self.options.setObjectName("options")
        self.options.addItem("")
        self.options.addItem("")
        self.options.addItem("")
        self.horizontalLayout.addWidget(self.options)
        self.layoutWidget1 = QtWidgets.QWidget(self.frame)
        self.layoutWidget1.setGeometry(QtCore.QRect(30, 270, 155, 501))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.prediction = QtWidgets.QPushButton(self.layoutWidget1)
        self.prediction.setStyleSheet("QPushButton {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QPushButton::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")
        self.prediction.setObjectName("prediction")
        self.prediction.clicked.connect(lambda: self.predicting())
        self.verticalLayout.addWidget(self.prediction)

        self.adjustment = QtWidgets.QPushButton(self.layoutWidget1)
        self.adjustment.setStyleSheet("QPushButton {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QPushButton::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")
        self.adjustment.setObjectName("adjustment")
        self.verticalLayout.addWidget(self.adjustment)
        self.adjustment.clicked.connect(lambda: self.adjust())

        self.saveheader = QtWidgets.QPushButton(self.layoutWidget1)
        self.saveheader.setStyleSheet("QPushButton {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QPushButton::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")
        self.saveheader.setObjectName("saveheader")
        self.verticalLayout.addWidget(self.saveheader)
        self.saveheader.clicked.connect(lambda: self.saveAs(reply=True))
        self.saveimg = QtWidgets.QPushButton(self.layoutWidget1)
        self.saveimg.setStyleSheet("QPushButton {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QPushButton::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")
        self.saveimg.setObjectName("saveimg")
        self.verticalLayout.addWidget(self.saveimg)
        self.saveimg.clicked.connect(lambda: self.saveAs(reply=False))

        self.table = QtWidgets.QTableWidget(self.centralwidget)
        self.table.setGeometry(QtCore.QRect(710, 230, 391, 581))
        self.table.setAutoFillBackground(True)
        self.table.setColumnCount(3)
        self.table.setObjectName("table")
        self.table.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(2, item)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(200, 230, 511, 31))
        self.frame_2.setStyleSheet(" QFrame{\n"
"background-color: rgb(230, 224, 255)\n"
"}\n"
"")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.last = QtWidgets.QPushButton(self.frame_2)
        self.last.setGeometry(QtCore.QRect(300, 0, 51, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.last.sizePolicy().hasHeightForWidth())
        self.last.setSizePolicy(sizePolicy)
        self.last.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.last.setStyleSheet("QPushButton {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QPushButton::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")

        self.last.setObjectName("last")
        self.last.clicked.connect(lambda: self.lastclick())
        self.next = QtWidgets.QPushButton(self.frame_2)
        self.next.setGeometry(QtCore.QRect(400, 0, 51, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.next.sizePolicy().hasHeightForWidth())
        self.next.setSizePolicy(sizePolicy)
        self.next.setStyleSheet("QPushButton {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QPushButton::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")
        self.next.setObjectName("next")

        self.next.clicked.connect(lambda: self.nextclick())
        self.slice = QtWidgets.QLineEdit(self.frame_2)
        self.slice.setGeometry(QtCore.QRect(100, 0, 31, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slice.sizePolicy().hasHeightForWidth())
        self.slice.setSizePolicy(sizePolicy)
        self.slice.setStyleSheet("QLineEdit {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QLineEdit::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")
        self.slice.setObjectName("slice")
        self.totalslices = QtWidgets.QLineEdit(self.frame_2)
        self.totalslices.setGeometry(QtCore.QRect(160, 0, 31, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.totalslices.sizePolicy().hasHeightForWidth())
        self.totalslices.setSizePolicy(sizePolicy)
        self.totalslices.setStyleSheet("QLineEdit {\n"
"background-color: rgb(170, 170, 255)\n"
"}\n"
"QLineEdit::hover{\n"
"background-color:rgb(145, 112, 255)\n"
"\n"
"}")
        self.totalslices.setObjectName("totalslices")
        self.label_5 = QtWidgets.QLabel(self.frame_2)
        self.label_5.setGeometry(QtCore.QRect(140, 0, 16, 31))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(1099, 0, 51, 801))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(310, 0, 671, 221))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.instruction = QtWidgets.QLabel(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.instruction.sizePolicy().hasHeightForWidth())
        self.instruction.setSizePolicy(sizePolicy)
        self.instruction.setStyleSheet("font-size: 20px;\n"
"font-weight: bold;")
        self.instruction.setScaledContents(False)
        self.instruction.setObjectName("instruction")
        self.horizontalLayout_2.addWidget(self.instruction)
        self.picture = QtWidgets.QLabel(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picture.sizePolicy().hasHeightForWidth())
        self.picture.setSizePolicy(sizePolicy)
        self.picture.setText("")
        self.picture.setPixmap(QtGui.QPixmap("instruction1.png"))
        self.picture.setScaledContents(True)
        self.picture.setObjectName("picture")
        self.horizontalLayout_2.addWidget(self.picture)
        self.eight = QtWidgets.QLabel(self.layoutWidget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.eight.sizePolicy().hasHeightForWidth())
        self.eight.setSizePolicy(sizePolicy)
        self.eight.setText("")
        self.eight.setPixmap(QtGui.QPixmap("instruction2.png"))
        self.eight.setScaledContents(True)
        self.eight.setObjectName("eight")
        self.horizontalLayout_2.addWidget(self.eight)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def setUpParams(self):
        self.OpenPath = ""
        self.adjusted = False
        self.isOpen = False
        self.predicted = False

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.show.setText(_translate("MainWindow", "TextLabel"))
        self.open.setText(_translate("MainWindow", "Open"))
        self.options.setItemText(0, _translate("MainWindow", "C0"))
        self.options.setItemText(1, _translate("MainWindow", "LGE"))
        self.options.setItemText(2, _translate("MainWindow", "T2"))
        self.prediction.setText(_translate("MainWindow", "Predict"))
        self.adjustment.setText(_translate("MainWindow", "Adjust"))
        self.saveheader.setText(_translate("MainWindow", "Save(ChangeHeader)"))
        self.saveimg.setText(_translate("MainWindow", "Save(ChangImg)"))
        item = self.table.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Filename"))
        item = self.table.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Prediction"))
        item = self.table.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "afteradjust"))
        self.last.setText(_translate("MainWindow", "Last"))
        self.next.setText(_translate("MainWindow", "Next"))
        self.label_5.setText(_translate("MainWindow", "/"))
        self.instruction.setText(_translate("MainWindow", "instruction:"))
    # 定义openfiles函数，连接open按键
    def openfiles(self):

        try:
            self.imgName, imgType = QFileDialog.getOpenFileName(self, "", os.getcwd(),
                                                               filter="All Files (*);;nii.gz Files (*.nii.gz);;mha Files (*.mha);;nii Files (*.nii)")

            if "T2" in self.imgName:
                self.name = "T2"
                self.options.setCurrentIndex(2)
            elif "LGE" in self.imgName:
                self.name = "LGE"
                self.options.setCurrentIndex(1)
            elif "C0" in self.imgName:
                self.name = "C0"
                self.options.setCurrentIndex(0)
            self.OpenPath = self.imgName
            self.img = SimpleITK.ReadImage(self.imgName)
            self.data = GetArrayFromImage(self.img)
            self.data = np.array(self.data).transpose(1, 2, 0)
            self.spacing = self.img.GetSpacing()
            self.origin = self.img.GetOrigin()
            self.Direction = self.img.GetDirection()
            self.imgDim = min(self.data.shape)
            self.imgIndex = int(self.imgDim / 2)
            self.viewnow()
            self.isOpen = True
            self.adjusted = False
            self.predicted = False
        except:
            pass
    # 定义viewnow和viewanother函数，连接显示图像界面
    def viewnow(self):
        minDim = list(self.data.shape).index(min(self.data.shape))
        if minDim == 0:
            self.img = np.zeros((self.data.shape[1], self.data.shape[2], min(self.data.shape)))
            for i in range(min(self.data.shape)):
                self.data[:, :, i] = self.data[i, :, :]
        if minDim == 1:
            self.img = np.zeros((self.img.shape[0], self.img.shape[2], min(self.img.shape)))
            for i in range(min(self.img.shape)):
                self.img[:, :, i] = self.img[:, i, :]
        if minDim == 2:
            self.data = self.data
        self.imgDim = self.data.shape[2]
        slice_data = self.data[:, :, self.imgIndex]
        slice_data_norm = (slice_data - np.min(slice_data)) * 255 / (np.max(slice_data) - np.min(slice_data))
        slice_data_uint8 = slice_data_norm.astype(np.uint8)
        channels = 1
        height, width = slice_data_uint8.shape
        bytesPerLine = channels * width
        depth_img = slice_data_uint8.copy()
        qImg = QtGui.QImage(
                depth_img, width, height, bytesPerLine, QtGui.QImage.Format_Indexed8
            )
        pixmap_slice = QPixmap.fromImage(qImg)
        self.show.setPixmap(pixmap_slice.scaled(self.show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.totalslices.setText(str(self.imgDim))
        self.slice.setText(str(self.imgIndex + 1))
        self.row_index = self.table.rowCount()
        self.table.insertRow(self.row_index)
        name_item = QTableWidgetItem(self.imgName.split("/")[-1])
        name_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(self.row_index, 0, name_item)
        age_item = QTableWidgetItem("")
        age_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(self.row_index, 1, age_item)
        gender_item = QTableWidgetItem("")
        gender_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(self.row_index, 2, gender_item)

    def viewanother(self):
        minDim = list(self.data.shape).index(min(self.data.shape))

        if minDim == 0:
            self.data = np.zeros((self.data.shape[1], self.data.shape[2], min(self.data.shape)))
            for i in range(min(self.data.shape)):
                self.data[:, :, i] = self.data[i, :, :]
        if minDim == 1:
            self.data = np.zeros((self.data.shape[0], self.data.shape[2], min(self.data.shape)))
            for i in range(min(self.data.shape)):
                self.data[:, :, i] = self.data[:, i, :]
        if minDim == 2:
            self.data = self.data
        self.imgDim = self.data.shape[2]
        slice_data = self.data[:, :, self.imgIndex]
        slice_data_norm = (slice_data - np.min(slice_data)) * 255 / (np.max(slice_data) - np.min(slice_data))
        slice_data_uint8 = slice_data_norm.astype(np.uint8)
        channels = 1
        height, width = slice_data_uint8.shape
        bytesPerLine = channels * width
        depth_img = slice_data_uint8.copy()
        qImg = QtGui.QImage(
                depth_img, width, height, bytesPerLine, QtGui.QImage.Format_Indexed8
            )
        pixmap_slice = QPixmap.fromImage(qImg)
        self.show.setPixmap(pixmap_slice.scaled(self.show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.totalslices.setText(str(self.imgDim))
        self.slice.setText(str(self.imgIndex + 1))

    # 定义lastclick函数，连接last按键
    def lastclick(self):
        try:
            if self.isOpen:
                try:
                    if self.imgIndex > 0:
                        self.imgIndex -= 1
                        self.viewanother()
                except IndexError:
                    pass
        except:
            pass

    # 定义next函数，连接next按键
    def nextclick(self):
        try:
            if self.isOpen:
                try:
                    if self.imgIndex < self.imgDim - 1:
                        self.imgIndex += 1
                        self.viewanother()
                except IndexError:
                    pass
        except:
            pass

    # 定义predicting和prediction函数，设置多线程，连接predict按键
    def predicting(self):
        if self.OpenPath == "":
            QMessageBox.information(self, "Tip", "Please open file first:)")
            return 0, False
        else:
            input_data = self.OpenPath
            self.prediction_thread = PredictionThread(input_data)
            self.prediction_thread.start()
            self.prediction_thread.finished_signal.connect(self.prediction_done)

    def prediction_done(self, result):
        self.predic_tion = result
        gender_item = QTableWidgetItem(str(self.predic_tion))
        gender_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(self.row_index, 1, gender_item)
        self.predicted = True

    # 定义adjust函数，连接adjust按键,adjusting调整变化后图像的头文件
    def adjusting(self):
        DIrection = self.img.GetDirection()
        direction_3x3 = np.reshape(np.array(list(DIrection)), (3, 3))
        direction_3x1x3 = np.expand_dims(direction_3x3, axis=(1))
        a = np.zeros((3, 1, 3))
        a[0] = direction_3x1x3[0:1, :, :]
        a[1] = direction_3x1x3[1:2, :, :]
        a[2] = direction_3x1x3[2:, :, :]
        self.DIrection = np.array(a)

        if self.predic_tion == "000":
            self.DIrection = self.DIrection
        if self.predic_tion == "001":
            self.DIrection[0] = -self.DIrection[0]
        if self.predic_tion == "010":
            self.DIrection[1] = -self.DIrection[1]
        if self.predic_tion == "011":
            self.DIrection[:2, :] = -self.DIrection[:2, :]
        if self.predic_tion == "100":
            self.DIrection[[0, 1]] = self.DIrection[[1, 0]]
        if self.predic_tion == "101":
            self.DIrection[[0, 1]] = self.DIrection[[1, 0]]
            self.DIrection[0] = -self.DIrection[0]
        if self.predic_tion == "110":
            self.DIrection[[0, 1]] = self.DIrection[[1, 0]]
            self.DIrection[1] = -self.DIrection[1]
        if self.predic_tion == "111":
            self.DIrection[[0, 1]] = self.DIrection[[1, 0]]
            self.DIrection[:2, :] = -self.DIrection[:2, :]
        self.DIrection = tuple(np.reshape(self.DIrection, (9,)).tolist())
        return True


    def adjust(self):
        if self.predicted:
            self.data = GetArrayFromImage(self.img)
            self.data = np.array(self.data).transpose(1, 2, 0)
            if self.predic_tion == "000":
                self.data = self.data
            if self.predic_tion == "001":
                self.data = np.fliplr(self.data)
            if self.predic_tion == "010":
                self.data = np.flipud(self.data)
            if self.predic_tion == "011":
                self.data = np.flipud(np.fliplr(self.data))
            if self.predic_tion == "100":
                self.data = self.data.transpose((1, 0, 2))
            if self.predic_tion == "101":
                self.data = np.flipud(self.data.transpose((1, 0, 2)))
            if self.predic_tion == "110":
                self.data = np.fliplr(self.data.transpose((1, 0, 2)))
            if self.predic_tion == "111":
                self.data = np.flipud(np.fliplr(self.data.transpose((1, 0, 2))))

            self.direction = '000'
            self.viewanother()
            gender_item = QTableWidgetItem(str(self.direction))
            gender_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(self.row_index, 2, gender_item)
            self.adjusting()
            self.adjusted = True

            return True
        else:
            QMessageBox.information(self, "Tip", "Please predict first:)")

    # 定义saveAs函数，连接save(changeheader)和save(changeimage)按键
    def saveAs(self, reply):
        if self.isOpen:
            try:
                if self.adjusted or self.direction == "000":
                    savePath, imgType = QFileDialog.getSaveFileName(self, "", "untitled_" + self.imgName.split("/")[-1],
                                                                    filter="nii.gz Files (*.nii.gz);;mha Files (*.mha);;nii Files (*.nii);;All Files (*)")
                    save_img = np.zeros((self.data.shape[2], self.data.shape[0], self.data.shape[1]))
                    for i in range(self.data.shape[2]):
                        save_img[i, :, :] = self.data[:, :, i]
                    if reply:
                        self.DIrection = self.DIrection
                    else:
                        self.DIrection = self.Direction
                    DIrection = self.DIrection
                    img_save = GetImageFromArray(save_img)
                    img_save.SetDirection(DIrection)
                    img_save.SetOrigin(self.origin)
                    img_save.SetSpacing(self.spacing)
                    WriteImage(img_save, savePath)
                else:
                    pass
            except:
                pass
        else:
            QMessageBox.information(self, "Tip", "Please open file first:)")
            pass
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
