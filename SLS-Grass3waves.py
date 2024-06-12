'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-05 16:55:37
LastEditors: Mingxin Zhang
LastEditTime: 2024-06-13 00:53:34
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QPixmap
from PyQt5 import QtGui
from pyautd3.link import TwinCAT, SOEM, Simulator, OnLostFunc
from pyautd3.gain import Focus
from pyautd3 import AUTD3, Controller, Silencer, Stop
from pyautd3.modulation import Fourier, Sine
from datetime import timedelta
import time
import math
from math import pi
import pySequentialLineSearch
import pyrealsense2 as rs
import cv2
import os
import ctypes
import platform

DEVICE_WIDTH = AUTD3.device_width()
DEVICE_HEIGHT = AUTD3.device_height()

# drawing the waveform
class SinusoidWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 200)
        self._amplitude = [1.0, 1.0, 1.0]
        self._frequency = [1.0, 1.0, 1.0]
        self._offset = [0.5, 0.5, 0.5]
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(palette)

    def setAmplitude(self, amplitude):
        self._amplitude = amplitude
        self.update()

    def setOffset(self, offset):
        self._offset = offset
        self.update()

    def setPhase(self, phase):
        self._phase = phase
        self.update()

    def setFrequency(self, frequency):
        self._frequency = frequency
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.blue, 2))

        width = self.width()
        height = self.height()
        x_scale = width / (0.3 * math.pi)
        y_scale = height

        path = QPainterPath()
        path.moveTo(0, height / 2)

        line_thickness = 1
        painter.setPen(QPen(Qt.blue, line_thickness))
        for x in range(width):
            t = x / x_scale
            y = 0.5 * self._amplitude[0] * math.sin(self._frequency[0] * t ) + self._offset[0]\
              + 0.5 * self._amplitude[1] * math.sin(self._frequency[1] * t ) + self._offset[1]\
              + 0.5 * self._amplitude[2] * math.sin(self._frequency[2] * t ) + self._offset[2]
            y = y / 3
            path.lineTo(x, height - y * y_scale)
        painter.drawPath(path)

        # draw the axis
        axis_thickness = 10
        painter.setPen(QPen(Qt.black, axis_thickness))
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)


# AUTD thread
class AUTDThread(QThread):
    SLS_para_signal = pyqtSignal(np.ndarray)
    position_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        # connect the signal to the slot function
        self.SLS_para_signal.connect(self.SLSSignal)
        self.video_thread = VideoThread()
        self.video_thread.position_signal.connect(self.PositionSignal)

        self._run_flag = True

        # initial parameters
        self.coordinate = np.array([0., 0., 230.])
        self.m = Sine(100)

        # import the HighPrecisionSleep() method
        dll = ctypes.cdll.LoadLibrary
        self.libc = dll(os.path.dirname(__file__) + '/cpp/' + platform.system().lower() +
                         '/HighPrecisionTimer.so') 

    # slot function to accept SLS parameters
    @pyqtSlot(np.ndarray)
    def SLSSignal(self, SLS_para):
        self.m = Fourier(Sine(freq=int(SLS_para[0])).with_amp(SLS_para[1]))
        self.m.add_component(Sine(freq=int(SLS_para[2])).with_amp(SLS_para[3]))
        self.m.add_component(Sine(freq=int(SLS_para[4])).with_amp(SLS_para[5]))
        self.f_horizontal = SLS_para[6]
    
    # slot function to accept coordinates
    @pyqtSlot(np.ndarray)
    def PositionSignal(self, coordinate):
        self.coordinate = coordinate
    
    def on_lost(self, msg: ctypes.c_char_p):
        print(msg.decode('utf-8'), end="")
        os._exit(-1)

    def stop(self):
        # set run flag to False and waits for thread to finish
        self._run_flag = False
        self.wait()

    def run(self):
        W_cos = math.cos(math.pi/12) * DEVICE_WIDTH
    
        on_lost_func = OnLostFunc(self.on_lost)

        autd = (
            Controller.builder()
            .add_device(AUTD3.from_euler_zyz([W_cos - (DEVICE_WIDTH - W_cos), DEVICE_HEIGHT - 10 + 12.5, 0.], [pi, pi/12, 0.]))
            .add_device(AUTD3.from_euler_zyz([W_cos - (DEVICE_WIDTH - W_cos), -10 - 12.5, 0.], [pi, pi/12, 0.]))
            .add_device(AUTD3.from_euler_zyz([-W_cos + (DEVICE_WIDTH - W_cos),  12.5, 0.], [0., pi/12, 0.]))
            .add_device(AUTD3.from_euler_zyz([-W_cos + (DEVICE_WIDTH - W_cos), -DEVICE_HEIGHT - 12.5, 0.], [0., pi/12, 0.]))
            # .advanced_mode()
            .open_with(Simulator(8080))
            # .open_with(SOEM().with_on_lost(on_lost_func))
            # .open_with(TwinCAT())
        )

        print('================================== Firmware information ====================================')
        firm_info_list = autd.firmware_info_list()
        for firm in firm_info_list:
            print(firm)
        print('============================================================================================')

        center = autd.geometry.center + np.array([0., 0., 0.])

        time_step = 0.01
        theta = 0
        theta_horizontal = 0
        config = Silencer()
        autd.send(config)

        print('press ctrl+c to finish...')

        try:
            while self._run_flag:
                stm_f = 10
                radius = 8
                
                horizontal_range_r = 15

                # ... change the radius and height here
                x = self.coordinate[0]
                y = self.coordinate[1]
                # D435i depth start point: -4.2 mm
                # the height difference between the transducer surface and the camera: 9 mm
                height = self.coordinate[2] - 52 - 4.2
                
                # update the focus information
                p = radius * np.array([np.cos(theta), np.sin(theta), 0])
                p += np.array([x, y, height])
                
                horizontal_step = horizontal_range_r * np.array([np.cos(theta_horizontal), 0, 0])
                
                f = Focus(center + horizontal_step + p)
                tic = time.time()
                autd.send((self.m, f), timeout=timedelta(milliseconds=0))

                theta += 2 * np.pi * stm_f * time_step
                theta_horizontal += 2 * np.pi * self.f_horizontal * time_step
                toc = time.time()
                send_time = toc - tic

                self.libc.HighPrecisionSleep(ctypes.c_float(time_step - send_time))  # cpp sleep function

        except KeyboardInterrupt:
            pass

        print('finish.')
        autd.send(Stop())
        autd.close()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    position_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 848, 100, rs.format.z16, 300)

    def run(self):
        # Start streaming
        self.pipeline.start(self.config)

        while self._run_flag:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            
            W = depth_frame.get_width()
            H = depth_frame.get_height()
            # the height range: 0 ~ 23 cm
            filter = rs.threshold_filter(min_dist=0, max_dist=0.25)
            depth_frame = filter.process(depth_frame)
            depth_img = np.asanyarray(depth_frame.get_data())
            # the contact area
            depth_img = depth_img[int(H/2)-5:int(H/2)+5, int(W/2)-5:int(W/2)+5]
            
            min_x, min_y = np.where(depth_img > 0)
            if min_x.size == 0 or min_y.size == 0:
                continue

            nonzero_indices = np.argwhere(depth_img != 0)
            nonzero_values = depth_img[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            
            height = np.mean(nonzero_values)

            # print('X:', x_dis, 'Y:', y_dis, 'Z:', height)
            # send the coodinate signal
            self.position_signal.emit(np.array([0, 0, height]))
            
            # draw the rendering area
            # cv2.circle(depth_img, (cent_y, cent_x), 5, (255, 255, 255), -1)
            depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img), cv2.COLORMAP_JET)
            self.change_pixmap_signal.emit(depth_img)

    def stop(self):
        # set run flag to False and waits for thread to finish
        self._run_flag = False
        self.wait()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sequential Line Search")
        # Start the threads
        self.autd_thread = AUTDThread()
        # The realsense thread is the class member of AUTDThread()
        self.video_thread = self.autd_thread.video_thread

        self.image_disp_w_h = 320

        self.image_label = QLabel(self)
        self.image_label.resize(self.image_disp_w_h, self.image_disp_w_h)

        whole_hbox = QHBoxLayout()
        whole_hbox.addWidget(self.image_label)

        self.horizontal_slider = QSlider(Qt.Horizontal)
        self.horizontal_slider.setRange(0, 999)
        self.horizontal_slider.setSliderPosition(0)

        self.vertical_sliders = []

        self.sinusoid_widget = SinusoidWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.sinusoid_widget)

        horizontal_layout = QHBoxLayout()
        labels = ["F_low", "A_low", "F_mid", "A_mid", "F_high", "A_high", "Moving Speed"]
        for i in range(len(labels)):
            vertical_slider = QSlider(Qt.Vertical)
            vertical_slider.setRange(0, 100)
            self.vertical_sliders.append(vertical_slider)
            vertical_slider.valueChanged.connect(lambda value, idx=i: self.updateSlider(value, idx))

            label = QLabel(labels[i])

            vertical_box = QVBoxLayout()
            vertical_box.addWidget(label, 1, Qt.AlignCenter | Qt.AlignTop)
            vertical_box.addWidget(vertical_slider, 0, Qt.AlignCenter | Qt.AlignTop)

            horizontal_layout.addLayout(vertical_box)

        layout.addLayout(horizontal_layout)
        layout.addWidget(self.horizontal_slider)

        self.optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=7)
        
        self.optimizer.set_gaussian_process_upper_confidence_bound_hyperparam(5.)

        self.horizontal_slider.valueChanged.connect(lambda value: 
                                                    self.updateValues(_update_optimizer_flag=False))

        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda value: self.updateValues(_update_optimizer_flag=True))
        layout.addWidget(next_button)

        whole_hbox.addLayout(layout)
        self.setLayout(whole_hbox)

        self.updateValues(_update_optimizer_flag=False)
        # connect its signal to the update_image slot
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.video_thread.start()
        self.autd_thread.start()

    def closeEvent(self, event):
        self.video_thread.stop()
        self.autd_thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_disp_w_h, self.image_disp_w_h, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def updateSlider(self, value, index):
        p = value / 100.0
        if index == 0:
            self.para_list[0] = int(10 + p * 20)    # low frequency 10~30Hz
        if index == 1:
            self.para_list[1] = p
        if index == 2:
            self.para_list[2] = int(30 + p * 70)    # mid frequency 30~100Hz
        if index == 3:
            self.para_list[3] = p
        if index == 4:
            self.para_list[4] = int(100 + p * 200)  # high frequency 100~300Hz
        if index == 5:
            self.para_list[5] = p
        if index == 6:
            self.para_list[6] = 0.2 + p * 0.8
        self.autd_thread.SLS_para_signal.emit(np.array(self.para_list))
        self.sinusoid_widget.setAmplitude([self.para_list[1], self.para_list[3], self.para_list[5]])
        self.sinusoid_widget.setFrequency([self.para_list[0], self.para_list[2], self.para_list[4]])

    def updateValues(self, _update_optimizer_flag):
        slider_position = self.horizontal_slider.value() / 999.0

        if _update_optimizer_flag:
            self.optimizer.submit_feedback_data(slider_position)
            print('Next')

        optmized_para = self.optimizer.calc_point_from_slider_position(slider_position)

        #stm_freq = 3 + optmized_para[0] * 17     # STM_freq: 3~20Hz
        #radius = 2 + optmized_para[1] * 3       # STM radius: 2~5mm
        
        freq_l = int(10 + optmized_para[0] * 20)    # low frequency 10~30Hz
        amp_l = optmized_para[1]

        freq_m = int(30 + optmized_para[2] * 70)    # mid frequency 30~100Hz
        amp_m = optmized_para[3]

        freq_h = int(100 + optmized_para[4] * 200)  # high frequency 100~300Hz
        amp_h = optmized_para[5]
        
        speed = 0.2 + optmized_para[6] * 0.8
        
        self.para_list = [freq_l, amp_l, freq_m, amp_m, freq_h, amp_h, speed]
        
        self.autd_thread.SLS_para_signal.emit(np.array(self.para_list))

        # offset = -0.5 * amp + 1
        self.sinusoid_widget.setAmplitude([self.para_list[1], self.para_list[3], self.para_list[5]])
        self.sinusoid_widget.setFrequency([self.para_list[0], self.para_list[2], self.para_list[4]])

        i = 0
        for vertical_slider in self.vertical_sliders:
            vertical_slider.setValue(int(optmized_para[i] * vertical_slider.maximum()))
            i += 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
