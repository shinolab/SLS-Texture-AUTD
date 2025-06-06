'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-05 16:55:37
LastEditors: Mingxin Zhang
LastEditTime: 2025-06-05 18:20:16
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QPixmap
from PyQt5 import QtGui
from pyautd3.link import SOEM, Simulator, OnLostFunc
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
import visualization
from multiprocessing import Process, Queue
from visualization import run_visualization_process

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

    # Visualize the waveform
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
            y = 0
            y = 0.5 * self._amplitude[0] * math.sin(self._frequency[0] * t ) + self._offset[0]\
              + 0.5 * self._amplitude[1] * math.sin(self._frequency[1] * t ) + self._offset[1]\
              + 0.5 * self._amplitude[2] * math.sin(self._frequency[2] * t ) + self._offset[2]
            y = y / 3
            path.lineTo(x, height - y * y_scale)
        painter.drawPath(path)

        # Draw the axis
        axis_thickness = 10
        painter.setPen(QPen(Qt.black, axis_thickness))
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)


# AUTD thread
class AUTDThread(QThread):
    # The signal to receive the SLS parameters
    SLS_para_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        # Connect the signal to the slot function
        self.SLS_para_signal.connect(self.SLSSignal)
        # The video thread of realsense
        self.video_thread = VideoThread()
        # Connect the signal of finger position to the slot function to update the focus
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

        time_step = 0.003   # The expected time step
        send_time = 0.0027  # The time cost of send infomation to AUTDs
        sleep_time = time_step - send_time  # The real sleep time
        theta = 0
        config = Silencer().disable()
        autd.send(config)

        print('press ctrl+c to finish...')

        try:
            while self._run_flag:
                stm_f = 10
                radius = 3

                # ... change the radius and height here
                x = self.coordinate[0]
                y = self.coordinate[1]
                # D435i depth start point: -4.2 mm
                # the height difference between the transducer origin and the camera: 52 mm
                height = self.coordinate[2] - 52 - 4.2
                
                # update the focus information
                p = radius * np.array([np.cos(theta), np.sin(theta), 0])
                p += np.array([x, y, height])
                # print(x, y, height)
                f = Focus(center + p)
                # tic = time.time()
                autd.send((self.m, f), timeout=timedelta(milliseconds=0))

                theta += 2 * np.pi * stm_f * time_step

                self.libc.HighPrecisionSleep(ctypes.c_float(sleep_time))  # cpp sleep function
                # toc = time.time()
                # print(toc-tic)

        except KeyboardInterrupt:
            pass

        print('finish.')
        autd.send(Stop())
        autd.close()


# The thread for the realsense
class VideoThread(QThread):
    # Send the image to visulization obtained by the realsense
    change_pixmap_signal = pyqtSignal(np.ndarray)
    # Send the finger position obtained by the realsense
    position_signal = pyqtSignal(np.ndarray)
    finger_x = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 848, 100, rs.format.z16, 300)
        
        self.positions = []
        self.timestamps = []

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
            # the height range: 0 ~ 21 cm
            filter = rs.threshold_filter(min_dist=0, max_dist=0.21)
            depth_frame = filter.process(depth_frame)
            depth_img = np.asanyarray(depth_frame.get_data())
            # the contact area, 100 x 100 pix
            depth_img = depth_img[int(H/2)-50:int(H/2)+50, int(W/2)-424:int(W/2)+424]
            
            # the avg height of 20 closest points
            min_x, min_y = np.where(depth_img > 0)
            if min_x.size == 0 or min_y.size == 0:
                continue
            
            nonzero_indices = np.argwhere(depth_img != 0)
            nonzero_values = depth_img[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            min_x, min_y = np.transpose(nonzero_indices[np.argsort(nonzero_values)[:20]])
            # mass_x and mass_y are the list of x indices and y indices of mass pixels
            cent_x = int(np.average(min_x))
            cent_y = int(np.average(min_y))
            height = depth_img[cent_x, cent_y]
            
            self.finger_x.emit(np.array([cent_y]))

            # calculate the coodinate using the fov
            # depth fov of D435i: 87° x 58°
            # rgb fov of D435i: 69° x 42°
            # ang_x = math.radians((cent_x - 50) / (W / 2) * (87 / 2))
            # ang_y = math.radians((cent_y - 50) / (H / 2) * (58 / 2))
            # x_dis = math.tan(ang_x) * height
            # y_dis = math.tan(ang_y) * height
            
            # use official functions to obtain the coodinate
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            cent_x_full_frame = cent_x + int(W/2) - 50
            cent_y_full_frame = cent_y + int(H/2) - 424
            point = rs.rs2_deproject_pixel_to_point(intrinsics,[cent_x_full_frame, cent_y_full_frame],height)
            x_dis, y_dis, height = point

            # print('X:', x_dis, 'Y:', y_dis, 'Z:', height)
            # send the coodinate signal
            self.position_signal.emit(np.array([y_dis, x_dis, height]))
            
            # temporal differential for obtaining the velocity
            current_time = time.time() 
            self.positions.append((x_dis, y_dis))
            self.timestamps.append(current_time)

            # 10 frames
            if len(self.positions) > 10:
                self.positions.pop(0)
                self.timestamps.pop(0)

            if len(self.positions) > 1:
                dx = self.positions[-1][0] - self.positions[0][0]
                dy = self.positions[-1][1] - self.positions[0][1]
                dt = self.timestamps[-1] - self.timestamps[0]
                vx = dx / dt
                vy = dy / dt
                # print("velocity_x:",vx)
                # print("velocity_y:",vy)
                # self.velocity_signal.emit(np.array([vx, vy]))
            
            # draw the rendering area
            cv2.circle(depth_img, (cent_y, cent_x), 5, (255, 255, 255), -1)
            depth_img = cv2.flip(cv2.applyColorMap(cv2.convertScaleAbs(depth_img), cv2.COLORMAP_JET), 0)
            self.change_pixmap_signal.emit(depth_img)

    def stop(self):
        # set run flag to False and waits for thread to finish
        self._run_flag = False
        self.wait()


class MainWindow(QWidget):
    def __init__(self, vis_queue=None, finger_position=None):
        super().__init__()
        self.setWindowTitle("Sequential Line Search")
        self.vis_queue = vis_queue  # 用于向 pygame 子进程发送参数
        self.finger_position = finger_position

        # Start the threads
        self.autd_thread = AUTDThread()
        self.video_thread = self.autd_thread.video_thread
        self.image_disp_w_h = 800

        self.image_label = QLabel(self)
        self.image_label.resize(self.image_disp_w_h, self.image_disp_w_h)

        whole_vbox = QVBoxLayout()
        whole_vbox.addWidget(self.image_label)

        self.horizontal_slider = QSlider(Qt.Horizontal)
        self.horizontal_slider.setRange(0, 999)
        self.horizontal_slider.setSliderPosition(0)

        self.vertical_sliders = []
        self.sinusoid_widget = SinusoidWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.sinusoid_widget)

        horizontal_layout = QHBoxLayout()
        labels = ["F_low", "A_low", "F_mid", "A_mid", "F_high", "A_high"]
        for i in range(len(labels)):
            vertical_slider = QSlider(Qt.Vertical)
            vertical_slider.setRange(0, 100)
            vertical_slider.setEnabled(False)
            self.vertical_sliders.append(vertical_slider)

            label = QLabel(labels[i])
            vertical_box = QVBoxLayout()
            vertical_box.addWidget(label, 1, Qt.AlignCenter | Qt.AlignTop)
            vertical_box.addWidget(vertical_slider, 0, Qt.AlignCenter | Qt.AlignTop)
            horizontal_layout.addLayout(vertical_box)

        layout.addLayout(horizontal_layout)
        layout.addWidget(self.horizontal_slider)

        self.optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=6)
        self.optimizer.set_hyperparams(kernel_signal_var=0.50,
                                       kernel_length_scale=0.10,
                                       kernel_hyperparams_prior_var=0.10)
        self.optimizer.set_gaussian_process_upper_confidence_bound_hyperparam(5.)

        self.horizontal_slider.valueChanged.connect(lambda value:
                                                    self.updateValues(_update_optimizer_flag=False))

        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda value: self.updateValues(_update_optimizer_flag=True))
        layout.addWidget(next_button)

        whole_vbox.addLayout(layout)
        self.setLayout(whole_vbox)

        self.updateValues(_update_optimizer_flag=False)

        # Connect signals
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.autd_thread.SLS_para_signal.connect(self.forward_to_visualization)
        self.video_thread.finger_x.connect(self.finger_position_visualization)

        # Start threads
        self.video_thread.start()
        self.autd_thread.start()
        self.autd_thread.SLS_para_signal.emit(np.array(self.para_list))

    def closeEvent(self, event):
        self.video_thread.stop()
        self.autd_thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_disp_w_h, self.image_disp_w_h, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def updateSlider(self, value, index):
        p = value / 100.0
        if index == 0:
            self.para_list[0] = int(10 + p * 20)
        if index == 1:
            self.para_list[1] = p
        if index == 2:
            self.para_list[2] = int(30 + p * 70)
        if index == 3:
            self.para_list[3] = p
        if index == 4:
            self.para_list[4] = int(100 + p * 200)
        if index == 5:
            self.para_list[5] = p

        self.autd_thread.SLS_para_signal.emit(np.array(self.para_list))
        self.sinusoid_widget.setAmplitude([self.para_list[1], self.para_list[3], self.para_list[5]])
        self.sinusoid_widget.setFrequency([self.para_list[0], self.para_list[2], self.para_list[4]])

    def updateValues(self, _update_optimizer_flag):
        slider_position = self.horizontal_slider.value() / 999.0

        if _update_optimizer_flag:
            self.optimizer.submit_feedback_data(slider_position)
            print('Next')

        optmized_para = self.optimizer.calc_point_from_slider_position(slider_position)

        freq_l = int(10 + optmized_para[0] * 20)
        amp_l = optmized_para[1]
        freq_m = int(30 + optmized_para[2] * 70)
        amp_m = optmized_para[3]
        freq_h = int(100 + optmized_para[4] * 200)
        amp_h = optmized_para[5]

        self.para_list = [freq_l, amp_l, freq_m, amp_m, freq_h, amp_h]

        self.autd_thread.SLS_para_signal.emit(np.array(self.para_list))
        self.sinusoid_widget.setAmplitude([amp_l, amp_m, amp_h])
        self.sinusoid_widget.setFrequency([freq_l, freq_m, freq_h])

        for i, vertical_slider in enumerate(self.vertical_sliders):
            vertical_slider.setValue(int(optmized_para[i] * vertical_slider.maximum()))

    @pyqtSlot(np.ndarray)
    def forward_to_visualization(self, para):
        if self.vis_queue is not None:
            self.vis_queue.put(list(para))
    
    @pyqtSlot(np.ndarray)
    def finger_position_visualization(self, pos):
        if self.finger_position is not None:
            self.finger_position.put(list(pos))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')  # macOS 推荐

    vis_queue = Queue()
    finger_position = Queue()
    vis_proc = Process(target=run_visualization_process, args=(vis_queue, finger_position))
    vis_proc.start()

    app = QApplication(sys.argv)
    window = MainWindow(vis_queue, finger_position)
    window.show()
    ret = app.exec_()

    vis_proc.terminate()
    vis_proc.join()
    sys.exit(ret)
