'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-05 16:55:37
LastEditors: Mingxin Zhang
LastEditTime: 2024-10-19 18:06:59
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
from pyautd3.modulation import Fourier, Sine, Static
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

 
# AUTD thread
class AUTDThread(QThread):
    def __init__(self):
        super().__init__()
        # The video thread of realsense
        self.video_thread = VideoThread()
        # Connect the signal of finger position to the slot function to update the focus
        self.video_thread.position_signal.connect(self.PositionSignal)

        self._run_flag = True

        # initial parameters
        self.coordinate = np.array([0., 0., 210., 1.0])
        self.m = Static().with_amp(self.coordinate[3])

        # import the HighPrecisionSleep() method
        dll = ctypes.cdll.LoadLibrary
        self.libc = dll(os.path.dirname(__file__) + '/cpp/' + platform.system().lower() +
                         '/HighPrecisionTimer.so') 
    
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
            # .open_with(Simulator(8080))
            .open_with(SOEM().with_on_lost(on_lost_func))
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
                radius = 5

                # ... change the radius and height here
                x = self.coordinate[0]
                y = self.coordinate[1]
                # D435i depth start point: -4.2 mm
                # the height difference between the transducer origin and the camera: 52 mm
                height = self.coordinate[2] - 52 - 4.2
                self.m = Static().with_amp(self.coordinate[3])
                
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

    def __init__(self):
        super().__init__()
        self._run_flag = True

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 848, 100, rs.format.z16, 300)

        self.heightmap = self.getHeightMap()
        
    def getHeightMap(self):
        # Test height map
        heightmap = np.load('10.npy')
        heightmap = heightmap[::4,::4]
        heightmap = heightmap[int(120/2)-50:int(120/2)+50, int(160/2)-50:int(160/2)+50]
        heightmap /= heightmap.max()
        return heightmap

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
            depth_img = depth_img[int(H/2)-50:int(H/2)+50, int(W/2)-50:int(W/2)+50]
            
            # the avg height of 10 closest points
            min_x, min_y = np.where(depth_img > 0)
            if min_x.size == 0 or min_y.size == 0:
                continue
            
            nonzero_indices = np.argwhere(depth_img != 0)
            nonzero_values = depth_img[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            min_x, min_y = np.transpose(nonzero_indices[np.argsort(nonzero_values)[:10]])
            # mass_x and mass_y are the list of x indices and y indices of mass pixels
            cent_x = int(np.average(min_x))
            cent_y = int(np.average(min_y))
            height = depth_img[cent_x, cent_y]

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
            cent_y_full_frame = cent_y + int(H/2) - 50
            point = rs.rs2_deproject_pixel_to_point(intrinsics,[cent_x_full_frame, cent_y_full_frame], height)
            x_dis, y_dis, height = point

            # print('X:', x_dis, 'Y:', y_dis, 'Z:', height)
            
            # send the coodinate signal
            gain = self.heightmap[cent_x, cent_y]
            self.position_signal.emit(np.array([y_dis, x_dis, height, gain]))
            
            # draw the rendering area
            cv2.circle(depth_img, (cent_y, cent_x), 5, (255, 255, 255), -1)
            
            heightmap = self.heightmap * 255
            heightmap = heightmap.astype(int)
            
            # heightmap = cv2.cvtColor(heightmap, cv2.COLOR_GRAY2RGB)

            depth_img = depth_img * 0.6 + heightmap * 0.4
            
            depth_img = cv2.convertScaleAbs(depth_img)

            self.change_pixmap_signal.emit(depth_img)

    def stop(self):
        # set run flag to False and waits for thread to finish
        self._run_flag = False
        self.wait()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Height Map Rendering")
        # Start the threads
        self.autd_thread = AUTDThread()
        # The realsense thread is the class member of AUTDThread()
        self.video_thread = self.autd_thread.video_thread

        self.image_disp_w_h = 320

        self.image_label = QLabel(self)
        self.image_label.resize(self.image_disp_w_h, self.image_disp_w_h)

        whole_hbox = QHBoxLayout()
        whole_hbox.addWidget(self.image_label)

        self.setLayout(whole_hbox)

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
