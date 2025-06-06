#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import math
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import sys
import random
from multiprocessing import Queue

# ---------- Constants ----------
SCREEN_W, SCREEN_H = 1000, 800
GRID_W = 900
BASELINE_Y = 380
RECT_W, RECT_H = 10, 80
AREA_CONST = RECT_W * RECT_H

PAD_OFFSET_X, PAD_OFFSET_Y = -130, 15 #contact area position of the finger from the adopted figure
PAD_W, PAD_H = 60, 40
RETURN_SPEED = 0.15

FREQ_RANGE = {
    "low": (10, 30),
    "mid": (30, 100),
    "high": (100, 300),
}
AMP_RANGE = (0.1, 0.9)
GRAY_LEVELS = {"low": 70, "mid": 130, "high": 200}

FINGER_IMG_PATH = "finger_image_upright-removebg-preview.png"  # Provide correct path
FINGER_SCALE = 2.0

params = [20, 0.5, 65, 0.5, 120, 0.5, 0.6]  # freq/amp/speed


class VisualizationThread(QThread):
    def __init__(self):
        super().__init__()
        
    @pyqtSlot(np.ndarray)
    def update_params(self, p):
        self.params = p
        print('received: ', self.params)

    # ---------- Utilities ----------
    def map_freq_to_cols(self, freq, freq_min, freq_max):
        # Use proportionality to frequency to determine count
        if freq <= 0:
            return 0
        base_density = 0.2  # scaling factor: higher = more fibers per Hz
        return int(base_density * freq + 5)  # Reduced total density by ~50%

    def map_amp_to_deform(self, amp):
        return AMP_RANGE[0] + amp * (AMP_RANGE[1] - AMP_RANGE[0])

    def build_mixed_fibers(self, distribution, freq_map, amp_map):
        total_cols = sum(distribution.values())
        labels = []
        for k, count in distribution.items():
            labels.extend([k] * count)

        sorted_labels = []
        positions = {k: 0 for k in distribution}
        weights = {k: distribution[k] / total_cols for k in distribution}
        for i in range(total_cols):
            best_k = min(distribution.keys(), key=lambda k: positions[k] / weights[k] if weights[k] > 0 else float('inf'))
            sorted_labels.append(best_k)
            positions[best_k] += 1

        spacings = []
        for fiber_type in sorted_labels:
            freq = freq_map[fiber_type]
            norm_freq = min(1.0, max(0.0, freq / 300.0))
            spacing = 20 - 20 * norm_freq  
            spacings.append(spacing)

        total_width = sum([RECT_W + s for s in spacings])
        x_start = (SCREEN_W - total_width) / 2
        fibers = []
        x_pos = x_start
        for i, fiber_type in enumerate(sorted_labels):
            freq = freq_map[fiber_type]
            spacing = spacings[i]
            fibers.append({
                "bx": x_pos + spacing / 2,  
                "by": BASELINE_Y,
                "scale": 1.0,
                "gray": GRAY_LEVELS[fiber_type],
                "type": fiber_type,
                "phase": random.uniform(0, 2 * math.pi),
                "freq": freq,
                "amp": amp_map[fiber_type]
            })
            x_pos += RECT_W + spacing

        return fibers

    def compute_jitter_x_offset(self, freq, fiber_pos, finger_pos, t):
        """
        Computes the X-direction jitter offset for a fiber based on frequency and proximity to the finger.
        """
        dx = fiber_pos[0] - finger_pos[0]
        dy = fiber_pos[1] - finger_pos[1]
        distance = math.hypot(dx, dy)

        if distance > 100:
            jitter_scale = 0.0
        elif distance > 20:
            jitter_scale = (100 - distance) / 80.0
        else:
            jitter_scale = 1.0

        norm = freq / 300
        norm = max(0, min(1, norm))

        jitter_amp = 2 * (1 - norm)             
        jitter_speed = 0.5 * math.pi * (5 + norm * 10)

        #offset_x = jitter_amp * jitter_scale * math.sin(jitter_speed * t)
        offset_x = jitter_amp * jitter_scale * ((t * jitter_speed) % 1.0 - 0.5) 
        return offset_x

    def draw_fiber_visual(self, surface, pos, size, freq, deform_scale, amp,is_near, t, phase, gray):
        norm = freq / 300.0
        base_h = size[1]
        raw_h = int(base_h * (amp**0.8) + 60)
        h = raw_h * deform_scale 
        w = size[0]
        rect = pygame.Rect(0, 0, w, h)
        rect.midbottom = pos

        matte_level = int(60 + 160 * (norm ** 1.5))
        matte_level = min(matte_level, 220)

        fill_surf = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)   
        fill_surf.fill((gray, gray, gray, 255)) 
        surface.blit(fill_surf, rect.topleft)
        pygame.draw.rect(surface, (90, 90, 90), rect, 1)
        

from multiprocessing import Queue

class VisualizationWithQueue(VisualizationThread):
    def __init__(self, param_queue: Queue, finger_position):
        super().__init__()
        self.param_queue = param_queue
        self.position_queue = finger_position
        self.params = [20, 0.5, 65, 0.5, 120, 0.5]

    def run(self):
        import pygame
        print('run pygame')
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Finger-Fabric: Wave-SLS Mapped")
        clock = pygame.time.Clock()

        finger_raw = pygame.image.load(FINGER_IMG_PATH).convert_alpha()
        fw, fh = finger_raw.get_size()
        finger_img = pygame.transform.smoothscale(finger_raw, (int(fw * FINGER_SCALE), int(fh * FINGER_SCALE)))

        finger_x, finger_y = SCREEN_W // 2, SCREEN_H // 2 + 45

        fibers = []
        last_params = self.params.copy()

        def update_fibers():
            freq_l, amp_l, freq_m, amp_m, freq_h, amp_h = self.params
            cols_l = self.map_freq_to_cols(freq_l, *FREQ_RANGE["low"])
            cols_m = self.map_freq_to_cols(freq_m, *FREQ_RANGE["mid"])
            cols_h = self.map_freq_to_cols(freq_h, *FREQ_RANGE["high"])
            distribution = {"low": cols_l, "mid": cols_m, "high": cols_h}
            freq_map = {"low": freq_l, "mid": freq_m, "high": freq_h}
            amp_map = {"low": amp_l, "mid": amp_m, "high": amp_h}
            return self.build_mixed_fibers(distribution, freq_map, amp_map), {
                "low": self.map_amp_to_deform(amp_l),
                "mid": self.map_amp_to_deform(amp_m),
                "high": self.map_amp_to_deform(amp_h),
            }

        fibers, max_deform = update_fibers()

        running = True
        while running:
            # 👇 监听参数更新
            while not self.param_queue.empty():
                self.params = self.param_queue.get()
            while not self.position_queue.empty():
                self.finger_position = self.position_queue.get()
                print(self.finger_position[0])
                finger_x = SCREEN_W * (self.finger_position[0] / 848)

            dt = clock.tick(60) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                    break
                elif e.type == pygame.MOUSEMOTION:
                    finger_x = e.pos[0]

            fibers, max_deform = update_fibers()

            pad_x = finger_x + PAD_OFFSET_X
            pad_y = finger_y + PAD_OFFSET_Y
            average_y = 0
            n_y = 0
            for f in fibers:
                dx = f["bx"] - pad_x
                dy = f["by"] - PAD_H / 2 - pad_y
                if abs(dx) <= PAD_W / 2:
                    dist = math.hypot(dx, dy)
                    influence = max(0, 1 - dist / (PAD_W / 2))
                    deform = max_deform[f["type"]]
                    f["scale"] = 0.75
                    raw_h = int(RECT_H * (f["amp"]**0.8) + 60)
                    average_y += raw_h
                    n_y += 1
                else:
                    f["scale"] += (1 - f["scale"]) * RETURN_SPEED
            print(average_y, n_y)
            average_y /= n_y

            screen.fill((30, 30, 30))
            current_time = pygame.time.get_ticks() / 1000.0
            for f in fibers:
                pos = (f["bx"], f["by"])
                dist = math.hypot(f["bx"] - pad_x, f["by"] - PAD_H / 2 - pad_y)
                is_near = dist <= 80
                self.draw_fiber_visual(screen, pos, (RECT_W, RECT_H),
                                       f["freq"], f["scale"], f["amp"],
                                       is_near, current_time, f["phase"], f["gray"])
            screen.blit(finger_img, finger_img.get_rect(center=(finger_x, finger_y - average_y)))
            pygame.display.flip()

        pygame.quit()


# 👇 添加进程入口
def run_visualization_process(queue: Queue, finger_position: Queue):
    VisualizationWithQueue(queue, finger_position).run()


# ---------- Entry ----------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    visualization = VisualizationThread()
    visualization.run()
    sys.exit(app.exec_())