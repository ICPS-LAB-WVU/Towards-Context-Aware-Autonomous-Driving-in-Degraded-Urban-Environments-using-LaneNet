"""LaneNet + YOLOv8n perception and context-aware driving controller.

No adversarial attack, traffic impairment, or ByteTrack dependency.
The LaneNet model is imported from the iCPS Duckietown-Lane-Segmentation
training package (install it with `pip install -e training/lane-segmentation`).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from duckietown_seg.models import create_model

from lane_center_estimation import LaneEstimate, SpatioTemporalLaneEstimator

CLASS_BOT = 0
CLASS_STOP_SIGN = 2
CLASS_TRAFFIC_LIGHT = 3


@dataclass
class DrivingOutput:
    velocity: float
    omega: float
    error_cm: float | None
    mode: str
    detections: np.ndarray
    segmentation: np.ndarray
    frame_bgr: np.ndarray


class PIDTrafficController:
    """Published PID gains and semantic traffic-rule state machine.

    Camera geometry and stop-duration thresholds are configurable. These values
    should be tuned against the exact physical testbed before driving.
    """

    def __init__(self):
        self.estimator = SpatioTemporalLaneEstimator(history_size=8)
        self.kp = float(os.getenv("PID_KP", "0.25"))
        self.ki = float(os.getenv("PID_KI", "0.01"))
        self.kd = float(os.getenv("PID_KD", "0.05"))
        self.kappa = float(os.getenv("PID_KAPPA_CM", "20"))
        self.vmax = float(os.getenv("MAX_VELOCITY", "0.30"))
        self.max_omega = float(os.getenv("MAX_OMEGA", "4.0"))
        self.stop_wait = float(os.getenv("STOP_WAIT_SEC", "3.0"))
        self.cooldown = float(os.getenv("STOP_COOLDOWN_SEC", "5.0"))
        self.stop_height_fraction = float(os.getenv("STOP_HEIGHT_FRACTION", "0.15"))
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = None
        self.state = "DRIVING"
        self.state_since = 0.0
        self.last_stop_finished = -float("inf")
        self.tl_cooldown_until = -float("inf")

    @staticmethod
    def _traffic_light_is_green(frame, det):
        x1, y1, x2, y2 = map(int, det[:4])
        h, w = frame.shape[:2]
        x1, x2 = max(x1, 0), min(x2, w)
        y1, y2 = max(y1, 0), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return False
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([40, 70, 50]), np.array([90, 255, 255]))
        red = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        red |= cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
        denom = float(crop.shape[0] * crop.shape[1])
        return cv2.countNonZero(green) / denom >= 0.03 and cv2.countNonZero(red) / denom <= 0.08

    def command(self, estimate: LaneEstimate, boxes: np.ndarray, frame_bgr: np.ndarray):
        now = time.monotonic()
        h, _ = frame_bgr.shape[:2]
        stops = [b for b in boxes if int(b[4]) == CLASS_STOP_SIGN and float(b[5]) > 0.6]
        lights = [b for b in boxes if int(b[4]) == CLASS_TRAFFIC_LIGHT and float(b[5]) > 0.6]

        if self.state == "STOP_WAIT":
            if now - self.state_since < self.stop_wait:
                return 0.0, 0.0
            self.state = "DRIVING"
            self.last_stop_finished = now

        if self.state == "RED_WAIT":
            if not lights or not any(self._traffic_light_is_green(frame_bgr, b) for b in lights):
                return 0.0, 0.0
            self.state = "DRIVING"
            self.tl_cooldown_until = now + 5.0

        if now - self.last_stop_finished > self.cooldown:
            if any(float(b[3] - b[1]) >= h * self.stop_height_fraction for b in stops):
                self.state, self.state_since = "STOP_WAIT", now
                return 0.0, 0.0

        if now >= self.tl_cooldown_until:
            for b in lights:
                if float(b[3] - b[1]) >= h * self.stop_height_fraction:
                    if not self._traffic_light_is_green(frame_bgr, b):
                        self.state, self.state_since = "RED_WAIT", now
                        return 0.0, 0.0

        # Preserve the original perception-control system's front-robot stopping cue.
        if estimate.center_px is not None:
            for b in boxes:
                if int(b[4]) == CLASS_BOT and float(b[5]) >= 0.5:
                    bot_cx = float(b[0] + b[2]) / 2.0
                    if (float(b[3] - b[1]) >= 0.35 * h and
                            abs(bot_cx - estimate.center_px) < estimate.lane_width_px / 2.5):
                        return 0.0, 0.0

        if estimate.error_cm is None:
            return 0.0, 0.0  # no valid lane estimate or history -> safe stop
        err = float(estimate.error_cm)
        dt = 0.05 if self.previous_time is None else min(max(now - self.previous_time, 0.01), 0.5)
        self.integral = float(np.clip(self.integral + err * dt, -100, 100))
        derivative = (err - self.previous_error) / dt
        omega = -float(np.clip(self.kp * err + self.ki * self.integral + self.kd * derivative,
                                -self.max_omega, self.max_omega))
        velocity = self.vmax * math.exp(-abs(err) / self.kappa)
        self.previous_error, self.previous_time = err, now
        return float(velocity), float(omega)


class PerceptionControl:
    def __init__(self, yolo_weights: str, lanenet_weights: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = YOLO(yolo_weights)
        self.segmenter = create_model(
            model_name="configurable_unet", in_channels=3, num_classes=4,
            base_channels=32, bilinear=True, use_depthwise=True,
            use_se_decoder=True, use_aspp=False,
        ).to(self.device)
        try:
            ckpt = torch.load(lanenet_weights, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(lanenet_weights, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            self.segmenter.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError("LaneNet checkpoint is incompatible with depthwise_se; "
                               "do not use the old vanilla U-Net checkpoint.") from exc
        self.segmenter.eval()
        self.controllers: dict[str, PIDTrafficController] = {}

    @torch.inference_mode()
    def process(self, frame_bgr: np.ndarray, vehicle_id: str) -> DrivingOutput:
        frame = cv2.resize(frame_bgr, (640, 640))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        inp = inp.float().div_(255.0).to(self.device)  # matches upstream training transforms
        logits = self.segmenter(inp)
        mask = logits.argmax(dim=1)[0].byte().cpu().numpy()

        det = self.detector.predict(frame, conf=0.25, imgsz=640, verbose=False)[0]
        bxs = []
        for b in det.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            bxs.append([x1, y1, x2, y2, int(b.cls[0]), float(b.conf[0])])
        boxes = np.asarray(bxs, dtype=np.float32).reshape(-1, 6)
        controller = self.controllers.setdefault(vehicle_id, PIDTrafficController())
        estimate = controller.estimator.estimate(mask)
        v, omega = controller.command(estimate, boxes, frame)
        return DrivingOutput(v, omega, estimate.error_cm, estimate.source, boxes, mask, frame)
