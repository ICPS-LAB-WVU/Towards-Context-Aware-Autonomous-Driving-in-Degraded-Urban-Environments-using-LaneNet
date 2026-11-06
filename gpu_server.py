#!/usr/bin/env python3
"""Clean TCP GPU inference server for the Duckietown context-aware driving paper.

Protocol is compatible with packages/my_package/src/vehicle_client.py:
  client: size,vehicle,frame_id,t_gen\n followed by exactly `size` JPEG bytes
  server: vehicle,frame_id,v,omega,t_server,aoi_server\n
The server also tolerates optional 7-column timing lines from older lab clients.
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time

import cv2
import numpy as np

from perception_control import PerceptionControl
from visualization import render

MAX_JPEG_BYTES = 12 * 1024 * 1024


def run_server(host: str, port: int, pipeline: PerceptionControl, show_gui=False):
    model_lock = threading.Lock()
    frames = {}
    frames_lock = threading.Lock()

    def serve_client(conn, addr):
        print(f"[connected] {addr}", flush=True)
        with conn:
            conn.settimeout(10.0)
            stream = conn.makefile("rb")
            with stream:
                while True:
                    try:
                        header = stream.readline(512)
                    except socket.timeout:
                        break
                    if not header:
                        break
                    fields = header.decode("ascii", errors="strict").strip().split(",")
                    if len(fields) == 7:
                        continue  # optional timing-statistics line from the ROS client
                    if len(fields) != 4:
                        raise ValueError(f"Invalid protocol header: {fields!r}")
                    size_str, vehicle, frame_str, generated_str = fields
                    size, frame_id, t_gen = int(size_str), int(frame_str), float(generated_str)
                    if not 0 < size <= MAX_JPEG_BYTES:
                        raise ValueError(f"Invalid JPEG size {size}")
                    payload = stream.read(size)
                    if len(payload) != size:
                        break
                    t_server = time.time()
                    v, omega = 0.0, 0.0
                    try:
                        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            raise ValueError("JPEG decoding failed")
                        # Shared CUDA models are serialized; controllers are per vehicle.
                        with model_lock:
                            outcome = pipeline.process(frame, vehicle)
                        v, omega = outcome.velocity, outcome.omega
                        if show_gui:
                            rendered = render(outcome)
                            with frames_lock:
                                frames[vehicle] = rendered
                    except Exception as exc:
                        print(f"[infer] {vehicle} frame={frame_id}: {exc}", flush=True)
                    t_server = time.time()
                    # Send safe zero if client and server clocks are not synchronized.
                    aoi = max(0.0, t_server - t_gen)
                    conn.sendall(f"{vehicle},{frame_id},{v:.4f},{omega:.4f},{t_server:.6f},{aoi:.6f}\n".encode("ascii"))
        print(f"[disconnected] {addr}", flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        server.settimeout(0.2 if show_gui else 2.0)
        print(f"[server] Listening on {host}:{port}", flush=True)
        try:
            while True:
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    conn = None
                if conn is not None:
                    threading.Thread(target=serve_client, args=(conn, addr), daemon=True).start()
                if show_gui:
                    with frames_lock:
                        current = list(frames.items())
                    for vehicle, im in current:
                        cv2.imshow(f"LaneNet | {vehicle}", im)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if show_gui:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="LaneNet context-aware Duckietown GPU server")
    parser.add_argument("--host", default=os.getenv("GPU_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GPU_SERVER_PORT", "5001")))
    parser.add_argument("--yolo-weights", default=os.getenv("YOLO_WEIGHTS", "weight/yolo.pt"))
    parser.add_argument("--lanenet-weights", default=os.getenv("LANENET_WEIGHTS", "weight/lanenet_depthwise_se.pth"))
    parser.add_argument("--show-gui", action="store_true")
    args = parser.parse_args()
    pipeline = PerceptionControl(args.yolo_weights, args.lanenet_weights)
    run_server(args.host, args.port, pipeline, args.show_gui)


if __name__ == "__main__":
    main()
