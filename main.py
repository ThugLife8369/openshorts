"""
OpenShorts Main Pipeline Runner
Complete, fully integrated version with automated AWS S3 uploading, 
Node.js runtime binding for yt-dlp, strict single-stream fallback, 
and full test suite compliance.
"""

import time
import cv2
import subprocess
import shutil
import argparse
import re
import sys
import threading
import unicodedata
import uuid
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
import boto3
from ultralytics import YOLO
from dotenv import load_dotenv

import transcribe_backends
import gemini_worker
from ffmpeg_utils import cut_clip, METADATA_SCRUB
from watermarked import mark_delivery

# Load environment variables
load_dotenv()

# --- Constants & Models ---
ASPECT_RATIO = 9 / 16
model = YOLO(os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt"))
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

JUMP_CONFIRM_FRAMES = max(int(os.environ.get("JUMP_CONFIRM_FRAMES", "3")), 1)
SCENE_CUT_RESET = os.environ.get("SCENE_CUT_RESET", "1") != "0"
DETECT_MAX_WIDTH = 640
DETECT_LOCK = threading.Lock()
DETECT_STRIDE = max(int(os.environ.get("DETECT_STRIDE", "4")), 1)
YOLO_FALLBACK_STRIDE = DETECT_STRIDE * 2

class SmoothedCameraman:
    def __init__(self, output_width, output_height, video_width, video_height, aspect_ratio=ASPECT_RATIO):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = aspect_ratio
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        self.crop_height = video_height
        self.crop_width = int(self.crop_height * aspect_ratio)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / aspect_ratio)
             
        self.safe_zone_radius = self.crop_width * 0.25
        self.jump_confirm_frames = JUMP_CONFIRM_FRAMES
        self._pending_target = None
        self._pending_count = 0
        self._snap_pending = False

    def begin_scene(self):
        self._pending_target = None
        self._pending_count = 0
        self._snap_pending = True

    def update_target(self, face_box):
        if not face_box:
            return
        x, y, w, h = face_box
        new_center = x + w / 2

        if self._snap_pending:
            self._snap_pending = False
            self._pending_target = None
            self._pending_count = 0
            self.target_center_x = new_center
            self.current_center_x = new_center
            return

        if abs(new_center - self.target_center_x) > self.safe_zone_radius:
            if (self._pending_target is not None
                    and abs(new_center - self._pending_target) <= self.safe_zone_radius):
                self._pending_count += 1
            else:
                self._pending_target = new_center
                self._pending_count = 1
            if self._pending_count < self.jump_confirm_frames:
                return

        self._pending_target = None
        self._pending_count = 0
        self.target_center_x = new_center
    
    def get_crop_box(self, force_snap=False):
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            if abs(diff) > self.safe_zone_radiusBased on the exact log you just shared, we successfully eliminated the `NoneType` Python crash in `yt-dlp` (the logs show `extractor_args: {}` now!). 

However, the bot block happened again because **the cookies were not injected into the runner**. Look closely at the environment variables the GitHub runner loaded right before it crashed:

```text
  GEMINI_API_KEY: ***
  ELEVENLABS_API_KEY: ***
  FAL_KEY: ***
  UPLOAD_POST_API_KEY: ***
  AWS_ACCESS_KEY_ID: ***
  AWS_SECRET_ACCESS_KEY: ***
  AWS_S3_BUCKET: ***
  AWS_REGION: ***
  AUTO_CAPTIONS: 1
  AUTO_HOOK: 1
