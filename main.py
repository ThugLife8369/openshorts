"""
OpenShorts Main Pipeline Runner
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

load_dotenv()

# --- Constants & Models ---
ASPECT_RATIO = 9 / 16
model = YOLO(os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt"))
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

DETECT_MAX_WIDTH = 640
DETECT_LOCK = threading.Lock()

def sanitize_filename(filename):
    filename = unicodedata.normalize('NFC', filename)
    filename = re.sub(r'[<>:"/\\|?*#]', '', filename)
    filename = filename.replace(' ', '_')
    return filename[:120]

def download_youtube_video(url, output_dir="."):
    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")

    # FIX: Write cookies locally to avoid /app/ path crashes on GitHub Actions
    cookies_path = os.path.join(output_dir, 'cookies.txt')
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    
    if cookies_env:
        try:
            with open(cookies_path, 'w') as f:
                f.write(cookies_env)
        except Exception as e:
            print(f"⚠️ Failed to write cookies: {e}")
            cookies_path = None
    else:
        cookies_path = 'cookies.txt' if os.path.exists('cookies.txt') else None

    _proxy = os.environ.get("PROXY_URL", "").strip() or None

    try:
        from yt_clients import hd_extractor_args
        hd_args = hd_extractor_args()
    except ImportError:
        hd_args = {}

    # FIX: Ensure extractor_args is strictly a dictionary to prevent NoneType .get() errors
    def _base_opts(extractor_args, proxy, cookies=True):
        return {
            'quiet': False, 'verbose': True, 'no_warnings': False,
            'cookiefile': cookies_path if (cookies and cookies_path) else None,
            'proxy': proxy, 'socket_timeout': 30, 'retries': 10,
            'nocheckcertificate': True, 'cachedir': False, 'noplaylist': True,
            'extractor_args': extractor_args if extractor_args is not None else {},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            },
        }

    with yt_dlp.YoutubeDL(_base_opts(hd_args, _proxy)) as ydl:
        info = ydl.extract_info(url, download=False, process=False)
    
    sanitized = sanitize_filename(info.get('title', 'youtube_video'))
    expected = os.path.join(output_dir, f'{sanitized}.mp4')
    if os.path.exists(expected):
        os.remove(expected)

    dl_opts = {
        **_base_opts(hd_args, _proxy),
        'format': 'bestvideo[vcodec^=avc1][height<=1080][ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': os.path.join(output_dir, f'{sanitized}.%(ext)s'),
        'merge_output_format': 'mp4',
    }
    
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        ydl.download([url])

    downloaded_file = expected
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break

    return downloaded_file, sanitized

def upload_to_s3(file_path):
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        print("⚠️ AWS_S3_BUCKET not configured. Skipping S3 upload.")
        return False
    try:
        s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "us-east-1"))
        file_name = os.path.basename(file_path)
        s3.upload_file(file_path, bucket, file_name)
        print(f"☁️ Uploaded {file_name} to S3.")
        return True
    except Exception as e:
        print(f"❌ S3 Upload failed: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input', type=str, help="Path to input video")
    group.add_argument('-u', '--url', type=str, help="YouTube URL")
    parser.add_argument('-o', '--output', type=str, default=".")
    parser.add_argument('--format', type=str, default="auto")
    args = parser.parse_args()

    output_dir = args.output if os.path.isdir(args.output) else "."
    
    if args.url:
        input_video, video_title = download_youtube_video(args.url, output_dir)
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]

    print(f"🎬 Processing: {video_title}")
    
    # 1. Transcribe
    transcript = transcribe_backends.transcribe(input_video)
    duration = transcript.get('duration', 60.0)
    
    # 2. Extract Clips
    clips = gemini_worker.get_viral_clips(transcript, duration) if hasattr(gemini_worker, 'get_viral_clips') else []
    if not clips:
        clips = [{"start": 0.0, "end": min(duration, 30.0)}]

    # 3. Process & Upload
    for i, clip in enumerate(clips):
        start, end = clip.get('start', 0.0), clip.get('end', 30.0)
        clip_path = os.path.join(output_dir, f"{video_title}_clip_{i+1}.mp4")
        
        cut_clip(input_video, clip_path, start, end, i + 1)
        served = mark_delivery(clip_path)
        upload_to_s3(served)
        print(f"✅ CLIP_READY: {os.path.basename(served)}")
