#!/usr/bin/env python3
"""
SETUP:
pip install torch torchvision mediapipe opencv-python numpy scipy

# Clone the mapping repo for learned weights
git clone https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame.git

USAGE:
    python mediapipe_to_flame_extractor.py --video_dir clips --output_dir flame_params \
        --mappings_dir mediapipe-blendshapes-to-flame/mappings
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("ERROR: MediaPipe required. Install with: pip install mediapipe")
    sys.exit(1)

class MP2FLAME:
    
    def __init__(self, mappings_dir: Optional[str] = None):
        self.use_learned = False
        
        if mappings_dir:
            mappings_dir = Path(mappings_dir)
            # Check for correct file names from the repo
            exp_file = mappings_dir / "bs2exp.npy"
            jaw_file = mappings_dir / "bs2jaw.npy"
            eye_file = mappings_dir / "bs2eye.npy"
            
            if exp_file.exists() and jaw_file.exists() and eye_file.exists():
                try:
                    # Load mapping matrices
                    # Shape: bs2exp (52, 100), bs2jaw (52, 3), bs2eye (52, 6)
                    self.bs2exp = np.load(exp_file)
                    self.bs2jaw = np.load(jaw_file)
                    self.bs2eye = np.load(eye_file)
                    
                    print(f"  bs2exp shape: {self.bs2exp.shape}")
                    print(f"  bs2jaw shape: {self.bs2jaw.shape}")
                    print(f"  bs2eye shape: {self.bs2eye.shape}")
                    
                    self.use_learned = True
                    print(f"✓ Loaded learned mappings from {mappings_dir}")
                except Exception as e:
                    print(f"Warning: Could not load mappings: {e}")
            else:
                print(f"Warning: Mapping files not found in {mappings_dir}")
                print(f"  Expected: bs2exp.npy, bs2jaw.npy, bs2eye.npy")
        
        if not self.use_learned:
            print("Using fallback analytical mapping (less accurate)")
    
    def convert(self, blendshape_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if blendshape_scores.ndim == 1:
            blendshape_scores = blendshape_scores[np.newaxis, :]
        
        if self.use_learned:
            expression = blendshape_scores @ self.bs2exp
            jaw_pose = blendshape_scores @ self.bs2jaw
            eye_pose = blendshape_scores @ self.bs2eye
        else:
            expression, jaw_pose, eye_pose = self._analytical_mapping(blendshape_scores)
        
        return expression.astype(np.float32), jaw_pose.astype(np.float32), eye_pose.astype(np.float32)
    
    def _analytical_mapping(self, bs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(bs)
        expression = np.zeros((N, 100), dtype=np.float32)
        expression[:, 0] = bs[:, 17] * 2.0   # jawOpen
        expression[:, 1] = bs[:, 18] * 1.5   # jawForward
        expression[:, 4] = bs[:, 27] * 2.0   # mouthFunnel
        expression[:, 5] = bs[:, 28] * 2.0   # mouthPucker
        expression[:, 8] = bs[:, 33] * 2.0   # mouthSmileLeft
        expression[:, 9] = bs[:, 34] * 2.0   # mouthSmileRight
        expression[:, 30] = bs[:, 3] * 2.0   # browDownLeft
        expression[:, 31] = bs[:, 4] * 2.0   # browDownRight
        expression[:, 32] = bs[:, 5] * 1.5   # browInnerUp
        expression[:, 40] = bs[:, 11] * 2.0  # eyeBlinkLeft
        expression[:, 41] = bs[:, 12] * 2.0  # eyeBlinkRight
        
        jaw_pose = np.zeros((N, 3), dtype=np.float32)
        jaw_pose[:, 0] = bs[:, 17] * 0.4  # jaw open -> pitch
        
        eye_pose = np.zeros((N, 6), dtype=np.float32)
        eye_pose[:, 0] = (bs[:, 13] - bs[:, 15]) * 0.3  # left eye pitch
        eye_pose[:, 3] = (bs[:, 14] - bs[:, 16]) * 0.3  # right eye pitch
        
        return expression, jaw_pose, eye_pose


def gamma_correct(frame_bgr: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.float32)
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(frame_bgr, table)

def compress_highlights(frame_bgr: np.ndarray, threshold: int = 215, strength: float = 0.5) -> np.ndarray:
    img = frame_bgr.astype(np.float32)
    mask = img > threshold
    img[mask] = threshold + (img[mask] - threshold) * (1.0 - strength)
    return np.clip(img, 0, 255).astype(np.uint8)

def clahe_luminance(frame_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def bilateral_smooth(frame_bgr: np.ndarray, d: int = 7, sigma_color: int = 40, sigma_space: int = 40) -> np.ndarray:
    return cv2.bilateralFilter(frame_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def unsharp_mask(frame_bgr: np.ndarray, sigma: float = 1.0, amount: float = 0.5) -> np.ndarray:
    blur = cv2.GaussianBlur(frame_bgr, (0, 0), sigma)
    return cv2.addWeighted(frame_bgr, 1 + amount, blur, -amount, 0)

def enhance_for_face(frame_bgr: np.ndarray) -> np.ndarray:
    """Full enhancement pipeline for face detection."""
    x = gamma_correct(frame_bgr, gamma=1.5)
    x = compress_highlights(x, threshold=215, strength=0.5)
    x = bilateral_smooth(x, d=7, sigma_color=40, sigma_space=40)
    x = clahe_luminance(x, clip_limit=2.0, tile_grid=(8, 8))
    x = unsharp_mask(x, sigma=1.0, amount=0.5)
    return x

ENHANCE_VARIANTS = {
    "orig": lambda x: x,
    "gamma": lambda x: gamma_correct(x, gamma=1.5),
    "gamma_hc": lambda x: compress_highlights(gamma_correct(x, gamma=1.5), threshold=215, strength=0.5),
    "gamma_hc_clahe": lambda x: clahe_luminance(compress_highlights(gamma_correct(x, gamma=1.5), threshold=215, strength=0.5)),
    "full": enhance_for_face,
}

#bounding box utility functions

def clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def expand_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, pad_frac: float = 0.3) -> Tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))
    return clip_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)

def bbox_xyxy_to_xywh(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1

def bbox_xywh_to_xyxy(box) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))

def bbox_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_area(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


#opencv tracker utility (CSRT if available, otherwise KCF fallback)
def get_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    # Fallback to KCF if CSRT not available
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    raise AttributeError("No suitable tracker found in OpenCV")


def estimate_face_from_pose(pose_landmarks, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if pose_landmarks is None:
        return None
    HEAD_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # nose, eyes, ears, mouth
    
    pts = []
    for idx in HEAD_INDICES:
        if idx < len(pose_landmarks):
            lm = pose_landmarks[idx]
            # Check visibility
            vis = getattr(lm, 'visibility', 1.0)
            if vis > 0.3:
                pts.append((lm.x * width, lm.y * height))
    
    if len(pts) < 3:
        return None
    
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    size = max(max(xs) - min(xs), max(ys) - min(ys), 60.0)
    
    # Expand to include full face
    x1 = int(cx - size * 1.2)
    x2 = int(cx + size * 1.2)
    y1 = int(cy - size * 1.5)
    y2 = int(cy + size * 1.2)
    
    return clip_bbox(x1, y1, x2, y2, width, height)

class RobustFaceExtractor:
    def __init__(self, model_path: Optional[str] = None):
        # Download model if needed
        if model_path and Path(model_path).exists():
            mp_path = model_path
        else:
            default_paths = [
                "face_landmarker.task",
                "models/face_landmarker.task",
            ]
            mp_path = None
            for p in default_paths:
                if Path(p).exists():
                    mp_path = str(p)
                    break
            
            if mp_path is None:
                print("Downloading face_landmarker.task...")
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                mp_path = "face_landmarker.task"
                urllib.request.urlretrieve(url, mp_path)
        
        base_options = python.BaseOptions(model_asset_path=mp_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        try:
            pose_path = "pose_landmarker_lite.task"
            if not Path(pose_path).exists():
                print("Downloading pose_landmarker_lite.task...")
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                urllib.request.urlretrieve(url, pose_path)
            
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            self.has_pose = True
        except Exception as e:
            print(f"Warning: Pose landmarker not available: {e}")
            self.has_pose = False
        
        print(f"MediaPipe initialised with: {mp_path}")
        
        # Tracker state
        self.tracker = None
        self.tracker_fail_count = 0
        self.last_bbox = None
        self.last_frame_bgr = None
    
    def _detect_on_image(self, image_rgb: np.ndarray) -> Optional[Dict]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            return None
        
        h, w = image_rgb.shape[:2]
        lmks = result.face_landmarks[0]
        landmarks_3d = np.array([[l.x * w, l.y * h, l.z * w] for l in lmks], dtype=np.float32)
        
        if result.face_blendshapes:
            blendshapes = np.array([b.score for b in result.face_blendshapes[0]], dtype=np.float32)
        else:
            blendshapes = np.zeros(52, dtype=np.float32)
        
        if result.facial_transformation_matrixes:
            transform = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
        else:
            transform = np.eye(4, dtype=np.float32)
        
        # Compute bbox from landmarks
        xs = landmarks_3d[:, 0]
        ys = landmarks_3d[:, 1]
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        
        return {
            'landmarks_3d': landmarks_3d,
            'blendshapes': blendshapes,
            'transform': transform,
            'bbox': bbox,
        }
    
    def _detect_on_crop(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], upscale: float = 2.0) -> Optional[Dict]:
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
            return None
        
        # Upscale small crops for better detection
        if upscale > 1.0:
            crop_bgr = cv2.resize(crop_bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = self._detect_on_image(crop_rgb)
        
        if result is None:
            return None
        
        # Remap landmarks to full frame coordinates
        crop_h, crop_w = crop_bgr.shape[:2]
        orig_crop_w = x2 - x1
        orig_crop_h = y2 - y1
        
        landmarks = result['landmarks_3d'].copy()
        # Scale back from upscaled crop to original crop size
        landmarks[:, 0] = landmarks[:, 0] / upscale
        landmarks[:, 1] = landmarks[:, 1] / upscale
        # Then to full frame
        landmarks[:, 0] = landmarks[:, 0] + x1
        landmarks[:, 1] = landmarks[:, 1] + y1
        
        result['landmarks_3d'] = landmarks
        
        # Update bbox
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        result['bbox'] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        
        return result
    
    def _get_pose_face_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if not self.has_pose:
            return None
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            result = self.pose_landmarker.detect(mp_image)
            if result.pose_landmarks:
                h, w = frame_bgr.shape[:2]
                return estimate_face_from_pose(result.pose_landmarks[0], w, h)
        except:
            pass
        
        return None
    
    def extract_frame(self, frame_bgr: np.ndarray, frame_idx: int = 0) -> Tuple[Optional[Dict], str]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Strategy 1: Direct detection
        result = self._detect_on_image(frame_rgb)
        if result is not None:
            self._init_tracker(frame_bgr, result['bbox'])
            self.last_bbox = result['bbox']
            return result, 'direct'
        
        # Strategy 2: Enhanced image variants
        for variant_name in ['gamma_hc_clahe', 'full', 'gamma']:
            enhanced = ENHANCE_VARIANTS[variant_name](frame_bgr)
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            result = self._detect_on_image(enhanced_rgb)
            if result is not None:
                self._init_tracker(frame_bgr, result['bbox'])
                self.last_bbox = result['bbox']
                return result, 'enhanced'
        
        # Strategy 3: CSRT tracker + crop detection
        if self.tracker is not None:
            ok, tracked_xywh = self.tracker.update(frame_bgr)
            if ok:
                tracked_box = bbox_xywh_to_xyxy(tracked_xywh)
                tracked_box = clip_bbox(*tracked_box, w, h)
                expanded_box = expand_bbox(*tracked_box, w, h, pad_frac=0.3)
                
                result = self._detect_on_crop(frame_bgr, expanded_box, upscale=2.5)
                if result is not None:
                    self._init_tracker(frame_bgr, result['bbox'])
                    self.last_bbox = result['bbox']
                    self.tracker_fail_count = 0
                    return result, 'tracker'
                
                self.tracker_fail_count += 1
                if self.tracker_fail_count > 15:
                    self.tracker = None
        
        # Strategy 4: Pose-based face region
        pose_bbox = self._get_pose_face_bbox(frame_bgr)
        if pose_bbox is not None:
            expanded_box = expand_bbox(*pose_bbox, w, h, pad_frac=0.4)
            result = self._detect_on_crop(frame_bgr, expanded_box, upscale=3.0)
            if result is not None:
                self._init_tracker(frame_bgr, result['bbox'])
                self.last_bbox = result['bbox']
                return result, 'crop'
        
        # Strategy 5: Last known bbox with expansion
        if self.last_bbox is not None:
            expanded_box = expand_bbox(*self.last_bbox, w, h, pad_frac=0.5)
            result = self._detect_on_crop(frame_bgr, expanded_box, upscale=3.0)
            if result is not None:
                self._init_tracker(frame_bgr, result['bbox'])
                self.last_bbox = result['bbox']
                return result, 'crop'
        
        return None, 'none'
    
    def _init_tracker(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]):
        try:
            self.tracker = get_csrt_tracker()
            xywh = bbox_xyxy_to_xywh(bbox)
            self.tracker.init(frame_bgr, xywh)
            self.tracker_fail_count = 0
        except Exception as e:
            self.tracker = None
    
    def extract_video(self, video_path: str, subsample: int = 1, max_frames: Optional[int] = None, verbose: bool = True) -> Optional[Dict[str, np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  Video: {total_frames} frames @ {fps:.1f} fps ({width}x{height})")
        
        all_landmarks = []
        all_blendshapes = []
        all_transforms = []
        frame_indices = []
        valid_mask = []
        source_counts = {'direct': 0, 'enhanced': 0, 'tracker': 0, 'crop': 0, 'none': 0}
        
        # Reset tracker state for new video
        self.tracker = None
        self.tracker_fail_count = 0
        self.last_bbox = None
        
        frame_idx = 0
        processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % subsample == 0:
                result, source = self.extract_frame(frame, frame_idx)
                source_counts[source] += 1
                
                if result is not None:
                    all_landmarks.append(result['landmarks_3d'])
                    all_blendshapes.append(result['blendshapes'])
                    all_transforms.append(result['transform'])
                    valid_mask.append(1.0)
                else:
                    # Placeholder for missing frames (will be interpolated)
                    all_landmarks.append(np.zeros((478, 3), dtype=np.float32))
                    all_blendshapes.append(np.zeros(52, dtype=np.float32))
                    all_transforms.append(np.eye(4, dtype=np.float32))
                    valid_mask.append(0.0)
                
                frame_indices.append(frame_idx)
                processed += 1
                
                if max_frames and processed >= max_frames:
                    break
            
            frame_idx += 1
            
            if verbose and frame_idx % 100 == 0:
                detected_pct = 100.0 * (1.0 - source_counts['none'] / max(1, processed))
                print(f"    Frame {frame_idx}/{total_frames} ({detected_pct:.1f}% detected)")
        
        cap.release()
        
        if not all_landmarks:
            return None
        
        # Convert to arrays
        landmarks = np.array(all_landmarks)
        blendshapes = np.array(all_blendshapes)
        transforms = np.array(all_transforms)
        valid_mask = np.array(valid_mask)
        
        # Interpolate missing frames
        landmarks, blendshapes = self._interpolate_missing(landmarks, blendshapes, valid_mask)
        
        raw_detected_pct = 100.0 * (1.0 - source_counts['none'] / max(1, processed))
        print(f"  Sources: direct={source_counts['direct']}, enhanced={source_counts['enhanced']}, "
              f"tracker={source_counts['tracker']}, crop={source_counts['crop']}, "
              f"none={source_counts['none']} ({raw_detected_pct:.1f}% raw)")
        
        return {
            'landmarks_3d': landmarks,
            'blendshapes': blendshapes,
            'transforms': transforms,
            'frame_indices': np.array(frame_indices),
            'valid_mask': valid_mask,
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
        }
    
    def _interpolate_missing(self, landmarks: np.ndarray, blendshapes: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        T = len(landmarks) #interpolate missing frames in the middle, but keep leading/trailing missing frames as zeros
        valid_idx = np.where(valid_mask > 0.5)[0]
        
        if len(valid_idx) == 0:
            return landmarks, blendshapes
        
        if len(valid_idx) == T:
            return landmarks, blendshapes  # All valid
        
        t_all = np.arange(T)
        
        # Interpolate landmarks
        landmarks_interp = landmarks.copy()
        for j in range(landmarks.shape[1]):
            for d in range(landmarks.shape[2]):
                vals = landmarks[valid_idx, j, d]
                landmarks_interp[:, j, d] = np.interp(t_all, valid_idx, vals)
        
        # Interpolate blendshapes
        blendshapes_interp = blendshapes.copy()
        for b in range(blendshapes.shape[1]):
            vals = blendshapes[valid_idx, b]
            blendshapes_interp[:, b] = np.interp(t_all, valid_idx, vals)
        
        return landmarks_interp.astype(np.float32), blendshapes_interp.astype(np.float32)
    
    def close(self):
        self.landmarker.close()
        if self.has_pose:
            self.pose_landmarker.close()

def extract_flame_from_video(video_path: str, output_dir: str, extractor: RobustFaceExtractor, converter: MP2FLAME, subsample: int = 1) -> Optional[Dict[str, np.ndarray]]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{video_path.stem}_flame.npz"
    
    if output_path.exists():
        print(f"  [CACHED] {output_path}")
        return dict(np.load(output_path))
    
    print(f"  Extracting: {video_path.name}")
    
    mp_data = extractor.extract_video(str(video_path), subsample=subsample)
    
    if mp_data is None or len(mp_data['blendshapes']) == 0:
        print(f"  [WARN] No faces detected in {video_path.name}")
        return None
    
    # Convert to FLAME parameters
    expression, jaw_pose, eye_pose = converter.convert(mp_data['blendshapes'])
    
    # Estimate global head orientation from transformation matrix
    transforms = mp_data['transforms']
    global_orient = np.zeros((len(transforms), 3), dtype=np.float32)
    
    for i, T in enumerate(transforms):
        R = T[:3, :3]
        try:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(R)
            global_orient[i] = r.as_rotvec()
        except:
            # Fallback
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            if sy > 1e-6:
                x = np.arctan2(R[2,1], R[2,2])
                y = np.arctan2(-R[2,0], sy)
                z = np.arctan2(R[1,0], R[0,0])
            else:
                x = np.arctan2(-R[1,2], R[1,1])
                y = np.arctan2(-R[2,0], sy)
                z = 0
            global_orient[i] = [x, y, z]
    
    result = {
        'expression': expression,
        'jaw_pose': jaw_pose,
        'eye_pose': eye_pose,
        'global_orient': global_orient,
        'mediapipe_blendshapes': mp_data['blendshapes'],
        'landmarks_3d': mp_data['landmarks_3d'],
        'valid_mask': mp_data['valid_mask'],
        'frame_indices': mp_data['frame_indices'],
        'fps': mp_data['fps'],
    }
    
    np.savez(output_path, **result)
    print(f"  [SAVED] {output_path} ({len(expression)} frames)")
    
    return result


def batch_extract(video_dir: str, output_dir: str, mappings_dir: Optional[str] = None, subsample: int = 1, max_videos: Optional[int] = None) -> Dict[str, Dict]:
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    videos = []
    for ext in ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']:
        videos.extend(video_dir.glob(ext))
    videos = sorted(videos)
    
    if max_videos:
        videos = videos[:max_videos]
    
    print(f"Found {len(videos)} videos")
    
    extractor = RobustFaceExtractor()
    converter = MP2FLAME(mappings_dir)
    
    results = {}
    success = 0
    failed = 0
    
    for i, video_path in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {video_path.name}")
        
        try:
            result = extract_flame_from_video(
                str(video_path),
                str(output_dir),
                extractor,
                converter,
                subsample=subsample,
            )
            
            if result is not None:
                results[video_path.stem] = result
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    extractor.close()
    
    print(f"\n{'='*50}")
    print(f"Extraction complete!")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    
    summary = {
        'total': len(videos),
        'success': success,
        'failed': failed,
        'videos': list(results.keys()),
    }
    
    with open(output_dir / 'extraction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def create_training_data(flame_dir: str, jsonl_path: str, output_path: str):
    #match videos in the jsonl metadata with the extracted FLAME params and create a combined jsonl for training
    flame_dir = Path(flame_dir)
    
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    
    print(f"Loaded {len(records)} metadata records")
    
    matched = []
    for r in records:
        video_name = r['base_video']
        flame_path = flame_dir / f"{video_name}_flame.npz"
        
        if flame_path.exists():
            matched.append({**r, 'flame_path': str(flame_path)})
    
    print(f"Matched {len(matched)} records with FLAME params")
    
    with open(output_path, 'w') as f:
        for r in matched:
            f.write(json.dumps(r) + '\n')
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract FLAME parameters (NO PyTorch3D!)")
    
    parser.add_argument('--video', type=str, help='Single video file')
    parser.add_argument('--video_dir', type=str, help='Directory of videos')
    parser.add_argument('--output_dir', type=str, default='flame_params')
    parser.add_argument('--mappings_dir', type=str, default=None,
                       help='Path to mediapipe-blendshapes-to-flame/mappings')
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--max_videos', type=int, default=None)
    
    parser.add_argument('--create_training', action='store_true')
    parser.add_argument('--jsonl', type=str)
    parser.add_argument('--training_output', type=str, default='training_data.jsonl')
    
    args = parser.parse_args()
    
    if args.create_training:
        if not args.jsonl:
            print("ERROR: --jsonl required for --create_training")
            return
        create_training_data(args.output_dir, args.jsonl, args.training_output)
        return
    
    if args.video:
        extractor = RobustFaceExtractor()
        converter = MP2FLAME(args.mappings_dir)
        extract_flame_from_video(args.video, args.output_dir, extractor, converter, args.subsample)
        extractor.close()
    elif args.video_dir:
        batch_extract(args.video_dir, args.output_dir, args.mappings_dir, args.subsample, args.max_videos)
    else:
        parser.print_help()
        print("\nExample:")
        print("  python mediapipe_to_flame_extractor.py --video_dir clips --output_dir flame_params \\")
        print("      --mappings_dir mediapipe-blendshapes-to-flame/mappings")


if __name__ == '__main__':
    main()
