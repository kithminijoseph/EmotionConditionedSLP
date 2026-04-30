#!/usr/bin/env python3
import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", message=".*align should be passed.*")
warnings.filterwarnings("ignore", message=".*scipy.sparse.csc.*deprecated.*")

def safe_to_numpy(x, dtype=None):
    if hasattr(x, "r"):
        x = x.r
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x


FACE_REGIONS = {
    "head_pose_refs": [1, 33, 263, 61, 291, 199],
    "left_brow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_brow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "nose": [1, 2, 98, 327, 168, 6, 197, 195, 5, 4],
    "upper_lip": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lower_lip": [146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "jaw": [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
    "left_cheek": [117, 118, 119, 100, 126, 142, 36, 205, 206],
    "right_cheek": [346, 347, 348, 329, 355, 371, 266, 425, 426],
}

NMF_CHANNELS = [
    "head_turn", "head_tilt_side", "head_tilt_fb", "head_nod", "head_shake", "head_jut",
    "brow_raise", "brow_furrow", "eye_aperture", "eye_gaze_h", "eye_gaze_v",
    "mouth_open", "mouth_spread", "lip_pucker", "tongue", "nose_wrinkle", "cheek_puff",
]

N_NMF_CHANNELS = len(NMF_CHANNELS)
N_EMOTIONS = 10

class TextEncoder(nn.Module):
    def __init__(self, output_dim: int = 256, use_bert: bool = True):
        super().__init__()
        self.use_bert = use_bert
        self.output_dim = output_dim

        if use_bert:
            try:
                from transformers import BertModel, BertTokenizer
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                self.bert = BertModel.from_pretrained("bert-base-uncased")
                self.bert = self.bert.cpu()
                self.bert.eval()
                for param in self.bert.parameters():
                    param.requires_grad = False
                self.proj = nn.Linear(768, output_dim)
                print("Loaded BERT encoder on CPU")
            except ImportError:
                print("transformers not installed, using fallback encoder")
                self.use_bert = False

        if not self.use_bert:
            self.embed = nn.Embedding(256, 64)
            self.lstm = nn.LSTM(64, output_dim // 2, batch_first=True, bidirectional=True)

    def force_bert_cpu(self):
        if self.use_bert and hasattr(self, "bert"):
            self.bert = self.bert.cpu()
            self.bert.eval()

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if self.use_bert:
            self.force_bert_cpu()

            tokens = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            tokens = {k: v.cpu() for k, v in tokens.items()}

            with torch.no_grad():
                outputs = self.bert(**tokens)

            cls_embed = outputs.last_hidden_state[:, 0, :]      # CPU
            cls_embed = cls_embed.to(device)                    # move only final embedding
            return self.proj(cls_embed)

        batch_size = len(texts)
        max_len = max(len(t) for t in texts) if texts else 1
        max_len = min(max_len, 128)

        char_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, text in enumerate(texts):
            for j, c in enumerate(text[:max_len]):
                char_ids[i, j] = ord(c) % 256

        embedded = self.embed(char_ids)
        output, _ = self.lstm(embedded)
        return output[:, -1, :]



class FiLMLayer(nn.Module):
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(condition).unsqueeze(1)
        beta = self.beta(condition).unsqueeze(1)
        return (1.0 + gamma) * x + beta


def parse_nmf_value(label: str, value: str) -> Dict[str, float]:
    result = {}

    if label == "head pos: turn":
        mag = 0.3 if "slightly" in value else 0.8 if "further" in value else 0.5
        result["head_turn"] = mag if "right" in value else -mag if "left" in value else 0

    elif label == "head pos: tilt side":
        mag = 0.3 if "slightly" in value else 0.8 if "further" in value else 0.5
        result["head_tilt_side"] = mag if "right" in value else -mag if "left" in value else 0

    elif label == "head pos: tilt fr/bk":
        mag = 0.3 if "slightly" in value else 0.8 if "further" in value else 0.5
        result["head_tilt_fb"] = mag if "back" in value else -mag if "front" in value or "forward" in value else 0

    elif label == "head pos: jut":
        mag = 0.3 if "slightly" in value else 0.6
        result["head_jut"] = mag if "forward" in value else -mag if "back" in value else 0

    elif label == "head mvmt: nod":
        mag = 0.3 if "slight" in value else 0.8 if "rapid" in value else 0.4 if "slow" in value else 0.5
        result["head_nod"] = mag

    elif label == "head mvmt: shake":
        mag = 0.3 if "slight" in value else 0.8 if "rapid" in value else 0.4 if "slow" in value else 0.5
        result["head_shake"] = mag

    elif label == "head mvmt: jut":
        mag = 0.3 if "slightly" in value else 0.6
        result["head_jut"] = mag if "forward" in value else -mag if "back" in value else 0

    elif label == "eye brows":
        mag = 0.3 if "slightly" in value else 0.8 if "further" in value else 0.5
        if "raised" in value and "furrowed" in value:
            result["brow_raise"] = mag * 0.5
            result["brow_furrow"] = mag * 0.5
        elif "raised" in value:
            result["brow_raise"] = mag
        elif "lowered" in value or "furrowed" in value:
            result["brow_raise"] = -mag
            if "furrowed" in value:
                result["brow_furrow"] = mag

    elif label == "eye aperture":
        if value == "blink":
            result["eye_aperture"] = -1.0
        elif "closed" in value:
            result["eye_aperture"] = -0.9
        elif "squint" in value:
            result["eye_aperture"] = -0.6 if "further" in value else -0.2 if "slightly" in value else -0.4
        elif "lowered" in value:
            result["eye_aperture"] = -0.5 if "further" in value else -0.15 if "slightly" in value else -0.3
        elif "wide" in value:
            result["eye_aperture"] = 0.3 if "slightly" in value else 0.6
        elif "wider" in value:
            result["eye_aperture"] = 0.8

    elif label == "eye gaze":
        if "up" in value:
            result["eye_gaze_v"] = 0.5
        elif "down" in value:
            result["eye_gaze_v"] = -0.5
        if "left" in value:
            result["eye_gaze_h"] = -0.5
        elif "right" in value:
            result["eye_gaze_h"] = 0.5

    elif label == "mouth":
        if "open" in value:
            result["mouth_open"] = 0.6
        if "spread" in value or "smile" in value:
            result["mouth_spread"] = 0.5
        if "pursed" in value:
            result["lip_pucker"] = 0.6
        if "tongue" in value:
            result["tongue"] = 0.7
        if "corners down" in value:
            result["mouth_spread"] = -0.4

    elif label == "nose":
        if "wrinkle" in value:
            result["nose_wrinkle"] = 0.5

    elif label == "cheeks":
        if "puffed" in value:
            result["cheek_puff"] = 0.6

    return result


def interpolate_nmf_to_frames(
    nmf_annotations: List[Dict],
    start_frame: int,
    end_frame: int,
    fps: float = 30.0
) -> np.ndarray:
    T = end_frame - start_frame + 1
    nmf_sequence = np.zeros((T, N_NMF_CHANNELS), dtype=np.float32)
    channel_idx = {name: i for i, name in enumerate(NMF_CHANNELS)}

    for nm in nmf_annotations:
        if nm.get("tag") != "NON_MANUAL":
            continue

        attrib = nm.get("attrib", {})
        nm_start = int(attrib.get("START_FRAME", start_frame))
        nm_end = int(attrib.get("END_FRAME", end_frame))

        label, value = "", ""
        onset_start, onset_end = nm_start, nm_start
        offset_start, offset_end = nm_end, nm_end

        for field in nm.get("children", []):
            tag = field.get("tag", "")
            if tag == "LABEL":
                label = field.get("text", "").strip("'")
            elif tag == "VALUE":
                value = field.get("text", "").strip("'")
            elif tag == "ONSET":
                onset_attrib = field.get("attrib", {})
                onset_start = int(onset_attrib.get("START_FRAME", nm_start))
                onset_end = int(onset_attrib.get("END_FRAME", nm_start))
            elif tag == "OFFSET":
                offset_attrib = field.get("attrib", {})
                offset_start = int(offset_attrib.get("START_FRAME", nm_end))
                offset_end = int(offset_attrib.get("END_FRAME", nm_end))

        channel_values = parse_nmf_value(label, value)

        for channel_name, val in channel_values.items():
            if channel_name not in channel_idx:
                continue
            ch_idx = channel_idx[channel_name]

            rel_onset_start = max(0, onset_start - start_frame)
            rel_onset_end = max(0, onset_end - start_frame)
            rel_main_start = max(0, nm_start - start_frame)
            rel_main_end = min(T, nm_end - start_frame + 1)
            rel_offset_start = min(T, offset_start - start_frame)
            rel_offset_end = min(T, offset_end - start_frame + 1)

            if rel_onset_end > rel_onset_start:
                for i, t in enumerate(range(rel_onset_start, min(rel_onset_end, rel_main_start))):
                    if t < T:
                        ramp = (i + 1) / (rel_onset_end - rel_onset_start + 1)
                        nmf_sequence[t, ch_idx] = max(nmf_sequence[t, ch_idx], val * ramp)

            for t in range(rel_main_start, rel_main_end):
                if 0 <= t < T:
                    nmf_sequence[t, ch_idx] = max(nmf_sequence[t, ch_idx], val)

            if rel_offset_end > rel_offset_start:
                duration = rel_offset_end - rel_offset_start
                for i, t in enumerate(range(rel_offset_start, rel_offset_end)):
                    if 0 <= t < T:
                        ramp = 1.0 - (i + 1) / (duration + 1)
                        nmf_sequence[t, ch_idx] = max(nmf_sequence[t, ch_idx], val * ramp)

    return nmf_sequence


class FaceMotionDataset(Dataset):
    def __init__(self, npz_dir: str, jsonl_path: str, seq_len: int = 64):
        self.npz_dir = Path(npz_dir)
        self.seq_len = seq_len

        self.records = []
        with open(jsonl_path, "r") as f:
            for line in f:
                r = json.loads(line)
                npz_name = f"{r['base_video']}_utt{r['utterance_id']}.npz"
                npz_path = self.npz_dir / npz_name
                if npz_path.exists():
                    r["npz_path"] = str(npz_path)
                    self.records.append(r)

        print(f"Loaded {len(self.records)} records")
        self._compute_global_neutral_face()

    def _compute_global_neutral_face(self):
        all_faces = []
        for r in self.records[:50]:
            try:
                data = np.load(r["npz_path"])
                face = data["face"]
                if face.max() > 0.001:
                    all_faces.append(face[len(face) // 2])
            except Exception:
                continue

        self.neutral_face = np.mean(all_faces, axis=0).astype(np.float32) if all_faces else np.zeros((468, 3), dtype=np.float32)
        print(f"Computed global neutral face from {len(all_faces)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        data = np.load(r["npz_path"])
        landmarks = data["face"]
        fps = float(data.get("fps", 30))

        if landmarks.shape[-1] == 2:
            landmarks = np.concatenate([landmarks, np.zeros((*landmarks.shape[:-1], 1))], axis=-1)

        T_total = len(landmarks)

        nmf_annotations = []
        translation = ""
        for child in r["xml_utterance"].get("children", []):
            if child.get("tag") == "NON_MANUALS":
                nmf_annotations = child.get("children", [])
            elif child.get("tag") == "TRANSLATION":
                translation = child.get("text", "").strip("'")

        nmf_sequence = interpolate_nmf_to_frames(nmf_annotations, r["start_frame"], r["end_frame"], fps)

        emosign = r.get("emosign", {})
        emotions = np.array([
            emosign.get("joy", 1) / 5.0,
            emosign.get("excited", 1) / 5.0,
            emosign.get("surprise_pos", 1) / 5.0,
            emosign.get("surprise_neg", 1) / 5.0,
            emosign.get("worry", 1) / 5.0,
            emosign.get("sadness", 1) / 5.0,
            emosign.get("fear", 1) / 5.0,
            emosign.get("disgust", 1) / 5.0,
            emosign.get("frustration", 1) / 5.0,
            emosign.get("anger", 1) / 5.0,
        ], dtype=np.float32)

        if T_total >= self.seq_len:
            start = np.random.randint(0, T_total - self.seq_len + 1)
            landmarks = landmarks[start:start + self.seq_len]
            nmf_sequence = nmf_sequence[start:start + self.seq_len]
        else:
            pad_len = self.seq_len - T_total
            landmarks = np.pad(landmarks, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
            nmf_sequence = np.pad(nmf_sequence, ((0, pad_len), (0, 0)), mode="edge")

        landmarks = landmarks.astype(np.float32)
        nmf_sequence = nmf_sequence.astype(np.float32)

        seq_neutral = landmarks[:min(5, len(landmarks))].mean(axis=0).astype(np.float32)
        delta_landmarks = landmarks - seq_neutral

        return {
            "landmarks": torch.from_numpy(landmarks).float(),
            "delta_landmarks": torch.from_numpy(delta_landmarks).float(),
            "nmf": torch.from_numpy(nmf_sequence).float(),
            "emotions": torch.from_numpy(emotions).float(),
            "neutral": torch.from_numpy(seq_neutral).float(),
            "text": translation,
        }


class NMFMotionGeneratorV7(nn.Module):
    def __init__(
        self,
        nmf_dim: int = N_NMF_CHANNELS,
        emotion_dim: int = N_EMOTIONS,
        text_dim: int = 256,
        hidden_dim: int = 256,
        n_landmarks: int = 468,
        neutral_face: Optional[torch.Tensor] = None,
        use_bert: bool = True
    ):
        super().__init__()

        self.n_landmarks = n_landmarks
        self.hidden_dim = hidden_dim

        self.text_encoder = TextEncoder(text_dim, use_bert=use_bert)

        self.nmf_proj = nn.Sequential(
            nn.Linear(nmf_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.film_emotion = FiLMLayer(hidden_dim, emotion_dim)
        self.film_text = FiLMLayer(hidden_dim, text_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0.1)

        self.post_gru = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6),
            nn.Tanh()
        )
        self.pose_scale = nn.Parameter(torch.tensor([0.3, 0.4, 0.2, 0.05, 0.05, 0.02], dtype=torch.float32))

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_landmarks * 3)
        )

        self.register_buffer("region_scales", self._build_region_scales())

        if neutral_face is None:
            neutral_face = torch.zeros(n_landmarks, 3, dtype=torch.float32)
        self.register_buffer("neutral_face", neutral_face)

    def force_text_encoder_cpu(self):
        if hasattr(self, "text_encoder") and hasattr(self.text_encoder, "force_bert_cpu"):
            self.text_encoder.force_bert_cpu()

    def _build_region_scales(self) -> torch.Tensor:
        scales = torch.ones(468, 3) * 0.04

        for idx in FACE_REGIONS["upper_lip"] + FACE_REGIONS["lower_lip"]:
            if idx < 468:
                scales[idx] = torch.tensor([0.06, 0.08, 0.05])

        for idx in FACE_REGIONS["jaw"]:
            if idx < 468:
                scales[idx] = torch.tensor([0.05, 0.10, 0.06])

        for idx in FACE_REGIONS["left_brow"] + FACE_REGIONS["right_brow"]:
            if idx < 468:
                scales[idx] = torch.tensor([0.03, 0.05, 0.03])

        for idx in FACE_REGIONS["left_eye"] + FACE_REGIONS["right_eye"]:
            if idx < 468:
                scales[idx] = torch.tensor([0.02, 0.03, 0.02])

        for idx in FACE_REGIONS["left_cheek"] + FACE_REGIONS["right_cheek"]:
            if idx < 468:
                scales[idx] = torch.tensor([0.03, 0.04, 0.03])

        for idx in FACE_REGIONS["nose"]:
            if idx < 468:
                scales[idx] = torch.tensor([0.02, 0.03, 0.02])

        return scales.float()

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        self.force_text_encoder_cpu()
        return self.text_encoder(texts, device)

    def forward(
        self,
        nmf: torch.Tensor,
        emotions: torch.Tensor,
        text_embed: torch.Tensor,
        neutral: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = nmf.shape

        x = self.nmf_proj(nmf)
        x = self.film_emotion(x, emotions)
        x = self.film_text(x, text_embed)
        x, _ = self.gru(x)
        x = self.post_gru(x)

        pose = self.pose_head(x) * self.pose_scale
        raw_deltas = self.delta_head(x).view(B, T, self.n_landmarks, 3)

        deltas = torch.tanh(raw_deltas) * self.region_scales

        if neutral is None:
            neutral_face = self.neutral_face.unsqueeze(0).expand(B, -1, -1)
        else:
            neutral_face = neutral

        neutral_seq = neutral_face.unsqueeze(1).expand(B, T, -1, -1)
        landmarks = neutral_seq + deltas
        landmarks = self._apply_pose(landmarks, pose)

        result = {
            "landmarks": landmarks,
            "deltas": deltas,
            "pose": pose,
        }

        if return_components:
            result["raw_deltas"] = raw_deltas
            result["neutral"] = neutral_seq

        return result

    def _apply_pose(self, landmarks: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        pitch = pose[:, :, 0:1]
        yaw = pose[:, :, 1:2]

        center = landmarks.mean(dim=2, keepdim=True)
        centered = landmarks - center

        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        x_rot = centered[:, :, :, 0:1] * cos_y.unsqueeze(2) - centered[:, :, :, 2:3] * sin_y.unsqueeze(2)
        z_rot = centered[:, :, :, 0:1] * sin_y.unsqueeze(2) + centered[:, :, :, 2:3] * cos_y.unsqueeze(2)

        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        y_rot = centered[:, :, :, 1:2] * cos_p.unsqueeze(2) - z_rot * sin_p.unsqueeze(2)
        z_rot2 = centered[:, :, :, 1:2] * sin_p.unsqueeze(2) + z_rot * cos_p.unsqueeze(2)

        rotated = torch.cat([x_rot, y_rot, z_rot2], dim=-1)
        translation = pose[:, :, 3:6].unsqueeze(2)

        return rotated + center + translation


def collate_fn(batch):
    return {
        "landmarks": torch.stack([b["landmarks"] for b in batch]),
        "delta_landmarks": torch.stack([b["delta_landmarks"] for b in batch]),
        "nmf": torch.stack([b["nmf"] for b in batch]),
        "emotions": torch.stack([b["emotions"] for b in batch]),
        "neutral": torch.stack([b["neutral"] for b in batch]),
        "texts": [b["text"] for b in batch],
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    model.force_text_encoder_cpu()

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        landmarks_gt = batch["landmarks"].to(device)
        delta_gt = batch["delta_landmarks"].to(device)
        nmf = batch["nmf"].to(device)
        emotions = batch["emotions"].to(device)
        neutral = batch["neutral"].to(device)
        texts = batch["texts"]

        optimizer.zero_grad()

        text_embed = model.encode_text(texts, device)
        outputs = model(nmf, emotions, text_embed, neutral=neutral)

        landmarks_pred = outputs["landmarks"]
        deltas_pred = outputs["deltas"]
        pose = outputs["pose"]

        recon_loss = F.mse_loss(landmarks_pred, landmarks_gt)
        delta_loss = F.mse_loss(deltas_pred, delta_gt)

        vel_pred = landmarks_pred[:, 1:] - landmarks_pred[:, :-1]
        vel_gt = landmarks_gt[:, 1:] - landmarks_gt[:, :-1]
        vel_loss = F.mse_loss(vel_pred, vel_gt)

        acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
        acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
        acc_loss = F.mse_loss(acc_pred, acc_gt)

        mouth_idx = FACE_REGIONS["upper_lip"] + FACE_REGIONS["lower_lip"]
        mouth_loss = F.mse_loss(
            landmarks_pred[:, :, mouth_idx],
            landmarks_gt[:, :, mouth_idx]
        )

        jaw_idx = FACE_REGIONS["jaw"]
        jaw_loss = F.mse_loss(
            landmarks_pred[:, :, jaw_idx],
            landmarks_gt[:, :, jaw_idx]
        )

        brow_idx = FACE_REGIONS["left_brow"] + FACE_REGIONS["right_brow"]
        brow_loss = F.mse_loss(
            landmarks_pred[:, :, brow_idx],
            landmarks_gt[:, :, brow_idx]
        )

        pred_mouth_open = torch.norm(
            landmarks_pred[:, :, 13] - landmarks_pred[:, :, 14], dim=-1
        )
        gt_mouth_open = torch.norm(
            landmarks_gt[:, :, 13] - landmarks_gt[:, :, 14], dim=-1
        )
        mouth_open_loss = F.mse_loss(pred_mouth_open, gt_mouth_open)

        head_nod_signal = nmf[:, :, NMF_CHANNELS.index("head_nod")]
        head_shake_signal = nmf[:, :, NMF_CHANNELS.index("head_shake")]

        nod_loss = F.mse_loss(pose[:, :, 0], head_nod_signal * 0.3)
        shake_loss = F.mse_loss(pose[:, :, 1], head_shake_signal * 0.3)

        loss = (
            recon_loss
            + 0.5 * delta_loss
            + 0.3 * vel_loss
            + 0.15 * acc_loss
            + 2.0 * mouth_loss
            + 1.0 * jaw_loss
            + 0.8 * brow_loss
            + 1.2 * mouth_open_loss
            + 0.4 * nod_loss
            + 0.4 * shake_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}

def generate_new(
    model,
    device: torch.device,
    text: str,
    emotions: Dict[str, float],
    nmf_sequence: Optional[np.ndarray] = None,
    nmf_spec: Optional[Dict[str, List[Tuple[int, int, float]]]] = None,
    n_frames: int = 60,
    fps: float = 30.0
) -> Dict[str, np.ndarray]:
    model.eval()
    model.force_text_encoder_cpu()

    if nmf_sequence is not None:
        T = len(nmf_sequence)
    elif nmf_spec is not None:
        T = n_frames
        nmf_sequence = np.zeros((T, N_NMF_CHANNELS), dtype=np.float32)
        channel_idx = {name: i for i, name in enumerate(NMF_CHANNELS)}

        for channel_name, spans in nmf_spec.items():
            if channel_name not in channel_idx:
                print(f"Warning: Unknown NMF channel '{channel_name}'")
                continue
            ch_idx = channel_idx[channel_name]

            for start, end, value in spans:
                duration = max(end - start, 1)
                onset_len = max(1, int(duration * 0.1))
                offset_len = max(1, int(duration * 0.1))

                for t in range(start, min(end, T)):
                    if t < start + onset_len:
                        ramp = (t - start + 1) / onset_len
                        nmf_sequence[t, ch_idx] = max(nmf_sequence[t, ch_idx], value * ramp)
                    elif t >= end - offset_len:
                        ramp = (end - t) / offset_len
                        nmf_sequence[t, ch_idx] = max(nmf_sequence[t, ch_idx], value * ramp)
                    else:
                        nmf_sequence[t, ch_idx] = max(nmf_sequence[t, ch_idx], value)
    else:
        T = n_frames
        nmf_sequence = np.zeros((T, N_NMF_CHANNELS), dtype=np.float32)

    emotion_names = [
        "joy", "excited", "surprise_pos", "surprise_neg", "worry",
        "sadness", "fear", "disgust", "frustration", "anger"
    ]
    emotion_arr = np.array([emotions.get(name, 0.0) for name in emotion_names], dtype=np.float32)

    nmf_tensor = torch.from_numpy(nmf_sequence).float().unsqueeze(0).to(device)
    emotion_tensor = torch.from_numpy(emotion_arr).float().unsqueeze(0).to(device)
    text_embed = model.encode_text([text], device)
    neutral = model.neutral_face.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            nmf_tensor,
            emotion_tensor,
            text_embed,
            neutral=neutral,
            return_components=True
        )

    return {
        "landmarks": outputs["landmarks"][0].cpu().numpy(),
        "deltas": outputs["deltas"][0].cpu().numpy(),
        "pose": outputs["pose"][0].cpu().numpy(),
        "nmf": nmf_sequence,
        "emotions": emotion_arr,
        "text": text,
    }


def generate_from_record(
    model,
    record: Dict,
    device: torch.device,
    fps: float = 30.0
) -> Dict[str, np.ndarray]:
    model.eval()
    model.force_text_encoder_cpu()

    nmf_annotations = []
    translation = ""
    for child in record["xml_utterance"].get("children", []):
        if child.get("tag") == "NON_MANUALS":
            nmf_annotations = child.get("children", [])
        elif child.get("tag") == "TRANSLATION":
            translation = child.get("text", "").strip("'")

    nmf_sequence = interpolate_nmf_to_frames(
        nmf_annotations, record["start_frame"], record["end_frame"], fps
    )

    emosign = record.get("emosign", {})
    emotions = np.array([
        emosign.get("joy", 1) / 5.0,
        emosign.get("excited", 1) / 5.0,
        emosign.get("surprise_pos", 1) / 5.0,
        emosign.get("surprise_neg", 1) / 5.0,
        emosign.get("worry", 1) / 5.0,
        emosign.get("sadness", 1) / 5.0,
        emosign.get("fear", 1) / 5.0,
        emosign.get("disgust", 1) / 5.0,
        emosign.get("frustration", 1) / 5.0,
        emosign.get("anger", 1) / 5.0,
    ], dtype=np.float32)

    nmf_tensor = torch.from_numpy(nmf_sequence).float().unsqueeze(0).to(device)
    emotions_tensor = torch.from_numpy(emotions).float().unsqueeze(0).to(device)
    text_embed = model.encode_text([translation], device)
    neutral = model.neutral_face.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            nmf_tensor,
            emotions_tensor,
            text_embed,
            neutral=neutral,
            return_components=True
        )

    return {
        "landmarks": outputs["landmarks"][0].cpu().numpy(),
        "deltas": outputs["deltas"][0].cpu().numpy(),
        "pose": outputs["pose"][0].cpu().numpy(),
        "nmf": nmf_sequence,
        "emotions": emotions,
        "text": translation,
    }

def landmarks_to_flame(landmarks: np.ndarray, flame_path: str) -> Dict[str, np.ndarray]:
    T = len(landmarks)

    expression = np.zeros((T, 100), dtype=np.float32)
    jaw_pose = np.zeros((T, 3), dtype=np.float32)
    global_orient = np.zeros((T, 3), dtype=np.float32)

    UPPER_LIP, LOWER_LIP = 13, 14
    LEFT_MOUTH, RIGHT_MOUTH = 61, 291
    LEFT_EYE_T, LEFT_EYE_B = 159, 145
    RIGHT_EYE_T, RIGHT_EYE_B = 386, 374
    LEFT_BROW, RIGHT_BROW = 107, 336
    NOSE_TIP, CHIN, FOREHEAD = 1, 152, 10

    for t in range(T):
        lm = landmarks[t]
        face_height = max(np.linalg.norm(lm[FOREHEAD] - lm[CHIN]), 1e-4)

        mouth_open = np.linalg.norm(lm[UPPER_LIP] - lm[LOWER_LIP]) / face_height
        jaw_pose[t, 0] = np.clip(mouth_open * 3.0, 0.0, 0.6)

        lip_width = np.linalg.norm(lm[LEFT_MOUTH] - lm[RIGHT_MOUTH]) / face_height
        expression[t, 0] = np.clip((lip_width - 0.25) * 5.0, -2.0, 2.0)

        left_eye = np.linalg.norm(lm[LEFT_EYE_T] - lm[LEFT_EYE_B])
        right_eye = np.linalg.norm(lm[RIGHT_EYE_T] - lm[RIGHT_EYE_B])
        eye_open = ((left_eye + right_eye) / 2.0) / face_height
        blink_val = np.clip((0.04 - eye_open) * 30.0, 0.0, 3.0)
        expression[t, 10] = blink_val
        expression[t, 11] = blink_val

        brow_h = ((lm[LEFT_BROW, 1] + lm[RIGHT_BROW, 1]) / 2.0) - lm[NOSE_TIP, 1]
        expression[t, 20] = np.clip((brow_h / face_height - 0.15) * 10.0, -2.0, 2.0)

        left_eye_c = (lm[33] + lm[133]) / 2.0
        right_eye_c = (lm[362] + lm[263]) / 2.0
        horizontal = right_eye_c - left_eye_c
        horizontal /= (np.linalg.norm(horizontal) + 1e-6)

        eye_center = (left_eye_c + right_eye_c) / 2.0
        vertical = eye_center - lm[NOSE_TIP]
        vertical /= (np.linalg.norm(vertical) + 1e-6)

        normal = np.cross(horizontal, vertical)
        normal /= (np.linalg.norm(normal) + 1e-6)

        global_orient[t, 0] = np.arctan2(vertical[2], vertical[1]) * 0.3
        global_orient[t, 1] = np.arctan2(normal[0], normal[2]) * 0.3
        global_orient[t, 2] = np.arctan2(horizontal[1], horizontal[0]) * 0.2

    return {
        "expression": expression,
        "jaw_pose": jaw_pose,
        "global_orient": global_orient,
    }


def render_flame_mesh(flame_params: Dict, flame_path: str, output_path: str) -> np.ndarray:
    with open(flame_path, "rb") as f:
        flame_data = pickle.load(f, encoding="latin1")

    v_template = safe_to_numpy(flame_data["v_template"], np.float32)
    shapedirs = safe_to_numpy(flame_data["shapedirs"], np.float32)
    faces = safe_to_numpy(flame_data["f"], np.int32)

    if shapedirs.ndim == 3 and shapedirs.shape[2] >= 400:
        shapedirs = shapedirs[:, :, 300:400]
    elif shapedirs.ndim == 3 and shapedirs.shape[2] >= 100:
        shapedirs = shapedirs[:, :, :100]
    else:
        raise ValueError(f"Unexpected shapedirs shape: {shapedirs.shape}")

    T = len(flame_params["expression"])
    n_verts = v_template.shape[0]
    vertices = np.zeros((T, n_verts, 3), dtype=np.float32)

    for t in range(T):
        v = v_template.copy()

        expr = flame_params["expression"][t]
        n_expr = min(len(expr), shapedirs.shape[2])
        for i in range(n_expr):
            v += shapedirs[:, :, i] * float(expr[i])

        jaw_rot = float(flame_params["jaw_pose"][t, 0])
        if abs(jaw_rot) > 0.01:
            mean_y = float(v[:, 1].mean())
            lower_mask = (v[:, 1] < mean_y - 0.02)

            if np.any(lower_mask):
                pivot = v[lower_mask].mean(axis=0)

                cos_j = np.cos(jaw_rot)
                sin_j = np.sin(jaw_rot)

                lower_v = v[lower_mask] - pivot
                new_y = lower_v[:, 1] * cos_j - lower_v[:, 2] * sin_j
                new_z = lower_v[:, 1] * sin_j + lower_v[:, 2] * cos_j
                lower_v[:, 1] = new_y
                lower_v[:, 2] = new_z
                v[lower_mask] = lower_v + pivot

        orient = flame_params["global_orient"][t]

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(orient[0]), -np.sin(orient[0])],
            [0, np.sin(orient[0]),  np.cos(orient[0])]
        ], dtype=np.float32)

        Ry = np.array([
            [np.cos(orient[1]), 0, np.sin(orient[1])],
            [0, 1, 0],
            [-np.sin(orient[1]), 0, np.cos(orient[1])]
        ], dtype=np.float32)

        Rz = np.array([
            [np.cos(orient[2]), -np.sin(orient[2]), 0],
            [np.sin(orient[2]),  np.cos(orient[2]), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        R = Rz @ Ry @ Rx

        center = v.mean(axis=0)
        v = (v - center) @ R.T + center

        Rx_fix = np.array([
            [1, 0, 0],
            [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)],
            [0, np.sin(-np.pi / 2),  np.cos(-np.pi / 2)]
        ], dtype=np.float32)

        Rz_fix = np.array([
            [np.cos(np.pi), -np.sin(np.pi), 0],
            [np.sin(np.pi),  np.cos(np.pi), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        R_fix = Rz_fix @ Rx_fix

        center = v.mean(axis=0)
        v = (v - center) @ R_fix.T + center

        v[:, 2] = -v[:, 2]
        vertices[t] = v

    np.savez(
        output_path,
        vertices=vertices,
        faces=faces,
        expression=flame_params["expression"],
        jaw_pose=flame_params["jaw_pose"],
        global_orient=flame_params["global_orient"],
    )
    return vertices


def render_3d_video(vertices: np.ndarray, faces: np.ndarray, output_path: str, fps: int = 30):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import imageio.v2 as imageio
    except ImportError as e:
        print(f"Missing dependency for matplotlib rendering: {e}")
        print("Try: pip install matplotlib imageio")
        return

    T = len(vertices)
    faces = np.asarray(faces, dtype=np.int32)

    all_xyz = vertices.reshape(-1, 3)
    x_min, y_min, z_min = all_xyz.min(axis=0)
    x_max, y_max, z_max = all_xyz.max(axis=0)

    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    z_mid = (z_min + z_max) / 2.0
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0 + 1e-6

    frames = []

    for t in range(T):
        v = vertices[t]

        fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        tris = v[faces]
        mesh = Poly3DCollection(tris, linewidths=0.05, alpha=1.0)
        mesh.set_facecolor((0.96, 0.80, 0.69, 1.0))
        mesh.set_edgecolor((0.35, 0.35, 0.35, 0.08))
        ax.add_collection3d(mesh)

        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)

        ax.view_init(elev=0, azim=-90)
        ax.set_axis_off()
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

        fig.tight_layout(pad=0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(img)
        plt.close(fig)

        if t % 10 == 0:
            print(f"  Rendered frame {t}/{T}")

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved 3D video to {output_path}")


def render_landmarks_video(landmarks: np.ndarray, output_path: str, fps: int = 30):
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed")
        return

    T, N, _ = landmarks.shape
    W, H = 512, 512

    lm_2d = landmarks[:, :, :2].copy()
    min_xy = lm_2d.min(axis=(0, 1))
    max_xy = lm_2d.max(axis=(0, 1))
    denom = max(float(np.max(max_xy - min_xy)), 1e-6)
    scale = min(W, H) * 0.8 / denom
    lm_2d = (lm_2d - min_xy) * scale + np.array([W * 0.1, H * 0.1])

    CONNECTIONS = [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
        (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
        (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
        (314, 405), (405, 321), (321, 375), (375, 291), (291, 61),
        (33, 133), (133, 173), (173, 246), (246, 33),
        (362, 263), (263, 466), (466, 398), (398, 362),
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for t in range(T):
        frame = np.ones((H, W, 3), dtype=np.uint8) * 255
        pts = lm_2d[t].astype(np.int32)

        for i, j in CONNECTIONS:
            if i < N and j < N:
                cv2.line(frame, tuple(pts[i]), tuple(pts[j]), (200, 200, 200), 1)

        for pt in pts:
            cv2.circle(frame, tuple(pt), 2, (0, 100, 255), -1)

        for idx in [13, 14, 61, 291]:
            if idx < N:
                cv2.circle(frame, tuple(pts[idx]), 4, (0, 0, 255), -1)

        cv2.putText(frame, f"Frame {t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        writer.write(frame)

    writer.release()
    print(f"Saved landmark video to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "generate", "generate_new", "analyse"])
    parser.add_argument("--npz_dir", default="clips_npz_track")
    parser.add_argument("--jsonl", default="out/metadata/merged_records.jsonl")
    parser.add_argument("--output_dir", default="nmf_motion_v7_cpu_bert_out")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--flame_model", default="flame2023.pkl")
    parser.add_argument("--record_idx", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--output_format", choices=["npz", "video", "obj", "all"], default="all")
    parser.add_argument("--render_3d", action="store_true")
    parser.add_argument("--no_bert", action="store_true")

    parser.add_argument("--text", type=str, default="Hello, how are you?")
    parser.add_argument("--n_frames", type=int, default=60)
    parser.add_argument("--emotion", type=str, default="neutral",
                        choices=["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"])
    parser.add_argument("--nmf_preset", type=str, default="speaking",
                        choices=["neutral", "speaking", "nodding", "questioning", "emphatic"])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mode == "train":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = FaceMotionDataset(args.npz_dir, args.jsonl, args.seq_len)
        dataloader = DataLoader(
            dataset,
            args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )

        neutral = torch.from_numpy(dataset.neutral_face).float()
        model = NMFMotionGeneratorV7(
            neutral_face=neutral,
            use_bert=not args.no_bert
        ).to(device)
        model.force_text_encoder_cpu()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        best_loss = float("inf")
        for epoch in range(args.epochs):
            metrics = train_epoch(model, dataloader, optimizer, device, epoch)
            scheduler.step()

            print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {metrics['loss']:.6f}")

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "neutral_face": neutral,
                    "loss": best_loss,
                }, output_dir / "best.pt")
                print(f"  Saved (loss={best_loss:.6f})")

        print(f"\nDone! Best loss: {best_loss:.6f}")

    elif args.mode == "generate":
        ckpt_path = args.checkpoint or Path(args.output_dir) / "best.pt"
        print(f"Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        neutral = ckpt.get("neutral_face", torch.zeros(468, 3))
        model = NMFMotionGeneratorV7(
            neutral_face=neutral,
            use_bert=not args.no_bert
        ).to(device)
        model.force_text_encoder_cpu()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        records = [json.loads(l) for l in open(args.jsonl)]
        record = records[args.record_idx]

        print(f"\nGenerating for record {args.record_idx}:")
        print(f"  Video: {record['base_video']}")

        result = generate_from_record(model, record, device)
        landmarks = result["landmarks"]

        print(f"  Generated {len(landmarks)} frames")
        print(f"  Text: {result['text'][:60]}...")
        print(f"  Emotions: joy={result['emotions'][0]:.2f}, anger={result['emotions'][9]:.2f}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        lm_path = output_dir / f"landmarks_{args.record_idx}.npz"
        np.savez(lm_path, **result)
        print(f"  Saved landmarks: {lm_path}")

        if args.output_format in ["video", "all"]:
            vid_path = output_dir / f"landmarks_{args.record_idx}.mp4"
            render_landmarks_video(landmarks, str(vid_path))

        if Path(args.flame_model).exists():
            flame_params = landmarks_to_flame(landmarks, args.flame_model)
            flame_npz_path = output_dir / f"flame_{args.record_idx}.npz"
            vertices = render_flame_mesh(flame_params, args.flame_model, str(flame_npz_path))
            print(f"  FLAME vertices: {vertices.shape}")

            if args.render_3d:
                with open(args.flame_model, "rb") as f:
                    faces = safe_to_numpy(pickle.load(f, encoding="latin1")["f"], np.int32)
                vid3d_path = output_dir / f"flame_3d_{args.record_idx}.mp4"
                render_3d_video(vertices, faces, str(vid3d_path))
        else:
            print(f"  FLAME model not found: {args.flame_model}")

        print("\nGeneration complete!")

    elif args.mode == "generate_new":
        ckpt_path = args.checkpoint or Path(args.output_dir) / "best.pt"
        print(f"Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        neutral = ckpt.get("neutral_face", torch.zeros(468, 3))
        model = NMFMotionGeneratorV7(
            neutral_face=neutral,
            use_bert=not args.no_bert
        ).to(device)
        model.force_text_encoder_cpu()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        EMOTION_PRESETS = {
            "neutral": {},
            "happy": {"joy": 0.8, "excited": 0.5},
            "sad": {"sadness": 0.7, "worry": 0.3},
            "angry": {"anger": 0.8, "frustration": 0.6},
            "surprised": {"surprise_pos": 0.8, "excited": 0.4},
            "fearful": {"fear": 0.7, "worry": 0.5, "surprise_neg": 0.3},
            "disgusted": {"disgust": 0.8, "frustration": 0.3},
        }

        NMF_PRESETS = {
            "neutral": {},
            "speaking": {
                "mouth_open": [(0, args.n_frames, 0.4)],
                "mouth_spread": [(0, args.n_frames, 0.2)],
            },
            "nodding": {
                "head_nod": [
                    (0, args.n_frames // 3, 0.5),
                    (args.n_frames // 3, 2 * args.n_frames // 3, 0.5),
                    (2 * args.n_frames // 3, args.n_frames, 0.5),
                ],
                "mouth_open": [(0, args.n_frames, 0.3)],
            },
            "questioning": {
                "brow_raise": [(0, args.n_frames, 0.6)],
                "head_tilt_side": [(0, args.n_frames, 0.3)],
                "eye_aperture": [(0, args.n_frames, 0.3)],
            },
            "emphatic": {
                "brow_raise": [(0, args.n_frames // 2, 0.7)],
                "brow_furrow": [(args.n_frames // 2, args.n_frames, 0.5)],
                "head_nod": [(args.n_frames // 4, 3 * args.n_frames // 4, 0.6)],
                "mouth_open": [(0, args.n_frames, 0.5)],
            },
        }

        emotions = EMOTION_PRESETS.get(args.emotion, {})
        nmf_spec = NMF_PRESETS.get(args.nmf_preset, {})

        print(f"\n=== GENERATING NEW SEQUENCE ===")
        print(f"  Text: {args.text}")
        print(f"  Emotion: {args.emotion} -> {emotions}")
        print(f"  NMF preset: {args.nmf_preset}")
        print(f"  Frames: {args.n_frames}")

        result = generate_new(
            model=model,
            device=device,
            text=args.text,
            emotions=emotions,
            nmf_spec=nmf_spec,
            n_frames=args.n_frames
        )

        landmarks = result["landmarks"]
        print(f"\n  Generated {len(landmarks)} frames")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_text = args.text[:20].replace(" ", "_").replace("?", "").replace("!", "")
        output_name = f"gen_{args.emotion}_{args.nmf_preset}_{safe_text}"

        lm_path = output_dir / f"{output_name}.npz"
        np.savez(lm_path, **result)
        print(f"  Saved: {lm_path}")

        if args.output_format in ["video", "all"]:
            vid_path = output_dir / f"{output_name}.mp4"
            render_landmarks_video(landmarks, str(vid_path))

        if Path(args.flame_model).exists():
            flame_params = landmarks_to_flame(landmarks, args.flame_model)
            flame_path = output_dir / f"{output_name}_flame.npz"
            vertices = render_flame_mesh(flame_params, args.flame_model, str(flame_path))
            print(f"  FLAME vertices: {vertices.shape}")

            if args.render_3d:
                with open(args.flame_model, "rb") as f:
                    faces = safe_to_numpy(pickle.load(f, encoding="latin1")["f"], np.int32)
                vid3d_path = output_dir / f"{output_name}_3d.mp4"
                render_3d_video(vertices, faces, str(vid3d_path))

        print("\n=== Generation complete! ===")
        print(f"Output: {output_dir / output_name}.*")

    elif args.mode == "analyse":
        dataset = FaceMotionDataset(args.npz_dir, args.jsonl, args.seq_len)
        print(f"\nRecords: {len(dataset)}")

        sample = dataset[0]
        print("Sample shapes:")
        print(f"  landmarks: {sample['landmarks'].shape}")
        print(f"  nmf: {sample['nmf'].shape}")
        print(f"  emotions: {sample['emotions'].shape}")
        print(f"  text: {sample['text'][:50]}...")

        nmf = sample["nmf"].numpy()
        print("\nNMF channel activity:")
        for i, name in enumerate(NMF_CHANNELS):
            act = np.abs(nmf[:, i]).mean()
            if act > 0.01:
                print(f"  {name}: {act:.3f}")


if __name__ == "__main__":
    main()