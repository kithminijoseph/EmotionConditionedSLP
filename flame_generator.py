"""
USAGE:
python flame_generator.py train \
  --flame_dir flame_params_for_clips \
  --jsonl merged_records.jsonl \
  --output_dir best_checkpoint_split_160 \
  --split

python flame_generator.py train \
  --flame_dir flame_params_for_clips \
  --jsonl merged_records.jsonl \
  --output_dir best_checkpoint_full_200 \
  --full

python flame_generator.py reconstruct \
  --checkpoint gen_v9/best.pt \
  --heldout_dir gen_v9/heldout_samples \
  --output_dir reconstructed_test_samples

python flame_generator.py generate \
  --checkpoint gen_v9_normalised_full/best.pt \
  --text "I really like that" \
  --emotion "joy=1.0,excited=0.55" \
  --nmf "mouth_spread=0.95,mouth_open=0.2,brow_raise=0.25" \
  --output happiness.npz
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

N_EXPRESSION = 100
N_JAW = 3
N_EYE = 6
N_ORIENT = 3
N_FLAME = N_EXPRESSION + N_JAW + N_EYE + N_ORIENT

NMF_CHANNELS = [
    "head_turn", "head_tilt_side", "head_tilt_fb", "head_nod", "head_shake", "head_jut",
    "brow_raise", "brow_furrow", "eye_aperture", "eye_gaze_h", "eye_gaze_v",
    "mouth_open", "mouth_spread", "lip_pucker", "tongue", "nose_wrinkle", "cheek_puff",
]
N_NMF = len(NMF_CHANNELS)
NMF_IDX = {n: i for i, n in enumerate(NMF_CHANNELS)}

EMOTION_CHANNELS = [
    "joy", "excited", "surprise_pos", "surprise_neg", "worry",
    "sadness", "fear", "disgust", "frustration", "anger"
]
N_EMOTION = len(EMOTION_CHANNELS)

MIN_FRAMES = 24
MAX_FRAMES = 180
FRAMES_PER_WORD = 12


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def text_to_duration(text: str) -> int:
    words = text.split()
    n_words = max(1, len(words))
    base_frames = n_words * FRAMES_PER_WORD
    n_pauses = text.count(',') + text.count(';') + text.count(':')
    n_stops = text.count('.') + text.count('!') + text.count('?')
    pause_frames = n_pauses * 4 + n_stops * 6
    avg_word_len = sum(len(w) for w in words) / n_words
    complexity_factor = 1.0 + (avg_word_len - 4) * 0.05
    duration = int((base_frames + pause_frames) * complexity_factor)
    return max(MIN_FRAMES, min(MAX_FRAMES, duration))


def text_to_ids(text: str, max_len: int = 64) -> np.ndarray:
    ids = np.fromiter((ord(c) % 256 for c in text[:max_len]), dtype=np.int64)
    return ids if len(ids) > 0 else np.array([0], dtype=np.int64)


def extract_phoneme_features(text: str) -> np.ndarray:
    total = max(1, len(text))
    return np.array([
        sum(c in "MBPmbp" for c in text) / total,
        sum(c in "OUWouw" for c in text) / total,
        sum(c in "AHah" for c in text) / total,
        sum(c in "FVSZfvsz" for c in text) / total,
        sum(c in "TDNtdn" for c in text) / total,
        sum(c in "EIei" for c in text) / total,
        float("!" in text),
        float("?" in text),
        min(len(text) / 30.0, 1.0),
        sum(c.isupper() for c in text) / total,
    ], dtype=np.float32)


def create_neutral_blend_curve(n_frames: int, onset_ratio: float = 0.12, offset_ratio: float = 0.12) -> np.ndarray:
    curve = np.ones(n_frames, dtype=np.float32)
    onset_frames = max(3, int(n_frames * onset_ratio))
    offset_frames = max(3, int(n_frames * offset_ratio))
    for t in range(onset_frames):
        curve[t] = 0.5 * (1 - np.cos(np.pi * t / onset_frames))
    for t in range(offset_frames):
        idx = n_frames - offset_frames + t
        curve[idx] = 0.5 * (1 + np.cos(np.pi * t / offset_frames))
    return curve


def build_nmf_sequence(nmf_spec: Dict[str, float], duration: int, use_neutral_blend: bool = True) -> np.ndarray:
    nmf = np.zeros((duration, N_NMF), dtype=np.float32)
    blend = create_neutral_blend_curve(duration) if use_neutral_blend else np.ones(duration, dtype=np.float32)
    for name, val in nmf_spec.items():
        if name in NMF_IDX:
            nmf[:, NMF_IDX[name]] = val * blend
    return nmf


def temporal_features(T: int) -> np.ndarray:
    if T <= 1:
        return np.zeros((T, 4), dtype=np.float32)
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    return np.stack([
        t,
        np.sin(2 * np.pi * t),
        np.cos(2 * np.pi * t),
        2 * t - 1.0,
    ], axis=-1).astype(np.float32)


class TextEncoder(nn.Module):
    def __init__(self, output_dim: int = 192):
        super().__init__()
        self.char_embed = nn.Embedding(256, 64)
        self.phoneme_proj = nn.Linear(10, 64)
        self.out = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor, phoneme_feats: torch.Tensor) -> torch.Tensor:
        emb = self.char_embed(token_ids)
        weights = token_mask.unsqueeze(-1)
        pooled = (emb * weights).sum(1) / token_mask.sum(1, keepdim=True).clamp(min=1.0)
        phon = self.phoneme_proj(phoneme_feats)
        return self.out(torch.cat([pooled, phon], dim=-1))


class EmotionAmplifier(nn.Module):
    def __init__(self, emotion_dim: int = N_EMOTION, hidden_dim: int = 128):
        super().__init__()
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mouth_head = nn.Linear(hidden_dim, 8)
        self.cheek_head = nn.Linear(hidden_dim, 4)
        self.nose_head = nn.Linear(hidden_dim, 4)
        self.brow_head = nn.Linear(hidden_dim, 6)
        self.amp_scale = nn.Parameter(torch.tensor(0.20))
        self.register_buffer('mouth_idx', torch.tensor([10, 11, 12, 13, 14, 15, 5, 3]))
        self.register_buffer('cheek_idx', torch.tensor([16, 17, 20, 21]))
        self.register_buffer('nose_idx', torch.tensor([18, 19, 22, 23]))
        self.register_buffer('brow_idx', torch.tensor([30, 31, 32, 33, 34, 35]))

    def forward(self, emotion: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.emotion_encoder(emotion)
        return {
            'mouth': torch.tanh(self.mouth_head(h)) * self.amp_scale,
            'cheek': torch.tanh(self.cheek_head(h)) * self.amp_scale * 0.6,
            'nose': torch.tanh(self.nose_head(h)) * self.amp_scale * 0.4,
            'brow': torch.tanh(self.brow_head(h)) * self.amp_scale * 0.75,
            'mouth_idx': self.mouth_idx,
            'cheek_idx': self.cheek_idx,
            'nose_idx': self.nose_idx,
            'brow_idx': self.brow_idx,
        }


class FiLM(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, feature_dim)
        self.shift = nn.Linear(cond_dim, feature_dim)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and cond.dim() == 2:
            cond = cond.unsqueeze(1)
        return self.scale(cond) * x + self.shift(cond)


class FLAMEGeneratorV9(nn.Module):
    def __init__(self, hidden_dim: int = 192, gru_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = TextEncoder(hidden_dim)
        self.nmf_encoder = nn.Sequential(
            nn.Linear(N_NMF + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emotion_encoder = nn.Sequential(
            nn.Linear(N_EMOTION, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )
        self.emotion_amplifier = EmotionAmplifier(N_EMOTION, 128)
        self.film_emo = FiLM(hidden_dim, hidden_dim)
        self.film_text = FiLM(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)
        self.post = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout))

        self.expression_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_EXPRESSION),
        )
        self.jaw_head = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.GELU(), nn.Linear(64, N_JAW), nn.Tanh())
        self.eye_head = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.GELU(), nn.Linear(64, N_EYE), nn.Tanh())
        self.orient_head = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.GELU(), nn.Linear(64, N_ORIENT), nn.Tanh())

        self.jaw_scale = nn.Parameter(torch.tensor([0.28, 0.12, 0.10]))
        self.eye_scale = nn.Parameter(torch.tensor([0.22] * 6))
        self.orient_scale = nn.Parameter(torch.tensor([0.22, 0.28, 0.18]))

        self.nmf_to_jaw = nn.Linear(N_NMF, 1)
        self.nmf_to_mouth = nn.Linear(N_NMF, 16)

    def _apply_emotion_amplification(self, expression: torch.Tensor, emo_amp: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, T, D = expression.shape
        delta = torch.zeros_like(expression)
        for region in ['mouth', 'cheek', 'nose', 'brow']:
            amp = emo_amp[region]
            idx = emo_amp[f'{region}_idx']
            n_idx = min(len(idx), amp.shape[-1])
            for i in range(n_idx):
                expr_idx = int(idx[i].item())
                if expr_idx < D:
                    delta[:, :, expr_idx] = delta[:, :, expr_idx] + amp[:, i:i+1]
        return expression + delta

    def forward(self, nmf: torch.Tensor, emotion: torch.Tensor, token_ids: torch.Tensor, token_mask: torch.Tensor, phoneme_feats: torch.Tensor, time_feats: torch.Tensor, blend_curve: Optional[torch.Tensor] = None, apply_neutral_blend: bool = False) -> Dict[str, torch.Tensor]:
        nmf_in = torch.cat([nmf, time_feats], dim=-1)
        nmf_feat = self.nmf_encoder(nmf_in)
        emo_feat = self.emotion_encoder(emotion)
        text_feat = self.text_encoder(token_ids, token_mask, phoneme_feats)
        emo_amp = self.emotion_amplifier(emotion)

        x = self.film_emo(nmf_feat, emo_feat)
        x = self.film_text(x, text_feat)
        x, _ = self.gru(x)
        x = self.post(x)

        expression = self.expression_head(x)
        jaw_pose = self.jaw_head(x) * self.jaw_scale
        eye_pose = self.eye_head(x) * self.eye_scale
        global_orient = self.orient_head(x) * self.orient_scale

        mouth_expr_front = expression[:, :, :16] + 0.35 * self.nmf_to_mouth(nmf)
        expression = torch.cat([mouth_expr_front, expression[:, :, 16:]], dim=-1)

        jaw0 = jaw_pose[:, :, 0:1] + 0.25 * torch.tanh(self.nmf_to_jaw(nmf))
        jaw_pose = torch.cat([torch.clamp(jaw0, -0.45, 0.45), jaw_pose[:, :, 1:]], dim=-1)

        expression = self._apply_emotion_amplification(expression, emo_amp)

        if apply_neutral_blend and blend_curve is not None:
            blend = blend_curve.unsqueeze(-1)
            expression = expression * blend
            jaw_pose = jaw_pose * blend
            eye_pose = eye_pose * (0.3 + 0.7 * blend)
            global_orient = global_orient * (0.2 + 0.8 * blend)

        flame_params = torch.cat([expression, jaw_pose, eye_pose, global_orient], dim=-1)
        return {
            'flame_params': flame_params,
            'expression': expression,
            'jaw_pose': jaw_pose,
            'eye_pose': eye_pose,
            'global_orient': global_orient,
            'emotion_amp': emo_amp,
        }


@dataclass
class Sample:
    flame_params: np.ndarray
    nmf: np.ndarray
    emotions: np.ndarray
    token_ids: np.ndarray
    phoneme_feats: np.ndarray
    text: str
    clip_id: str


class FLAMEDataset(Dataset):
    def __init__(self, samples: List[Sample], seq_len: int = 64):
        self.samples = samples
        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        T = len(s.flame_params)
        if T > self.seq_len:
            start = np.random.randint(0, T - self.seq_len + 1)
            flame = s.flame_params[start:start + self.seq_len]
            nmf = s.nmf[start:start + self.seq_len]
        else:
            pad = self.seq_len - T
            flame = np.pad(s.flame_params, ((0, pad), (0, 0)), mode='edge')
            nmf = np.pad(s.nmf, ((0, pad), (0, 0)), mode='edge')
        tfeat = temporal_features(len(flame))
        blend = np.ones(len(flame), dtype=np.float32)  # no neutral suppression during training
        return {
            'flame_params': torch.from_numpy(flame).float(),
            'nmf': torch.from_numpy(nmf).float(),
            'emotions': torch.from_numpy(s.emotions).float(),
            'token_ids': torch.from_numpy(s.token_ids).long(),
            'phoneme_feats': torch.from_numpy(s.phoneme_feats).float(),
            'time_feats': torch.from_numpy(tfeat).float(),
            'blend_curve': torch.from_numpy(blend).float(),
        }


def extract_emotions(rec) -> np.ndarray:
    e = rec.get('emosign', {})
    return np.array([max(0.0, min(1.0, (e.get(n, 1) - 1) / 3.0)) for n in EMOTION_CHANNELS], dtype=np.float32)


def extract_text(rec) -> str:
    xml = rec.get('xml_utterance', {})
    for child in xml.get('children', []):
        if child.get('tag') == 'TRANSLATION':
            return child.get('text', '').strip("'")
    return ""

def parse_nmf_label(label: str, value: str) -> Dict[str, float]:
    result = {}
    if 'head pos: turn' in label:
        mag = 0.3 if 'slightly' in value else 0.7
        if 'right' in value:
            result['head_turn'] = mag
        elif 'left' in value:
            result['head_turn'] = -mag
    elif 'head mvmt: nod' in label:
        result['head_nod'] = 0.55
    elif 'head mvmt: shake' in label:
        result['head_shake'] = 0.55
    elif 'eye brows' in label:
        mag = 0.3 if 'slightly' in value else 0.75
        if 'raised' in value:
            result['brow_raise'] = mag
        elif 'lowered' in value or 'furrowed' in value:
            result['brow_furrow'] = mag
    elif 'eye aperture' in label:
        if 'blink' in value or 'closed' in value:
            result['eye_aperture'] = -0.9
        elif 'squint' in value:
            result['eye_aperture'] = -0.5
        elif 'wide' in value:
            result['eye_aperture'] = 0.6
    elif 'mouth' in label:
        if 'open' in value:
            result['mouth_open'] = 0.75
        if 'spread' in value or 'smile' in value:
            result['mouth_spread'] = 0.55
    return result

def extract_nmf_from_record(rec, T: int) -> np.ndarray:
    nmf = np.zeros((T, N_NMF), dtype=np.float32)
    xml = rec.get('xml_utterance', {})
    start_frame = rec.get('start_frame', 0)
    end_frame = rec.get('end_frame', T)
    duration = max(1, end_frame - start_frame)
    for child in xml.get('children', []):
        if child.get('tag') != 'NON_MANUALS':
            continue
        for nm in child.get('children', []):
            if nm.get('tag') != 'NON_MANUAL':
                continue
            att = nm.get('attrib', {})
            s = int(att.get('START_FRAME', start_frame)) - start_frame
            e = int(att.get('END_FRAME', end_frame)) - start_frame
            s_scaled = int(s * T / duration)
            e_scaled = int(e * T / duration)
            label, value = '', ''
            for f in nm.get('children', []):
                if f.get('tag') == 'LABEL':
                    label = f.get('text', '').strip("'").lower()
                elif f.get('tag') == 'VALUE':
                    value = f.get('text', '').strip("'").lower()
            vals = parse_nmf_label(label, value)
            for ch, v in vals.items():
                idx = NMF_IDX.get(ch)
                if idx is None:
                    continue
                lo, hi = max(0, s_scaled), min(T, e_scaled + 1)
                if hi > lo:
                    if v >= 0:
                        nmf[lo:hi, idx] = np.maximum(nmf[lo:hi, idx], v)
                    else:
                        nmf[lo:hi, idx] = np.minimum(nmf[lo:hi, idx], v)
    return nmf

def load_samples(flame_dir: str, jsonl_path: str) -> List[Sample]:
    flame_path = Path(flame_dir)
    samples = []
    with open(jsonl_path, 'r') as f:
        records = [json.loads(line) for line in f]
    for rec in records:
        video = rec.get('base_video')
        utt = rec.get('utterance_id')
        if not video or utt is None:
            continue
        npz_path = flame_path / f"{video}_utt{utt}_flame.npz"
        if not npz_path.exists():
            continue
        try:
            data = np.load(npz_path)
            expr = data['expression'].astype(np.float32)
            if len(expr) < 10:
                continue
            jaw = data.get('jaw_pose', np.zeros((len(expr), 3), dtype=np.float32)).astype(np.float32)
            eye = data.get('eye_pose', np.zeros((len(expr), 6), dtype=np.float32)).astype(np.float32)
            orient = data.get('global_orient', np.zeros((len(expr), 3), dtype=np.float32)).astype(np.float32)
            flame_params = np.concatenate([expr, jaw, eye, orient], axis=-1)
            T = len(flame_params)
            nmf = extract_nmf_from_record(rec, T)
            emotions = extract_emotions(rec)
            text = extract_text(rec)
            samples.append(Sample(
                flame_params=flame_params,
                nmf=nmf,
                emotions=emotions,
                token_ids=text_to_ids(text),
                phoneme_feats=extract_phoneme_features(text),
                text=text,
                clip_id=f"{video}_utt{utt}",
            ))
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue
    print(f"Loaded {len(samples)} valid samples")
    return samples

def save_heldout_samples(samples: List[Sample], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s in samples:
        out_path = out_dir / f"{s.clip_id}.npz"
        np.savez(
            out_path,
            clip_id=s.clip_id,
            text=s.text,
            emotions=s.emotions,
            nmf=s.nmf,
            flame_params=s.flame_params,
            token_ids=s.token_ids,
            phoneme_feats=s.phoneme_feats,
            fps=np.float32(30.0),
        )
        manifest.append({'clip_id': s.clip_id, 'file': out_path.name, 'duration': len(s.flame_params)})
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

def collate_fn(batch):
    max_len = max(b['token_ids'].shape[0] for b in batch)
    token_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    token_mask = torch.zeros(len(batch), max_len)
    for i, b in enumerate(batch):
        L = len(b['token_ids'])
        token_ids[i, :L] = b['token_ids']
        token_mask[i, :L] = 1.0
    return {
        'flame_params': torch.stack([b['flame_params'] for b in batch]),
        'nmf': torch.stack([b['nmf'] for b in batch]),
        'emotions': torch.stack([b['emotions'] for b in batch]),
        'token_ids': token_ids,
        'token_mask': token_mask,
        'phoneme_feats': torch.stack([b['phoneme_feats'] for b in batch]),
        'time_feats': torch.stack([b['time_feats'] for b in batch]),
        'blend_curve': torch.stack([b['blend_curve'] for b in batch]),
    }

def compute_loss(out: Dict, target: torch.Tensor, nmf: torch.Tensor) -> tuple[torch.Tensor, Dict]:
    pred = out['flame_params']
    expr_loss = F.smooth_l1_loss(out['expression'], target[:, :, :100])
    jaw_loss = F.smooth_l1_loss(out['jaw_pose'], target[:, :, 100:103])
    eye_loss = F.smooth_l1_loss(out['eye_pose'], target[:, :, 103:109])
    orient_loss = F.smooth_l1_loss(out['global_orient'], target[:, :, 109:112])

    vel_pred = pred[:, 1:] - pred[:, :-1]
    vel_gt = target[:, 1:] - target[:, :-1]
    vel_loss = F.smooth_l1_loss(vel_pred, vel_gt)

    accel_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
    accel_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
    accel_loss = F.smooth_l1_loss(accel_pred, accel_gt)

    mouth_open = nmf[:, :, NMF_IDX['mouth_open']:NMF_IDX['mouth_open']+1]
    jaw_ctrl = F.mse_loss(out['jaw_pose'][:, :, 0:1], 0.30 * mouth_open)

    mouth_spread = nmf[:, :, NMF_IDX['mouth_spread']:NMF_IDX['mouth_spread']+1]
    smile_pred = 0.5 * (out['expression'][:, :, 10:11] + out['expression'][:, :, 11:12])
    frown_pred = 0.5 * (out['expression'][:, :, 12:13] + out['expression'][:, :, 13:14])
    smile_loss = F.mse_loss(smile_pred, 0.22 * mouth_spread.clamp(min=0))
    frown_loss = F.mse_loss(frown_pred, 0.22 * (-mouth_spread).clamp(min=0))

    amp_reg = 0.0
    for k, v in out['emotion_amp'].items():
        if isinstance(v, torch.Tensor) and v.requires_grad:
            amp_reg = amp_reg + v.abs().mean()

    total = (
        2.3 * expr_loss +
        2.8 * jaw_loss +
        0.8 * eye_loss +
        0.6 * orient_loss +
        0.35 * vel_loss +
        0.20 * accel_loss +
        0.80 * jaw_ctrl +
        0.45 * smile_loss +
        0.45 * frown_loss +
        0.02 * amp_reg
    )
    return total, {
        'loss': float(total.item()),
        'expr': float(expr_loss.item()),
        'jaw': float(jaw_loss.item()),
        'vel': float(vel_loss.item()),
    }


def train(args):
    set_seed(args.seed)
    device = pick_device(args.force_cpu)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_samples(args.flame_dir, args.jsonl)
    if len(all_samples) < 10:
        print("ERROR: Not enough samples")
        return

    if args.full:
        train_samples = all_samples
        test_samples = []
        print(f"Training on FULL dataset: {len(train_samples)} samples")
    else:
        n_test = max(1, int(len(all_samples) * 0.2))
        random.shuffle(all_samples)
        test_samples = all_samples[:n_test]
        train_samples = all_samples[n_test:]
        print(f"Train: {len(train_samples)}, Held-out: {len(test_samples)}")
        save_heldout_samples(test_samples, out_dir / 'heldout_samples')
        print(f"Saved full held-out sample pack to: {out_dir / 'heldout_samples'}")

    train_ds = FLAMEDataset(train_samples, seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True)

    test_loader = None
    if test_samples:
        test_ds = FLAMEDataset(test_samples, seq_len=args.seq_len)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers)

    model = FLAMEGeneratorV9(hidden_dim=args.hidden_dim, gru_layers=args.gru_layers, dropout=args.dropout).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_metrics = []
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(
                batch['nmf'], batch['emotions'], batch['token_ids'], batch['token_mask'],
                batch['phoneme_feats'], batch['time_feats'],
                batch['blend_curve'], apply_neutral_blend=False
            )
            loss, metrics = compute_loss(out, batch['flame_params'], batch['nmf'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_metrics.append(metrics)

        avg_train = {k: float(np.mean([m[k] for m in train_metrics])) for k in train_metrics[0]}

        if test_loader and epoch % 5 == 0:
            model.eval()
            val_metrics = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    out = model(
                        batch['nmf'], batch['emotions'], batch['token_ids'], batch['token_mask'],
                        batch['phoneme_feats'], batch['time_feats'],
                        batch['blend_curve'], apply_neutral_blend=False
                    )
                    _, metrics = compute_loss(out, batch['flame_params'], batch['nmf'])
                    val_metrics.append(metrics)
            avg_val = {k: float(np.mean([m[k] for m in val_metrics])) for k in val_metrics[0]}
            val_loss = avg_val['loss']
        else:
            val_loss = avg_train['loss']

        scheduler.step()
        print(f"Epoch {epoch:03d}/{args.epochs} | train {avg_train['loss']:.5f} | val {val_loss:.5f} | jaw {avg_train['jaw']:.5f} | vel {avg_train['vel']:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'config': {'hidden_dim': args.hidden_dim, 'gru_layers': args.gru_layers, 'dropout': args.dropout},
                'epoch': epoch,
                'best_loss': best_loss,
            }, out_dir / 'best.pt')

    print(f"Training complete. Best loss: {best_loss:.6f}")


@torch.no_grad()
def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = FLAMEGeneratorV9(
        hidden_dim=cfg.get('hidden_dim', 192),
        gru_layers=cfg.get('gru_layers', 2),
        dropout=cfg.get('dropout', 0.1)
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


@torch.no_grad()
def generate(args):
    device = pick_device(args.force_cpu)
    model = load_model(args, device)

    emotions = {}
    if args.emotion:
        for item in args.emotion.split(','):
            if '=' in item:
                k, v = item.split('=')
                emotions[k.strip()] = float(v)

    nmf_spec = {}
    if args.nmf:
        for item in args.nmf.split(','):
            if '=' in item:
                k, v = item.split('=')
                nmf_spec[k.strip()] = float(v)

    duration = args.duration if args.duration > 0 else text_to_duration(args.text)
    print(f"Text: {args.text!r}")
    print(f"Duration: {duration} frames ({duration/30:.1f}s)")

    nmf_seq = build_nmf_sequence(nmf_spec, duration, use_neutral_blend=not args.no_neutral)
    nmf_t = torch.from_numpy(nmf_seq).unsqueeze(0).to(device)

    emo_vec = np.array([emotions.get(n, 0.0) for n in EMOTION_CHANNELS], dtype=np.float32)
    emo_t = torch.from_numpy(emo_vec).unsqueeze(0).to(device)

    token_ids_np = text_to_ids(args.text)
    token_ids = torch.from_numpy(token_ids_np).unsqueeze(0).to(device)
    token_mask = torch.ones(1, token_ids.shape[1], device=device)
    phoneme_feats = torch.from_numpy(extract_phoneme_features(args.text)).unsqueeze(0).to(device)
    time_feats = torch.from_numpy(temporal_features(duration)).unsqueeze(0).to(device)
    blend_curve = torch.from_numpy(create_neutral_blend_curve(duration)).unsqueeze(0).to(device)

    out = model(
        nmf_t, emo_t, token_ids, token_mask, phoneme_feats, time_feats,
        blend_curve, apply_neutral_blend=not args.no_neutral
    )

    expression = out['expression'][0].cpu().numpy()
    jaw_pose = out['jaw_pose'][0].cpu().numpy()
    eye_pose = out['eye_pose'][0].cpu().numpy()
    global_orient = out['global_orient'][0].cpu().numpy()
    flame_params = np.concatenate([expression, jaw_pose, eye_pose, global_orient], axis=-1)

    out_path = args.output if args.output.endswith('.npz') else args.output + '.npz'
    np.savez(
        out_path,
        flame_params=flame_params,
        expression=expression,
        jaw_pose=jaw_pose,
        eye_pose=eye_pose,
        global_orient=global_orient,
        nmf=nmf_seq,
        emotions=emo_vec,
        text=args.text,
        fps=np.float32(30.0),
    )
    print(f"Saved: {out_path}")
    print(f"Jaw range: [{jaw_pose[:,0].min():.3f}, {jaw_pose[:,0].max():.3f}]")


@torch.no_grad()
def reconstruct(args):
    device = pick_device(args.force_cpu)
    model = load_model(args, device)

    heldout_dir = Path(args.heldout_dir)
    files = sorted([p for p in heldout_dir.glob('*.npz') if p.name != 'manifest.json'])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        d = np.load(path, allow_pickle=True)
        clip_id = str(d['clip_id']) if np.asarray(d['clip_id']).shape == () else str(d['clip_id'].tolist())
        text = str(d['text']) if np.asarray(d['text']).shape == () else str(d['text'].tolist())
        emotions = d['emotions'].astype(np.float32)
        nmf = d['nmf'].astype(np.float32)
        duration = len(nmf)

        token_ids_np = d['token_ids'].astype(np.int64)
        phoneme_np = d['phoneme_feats'].astype(np.float32)

        nmf_t = torch.from_numpy(nmf).unsqueeze(0).to(device)
        emo_t = torch.from_numpy(emotions).unsqueeze(0).to(device)
        token_ids = torch.from_numpy(token_ids_np).unsqueeze(0).to(device)
        token_mask = torch.ones(1, token_ids.shape[1], device=device)
        phoneme_feats = torch.from_numpy(phoneme_np).unsqueeze(0).to(device)
        time_feats = torch.from_numpy(temporal_features(duration)).unsqueeze(0).to(device)
        blend_curve = torch.from_numpy(create_neutral_blend_curve(duration)).unsqueeze(0).to(device)

        out = model(
            nmf_t, emo_t, token_ids, token_mask, phoneme_feats, time_feats,
            blend_curve, apply_neutral_blend=not args.no_neutral
        )
        flame_params = out['flame_params'][0].cpu().numpy()

        out_path = out_dir / f"{clip_id}_recon.npz"
        np.savez(
            out_path,
            flame_params=flame_params,
            expression=out['expression'][0].cpu().numpy(),
            jaw_pose=out['jaw_pose'][0].cpu().numpy(),
            eye_pose=out['eye_pose'][0].cpu().numpy(),
            global_orient=out['global_orient'][0].cpu().numpy(),
            fps=np.float32(30.0),
            text=text,
            emotions=emotions,
            nmf=nmf,
        )
        print(f"Saved: {out_path}")

    print(f"Reconstructed {len(files)} held-out samples to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="FLAME Generator")
    sub = parser.add_subparsers(dest='mode')

    train_p = sub.add_parser('train')
    train_p.add_argument('--flame_dir', required=True)
    train_p.add_argument('--jsonl', required=True)
    train_p.add_argument('--output_dir', default='flame_gen_v9')
    train_p.add_argument('--epochs', type=int, default=150)
    train_p.add_argument('--batch_size', type=int, default=16)
    train_p.add_argument('--lr', type=float, default=3e-4)
    train_p.add_argument('--seq_len', type=int, default=64)
    train_p.add_argument('--hidden_dim', type=int, default=192)
    train_p.add_argument('--gru_layers', type=int, default=2)
    train_p.add_argument('--dropout', type=float, default=0.1)
    train_p.add_argument('--num_workers', type=int, default=2)
    train_p.add_argument('--seed', type=int, default=42)
    train_p.add_argument('--force_cpu', action='store_true')
    train_p.add_argument('--split', action='store_true')
    train_p.add_argument('--full', action='store_true')

    gen_p = sub.add_parser('generate')
    gen_p.add_argument('--checkpoint', required=True)
    gen_p.add_argument('--text', required=True)
    gen_p.add_argument('--emotion', default='')
    gen_p.add_argument('--nmf', default='')
    gen_p.add_argument('--duration', type=int, default=0)
    gen_p.add_argument('--output', default='generated_v9.npz')
    gen_p.add_argument('--force_cpu', action='store_true')
    gen_p.add_argument('--no_neutral', action='store_true')

    rec_p = sub.add_parser('reconstruct')
    rec_p.add_argument('--checkpoint', required=True)
    rec_p.add_argument('--heldout_dir', required=True)
    rec_p.add_argument('--output_dir', default='reconstructions_v9')
    rec_p.add_argument('--force_cpu', action='store_true')
    rec_p.add_argument('--no_neutral', action='store_true')

    args = parser.parse_args()
    if args.mode == 'train':
        if not args.split and not args.full:
            print("Please specify --split or --full")
            return
        train(args)
    elif args.mode == 'generate':
        generate(args)
    elif args.mode == 'reconstruct':
        reconstruct(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
