#!/usr/bin/env python3
"""
USAGE:
# Full grid search
python hyperparam_tuning.py grid --flame_dir flame_params --jsonl merged.jsonl --output_dir hp_grid
    
# Analyse results
python hyperparam_tuning.py analyse --results_dir hp_grid --output_dir hp_analysis
"""

import argparse
import copy
import json
import itertools
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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


@dataclass
class HyperparamConfig:
    """Complete hyperparameter configuration."""
    # Architecture
    hidden_dim: int = 192
    gru_layers: int = 2
    dropout: float = 0.1
    
    # Training
    lr: float = 3e-4
    batch_size: int = 16
    seq_len: int = 64
    epochs: int = 80  # Shorter for HP search
    weight_decay: float = 0.01
    
    # Loss weights
    expr_weight: float = 2.3
    jaw_weight: float = 2.8
    eye_weight: float = 0.8
    orient_weight: float = 0.6
    vel_weight: float = 0.35
    accel_weight: float = 0.20
    jaw_ctrl_weight: float = 0.80
    smile_weight: float = 0.45
    amp_reg_weight: float = 0.02
    
    # Control gains
    nmf_to_mouth_scale: float = 0.35
    nmf_to_jaw_scale: float = 0.25
    emotion_amp_init: float = 0.20
    
    # Derived name
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"hd{self.hidden_dim}_gl{self.gru_layers}_lr{self.lr:.0e}_bs{self.batch_size}"


# Default search spaces
SEARCH_SPACES = {
    'quick': {
        'hidden_dim': [128, 192],
        'gru_layers': [2],
        'lr': [1e-4, 3e-4],
    },
    'architecture': {
        'hidden_dim': [128, 192, 256, 384],
        'gru_layers': [1, 2, 3],
        'dropout': [0.05, 0.1, 0.15, 0.2],
    },
    'training': {
        'lr': [1e-4, 2e-4, 3e-4, 5e-4, 1e-3],
        'batch_size': [8, 16, 32],
        'seq_len': [32, 64, 96, 128],
    },
    'loss_weights': {
        'expr_weight': [1.5, 2.0, 2.3, 3.0],
        'jaw_weight': [2.0, 2.5, 2.8, 3.5],
        'vel_weight': [0.2, 0.35, 0.5, 0.7],
        'accel_weight': [0.1, 0.2, 0.3, 0.4],
    },
    'control_gains': {
        'nmf_to_mouth_scale': [0.2, 0.35, 0.5, 0.7],
        'nmf_to_jaw_scale': [0.15, 0.25, 0.35, 0.5],
        'emotion_amp_init': [0.1, 0.2, 0.3, 0.4],
    },
    'full': {
        'hidden_dim': [128, 192, 256],
        'gru_layers': [1, 2, 3],
        'dropout': [0.1, 0.15],
        'lr': [1e-4, 3e-4, 5e-4],
        'batch_size': [16, 32],
        'vel_weight': [0.2, 0.35, 0.5],
    },
}


@dataclass
class ExperimentResult:
    config: Dict
    config_name: str
    
    # Final metrics
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    
    # Per-component losses
    best_expr_loss: float = 0.0
    best_jaw_loss: float = 0.0
    best_eye_loss: float = 0.0
    best_orient_loss: float = 0.0
    best_vel_loss: float = 0.0
    
    # Training history
    train_history: List[float] = field(default_factory=list)
    val_history: List[float] = field(default_factory=list)
    
    # Timing
    training_time_sec: float = 0.0
    
    # Status
    converged: bool = False
    early_stopped: bool = False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def temporal_features(T: int) -> np.ndarray:
    if T <= 1:
        return np.zeros((T, 4), dtype=np.float32)
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    return np.stack([t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), 2 * t - 1.0], axis=-1).astype(np.float32)


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

    def forward(self, token_ids, token_mask, phoneme_feats):
        emb = self.char_embed(token_ids)
        weights = token_mask.unsqueeze(-1)
        pooled = (emb * weights).sum(1) / token_mask.sum(1, keepdim=True).clamp(min=1.0)
        phon = self.phoneme_proj(phoneme_feats)
        return self.out(torch.cat([pooled, phon], dim=-1))


class EmotionAmplifier(nn.Module):
    def __init__(self, emotion_dim: int = N_EMOTION, hidden_dim: int = 128, amp_init: float = 0.20):
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
        self.amp_scale = nn.Parameter(torch.tensor(amp_init))
        self.register_buffer('mouth_idx', torch.tensor([10, 11, 12, 13, 14, 15, 5, 3]))
        self.register_buffer('cheek_idx', torch.tensor([16, 17, 20, 21]))
        self.register_buffer('nose_idx', torch.tensor([18, 19, 22, 23]))
        self.register_buffer('brow_idx', torch.tensor([30, 31, 32, 33, 34, 35]))

    def forward(self, emotion):
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

    def forward(self, x, cond):
        if x.dim() == 3 and cond.dim() == 2:
            cond = cond.unsqueeze(1)
        return self.scale(cond) * x + self.shift(cond)


class ConfigurableFLAMEGenerator(nn.Module):
    
    def __init__(self, config: HyperparamConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        
        self.text_encoder = TextEncoder(hidden_dim)
        self.nmf_encoder = nn.Sequential(
            nn.Linear(N_NMF + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.emotion_encoder = nn.Sequential(
            nn.Linear(N_EMOTION, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )
        self.emotion_amplifier = EmotionAmplifier(N_EMOTION, 128, config.emotion_amp_init)
        self.film_emo = FiLM(hidden_dim, hidden_dim)
        self.film_text = FiLM(hidden_dim, hidden_dim)
        
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, 
            num_layers=config.gru_layers, 
            batch_first=True,
            bidirectional=True, 
            dropout=config.dropout if config.gru_layers > 1 else 0.0
        )
        self.post = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(config.dropout))

        self.expression_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
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
        
        # Store configurable gains
        self.nmf_to_mouth_scale = config.nmf_to_mouth_scale
        self.nmf_to_jaw_scale = config.nmf_to_jaw_scale

    def _apply_emotion_amplification(self, expression, emo_amp):
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

    def forward(self, nmf, emotion, token_ids, token_mask, phoneme_feats, time_feats,
                blend_curve=None, apply_neutral_blend=False):
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

        # Apply NMF control with configurable gains
        mouth_expr_front = expression[:, :, :16] + self.nmf_to_mouth_scale * self.nmf_to_mouth(nmf)
        expression = torch.cat([mouth_expr_front, expression[:, :, 16:]], dim=-1)

        jaw0 = jaw_pose[:, :, 0:1] + self.nmf_to_jaw_scale * torch.tanh(self.nmf_to_jaw(nmf))
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


def compute_loss_configurable(out, target, nmf, config: HyperparamConfig):
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
        config.expr_weight * expr_loss +
        config.jaw_weight * jaw_loss +
        config.eye_weight * eye_loss +
        config.orient_weight * orient_loss +
        config.vel_weight * vel_loss +
        config.accel_weight * accel_loss +
        config.jaw_ctrl_weight * jaw_ctrl +
        config.smile_weight * smile_loss +
        config.smile_weight * frown_loss +
        config.amp_reg_weight * amp_reg
    )
    
    return total, {
        'loss': float(total.item()),
        'expr': float(expr_loss.item()),
        'jaw': float(jaw_loss.item()),
        'eye': float(eye_loss.item()),
        'orient': float(orient_loss.item()),
        'vel': float(vel_loss.item()),
        'accel': float(accel_loss.item()),
    }

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
        blend = np.ones(len(flame), dtype=np.float32)
        return {
            'flame_params': torch.from_numpy(flame).float(),
            'nmf': torch.from_numpy(nmf).float(),
            'emotions': torch.from_numpy(s.emotions).float(),
            'token_ids': torch.from_numpy(s.token_ids).long(),
            'phoneme_feats': torch.from_numpy(s.phoneme_feats).float(),
            'time_feats': torch.from_numpy(tfeat).float(),
            'blend_curve': torch.from_numpy(blend).float(),
        }


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


def load_samples_from_v9(flame_dir: str, jsonl_path: str) -> List[Sample]:
    try:
        import importlib.util
        gen_path = Path(__file__).parent / 'flame_generator.py'
        if not gen_path.exists():
            gen_path = Path('flame_generator.py')
        
        if gen_path.exists():
            spec = importlib.util.spec_from_file_location("flame_generator", gen_path)
            gen_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen_module)
            return gen_module.load_samples(flame_dir, jsonl_path)
    except Exception as e:
        print(f"Could not import flame_generator: {e}")
    
    # Fallback: minimal implementation
    print("Using fallback sample loading...")
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
            
            jaw = data.get('jaw_pose', np.zeros((len(expr), 3))).astype(np.float32)
            eye = data.get('eye_pose', np.zeros((len(expr), 6))).astype(np.float32)
            orient = data.get('global_orient', np.zeros((len(expr), 3))).astype(np.float32)
            flame_params = np.concatenate([expr, jaw, eye, orient], axis=-1)
            
            T = len(flame_params)
            nmf = np.zeros((T, N_NMF), dtype=np.float32)
            
            # Extract emotions
            e = rec.get('emosign', {})
            emotions = np.array([max(0.0, min(1.0, (e.get(n, 1) - 1) / 3.0)) 
                                for n in EMOTION_CHANNELS], dtype=np.float32)
            
            # Extract text
            text = ""
            xml = rec.get('xml_utterance', {})
            for child in xml.get('children', []):
                if child.get('tag') == 'TRANSLATION':
                    text = child.get('text', '').strip("'")
                    break
            
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
            continue
    
    print(f"Loaded {len(samples)} samples")
    return samples

def train_single_config(config: HyperparamConfig, train_samples: List[Sample], val_samples: List[Sample], device: torch.device, early_stop_patience: int = 15, verbose: bool = True) -> ExperimentResult:
    
    result = ExperimentResult(
        config=asdict(config),
        config_name=config.name,
    )
    
    start_time = time.time()
    
    # Create datasets
    train_ds = FLAMEDataset(train_samples, seq_len=config.seq_len)
    val_ds = FLAMEDataset(val_samples, seq_len=config.seq_len)
    
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Create model
    model = ConfigurableFLAMEGenerator(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    best_val_loss = float('inf')
    best_metrics = {}
    patience_counter = 0
    
    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            out = model(
                batch['nmf'], batch['emotions'], batch['token_ids'], batch['token_mask'],
                batch['phoneme_feats'], batch['time_feats'], batch['blend_curve'], 
                apply_neutral_blend=False
            )
            
            loss, metrics = compute_loss_configurable(out, batch['flame_params'], batch['nmf'], config)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(metrics['loss'])
        
        avg_train_loss = np.mean(train_losses)
        result.train_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        val_metrics_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                out = model(
                    batch['nmf'], batch['emotions'], batch['token_ids'], batch['token_mask'],
                    batch['phoneme_feats'], batch['time_feats'], batch['blend_curve'],
                    apply_neutral_blend=False
                )
                
                _, metrics = compute_loss_configurable(out, batch['flame_params'], batch['nmf'], config)
                val_losses.append(metrics['loss'])
                val_metrics_list.append(metrics)
        
        avg_val_loss = np.mean(val_losses)
        result.val_history.append(avg_val_loss)
        
        scheduler.step()
        
        # Check for best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = {k: np.mean([m[k] for m in val_metrics_list]) for k in val_metrics_list[0]}
            result.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: train={avg_train_loss:.5f}, val={avg_val_loss:.5f}, best={best_val_loss:.5f}")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            result.early_stopped = True
            if verbose:
                print(f"  Early stopped at epoch {epoch}")
            break
    
    # Store results
    result.best_val_loss = best_val_loss
    result.final_train_loss = result.train_history[-1]
    result.final_val_loss = result.val_history[-1]
    result.best_expr_loss = best_metrics.get('expr', 0)
    result.best_jaw_loss = best_metrics.get('jaw', 0)
    result.best_eye_loss = best_metrics.get('eye', 0)
    result.best_orient_loss = best_metrics.get('orient', 0)
    result.best_vel_loss = best_metrics.get('vel', 0)
    result.training_time_sec = time.time() - start_time
    result.converged = not result.early_stopped or result.best_epoch > config.epochs // 2
    
    return result

def generate_grid_configs(search_space: Dict[str, List], base_config: HyperparamConfig = None) -> List[HyperparamConfig]:
    base = base_config or HyperparamConfig()
    
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    configs = []
    for combo in itertools.product(*values):
        cfg = copy.copy(base)
        for key, val in zip(keys, combo):
            setattr(cfg, key, val)
        cfg.name = "_".join([f"{k[:3]}{v}" for k, v in zip(keys, combo)])
        configs.append(cfg)
    
    return configs


def generate_random_configs(search_space: Dict[str, List], n_trials: int, base_config: HyperparamConfig = None) -> List[HyperparamConfig]:
    base = base_config or HyperparamConfig()
    
    configs = []
    for i in range(n_trials):
        cfg = copy.copy(base)
        name_parts = []
        for key, values in search_space.items():
            val = random.choice(values)
            setattr(cfg, key, val)
            name_parts.append(f"{key[:3]}{val}")
        cfg.name = f"trial{i:03d}_" + "_".join(name_parts[:3])
        configs.append(cfg)
    
    return configs


def run_hyperparameter_search(configs: List[HyperparamConfig], samples: List[Sample], output_dir: Path, n_folds: int = 3, seed: int = 42, verbose: bool = True) -> List[ExperimentResult]:

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Testing {len(configs)} configurations with {n_folds}-fold CV")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create folds
    indices = list(range(len(samples)))
    random.shuffle(indices)
    fold_size = len(indices) // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    
    all_results = []
    
    for cfg_idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {cfg_idx+1}/{len(configs)}: {config.name}")
        print(f"{'='*60}")
        
        fold_results = []
        
        for fold_idx in range(n_folds):
            if verbose:
                print(f"\n  Fold {fold_idx+1}/{n_folds}")
            
            # Split
            val_indices = folds[fold_idx]
            train_indices = [i for i in indices if i not in val_indices]
            
            train_samples = [samples[i] for i in train_indices]
            val_samples = [samples[i] for i in val_indices]
            
            # Train
            result = train_single_config(
                config, train_samples, val_samples, device,
                early_stop_patience=15, verbose=verbose
            )
            fold_results.append(result)
        
        # Average across folds
        avg_result = ExperimentResult(
            config=asdict(config),
            config_name=config.name,
            best_val_loss=np.mean([r.best_val_loss for r in fold_results]),
            best_epoch=int(np.mean([r.best_epoch for r in fold_results])),
            final_train_loss=np.mean([r.final_train_loss for r in fold_results]),
            final_val_loss=np.mean([r.final_val_loss for r in fold_results]),
            best_expr_loss=np.mean([r.best_expr_loss for r in fold_results]),
            best_jaw_loss=np.mean([r.best_jaw_loss for r in fold_results]),
            best_eye_loss=np.mean([r.best_eye_loss for r in fold_results]),
            best_orient_loss=np.mean([r.best_orient_loss for r in fold_results]),
            best_vel_loss=np.mean([r.best_vel_loss for r in fold_results]),
            training_time_sec=np.sum([r.training_time_sec for r in fold_results]),
            converged=all(r.converged for r in fold_results),
        )
        
        # Store individual fold histories for analysis
        avg_result.train_history = [np.mean([f.train_history[i] if i < len(f.train_history) else f.train_history[-1] 
                                             for f in fold_results]) 
                                    for i in range(max(len(f.train_history) for f in fold_results))]
        avg_result.val_history = [np.mean([f.val_history[i] if i < len(f.val_history) else f.val_history[-1]
                                          for f in fold_results])
                                 for i in range(max(len(f.val_history) for f in fold_results))]
        
        all_results.append(avg_result)
        
        print(f"\n  Result: val_loss={avg_result.best_val_loss:.5f}, "
              f"epoch={avg_result.best_epoch}, time={avg_result.training_time_sec:.1f}s")
        
        # Save intermediate results
        save_results(all_results, output_dir / 'results_intermediate.json')
    
    # Sort by validation loss
    all_results.sort(key=lambda r: r.best_val_loss)
    
    # Save final results
    save_results(all_results, output_dir / 'results_final.json')
    
    return all_results


def save_results(results: List[ExperimentResult], path: Path):
    data = []
    for r in results:
        d = {
            'config': r.config,
            'config_name': r.config_name,
            'best_val_loss': r.best_val_loss,
            'best_epoch': r.best_epoch,
            'final_train_loss': r.final_train_loss,
            'final_val_loss': r.final_val_loss,
            'best_expr_loss': r.best_expr_loss,
            'best_jaw_loss': r.best_jaw_loss,
            'best_eye_loss': r.best_eye_loss,
            'best_orient_loss': r.best_orient_loss,
            'best_vel_loss': r.best_vel_loss,
            'training_time_sec': r.training_time_sec,
            'converged': r.converged,
            'early_stopped': r.early_stopped,
            'train_history': r.train_history,
            'val_history': r.val_history,
        }
        data.append(d)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def load_results(path: Path) -> List[Dict]:
    """Load results from JSON."""
    with open(path) as f:
        return json.load(f)

def plot_hyperparameter_results(results: List[Dict], output_dir: Path):
    
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    
    # Extract data
    configs = [r['config'] for r in results]
    val_losses = [r['best_val_loss'] for r in results]
    names = [r['config_name'] for r in results]
    
    # Figure 1: Overall ranking
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n_show = min(20, len(results))
    y_pos = np.arange(n_show)
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_show))
    ax.barh(y_pos, val_losses[:n_show], color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n[:30] for n in names[:n_show]], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Validation Loss')
    ax.set_title(f'Top {n_show} Configurations (Lower = Better)')
    
    # Add value labels
    for i, v in enumerate(val_losses[:n_show]):
        ax.text(v + 0.0001, i, f'{v:.5f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_ranking.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_ranking.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 2: Parameter impact analysis
    param_names = ['hidden_dim', 'gru_layers', 'dropout', 'lr', 'batch_size', 
                   'seq_len', 'vel_weight', 'expr_weight']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        if i >= len(axes):
            break
        
        ax = axes[i]
        param_values = [c.get(param, None) for c in configs]
        
        if param_values[0] is None:
            ax.set_visible(False)
            continue
        
        # Group by parameter value
        unique_vals = sorted(set(param_values))
        grouped_losses = {v: [] for v in unique_vals}
        
        for pv, vl in zip(param_values, val_losses):
            grouped_losses[pv].append(vl)
        
        means = [np.mean(grouped_losses[v]) for v in unique_vals]
        stds = [np.std(grouped_losses[v]) for v in unique_vals]
        
        x = np.arange(len(unique_vals))
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in unique_vals], rotation=45, ha='right')
        ax.set_ylabel('Val Loss')
        ax.set_title(param)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_parameter_impact.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_parameter_impact.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 3: Training curves for top configs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_curves = min(5, len(results))
    colors = plt.cm.tab10(np.linspace(0, 1, n_curves))
    
    for i in range(n_curves):
        train_hist = results[i].get('train_history', [])
        val_hist = results[i].get('val_history', [])
        name = results[i]['config_name'][:20]
        
        if train_hist:
            axes[0].plot(train_hist, color=colors[i], label=name, alpha=0.8)
        if val_hist:
            axes[1].plot(val_hist, color=colors[i], label=name, alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Curves (Top 5)')
    axes[0].legend(fontsize=8)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Curves (Top 5)')
    axes[1].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_training_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_training_curves.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 4: Component-wise loss breakdown
    fig, ax = plt.subplots(figsize=(12, 6))
    
    components = ['expr', 'jaw', 'eye', 'orient', 'vel']
    comp_labels = ['Expression', 'Jaw', 'Eye', 'Orientation', 'Velocity']
    
    n_configs = min(10, len(results))
    x = np.arange(len(components))
    width = 0.08
    
    for i in range(n_configs):
        r = results[i]
        values = [
            r.get('best_expr_loss', 0),
            r.get('best_jaw_loss', 0),
            r.get('best_eye_loss', 0),
            r.get('best_orient_loss', 0),
            r.get('best_vel_loss', 0),
        ]
        offset = (i - n_configs/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=r['config_name'][:15], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel('Loss')
    ax.set_title('Component-wise Loss Breakdown')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_component_breakdown.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_component_breakdown.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 5: Convergence analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = [r.get('best_epoch', 0) for r in results]
    times = [r.get('training_time_sec', 0) for r in results]
    
    axes[0].scatter(epochs, val_losses, alpha=0.6, edgecolor='black')
    axes[0].set_xlabel('Best Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Convergence Speed vs Performance')
    
    axes[1].scatter(times, val_losses, alpha=0.6, edgecolor='black', color='orange')
    axes[1].set_xlabel('Training Time (s)')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Training Time vs Performance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_convergence.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_convergence.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved figures to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="FLAME Generator Hyperparameter Tuning")
    subparsers = parser.add_subparsers(dest='command')
    
    # Quick search
    quick_p = subparsers.add_parser('quick', help='Quick test with few configs')
    quick_p.add_argument('--flame_dir', required=True)
    quick_p.add_argument('--jsonl', required=True)
    quick_p.add_argument('--output_dir', default='hp_quick')
    quick_p.add_argument('--n_folds', type=int, default=2)
    quick_p.add_argument('--seed', type=int, default=42)
    
    # Grid search
    grid_p = subparsers.add_parser('grid', help='Full grid search')
    grid_p.add_argument('--flame_dir', required=True)
    grid_p.add_argument('--jsonl', required=True)
    grid_p.add_argument('--output_dir', default='hp_grid')
    grid_p.add_argument('--space', default='full', choices=list(SEARCH_SPACES.keys()))
    grid_p.add_argument('--n_folds', type=int, default=3)
    grid_p.add_argument('--seed', type=int, default=42)
    
    # Random search
    random_p = subparsers.add_parser('random', help='Random search')
    random_p.add_argument('--flame_dir', required=True)
    random_p.add_argument('--jsonl', required=True)
    random_p.add_argument('--output_dir', default='hp_random')
    random_p.add_argument('--space', default='full', choices=list(SEARCH_SPACES.keys()))
    random_p.add_argument('--n_trials', type=int, default=30)
    random_p.add_argument('--n_folds', type=int, default=3)
    random_p.add_argument('--seed', type=int, default=42)
    
    # Analyse existing results
    analyse_p = subparsers.add_parser('analyse', help='Analyse results')
    analyse_p.add_argument('--results_dir', required=True)
    analyse_p.add_argument('--output_dir', default='hp_analysis')
    
    args = parser.parse_args()
    
    if args.command in ['quick', 'grid', 'random']:
        # Load data
        samples = load_samples_from_v9(args.flame_dir, args.jsonl)
        if len(samples) < 20:
            print("ERROR: Not enough samples")
            return
        
        output_dir = Path(args.output_dir)
        
        # Generate configs
        if args.command == 'quick':
            search_space = SEARCH_SPACES['quick']
            configs = generate_grid_configs(search_space)
        elif args.command == 'grid':
            search_space = SEARCH_SPACES[args.space]
            configs = generate_grid_configs(search_space)
        else:  # random
            search_space = SEARCH_SPACES[args.space]
            configs = generate_random_configs(search_space, args.n_trials)
        
        print(f"Generated {len(configs)} configurations")
        
        # Run search
        results = run_hyperparameter_search(
            configs, samples, output_dir,
            n_folds=args.n_folds, seed=args.seed
        )
        
        # Generate visualizations
        if HAS_MATPLOTLIB:
            plot_hyperparameter_results([asdict(r) if hasattr(r, '__dataclass_fields__') else r 
                                        for r in results], output_dir / 'figures')
        
        # Generate reports
        results_data = load_results(output_dir / 'results_final.json')
        print(f"\n{'='*60}")
        print("HYPERPARAMETER SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"Best config: {results[0].config_name}")
        print(f"Best val loss: {results[0].best_val_loss:.5f}")
        print(f"Results saved to: {output_dir}")
    
    elif args.command == 'analyse':
        results_dir = Path(args.results_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        results_path = results_dir / 'results_final.json'
        if not results_path.exists():
            results_path = results_dir / 'results_intermediate.json'
        
        results = load_results(results_path)
        print(f"Loaded {len(results)} results")
        
        # Generate visualizations
        if HAS_MATPLOTLIB:
            plot_hyperparameter_results(results, output_dir / 'figures')
        
        print(f"Analysis saved to: {output_dir}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
