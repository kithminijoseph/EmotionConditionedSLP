#!/usr/bin/env python3
"""
USAGE:
    # Step 1: Train ablation models
    python ablation_comparison.py train_ablation --flame_dir flame_params --jsonl merged_records.jsonl --output_dir ablation_study
    
    # Step 2: Evaluate all conditions
    python ablation_comparison.py evaluate --ablation_dir ablation_study --output_dir results
    
    # Step 3: Generate dissertation figures
    python ablation_comparison.py figures --results_dir results --output_dir dissertation_figures
"""

import argparse
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

warnings.filterwarnings('ignore')
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from scipy import stats
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

N_EXPRESSION = 100
N_JAW = 3
N_EYE = 6
N_ORIENT = 3
N_FLAME = N_EXPRESSION + N_JAW + N_EYE + N_ORIENT  # 112

NMF_CHANNELS = [
    "head_turn", "head_tilt_side", "head_tilt_fb", "head_nod", "head_shake", "head_jut",
    "brow_raise", "brow_furrow", "eye_aperture", "eye_gaze_h", "eye_gaze_v",
    "mouth_open", "mouth_spread", "lip_pucker", "tongue", "nose_wrinkle", "cheek_puff",
]
N_NMF = len(NMF_CHANNELS)

EMOTION_CHANNELS = [
    "joy", "excited", "surprise_pos", "surprise_neg", "worry",
    "sadness", "fear", "disgust", "frustration", "anger"
]
N_EMOTION = len(EMOTION_CHANNELS)

# Parameter slices
PARAM_SLICES = {
    'expression': (0, 100),
    'jaw_pose': (100, 103),
    'eye_pose': (103, 109),
    'global_orient': (109, 112),
}

# Expression region indices
EXPRESSION_REGIONS = {
    'mouth_open': [0, 1, 2, 3, 4],           # Jaw/mouth opening
    'smile_frown': [10, 11, 12, 13],          # Smile left/right, frown left/right
    'lips': [5, 6, 7, 8, 9, 14, 15],          # Lip shapes
    'brows': [30, 31, 32, 33, 34, 35],        # Eyebrow movements
    'eyes_expr': [40, 41, 42, 43, 44, 45],    # Eye expressions (squint, wide)
    'cheeks': [16, 17, 20, 21],               # Cheek movements
    'nose': [18, 19, 22, 23],                 # Nose wrinkle etc
}

# Ablation conditions
ABLATION_CONDITIONS = {
    'text_only': {'use_emotion': False, 'use_nmf': False},
    'text_emotion': {'use_emotion': True, 'use_nmf': False},
    'text_nmf': {'use_emotion': False, 'use_nmf': True},
    'full_model': {'use_emotion': True, 'use_nmf': True},
}

CONDITION_LABELS = {
    'text_only': 'Text Only',
    'text_emotion': 'Text + Emotion',
    'text_nmf': 'Text + NMF',
    'full_model': 'Full Model\n(Text + NMF + Emotion)',
}

CONDITION_COLORS = {
    'text_only': '#1f77b4',
    'text_emotion': '#ff7f0e', 
    'text_nmf': '#2ca02c',
    'full_model': '#d62728',
}

@dataclass
class ReconstructionMetrics:
    # Overall
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    max_error: float = 0.0
    correlation: float = 0.0
    
    # Per-component MSE
    mse_expression: float = 0.0
    mse_jaw_pose: float = 0.0
    mse_eye_pose: float = 0.0
    mse_global_orient: float = 0.0
    
    # Per-component MAE
    mae_expression: float = 0.0
    mae_jaw_pose: float = 0.0
    mae_eye_pose: float = 0.0
    mae_global_orient: float = 0.0
    
    # Per-component correlation
    corr_expression: float = 0.0
    corr_jaw_pose: float = 0.0
    corr_eye_pose: float = 0.0
    corr_global_orient: float = 0.0
    
    # Per-region expression MSE
    mse_mouth_open: float = 0.0
    mse_smile_frown: float = 0.0
    mse_lips: float = 0.0
    mse_brows: float = 0.0
    mse_eyes_expr: float = 0.0
    mse_cheeks: float = 0.0
    mse_nose: float = 0.0


@dataclass
class TemporalMetrics:
    # Velocity (1st derivative)
    velocity_mse: float = 0.0
    velocity_mae: float = 0.0
    velocity_corr: float = 0.0
    
    # Acceleration (2nd derivative)
    accel_mse: float = 0.0
    accel_mae: float = 0.0
    
    # Jerk (3rd derivative) - smoothness
    jerk_mean_pred: float = 0.0
    jerk_mean_target: float = 0.0
    jerk_peak_pred: float = 0.0
    jerk_peak_target: float = 0.0
    jerk_auc_pred: float = 0.0      # Area under jerk curve
    jerk_auc_target: float = 0.0
    
    # Smoothness ratio (pred/target, 1.0 = same smoothness)
    smoothness_ratio: float = 0.0
    
    # Per-component velocity MSE
    vel_mse_expression: float = 0.0
    vel_mse_jaw: float = 0.0
    vel_mse_eye: float = 0.0
    vel_mse_orient: float = 0.0


@dataclass
class DistributionMetrics:
    # Expression statistics
    expr_mean_range: float = 0.0        # Mean per-dimension range
    expr_std: float = 0.0               # Mean per-dimension std
    expr_dynamic_range: float = 0.0     # Overall min-max range
    
    # Jaw statistics  
    jaw_mean_range: float = 0.0
    jaw_std: float = 0.0
    jaw_dynamic_range: float = 0.0
    
    # Distribution divergence (pred vs target)
    expr_kl_divergence: float = 0.0
    jaw_kl_divergence: float = 0.0


@dataclass
class ControlMetrics:
    # NMF control accuracy
    nmf_mouth_open_corr: float = 0.0    # Correlation between nmf mouth_open and jaw
    nmf_mouth_spread_corr: float = 0.0  # Correlation between nmf mouth_spread and smile
    nmf_brow_corr: float = 0.0          # Correlation between nmf brow and brow expressions
    nmf_head_turn_corr: float = 0.0     # Correlation between nmf head_turn and orient
    
    # Emotion responsiveness
    emotion_expr_variance: float = 0.0  # How much expression varies with emotion


@dataclass
class SampleResult:
    clip_id: str
    duration: int
    condition: str = ''
    reconstruction: ReconstructionMetrics = field(default_factory=ReconstructionMetrics)
    temporal: TemporalMetrics = field(default_factory=TemporalMetrics)
    distribution: DistributionMetrics = field(default_factory=DistributionMetrics)
    control: ControlMetrics = field(default_factory=ControlMetrics)

def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x_flat = x.flatten()
    y_flat = y.flatten()
    if len(x_flat) < 2 or np.std(x_flat) < 1e-8 or np.std(y_flat) < 1e-8:
        return 0.0
    return float(np.corrcoef(x_flat, y_flat)[0, 1])


def compute_reconstruction_metrics(pred: np.ndarray, target: np.ndarray) -> ReconstructionMetrics:
    metrics = ReconstructionMetrics()
    
    # Align lengths
    T = min(len(pred), len(target))
    if T == 0:
        return metrics
    pred = pred[:T].astype(np.float64)
    target = target[:T].astype(np.float64)
    
    # Overall metrics
    diff = pred - target
    metrics.mse = float(np.mean(diff ** 2))
    metrics.mae = float(np.mean(np.abs(diff)))
    metrics.rmse = float(np.sqrt(metrics.mse))
    metrics.max_error = float(np.max(np.abs(diff)))
    metrics.correlation = safe_corrcoef(pred, target)
    
    # Per-component metrics
    for comp_name, (start, end) in PARAM_SLICES.items():
        p = pred[:, start:end]
        t = target[:, start:end]
        
        mse = float(np.mean((p - t) ** 2))
        mae = float(np.mean(np.abs(p - t)))
        corr = safe_corrcoef(p, t)
        
        setattr(metrics, f'mse_{comp_name}', mse)
        setattr(metrics, f'mae_{comp_name}', mae)
        setattr(metrics, f'corr_{comp_name}', corr)
    
    # Per-region expression metrics
    expr_pred = pred[:, :N_EXPRESSION]
    expr_target = target[:, :N_EXPRESSION]
    
    for region_name, indices in EXPRESSION_REGIONS.items():
        valid_idx = [i for i in indices if i < N_EXPRESSION]
        if valid_idx:
            p = expr_pred[:, valid_idx]
            t = expr_target[:, valid_idx]
            mse = float(np.mean((p - t) ** 2))
            setattr(metrics, f'mse_{region_name}', mse)
    
    return metrics


def compute_temporal_metrics(pred: np.ndarray, target: np.ndarray, fps: float = 30.0) -> TemporalMetrics:
    metrics = TemporalMetrics()
    
    T = min(len(pred), len(target))
    if T < 4:
        return metrics
    
    pred = pred[:T].astype(np.float64)
    target = target[:T].astype(np.float64)
    dt = 1.0 / fps
    
    # Velocity (1st derivative)
    vel_pred = np.diff(pred, axis=0) / dt
    vel_target = np.diff(target, axis=0) / dt
    
    metrics.velocity_mse = float(np.mean((vel_pred - vel_target) ** 2))
    metrics.velocity_mae = float(np.mean(np.abs(vel_pred - vel_target)))
    metrics.velocity_corr = safe_corrcoef(vel_pred, vel_target)
    
    # Per-component velocity MSE
    for comp_name, (start, end) in PARAM_SLICES.items():
        vp = vel_pred[:, start:end]
        vt = vel_target[:, start:end]
        mse = float(np.mean((vp - vt) ** 2))
        # Map component names
        short_name = comp_name.replace('_pose', '').replace('global_', '')
        setattr(metrics, f'vel_mse_{short_name}', mse)
    
    # Acceleration (2nd derivative)
    if len(vel_pred) > 1:
        accel_pred = np.diff(vel_pred, axis=0) / dt
        accel_target = np.diff(vel_target, axis=0) / dt
        
        metrics.accel_mse = float(np.mean((accel_pred - accel_target) ** 2))
        metrics.accel_mae = float(np.mean(np.abs(accel_pred - accel_target)))
        
        # Jerk (3rd derivative)
        if len(accel_pred) > 1:
            jerk_pred = np.diff(accel_pred, axis=0) / dt
            jerk_target = np.diff(accel_target, axis=0) / dt
            
            # Jerk magnitude per frame (L2 norm across dimensions)
            jerk_mag_pred = np.sqrt(np.sum(jerk_pred ** 2, axis=1))
            jerk_mag_target = np.sqrt(np.sum(jerk_target ** 2, axis=1))
            
            metrics.jerk_mean_pred = float(np.mean(jerk_mag_pred))
            metrics.jerk_mean_target = float(np.mean(jerk_mag_target))
            metrics.jerk_peak_pred = float(np.max(jerk_mag_pred))
            metrics.jerk_peak_target = float(np.max(jerk_mag_target))
            metrics.jerk_auc_pred = float(np.sum(jerk_mag_pred) * dt)
            metrics.jerk_auc_target = float(np.sum(jerk_mag_target) * dt)
            
            # Smoothness ratio
            if metrics.jerk_mean_target > 1e-8:
                metrics.smoothness_ratio = metrics.jerk_mean_pred / metrics.jerk_mean_target
    
    return metrics


def compute_distribution_metrics(pred: np.ndarray, target: np.ndarray) -> DistributionMetrics:
    metrics = DistributionMetrics()
    
    T = min(len(pred), len(target))
    if T == 0:
        return metrics
    
    pred = pred[:T]
    target = target[:T]
    
    # Expression statistics (pred)
    expr_pred = pred[:, :N_EXPRESSION]
    metrics.expr_mean_range = float(np.mean(np.ptp(expr_pred, axis=0)))
    metrics.expr_std = float(np.mean(np.std(expr_pred, axis=0)))
    metrics.expr_dynamic_range = float(np.ptp(expr_pred))
    
    # Jaw statistics (pred)
    jaw_pred = pred[:, 100:103]
    metrics.jaw_mean_range = float(np.mean(np.ptp(jaw_pred, axis=0)))
    metrics.jaw_std = float(np.mean(np.std(jaw_pred, axis=0)))
    metrics.jaw_dynamic_range = float(np.ptp(jaw_pred))
    
    # KL divergence approximation via histogram comparison
    if HAS_SCIPY:
        try:
            # Expression KL divergence
            expr_target = target[:, :N_EXPRESSION]
            hist_pred, bins = np.histogram(expr_pred.flatten(), bins=50, density=True)
            hist_target, _ = np.histogram(expr_target.flatten(), bins=bins, density=True)
            # Add small epsilon to avoid log(0)
            hist_pred = hist_pred + 1e-10
            hist_target = hist_target + 1e-10
            metrics.expr_kl_divergence = float(stats.entropy(hist_pred, hist_target))
            
            # Jaw KL divergence  
            jaw_target = target[:, 100:103]
            hist_pred_j, bins_j = np.histogram(jaw_pred.flatten(), bins=20, density=True)
            hist_target_j, _ = np.histogram(jaw_target.flatten(), bins=bins_j, density=True)
            hist_pred_j = hist_pred_j + 1e-10
            hist_target_j = hist_target_j + 1e-10
            metrics.jaw_kl_divergence = float(stats.entropy(hist_pred_j, hist_target_j))
        except:
            pass
    
    return metrics


def compute_control_metrics(pred: np.ndarray, nmf: np.ndarray, emotions: np.ndarray) -> ControlMetrics:
    metrics = ControlMetrics()
    
    T = min(len(pred), len(nmf))
    if T < 2:
        return metrics
    
    pred = pred[:T]
    nmf = nmf[:T]
    
    # NMF indices
    NMF_IDX = {n: i for i, n in enumerate(NMF_CHANNELS)}
    
    # Mouth open -> Jaw correlation
    if 'mouth_open' in NMF_IDX:
        nmf_mouth = nmf[:, NMF_IDX['mouth_open']]
        jaw_open = pred[:, 100]  # First jaw component
        metrics.nmf_mouth_open_corr = safe_corrcoef(nmf_mouth, jaw_open)
    
    # Mouth spread -> Smile correlation
    if 'mouth_spread' in NMF_IDX:
        nmf_spread = nmf[:, NMF_IDX['mouth_spread']]
        smile = 0.5 * (pred[:, 10] + pred[:, 11])  # Avg of smile L/R
        metrics.nmf_mouth_spread_corr = safe_corrcoef(nmf_spread, smile)
    
    # Brow raise -> Brow expression correlation
    if 'brow_raise' in NMF_IDX:
        nmf_brow = nmf[:, NMF_IDX['brow_raise']]
        brow_expr = np.mean(pred[:, 30:36], axis=1)  # Avg brow expressions
        metrics.nmf_brow_corr = safe_corrcoef(nmf_brow, brow_expr)
    
    # Head turn -> Global orient correlation
    if 'head_turn' in NMF_IDX:
        nmf_turn = nmf[:, NMF_IDX['head_turn']]
        head_yaw = pred[:, 110]  # Y-axis rotation
        metrics.nmf_head_turn_corr = safe_corrcoef(nmf_turn, head_yaw)
    
    # Emotion responsiveness (variance of expression explained by emotion)
    if len(emotions) == N_EMOTION:
        # Simple metric: how much the dominant emotion affects expression variance
        dominant_emo = np.argmax(emotions)
        metrics.emotion_expr_variance = float(np.var(pred[:, :N_EXPRESSION]))
    
    return metrics


def evaluate_single_sample(pred_path: Path, target_path: Path, condition: str = '', fps: float = 30.0) -> SampleResult:
    
    pred_data = np.load(pred_path, allow_pickle=True)
    target_data = np.load(target_path, allow_pickle=True)
    
    pred = pred_data['flame_params'].astype(np.float32)
    target = target_data['flame_params'].astype(np.float32)
    
    # Get clip ID
    clip_id = pred_path.stem.replace('_recon', '')
    
    # Get NMF and emotions if available
    nmf = pred_data.get('nmf', np.zeros((len(pred), N_NMF)))
    if isinstance(nmf, np.ndarray) and len(nmf) > 0:
        nmf = nmf.astype(np.float32)
    else:
        nmf = np.zeros((len(pred), N_NMF), dtype=np.float32)
    
    emotions = pred_data.get('emotions', np.zeros(N_EMOTION))
    if isinstance(emotions, np.ndarray):
        emotions = emotions.astype(np.float32)
    else:
        emotions = np.zeros(N_EMOTION, dtype=np.float32)
    
    # Compute all metrics
    result = SampleResult(
        clip_id=clip_id,
        duration=min(len(pred), len(target)),
        condition=condition,
        reconstruction=compute_reconstruction_metrics(pred, target),
        temporal=compute_temporal_metrics(pred, target, fps),
        distribution=compute_distribution_metrics(pred, target),
        control=compute_control_metrics(pred, nmf, emotions),
    )
    
    return result

def aggregate_results(results: List[SampleResult]) -> Dict[str, Any]:
    if not results:
        return {}
    
    agg = {
        'n_samples': len(results),
        'total_frames': sum(r.duration for r in results),
        'mean_duration': float(np.mean([r.duration for r in results])),
    }
    
    # Helper to aggregate a list of values
    def agg_values(values, prefix):
        values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
        if not values:
            return {}
        return {
            f'{prefix}_mean': float(np.mean(values)),
            f'{prefix}_std': float(np.std(values)),
            f'{prefix}_median': float(np.median(values)),
            f'{prefix}_min': float(np.min(values)),
            f'{prefix}_max': float(np.max(values)),
        }
    
    # Aggregate reconstruction metrics
    for field_name in ['mse', 'mae', 'rmse', 'correlation',
                       'mse_expression', 'mse_jaw_pose', 'mse_eye_pose', 'mse_global_orient',
                       'mae_expression', 'mae_jaw_pose', 'mae_eye_pose', 'mae_global_orient',
                       'corr_expression', 'corr_jaw_pose', 'corr_eye_pose', 'corr_global_orient',
                       'mse_mouth_open', 'mse_smile_frown', 'mse_lips', 'mse_brows', 
                       'mse_eyes_expr', 'mse_cheeks', 'mse_nose']:
        values = [getattr(r.reconstruction, field_name, 0) for r in results]
        agg.update(agg_values(values, f'recon_{field_name}'))
    
    # Aggregate temporal metrics
    for field_name in ['velocity_mse', 'velocity_mae', 'velocity_corr',
                       'accel_mse', 'accel_mae',
                       'jerk_mean_pred', 'jerk_peak_pred', 'jerk_auc_pred',
                       'smoothness_ratio',
                       'vel_mse_expression', 'vel_mse_jaw', 'vel_mse_eye', 'vel_mse_orient']:
        values = [getattr(r.temporal, field_name, 0) for r in results]
        agg.update(agg_values(values, f'temporal_{field_name}'))
    
    # Aggregate distribution metrics
    for field_name in ['expr_mean_range', 'expr_std', 'expr_dynamic_range',
                       'jaw_mean_range', 'jaw_std', 'jaw_dynamic_range',
                       'expr_kl_divergence', 'jaw_kl_divergence']:
        values = [getattr(r.distribution, field_name, 0) for r in results]
        agg.update(agg_values(values, f'dist_{field_name}'))
    
    # Aggregate control metrics
    for field_name in ['nmf_mouth_open_corr', 'nmf_mouth_spread_corr', 
                       'nmf_brow_corr', 'nmf_head_turn_corr', 'emotion_expr_variance']:
        values = [getattr(r.control, field_name, 0) for r in results]
        agg.update(agg_values(values, f'ctrl_{field_name}'))
    
    return agg

def create_ablation_generator(condition: str, base_config: Dict):
    cfg = ABLATION_CONDITIONS[condition]
    
    # This returns a wrapper that modifies inputs during forward pass
    class AblationWrapper:
        def __init__(self, use_emotion: bool, use_nmf: bool):
            self.use_emotion = use_emotion
            self.use_nmf = use_nmf
        
        def modify_batch(self, batch: Dict) -> Dict:
            modified = batch.copy()
            if not self.use_nmf:
                # Zero out NMF
                modified['nmf'] = torch.zeros_like(batch['nmf'])
            if not self.use_emotion:
                # Zero out emotions
                modified['emotions'] = torch.zeros_like(batch['emotions'])
            return modified
    
    return AblationWrapper(cfg['use_emotion'], cfg['use_nmf'])


def train_ablation_models(flame_dir: str, jsonl_path: str, output_dir: str, epochs: int = 150, seed: int = 42, conditions: List[str] = None):
    if not HAS_TORCH:
        print("ERROR: PyTorch required for training")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conditions = conditions or list(ABLATION_CONDITIONS.keys())
    
    # Import generator module
    try:
        import importlib.util
        gen_path = Path(__file__).parent / 'flame_generator_v9.py'
        if not gen_path.exists():
            gen_path = Path('flame_generator_v9.py')
        
        spec = importlib.util.spec_from_file_location("flame_generator_v9", gen_path)
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
    except Exception as e:
        print(f"ERROR: Could not import flame_generator_v9.py: {e}")
        return
    
    # Load samples once
    all_samples = gen_module.load_samples(flame_dir, jsonl_path)
    if len(all_samples) < 10:
        print("ERROR: Not enough samples")
        return
    
    # Create fixed train/test split
    random.seed(seed)
    n_test = max(1, int(len(all_samples) * 0.2))
    shuffled = all_samples.copy()
    random.shuffle(shuffled)
    test_samples = shuffled[:n_test]
    train_samples = shuffled[n_test:]
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
    
    # Save held-out samples (same for all conditions)
    heldout_dir = output_dir / 'heldout_samples'
    gen_module.save_heldout_samples(test_samples, heldout_dir)
    print(f"Saved held-out samples to: {heldout_dir}")
    
    # Save split info
    split_info = {
        'n_train': len(train_samples),
        'n_test': len(test_samples),
        'seed': seed,
        'test_ids': [s.clip_id for s in test_samples],
    }
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Train each condition
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Training: {condition}")
        print(f"Config: {ABLATION_CONDITIONS[condition]}")
        print(f"{'='*60}")
        
        cond_dir = output_dir / condition
        cond_dir.mkdir(exist_ok=True)
        
        # Create ablation wrapper
        wrapper = create_ablation_generator(condition, {})
        
        # Train with modified inputs
        device = gen_module.pick_device()
        
        train_ds = gen_module.FLAMEDataset(train_samples, seq_len=64)
        train_loader = DataLoader(
            train_ds, batch_size=16, shuffle=True,
            collate_fn=gen_module.collate_fn, num_workers=2, drop_last=True
        )
        
        test_ds = gen_module.FLAMEDataset(test_samples, seq_len=64)
        test_loader = DataLoader(
            test_ds, batch_size=16, shuffle=False,
            collate_fn=gen_module.collate_fn, num_workers=2
        )
        
        model = gen_module.FLAMEGeneratorV9(hidden_dim=192, gru_layers=2, dropout=0.1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_loss = float('inf')
        train_history = []
        val_history = []
        
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_losses = []
            
            for batch in train_loader:
                # Apply ablation modification
                batch = wrapper.modify_batch(batch)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                out = model(
                    batch['nmf'], batch['emotions'], batch['token_ids'], batch['token_mask'],
                    batch['phoneme_feats'], batch['time_feats'],
                    batch['blend_curve'], apply_neutral_blend=False
                )
                
                loss, metrics = gen_module.compute_loss(out, batch['flame_params'], batch['nmf'])
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(metrics)
            
            avg_train = {k: np.mean([m[k] for m in train_losses]) for k in train_losses[0]}
            train_history.append(avg_train)
            
            # Validation
            if epoch % 5 == 0:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        batch = wrapper.modify_batch(batch)
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        
                        out = model(
                            batch['nmf'], batch['emotions'], batch['token_ids'], batch['token_mask'],
                            batch['phoneme_feats'], batch['time_feats'],
                            batch['blend_curve'], apply_neutral_blend=False
                        )
                        
                        _, metrics = gen_module.compute_loss(out, batch['flame_params'], batch['nmf'])
                        val_losses.append(metrics)
                
                avg_val = {k: np.mean([m[k] for m in val_losses]) for k in val_losses[0]}
                val_history.append({'epoch': epoch, **avg_val})
                val_loss = avg_val['loss']
            else:
                val_loss = avg_train['loss']
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:03d}/{epochs} | train {avg_train['loss']:.5f} | val {val_loss:.5f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'config': {'hidden_dim': 192, 'gru_layers': 2, 'dropout': 0.1},
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'condition': condition,
                }, cond_dir / 'best.pt')
        
        # Save training history
        with open(cond_dir / 'train_history.json', 'w') as f:
            json.dump({'train': train_history, 'val': val_history}, f, indent=2)
        
        print(f"  Best loss: {best_loss:.6f}")
        print(f"  Saved to: {cond_dir / 'best.pt'}")
    
    print(f"\n{'='*60}")
    print("Ablation training complete!")
    print(f"{'='*60}")


def run_ablation_reconstruction(ablation_dir: str, conditions: List[str] = None):
    if not HAS_TORCH:
        print("ERROR: PyTorch required")
        return
    
    ablation_dir = Path(ablation_dir)
    heldout_dir = ablation_dir / 'heldout_samples'
    
    if not heldout_dir.exists():
        print(f"ERROR: Held-out samples not found at {heldout_dir}")
        return
    
    conditions = conditions or list(ABLATION_CONDITIONS.keys())
    
    # Import generator
    try:
        import importlib.util
        gen_path = Path(__file__).parent / 'flame_generator.py'
        if not gen_path.exists():
            gen_path = Path('flame_generator.py')
        
        spec = importlib.util.spec_from_file_location("flame_generator", gen_path)
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
    except Exception as e:
        print(f"ERROR: Could not import flame_generator.py: {e}")
        return
    
    device = gen_module.pick_device()
    
    for condition in conditions:
        print(f"\nReconstructing: {condition}")
        
        cond_dir = ablation_dir / condition
        checkpoint_path = cond_dir / 'best.pt'
        
        if not checkpoint_path.exists():
            print(f"  Checkpoint not found: {checkpoint_path}")
            continue
        
        recon_dir = cond_dir / 'reconstructions'
        recon_dir.mkdir(exist_ok=True)
        
        # Load model
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt.get('config', {})
        model = gen_module.FLAMEGeneratorV9(
            hidden_dim=cfg.get('hidden_dim', 192),
            gru_layers=cfg.get('gru_layers', 2),
            dropout=cfg.get('dropout', 0.1)
        ).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        # Get ablation wrapper
        wrapper = create_ablation_generator(condition, {})
        
        # Reconstruct each held-out sample
        files = sorted([p for p in heldout_dir.glob('*.npz') if p.stem != 'manifest'])
        
        with torch.no_grad():
            for path in files:
                d = np.load(path, allow_pickle=True)
                
                clip_id = str(d['clip_id'].item()) if d['clip_id'].shape == () else str(d['clip_id'])
                text = str(d['text'].item()) if d['text'].shape == () else str(d['text'])
                emotions = d['emotions'].astype(np.float32)
                nmf = d['nmf'].astype(np.float32)
                duration = len(nmf)
                
                token_ids_np = d['token_ids'].astype(np.int64)
                phoneme_np = d['phoneme_feats'].astype(np.float32)
                
                # Prepare tensors
                nmf_t = torch.from_numpy(nmf).unsqueeze(0).to(device)
                emo_t = torch.from_numpy(emotions).unsqueeze(0).to(device)
                token_ids = torch.from_numpy(token_ids_np).unsqueeze(0).to(device)
                token_mask = torch.ones(1, token_ids.shape[1], device=device)
                phoneme_feats = torch.from_numpy(phoneme_np).unsqueeze(0).to(device)
                time_feats = torch.from_numpy(gen_module.temporal_features(duration)).unsqueeze(0).to(device)
                blend_curve = torch.from_numpy(gen_module.create_neutral_blend_curve(duration)).unsqueeze(0).to(device)
                
                # Apply ablation
                batch = {
                    'nmf': nmf_t,
                    'emotions': emo_t,
                    'token_ids': token_ids,
                    'token_mask': token_mask,
                    'phoneme_feats': phoneme_feats,
                    'time_feats': time_feats,
                    'blend_curve': blend_curve,
                }
                batch = wrapper.modify_batch(batch)
                
                out = model(
                    batch['nmf'], batch['emotions'], batch['token_ids'], batch['token_mask'],
                    batch['phoneme_feats'], batch['time_feats'],
                    batch['blend_curve'], apply_neutral_blend=False
                )
                
                flame_params = out['flame_params'][0].cpu().numpy()
                
                out_path = recon_dir / f"{clip_id}_recon.npz"
                np.savez(
                    out_path,
                    flame_params=flame_params,
                    expression=out['expression'][0].cpu().numpy(),
                    jaw_pose=out['jaw_pose'][0].cpu().numpy(),
                    eye_pose=out['eye_pose'][0].cpu().numpy(),
                    global_orient=out['global_orient'][0].cpu().numpy(),
                    nmf=nmf,
                    emotions=emotions,
                    text=text,
                    fps=np.float32(30.0),
                )
        
        print(f"  Reconstructed {len(files)} samples to {recon_dir}")

def evaluate_ablation_study(ablation_dir: str, output_dir: str, conditions: List[str] = None) -> Dict[str, Dict]:

    ablation_dir = Path(ablation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    heldout_dir = ablation_dir / 'heldout_samples'
    if not heldout_dir.exists():
        print(f"ERROR: Held-out samples not found at {heldout_dir}")
        return {}
    
    conditions = conditions or list(ABLATION_CONDITIONS.keys())
    
    all_results = {}
    all_sample_results = {}
    
    for condition in conditions:
        print(f"\nEvaluating: {condition}")
        
        recon_dir = ablation_dir / condition / 'reconstructions'
        if not recon_dir.exists():
            print(f"  Reconstructions not found: {recon_dir}")
            continue
        
        # Find matching pairs
        recon_files = {p.stem.replace('_recon', ''): p for p in recon_dir.glob('*_recon.npz')}
        heldout_files = {p.stem: p for p in heldout_dir.glob('*.npz') if p.stem != 'manifest'}
        
        common_ids = set(recon_files.keys()) & set(heldout_files.keys())
        print(f"  Found {len(common_ids)} matching pairs")
        
        sample_results = []
        for clip_id in sorted(common_ids):
            try:
                result = evaluate_single_sample(
                    recon_files[clip_id],
                    heldout_files[clip_id],
                    condition
                )
                sample_results.append(result)
            except Exception as e:
                print(f"  Error evaluating {clip_id}: {e}")
        
        if sample_results:
            aggregated = aggregate_results(sample_results)
            all_results[condition] = aggregated
            all_sample_results[condition] = sample_results
            
            # Save per-condition results
            cond_output = output_dir / condition
            cond_output.mkdir(exist_ok=True)
            
            with open(cond_output / 'aggregated_metrics.json', 'w') as f:
                json.dump(aggregated, f, indent=2)
            
            # Save per-sample results
            per_sample = [
                {
                    'clip_id': r.clip_id,
                    'duration': r.duration,
                    'mse': r.reconstruction.mse,
                    'mae': r.reconstruction.mae,
                    'correlation': r.reconstruction.correlation,
                    'velocity_mse': r.temporal.velocity_mse,
                    'jerk_mean': r.temporal.jerk_mean_pred,
                    'smoothness_ratio': r.temporal.smoothness_ratio,
                }
                for r in sample_results
            ]
            with open(cond_output / 'per_sample_metrics.json', 'w') as f:
                json.dump(per_sample, f, indent=2)
    
    # Save combined results
    with open(output_dir / 'ablation_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Condition':<20} {'MSE':>10} {'MAE':>10} {'Corr':>10} {'Vel MSE':>10} {'Jerk':>10}")
    print("-"*80)
    
    for cond in conditions:
        if cond in all_results:
            r = all_results[cond]
            print(f"{CONDITION_LABELS[cond]:<20} "
                  f"{r.get('recon_mse_mean', 0):>10.5f} "
                  f"{r.get('recon_mae_mean', 0):>10.5f} "
                  f"{r.get('recon_correlation_mean', 0):>10.3f} "
                  f"{r.get('temporal_velocity_mse_mean', 0):>10.5f} "
                  f"{r.get('temporal_jerk_mean_pred_mean', 0):>10.2f}")
    
    return all_results, all_sample_results

def generate_all_figures(results_dir: str, output_dir: str, conditions: List[str] = None):
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib required for figures")
        return
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_dir / 'ablation_comparison.json') as f:
        all_results = json.load(f)
    
    conditions = conditions or [c for c in ABLATION_CONDITIONS.keys() if c in all_results]
    
    # Set style
    if HAS_SEABORN:
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })
    
    # Figure 1: Overall MSE comparison
    fig1_overall_mse(all_results, conditions, output_dir)
    
    # Figure 2: Per-component error breakdown
    fig2_component_breakdown(all_results, conditions, output_dir)
    
    # Figure 3: Temporal quality metrics
    fig3_temporal_quality(all_results, conditions, output_dir)
    
    # Figure 4: Regional expression error
    fig4_regional_error(all_results, conditions, output_dir)
    
    # Figure 5: Control fidelity (NMF correlation)
    fig5_control_fidelity(all_results, conditions, output_dir)
    
    # Figure 6: Combined summary figure
    fig6_summary(all_results, conditions, output_dir)

    
    print(f"\nAll figures saved to: {output_dir}")


def fig1_overall_mse(results: Dict, conditions: List[str], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE comparison
    mse_means = [results[c].get('recon_mse_mean', 0) for c in conditions]
    mse_stds = [results[c].get('recon_mse_std', 0) for c in conditions]
    
    x = np.arange(len(conditions))
    colors = [CONDITION_COLORS[c] for c in conditions]
    labels = [CONDITION_LABELS[c] for c in conditions]
    
    bars = axes[0].bar(x, mse_means, yerr=mse_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('(a) Overall Reconstruction MSE')
    
    # Add value labels
    for bar, val in zip(bars, mse_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + mse_stds[bars.index(bar)] + 0.0005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Correlation comparison
    corr_means = [results[c].get('recon_correlation_mean', 0) for c in conditions]
    corr_stds = [results[c].get('recon_correlation_std', 0) for c in conditions]
    
    bars = axes[1].bar(x, corr_means, yerr=corr_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('(b) Reconstruction Correlation')
    axes[1].set_ylim(0, 1.0)
    
    for bar, val in zip(bars, corr_means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_overall_mse.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_overall_mse.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig1_overall_mse.png/pdf")


def fig2_component_breakdown(results: Dict, conditions: List[str], output_dir: Path):
    
    components = ['expression', 'jaw_pose', 'eye_pose', 'global_orient']
    comp_labels = ['Expression\n(100D)', 'Jaw Pose\n(3D)', 'Eye Pose\n(6D)', 'Head Orient\n(3D)']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(components))
    width = 0.2
    
    for i, cond in enumerate(conditions):
        mse_values = [results[cond].get(f'recon_mse_{c}_mean', 0) for c in components]
        offset = (i - len(conditions)/2 + 0.5) * width
        bars = ax.bar(x + offset, mse_values, width, label=CONDITION_LABELS[cond],
                     color=CONDITION_COLORS[cond], alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Per-Component Reconstruction Error')
    ax.legend(loc='upper right')
    ax.set_yscale('log')  # Log scale for better visibility
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_component_breakdown.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_component_breakdown.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig2_component_breakdown.png/pdf")


def fig3_temporal_quality(results: Dict, conditions: List[str], output_dir: Path):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(conditions))
    colors = [CONDITION_COLORS[c] for c in conditions]
    labels = [CONDITION_LABELS[c] for c in conditions]
    
    # Velocity MSE
    vel_mse = [results[c].get('temporal_velocity_mse_mean', 0) for c in conditions]
    axes[0].bar(x, vel_mse, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].set_ylabel('Velocity MSE')
    axes[0].set_title('(a) Motion Velocity Error')
    
    # Acceleration MSE
    accel_mse = [results[c].get('temporal_accel_mse_mean', 0) for c in conditions]
    axes[1].bar(x, accel_mse, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1].set_ylabel('Acceleration MSE')
    axes[1].set_title('(b) Motion Acceleration Error')
    
    # Jerk (smoothness)
    jerk = [results[c].get('temporal_jerk_mean_pred_mean', 0) for c in conditions]
    axes[2].bar(x, jerk, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha='right')
    axes[2].set_ylabel('Mean Jerk')
    axes[2].set_title('(c) Motion Smoothness\n(Lower = Smoother)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_temporal_quality.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_temporal_quality.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_temporal_quality.png/pdf")


def fig4_regional_error(results: Dict, conditions: List[str], output_dir: Path):
    
    regions = ['mouth_open', 'smile_frown', 'lips', 'brows', 'eyes_expr', 'cheeks', 'nose']
    region_labels = ['Mouth\nOpen', 'Smile/\nFrown', 'Lips', 'Brows', 'Eyes', 'Cheeks', 'Nose']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(regions))
    width = 0.2
    
    for i, cond in enumerate(conditions):
        mse_values = [results[cond].get(f'recon_mse_{r}_mean', 0) for r in regions]
        offset = (i - len(conditions)/2 + 0.5) * width
        ax.bar(x + offset, mse_values, width, label=CONDITION_LABELS[cond],
              color=CONDITION_COLORS[cond], alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels)
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Expression Error by Facial Region')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_regional_error.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_regional_error.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_regional_error.png/pdf")


def fig5_control_fidelity(results: Dict, conditions: List[str], output_dir: Path):
    # Only show conditions that use NMF
    nmf_conditions = [c for c in conditions if ABLATION_CONDITIONS[c]['use_nmf']]
    
    if not nmf_conditions:
        print("  Skipping fig5 (no NMF conditions)")
        return
    
    controls = ['nmf_mouth_open_corr', 'nmf_mouth_spread_corr', 'nmf_brow_corr', 'nmf_head_turn_corr']
    control_labels = ['Mouth Open\n→ Jaw', 'Mouth Spread\n→ Smile', 'Brow Raise\n→ Brows', 'Head Turn\n→ Yaw']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(controls))
    width = 0.35
    
    for i, cond in enumerate(nmf_conditions):
        corr_values = [results[cond].get(f'ctrl_{c}_mean', 0) for c in controls]
        offset = (i - len(nmf_conditions)/2 + 0.5) * width
        ax.bar(x + offset, corr_values, width, label=CONDITION_LABELS[cond],
              color=CONDITION_COLORS[cond], alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(control_labels)
    ax.set_ylabel('Correlation')
    ax.set_title('NMF Control Signal Fidelity')
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_control_fidelity.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_control_fidelity.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_control_fidelity.png/pdf")


def fig6_summary(results: Dict, conditions: List[str], output_dir: Path):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = [CONDITION_COLORS[c] for c in conditions]
    labels = [CONDITION_LABELS[c] for c in conditions]
    x = np.arange(len(conditions))
    
    # (a) Overall MSE
    ax1 = fig.add_subplot(gs[0, 0])
    mse_means = [results[c].get('recon_mse_mean', 0) for c in conditions]
    mse_stds = [results[c].get('recon_mse_std', 0) for c in conditions]
    ax1.bar(x, mse_means, yerr=mse_stds, capsize=4, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('MSE')
    ax1.set_title('(a) Reconstruction Error')
    
    # (b) Correlation
    ax2 = fig.add_subplot(gs[0, 1])
    corr_means = [results[c].get('recon_correlation_mean', 0) for c in conditions]
    ax2.bar(x, corr_means, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('Correlation')
    ax2.set_title('(b) Reconstruction Correlation')
    ax2.set_ylim(0, 1)
    
    # (c) Velocity MSE
    ax3 = fig.add_subplot(gs[0, 2])
    vel_mse = [results[c].get('temporal_velocity_mse_mean', 0) for c in conditions]
    ax3.bar(x, vel_mse, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax3.set_ylabel('Velocity MSE')
    ax3.set_title('(c) Motion Dynamics Error')
    
    # (d) Component breakdown
    ax4 = fig.add_subplot(gs[1, 0])
    components = ['expression', 'jaw_pose', 'eye_pose', 'global_orient']
    comp_labels_short = ['Expr', 'Jaw', 'Eye', 'Orient']
    width = 0.2
    for i, cond in enumerate(conditions):
        mse_values = [results[cond].get(f'recon_mse_{c}_mean', 0) for c in components]
        offset = (i - len(conditions)/2 + 0.5) * width
        ax4.bar(np.arange(len(components)) + offset, mse_values, width, 
               color=CONDITION_COLORS[cond], alpha=0.8)
    ax4.set_xticks(np.arange(len(components)))
    ax4.set_xticklabels(comp_labels_short)
    ax4.set_ylabel('MSE')
    ax4.set_title('(d) Per-Component Error')
    ax4.set_yscale('log')
    
    # (e) Jerk comparison
    ax5 = fig.add_subplot(gs[1, 1])
    jerk = [results[c].get('temporal_jerk_mean_pred_mean', 0) for c in conditions]
    ax5.bar(x, jerk, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax5.set_ylabel('Mean Jerk')
    ax5.set_title('(e) Motion Smoothness (↓ better)')
    
    # (f) Improvement over baseline
    ax6 = fig.add_subplot(gs[1, 2])
    baseline_mse = results['text_only'].get('recon_mse_mean', 1)
    improvements = [(1 - results[c].get('recon_mse_mean', 0) / baseline_mse) * 100 
                    for c in conditions]
    bars = ax6.bar(x, improvements, color=colors, alpha=0.8, edgecolor='black')
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax6.set_ylabel('Improvement (%)')
    ax6.set_title('(f) MSE Improvement over Baseline')
    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
        ax6.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}%', 
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    plt.savefig(output_dir / 'fig6_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_summary.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig6_summary.png/pdf")

def main():
    parser = argparse.ArgumentParser(description="FLAME Generator Ablation Study & Evaluation")
    subparsers = parser.add_subparsers(dest='command')
    
    # Train ablation models
    train_p = subparsers.add_parser('train_ablation', help='Train models for ablation study')
    train_p.add_argument('--flame_dir', required=True, help='Directory with FLAME NPZ files')
    train_p.add_argument('--jsonl', required=True, help='Metadata JSONL file')
    train_p.add_argument('--output_dir', default='ablation_study', help='Output directory')
    train_p.add_argument('--epochs', type=int, default=150)
    train_p.add_argument('--seed', type=int, default=42)
    train_p.add_argument('--conditions', nargs='+', default=None, 
                        help='Specific conditions to train (default: all)')
    
    # Run reconstruction
    recon_p = subparsers.add_parser('reconstruct', help='Run reconstruction for all conditions')
    recon_p.add_argument('--ablation_dir', required=True, help='Ablation study directory')
    recon_p.add_argument('--conditions', nargs='+', default=None)
    
    # Evaluate
    eval_p = subparsers.add_parser('evaluate', help='Evaluate all ablation conditions')
    eval_p.add_argument('--ablation_dir', required=True, help='Ablation study directory')
    eval_p.add_argument('--output_dir', default='evaluation_results', help='Results output directory')
    eval_p.add_argument('--conditions', nargs='+', default=None)
    
    # Generate figures
    fig_p = subparsers.add_parser('figures', help='Generate dissertation figures')
    fig_p.add_argument('--results_dir', required=True, help='Evaluation results directory')
    fig_p.add_argument('--output_dir', default='dissertation_figures', help='Figures output directory')
    fig_p.add_argument('--conditions', nargs='+', default=None)
    
    # Full pipeline
    full_p = subparsers.add_parser('full', help='Run complete ablation pipeline')
    full_p.add_argument('--flame_dir', required=True)
    full_p.add_argument('--jsonl', required=True)
    full_p.add_argument('--output_dir', default='ablation_full', help='Base output directory')
    full_p.add_argument('--epochs', type=int, default=150)
    full_p.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.command == 'train_ablation':
        train_ablation_models(
            args.flame_dir, args.jsonl, args.output_dir,
            epochs=args.epochs, seed=args.seed, conditions=args.conditions
        )
    
    elif args.command == 'reconstruct':
        run_ablation_reconstruction(args.ablation_dir, args.conditions)
    
    elif args.command == 'evaluate':
        evaluate_ablation_study(args.ablation_dir, args.output_dir, args.conditions)
    
    elif args.command == 'figures':
        generate_all_figures(args.results_dir, args.output_dir, args.conditions)
    
    elif args.command == 'full':
        # Run complete pipeline
        ablation_dir = Path(args.output_dir) / 'models'
        results_dir = Path(args.output_dir) / 'results'
        figures_dir = Path(args.output_dir) / 'figures'
        
        print("\n" + "="*60)
        print("STEP 1: Training ablation models")
        print("="*60)
        train_ablation_models(args.flame_dir, args.jsonl, str(ablation_dir),
                             epochs=args.epochs, seed=args.seed)
        
        print("\n" + "="*60)
        print("STEP 2: Running reconstruction")
        print("="*60)
        run_ablation_reconstruction(str(ablation_dir))
        
        print("\n" + "="*60)
        print("STEP 3: Evaluating results")
        print("="*60)
        evaluate_ablation_study(str(ablation_dir), str(results_dir))
        
        print("\n" + "="*60)
        print("STEP 4: Generating figures")
        print("="*60)
        generate_all_figures(str(results_dir), str(figures_dir))
        
        print("\n" + "="*60)
        print("COMPLETE!")
        print(f"Results: {results_dir}")
        print(f"Figures: {figures_dir}")
        print("="*60)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
