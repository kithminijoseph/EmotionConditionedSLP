import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

EXPRESSION_REGIONS = {
    'mouth_open': list(range(0, 5)),
    'smile_frown': [10, 11, 12, 13],
    'lips': [5, 6, 7, 8, 9, 14, 15],
    'brows': [30, 31, 32, 33, 34, 35],
    'eyes_expr': [40, 41, 42, 43, 44, 45],
    'cheeks': [16, 17, 20, 21],
    'nose': [18, 19, 22, 23],
}

EMOTION_RELATED_REGIONS = ['smile_frown', 'brows', 'eyes_expr', 'cheeks']
NMF_RELATED_REGIONS = ['mouth_open', 'lips']

EMOTION_NAMES = [
    "joy", "excited", "surprise_pos", "surprise_neg", "worry",
    "sadness", "fear", "disgust", "frustration", "anger"
]

CONDITION_COLORS = {
    'text_only': '#1f77b4',
    'text_emotion': '#ff7f0e',
    'text_nmf': '#2ca02c',
    'full_model': '#d62728',
}

def analyse_where_emotion_helps(heldout_dir: str, ablation_dir: str, output_dir: str):  
    heldout_dir = Path(heldout_dir)
    ablation_dir = Path(ablation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    text_nmf_recon = ablation_dir / 'text_nmf' / 'reconstructions'
    full_model_recon = ablation_dir / 'full_model' / 'reconstructions'
    
    if not text_nmf_recon.exists() or not full_model_recon.exists():
        print("ERROR: Reconstruction directories not found")
        return
    
    # Collect per-sample analysis
    samples = []
    heldout_files = {p.stem: p for p in heldout_dir.glob('*.npz') if p.stem != 'manifest'}
    
    for clip_id, target_path in heldout_files.items():
        text_nmf_path = text_nmf_recon / f'{clip_id}_recon.npz'
        full_path = full_model_recon / f'{clip_id}_recon.npz'
        
        if not text_nmf_path.exists() or not full_path.exists():
            continue
        
        # Load data
        target_data = np.load(target_path, allow_pickle=True)
        target = target_data['flame_params'].astype(np.float32)
        emotions = target_data.get('emotions', np.zeros(10)).astype(np.float32)
        
        text_nmf_pred = np.load(text_nmf_path)['flame_params'].astype(np.float32)
        full_pred = np.load(full_path)['flame_params'].astype(np.float32)
        
        T = min(len(target), len(text_nmf_pred), len(full_pred))
        target = target[:T]
        text_nmf_pred = text_nmf_pred[:T]
        full_pred = full_pred[:T]
        
        # Compute metrics
        sample = {
            'clip_id': clip_id,
            'duration': T,
            'emotion_intensity': float(np.max(emotions)),
            'dominant_emotion': EMOTION_NAMES[int(np.argmax(emotions))],
            'emotions': emotions.tolist(),
        }
        
        # Overall MSE
        sample['text_nmf_mse'] = float(np.mean((text_nmf_pred - target) ** 2))
        sample['full_model_mse'] = float(np.mean((full_pred - target) ** 2))
        sample['emotion_improvement'] = sample['text_nmf_mse'] - sample['full_model_mse']
        sample['emotion_improvement_pct'] = (sample['emotion_improvement'] / sample['text_nmf_mse']) * 100
        
        # Per-region analysis
        for region, indices in EXPRESSION_REGIONS.items():
            valid_idx = [i for i in indices if i < 100]
            if valid_idx:
                text_nmf_region = float(np.mean((text_nmf_pred[:, valid_idx] - target[:, valid_idx]) ** 2))
                full_region = float(np.mean((full_pred[:, valid_idx] - target[:, valid_idx]) ** 2))
                sample[f'{region}_text_nmf_mse'] = text_nmf_region
                sample[f'{region}_full_mse'] = full_region
                sample[f'{region}_improvement'] = text_nmf_region - full_region
        
        # Velocity analysis
        vel_text_nmf = np.diff(text_nmf_pred, axis=0)
        vel_full = np.diff(full_pred, axis=0)
        vel_target = np.diff(target, axis=0)
        
        sample['text_nmf_vel_mse'] = float(np.mean((vel_text_nmf - vel_target) ** 2))
        sample['full_vel_mse'] = float(np.mean((vel_full - vel_target) ** 2))
        sample['vel_improvement'] = sample['text_nmf_vel_mse'] - sample['full_vel_mse']
        
        samples.append(sample)
    
    print(f"Analysed {len(samples)} samples")
    
    # Compute aggregate statistics
    analysis = compute_detailed_statistics(samples)
    
    # Save results
    with open(output_dir / 'detailed_emotion_analysis.json', 'w') as f:
        json.dump({
            'samples': samples,
            'analysis': analysis,
        }, f, indent=2)
    
    # Print key findings
    print_analysis_summary(analysis)
    
    # Generate visualizations
    if HAS_MATPLOTLIB:
        create_analysis_figures(samples, analysis, output_dir)
    
    return analysis

def compute_detailed_statistics(samples: List[Dict]) -> Dict:
    analysis = {
        'n_samples': len(samples),
    }
    
    # Overall statistics
    improvements = [s['emotion_improvement_pct'] for s in samples]
    analysis['mean_improvement_pct'] = float(np.mean(improvements))
    analysis['std_improvement_pct'] = float(np.std(improvements))
    analysis['median_improvement_pct'] = float(np.median(improvements))
    analysis['samples_where_emotion_helps'] = sum(1 for x in improvements if x > 0)
    analysis['samples_where_emotion_hurts'] = sum(1 for x in improvements if x < 0)
    analysis['pct_samples_emotion_helps'] = analysis['samples_where_emotion_helps'] / len(samples) * 100
    
    # By emotion intensity
    high_emotion = [s for s in samples if s['emotion_intensity'] > 0.5]
    low_emotion = [s for s in samples if s['emotion_intensity'] <= 0.5]
    
    if high_emotion:
        high_improvements = [s['emotion_improvement_pct'] for s in high_emotion]
        analysis['high_emotion_n'] = len(high_emotion)
        analysis['high_emotion_mean_improvement'] = float(np.mean(high_improvements))
        analysis['high_emotion_helps_pct'] = sum(1 for x in high_improvements if x > 0) / len(high_improvements) * 100
    
    if low_emotion:
        low_improvements = [s['emotion_improvement_pct'] for s in low_emotion]
        analysis['low_emotion_n'] = len(low_emotion)
        analysis['low_emotion_mean_improvement'] = float(np.mean(low_improvements))
        analysis['low_emotion_helps_pct'] = sum(1 for x in low_improvements if x > 0) / len(low_improvements) * 100
    
    # By region
    for region in EXPRESSION_REGIONS.keys():
        key = f'{region}_improvement'
        region_improvements = [s[key] for s in samples if key in s]
        if region_improvements:
            analysis[f'{region}_mean_improvement'] = float(np.mean(region_improvements))
            analysis[f'{region}_helps_pct'] = sum(1 for x in region_improvements if x > 0) / len(region_improvements) * 100
    
    # Emotion-related vs NMF-related regions
    emotion_region_means = [analysis.get(f'{r}_mean_improvement', 0) for r in EMOTION_RELATED_REGIONS]
    nmf_region_means = [analysis.get(f'{r}_mean_improvement', 0) for r in NMF_RELATED_REGIONS]
    
    analysis['emotion_regions_mean_improvement'] = float(np.mean(emotion_region_means))
    analysis['nmf_regions_mean_improvement'] = float(np.mean(nmf_region_means))
    analysis['emotion_helps_more_in_emotion_regions'] = analysis['emotion_regions_mean_improvement'] > analysis['nmf_regions_mean_improvement']
    
    # Velocity improvement
    vel_improvements = [s['vel_improvement'] for s in samples]
    analysis['velocity_mean_improvement'] = float(np.mean(vel_improvements))
    analysis['velocity_helps_pct'] = sum(1 for x in vel_improvements if x > 0) / len(vel_improvements) * 100
    
    # Statistical tests
    if HAS_SCIPY:
        text_nmf_mses = [s['text_nmf_mse'] for s in samples]
        full_mses = [s['full_model_mse'] for s in samples]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(text_nmf_mses, full_mses)
        analysis['overall_ttest_t'] = float(t_stat)
        analysis['overall_ttest_p'] = float(p_value)
        analysis['overall_significant_05'] = p_value < 0.05
        analysis['overall_significant_01'] = p_value < 0.01
        
        # Effect size
        diff = np.array(text_nmf_mses) - np.array(full_mses)
        if np.std(diff) > 0:
            analysis['cohens_d'] = float(np.mean(diff) / np.std(diff))
        
        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_p = stats.wilcoxon(text_nmf_mses, full_mses)
            analysis['wilcoxon_stat'] = float(w_stat)
            analysis['wilcoxon_p'] = float(w_p)
        except:
            pass
        
        # Test for emotion-related regions specifically
        emotion_region_text_nmf = []
        emotion_region_full = []
        for s in samples:
            text_nmf_avg = np.mean([s.get(f'{r}_text_nmf_mse', 0) for r in EMOTION_RELATED_REGIONS])
            full_avg = np.mean([s.get(f'{r}_full_mse', 0) for r in EMOTION_RELATED_REGIONS])
            emotion_region_text_nmf.append(text_nmf_avg)
            emotion_region_full.append(full_avg)
        
        t_stat_emo, p_value_emo = stats.ttest_rel(emotion_region_text_nmf, emotion_region_full)
        analysis['emotion_regions_ttest_t'] = float(t_stat_emo)
        analysis['emotion_regions_ttest_p'] = float(p_value_emo)
        analysis['emotion_regions_significant'] = p_value_emo < 0.05
    
    return analysis


def print_analysis_summary(analysis: Dict):
    print("DETAILED EMOTION CONTRIBUTION ANALYSIS")
    print(f"OVERALL RESULTS (N={analysis['n_samples']} samples)")
    print(f"Mean improvement from emotion: {analysis['mean_improvement_pct']:.2f}% ± {analysis['std_improvement_pct']:.2f}%")
    print(f"Samples where emotion helps: {analysis['samples_where_emotion_helps']}/{analysis['n_samples']} ({analysis['pct_samples_emotion_helps']:.1f}%)")
    
    if 'overall_ttest_p' in analysis:
        sig_str = "SIGNIFICANT" if analysis['overall_significant_05'] else "Not significant"
        print(f"   Statistical significance: p={analysis['overall_ttest_p']:.4f} {sig_str}")
        if 'cohens_d' in analysis:
            print(f"   Effect size (Cohen's d): {analysis['cohens_d']:.3f}")
    
    print(f"BY EMOTION INTENSITY")
    if 'high_emotion_n' in analysis:
        print(f"High emotion samples (intensity > 0.5): N={analysis['high_emotion_n']}")
        print(f"Mean improvement: {analysis['high_emotion_mean_improvement']:.2f}%")
        print(f"Emotion helps in: {analysis['high_emotion_helps_pct']:.1f}% of samples")
    if 'low_emotion_n' in analysis:
        print(f"Low emotion samples (intensity ≤ 0.5): N={analysis['low_emotion_n']}")
        print(f"Mean improvement: {analysis['low_emotion_mean_improvement']:.2f}%")
        print(f"Emotion helps in: {analysis['low_emotion_helps_pct']:.1f}% of samples")
    
    print(f"BY FACIAL REGION")
    print(f"Emotion-related regions (smile/frown, brows, eyes, cheeks):")
    print(f"Mean improvement: {analysis['emotion_regions_mean_improvement']*1000:.3f} ×10⁻³")
    if 'emotion_regions_significant' in analysis:
        sig_str = "SIGNIFICANT" if analysis['emotion_regions_significant'] else "Not significant"
        print(f"Statistical test: p={analysis['emotion_regions_ttest_p']:.4f} {sig_str}")
    
    print(f"NMF-related regions (mouth open, lips):")
    print(f"Mean improvement: {analysis['nmf_regions_mean_improvement']*1000:.3f} ×10⁻³")
    
    if analysis['emotion_helps_more_in_emotion_regions']:
        print(f"KEY FINDING: Emotion helps MORE in emotion-related regions!")
    
    print(f"TEMPORAL QUALITY (Velocity)")
    print(f"Mean velocity improvement: {analysis['velocity_mean_improvement']*1000:.3f} ×10⁻³")
    print(f"Emotion improves velocity in: {analysis['velocity_helps_pct']:.1f}% of samples")


def create_analysis_figures(samples: List[Dict], analysis: Dict, output_dir: Path):
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    intensities = [s['emotion_intensity'] for s in samples]
    improvements = [s['emotion_improvement_pct'] for s in samples]
    colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
    
    axes[0, 0].scatter(intensities, improvements, c=colors, alpha=0.6, edgecolor='black', s=80)
    axes[0, 0].axhline(0, color='gray', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Emotion Intensity (max score)', fontsize=12)
    axes[0, 0].set_ylabel('MSE Improvement from Emotion (%)', fontsize=12)
    axes[0, 0].set_title('(a) Emotion Contribution vs Emotion Intensity', fontsize=13)
    
    # Add regression line
    if HAS_SCIPY:
        slope, intercept, r, p, se = stats.linregress(intensities, improvements)
        x_line = np.linspace(0, max(intensities), 100)
        axes[0, 0].plot(x_line, slope * x_line + intercept, 'b-', linewidth=2,
                       label=f'Trend: r={r:.3f}, p={p:.3f}')
        axes[0, 0].legend(fontsize=10)

    regions = list(EXPRESSION_REGIONS.keys())
    region_improvements = [analysis.get(f'{r}_mean_improvement', 0) * 1000 for r in regions]
    
    bar_colors = []
    for r in regions:
        if r in EMOTION_RELATED_REGIONS:
            bar_colors.append('#ff7f0e')  # Orange for emotion-related
        else:
            bar_colors.append('#2ca02c')  # Green for NMF-related
    
    bars = axes[0, 1].bar(range(len(regions)), region_improvements, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[0, 1].set_xticks(range(len(regions)))
    axes[0, 1].set_xticklabels([r.replace('_', '\n') for r in regions], rotation=45, ha='right', fontsize=10)
    axes[0, 1].axhline(0, color='gray', linestyle='--', linewidth=2)
    axes[0, 1].set_ylabel('Mean MSE Improvement (×10⁻³)', fontsize=12)
    axes[0, 1].set_title('(b) Emotion Contribution by Facial Region', fontsize=13)
    
    # Legend
    orange_patch = plt.Rectangle((0,0), 1, 1, fc='#ff7f0e', alpha=0.8)
    green_patch = plt.Rectangle((0,0), 1, 1, fc='#2ca02c', alpha=0.8)
    axes[0, 1].legend([orange_patch, green_patch], ['Emotion-related', 'NMF-related'], loc='upper right')
    
    # (c) High vs Low emotion comparison
    categories = ['High Emotion\n(intensity > 0.5)', 'Low Emotion\n(intensity ≤ 0.5)']
    means = [analysis.get('high_emotion_mean_improvement', 0), analysis.get('low_emotion_mean_improvement', 0)]
    ns = [analysis.get('high_emotion_n', 0), analysis.get('low_emotion_n', 0)]
    
    colors = ['#ff7f0e', '#1f77b4']
    bars = axes[1, 0].bar(categories, means, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 0].axhline(0, color='gray', linestyle='--', linewidth=2)
    axes[1, 0].set_ylabel('Mean MSE Improvement (%)', fontsize=12)
    axes[1, 0].set_title('(c) Emotion Helps More for High-Emotion Samples', fontsize=13)
    
    # Add sample counts
    for bar, n in zip(bars, ns):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'N={n}', ha='center', fontsize=10)
    
    # (d) Improvement distribution with key statistics
    all_improvements = [s['emotion_improvement_pct'] for s in samples]
    
    n_bins = 20
    axes[1, 1].hist(all_improvements, bins=n_bins, edgecolor='black', alpha=0.7, color='#9467bd')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
    axes[1, 1].axvline(np.mean(all_improvements), color='blue', linestyle='-', linewidth=2,
                      label=f'Mean: {np.mean(all_improvements):.2f}%')
    
    # Add annotation for samples where emotion helps
    helps_pct = analysis['pct_samples_emotion_helps']
    axes[1, 1].annotate(f'Emotion helps in\n{helps_pct:.1f}% of samples',
                       xy=(max(all_improvements) * 0.6, axes[1, 1].get_ylim()[1] * 0.8),
                       fontsize=11, ha='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    axes[1, 1].set_xlabel('MSE Improvement from Emotion (%)', fontsize=12)
    axes[1, 1].set_ylabel('Number of Samples', fontsize=12)
    axes[1, 1].set_title('(d) Distribution of Emotion Contribution', fontsize=13)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'emotion_detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'emotion_detailed_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'emotion_detailed_analysis.png'}")



def visualise_sample_comparison(sample_id: str, heldout_dir: str, ablation_dir: str, output_dir: str, fps: float = 30.0):
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualisation")
        return
    
    heldout_path = Path(heldout_dir) / f'{sample_id}.npz'
    if not heldout_path.exists():
        print(f"Sample not found: {heldout_path}")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load target
    target_data = np.load(heldout_path, allow_pickle=True)
    target = target_data['flame_params'].astype(np.float32)
    emotions = target_data.get('emotions', np.zeros(10))
    text = str(target_data.get('text', ''))
    
    # Load reconstructions
    conditions = ['text_only', 'text_emotion', 'text_nmf', 'full_model']
    reconstructions = {}
    
    for cond in conditions:
        recon_path = Path(ablation_dir) / cond / 'reconstructions' / f'{sample_id}_recon.npz'
        if recon_path.exists():
            reconstructions[cond] = np.load(recon_path)['flame_params'].astype(np.float32)
    
    if not reconstructions:
        print("No reconstructions found")
        return
    
    # Align lengths
    T = min(len(target), *[len(r) for r in reconstructions.values()])
    target = target[:T]
    reconstructions = {k: v[:T] for k, v in reconstructions.items()}
    
    t = np.arange(T) / fps
    
    # Create figure
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Title
    emotion_str = ", ".join([f"{EMOTION_NAMES[i]}={emotions[i]:.2f}" 
                            for i in range(len(emotions)) if emotions[i] > 0.1])
    fig.suptitle(f'Sample: {sample_id}\nText: "{text[:60]}..."\nEmotions: {emotion_str}', 
                fontsize=12, y=0.98)
    
    # Row 1: Key parameter time series
    params_to_plot = [
        ('Jaw Open (expr 0)', 0),
        ('Smile L (expr 10)', 10),
        ('Brow Up L (expr 30)', 30),
    ]
    
    for col, (name, idx) in enumerate(params_to_plot):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, target[:, idx], 'k-', linewidth=2, label='Target', alpha=0.8)
        
        for cond, pred in reconstructions.items():
            ax.plot(t, pred[:, idx], '--', color=CONDITION_COLORS[cond], 
                   linewidth=1.5, label=cond.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(name)
        ax.set_title(name)
        if col == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Row 2: Per-frame MSE
    ax = fig.add_subplot(gs[1, :])
    
    for cond, pred in reconstructions.items():
        frame_mse = np.mean((pred - target) ** 2, axis=1)
        ax.plot(t, frame_mse, color=CONDITION_COLORS[cond], linewidth=1.5, 
               label=cond.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Per-Frame MSE')
    ax.set_title('Per-Frame Reconstruction Error')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    
    # Row 3: Velocity comparison
    ax = fig.add_subplot(gs[2, :])
    
    vel_target = np.linalg.norm(np.diff(target, axis=0), axis=1)
    ax.plot(t[1:], vel_target, 'k-', linewidth=2, label='Target', alpha=0.8)
    
    for cond, pred in reconstructions.items():
        vel_pred = np.linalg.norm(np.diff(pred, axis=0), axis=1)
        ax.plot(t[1:], vel_pred, '--', color=CONDITION_COLORS[cond], 
               linewidth=1.5, label=cond.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (L2 norm)')
    ax.set_title('Motion Velocity')
    ax.legend(loc='upper right')
    
    # Row 4: Cumulative error
    ax = fig.add_subplot(gs[3, :])
    
    for cond, pred in reconstructions.items():
        frame_mse = np.mean((pred - target) ** 2, axis=1)
        cumulative = np.cumsum(frame_mse)
        ax.plot(t, cumulative, color=CONDITION_COLORS[cond], linewidth=2,
               label=f'{cond.replace("_", " ").title()} (final: {cumulative[-1]:.4f})', alpha=0.8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative MSE')
    ax.set_title('Cumulative Reconstruction Error Over Time')
    ax.legend(loc='upper left')
    
    plt.savefig(output_dir / f'{sample_id}_detailed_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'{sample_id}_detailed_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / f'{sample_id}_detailed_comparison.png'}")

def main():
    parser = argparse.ArgumentParser(description="Detailed Ablation Analysis")
    subparsers = parser.add_subparsers(dest='command')
    
    # Analyse emotion contribution
    analyse_p = subparsers.add_parser('analyse', help='Analyse where emotion helps')
    analyse_p.add_argument('--heldout_dir', required=True)
    analyse_p.add_argument('--ablation_dir', required=True)
    analyse_p.add_argument('--output_dir', default='detailed_analysis')
    
    # Visualise specific sample
    vis_p = subparsers.add_parser('visualise', help='Visualise single sample')
    vis_p.add_argument('--sample_id', required=True)
    vis_p.add_argument('--heldout_dir', required=True)
    vis_p.add_argument('--ablation_dir', required=True)
    vis_p.add_argument('--output_dir', default='sample_visualisations')
    
    args = parser.parse_args()
    
    if args.command == 'analyse':
        analyse_where_emotion_helps(args.heldout_dir, args.ablation_dir, args.output_dir)
    elif args.command == 'visualise':
        visualise_sample_comparison(args.sample_id, args.heldout_dir, args.ablation_dir, args.output_dir)
    else:
        parser.print_help()
if __name__ == '__main__':
    main()
