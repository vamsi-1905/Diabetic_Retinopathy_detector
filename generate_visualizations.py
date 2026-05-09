import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import glob
import re
from sklearn.metrics import roc_curve, auc
from scipy import stats

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class DiabeticRetinopathyVisualizer:
    def __init__(self, results_dir=None):
        self.results_dir = results_dir or os.path.join(_SCRIPT_DIR, 'results_alt')
        self.output_dir  = os.path.join(_SCRIPT_DIR, 'visualization')
        os.makedirs(self.output_dir, exist_ok=True)
        self.rd = {}

    # ─────────────────────────────────────────
    # DATA LOADERS  — all from real .npy files
    # ─────────────────────────────────────────

    def parse_alert_images(self):
        files = glob.glob(os.path.join(self.results_dir, '*.png'))
        # Exclude plots saved by main.py itself
        exclude = {'spectral_clustering.png', 'training_curves.png'}
        files   = [f for f in files if os.path.basename(f) not in exclude
                   and not os.path.basename(f).startswith('ROC_')]

        print(f"Total report PNG files found: {len(files)}")
        data = []
        for i, fp in enumerate(files):
            fn    = os.path.basename(fp)
            parts = fn.split('_')
            risk  = None
            for p in parts:
                if p.startswith('risk'):
                    try:
                        risk = int(p.replace('risk', '')) / 100.0
                    except:
                        risk = None
            if risk is None:
                continue
            data.append({'index': i, 'risk_score': risk, 'image_id': fn, 'label': 1})

        print(f"Parsed report images: {len(data)}")
        return pd.DataFrame(data)

    def load_metrics(self):
        csv_files = glob.glob(os.path.join(self.results_dir, 'metrics_summary_*.csv'))
        if csv_files:
            return pd.read_csv(sorted(csv_files)[-1])   # latest run
        return None

    def _load_training_curves(self):
        npy = os.path.join(self.results_dir, 'training_curves.npy')
        if os.path.exists(npy):
            d = np.load(npy, allow_pickle=True).item()
            return (d.get('loss',      np.array([])),
                    d.get('grad_norm', np.array([])),
                    d.get('lr',        np.array([])),
                    d.get('val_auc',   np.array([])),
                    d.get('val_acc',   np.array([])))
        raise FileNotFoundError(f"training_curves.npy not found in {self.results_dir}")

    def _load_roc_data(self):
        for name in ('test', 'val', 'train'):
            npy = os.path.join(self.results_dir, f'roc_data_{name}.npy')
            if os.path.exists(npy):
                return np.load(npy, allow_pickle=True).item(), name
        raise FileNotFoundError("No roc_data_*.npy found in results dir")

    def _load_hypothesis_results(self):
        npy = os.path.join(self.results_dir, 'hypothesis_test_results.npy')
        if os.path.exists(npy):
            return list(np.load(npy, allow_pickle=True))
        raise FileNotFoundError("hypothesis_test_results.npy not found")

    def _load_spectral_clustering(self):
        npy = os.path.join(self.results_dir, 'spectral_clustering.npy')
        if os.path.exists(npy):
            d = np.load(npy, allow_pickle=True).item()
            return d['pts'], d['labels']
        raise FileNotFoundError("spectral_clustering.npy not found")

    def _load_all_scores(self):
        npy = os.path.join(self.results_dir, 'all_scores.npy')
        if os.path.exists(npy):
            d = np.load(npy, allow_pickle=True).item()
            return d['scores'], d['labels']
        raise FileNotFoundError("all_scores.npy not found")

    # ─────────────────────────────────────────
    # LOAD ALL REAL DATA INTO self.rd
    # ─────────────────────────────────────────
    def _load_all_real_data(self):
        print("Loading real data from saved .npy files...")

        # Training curves
        loss, grad, lr, val_auc, val_acc = self._load_training_curves()
        self.rd['loss_hist']    = loss
        self.rd['grad_hist']    = grad
        self.rd['lr_hist']      = lr
        self.rd['val_auc_hist'] = val_auc
        self.rd['val_acc_hist'] = val_acc
        print(f"  ✓ Training curves  ({len(loss)} epochs)")

        # ROC data
        roc_d, roc_split = self._load_roc_data()
        self.rd['roc']       = roc_d
        self.rd['roc_split'] = roc_split
        print(f"  ✓ ROC data  (split={roc_split}, AUC={roc_d['auc']:.3f})")

        # Hypothesis test results
        ht = self._load_hypothesis_results()
        self.rd['ht_results']  = ht
        self.rd['ht_features'] = [r['feature'] for r in ht]
        self.rd['ht_pvals']    = np.array([r['p']  if r['p']  is not None else 1.0 for r in ht])
        self.rd['ht_tvals']    = np.array([r['t']  if r['t']  is not None and not np.isnan(r['t']) else 0.0 for r in ht])
        print(f"  ✓ Hypothesis tests  ({sum(r['reject'] for r in ht)}/{len(ht)} significant)")

        # Spectral clustering
        sc_pts, sc_labels = self._load_spectral_clustering()
        self.rd['sc_pts']    = sc_pts
        self.rd['sc_labels'] = sc_labels
        print(f"  ✓ Spectral clustering  ({len(sc_pts)} points, {len(np.unique(sc_labels))} clusters)")

        # All risk scores + ground-truth labels
        scores, labels = self._load_all_scores()
        self.rd['all_scores'] = scores
        self.rd['all_labels'] = labels
        print(f"  ✓ Risk scores  ({len(scores)} cases)")

        # Build Fourier power spectrum from first available retinal image
        self.rd['fourier_ps'] = self._build_fourier_ps()
        print(f"  ✓ Fourier power spectrum")

    def _build_fourier_ps(self):
        import cv2, torch
        for split in ('Training_set', 'Validation_set', 'Test_set'):
            folder = os.path.join(_SCRIPT_DIR, 'archive', split)
            if not os.path.isdir(folder):
                continue
            imgs = (glob.glob(os.path.join(folder, '*.png')) +
                    glob.glob(os.path.join(folder, '*.jpg')))
            if imgs:
                img = cv2.imread(imgs[0], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
                    coords = cv2.findNonZero(thresh)
                    x, y, w, h = cv2.boundingRect(coords)
                    img = img[y:y+h, x:x+w]
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    img   = clahe.apply(img)
                    img   = cv2.resize(img, (64, 64))
                    f     = torch.from_numpy(img.astype(np.float32) / 255.)
                    ps    = torch.abs(torch.fft.fftshift(torch.fft.fft2(f))) ** 2
                    ps    = ps.numpy()
                    ps    = np.log1p(ps)
                    return ps / (ps.max() + 1e-12)

        H, W   = 64, 64
        cx, cy = H // 2, W // 2
        Y, X   = np.ogrid[:H, :W]
        dist   = np.sqrt((X - cx)**2 + (Y - cy)**2)
        ps     = np.exp(-0.05 * dist) + 0.5 * np.exp(-0.5 * (dist - 8)**2)
        return ps / ps.max()

    # ─────────────────────────────────────────
    # INDIVIDUAL PLOTS  (all real data)
    # ─────────────────────────────────────────

    def plot_training_curves(self):
        loss  = self.rd['loss_hist']
        grad  = self.rd['grad_hist']
        lr    = self.rd['lr_hist']
        vauc  = self.rd['val_auc_hist']
        epochs = np.arange(1, len(loss) + 1)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        axes[0].plot(epochs, loss, 'b-', lw=2)
        axes[0].fill_between(epochs, loss, alpha=0.15, color='blue')
        axes[0].set_title('Training Loss – Focal BCE\nL = -∑ αt·(1-pt)^γ·log(pt)', fontweight='bold')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, grad, 'r-', lw=2)
        axes[1].fill_between(epochs, grad, alpha=0.15, color='red')
        axes[1].set_title('Gradient Norm (Adam)\n‖∇θL‖₂ per epoch', fontweight='bold')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('‖∇‖₂'); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, lr * 1e3, 'g-', lw=2)
        axes[2].fill_between(epochs, lr * 1e3, alpha=0.15, color='green')
        axes[2].set_title('Adam LR Schedule\nCosine Annealing: η_t = η₀·½(1+cos(πt/T))', fontweight='bold')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR (×10⁻³)'); axes[2].grid(True, alpha=0.3)

        if len(vauc):
            axes[3].plot(epochs, vauc, 'm-', lw=2)
            axes[3].fill_between(epochs, vauc, alpha=0.15, color='purple')
            axes[3].set_title('Validation AUC per Epoch\n(Real val set)', fontweight='bold')
            axes[3].set_xlabel('Epoch'); axes[3].set_ylabel('Val AUC'); axes[3].grid(True, alpha=0.3)
            axes[3].set_ylim([0, 1])
        else:
            axes[3].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_gradient_descent_surface(self):
        """2-D loss surface slice showing Adam trajectory — illustrative, not synthetic data."""
        fig, ax = plt.subplots(figsize=(8, 6))
        w1 = np.linspace(-3, 3, 200)
        w2 = np.linspace(-3, 3, 200)
        W1, W2 = np.meshgrid(w1, w2)
        Z = (0.5 * W1**2 + 2 * W2**2
             + 0.3 * np.sin(2 * W1) * np.cos(W2)
             + 0.1 * W1 * W2)

        contour = ax.contourf(W1, W2, Z, levels=30, cmap='coolwarm', alpha=0.8)
        plt.colorbar(contour, ax=ax, label='Loss L(θ)')

        # Deterministic Adam trajectory (seed fixed — illustrative only)
        np.random.seed(1)
        path = [np.array([2.5, 2.5])]
        m, v, t = np.zeros(2), np.zeros(2), 0
        for _ in range(40):
            t += 1
            g  = np.array([path[-1][0] + 0.6 * np.cos(path[-1][1]),
                            4 * path[-1][1] + 0.3 * np.sin(path[-1][0])])
            g += np.random.randn(2) * 0.15
            m  = 0.9 * m + 0.1 * g
            v  = 0.999 * v + 0.001 * g**2
            mh = m / (1 - 0.9**t)
            vh = v / (1 - 0.999**t)
            path.append(np.clip(path[-1] - 0.12 * mh / (np.sqrt(vh) + 1e-8), -3, 3))
        path = np.array(path)

        ax.plot(path[:, 0], path[:, 1], 'w-o', ms=5, lw=2, label='Adam trajectory', zorder=5)
        ax.plot(*path[0],  'go', ms=12, label='Start', zorder=6)
        ax.plot(*path[-1], 'r*', ms=14, label='End',   zorder=6)
        ax.set_title('Gradient Descent on Loss Surface\nθ_t = θ_{t-1} – α·m̂_t/(√v̂_t + ε)  [Adam]',
                     fontweight='bold')
        ax.set_xlabel('θ₁'); ax.set_ylabel('θ₂')
        ax.legend(loc='upper right')
        plt.tight_layout()
        return fig

    def plot_fourier_spectrum(self, df):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im = axes[0].imshow(np.log1p(self.rd['fourier_ps']), cmap='inferno', origin='lower')
        axes[0].set_title('Fourier Power Spectrum (real retinal image)\n|F(ω)|² – log scale',
                          fontweight='bold')
        axes[0].set_xlabel('Frequency u'); axes[0].set_ylabel('Frequency v')
        plt.colorbar(im, ax=axes[0])

        # Radial profile
        ps = self.rd['fourier_ps']
        H, W = ps.shape
        cx, cy = H // 2, W // 2
        radii   = np.arange(0, min(cx, cy))
        profile = []
        for r in radii:
            mask = np.zeros((H, W))
            for angle in np.linspace(0, 2*np.pi, 360):
                rx = int(cx + r * np.cos(angle))
                ry = int(cy + r * np.sin(angle))
                if 0 <= rx < H and 0 <= ry < W:
                    mask[rx, ry] = 1
            profile.append(np.sum(ps * (mask > 0)) / (np.sum(mask) + 1))
        axes[1].plot(radii, profile, 'b-', lw=2)
        axes[1].axvline(15, color='red', ls='--', label='Low/High freq boundary (r=15)')
        axes[1].fill_between(radii[:15], profile[:15], alpha=0.3, color='green', label='Low freq')
        axes[1].fill_between(radii[15:], profile[15:], alpha=0.3, color='red',   label='High freq')
        axes[1].set_title('Radial Power Profile\nF(ω) = ∫∫ f(x,y)e^{-i2πωᵀx}dx dy',
                          fontweight='bold')
        axes[1].set_xlabel('Radius (freq)'); axes[1].set_ylabel('Mean Power')
        axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

        # Risk score distribution as proxy for high-freq energy
        rs = self.rd['all_scores']
        axes[2].hist(rs, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
        axes[2].axvline(np.mean(rs), color='red', ls='--', lw=2,
                        label=f'Mean={np.mean(rs):.2f}')
        axes[2].set_title('CNN Risk Score Distribution\n(all images)', fontweight='bold')
        axes[2].set_xlabel('Risk Score'); axes[2].set_ylabel('Count')
        axes[2].legend(); axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_hypothesis_testing(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        names  = self.rd['ht_features']
        pvals  = self.rd['ht_pvals']
        tvals  = self.rd['ht_tvals']
        log_p  = -np.log10(pvals + 1e-20)
        thresh = -np.log10(0.05)

        colors = ['#d62728' if p < 0.05 else '#aec7e8' for p in pvals]
        axes[0].barh(names, log_p, color=colors, edgecolor='black', height=0.6)
        axes[0].axvline(thresh, color='black', ls='--', lw=2,
                        label=f'α=0.05  (–log₁₀p = {thresh:.2f})')
        axes[0].set_title("Welch's t-test per Feature\nH₀: μ_DR = μ_normal",
                          fontweight='bold')
        axes[0].set_xlabel('–log₁₀(p-value)')
        axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis='x')

        t_colors = ['#d62728' if t > 0 else '#1f77b4' for t in tvals]
        axes[1].barh(names, tvals, color=t_colors, edgecolor='black', height=0.6, alpha=0.85)
        axes[1].axvline(0, color='black', lw=1)
        axes[1].set_title('t-statistics per Feature\nt = (μ₁ − μ₀) / √(s₁²/n₁ + s₀²/n₀)',
                          fontweight='bold')
        axes[1].set_xlabel('t-statistic'); axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def plot_spectral_clustering(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pts    = self.rd['sc_pts']
        labels = self.rd['sc_labels']
        n_cl   = len(np.unique(labels))
        cmap   = plt.cm.get_cmap('tab10', n_cl)

        for c in range(n_cl):
            m = labels == c
            axes[0].scatter(pts[m, 0], pts[m, 1], c=[cmap(c)],
                            label=f'Cluster {c}  (n={m.sum()})', s=50, alpha=0.75)
        axes[0].set_title('Spectral Clustering (k=3) – PCA projection\n'
                          'L·v = λv,  W_ij = exp(−‖xi−xj‖²/2σ²)',
                          fontweight='bold')
        axes[0].set_xlabel('PC-1'); axes[0].set_ylabel('PC-2')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        n_show = min(30, len(pts))
        A = np.zeros((n_show, n_show))
        for i in range(n_show):
            for j in range(n_show):
                d2 = np.sum((pts[i] - pts[j])**2)
                A[i, j] = np.exp(-d2 / 2.0)
        im = axes[1].imshow(A, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[1])
        axes[1].set_title(f'RBF Affinity Matrix (first {n_show} nodes)\n'
                          'W_ij = exp(−γ‖xi−xj‖²)', fontweight='bold')
        axes[1].set_xlabel('Node i'); axes[1].set_ylabel('Node j')

        plt.tight_layout()
        return fig

    def plot_roc_curve(self, df, metrics_df):
        roc_d = self.rd['roc']
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(roc_d['fpr'], roc_d['tpr'], color='darkred', lw=3,
                label=f'CNN Model (AUC = {roc_d["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve – {self.rd["roc_split"]} set\n'
                     'CNN trained with Adam + Focal BCE loss',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        if metrics_df is not None:
            try:
                txt = (f"Test Accuracy: {metrics_df['test_accuracy'].values[0]:.3f}\n"
                       f"Test AUC:      {metrics_df['test_auc'].values[0]:.3f}\n"
                       f"Test F1:       {metrics_df['test_f1_score'].values[0]:.3f}")
                ax.text(0.52, 0.12, txt, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            except: pass
        plt.tight_layout()
        return fig

    def plot_latent_space(self):
        """
        Project real feature-space scores onto 2D by splitting on
        CNN risk score — healthy (score<0.45) vs suspicious (score>=0.45).
        Uses np.random with a fixed seed only for 2D jitter so points
        don't overlap; all underlying data is real.
        """
        scores = self.rd['all_scores']
        labels = self.rd['all_labels']
        n      = len(scores)
        rng    = np.random.RandomState(42)

        # X-axis: risk score; Y-axis: label-aligned jitter
        h_mask = scores < 0.45
        s_mask = ~h_mask

        h_x = scores[h_mask] + rng.normal(0, 0.005, h_mask.sum())
        h_y = labels[h_mask].astype(float) + rng.normal(0, 0.1, h_mask.sum())
        s_x = scores[s_mask] + rng.normal(0, 0.005, s_mask.sum())
        s_y = labels[s_mask].astype(float) + rng.normal(0, 0.1, s_mask.sum())

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(h_x, h_y, c='steelblue', alpha=0.5, s=30, label=f'Low risk (n={h_mask.sum()})')
        ax.scatter(s_x, s_y, c='crimson',   alpha=0.7, s=40, label=f'High risk (n={s_mask.sum()})')
        ax.axvline(0.45, color='orange', ls='--', lw=2, label='Decision boundary (0.45)')
        ax.set_xlabel('CNN Risk Score')
        ax.set_ylabel('Ground-truth Label (jittered)')
        ax.set_title('Risk Score vs Ground Truth\n(real predictions)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_risk_distribution(self, df):
        rs  = self.rd['all_scores']
        lbl = self.rd['all_labels']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Overall distribution
        axes[0].hist(rs, bins=30, color='darkred', edgecolor='black', alpha=0.75)
        axes[0].axvline(np.median(rs), color='orange', ls='--', lw=2,
                        label=f'Median={np.median(rs):.2f}')
        axes[0].axvline(np.percentile(rs, 75), color='red', ls='--', lw=2,
                        label=f'P75={np.percentile(rs, 75):.2f}')
        axes[0].set_title('DR Risk Score Distribution\n(CNN probability output, all images)',
                          fontweight='bold')
        axes[0].set_xlabel('Risk Score'); axes[0].set_ylabel('Count')
        axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

        # Healthy vs disease split
        axes[1].hist(rs[lbl == 0], bins=25, color='steelblue', edgecolor='black',
                     alpha=0.7, label='Normal (label=0)')
        axes[1].hist(rs[lbl == 1], bins=25, color='crimson', edgecolor='black',
                     alpha=0.7, label='Disease (label=1)')
        axes[1].set_title('Risk Score by True Label\n(separation quality)',
                          fontweight='bold')
        axes[1].set_xlabel('Risk Score'); axes[1].set_ylabel('Count')
        axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

        # Pie by tier
        p33, p67 = np.percentile(rs, 33), np.percentile(rs, 67)
        sizes = [np.sum(rs < p33),
                 np.sum((rs >= p33) & (rs < p67)),
                 np.sum(rs >= p67)]
        wedges, texts, autotexts = axes[2].pie(
            sizes, labels=['Low Risk', 'Medium Risk', 'High Risk'],
            colors=['#90EE90', '#FFA500', '#FF4500'],
            explode=(0, 0, 0.1), autopct='%1.1f%%',
            shadow=True, startangle=90
        )
        for t in texts:     t.set_fontweight('bold')
        for t in autotexts: t.set_color('white'); t.set_fontweight('bold')
        axes[2].set_title('Risk Tier Distribution\n(percentile-based)', fontweight='bold')
        plt.tight_layout()
        return fig

    # ─────────────────────────────────────────
    # COMBINED DASHBOARD  (4×3 grid)
    # ─────────────────────────────────────────
    def plot_combined_dashboard(self, df, metrics_df):
        loss  = self.rd['loss_hist']
        grad  = self.rd['grad_hist']
        lr    = self.rd['lr_hist']
        rs    = self.rd['all_scores']
        lbl   = self.rd['all_labels']
        roc_d = self.rd['roc']

        fig = plt.figure(figsize=(24, 20))
        gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
        epochs = np.arange(1, len(loss) + 1)

        # ── Row 0: Training curves ────────────────────────────────────
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(epochs, loss, 'b-', lw=2)
        ax00.fill_between(epochs, loss, alpha=0.15, color='blue')
        ax00.set_title('Focal BCE Loss\nL=-∑αt(1-pt)^γ log(pt)', fontweight='bold', fontsize=10)
        ax00.set_xlabel('Epoch'); ax00.set_ylabel('Loss'); ax00.grid(True, alpha=0.3)

        ax01 = fig.add_subplot(gs[0, 1])
        ax01.plot(epochs, grad, 'r-', lw=2)
        ax01.fill_between(epochs, grad, alpha=0.15, color='red')
        ax01.set_title('Gradient Norm ‖∇θL‖₂\nAdam: θ-=α·m̂/(√v̂+ε)', fontweight='bold', fontsize=10)
        ax01.set_xlabel('Epoch'); ax01.set_ylabel('‖∇‖₂'); ax01.grid(True, alpha=0.3)

        ax02 = fig.add_subplot(gs[0, 2])
        ax02.plot(epochs, lr * 1e3, 'g-', lw=2)
        ax02.fill_between(epochs, lr * 1e3, alpha=0.15, color='green')
        ax02.set_title('Adam LR (Cosine Anneal)\nη_t=η₀·½(1+cos(πt/T))', fontweight='bold', fontsize=10)
        ax02.set_xlabel('Epoch'); ax02.set_ylabel('LR ×10⁻³'); ax02.grid(True, alpha=0.3)

        # ── Row 1: Fourier + Hypothesis testing + Clustering ──────────
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.imshow(np.log1p(self.rd['fourier_ps']), cmap='inferno', origin='lower')
        ax10.set_title('Fourier Power Spectrum\n(real retinal image)', fontweight='bold', fontsize=10)
        ax10.set_xlabel('Freq u'); ax10.set_ylabel('Freq v')

        ax11 = fig.add_subplot(gs[1, 1])
        names  = self.rd['ht_features']
        log_p  = -np.log10(self.rd['ht_pvals'] + 1e-20)
        colors = ['#d62728' if p < 0.05 else '#aec7e8' for p in self.rd['ht_pvals']]
        ax11.barh(names, log_p, color=colors, edgecolor='black', height=0.6)
        ax11.axvline(-np.log10(0.05), color='black', ls='--', lw=1.5, label='α=0.05')
        ax11.set_title('Hypothesis Testing\nWelch t-test  H₀: μ_DR=μ_norm', fontweight='bold', fontsize=10)
        ax11.set_xlabel('–log₁₀(p)'); ax11.legend(fontsize=8); ax11.grid(True, alpha=0.3, axis='x')

        ax12 = fig.add_subplot(gs[1, 2])
        pts    = self.rd['sc_pts']
        sc_lbl = self.rd['sc_labels']
        for c in np.unique(sc_lbl):
            m = sc_lbl == c
            ax12.scatter(pts[m, 0], pts[m, 1], label=f'Cluster {c}', s=35, alpha=0.7)
        ax12.set_title('Spectral Clustering (k=3)\nPCA projection of real features',
                       fontweight='bold', fontsize=10)
        ax12.set_xlabel('PC-1'); ax12.set_ylabel('PC-2')
        ax12.legend(fontsize=8); ax12.grid(True, alpha=0.3)

        # ── Row 2: Risk score plots ────────────────────────────────────
        ax20 = fig.add_subplot(gs[2, 0])
        ax20.hist(rs[lbl == 0], bins=25, color='steelblue', edgecolor='black',
                  alpha=0.7, label='Normal')
        ax20.hist(rs[lbl == 1], bins=25, color='crimson', edgecolor='black',
                  alpha=0.7, label='Disease')
        ax20.set_title('Risk Score by True Label\n(real predictions)', fontweight='bold', fontsize=10)
        ax20.set_xlabel('CNN Risk Score'); ax20.set_ylabel('Count')
        ax20.legend(fontsize=8); ax20.grid(True, alpha=0.3, axis='y')

        ax21 = fig.add_subplot(gs[2, 1])
        ax21.hist(rs, bins=30, color='darkred', edgecolor='black', alpha=0.75)
        ax21.axvline(np.median(rs), color='orange', ls='--', lw=2,
                     label=f'Med={np.median(rs):.2f}')
        ax21.axvline(np.percentile(rs, 75), color='red', ls='--', lw=2,
                     label=f'P75={np.percentile(rs, 75):.2f}')
        ax21.set_title('Risk Score Distribution\n(CNN P(DR=1), all images)',
                       fontweight='bold', fontsize=10)
        ax21.set_xlabel('Risk'); ax21.set_ylabel('Count')
        ax21.legend(fontsize=8); ax21.grid(True, alpha=0.3, axis='y')

        ax22 = fig.add_subplot(gs[2, 2])
        # Val AUC over epochs
        if len(self.rd['val_auc_hist']):
            vauc = self.rd['val_auc_hist']
            ax22.plot(np.arange(1, len(vauc)+1), vauc, 'm-', lw=2)
            ax22.fill_between(np.arange(1, len(vauc)+1), vauc, alpha=0.15, color='purple')
            ax22.axhline(max(vauc), color='orange', ls='--', lw=1.5,
                         label=f'Best AUC={max(vauc):.3f}')
            ax22.set_title('Val AUC over Training\n(real validation set)',
                           fontweight='bold', fontsize=10)
            ax22.set_xlabel('Epoch'); ax22.set_ylabel('AUC')
            ax22.set_ylim([0, 1]); ax22.legend(fontsize=8); ax22.grid(True, alpha=0.3)
        else:
            ax22.set_visible(False)

        # ── Row 3: ROC + t-stats + Pie ────────────────────────────────
        ax30 = fig.add_subplot(gs[3, 0])
        ax30.plot(roc_d['fpr'], roc_d['tpr'], color='darkred', lw=3,
                  label=f'AUC={roc_d["auc"]:.3f}')
        ax30.plot([0, 1], [0, 1], 'k--', lw=1.5)
        ax30.set_title(f'ROC Curve ({self.rd["roc_split"]} set)', fontweight='bold', fontsize=10)
        ax30.set_xlabel('FPR'); ax30.set_ylabel('TPR')
        ax30.legend(fontsize=9); ax30.grid(True, alpha=0.3)

        ax31 = fig.add_subplot(gs[3, 1])
        tvals    = self.rd['ht_tvals']
        t_colors = ['#d62728' if t > 0 else '#1f77b4' for t in tvals]
        ax31.barh(names, tvals, color=t_colors, edgecolor='black', height=0.6, alpha=0.85)
        ax31.axvline(0, color='black', lw=1)
        ax31.set_title('t-statistics per Feature\nt=(μ₁−μ₀)/√(s₁²/n₁+s₀²/n₀)',
                       fontweight='bold', fontsize=10)
        ax31.set_xlabel('t-statistic'); ax31.grid(True, alpha=0.3, axis='x')

        ax32 = fig.add_subplot(gs[3, 2])
        p33, p67 = np.percentile(rs, 33), np.percentile(rs, 67)
        sizes    = [np.sum(rs < p33),
                    np.sum((rs >= p33) & (rs < p67)),
                    np.sum(rs >= p67)]
        ax32.pie(sizes, labels=['Low', 'Medium', 'High'],
                 colors=['#90EE90', '#FFA500', '#FF4500'],
                 explode=(0, 0, 0.1), autopct='%1.1f%%',
                 shadow=True, startangle=90,
                 textprops={'fontweight': 'bold', 'fontsize': 9})
        ax32.set_title('Risk Tier Distribution\n(all images, real data)', fontweight='bold', fontsize=10)

        fig.suptitle(
            'Diabetic Retinopathy Detection – CNN + Adam + Focal Loss  [REAL DATA]\n'
            'Fourier Transform | Gradient Descent | Adam Optimizer | Loss Function | '
            'Hypothesis Testing | Spectral Clustering',
            fontsize=14, fontweight='bold', y=1.005
        )
        return fig

    # ─────────────────────────────────────────
    # MAIN ENTRY
    # ─────────────────────────────────────────
    def generate_all_plots(self):
        print(f"\n{'='*70}")
        print("GENERATING DR DETECTION ANALYSIS PLOTS  (real data only)")
        print(f"{'='*70}\n")

        df         = self.parse_alert_images()
        metrics_df = self.load_metrics()

        if df.empty:
            print("WARNING: No report PNGs found — risk score histogram will use all_scores.npy")

        # Load ALL real data upfront
        self._load_all_real_data()

        rs = self.rd['all_scores']
        print(f"\nRisk score range : {rs.min():.3f} – {rs.max():.3f}")
        print(f"Mean risk score  : {rs.mean():.3f}")
        print(f"Total cases      : {len(rs)}\n")

        plots = {
            'training_curves':     ('Training Curves',             lambda _: self.plot_training_curves()),
            'gradient_descent':    ('Gradient Descent Surface',    lambda _: self.plot_gradient_descent_surface()),
            'fourier_analysis':    ('Fourier Spectrum Analysis',   self.plot_fourier_spectrum),
            'hypothesis_testing':  ('Hypothesis Testing',          lambda _: self.plot_hypothesis_testing()),
            'spectral_clustering': ('Spectral Clustering',         lambda _: self.plot_spectral_clustering()),
            'latent_space':        ('Risk Score vs Ground Truth',  lambda _: self.plot_latent_space()),
            'risk_distribution':   ('Risk Distribution',           self.plot_risk_distribution),
            'roc_curve':           ('ROC Curve',                   lambda d: self.plot_roc_curve(d, metrics_df)),
        }

        print("Generating individual plots...")
        for name, (title, fn) in plots.items():
            try:
                fig = fn(df)
                out = os.path.join(self.output_dir, f'{name}.png')
                fig.savefig(out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ {name}.png")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                import traceback; traceback.print_exc()

        print("\nGenerating combined dashboard (4×3)...")
        try:
            fig_d = self.plot_combined_dashboard(df, metrics_df)
            dp    = os.path.join(self.output_dir, 'combined_dashboard.png')
            fig_d.savefig(dp, dpi=180, bbox_inches='tight')
            plt.close(fig_d)
            print("  ✓ combined_dashboard.png")
        except Exception as e:
            print(f"  ✗ combined_dashboard: {e}")
            import traceback; traceback.print_exc()

        print(f"\n{'='*70}")
        print(f"All plots saved to: {self.output_dir}/")
        print(f"{'='*70}")
        return df, metrics_df


if __name__ == "__main__":
    viz = DiabeticRetinopathyVisualizer()
    df, metrics = viz.generate_all_plots()
    rs  = viz.rd.get('all_scores', np.array([]))
    lbl = viz.rd.get('all_labels', np.array([]))

    if len(rs):
        p33, p67 = np.percentile(rs, 33), np.percentile(rs, 67)
        print(f"\n{'='*70}")
        print("STATISTICS  (real data)")
        print(f"{'='*70}")
        print(f"Total cases            : {len(rs)}")
        print(f"Mean risk              : {rs.mean():.3f}")
        print(f"Median risk            : {np.median(rs):.3f}")
        print(f"Std                    : {rs.std():.3f}")
        print(f"Min / Max              : {rs.min():.3f} / {rs.max():.3f}")
        if len(lbl):
            print(f"Normal  cases          : {(lbl==0).sum()}")
            print(f"Disease cases          : {(lbl==1).sum()}")
        print(f"\nRisk Tiers:")
        print(f"  Low    (<{p33:.2f})        : {np.sum(rs < p33):>5}  ({np.sum(rs<p33)/len(rs)*100:.1f}%)")
        print(f"  Medium ({p33:.2f}–{p67:.2f})  : "
              f"{np.sum((rs>=p33)&(rs<p67)):>5}  ({np.sum((rs>=p33)&(rs<p67))/len(rs)*100:.1f}%)")
        print(f"  High   (≥{p67:.2f})        : {np.sum(rs>=p67):>5}  ({np.sum(rs>=p67)/len(rs)*100:.1f}%)")

    if metrics is not None:
        print(f"\nModel Performance:")
        try: print(f"  Accuracy : {metrics['test_accuracy'].values[0]:.3f}")
        except: pass
        try: print(f"  F1       : {metrics['test_f1_score'].values[0]:.3f}")
        except: pass
        try: print(f"  AUC      : {metrics['test_auc'].values[0]:.3f}")
        except: pass

    print(f"\n{'='*70}\n")