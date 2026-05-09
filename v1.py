import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, f1_score
from sklearn.cluster import SpectralClustering

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'=' * 70}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"{'=' * 70}\n")


## path files
class CFG:
    BASE_DIR = 'math/archive'

    DATA_DIR_TRAIN = os.path.join(BASE_DIR, 'Training_set')
    CSV_PATH_TRAIN = os.path.join(BASE_DIR, 'Training_set', 'RFMiD_Training_Labels.csv')

    DATA_DIR_VAL   = os.path.join(BASE_DIR, 'Validation_set')
    CSV_PATH_VAL   = os.path.join(BASE_DIR, 'Validation_set', 'RFMiD_Validation_Labels.csv')

    DATA_DIR_TEST  = os.path.join(BASE_DIR, 'Test_set')
    CSV_PATH_TEST  = os.path.join(BASE_DIR, 'Test_set', 'RFMiD_Testing_Labels.csv')
    DISEASE_COLUMN = 'Disease_Risk'
    IMG_SIZE = (128, 128)

def classify_risk(risk_score):
    if risk_score < CFG.LOW_RISK_THRESHOLD:
        return "LOW"
    elif risk_score < CFG.MED_RISK_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"

   

  ## CNN params
    HIDDEN_DIM       = 128
    DROPOUT          = 0.4
    EPOCHS           = 200
    BATCH_SIZE       = 32
    LR               = 1e-3           # Adam initial lr
    WEIGHT_DECAY     = 1e-4
    MAX_TRAIN        = 300
    MAX_EVAL         = 100

    #  model thresholds
    HOSVD_RANK_THRESHOLD          = 35
    QR_CONDITION_THRESHOLD        = 100
    FOURIER_RATIO_THRESHOLD       = 2.5
    SUSPICIOUS_PROBABILITY_THRESHOLD   = 0.5
    COLLECTION_PROBABILITY_THRESHOLD   = 0.01
    LOW_RISK_THRESHOLD            = 0.2
    MED_RISK_THRESHOLD            = 0.5

    # Hypothesis testing
    ALPHA                         = 0.05   # LOS

    # Spectral clustering
    N_CLUSTERS                    = 3 # LOW,MEDIUM,HIGH RISK

    REPORT_IMG_SIZE  = (300, 300)
    OUTPUT_DIR       = 'math/results'
    os.makedirs(output_dir, exist_ok=True)



# Mathematical Formulation 
 
class MathematicalMethodsGPU:

    @staticmethod
    def finite_difference_pde_laplacian(img_gray):
        u = torch.from_numpy(img_gray.astype(np.float32) / 255.0).to(device)
        u = u.unsqueeze(0).unsqueeze(0)
        laplacian_kernel = torch.tensor([
            [0, 1, 0], [1, -4, 1], [0, 1, 0]
        ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        laplacian      = F.conv2d(u, laplacian_kernel, padding=1)
        edge_magnitude = torch.abs(laplacian)
        edge_score     = float(torch.mean(edge_magnitude).cpu())
        return {
            'laplacian':       laplacian.squeeze().cpu().numpy(),
            'edge_magnitude':  edge_magnitude.squeeze().cpu().numpy(),
            'edge_score':      edge_score,
            'formula':         '∇²u = ∂²u/∂x² + ∂²u/∂y²'
        }

    @staticmethod
    def hosvd_tensor_decomposition(img_rgb):
        img_small = cv2.resize(img_rgb, (64, 64))
        tensor    = torch.from_numpy(img_small.astype(np.float32) / 255.0).to(device)
        X1        = tensor.reshape(64, 64 * 3)
        U1, S1, V1 = torch.linalg.svd(X1, full_matrices=False)
        threshold  = 0.01 * S1[0]
        rank       = int(torch.sum(S1 > threshold).cpu())
        S1_sq      = S1 ** 2
        energy_ratio = float(torch.sum(S1_sq[:10]).cpu() / (torch.sum(S1_sq).cpu() + 1e-12))
        p            = S1_sq / (torch.sum(S1_sq) + 1e-12)
        entropy      = float(-torch.sum(p * torch.log(p + 1e-12)).cpu())
        return {
            'singular_values': S1.cpu().numpy(),
            'rank':            rank,
            'energy_ratio':    energy_ratio,
            'entropy':         entropy,
            'formula':         'X = S ×₁ U₁ ×₂ U₂ ×₃ U₃'
        }

    @staticmethod
    def qr_decomposition_texture(img_gray):
        img_small = cv2.resize(img_gray, (64, 64))
        patches   = [img_small[i:i+8, j:j+8].flatten()
                     for i in range(0, 56, 8) for j in range(0, 56, 8)]
        A         = torch.from_numpy(np.array(patches).T.astype(np.float32)).to(device)
        Q, R      = torch.linalg.qr(A)
        diag_R    = torch.abs(torch.diag(R))
        sigma_max = float(torch.max(diag_R).cpu()) if diag_R.numel() else 1.0
        sigma_min = float(torch.min(diag_R[diag_R > 1e-6]).cpu()) if torch.any(diag_R > 1e-6) else 1.0
        cond      = float(sigma_max / (sigma_min + 1e-12))
        rank      = int(torch.sum(diag_R > 1e-3).cpu())
        return {
            'condition_number': cond,
            'rank':             rank,
            'diag_R':           diag_R.cpu().numpy(),
            'formula':          'A = QR, κ(A) = σ_max/σ_min'
        }

    @staticmethod
    def multivariate_gaussian(img_rgb):
        pixels  = torch.from_numpy(img_rgb.reshape(-1, 3).astype(np.float32) / 255.0).to(device)
        n       = pixels.shape[0]
        mu      = torch.mean(pixels, dim=0)
        centered = pixels - mu
        cov     = (1.0 / max(n, 1)) * (centered.T @ centered)
        cov_reg = cov + 1e-8 * torch.eye(3, device=device)
        eigenvalues = torch.linalg.eigvalsh(cov_reg)
        det   = float(torch.linalg.det(cov_reg).cpu())
        trace = float(torch.trace(cov).cpu())
        return {
            'mean':        mu.cpu().numpy(),
            'covariance':  cov.cpu().numpy(),
            'eigenvalues': eigenvalues.cpu().numpy(),
            'determinant': det,
            'trace':       trace,
            'formula':     'p(x) = 1/√((2π)^k|Σ|) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))'
        }

    @staticmethod
    def fourier_frequency_analysis(img_gray):
        """2-D Fourier Transform – separates low/high-freq energy."""
        f   = torch.from_numpy(img_gray.astype(np.float32) / 255.0).to(device)
        F_s = torch.fft.fftshift(torch.fft.fft2(f))
        power = torch.abs(F_s) ** 2
        h, w  = f.shape
        ch, cw = h // 2, w // 2
        r     = 15
        low_freq  = float(torch.sum(power[ch-r:ch+r, cw-r:cw+r]).cpu())
        total     = float(torch.sum(power).cpu())
        high_freq = float(max(total - low_freq, 0.0))
        freq_ratio = float(high_freq / (low_freq + 1e-12))

        # Spectral entropy: how spread is the frequency energy
        p_spectrum = power / (total + 1e-12)
        spectral_entropy = float(-torch.sum(p_spectrum * torch.log(p_spectrum + 1e-20)).cpu())

        return {
            'power_spectrum':   power.cpu().numpy(),
            'low_freq_energy':  low_freq,
            'high_freq_energy': high_freq,
            'freq_ratio':       freq_ratio,
            'spectral_entropy': spectral_entropy,
            'formula':          'F(ω) = ∫∫ f(x,y)e^(-i2πωᵀx) dx dy'
        }

    @staticmethod
    def detect_lesions_morphology(img_gray):
        inverted  = 255 - img_gray
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat    = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel)
        _, ma_bin = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
        nl, _, st, _ = cv2.connectedComponentsWithStats(ma_bin, 8)
        micro_count  = sum(1 for i in range(1, nl) if 3 < st[i, cv2.CC_STAT_AREA] < 50)

        kernel_l   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        blackhat   = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel_l)
        _, hem_bin = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        nl2, _, st2, _ = cv2.connectedComponentsWithStats(hem_bin, 8)
        hem_count  = sum(1 for i in range(1, nl2) if 50 < st2[i, cv2.CC_STAT_AREA] < 500)

        _, bright   = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        kernel_exu  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        exudates    = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel_exu)
        nl3, _, st3, _ = cv2.connectedComponentsWithStats(exudates, 8)
        exu_count   = sum(1 for i in range(1, nl3) if st3[i, cv2.CC_STAT_AREA] > 20)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh   = clahe.apply(img_gray)
        sx    = cv2.Sobel(enh, cv2.CV_64F, 1, 0, ksize=3)
        sy    = cv2.Sobel(enh, cv2.CV_64F, 0, 1, ksize=3)
        vmag  = np.sqrt(sx**2 + sy**2)
        vessel_density = float(np.sum(vmag > np.percentile(vmag, 90)) / vmag.size)

        def to_mask(m):
            m = (m > 0).astype(np.uint8) * 255
            return cv2.resize(m, CFG.IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        return {
            'microaneurysms':  to_mask(ma_bin),
            'micro_count':     int(micro_count),
            'hemorrhages':     to_mask(hem_bin),
            'hem_count':       int(hem_count),
            'exudates':        to_mask(exudates),
            'exu_count':       int(exu_count),
            'vessels':         to_mask((vmag > np.percentile(vmag, 90)).astype(np.uint8) * 255),
            'vessel_density':  vessel_density
        }


## 14 fold feature extraction

def extract_features(img_rgb):
    img_rgb = cv2.resize(img_rgb, CFG.IMG_SIZE)
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    M       = MathematicalMethodsGPU()

    pde     = M.finite_difference_pde_laplacian(gray)
    hosvd   = M.hosvd_tensor_decomposition(img_rgb)
    qr_r    = M.qr_decomposition_texture(gray)
    gauss   = M.multivariate_gaussian(img_rgb)
    fourier = M.fourier_frequency_analysis(gray)   
    lesions = M.detect_lesions_morphology(gray)

    fv = np.array([
        pde['edge_score'],
        hosvd['rank'],
        hosvd['energy_ratio'],
        hosvd['entropy'],
        qr_r['condition_number'],
        qr_r['rank'],
        gauss['determinant'],
        gauss['trace'],
        fourier['freq_ratio'],
        fourier['spectral_entropy'],      ## fourier spectral transform
        lesions['micro_count'],
        lesions['hem_count'],
        lesions['exu_count'],
        lesions['vessel_density'],
    ], dtype=np.float32)

    analysis_dict = {
        'pde': pde, 'hosvd': hosvd, 'qr': qr_r,
        'gaussian': gauss, 'fourier': fourier, 'lesions': lesions
    }
    return fv, analysis_dict


FEATURE_DIM = 14



# NEURAL NETWORK Model  (MLP with BatchNorm + Dropout)
## Batchnorm = batch-normalisation to have zero standard mean and unit variance
## Dropout = deactivating neurons of the model to prevent overfitting of the model

class DRClassifierNN(nn.Module):
    """
    Feed-forward classifier for hand-crafted retinal features.
    Uses BN + dropout; trained with Adam + BCE loss.
    """
    def __init__(self, input_dim: int = FEATURE_DIM,
                 hidden_dim: int = CFG.HIDDEN_DIM,
                 dropout: float  = CFG.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),

            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # logits


class RetinalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(np.array(y)).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# LOSS FUNCTION = (weighted BCE with focal modifier)

class FocalBCELoss(nn.Module):
    """
    Binary cross-entropy with focal weighting:
        L = -αt * (1-pt)^γ * log(pt)
    Downweights easy negatives; focuses training on hard cases.
    Formula: L_focal = -∑ αt · (1-pt)^γ · log(pt)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 pos_weight: float = 1.0):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device)
        )
        pt      = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(targets, self.alpha),
                              torch.full_like(targets, 1 - self.alpha))
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss    = focal_weight * F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        return loss.mean()



# TRAINING  (Adam optimizer + gradient tracking)

def train_model(csv_path, data_dir):
    print(f"\n[PHASE 1] Feature extraction + CNN training...")
    df = pd.read_csv(csv_path)

    X_raw, y_raw = [], []
    processed = 0
    for _, row in df.iterrows():
        image_id  = str(row['ID'])
        true_label = int(row[CFG.DISEASE_COLUMN])
        img_path  = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(data_dir, image_id + ext)
            if os.path.exists(p):
                img_path = p; break
        if not img_path: continue
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            fv, _ = extract_features(img_rgb)
            X_raw.append(fv); y_raw.append(true_label); processed += 1
        except Exception as e:
            print(f"  Skip {image_id}: {e}")
        if processed % 100 == 0 and processed > 0:
            print(f"  ...{processed}/{min(CFG.MAX_TRAIN, len(df))} features extracted")
        if processed >= CFG.MAX_TRAIN:
            break

    print(f"\n  Training on {len(X_raw)} samples.")
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_raw).astype(np.float32)

    # validation set
    val_df = pd.read_csv(CFG.CSV_PATH_VAL)

    X_val_raw, y_val_raw = [], []

    for _, row in val_df.iterrows():
        image_id = str(row['ID'])
        label = int(row[CFG.DISEASE_COLUMN])

        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(CFG.DATA_DIR_VAL, image_id + ext)
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is None: continue

                try:
                    fv, _ = extract_features(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    X_val_raw.append(fv)
                    y_val_raw.append(label)
                except:
                    pass
                break

    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
    X_val_tensor = torch.from_numpy(X_val_scaled).to(device)
    y_val_tensor = torch.from_numpy(np.array(y_val_raw)).float().to(device)

    # class imbalance weights
    pos_count = sum(y_raw); neg_count = len(y_raw) - pos_count
    pos_weight = neg_count / max(pos_count, 1)

    dataset   = RetinalDataset(X_scaled, y_raw)
    loader    = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)

    model     = DRClassifierNN().to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr          = CFG.LR,
                           betas       = (0.9, 0.999),
                           eps         = 1e-8,
                           weight_decay= CFG.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)

    criterion = FocalBCELoss(pos_weight=pos_weight)

    loss_history      = []
    grad_norm_history = []

    print(f"\n  Adam optimizer: lr={CFG.LR}, β=(0.9,0.999), wd={CFG.WEIGHT_DECAY}")
    print(f"  Loss function : FocalBCE(α=0.75, γ=2.0, pos_weight={pos_weight:.2f})")
    print(f"  Epochs        : {CFG.EPOCHS}\n")

    best_auc = 0

    for epoch in range(CFG.EPOCHS):
        model.train()
        epoch_loss  = 0.0
        epoch_gnorm = 0.0
        n_batches   = 0

        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)

            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss  += loss.item()
            epoch_gnorm += total_norm
            n_batches   += 1

        avg_loss  = epoch_loss  / max(n_batches, 1)
        avg_gnorm = epoch_gnorm / max(n_batches, 1)
        loss_history.append(avg_loss)
        grad_norm_history.append(avg_gnorm)
        scheduler.step()

        # testing on validation dataset 
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_probs  = torch.sigmoid(val_logits).cpu().numpy()
            val_preds  = (val_probs >= 0.5).astype(int)

            val_acc = accuracy_score(y_val_raw, val_preds)
            val_auc = roc_auc_score(y_val_raw, val_probs)

        model.train()

        # Saving best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:>3}/{CFG.EPOCHS}] "
                  f"Loss={avg_loss:.4f}  ‖∇‖={avg_gnorm:.4f}  "
                  f"ValAcc={val_acc:.3f}  ValAUC={val_auc:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    print("\n  CNN training complete.")
    return model, scaler, loss_history, grad_norm_history


def _save_training_curves(loss_history, grad_norm_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(loss_history) + 1)

    ax1.plot(epochs, loss_history, 'b-o', markersize=4, linewidth=2)
    ax1.set_title('Training Loss (Focal BCE)\nL = -∑ αt·(1-pt)^γ·log(pt)',
                  fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, grad_norm_history, 'r-o', markersize=4, linewidth=2)
    ax2.set_title('Gradient Norm per Epoch\n‖∇θL‖₂  – Adam update tracking',
                  fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('‖∇‖₂')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()



# HYPOTHESIS TESTING  (t-test on feature distributions with 0.05 Los)

def run_hypothesis_testing(X_all, y_all, feature_names):
    """
    Two-sample Welch's t-test for each feature:
        H₀: μ_disease = μ_normal
        H₁: μ_disease ≠ μ_normal
    Reports t-statistic, p-value, reject H₀ at α=0.05
    """
    print(f"\n[HYPOTHESIS TESTING]  Welch's t-test per feature  (α={CFG.ALPHA})")
    print(f"  H₀: E[feature | DR=1] = E[feature | DR=0]")
    print(f"  {'Feature':<25} {'t-stat':>8} {'p-value':>12} {'Reject H₀':>12}")
    print(f"  {'-'*60}")

    X_all = np.array(X_all); y_all = np.array(y_all)
    results = []
    for i, name in enumerate(feature_names):
        g1 = X_all[y_all == 1, i]
        g0 = X_all[y_all == 0, i]
        if len(g1) < 2 or len(g0) < 2:
            results.append({'feature': name, 't': np.nan, 'p': np.nan, 'reject': False})
            continue
        t, p = stats.ttest_ind(g1, g0, equal_var=False)
        reject = p < CFG.ALPHA
        flag = '  ✓ SIGNIFICANT' if reject else ''
        print(f"  {name:<25} {t:>8.3f} {p:>12.4e} {str(reject):>12}{flag}")
        results.append({'feature': name, 't': t, 'p': p, 'reject': reject})

    sig_count = sum(r['reject'] for r in results)
    print(f"\n  {sig_count}/{len(feature_names)} features significantly different (p<{CFG.ALPHA})")
    return results



# SPECTRAL CLUSTERING  on feature embeddings

def run_spectral_clustering(X_scaled, n_clusters: int = CFG.N_CLUSTERS):
    """
    Spectral Clustering on the normalised feature matrix.
    Constructs an affinity graph W from RBF kernel:
        W_ij = exp(-‖x_i - x_j‖² / (2σ²))
    Solves: L·v = λv  where L = D⁻¹/²(D-W)D⁻¹/²
    Clusters the top-k eigenvectors via k-means.
    """
    print(f"\n[SPECTRAL CLUSTERING]  k={n_clusters} clusters on feature space")
    max_sc = min(len(X_scaled), 200)
    idx    = np.random.choice(len(X_scaled), max_sc, replace=False)
    X_sub  = X_scaled[idx]

    sc = SpectralClustering(
        n_clusters       = n_clusters,
        affinity         = 'rbf',          # W_ij = exp(-γ‖xi-xj‖²)
        gamma            = 1.0,
        assign_labels    = 'kmeans',
        random_state     = 42,
        n_init           = 10,
    )
    labels = sc.fit_predict(X_sub)
    counts = {c: int(np.sum(labels == c)) for c in range(n_clusters)}
    print(f"  Cluster sizes: {counts}")

    # clustering results
    from sklearn.decomposition import PCA
    pca    = PCA(n_components=2, random_state=42)
    X_2d   = pca.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap    = plt.cm.get_cmap('tab10', n_clusters)
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap(c)], label=f'Cluster {c}', s=40, alpha=0.7)
    ax.set_title(f'Spectral Clustering (k={n_clusters}) – PCA Projection\n'
                 f'L·v = λv,  W_ij = exp(-‖xi-xj‖²/2σ²)',
                 fontweight='bold')
    ax.set_xlabel('PC-1'); ax.set_ylabel('PC-2')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, 'spectral_clustering.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: spectral_clustering.png")
    return labels, idx



# PREDICT with trained NN

def predict_proba(model, scaler, features_np):
    """Returns P(DR=1) for a single feature vector."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(
            scaler.transform(features_np.reshape(1, -1)).astype(np.float32)
        ).to(device)
        logit = model(x)
        prob  = torch.sigmoid(logit).cpu().item()
    return prob


## suspicious case searching

def find_suspicious_cases(csv_path, data_dir, model, scaler):
    df        = pd.read_csv(csv_path)
    normal_df = df[df[CFG.DISEASE_COLUMN] == 0].copy()

    print(f"\n[PHASE 2] Scanning {len(normal_df)} NORMAL-labelled images for hidden DR...")
    suspicious = []
    processed  = 0

    for _, row in normal_df.iterrows():
        image_id = str(row['ID'])
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(data_dir, image_id + ext)
            if os.path.exists(p): img_path = p; break
        if not img_path: continue
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            features, analysis_dict = extract_features(img_rgb)
        except Exception as e:
            print(f"  Skip {image_id}: {e}"); continue
        
        risk_label = classify_risk(risk_score)
        analysis_dict['risk_label'] = risk_label

        risk_score = predict_proba(model, scaler, features)

        if risk_score > CFG.COLLECTION_PROBABILITY_THRESHOLD:
            analysis_dict['risk_score']   = risk_score
            analysis_dict['is_suspicious'] = risk_score > CFG.SUSPICIOUS_PROBABILITY_THRESHOLD
            analysis_dict['alerts'] = []
            if analysis_dict['hosvd']['rank'] > CFG.HOSVD_RANK_THRESHOLD:
                analysis_dict['alerts'].append(
                    f"{analysis_dict['lesions']['micro_count']} MICROANEURYSMS detected")
            if analysis_dict['qr']['condition_number'] > CFG.QR_CONDITION_THRESHOLD:
                analysis_dict['alerts'].append(
                    f"{analysis_dict['lesions']['hem_count']} HEMORRHAGES detected")
            if analysis_dict['fourier']['freq_ratio'] > CFG.FOURIER_RATIO_THRESHOLD:
                analysis_dict['alerts'].append(
                    f"High Freq Ratio ({analysis_dict['fourier']['freq_ratio']:.2f}) "
                    f"[Spectral Entropy={analysis_dict['fourier']['spectral_entropy']:.2f}]")
            suspicious.append({ 'image_id': image_id, 'risk_score': risk_score,'risk_label': risk_label,'analysis': analysis_dict})

        processed += 1
        if processed % 100 == 0:
            print(f"  Analysed {processed}/{len(normal_df)}")
        if processed >= 400: break

    suspicious.sort(key=lambda x: x['risk_score'], reverse=True)
    print(f"\n  Found {len(suspicious)} suspicious cases "
          f"(threshold={CFG.COLLECTION_PROBABILITY_THRESHOLD})")
    return suspicious
   
def process_all_images(csv_path, data_dir, model, scaler):
    print(f"\n[PHASE X] Classifying ALL images and generating reports...\n")

    df = pd.read_csv(csv_path)
    os.makedirs('math/results', exist_ok=True)

    count = 0

    for _, row in df.iterrows():
        image_id = str(row['ID'])

        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(data_dir, image_id + ext)
            if os.path.exists(p):
                img_path = p
                break
        if not img_path:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            features, analysis_dict = extract_features(img_rgb)
        except:
            continue

        # prediction
        features_scaled = scaler.transform(features.reshape(1, -1))
        prob = model.predict_proba(features_scaled)[0]
        risk_score = float(prob[1])

        # classify
        if risk_score < CFG.LOW_RISK_THRESHOLD:
            risk_label = "LOW"
        elif risk_score < CFG.MED_RISK_THRESHOLD:
            risk_label = "MEDIUM"
        else:
            risk_label = "HIGH"

        analysis_dict['risk_score'] = risk_score
        analysis_dict['risk_label'] = risk_label
        analysis_dict['alerts'] = []

        case = {
            'image_id': image_id,
            'risk_score': risk_score,
            'analysis': analysis_dict
        }

        # generate report for test images
        generate_suspicious_report(case, data_dir, count)

        count += 1

        if count % 50 == 0:
            print(f"Processed {count} images...")

    print(f"\nDone. Generated reports for {count} images.")




# results generation
 
def generate_suspicious_report(case, data_dir, idx):
    image_id   = case['image_id']
    analysis   = case['analysis']
    risk_score = analysis['risk_score']

    rl  = ('LOW'    if risk_score < CFG.LOW_RISK_THRESHOLD else
           'MEDIUM' if risk_score < CFG.MED_RISK_THRESHOLD else 'HIGH')
    rc  = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}[rl]
    bc  = {'LOW': 'lightgreen', 'MEDIUM': '#FFF2CC', 'HIGH': '#FFE4E1'}[rl]
    ec  = rc

    if rl == 'LOW':
        rec_text = (f"RECOMMENDATION:\n\nHealthy – No immediate action\n"
                    f"Routine follow-up in 12 months\n\n"
                    f"CNN Prob: {risk_score*100:.1f}%\nLow risk indicators")
        alerts_text = "No major clinical alerts"
    elif rl == 'MEDIUM':
        rec_text = (f"RECOMMENDATION:\n\nMedium Risk – Monitor\n"
                    f"Revisit in 3-6 months\n\nCNN Prob: {risk_score*100:.1f}%")
        alerts_text = ("CLINICAL ALERTS:\n\n" +
                       "\n".join(f"- {a}" for a in analysis['alerts'][:6])
                       if analysis['alerts'] else "Medium risk indicators")
    else:
        rec_text = (f"RECOMMENDATION:\n\nHIGH RISK – Urgent\n"
                    f"Re-examine immediately\n\nCNN Prob: {risk_score*100:.1f}%\n"
                    f"Multiple abnormalities")
        alerts_text = ("CLINICAL ALERTS:\n\n" +
                       "\n".join(f"- {a}" for a in analysis['alerts'][:6])
                       if analysis['alerts'] else "High risk indicators")

    img_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        p = os.path.join(data_dir, image_id + ext)
        if os.path.exists(p): img_path = p; break
    if not img_path: return

    img = cv2.imread(img_path)
    if img is None: return
    img_rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), CFG.REPORT_IMG_SIZE)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f'{rl} RISK – ID: {image_id} (Labeled: NORMAL)',
                 fontsize=20, fontweight='bold', color=rc, y=0.98)

    def ax_img(pos, arr, title, cmap=None):
        a = plt.subplot(2, 5, pos)
        a.imshow(arr, cmap=cmap); a.set_title(title, fontweight='bold', fontsize=11)
        a.axis('off'); return a

    ax_img(1,  img_rgb,                              'Original\n(Labeled NORMAL)')
    ax_img(2,  analysis['lesions']['microaneurysms'], f'Microaneurysms\nCount: {analysis["lesions"]["micro_count"]}', 'Reds')
    ax_img(3,  analysis['lesions']['hemorrhages'],    f'Hemorrhages\nCount: {analysis["lesions"]["hem_count"]}',      'Reds')
    ax_img(4,  analysis['lesions']['exudates'],       f'Exudates\nCount: {analysis["lesions"]["exu_count"]}',        'YlOrRd')

    ax5 = plt.subplot(2, 5, 5)
    ax5.text(0.5, 0.5, alerts_text, ha='center', va='center', fontsize=10,
             family='monospace', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=rc, alpha=0.3, edgecolor=ec, linewidth=3))
    ax5.axis('off'); ax5.set_title('Clinical Alerts', fontweight='bold', fontsize=12, color=rc)

    ax_img(6,  analysis['lesions']['vessels'], f'Vessel Analysis\nDensity: {analysis["lesions"]["vessel_density"]:.3f}', 'hot')

    # Overlay
    ax7   = plt.subplot(2, 5, 7)
    ov    = img_rgb.copy(); H, W = ov.shape[:2]
    for mask_key, color in [('microaneurysms', [255,0,0]),
                             ('hemorrhages',   [255,165,0]),
                             ('exudates',      [255,255,0])]:
        m = cv2.resize(analysis['lesions'][mask_key], (W, H), interpolation=cv2.INTER_NEAREST)
        ov[m > 0] = color
    ax7.imshow(ov)
    ax7.set_title('Overlay\nRed:Micro Orange:Hem Yellow:Exu', fontweight='bold', fontsize=11)
    ax7.axis('off')

    ax8 = plt.subplot(2, 5, 8)
    quant = (f"Quantitative:\n\n"
             f"Micro: {analysis['lesions']['micro_count']}\n"
             f"Hem:   {analysis['lesions']['hem_count']}\n"
             f"Exu:   {analysis['lesions']['exu_count']}\n\n"
             f"Vessel Density: {analysis['lesions']['vessel_density']:.3f}\n"
             f"Freq Ratio:     {analysis['fourier']['freq_ratio']:.2f}\n"
             f"Spectral Ent:   {analysis['fourier']['spectral_entropy']:.2f}\n"
             f"Texture Rank:   {analysis['hosvd']['rank']}")
    ax8.text(0.5, 0.5, quant, ha='center', va='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax8.axis('off'); ax8.set_title('Quantitative', fontweight='bold', fontsize=12)

    ax9 = plt.subplot(2, 5, 9)
    risk_txt = (f"RISK ASSESSMENT\n\nCNN PROB: {risk_score*100:.1f}%\n"
                f"RISK LEVEL: {rl}\n\n"
                f"GPU Math Methods:\n"
                f"• PDE edge:      {analysis['pde']['edge_score']:.4f}\n"
                f"• HOSVD rank:    {analysis['hosvd']['rank']}\n"
                f"• QR κ(A):       {analysis['qr']['condition_number']:.1f}\n"
                f"• Fourier γ:     {analysis['fourier']['freq_ratio']:.2f}\n"
                f"• Spectral ent:  {analysis['fourier']['spectral_entropy']:.2f}")
    ax9.text(0.5, 0.5, risk_txt, ha='center', va='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor=rc, alpha=0.4, edgecolor='black', linewidth=2))
    ax9.axis('off'); ax9.set_title('Risk Assessment', fontweight='bold', fontsize=12)

    ax10 = plt.subplot(2, 5, 10)
    ax10.text(0.5, 0.5, rec_text, ha='center', va='center', fontsize=10,
              family='monospace', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor=bc, edgecolor=ec, linewidth=3))
    ax10.axis('off'); ax10.set_title('Recommendation', fontweight='bold', fontsize=12, color=rc)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(CFG.OUTPUT_DIR,
        f'{rl}_ALERT_{idx+1:03d}_risk{risk_score*100:.0f}_{image_id}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


# result evaluation

def evaluate_dataset(csv_path, data_dir, model, scaler, dataset_name='dataset'):
    df = pd.read_csv(csv_path)
    y_true, y_scores, y_pred = [], [], []
    processed = 0

    for _, row in df.iterrows():
        image_id   = str(row['ID'])
        true_label = int(row.get(CFG.DISEASE_COLUMN, 0))
        img_path   = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(data_dir, image_id + ext)
            if os.path.exists(p): img_path = p; break
        if not img_path: continue
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            features, _ = extract_features(img_rgb)
        except Exception: continue

        prob  = predict_proba(model, scaler, features)
        pred  = int(prob >= 0.5)
        y_true.append(true_label); y_scores.append(prob); y_pred.append(pred)
        processed += 1
        if processed % 50 == 0:
            print(f"  [{dataset_name}] {processed} processed...")
        if processed >= CFG.MAX_EVAL: break

    metrics = {'processed': processed}
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    if processed == 0:
        metrics.update({'accuracy': float('nan'), 'f1_score': float('nan'), 'auc': float('nan')})
        return metrics

    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    try:    metrics['f1_score'] = float(f1_score(y_true, y_pred))
    except: metrics['f1_score'] = float('nan')
    try:    metrics['auc'] = float(roc_auc_score(y_true, y_scores))
    except: metrics['auc'] = float('nan')

    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'CNN ROC (AUC={roc_auc_val:.3f})')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title(f'ROC – {dataset_name} (Trained CNN)')
        plt.legend(loc='lower right')
        roc_path = os.path.join(CFG.OUTPUT_DIR, f'ROC_{dataset_name}.png')
        plt.savefig(roc_path, dpi=150, bbox_inches='tight'); plt.close()
        metrics['roc_path'] = roc_path
    except: metrics['roc_path'] = None

    return metrics


## main function

FEATURE_NAMES = [
    'pde_edge_score', 'hosvd_rank', 'hosvd_energy_ratio', 'hosvd_entropy',
    'qr_condition_number', 'qr_rank', 'gauss_determinant', 'gauss_trace',
    'fourier_freq_ratio', 'fourier_spectral_entropy',
    'micro_count', 'hem_count', 'exu_count', 'vessel_density'
]

if __name__ == "__main__":
    run_id = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print("GPU-ACCELERATED DR DETECTION  |  CNN + Adam + Focal Loss")
    print(f"{'='*70}\n")

    # ── Phase 1: train CNN
    model, scaler, loss_hist, grad_hist = train_model(CFG.CSV_PATH_TRAIN, CFG.DATA_DIR_TRAIN)

    # ── Collect all features for hypothesis testing + spectral clustering
    print("\n[AUX] Collecting features for statistical analysis...")
    df_train = pd.read_csv(CFG.CSV_PATH_TRAIN)
    X_all, y_all = [], []
    for _, row in df_train.iterrows():
        iid = str(row['ID'])
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(CFG.DATA_DIR_TRAIN, iid + ext)
            if os.path.exists(p): img_path = p; break
        if not img_path: continue
        img = cv2.imread(img_path)
        if img is None: continue
        try:
            fv, _ = extract_features(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            X_all.append(fv); y_all.append(int(row[CFG.DISEASE_COLUMN]))
        except: pass
        if len(X_all) >= 200: break

    X_all_scaled = scaler.transform(X_all)

    #Hypothesis Testing
    ht_results = run_hypothesis_testing(X_all_scaled, y_all, FEATURE_NAMES)

    #Spectral Clustering
    sc_labels, sc_idx = run_spectral_clustering(X_all_scaled)

    # ── Phase 2: find suspicious cases
    print("\n[PHASE 2a] Training set scan...")
    train_sus = find_suspicious_cases(CFG.CSV_PATH_TRAIN, CFG.DATA_DIR_TRAIN, model, scaler)
    print("\n[PHASE 2b] Test set scan...")
    test_sus  = find_suspicious_cases(CFG.CSV_PATH_TEST,  CFG.DATA_DIR_TEST,  model, scaler)

    all_sus = sorted(train_sus + test_sus, key=lambda x: x['risk_score'], reverse=True)
    print(f"\nTotal suspicious: {len(all_sus)}")
    for i, c in enumerate(all_sus[:10], 1):
        print(f"  {i}. ID={c['image_id']}  Risk={c['risk_score']*100:.1f}%")

    print("\n[PHASE 3] Generating reports...")
    low_c   = [c for c in all_sus if c['risk_score'] <  CFG.LOW_RISK_THRESHOLD]
    med_c   = [c for c in all_sus if CFG.LOW_RISK_THRESHOLD <= c['risk_score'] < CFG.MED_RISK_THRESHOLD]
    high_c  = [c for c in all_sus if c['risk_score'] >= CFG.MED_RISK_THRESHOLD]

    cases_to_report = []
    if low_c:  cases_to_report.append(low_c[0])
    if med_c:  cases_to_report.append(med_c[0])
    if high_c: cases_to_report.append(high_c[0])
    while len(cases_to_report) < 3 and all_sus:
        cases_to_report.append(all_sus.pop(0))

    generated = 0
    for idx, case in enumerate(cases_to_report):
        dd = (CFG.DATA_DIR_TRAIN if any(os.path.exists(os.path.join(CFG.DATA_DIR_TRAIN, case['image_id']+e))
                                        for e in ['.png','.jpg','.jpeg'])
              else CFG.DATA_DIR_TEST)
        generate_suspicious_report(case, dd, idx)
        generated += 1
    print(f"  Generated {generated} reports.")

    # evaluate
    print(f"\n[PHASE 4] Evaluating on test set...")
    test_metrics = evaluate_dataset(CFG.CSV_PATH_TEST, CFG.DATA_DIR_TEST, model, scaler, 'test')

    # result as csv 
    ht_sig = [r['feature'] for r in ht_results if r['reject']]
    metrics_summary = {
        'run_id':            run_id,
        'model':             'DRClassifierNN (MLP)',
        'optimizer':         'Adam (β1=0.9, β2=0.999)',
        'loss_fn':           'FocalBCE (α=0.75, γ=2.0)',
        'epochs':            CFG.EPOCHS,
        'final_train_loss':  round(loss_hist[-1], 5) if loss_hist else float('nan'),
        'final_grad_norm':   round(grad_hist[-1], 5) if grad_hist else float('nan'),
        'significant_features': str(ht_sig),
        'test_processed':    test_metrics.get('processed', 0),
        'test_accuracy':     test_metrics.get('accuracy',  float('nan')),
        'test_f1_score':     test_metrics.get('f1_score',  float('nan')),
        'test_auc':          test_metrics.get('auc',       float('nan')),
    }
    mdf = pd.DataFrame([metrics_summary])
    csv_path = os.path.join(CFG.OUTPUT_DIR, f'metrics_summary_{run_id}.csv')
    mdf.to_csv(csv_path, index=False)

    # summary
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print("Train Accuracy:", accuracy_score(y_train, model.predict(X_train_scaled)))
    print(f"{'='*70}")
    print(f"  Model            : DRClassifierNN (MLP, {FEATURE_DIM} features → 1 logit)")
    print(f"  Optimizer        : Adam (lr={CFG.LR}, β=(0.9,0.999))")
    print(f"  Loss             : Focal BCE (α=0.75, γ=2.0)")
    print(f"  Gradient Descent : {CFG.EPOCHS} epochs, cosine LR schedule")
    print(f"  Fourier used for : freq_ratio + spectral_entropy features")
    print(f"  Hypothesis Tests : {sum(r['reject'] for r in ht_results)}/{len(ht_results)} features significant")
    print(f"  Spectral Clust.  : k={CFG.N_CLUSTERS} clusters on feature space")
    print(f"  Train suspicious : {len(train_sus)}")
    print(f"  Test suspicious  : {len(test_sus)}")
    print(f"  Reports saved to : {CFG.OUTPUT_DIR}/")
    print(f"  Metrics CSV      : {csv_path}")
    try:
        print(f"  Test Accuracy    : {test_metrics['accuracy']:.3f}")
        print(f"  Test F1          : {test_metrics['f1_score']:.3f}")
        print(f"  Test AUC         : {test_metrics['auc']:.3f}")
    except: pass

    if torch.cuda.is_available():
        print(f"\n  Peak GPU Memory  : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\n")