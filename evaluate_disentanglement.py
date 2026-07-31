import argparse
import os
import glob
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from recbole_cdr.quick_start.quick_start import load_data_and_model

# ==========================================
# CHANGE TO THE DESIRED CHECKPOINT PATH
# ==========================================
DEFAULT_MODEL_PATH = "saved/DGCDR-Jul-14-2026_12-22-57.pth"

# dCor needs the full N x N distance matrix, so cost grows with the square of
# the user count: at 35k overlapping users the computation holds ~19 GB in RAM.
# Sub-sampling barely moves the estimate (0.754 at 5k vs 0.743 at 35k on
# Elec->Cloth) so it is sampled by default.
DEFAULT_DCOR_SAMPLE = 5000
DEFAULT_DCOR_PERMUTATIONS = 5

def get_latest_checkpoint(checkpoint_dir='saved'):
    pth_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not pth_files:
        return None
    # Sort by modification time (most recent first)
    pth_files.sort(key=os.path.getmtime, reverse=True)
    return pth_files[0]

def compute_distance_matrix(X):
    # X: [N, D]
    # torch.cdist instead of the expanded ||a||^2 - 2ab + ||b||^2 form, which
    # loses precision to catastrophic cancellation when the norms are large
    # (median relative error ~1e-4 in that regime) and gives a non-zero
    # diagonal.
    return torch.cdist(X, X)

def double_center(D):
    # D: [N, N]
    row_means = torch.mean(D, dim=1, keepdim=True)
    col_means = torch.mean(D, dim=0, keepdim=True)
    grand_mean = torch.mean(D)
    return D - row_means - col_means + grand_mean

def distance_correlation(X, Y):
    """
    Computes the distance correlation (dCor) between X and Y.
    dCor = 0 if and only if X and Y are independent -- but only in the limit.
    This is the biased (V-statistic) estimator, which is strongly positive on
    independent data at finite N and high D: independent Gaussians at N=5000,
    D=256 give dCor = 0.41. A raw dCor is therefore NOT interpretable against
    an "ideal 0.0"; compare it with the permutation baseline from
    :func:`distance_correlation_with_null` instead.
    """
    N = X.size(0)
    if N <= 1:
        return 0.0

    # 1. Distance matrices
    D_X = compute_distance_matrix(X)
    D_Y = compute_distance_matrix(Y)

    # 2. Double centering
    A = double_center(D_X)
    B = double_center(D_Y)

    # 3. Covariance and variances
    dcov2 = torch.sum(A * B) / (N * N)
    dvarX2 = torch.sum(A * A) / (N * N)
    dvarY2 = torch.sum(B * B) / (N * N)

    if dvarX2 <= 0.0 or dvarY2 <= 0.0:
        return 0.0

    # dcov2 is non-negative in exact arithmetic; clamp so float error near zero
    # cannot produce a NaN.
    ratio = torch.clamp(dcov2, min=0.0) / torch.sqrt(dvarX2 * dvarY2)
    dcor = torch.sqrt(torch.clamp(ratio, min=0.0))
    return dcor.item()

def distance_correlation_with_null(X, Y, sample_size=DEFAULT_DCOR_SAMPLE,
                                   permutations=DEFAULT_DCOR_PERMUTATIONS, seed=42):
    """dCor together with the baseline it has to be read against.

    Shuffling the rows of Y destroys the pairing while keeping both marginal
    distributions intact, so the resulting dCor is exactly the value this
    estimator returns when the two spaces are independent, at this N and D and
    on these embeddings. The gap between the two is the evidence of dependence;
    the raw dCor on its own is not.

    Returns:
        (dcor, null_mean, null_std, n_used)
    """
    generator = torch.Generator().manual_seed(seed)

    n = X.size(0)
    if sample_size and n > sample_size:
        idx = torch.randperm(n, generator=generator)[:sample_size]
        X, Y = X[idx], Y[idx]

    observed = distance_correlation(X, Y)

    nulls = []
    for _ in range(permutations):
        perm = torch.randperm(Y.size(0), generator=generator)
        nulls.append(distance_correlation(X, Y[perm]))

    return observed, float(np.mean(nulls)), float(np.std(nulls)), X.size(0)

def run_probe(classifier_factory, X, y, seed=42):
    """Fit a domain probe and return its held-out accuracy.

    The features are standardised, fitting the scaler on the training split
    only. Without this the probe badly understates the leakage it is meant to
    detect: these embeddings have entries around 0.02, so the default L2
    penalty dominates the loss and the model underfits. On the CDs/Instruments
    checkpoint the linear probe reads 85.4% unscaled against 92.5% scaled --
    a 7-point error, all of it in the direction that flatters the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    classifier = classifier_factory().fit(scaler.transform(X_train), y_train)
    return accuracy_score(y_test, classifier.predict(scaler.transform(X_test)))

def evaluate_embeddings(model_path, output_dir='disentanglement_eval',
                        dcor_sample=DEFAULT_DCOR_SAMPLE,
                        dcor_permutations=DEFAULT_DCOR_PERMUTATIONS, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data and model checkpoint from {model_path}...")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)
    model.eval()

    print("\nExtracting embeddings via model.forward()...")
    with torch.no_grad():
        user_disentangled_list, item_disentangled_list, _, _, _, _ = model.forward()

    if not user_disentangled_list:
        raise ValueError("Error: The loaded model does not have preference disentanglement enabled.")

    # Extract user common and specific features for overlapping users
    source_c = user_disentangled_list[0].cpu()
    target_c = user_disentangled_list[1].cpu()
    source_s = user_disentangled_list[2].cpu()
    target_s = user_disentangled_list[3].cpu()

    print(f"Extracted features for {source_c.size(0)} overlapping users.")
    print(f"Embedding size: {source_c.size(1)}")

    # --- Metric 1: Average Absolute Cosine Similarity ---
    source_cos = torch.nn.functional.cosine_similarity(source_c, source_s, dim=1).abs().mean().item()
    target_cos = torch.nn.functional.cosine_similarity(target_c, target_s, dim=1).abs().mean().item()

    # --- Metric 2: Distance Correlation (dCor), against its permutation null ---
    print(f"\nComputing dCor (sample {dcor_sample}, {dcor_permutations} permutations)...")
    source_dcor, source_null, source_null_sd, n_used = distance_correlation_with_null(
        source_c, source_s, dcor_sample, dcor_permutations, seed)
    target_dcor, target_null, target_null_sd, _ = distance_correlation_with_null(
        target_c, target_s, dcor_sample, dcor_permutations, seed)

    # --- Metric 3: Probing Datasets Preparation ---
    # The probe asks whether a vector produced by the source-domain encoder can
    # be told apart from one produced by the target-domain encoder.
    X_common = torch.cat([source_c, target_c], dim=0).numpy()
    X_specific = torch.cat([source_s, target_s], dim=0).numpy()
    y = np.array([0] * len(source_c) + [1] * len(target_c))

    # --- Metric 4: Linear Probe (Logistic Regression) ---
    print("Fitting probes...")
    acc_c_linear = run_probe(lambda: LogisticRegression(max_iter=1000), X_common, y, seed)
    acc_s_linear = run_probe(lambda: LogisticRegression(max_iter=1000), X_specific, y, seed)

    # --- Metric 5: Non-Linear Probe (Multi-Layer Perceptron) ---
    # A 2-layer MLP (100 -> 50 -> ReLU) to capture non-linear domain leakage
    mlp = lambda: MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                                max_iter=1000, random_state=seed)
    acc_c_mlp = run_probe(mlp, X_common, y, seed)
    acc_s_mlp = run_probe(mlp, X_specific, y, seed)

    # Calculate Disentanglement Scores
    dis_score_linear = acc_s_linear - acc_c_linear
    dis_score_mlp = acc_s_mlp - acc_c_mlp

    # Save metrics to TXT and CSV with unique name
    import csv
    from datetime import datetime

    model_name = os.path.basename(model_path).replace('.pth', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    readable_timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')

    # Echo the weights the checkpoint was trained with. Without this a sweep
    # produces reports that cannot be told apart, and the log files have to be
    # consulted to find out which point each one is.
    hyper = " | ".join(
        f"{name}={config[name]}" for name in
        ('cl_sim_weight', 'cl_org_weight', 'cl_decoder_weight', 'item_cl_weight',
         'temperature', 'fuse_mode')
        if config[name] is not None
    )
    domains = f"{config['source_domain']['dataset']} -> {config['target_domain']['dataset']}"

    # Generate Text Report with unique name
    report = f"""
======================================================================
                     DISENTANGLEMENT METRICS REPORT
======================================================================
Model Checkpoint    : {model_name}
Timestamp           : {readable_timestamp}
Domains             : {domains}
Hyper-parameters    : {hyper}

1. FEATURE INDEPENDENCE METRICS (e^c vs e^s)
----------------------------------------------------------------------
Cosine Similarity   | Source: {source_cos:.4f} | Target: {target_cos:.4f}  (Ideal: ~0.00)

Distance Correlation, on {n_used} sampled users, against the permutation
baseline (what this estimator returns when the two spaces are independent):
  - Source: {source_dcor:.4f}  vs null {source_null:.4f} +/- {source_null_sd:.4f}  -> gap {source_dcor - source_null:+.4f}
  - Target: {target_dcor:.4f}  vs null {target_null:.4f} +/- {target_null_sd:.4f}  -> gap {target_dcor - target_null:+.4f}
  (Ideal: gap ~0.00. The raw dCor has no meaningful "ideal 0" -- it is
   positively biased at finite N and high dimension.)

2. ADVERSARIAL DOMAIN PROBING ACCURACIES
----------------------------------------------------------------------
LINEAR PROBE (Logistic Regression):
  - Common Features (e^c) Accuracy   : {acc_c_linear * 100:.2f}%          (Ideal: ~50.00%)
  - Specific Features (e^s) Accuracy : {acc_s_linear * 100:.2f}%          (Ideal: >85.00%)
  - Disentanglement Score            : {dis_score_linear * 100:.2f}%          (Ideal: >35.00%)

NON-LINEAR PROBE (2-Layer MLP):
  - Common Features (e^c) Accuracy   : {acc_c_mlp * 100:.2f}%          (Ideal: ~50.00%)
  - Specific Features (e^s) Accuracy : {acc_s_mlp * 100:.2f}%          (Ideal: >85.00%)
  - Disentanglement Score            : {dis_score_mlp * 100:.2f}%          (Ideal: >35.00%)

======================================================================
                           GUIDE & LEGEND
======================================================================
* Cosine Similarity measures whether the two spaces are orthogonal.
  Lower is better (~0.0). Note that orthogonality is NOT independence:
  two spaces can be perpendicular and still fully predict each other.
* Distance Correlation (dCor) measures statistical dependence, including
  the non-linear kind that cosine cannot see. Read the GAP over the
  permutation null, never the raw value.
* The null depends on how many users were sampled, so gaps are only
  comparable across checkpoints when --dcor_sample matches. Datasets
  with fewer overlapping users than the sample size are capped by their
  own size (reported above as the sampled count).
* Domain Probing Accuracy on Shared Features (e^c) measures domain
  invariance. Lower is better (ideal: ~50.00% / random chance).
* Domain Probing Accuracy on Specific Features (e^s) measures domain
  retention. Higher is better (ideal: >85.00%).
* Disentanglement Score = Accuracy(e^s) - Accuracy(e^c).
  Higher is better (ideal: >35.00%).
* Probe features are standardised (scaler fitted on the training split
  only). Skipping this understates leakage by several points.
======================================================================
"""

    txt_path = os.path.join(output_dir, f'report_{model_name}_{timestamp}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # Append to central history CSV
    csv_path = os.path.join(output_dir, 'disentanglement_history.csv')
    header = [
        'Timestamp', 'Model_Checkpoint', 'CosSim_Src', 'CosSim_Tgt',
        'dCor_Src', 'dCor_Tgt', 'dCorNull_Src', 'dCorNull_Tgt',
        'dCorGap_Src', 'dCorGap_Tgt', 'dCor_N',
        'LinearProbe_Common', 'LinearProbe_Specific', 'DisentanglementScore_Linear',
        'MLPProbe_Common', 'MLPProbe_Specific', 'DisentanglementScore_MLP'
    ]

    write_header = True
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            existing = f.readline().strip()
        if existing == ','.join(header):
            write_header = False
        elif existing:
            # Never truncate history in place: rows written under a different
            # set of columns are preserved next to the new file.
            backup = csv_path.replace('.csv', f'_legacy_{timestamp}.csv')
            os.rename(csv_path, backup)
            print(f"CSV columns changed; previous history preserved at {backup}")

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            timestamp, model_name,
            f"{source_cos:.4f}", f"{target_cos:.4f}",
            f"{source_dcor:.4f}", f"{target_dcor:.4f}",
            f"{source_null:.4f}", f"{target_null:.4f}",
            f"{source_dcor - source_null:.4f}", f"{target_dcor - target_null:.4f}",
            n_used,
            f"{acc_c_linear:.4f}", f"{acc_s_linear:.4f}", f"{dis_score_linear:.4f}",
            f"{acc_c_mlp:.4f}", f"{acc_s_mlp:.4f}", f"{dis_score_mlp:.4f}"
        ])

    # --- Print Results ---
    print(report)
    print(f"Report saved to TXT:     {txt_path}")
    print(f"Appended to history CSV: {csv_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Disentanglement Quality")
    parser.add_argument('--model_path', '-m', type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to checkpoint .pth file (defaults to DEFAULT_MODEL_PATH configured in the script)")
    parser.add_argument('--output_dir', '-o', type=str, default='disentanglement_eval', help="Output directory")
    parser.add_argument('--dcor_sample', type=int, default=DEFAULT_DCOR_SAMPLE,
                        help="Users sampled for dCor (0 = all; memory grows with the square)")
    parser.add_argument('--dcor_permutations', type=int, default=DEFAULT_DCOR_PERMUTATIONS,
                        help="Permutations used to estimate the dCor null baseline")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_path = args.model_path
    if not model_path or not os.path.exists(model_path):
        latest = get_latest_checkpoint('saved')
        if latest:
            print(f"Configured path '{model_path}' not found or empty. Falling back to latest checkpoint: {latest}")
            model_path = latest
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{model_path}' and no checkpoints found in 'saved/'.")

    evaluate_embeddings(model_path, args.output_dir, args.dcor_sample,
                        args.dcor_permutations, args.seed)
