# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q numpy pandas scipy scikit-learn imbalanced-learn cleanlab matplotlib

# ============================================================================
# AMEND — STANDALONE LABEL-NOISE CLEANER : EXPERIMENT SUITE  v2
# AMEND-ONLY BUILD  (ladder collapsed to the single headline variant,
# with end-task FPR and detection FPR columns added)
#
# WHAT CHANGED vs the general suite
# ----------------------------------------------------------------------------
#   * AMEND_LADDER holds only AMEND; the agree-weight sweep is removed.
#   * AMEND_DEFAULTS / ABLATION_DEFAULT carry the WA terms
#     (agree_weight=0.3, anchored_rescue=True), so Exp1's ablation centre and
#     every AMEND call is the WA configuration.
#   * Every experiment's cleaner/method list is AMEND (+ the no-clean
#     "None" control where the Friedman/Wilcoxon stats need >=2 columns).
#   * endtask_scores now returns FPR (= FP/(FP+TN) = 1 - specificity) for the
#     downstream classifier on the clean test fold.
#   * detection_scores exposes FPR as an alias of FRR (FP/(FP+TN) over clean
#     vs removed points) so the cleaner's false-positive rate prints under the
#     FPR name too. DET_KEYS and Exp3's DET_REPORT/END_METRICS list it.
#
# NOTE: end-task FPR (classifier behaviour on the CLEAN TEST fold) and
# detection FPR/FRR (cleaner behaviour on the NOISY TRAIN fold) measure
# different things and will not resemble each other — label them clearly.
#
# PROTOCOL (unchanged): clean-then-classify. Label noise is injected into the
# TRAIN fold only; the cleaner is fit on the noisy train fold; classifiers are
# trained on the cleaned result; evaluation is on the CLEAN test fold. Means
# over seeds x StratifiedKFold folds.
#
# Colab setup, if either package is missing:
#   !pip install -q imbalanced-learn cleanlab
# ============================================================================

import os
import json
import time
import warnings
warnings.simplefilter("once")

import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.stats import wilcoxon, rankdata, f as f_dist

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score)

try:
    from google.colab import drive
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

# imbalanced-learn baseline cleaners
try:
    from imblearn.under_sampling import (EditedNearestNeighbours, TomekLinks,
                                         InstanceHardnessThreshold, AllKNN,
                                         NeighbourhoodCleaningRule)
    _IMBLEARN = True
except ImportError:
    print("Note: imbalanced-learn not available — baseline cleaners disabled "
          "(install with: pip install imbalanced-learn)")
    _IMBLEARN = False

# Confident Learning reference implementation (optional; a self-contained
# equivalent below is used when it is missing).
try:
    from cleanlab.filter import find_label_issues
    _CLEANLAB = True
except ImportError:
    _CLEANLAB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- quiet Colab/Jupyter + pandas deprecation spam (cosmetic only) ---
import logging as _logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass
_logging.getLogger("jupyter_client").setLevel(_logging.ERROR)
_logging.getLogger("jupyter_client.session").setLevel(_logging.ERROR)


# ============================================================================
# CONFIG
# ============================================================================
FOLDER_PATH   = "/content/drive/My Drive/Results/DataSet/"
RESULTS_DIR   = "/content/drive/My Drive/Results_AMEND/"

# --- crash resume ----------------------------------------------------------
RESUME        = True
FRESH_START   = False
CHECKPOINT_DIR = None                # None -> RESULTS_DIR/_checkpoints

# SEEDS       = [42, 7, 13]          # original quick run (kept for reference)
SEEDS         = [42, 7, 13, 21, 99, 123, 256, 512, 1024, 2024]   # full-rigor 10-seed run
N_SPLITS      = 5

# --- AMEND configuration : AMEND is the resolved default -----------------
# FULLY ADAPTIVE WA:
#   * k / r / proj_dim = "auto"  -> neighbourhood size (sqrt(n_min)), local
#     intrinsic rank (energy=0.90), and JL projection width chosen from the data.
#   * contamination = "gmm"      -> the removal threshold tau is fit per fold by
#     a 2-component Gaussian mixture on the NS score distribution, instead of the
#     fixed-gap heuristic. max_tau=0.5 caps it. This is what makes the CUT
#     adaptive to the actual noise level, not just the neighbourhood geometry.
#   * SCS-only (use_fns=False, alpha ignored), agree_weight=0.3, anchored rescue.
AMEND_DEFAULTS = dict(k="auto", r="auto", proj_dim="auto",
                      alpha=1.0, use_fns=False,
                      agree_weight=0.3, anchored_rescue=True,
                      contamination="gmm", max_tau=0.5)

# --- the ladder, collapsed to AMEND only ---------------------------------
# Each entry is (extra AMENDCleaner kwargs, pass sample weights to classifier).
# All inherit AMEND_DEFAULTS.
AMEND_LADDER = {
    "AMEND": (dict(agree_weight=0.3, anchored_rescue=True,
                      contamination="gmm", max_tau=0.5), True),
}
REF_CLEANER = "AMEND"      # headline variant for the Wilcoxon comparisons

# Centre of the Exp1 ablation = the (fully adaptive) AMEND configuration.
ABLATION_DEFAULT = dict(k="auto", r="auto", proj_dim="auto",
                        alpha=1.0, use_fns=False,
                        agree_weight=0.3, anchored_rescue=True,
                        contamination="gmm", max_tau=0.5)

# --- classifiers -----------------------------------------------------------
ALL_CLASSIFIERS = ["RF", "SVM", "kNN", "LR", "DT"]
EXP3_CLASSIFIERS = ["RF", "SVM", "kNN", "LR", "DT"]
EXP1_CLASSIFIERS = ["RF", "SVM", "kNN", "LR", "DT"]
EXP2_CLASSIFIERS = ["RF", "SVM", "kNN", "LR", "DT"]

# Pass AMEND's soft weights to classifiers that accept sample_weight
# (RF / SVM / LR / DT do; kNN does not and silently ignores them).
USE_SAMPLE_WEIGHTS = True

# A kept point with weight <= this counts as "flagged" for soft detection.
FLAG_WEIGHT_THRESHOLD = 0.5

# --- experiment-specific knobs --------------------------------------------
NOISE_RATE_ABLATION   = 0.15         # injected noise used in Exp1 / Exp3
ABLATION_SCHEME       = "uniform"
EXP2_RATES            = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
EXP2_SCHEMES          = ["uniform", "minority_to_maj", "maj_to_minority"]

# --- Q1 BASELINE SET -------------------------------------------------------
# AMEND compared against established label-noise cleaners so the
# Friedman / Wilcoxon (Holm) comparative stats and the critical-difference
# plots have >=2 real methods. available_cleaners() silently drops any that
# this runtime cannot build (e.g. imblearn missing), so the run never dies
# on an absent dependency.
Q1_BASELINES          = ["CL", "AUM", "ENN", "AllKNN", "IHT"]

# Exp2 noise-sweep: keep it lighter (sweep is rate x scheme x cleaner x clf).
# AMEND + None + the single strongest comparator (CL).
EXP2_CLEANERS         = ["AMEND", "CL", "None"]
# Exp3 head-to-head: the full comparative panel.
EXP3_CLEANERS         = ["None", "AMEND"] + Q1_BASELINES

# Limit which datasets each experiment touches (None = all binary datasets).
ABLATION_DATASETS     = None   # all 13
ROBUSTNESS_DATASETS   = None
BASELINE_DATASETS     = None         # Friedman needs >= 3 datasets

# Exp4 runtime grids (synthetic data — no Drive needed)
EXP4_CLEANERS = ["AMEND", "CL"]
EXP4_N_GRID   = [200, 500, 1000, 2000, 4000]
EXP4_D_GRID   = [20, 50, 100, 300, 1000]
EXP4_D_FIXED  = 50
EXP4_N_FIXED  = 1000

# Same override hook as the v6.5 handler, if auto-detection picks the wrong col
LABEL_OVERRIDES = {
    # 'GLIOMA':     'class',
    # 'burczynski': 'target',
}

# Exp5 = metric-fair ranking comparison; Exp6 = the minority-size crossover.
EXP5_METHODS = ["AMEND", "CL", "AUM-score", "kNN-disagree", "IHT-score"]
EXP5_DATASETS = None
EXP6_METHODS = ["AMEND", "CL"]
EXP6_MIN_GRID = [5, 8, 12, 16, 20, 30, 45]
EXP6_DATASETS = None          # only datasets with enough minority are used

RUN_EXP1 = True
RUN_EXP2 = True
RUN_EXP3 = True
RUN_EXP4 = True
RUN_EXP5 = True
RUN_EXP6 = True

# Counts cleaner crashes instead of hiding them.
_CLEANER_FAILURES = defaultdict(int)
_CLASSIFIER_FAILURES = defaultdict(int)


# ============================================================================
# AMEND CORE 1/3 : SPECTRAL COHERENCE SCORER
# ============================================================================
class SpectralCoherenceScorer:
    """Per-sample noise scores from same-class local-subspace agreement.

    NS = alpha*(1 - SCS) + (1 - alpha)*FNS
      SCS : alignment of a point's local subspace with its same-class neighbors'
      FNS : fraction of a point's energy off its own local subspace
    use_scs / use_fns switch the two terms for ablation.
    """

    def __init__(self, k=8, r=5, alpha=0.5, use_scs=True, use_fns=True,
                 energy=0.90, agree_weight=0.0):
        self.k = k
        self.r = r
        self.alpha = alpha
        self.use_scs = use_scs
        self.use_fns = use_fns
        self.energy = energy          # r="auto": local energy to retain
        self.agree_weight = agree_weight

    def _resolve_r(self, X, indices, k, d, max_probe=60):
        """Adaptive rank: the local intrinsic dimension of the data."""
        if self.r != "auto":
            return max(1, min(int(self.r), d - 1, max(k - 1, 1)))
        rng = np.random.RandomState(0)
        n = len(X)
        probe = rng.choice(n, size=min(max_probe, n), replace=False)
        counts = []
        for i in probe:
            M = X[indices[i]]
            M = M - M.mean(axis=0)
            try:
                sv = np.linalg.svd(M, compute_uv=False) ** 2
            except Exception:
                continue
            tot = sv.sum()
            if tot <= 0:
                continue
            counts.append(int(np.searchsorted(np.cumsum(sv) / tot, self.energy)) + 1)
        r = int(np.median(counts)) if counts else 5
        return max(1, min(r, d - 1, max(k - 1, 1)))

    def fit_score(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n, d = X.shape
        k = min(self.k, n - 1)

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        indices = indices[:, 1:]

        r = self._resolve_r(X, indices, k, d)
        self.r_used_ = r

        CHUNK = int(np.clip(2.0e8 / max(d * d * 8.0, 1.0), 1, 500))
        CHUNK = max(1, min(CHUNK, n))
        U_all = np.zeros((n, d, r), dtype=np.float32)
        gam = min(0.5, d / (k + d))
        I_d = np.eye(d, dtype=np.float64)
        use_partial = (d > 80) and (r < d // 3)

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            X_nbrs = X[indices[start:end]]
            centered = X_nbrs - X_nbrs.mean(axis=1, keepdims=True)
            covs = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
            del X_nbrs, centered
            tr = np.trace(covs, axis1=1, axis2=2)
            covs_reg = (1 - gam) * covs + gam * (tr / max(d, 1))[:, None, None] * I_d
            del covs, tr
            for i in range(end - start):
                try:
                    if use_partial:
                        _, ev = eigsh(covs_reg[i], k=r, which="LM")
                    else:
                        _, ev = eigh(covs_reg[i]); ev = ev[:, -r:]
                    U_all[start + i] = ev.astype(np.float32)
                except Exception:
                    U_all[start + i] = np.eye(d, r, dtype=np.float32)
            del covs_reg

        scs = np.zeros(n)
        same = (y[indices] == y[:, None])
        for i in range(n):
            sc_idx = indices[i, same[i]]
            m = len(sc_idx)
            if m == 0:
                continue
            prods = np.einsum("dr,mds->mrs", U_all[i], U_all[sc_idx])
            scs[i] = float(np.sum(prods * prods)) / (m * r)

        U64 = U_all.astype(np.float64)
        UtX = np.einsum("ndr,nd->nr", U64, X)
        proj = np.einsum("ndr,nr->nd", U64, UtX)
        res_sq = np.sum((X - proj) ** 2, axis=1)
        fns = res_sq / np.maximum(np.sum(X ** 2, axis=1), 1e-12)

        a = self.alpha
        if self.use_scs and self.use_fns:
            base = a * (1.0 - scs) + (1.0 - a) * fns
        elif self.use_scs:
            base = 1.0 - scs
        elif self.use_fns:
            base = fns
        else:
            raise ValueError("At least one of use_scs / use_fns must be True.")

        dis = 1.0 - same.mean(axis=1)
        aw = float(self.agree_weight)
        ns = (1.0 - aw) * base + aw * dis if aw > 0 else base

        subspaces = [U_all[i].astype(np.float64) for i in range(n)]
        del U_all, U64
        return {"NS": ns, "SCS": scs, "FNS": fns, "DIS": dis, "BASE": base,
                "subspaces": subspaces, "nn_indices": indices}


# ============================================================================
# AMEND CORE 2/3 : SOFT CLEANER
# ============================================================================
class SoftCleaner:
    """Turn NS scores into a keep/remove/quarantine decision."""

    def __init__(self, contamination="auto", quarantine_weight=0.1,
                 rescue_threshold=0.45, min_rescue=2,
                 flatness_threshold=0.015, max_tau=0.20, flat_rel=None,
                 class_symmetric=False, anchored_rescue=False, random_state=0):
        self.contamination = contamination
        self.qw = quarantine_weight
        self.rescue_thr = rescue_threshold
        self.min_rescue = min_rescue
        self.flatness_threshold = flatness_threshold
        self.max_tau = max_tau
        self.flat_rel = flat_rel
        self.class_symmetric = class_symmetric
        self.anchored_rescue = anchored_rescue
        self.random_state = random_state

    def _estimate_contamination_gmm(self, ns):
        """P2: two-component Gaussian mixture on the score distribution."""
        n = len(ns)
        if n < 20 or np.std(ns) < 1e-9:
            return self._estimate_contamination(ns)
        x = np.asarray(ns).reshape(-1, 1)
        try:
            gm = GaussianMixture(n_components=2, covariance_type="full",
                                 random_state=self.random_state,
                                 n_init=2).fit(x)
        except Exception:
            return self._estimate_contamination(ns)
        noisy = int(np.argmax(gm.means_.ravel()))
        tau = float(np.mean(gm.predict_proba(x)[:, noisy] > 0.5))
        if tau <= 0.0 or tau >= 0.95:
            return self._estimate_contamination(ns)
        return float(np.clip(tau, 0.02, self.max_tau))

    def _pick_tau(self, ns):
        if self.contamination == "gmm":
            return self._estimate_contamination_gmm(ns)
        if self.contamination == "auto":
            return self._estimate_contamination(ns)
        return float(self.contamination)

    def _anchor_rescue(self, q_subs, anchor_subs):
        """P4: rescue a quarantined minority point only if its subspace coheres
        with the CONFIDENT minority manifold, rather than with other suspects."""
        out = np.zeros(len(q_subs), dtype=bool)
        if not anchor_subs or not len(q_subs):
            return out
        r = q_subs[0].shape[1]
        for i, q in enumerate(q_subs):
            best = 0.0
            for a in anchor_subs:
                pmat = q.T @ a
                best = max(best, float(np.sum(pmat * pmat)) / r)
            out[i] = best >= self.rescue_thr
        return out

    def _estimate_contamination(self, ns):
        s = np.sort(ns); n = len(s)
        if n < 20:
            return 0.05
        start = int(0.70 * n)
        tail = s[start:]; diffs = np.diff(tail)
        if len(diffs) < 3:
            return 0.05
        med_gap = np.median(diffs) + 1e-12
        big = np.where(diffs > 3.0 * med_gap)[0]
        if len(big) == 0:
            return 0.03
        boundary = start + big[0] + 1
        return float(np.clip((n - boundary) / n, 0.02, self.max_tau))

    def _subcluster_rescue(self, subs):
        n = len(subs)
        if n < self.min_rescue:
            return np.zeros(n, dtype=bool)
        r = subs[0].shape[1]
        aff = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                p = subs[i].T @ subs[j]
                aff[i, j] = aff[j, i] = np.sum(p * p) / r
        np.fill_diagonal(aff, 1.0)
        rescued = np.zeros(n, dtype=bool); visited = np.zeros(n, dtype=bool)
        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]; stack = [i]; visited[i] = True
            while stack:
                cur = stack.pop()
                for j in np.where((aff[cur] >= self.rescue_thr) & (~visited))[0]:
                    visited[j] = True; cluster.append(j); stack.append(j)
            if len(cluster) >= self.min_rescue:
                rescued[cluster] = True
        return rescued

    def clean(self, X, y, scores, subspaces):
        ns = scores["NS"]; n = len(X)
        cc = Counter(y); min_cls = min(cc, key=cc.get)
        is_min = (y == min_cls)

        flat = (ns.std() < self.flatness_threshold) or ((ns.max() - ns.min()) < 0.05)
        if self.flat_rel is not None:
            flat = flat or (ns.std() < self.flat_rel * max(abs(ns.mean()), 1e-12))
        if flat:
            return X, y, np.ones(n), np.ones(n, dtype=bool), 0.0

        tau = self._pick_tau(ns)

        if self.class_symmetric:
            th_hard = np.quantile(ns, 1 - tau / 2)
            th_soft = np.quantile(ns, 1 - 2 * tau)
            if th_soft >= th_hard:
                th_soft = th_hard * 0.8
            hard = ns > th_hard
            soft = (ns > th_soft) & (~hard)
            weights = np.ones(n)
            remove = hard & (~is_min); quarantine = hard & is_min
            weights[quarantine] = self.qw
            if soft.any():
                weights[soft] = np.clip(
                    1 - (ns[soft] - th_soft) / max(th_hard - th_soft, 1e-12), 0.1, 1.0)
        else:
            ns_maj, ns_min = ns[~is_min], ns[is_min]
            th_maj_hard = np.quantile(ns_maj, 1 - tau / 2) if len(ns_maj) else np.inf
            th_maj_soft = np.quantile(ns_maj, 1 - 2 * tau) if len(ns_maj) else np.inf
            if th_maj_soft >= th_maj_hard:
                th_maj_soft = th_maj_hard * 0.8
            tmn = tau / 2
            th_min_hard = np.quantile(ns_min, 1 - tmn / 2) if len(ns_min) else np.inf
            th_min_soft = np.quantile(ns_min, 1 - 2 * tmn) if len(ns_min) else np.inf
            if th_min_soft >= th_min_hard:
                th_min_soft = th_min_hard * 0.8

            weights = np.ones(n)
            hard = np.zeros(n, dtype=bool); soft = np.zeros(n, dtype=bool)
            hard[~is_min] = ns[~is_min] > th_maj_hard
            soft[~is_min] = (ns[~is_min] > th_maj_soft) & (~hard[~is_min])
            hard[is_min] = ns[is_min] > th_min_hard
            soft[is_min] = (ns[is_min] > th_min_soft) & (~hard[is_min])
            remove = hard & (~is_min); quarantine = hard & is_min
            weights[quarantine] = self.qw
            for cmask, ts, th in [(~is_min, th_maj_soft, th_maj_hard),
                                  (is_min, th_min_soft, th_min_hard)]:
                sm = soft & cmask
                if sm.any():
                    weights[sm] = np.clip(
                        1 - (ns[sm] - ts) / max(th - ts, 1e-12), 0.1, 1.0)

        q_idx = np.where(quarantine)[0]
        if len(q_idx) >= self.min_rescue:
            if self.anchored_rescue:
                min_idx = np.where(is_min)[0]
                conf = min_idx[(~hard[min_idx]) & (~soft[min_idx])]
                rescued = self._anchor_rescue([subspaces[i] for i in q_idx],
                                              [subspaces[i] for i in conf])
            else:
                rescued = self._subcluster_rescue([subspaces[i] for i in q_idx])
            weights[q_idx[rescued]] = 0.8

        keep = ~remove
        return X[keep], y[keep], weights[keep], keep, tau


# ============================================================================
# AMEND CORE 3/3 : SKLEARN-STYLE CLEANER
# ============================================================================
class AMENDCleaner(BaseEstimator):
    """Standalone spectral-coherence label-noise cleaner (imblearn-style API).

    fit_resample(X, y) -> (X_clean, y_clean)
    After fit: keep_mask_, sample_weight_, noise_scores_, scs_, fns_, tau_,
               n_removed_, skipped_
    """

    def __init__(self, k=8, r=5, alpha=0.5, contamination="auto",
                 use_scs=True, use_fns=True, class_symmetric=False,
                 standardize=True, proj_dim=None, projection_threshold=150,
                 max_tau=0.20, flat_rel=None, energy=0.90, jl_eps=0.5,
                 max_proj_dim=128, agree_weight=0.0, anchored_rescue=False,
                 random_state=42):
        self.k = k
        self.r = r
        self.alpha = alpha
        self.contamination = contamination
        self.use_scs = use_scs
        self.use_fns = use_fns
        self.class_symmetric = class_symmetric
        self.standardize = standardize
        self.proj_dim = proj_dim
        self.projection_threshold = projection_threshold
        self.max_tau = max_tau
        self.flat_rel = flat_rel
        self.energy = energy
        self.jl_eps = jl_eps
        self.max_proj_dim = max_proj_dim
        self.agree_weight = agree_weight
        self.anchored_rescue = anchored_rescue
        self.random_state = random_state

    def _auto_k(self, n, n_min):
        """Neighbourhood size from the data, never larger than the rarest class."""
        if self.k != "auto":
            return int(self.k)
        k = int(np.clip(round(np.sqrt(n_min)), 3, 12))
        return int(max(2, min(k, n_min - 1)))

    def _auto_proj_dim(self, n, d):
        """Projection width from Johnson-Lindenstrauss rather than a constant."""
        if self.proj_dim not in (None, "auto"):
            return int(self.proj_dim)
        if self.proj_dim is None:
            return None                       # legacy heuristic path
        try:
            from sklearn.random_projection import johnson_lindenstrauss_min_dim
            jl = int(johnson_lindenstrauss_min_dim(n_samples=max(int(n), 10),
                                                   eps=float(self.jl_eps)))
        except Exception:
            jl = 128
        return int(min(d, max(32, min(jl, int(self.max_proj_dim)))))

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
        n, d = X.shape
        n_min = min(Counter(y).values())
        k_res = self._auto_k(n, n_min)
        self.k_used_ = k_res

        if n_min < max(3, k_res + 1):
            self.keep_mask_ = np.ones(n, dtype=bool)
            self.sample_weight_ = np.ones(n)
            self.noise_scores_ = np.zeros(n); self.tau_ = 0.0
            self.skipped_ = True; self.n_removed_ = 0
            return X, y
        self.skipped_ = False

        pdim = self._auto_proj_dim(n, d)
        Xw = X
        if pdim is not None and d > pdim:
            Xw = GaussianRandomProjection(
                n_components=min(pdim, max(2, n - 1)),
                random_state=self.random_state).fit_transform(X)
        elif pdim is None and d > self.projection_threshold:
            pd_ = min(max(int(np.log(max(n, 2)) * 8), 32), 64, d)
            Xw = GaussianRandomProjection(
                n_components=pd_, random_state=self.random_state).fit_transform(X)
        self.proj_dim_used_ = Xw.shape[1]
        if self.standardize:
            Xw = StandardScaler().fit_transform(Xw)

        k_eff = min(k_res, n - 1)
        r_eff = (self.r if self.r == "auto"
                 else max(min(int(self.r), Xw.shape[1] - 1, max(k_eff - 1, 1)), 1))

        scorer = SpectralCoherenceScorer(
            k=k_eff, r=r_eff, alpha=self.alpha, energy=self.energy,
            use_scs=self.use_scs, use_fns=self.use_fns,
            agree_weight=self.agree_weight)
        scores = scorer.fit_score(Xw, y)

        cleaner = SoftCleaner(
            contamination=self.contamination, max_tau=self.max_tau,
            flat_rel=self.flat_rel, class_symmetric=self.class_symmetric,
            anchored_rescue=self.anchored_rescue,
            random_state=self.random_state)
        Xc, yc, w, keep, tau = cleaner.clean(X, y, scores, scores["subspaces"])

        self.keep_mask_ = keep; self.sample_weight_ = w
        self.noise_scores_ = scores["NS"]; self.scs_ = scores["SCS"]
        self.fns_ = scores["FNS"]; self.tau_ = float(tau)
        self.dis_ = scores.get("DIS"); self.base_score_ = scores.get("BASE")
        self.r_used_ = getattr(scorer, "r_used_", r_eff)
        self.n_removed_ = int((~keep).sum())
        return Xc, yc

    def fit(self, X, y):
        self.fit_resample(X, y); return self


# ============================================================================
# BASELINE : CONFIDENT LEARNING  (Northcutt, Jiang & Chuang, JAIR 2021)
# ============================================================================
class ConfidentLearningCleaner(BaseEstimator):
    """Confident Learning baseline (kept available for Exp6's AMEND-vs-CL
    crossover machinery; not run by default in this AMEND-only build)."""

    def __init__(self, base_estimator=None, cv=5, random_state=42,
                 min_examples_per_class=2, standardize=True):
        self.base_estimator = base_estimator
        self.cv = cv
        self.random_state = random_state
        self.min_examples_per_class = min_examples_per_class
        self.standardize = standardize

    def _estimator(self):
        if self.base_estimator is not None:
            return self.base_estimator
        steps = ([("sc", StandardScaler())] if self.standardize else []) + [
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=self.random_state))]
        return Pipeline(steps)

    @staticmethod
    def _confident_joint_prune(yi, P, K, min_per_class):
        n = len(yi)
        t = np.array([P[yi == j, j].mean() if (yi == j).any() else 1.0
                      for j in range(K)])
        above = P >= t[None, :]
        has = above.any(axis=1)
        conf = np.argmax(np.where(above, P, -np.inf), axis=1)

        C = np.zeros((K, K), dtype=int)
        for i in np.where(has)[0]:
            C[yi[i], conf[i]] += 1

        remove = np.zeros(n, dtype=bool)
        for a in range(K):
            for b in range(K):
                if a == b or C[a, b] <= 0:
                    continue
                cand = np.where(has & (yi == a) & (conf == b))[0]
                if len(cand) == 0:
                    continue
                margin = P[cand, b] - P[cand, a]
                remove[cand[np.argsort(-margin)][:min(C[a, b], len(cand))]] = True

        for a in range(K):
            idx = np.where(yi == a)[0]
            short = min_per_class - int((~remove[idx]).sum())
            if short > 0:
                back = idx[remove[idx]]
                remove[back[np.argsort(-P[back, a])][:short]] = False
        return ~remove

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
        n = len(y)
        classes = np.unique(y)
        cnt = Counter(y)
        n_min = min(cnt.values())

        k_cv = int(max(2, min(self.cv, n_min)))
        if n_min < 2 or len(classes) < 2:
            self.keep_mask_ = np.ones(n, dtype=bool)
            self.sample_indices_ = np.arange(n)
            self.n_removed_ = 0; self.skipped_ = True
            return X, y
        self.skipped_ = False

        yi = np.searchsorted(classes, y)
        try:
            P = cross_val_predict(
                self._estimator(), X, yi, method="predict_proba",
                cv=StratifiedKFold(n_splits=k_cv, shuffle=True,
                                   random_state=self.random_state))
        except Exception:
            self.keep_mask_ = np.ones(n, dtype=bool)
            self.sample_indices_ = np.arange(n)
            self.n_removed_ = 0; self.skipped_ = True
            return X, y

        keep = None
        if _CLEANLAB:
            try:
                issues = find_label_issues(
                    labels=yi, pred_probs=P, filter_by="prune_by_noise_rate",
                    min_examples_per_class=self.min_examples_per_class)
                keep = ~np.asarray(issues, dtype=bool)
            except Exception:
                keep = None
        if keep is None:
            keep = self._confident_joint_prune(
                yi, P, len(classes), self.min_examples_per_class)

        self.keep_mask_ = keep
        self.sample_indices_ = np.where(keep)[0]
        self.n_removed_ = int((~keep).sum())
        self.used_cleanlab_ = bool(_CLEANLAB)
        return X[keep], y[keep]

    def fit(self, X, y):
        self.fit_resample(X, y); return self


class AUMCleaner(BaseEstimator):
    """Area-Under-the-Margin baseline (Pleiss et al., NeurIPS 2020).
    Kept available for completeness; not run by default in this build."""

    def __init__(self, n_estimators=125, thresh_pct=99, random_state=42):
        self.n_estimators = n_estimators
        self.thresh_pct = thresh_pct
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y); n = len(y)
        self.keep_mask_ = np.ones(n, dtype=bool)
        self.sample_indices_ = np.arange(n)
        self.skipped_ = True
        cc = Counter(y)
        if min(cc.values()) < 5:
            return X, y
        rng = np.random.RandomState(self.random_state)
        classes = np.array(sorted(cc))
        thr_idx = rng.choice(n, size=min(max(int(0.02 * n), 5), n), replace=False)
        y_thr = np.array([rng.choice(classes[classes != y[i]]) for i in thr_idx])
        Xa = np.vstack([X, X[thr_idx]]); ya = np.concatenate([y, y_thr])
        is_thr = np.zeros(len(ya), dtype=bool); is_thr[n:] = True
        try:
            gb = GradientBoostingClassifier(
                n_estimators=self.n_estimators, max_depth=3,
                random_state=self.random_state).fit(Xa, ya)
            lab_pos = (ya == gb.classes_[1])
            margins = [np.where(lab_pos, np.ravel(dec), -np.ravel(dec))
                       for dec in gb.staged_decision_function(Xa)]
            aum = np.mean(np.vstack(margins), axis=0)
        except Exception:
            return X, y
        self.aum_ = aum[:n]
        keep = aum[:n] >= np.percentile(aum[is_thr], self.thresh_pct)
        if keep.sum() < min(cc.values()):
            keep = np.ones(n, dtype=bool)
        self.keep_mask_ = keep
        self.sample_indices_ = np.where(keep)[0]
        self.n_removed_ = int((~keep).sum())
        self.skipped_ = False
        return X[keep], y[keep]

    def fit(self, X, y):
        self.fit_resample(X, y); return self


# ============================================================================
# DATA LOADING
# ============================================================================
def _is_label_like_dtype(s):
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return True
    name = str(dt).lower()
    if name.startswith(("object", "str", "string", "bool", "category")):
        return True
    return pd.api.types.is_integer_dtype(dt)


def _is_text_dtype(s):
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return True
    return str(dt).lower().startswith(("object", "str", "string", "category"))


def detect_label_column(df, max_unique_ratio=0.2, max_unique_count=20):
    candidates = []
    for col in df.columns:
        u = df[col].nunique(dropna=True)
        ratio = u / max(1, df.shape[0])
        if 2 <= u <= max_unique_count and ratio < max_unique_ratio:
            if _is_label_like_dtype(df[col]):
                candidates.append((col, u))
    return min(candidates, key=lambda x: x[1])[0] if candidates else None


def resolve_label_column(name, df):
    if name in LABEL_OVERRIDES and LABEL_OVERRIDES[name] in df.columns:
        return LABEL_OVERRIDES[name]
    return detect_label_column(df)


def encode_dataset(df, label_col):
    df = df.copy().dropna(axis=1, how="all")
    y = df[label_col]
    if _is_text_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = LabelEncoder().fit_transform(y.fillna(-999))
    X = df.drop(columns=[label_col])
    for c in X.columns:
        if _is_text_dtype(X[c]):
            try:
                X[c] = LabelEncoder().fit_transform(X[c].astype(str).fillna("NA"))
            except Exception:
                X[c] = pd.factorize(X[c].astype(str).fillna("NA"))[0]
    Xa = np.array(X.to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    n_inf = int(np.isinf(Xa).sum())
    if n_inf:
        print(f"      ! {n_inf} infinite value(s) -> NaN (median-imputed per fold)")
        Xa[np.isinf(Xa)] = np.nan
    return Xa, np.asarray(y)


def load_binary_datasets(folder, ext=".csv"):
    """Return list of (name, X, y) for every binary CSV in the Drive folder."""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    out, audit = [], []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(ext):
            continue
        name = fn[:-len(ext)]
        try:
            df = pd.read_csv(os.path.join(folder, fn))
        except Exception as e:
            print(f"  x could not read {fn}: {e}"); continue
        label = resolve_label_column(name, df)
        if label is None:
            print(f"  - {name}: no label column detected — skipped"); continue
        if df[label].nunique() != 2:
            print(f"  - {name}: label '{label}' not binary "
                  f"({df[label].nunique()} classes) — skipped"); continue
        try:
            X, y = encode_dataset(df, label)
        except Exception as e:
            print(f"  x {name}: could not encode ({type(e).__name__}: {e}) "
                  f"— skipped"); continue
        out.append((name, X, y))
        cc = Counter(y); mn = min(cc, key=cc.get)
        audit.append({"Dataset": name, "Label": label, "Rows": len(y),
                      "Features": X.shape[1], "Minority": int(cc[mn]),
                      "MinorityLabel": int(mn),
                      "IR": round(max(cc.values()) / min(cc.values()), 2)})
    if audit:
        print("\nBinary datasets loaded (verify against your paper table):")
        print(pd.DataFrame(audit).to_string(index=False))
    return out


def minority_label(y):
    """Positive class for all end-task metrics = the minority class."""
    cc = Counter(np.asarray(y))
    return min(cc, key=cc.get)


# ============================================================================
# NOISE INJECTION
# ============================================================================
def inject_label_noise(y, rate, scheme="uniform", random_state=0):
    rng = np.random.RandomState(random_state)
    y = np.asarray(y).copy()
    cc = Counter(y); min_cls = min(cc, key=cc.get); maj_cls = max(cc, key=cc.get)
    flip = np.zeros(len(y), dtype=bool)
    if scheme == "uniform":
        flip = rng.rand(len(y)) < rate
    elif scheme == "minority_to_maj":
        idx = np.where(y == min_cls)[0]; flip[idx[rng.rand(len(idx)) < rate]] = True
    elif scheme == "maj_to_minority":
        idx = np.where(y == maj_cls)[0]; flip[idx[rng.rand(len(idx)) < rate]] = True
    else:
        raise ValueError(scheme)
    y_noisy = y.copy()
    y_noisy[flip] = np.where(y_noisy[flip] == min_cls, maj_cls, min_cls)
    return y_noisy, flip


# ============================================================================
# CLEANERS
# ============================================================================
def build_cleaner(cname, seed, amend_kwargs=None):
    if cname in AMEND_LADDER or cname == "AMEND":
        extra, use_w = AMEND_LADDER.get(cname, (dict(), False))
        kw = dict(AMEND_DEFAULTS); kw.update(extra); kw["random_state"] = seed
        if amend_kwargs:
            kw.update(amend_kwargs); kw["random_state"] = seed
        cl = AMENDCleaner(**kw)
        cl._use_weights = bool(use_w)
        return cl
    if cname == "AUM":
        return AUMCleaner(random_state=seed)
    if cname == "CL":
        return ConfidentLearningCleaner(random_state=seed)
    if cname == "None":
        return None
    if not _IMBLEARN:
        return None
    return {"ENN": EditedNearestNeighbours(),
            "AllKNN": AllKNN(),
            "Tomek": TomekLinks(),
            "NCR": NeighbourhoodCleaningRule(),
            "IHT": InstanceHardnessThreshold(random_state=seed)}.get(cname)


def cleaner_tag(cname):
    """Checkpoint identity of a cleaner."""
    if cname not in AMEND_LADDER and cname != "AMEND":
        return cname
    kw = dict(AMEND_DEFAULTS)
    extra, use_w = AMEND_LADDER.get(cname, (dict(), False))
    kw.update(extra)
    bits = [f"k{kw.get('k', 8)}", f"r{kw.get('r', 5)}", f"a{kw.get('alpha', 0.5)}",
            f"p{kw.get('proj_dim', None)}"]
    if not kw.get("use_scs", True):
        bits.append("noSCS")
    if not kw.get("use_fns", True):
        bits.append("noFNS")
    if kw.get("class_symmetric", False):
        bits.append("sym")
    if kw.get("contamination", "auto") != "auto":
        bits.append(f"c{kw['contamination']}@{kw.get('max_tau', 0.2)}")
    if kw.get("agree_weight", 0.0):
        bits.append(f"aw{kw['agree_weight']:g}")
    if kw.get("anchored_rescue", False):
        bits.append("anch")
    if use_w:
        bits.append("W")
    return f"{cname}[" + ",".join(bits) + "]"


def available_cleaners(names, verbose=True):
    """Drop cleaners this runtime cannot build, instead of running them as nothing."""
    ok, missing = [], []
    for nm in names:
        if nm in ("None", "CL", "AUM") or nm in AMEND_LADDER or _IMBLEARN:
            ok.append(nm)
        else:
            missing.append(nm)
    if missing and verbose:
        print(f"  ! DROPPED (imbalanced-learn not installed, and running them "
              f"would silently duplicate the 'None' row): {missing}")
    return ok


def _keep_mask_from(cleaner, n):
    """Recover a full-length boolean keep mask from any cleaner."""
    if cleaner is None:
        return np.ones(n, dtype=bool)
    km = getattr(cleaner, "keep_mask_", None)
    if km is not None and len(km) == n:
        return np.asarray(km, dtype=bool)
    si = getattr(cleaner, "sample_indices_", None)
    if si is not None:
        m = np.zeros(n, dtype=bool); m[np.asarray(si, dtype=int)] = True
        return m
    return None


def apply_cleaner(cleaner, X, y, cname="?"):
    """-> (X_clean, y_clean, keep_mask_full, sample_weights_kept, seconds)"""
    n = len(y)
    t0 = time.perf_counter()
    if cleaner is None:
        return X, y, np.ones(n, dtype=bool), np.ones(n), 0.0
    try:
        Xc, yc = cleaner.fit_resample(X, y)
        dt = time.perf_counter() - t0
        keep = _keep_mask_from(cleaner, n)
        w = getattr(cleaner, "sample_weight_", None)
        if w is None or len(w) != len(yc):
            w = np.ones(len(yc))
        return Xc, yc, keep, np.asarray(w, dtype=float), dt
    except Exception as e:
        _CLEANER_FAILURES[f"{cname}: {type(e).__name__}"] += 1
        return X, y, np.ones(n, dtype=bool), np.ones(n), time.perf_counter() - t0


# ============================================================================
# CLASSIFIERS  (multi-classifier evaluation)
# ============================================================================
_SUPPORTS_WEIGHT = {"RF", "SVM", "LR", "DT"}


def build_classifier(name, seed):
    if name == "RF":
        return Pipeline([("clf", RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=seed, n_jobs=-1))])
    if name == "SVM":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", SVC(C=1.0, kernel="rbf", gamma="scale",
                                     class_weight="balanced", random_state=seed))])
    if name == "kNN":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])
    if name == "LR":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                                    class_weight="balanced",
                                                    random_state=seed))])
    if name == "DT":
        return Pipeline([("clf", DecisionTreeClassifier(
            class_weight="balanced", random_state=seed))])
    raise ValueError(f"Unknown classifier: {name}")


def _pos_scores(clf, Xte, pos_label):
    """Continuous score for the POSITIVE (minority) class, for AUC."""
    classes = list(clf.classes_)
    if hasattr(clf, "predict_proba"):
        try:
            return clf.predict_proba(Xte)[:, classes.index(pos_label)]
        except Exception:
            pass
    if hasattr(clf, "decision_function"):
        df = np.ravel(clf.decision_function(Xte))
        return df if classes[-1] == pos_label else -df
    return clf.predict(Xte).astype(float)


def endtask_scores(y_true, y_pred, y_score, pos_label):
    neg = [c for c in np.unique(np.r_[y_true, y_pred]) if c != pos_label]
    neg = neg[0] if neg else (1 - pos_label)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[neg, pos_label]).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    try:
        auc = roc_auc_score((np.asarray(y_true) == pos_label).astype(int), y_score)
    except ValueError:
        auc = np.nan
    return dict(Recall=float(rec), Precision=float(prec), F1=float(f1),
                GMean=float(np.sqrt(rec * spec)), FPR=float(fpr),
                AUC=float(auc))


# ============================================================================
# NOISE-DETECTION METRICS
# ============================================================================
DET_KEYS = ["NoisePrec", "NoiseRec", "NoiseF1", "FRR", "FPR", "RemovalRate",
            "FlagPrec", "FlagRec", "FlagF1", "FlagRate", "NoiseRetained",
            "SkipRate"]


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(prec), float(rec), float(f1)


def detection_scores(flipped, keep, weights_kept=None):
    """How well the cleaner identified the injected label noise.

    FRR / FPR (identical here) = FP / (FP + TN): fraction of genuinely clean
    points discarded — the cleaner's false-positive rate, lower is better.
    """
    if keep is None:
        return {k: np.nan for k in DET_KEYS}
    keep = np.asarray(keep, dtype=bool)
    flipped = np.asarray(flipped, dtype=bool)
    n = len(flipped)
    clean = ~flipped
    removed = ~keep

    tp = int((removed & flipped).sum()); fp = int((removed & clean).sum())
    fn = int((keep & flipped).sum());    tn = int((keep & clean).sum())
    prec, rec, f1 = _prf(tp, fp, fn)
    frr = fp / (fp + tn) if (fp + tn) else 0.0

    flagged = removed.copy()
    if weights_kept is not None and len(weights_kept) == int(keep.sum()):
        kept_idx = np.where(keep)[0]
        flagged[kept_idx[np.asarray(weights_kept) <= FLAG_WEIGHT_THRESHOLD]] = True
    ftp = int((flagged & flipped).sum()); ffp = int((flagged & clean).sum())
    ffn = int((~flagged & flipped).sum())
    fprec, frec, ff1 = _prf(ftp, ffp, ffn)

    return {"NoisePrec": prec, "NoiseRec": rec, "NoiseF1": f1,
            "FRR": float(frr), "FPR": float(frr),
            "RemovalRate": float(removed.sum() / max(n, 1)),
            "FlagPrec": fprec, "FlagRec": frec, "FlagF1": ff1,
            "FlagRate": float(flagged.sum() / max(n, 1)),
            "NoiseRetained": float(1.0 - rec)}


# ============================================================================
# CHECKPOINTING / CRASH RESUME
# ============================================================================
def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Checkpoint:
    """Append-only resume log for one experiment."""

    def __init__(self, name, config, directory, enabled=True, fresh=False):
        self.name = name
        self.enabled = bool(enabled)
        self.store = {}
        self.reused = 0
        self.computed = 0
        self._fh = None
        self._closed = False
        self.path = os.path.join(directory, f"{name}.jsonl")
        self.meta_path = os.path.join(directory, f"{name}.meta.json")
        if not self.enabled:
            return
        os.makedirs(directory, exist_ok=True)
        sig = json.dumps(config, sort_keys=True, default=_jsonable)
        if fresh:
            self._archive("FRESH_START requested")
        elif os.path.exists(self.meta_path):
            try:
                with open(self.meta_path) as fh:
                    old = fh.read()
            except Exception:
                old = None
            if old is not None and old != sig:
                self._archive("protocol config changed since the last run")
        try:
            with open(self.meta_path, "w") as fh:
                fh.write(sig)
            self._load()
            self._fh = open(self.path, "a")
        except Exception as e:
            print(f"  [checkpoint:{name}] disabled — cannot write log ({e})")
            self.enabled = False

    def _archive(self, why):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        moved = False
        for p in (self.path, self.meta_path):
            if os.path.exists(p):
                try:
                    os.rename(p, f"{p}.{stamp}.bak"); moved = True
                except Exception:
                    pass
        if moved:
            print(f"  [checkpoint:{self.name}] {why} — previous log archived "
                  f"(.{stamp}.bak)")

    def _load(self):
        if not os.path.exists(self.path):
            return
        raw = defaultdict(lambda: ([], []))
        done, bad = set(), 0
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                k, kind = rec.get("k"), rec.get("kind")
                if kind == "__done__":
                    done.add(k)
                elif kind == "end":
                    raw[k][0].append(rec["r"])
                elif kind == "det":
                    raw[k][1].append(rec["r"])
        self.store = {k: v for k, v in raw.items() if k in done}
        for k in done:
            self.store.setdefault(k, ([], []))
        if bad:
            print(f"  [checkpoint:{self.name}] {bad} truncated line(s) ignored")
        if self.store:
            print(f"  [checkpoint:{self.name}] resuming — "
                  f"{len(self.store)} unit(s) already complete")

    @staticmethod
    def key(*parts):
        return "|".join(str(p) for p in parts)

    def has(self, k):
        return self.enabled and k in self.store

    def get(self, k):
        self.reused += 1
        e, d = self.store[k]
        return [dict(r) for r in e], [dict(r) for r in d]

    def put(self, k, end_rows, det_rows):
        self.computed += 1
        self.store[k] = (end_rows, det_rows)
        if self.enabled and self._closed:
            if not getattr(self, "_warned_closed", False):
                self._warned_closed = True
                print(f"  !! [checkpoint:{self.name}] write after close — this "
                      f"experiment is running outside the checkpointed block "
                      f"and will NOT resume. Fix the call ordering in main().")
            return
        if not self.enabled or self._fh is None:
            return
        try:
            for kind, rows in (("end", end_rows), ("det", det_rows)):
                for r in rows:
                    self._fh.write(json.dumps(
                        {"k": k, "kind": kind, "r": r}, default=_jsonable) + "\n")
            self._fh.write(json.dumps({"k": k, "kind": "__done__"}) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except Exception as e:
            print(f"  [checkpoint:{self.name}] write failed ({e}) — this unit "
                  "will be recomputed if the run is interrupted")

    def report(self):
        if not self.enabled:
            return
        print(f"  [checkpoint:{self.name}] {self.computed} unit(s) computed, "
              f"{self.reused} reused from the log")

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        self._closed = True


class _NullCheckpoint(Checkpoint):
    """Stand-in used when RESUME is off, so call sites stay identical."""

    def __init__(self):
        self.name = "off"; self.enabled = False; self.store = {}
        self.reused = 0; self.computed = 0; self._fh = None
        self._closed = False

    def report(self):
        pass

    def close(self):
        pass


# ============================================================================
# EVALUATION HARNESS
# ============================================================================
def clean_then_classify_one_fold(X, y, cleaner_factory, seed, n_splits, fold,
                                 noise_rate=0.0, scheme="uniform",
                                 classifiers=("RF",), want_det=True,
                                 pos_label=None, cname="?"):
    """A SINGLE (seed, fold) of clean-then-classify.

    This is the smallest re-runnable unit: the StratifiedKFold split is rebuilt
    deterministically from `seed`, then only the requested `fold` is evaluated.
    Rebuilding the whole split each call is cheap next to cleaning/fitting, and
    it keeps the fold assignment identical to the old per-seed loop, so
    per-fold and per-seed checkpoints agree numerically.
    """
    if pos_label is None:
        pos_label = minority_label(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    end_rows, det_rows = [], []

    splits = list(skf.split(X, y))
    if fold >= len(splits):
        return end_rows, det_rows
    tr, te = splits[fold]

    imp = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
    ytr, yte = y[tr].copy(), y[te].copy()

    flipped = np.zeros(len(ytr), dtype=bool)
    if noise_rate > 0:
        ytr, flipped = inject_label_noise(
            ytr, noise_rate, scheme, random_state=seed * 1000 + fold)

    cleaner = cleaner_factory(seed)
    Xc, yc, keep, w, dt = apply_cleaner(cleaner, Xtr, ytr, cname)

    if want_det and noise_rate > 0:
        d = detection_scores(flipped, keep, w)
        d["SkipRate"] = float(bool(getattr(cleaner, "skipped_", False)))
        d["clean_seconds"] = dt
        d["n_after"] = len(yc)
        d["kept_frac"] = len(yc) / max(len(ytr), 1)
        det_rows.append(d)

    if len(np.unique(yc)) < 2:
        return end_rows, det_rows  # cleaning wiped out a class — fold unusable

    sw = None
    if (USE_SAMPLE_WEIGHTS and getattr(cleaner, "_use_weights", True)
            and w is not None and len(w) == len(yc)
            and not np.allclose(w, 1.0)):
        sw = w

    for cl_name in classifiers:
        try:
            clf = build_classifier(cl_name, seed)
            if sw is not None and cl_name in _SUPPORTS_WEIGHT:
                clf.fit(Xc, yc, clf__sample_weight=sw)
            else:
                clf.fit(Xc, yc)
            yp = clf.predict(Xte)
            sc = _pos_scores(clf, Xte, pos_label)
            row = endtask_scores(yte, yp, sc, pos_label)
        except Exception as e:
            _CLASSIFIER_FAILURES[f"{cl_name}: {type(e).__name__}"] += 1
            continue
        row["Classifier"] = cl_name
        row["clean_seconds"] = dt
        row["n_after"] = len(yc)
        end_rows.append(row)

    return end_rows, det_rows


def clean_then_classify_cv(X, y, cleaner_factory, seed, n_splits,
                           noise_rate=0.0, scheme="uniform",
                           classifiers=("RF",), want_det=True,
                           pos_label=None, cname="?"):
    """One seed's worth of CV = every fold concatenated. Kept for callers that
    want a whole seed at once; multi_seed now goes fold-by-fold for resume."""
    e_all, d_all = [], []
    for fold in range(n_splits):
        e, d = clean_then_classify_one_fold(
            X, y, cleaner_factory, seed, n_splits, fold, noise_rate, scheme,
            classifiers, want_det, pos_label, cname)
        e_all += e; d_all += d
    return e_all, d_all


def multi_seed(X, y, fac, seeds, n_splits, nr, scheme,
               classifiers=("RF",), want_det=True, pos_label=None, cname="?",
               ck=None, key_parts=None):
    """Run one configuration over all seeds, ONE CHECKPOINTED UNIT PER FOLD.

    Granularity was one unit = one seed (all its folds); a disconnect mid-seed
    threw away every fold already computed for that seed. Now each (seed, fold)
    is its own unit, written and fsync'd the instant it finishes, so a kill
    costs at most a single fold. The per-seed key still exists as a fast path:
    if an OLD log has the whole seed done under the seed-level key, it is reused
    wholesale instead of forcing a recompute into the finer keys.
    """
    e_all, d_all = [], []
    have_keys = (ck is not None and key_parts is not None)
    for s in seeds:
        # Back-compat: reuse a legacy whole-seed unit if the old log has one.
        seed_key = Checkpoint.key(*key_parts, f"seed{s}") if have_keys else None
        if seed_key is not None and ck.has(seed_key):
            e, d = ck.get(seed_key)
            e_all += e; d_all += d
            continue
        for fold in range(n_splits):
            k = (Checkpoint.key(*key_parts, f"seed{s}", f"fold{fold}")
                 if have_keys else None)
            if k is not None and ck.has(k):
                e, d = ck.get(k)
            else:
                e, d = clean_then_classify_one_fold(
                    X, y, fac, s, n_splits, fold, nr, scheme,
                    classifiers, want_det, pos_label, cname)
                if k is not None:
                    ck.put(k, e, d)
            e_all += e; d_all += d
    return e_all, d_all


def agg(rows, by=None):
    """Mean over folds/seeds. `by` groups (e.g. by Classifier)."""
    if not rows:
        return pd.DataFrame() if by else {}
    df = pd.DataFrame(rows)
    if by:
        return df.groupby(by, as_index=False).mean(numeric_only=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return df.mean(numeric_only=True).to_dict()


def _r4(d):
    return {k: (round(v, 4) if isinstance(v, (int, float, np.floating)) and v == v
                else v) for k, v in d.items()}


def _subset(datasets, names):
    return datasets if names is None else [t for t in datasets if t[0] in names]


_DATASET_FAILURES = {}


def iter_datasets(datasets, experiment):
    """Yield datasets, surviving one that blows up."""
    import gc
    for name, X, y in datasets:
        try:
            yield name, X, y
        except GeneratorExit:
            raise
        except Exception as e:
            _DATASET_FAILURES[f"{experiment}/{name}"] = f"{type(e).__name__}: {e}"
            print(f"  !! {experiment}: dataset '{name}' FAILED "
                  f"({type(e).__name__}: {e}) — skipped, run continues")
        finally:
            gc.collect()


# ============================================================================
# EXPERIMENT 1 : INTERNAL ABLATION
# ============================================================================
def run_exp1(datasets, seeds, n_splits, noise_rate, scheme, results_dir,
             classifiers=("RF",), ck=None):
    ck = ck or _NullCheckpoint()
    print("\n" + "=" * 74)
    print("EXP 1 — INTERNAL ABLATION (k / r / alpha / terms / rule)")
    print("        end-task metrics (incl. FPR) + noise-detection incl. FRR/FPR")
    print("=" * 74)
    DEFAULT = dict(ABLATION_DEFAULT)
    GRID = {"k": [5, 8, 12, 20], "r": [2, 3, 5, 8],
            "alpha": [0.0, 0.25, 0.5, 0.75, 1.0]}
    TERMS = [("both", dict(use_scs=True, use_fns=True, alpha=0.5)),
             ("scs_only", dict(use_scs=True, use_fns=False)),
             ("fns_only", dict(use_scs=False, use_fns=True))]
    RULES = [("asymmetric", dict(class_symmetric=False)),
             ("symmetric", dict(class_symmetric=True))]
    COMBOS = [("k5_scsonly",  dict(k=5, alpha=1.0, use_scs=True, use_fns=False)),
              ("k8_scsonly",  dict(k=8, alpha=1.0, use_scs=True, use_fns=False)),
              ("k5_alpha0.5", dict(k=5, alpha=0.5, use_scs=True, use_fns=True)),
              ("k8_alpha0.5", dict(k=8, alpha=0.5, use_scs=True, use_fns=True)),
              ("adaptive_all", dict(k="auto", r="auto", proj_dim="auto",
                                    alpha=1.0, use_fns=False)),
              ("fixed_k5r5",   dict(k=5, r=5, proj_dim=None,
                                    alpha=1.0, use_fns=False))]

    def fac(extra):
        def f(seed):
            kw = dict(DEFAULT); kw.update(extra); kw["random_state"] = seed
            return AMENDCleaner(**kw)
        return f

    rows = []
    for name, X, y in iter_datasets(datasets, "exp1"):
      try:
        pos = minority_label(y)
        end, _ = multi_seed(X, y, lambda s: None, seeds, n_splits, noise_rate,
                            scheme, classifiers, False, pos, "None",
                            ck=ck, key_parts=(name, "control", "no_clean"))
        rows += _exp1_rows(name, "control", "no_clean", end, [], classifiers)

        def _extra(param, v):
            if param == "alpha":
                return dict(alpha=v, use_scs=True, use_fns=True)
            return {param: v}

        configs = ([(p, v, _extra(p, v)) for p, vs in GRID.items() for v in vs]
                   + [("terms", lab, ex) for lab, ex in TERMS]
                   + [("rule", lab, ex) for lab, ex in RULES]
                   + [("combo", lab, ex) for lab, ex in COMBOS])
        for factor, value, extra in configs:
            end, det = multi_seed(X, y, fac(extra), seeds, n_splits, noise_rate,
                                  scheme, classifiers, True, pos, "AMEND",
                                  ck=ck,
                                  key_parts=(name, factor, value,
                                             _cfg_tag(DEFAULT, extra)))
            rows += _exp1_rows(name, factor, value, end, det, classifiers)
        print(f"  done: {name}")
      except Exception as e:
        _DATASET_FAILURES[f"exp1/{name}"] = f"{type(e).__name__}: {e}"
        print(f"  !! exp1: '{name}' FAILED ({type(e).__name__}: {e}) — skipped")
    ck.report()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp1_ablation.csv"), index=False)
    print("\nMean end-task F1 by factor/value (across datasets):")
    print(df.groupby(["Factor", "Value"])["F1"].mean().round(4).to_string())
    det_cols = [c for c in ["NoiseF1", "NoiseRec", "NoisePrec", "FRR", "FPR",
                            "SkipRate"]
                if c in df.columns]
    if det_cols:
        print("\nMean noise-detection by factor/value:")
        print(df.dropna(subset=det_cols).groupby(["Factor", "Value"])[det_cols]
                .mean().round(4).to_string())
    for fac_name, grp in df[df.Factor != "control"].groupby("Factor"):
        if grp.Value.nunique() > 1 and grp.groupby("Value").F1.mean().round(6).nunique() == 1:
            print(f"  !! FACTOR '{fac_name}' HAD NO EFFECT — every value produced "
                  f"identical results. That is a wiring bug, not a finding; the "
                  f"swept parameter is being overridden or ignored.")
    print("[exp1] saved exp1_ablation.csv")
    return df


def _cfg_tag(default, extra):
    """Stable short name for a resolved ablation configuration."""
    kw = dict(default); kw.update(extra)
    return ",".join(f"{k}={kw[k]}" for k in sorted(kw))


def _exp1_rows(dataset, factor, value, end_rows, det_rows, classifiers):
    d = _r4(agg(det_rows)) if det_rows else {k: np.nan for k in DET_KEYS}
    out = []
    per_clf = agg(end_rows, by="Classifier")
    for cl in classifiers:
        r = {"Dataset": dataset, "Factor": factor, "Value": value,
             "Classifier": cl}
        if len(per_clf) and (per_clf["Classifier"] == cl).any():
            e = per_clf[per_clf["Classifier"] == cl].iloc[0].to_dict()
            e.pop("Classifier", None)
            r.update(_r4(e))
        r.update(d)
        out.append(r)
    return out


# ============================================================================
# EXPERIMENT 2 : NOISE ROBUSTNESS
# ============================================================================
def run_exp2(datasets, seeds, n_splits, cleaners, rates, schemes, results_dir,
             classifiers=("RF",), ck=None):
    ck = ck or _NullCheckpoint()
    print("\n" + "=" * 74)
    print("EXP 2 — NOISE ROBUSTNESS (end-task F1 + detection vs injected noise)")
    print("=" * 74)
    rows = []
    for name, X, y in datasets:
      try:
        pos = minority_label(y)
        for scheme in schemes:
            for rate in rates:
                for cname in cleaners:
                    fac = (lambda cn: (lambda s: build_cleaner(cn, s)))(cname)
                    e_all, d_all = multi_seed(
                        X, y, fac, seeds, n_splits, rate, scheme, classifiers,
                        True, pos, cname, ck=ck,
                        key_parts=(name, scheme, rate, cleaner_tag(cname)))
                    det = _r4(agg(d_all)) if d_all else {k: np.nan for k in DET_KEYS}
                    per_clf = agg(e_all, by="Classifier")
                    for cl in classifiers:
                        row = {"Dataset": name, "Scheme": scheme,
                               "NoiseRate": rate, "Cleaner": cname,
                               "Classifier": cl}
                        if len(per_clf) and (per_clf["Classifier"] == cl).any():
                            e = per_clf[per_clf["Classifier"] == cl].iloc[0].to_dict()
                            e.pop("Classifier", None)
                            row.update(_r4(e))
                        row.update(det)
                        rows.append(row)
            print(f"  {name} / {scheme} done")
      except Exception as e:
        _DATASET_FAILURES[f"exp2/{name}"] = f"{type(e).__name__}: {e}"
        print(f"  !! exp2: '{name}' FAILED ({type(e).__name__}: {e}) — skipped, "
              f"run continues with the remaining datasets")
      finally:
        __import__("gc").collect()
    ck.report()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp2_noise.csv"), index=False)

    _exp2_plots(df, results_dir, classifiers[0])

    print("\nUniform-noise end-task F1 by rate x cleaner "
          f"(classifier={classifiers[0]}):")
    sub = df[(df.Scheme == "uniform") & (df.Classifier == classifiers[0])]
    print(sub.pivot_table(index="NoiseRate", columns="Cleaner",
                          values="F1").round(4).to_string())
    print("\nUniform-noise DETECTION F1 by rate x cleaner:")
    print(sub[sub.NoiseRate > 0].pivot_table(
        index="NoiseRate", columns="Cleaner", values="NoiseF1")
        .round(4).to_string())
    print("\nUniform-noise FALSE REMOVAL RATE (FRR/FPR) by rate x cleaner:")
    print(sub[sub.NoiseRate > 0].pivot_table(
        index="NoiseRate", columns="Cleaner", values="FRR")
        .round(4).to_string())
    print("[exp2] saved exp2_noise.csv (+ plots)")
    return df


def _exp2_plots(df, results_dir, clf_name):
    panels = [("F1", "End-task F1 (mean across datasets)",
               "Robustness to label noise: clean-then-classify F1",
               "exp2_noise_f1.png", False),
              ("NoiseF1", "Noise-detection F1",
               "Noise detection quality vs injected noise",
               "exp2_detection_f1.png", True),
              ("NoiseRec", "Noise-detection recall",
               "Share of injected noise removed", "exp2_detection_recall.png", True),
              ("FRR", "False removal rate (clean points discarded)",
               "Collateral damage: clean points wrongly removed",
               "exp2_false_removal_rate.png", True)]
    for col, ylab, title, fname, drop_zero in panels:
        try:
            sub = df[(df.Scheme == "uniform") & (df.Classifier == clf_name)]
            if drop_zero:
                sub = sub[sub.NoiseRate > 0]
            piv = sub.pivot_table(index="NoiseRate", columns="Cleaner", values=col)
            if piv.empty or piv.isna().all().all():
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for c in piv.columns:
                ax.plot(piv.index, piv[c], marker="o", label=c,
                        linewidth=(2.5 if c.startswith("AMEND") else 1.3),
                        alpha=(1.0 if c.startswith("AMEND") else 0.7))
            ax.set_xlabel("Injected label-noise rate (uniform)")
            ax.set_ylabel(ylab); ax.set_title(title)
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir, fname), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"  plot {fname} skipped ({e})")


# ============================================================================
# EXPERIMENT 3 : BASELINE COMPARISON, MULTI-CLASSIFIER, + STATS
# ============================================================================
def cliffs_delta(a, b):
    """Non-parametric effect size."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not len(a) or not len(b):
        return np.nan
    gt = int(sum(x > yy for x in a for yy in b))
    lt = int(sum(x < yy for x in a for yy in b))
    return (gt - lt) / (len(a) * len(b))


_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
                8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
                14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489,
                19: 3.517, 20: 3.544}


def _friedman_wilcoxon(piv, metric, ref, alpha=0.05, lower_is_better=False):
    """Friedman + Iman-Davenport, then Holm-corrected Wilcoxon vs `ref`."""
    piv = piv.dropna()
    algos = piv.columns.tolist(); n, k = piv.shape
    if n < 3 or k < 2:
        return f"[{metric}] need >=3 datasets & >=2 cleaners (got {piv.shape})\n\n"
    sign = 1.0 if lower_is_better else -1.0
    ranks = np.vstack([rankdata(sign * piv.iloc[i].values) for i in range(n)])
    avg = ranks.mean(axis=0)
    chi2 = (12 * n / (k * (k + 1))) * (np.sum(avg ** 2) - k * (k + 1) ** 2 / 4)
    denom = (n * (k - 1) - chi2)
    FF = ((n - 1) * chi2) / denom if denom else float("nan")
    p = 1 - f_dist.cdf(FF, k - 1, (k - 1) * (n - 1))
    lines = [f"=== {metric} "
             f"({'lower is better' if lower_is_better else 'higher is better'}) ===",
             f"N datasets={n}  K cleaners={k}",
             "Avg ranks (lower=better): " +
             ", ".join(f"{a}={r:.2f}" for a, r in
                       sorted(zip(algos, avg), key=lambda t: t[1])),
             f"Friedman chi2={chi2:.4f}  Iman-Davenport F={FF:.4f}  "
             f"p={p:.5f}  {'SIG' if p < alpha else 'ns'}",
             f"Nemenyi CD(0.05)={_NEMENYI_Q05.get(k, 3.544) * np.sqrt(k * (k + 1) / (6.0 * n)):.3f}"]
    if ref in piv.columns:
        prs = []
        for a in [x for x in algos if x != ref]:
            d = piv[ref].values - piv[a].values
            eff = cliffs_delta(piv[ref].values, piv[a].values)
            if int((d != 0).sum()) < 5:
                prs.append((a, float(np.mean(d)), np.nan, eff))
            else:
                _, pv = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
                prs.append((a, float(np.mean(d)), float(pv), eff))
        testable = sorted([x for x in prs if x[2] == x[2]], key=lambda t: t[2])
        m = len(testable); holm = []; run = 0.0
        for i, (_, _, pv, _) in enumerate(testable):
            adj = max(min(pv * (m - i), 1.0), run); run = adj; holm.append(adj)
        holm_map = {testable[i][0]: holm[i] for i in range(m)}
        lines.append(f"Wilcoxon vs {ref} (Holm-adjusted):")
        for a, md, pv, eff in prs:
            ph = holm_map.get(a, np.nan)
            sig = "SIG" if (ph == ph and ph < alpha) else "   "
            lines.append(f"   {ref} vs {a:14s} dmean={md:+.4f}  "
                         f"p_holm={(ph if ph == ph else float('nan')):.4f}  "
                         f"delta={eff:+.3f}  {sig}")
    return "\n".join(lines) + "\n\n"


def run_exp3(datasets, seeds, n_splits, cleaners, noise_rate, scheme,
             results_dir, classifiers=("RF",), ck=None):
    ck = ck or _NullCheckpoint()
    print("\n" + "=" * 74)
    print("EXP 3 — AMEND vs no-clean control x MULTIPLE CLASSIFIERS")
    print(f"        classifiers: {', '.join(classifiers)}")
    print(f"        injected noise: {noise_rate:.0%} ({scheme}), train fold only")
    print("=" * 74)
    end_rows, det_rows = [], []
    for name, X, y in datasets:
      try:
        pos = minority_label(y)
        for cname in cleaners:
            fac = (lambda cn: (lambda s: build_cleaner(cn, s)))(cname)
            e_all, d_all = multi_seed(X, y, fac, seeds, n_splits, noise_rate,
                                      scheme, classifiers, True, pos, cname,
                                      ck=ck,
                                      key_parts=(name, cleaner_tag(cname)))
            per_clf = agg(e_all, by="Classifier")
            for cl in classifiers:
                row = {"Dataset": name, "Cleaner": cname, "Classifier": cl}
                if len(per_clf) and (per_clf["Classifier"] == cl).any():
                    e = per_clf[per_clf["Classifier"] == cl].iloc[0].to_dict()
                    e.pop("Classifier", None)
                    row.update(_r4(e))
                end_rows.append(row)
            if d_all:
                dr = {"Dataset": name, "Cleaner": cname}
                dr.update(_r4(agg(d_all)))
                det_rows.append(dr)
        print(f"  done: {name}")
      except Exception as e:
        _DATASET_FAILURES[f"exp3/{name}"] = f"{type(e).__name__}: {e}"
        print(f"  !! exp3: '{name}' FAILED ({type(e).__name__}: {e}) — skipped")
      finally:
        __import__("gc").collect()
    ck.report()

    df = pd.DataFrame(end_rows)
    ddf = pd.DataFrame(det_rows)
    df.to_csv(os.path.join(results_dir, "exp3_baselines.csv"), index=False)
    ddf.to_csv(os.path.join(results_dir, "exp3_detection.csv"), index=False)

    END_METRICS = ["Recall", "Precision", "F1", "GMean", "FPR", "AUC"]
    DET_REPORT = ["NoisePrec", "NoiseRec", "NoiseF1", "FRR", "FPR", "RemovalRate",
                  "FlagPrec", "FlagRec", "FlagF1", "SkipRate"]

    # ---- end-task summaries ----
    print("\nEnd-task F1 — cleaner x classifier (mean across datasets):")
    f1_tab = df.pivot_table(index="Cleaner", columns="Classifier",
                            values="F1").round(4)
    f1_tab["MEAN"] = f1_tab.mean(axis=1).round(4)
    f1_tab = f1_tab.sort_values("MEAN", ascending=False)
    print(f1_tab.to_string())

    overall = (df.groupby("Cleaner")[END_METRICS].mean().round(4)
                 .sort_values("F1", ascending=False))
    print("\nEnd-task, averaged over datasets AND classifiers (incl. FPR):")
    print(overall.to_string())

    # ---- detection summary ----
    det_summary = pd.DataFrame()
    if not ddf.empty:
        cols = [c for c in DET_REPORT if c in ddf.columns]
        det_summary = (ddf.groupby("Cleaner")[cols].mean().round(4)
                          .sort_values("NoiseF1", ascending=False))
        print("\nNOISE DETECTION (mean across datasets) — how well each cleaner")
        print(f"identified the {noise_rate:.0%} injected flips:")
        print(det_summary.to_string())
        print("\n  NoisePrec/Rec/F1 = precision/recall/F1 of REMOVED vs injected-noise")
        print("  FRR / FPR        = false removal rate: clean points discarded /")
        print("                     all clean points (identical; lower is better)")
        print("  SkipRate         = fraction of folds where the cleaner refused to run")
        print("  Flag*            = same, counting AMEND's quarantined points as")
        print("                     detections")

    # ---- statistics ----
    txt = ["AMEND EXPERIMENT 3 — STATISTICAL REPORT",
           f"seeds={seeds}  folds={n_splits}  noise={noise_rate} ({scheme})",
           f"classifiers={list(classifiers)}",
           f"sample weights passed to classifiers: {USE_SAMPLE_WEIGHTS}",
           f"NOTE: with N={df.Dataset.nunique()} datasets the smallest two-sided "
           f"Wilcoxon p attainable is {2 ** -(df.Dataset.nunique() - 1):.4f} "
           "before Holm correction;",
           "      pairs with fewer than 5 non-zero differences are reported as nan.",
           "", "END-TASK F1 — cleaner x classifier", f1_tab.to_string(), "",
           "END-TASK, averaged over datasets and classifiers", overall.to_string(),
           "", ""]
    if not det_summary.empty:
        txt += ["NOISE DETECTION (mean across datasets)",
                det_summary.to_string(), "", ""]

    txt.append("#" * 70)
    txt.append("# PER-CLASSIFIER TESTS ACROSS DATASETS (cleaners compared)")
    txt.append("#" * 70 + "\n")
    for cl in classifiers:
        sub = df[df.Classifier == cl]
        txt.append(f"----------  CLASSIFIER: {cl}  ----------\n")
        for metric in ["Recall", "F1", "GMean", "FPR", "AUC"]:
            piv = sub.pivot_table(index="Dataset", columns="Cleaner", values=metric)
            txt.append(_friedman_wilcoxon(piv, f"{cl} / {metric}", ref=REF_CLEANER,
                                          lower_is_better=(metric == "FPR")))

    if not ddf.empty:
        txt.append("#" * 70)
        txt.append("# NOISE-DETECTION TESTS ACROSS DATASETS "
                   "(classifier-independent)")
        txt.append("#" * 70 + "\n")
        det_no_none = ddf[ddf.Cleaner != "None"]
        for metric, lower in [("NoiseF1", False), ("NoiseRec", False),
                              ("NoisePrec", False), ("FRR", True),
                              ("FlagF1", False)]:
            if metric not in det_no_none.columns:
                continue
            piv = det_no_none.pivot_table(index="Dataset", columns="Cleaner",
                                          values=metric)
            txt.append(_friedman_wilcoxon(piv, metric, ref=REF_CLEANER,
                                          lower_is_better=lower))

    with open(os.path.join(results_dir, "exp3_stats.txt"), "w") as fh:
        fh.write("\n".join(txt))

    _exp3_plots(df, ddf, results_dir)
    print("[exp3] saved exp3_baselines.csv, exp3_detection.csv, "
          "exp3_stats.txt (+ plots)")
    return df, ddf


def _exp3_plots(df, ddf, results_dir):
    try:
        piv = df.pivot_table(index="Cleaner", columns="Classifier", values="F1")
        piv = piv.reindex(piv.mean(axis=1).sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(piv.index)); w = 0.8 / max(len(piv.columns), 1)
        for i, cl in enumerate(piv.columns):
            ax.bar(x + i * w - 0.4 + w / 2, piv[cl].values, w, label=cl)
        ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=20)
        ax.set_ylabel("End-task F1 (mean across datasets)")
        ax.set_title("Cleaner x classifier: does the gain survive the learner?")
        ax.legend(ncol=len(piv.columns), fontsize=8); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "exp3_f1_by_classifier.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  plot exp3_f1_by_classifier skipped ({e})")

    try:
        if ddf.empty:
            return
        cols = ["NoiseRec", "FRR"] + (["FlagRec"] if "FlagRec" in ddf else [])
        s = ddf[ddf.Cleaner != "None"].groupby("Cleaner")[cols].mean()
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for cl, r in s.iterrows():
            big = cl.startswith("AMEND")
            ax.scatter(r["FRR"], r["NoiseRec"], s=(200 if big else 90),
                       marker=("*" if big else "o"), zorder=3)
            ax.annotate(cl, (r["FRR"], r["NoiseRec"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=9)
            if big and "FlagRec" in r and r["FlagRec"] > r["NoiseRec"] + 1e-9:
                ax.scatter(r["FRR"], r["FlagRec"], s=200, marker="*",
                           facecolors="none", edgecolors="C4", linewidths=1.6,
                           zorder=3, label="AMEND incl. quarantined")
                ax.annotate("", xy=(r["FRR"], r["FlagRec"]),
                            xytext=(r["FRR"], r["NoiseRec"]),
                            arrowprops=dict(arrowstyle="->", color="C4",
                                            lw=1.2, ls="--"))
        ax.set_xlabel("False removal rate (FRR/FPR) — lower better")
        ax.set_ylabel("Noise detection recall — higher better")
        ax.set_title("Detection trade-off: noise caught vs clean data destroyed")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "exp3_detection_tradeoff.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  plot exp3_detection_tradeoff skipped ({e})")


# ============================================================================
# EXPERIMENT 4 : RUNTIME & SCALABILITY  (synthetic data)
# ============================================================================
def _make_manifold(n, d, imbalance=0.3, seed=0):
    rng = np.random.RandomState(seed)
    n_min = max(int(n * imbalance), 5); n_maj = n - n_min; latent = 5
    def blob(m, shift):
        return rng.randn(m, latent) @ rng.randn(latent, d) + shift + 0.1 * rng.randn(m, d)
    X = np.vstack([blob(n_maj, 0.0), blob(n_min, 3.0)])
    y = np.r_[np.zeros(n_maj), np.ones(n_min)].astype(int)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _time_clean(cname, X, y, seed, repeats=3):
    """Median wall-clock of fit_resample."""
    ts = []
    for _ in range(repeats):
        cl = build_cleaner(cname, seed)
        if cl is None:
            return np.nan
        t0 = time.perf_counter()
        try:
            cl.fit_resample(X, y)
        except Exception:
            return np.nan
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) if ts else np.nan


def run_exp4(cleaners, n_grid, d_grid, d_fixed, n_fixed, seed, results_dir,
             ck=None):
    ck = ck or _NullCheckpoint()
    print("\n" + "=" * 74)
    print("EXP 4 — RUNTIME & SCALABILITY (synthetic manifold data)")
    print("=" * 74)
    rows = []

    def _cached(axis, n, d, cname, compute):
        """One timing measurement = one checkpointed unit."""
        k = Checkpoint.key("exp4", axis, n, d, cleaner_tag(cname))
        if ck.has(k):
            e, _ = ck.get(k)
            if e:
                return e[0]
        row = {"Axis": axis, "n": n, "d": d, "Cleaner": cname,
               "seconds": round(float(compute()), 4)}
        ck.put(k, [row], [])
        return row

    for n in n_grid:
        X = y = None
        for c in cleaners:
            k = Checkpoint.key("exp4", "n", n, d_fixed, cleaner_tag(c))
            if not ck.has(k) and X is None:
                X, y = _make_manifold(n, d_fixed, seed=seed)
            rows.append(_cached("n", n, d_fixed, c,
                                lambda c=c: _time_clean(c, X, y, seed)))
        print(f"  n={n} done")

    for d in d_grid:
        X = y = None
        todo = [c for c in cleaners] + ["AMEND(proj64)"]
        if any(not ck.has(Checkpoint.key("exp4", "d", n_fixed, d, cleaner_tag(c)))
               for c in todo):
            X, y = _make_manifold(n_fixed, d, seed=seed)
        for c in cleaners:
            rows.append(_cached("d", n_fixed, d, c,
                                lambda c=c: _time_clean(c, X, y, seed)))

        def _proj():
            cl = AMENDCleaner(proj_dim=64, random_state=seed)
            t0 = time.perf_counter()
            try:
                cl.fit_resample(X, y)
                return time.perf_counter() - t0
            except Exception:
                return np.nan

        rows.append(_cached("d", n_fixed, d, "AMEND(proj64)", _proj))
        print(f"  d={d} done")

    ck.report()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp4_runtime.csv"), index=False)

    sub = df[df.Axis == "n"]; exps = []
    for c in sub.Cleaner.unique():
        s = sub[sub.Cleaner == c].dropna()
        if len(s) >= 3 and (s.seconds > 0).all():
            beta = np.polyfit(np.log(s.n), np.log(s.seconds), 1)[0]
            exps.append({"Cleaner": c, "beta_n": round(float(beta), 3)})
    if exps:
        edf = pd.DataFrame(exps)
        edf.to_csv(os.path.join(results_dir, "exp4_runtime_exponents.csv"),
                   index=False)
        print("\nEmpirical scaling in n (time ~ n^beta):")
        print(edf.to_string(index=False))

    try:
        for axis, xcol, fname in [("n", "n", "exp4_runtime_n.png"),
                                  ("d", "d", "exp4_runtime_d.png")]:
            s = df[df.Axis == axis]
            fig, ax = plt.subplots(figsize=(8, 5))
            for c in s.Cleaner.unique():
                ss = s[s.Cleaner == c].sort_values(xcol)
                ax.plot(ss[xcol], ss.seconds, marker="o", label=c,
                        linewidth=(2.5 if c.startswith("AMEND") else 1.3))
            ax.set_xlabel(xcol); ax.set_ylabel("clean time (s)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"Cleaning runtime vs {xcol} (log-log)")
            ax.legend(); ax.grid(alpha=0.3, which="both")
            fig.tight_layout(); fig.savefig(os.path.join(results_dir, fname), dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"  plots skipped ({e})")
    print("[exp4] saved exp4_runtime.csv (+ plots)")
    return df


# ============================================================================
# EXPERIMENT 5 : METRIC-FAIR DETECTION  (ranking quality, not removal counts)
# ============================================================================
def noise_ranking(method, X, y, seed):
    """Continuous 'this label looks wrong' score, one per sample (higher = worse)."""
    n = len(y)
    if method in AMEND_LADDER or method == "AMEND":
        c = build_cleaner(method, seed)
        c.fit_resample(X, y)
        if getattr(c, "skipped_", False):
            return None
        return np.asarray(c.noise_scores_, dtype=float)

    if method == "AUM-score":
        c = AUMCleaner(random_state=seed)
        c.fit_resample(X, y)
        if getattr(c, "skipped_", True) or not hasattr(c, "aum_"):
            return None
        return -np.asarray(c.aum_, dtype=float)

    classes = np.unique(y)
    if len(classes) < 2:
        return None
    yi = np.searchsorted(classes, y)
    n_min = int(min(Counter(yi).values()))

    if method in ("CL", "IHT-score"):
        if n_min < 2:
            return None
        cv = StratifiedKFold(n_splits=int(max(2, min(5, n_min))), shuffle=True,
                             random_state=seed)
        est = (Pipeline([("sc", StandardScaler()),
                         ("c", LogisticRegression(max_iter=2000,
                                                  class_weight="balanced",
                                                  random_state=seed))])
               if method == "CL" else
               RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=seed, n_jobs=-1))
        try:
            P = cross_val_predict(est, X, yi, cv=cv, method="predict_proba")
        except Exception:
            return None
        return 1.0 - P[np.arange(n), yi]

    if method == "kNN-disagree":
        k = int(max(1, min(5, n - 1)))
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
        idx = nn.kneighbors(X, return_distance=False)[:, 1:]
        return (yi[idx] != yi[:, None]).mean(axis=1).astype(float)

    if method == "Tomek-link":
        nn = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(X)
        nb = nn.kneighbors(X, return_distance=False)[:, 1]
        return ((yi[nb] != yi) & (nb[nb] == np.arange(n))).astype(float)
    return None


def run_exp5(datasets, seeds, n_splits, methods, noise_rate, scheme, results_dir,
             ck=None):
    ck = ck or _NullCheckpoint()
    print("\n" + "=" * 74)
    print("EXP 5 — METRIC-FAIR DETECTION (ROC-AUC / AP of each method's ranking)")
    print(f"        injected noise: {noise_rate:.0%} ({scheme}); no thresholds applied")
    print("=" * 74)
    rows = []
    for name, X, y in datasets:
        for m in methods:
            for seed in seeds:
                key = Checkpoint.key("exp5", name, cleaner_tag(m),
                                     noise_rate, scheme, seed)
                if ck.has(key):
                    e, _ = ck.get(key); rows += e; continue
                got = []
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed)
                for fold, (tr, _) in enumerate(skf.split(X, y)):
                    Xtr = SimpleImputer(strategy="median").fit_transform(X[tr])
                    ytr, flip = inject_label_noise(
                        y[tr], noise_rate, scheme, random_state=seed * 1000 + fold)
                    if flip.sum() < 2 or flip.all():
                        continue
                    try:
                        sc = noise_ranking(m, Xtr, ytr, seed)
                    except Exception:
                        sc = None
                    if sc is None or not np.isfinite(sc).all():
                        continue
                    f = flip.astype(int)
                    got.append({"Dataset": name, "Method": m,
                                "AUC": float(roc_auc_score(f, sc)),
                                "AP": float(average_precision_score(f, sc)),
                                "LiftAP": float(average_precision_score(f, sc)
                                                / max(f.mean(), 1e-9))})
                ck.put(key, got, [])
                rows += got
        print(f"  done: {name}")
    ck.report()
    df = pd.DataFrame(rows)
    if df.empty:
        print("[exp5] no usable folds"); return df
    df.to_csv(os.path.join(results_dir, "exp5_ranking.csv"), index=False)

    per_ds = df.groupby(["Dataset", "Method"])[["AUC", "AP", "LiftAP"]].mean().reset_index()
    summ = per_ds.groupby("Method")[["AUC", "AP", "LiftAP"]].mean().round(4) \
                 .sort_values("AUC", ascending=False)
    print("\nRanking quality (mean across datasets):")
    print(summ.to_string())
    print("\n  AUC 0.50 = the score is uninformative about label noise")
    print("  LiftAP   = average precision over the noise rate; 1.0 = chance")

    txt = ["EXP 5 — METRIC-FAIR DETECTION", f"noise={noise_rate} ({scheme})",
           f"seeds={list(seeds)} folds={n_splits}", "", summ.to_string(), "", ""]
    for metric in ["AUC", "AP"]:
        piv = per_ds.pivot_table(index="Dataset", columns="Method", values=metric)
        txt.append(_friedman_wilcoxon(piv, metric, ref="AMEND"
                                      if "AMEND" in piv.columns else "AMEND"))
    with open(os.path.join(results_dir, "exp5_stats.txt"), "w") as fh:
        fh.write("\n".join(txt))

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        o = summ.sort_values("AUC")
        ax.barh(o.index, o.AUC, color=["#c44" if i.startswith("AMEND") else "#89a"
                                       for i in o.index])
        ax.axvline(0.5, ls="--", c="k", lw=1)
        ax.set_xlabel("ROC-AUC of the noise ranking (0.5 = no signal)")
        ax.set_title("Metric-fair detection: every method scored on its own ranking")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "exp5_ranking_auc.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  plot skipped ({e})")
    print("[exp5] saved exp5_ranking.csv (+ stats, plot)")
    return df


# ============================================================================
# EXPERIMENT 6 : MINORITY-SIZE CROSSOVER
# ============================================================================
def run_exp6(datasets, seeds, n_splits, methods, min_grid, noise_rate, scheme,
             results_dir, ck=None):
    ck = ck or _NullCheckpoint()
    print("\n" + "=" * 74)
    print("EXP 6 — MINORITY-SIZE CROSSOVER (detection AUC as the rare class shrinks)")
    print("=" * 74)
    rows = []
    for name, X, y in datasets:
        pos = minority_label(y)
        avail = int((y == pos).sum())
        for target in [t for t in min_grid if t <= avail]:
            for seed in seeds:
                key = Checkpoint.key("exp6", name, target, noise_rate, scheme,
                                     seed, "|".join(cleaner_tag(m)
                                                    for m in methods))
                if ck.has(key):
                    e, _ = ck.get(key); rows += e; continue
                rng = np.random.RandomState(seed)
                keep = np.r_[np.where(y != pos)[0],
                             rng.choice(np.where(y == pos)[0], target, replace=False)]
                Xs, ys = X[keep], y[keep]
                got = []
                if min(Counter(ys).values()) >= max(3, n_splits):
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                          random_state=seed)
                    for fold, (tr, _) in enumerate(skf.split(Xs, ys)):
                        Xtr = SimpleImputer(strategy="median").fit_transform(Xs[tr])
                        ytr, flip = inject_label_noise(
                            ys[tr], noise_rate, scheme,
                            random_state=seed * 1000 + fold)
                        if flip.sum() < 2 or flip.all():
                            continue
                        for m in methods:
                            try:
                                sc = noise_ranking(m, Xtr, ytr, seed)
                            except Exception:
                                sc = None
                            if sc is None or not np.isfinite(sc).all():
                                continue
                            f = flip.astype(int)
                            got.append({"Dataset": name, "n_minority": target,
                                        "Method": m,
                                        "AUC": float(roc_auc_score(f, sc)),
                                        "AP": float(average_precision_score(f, sc))})
                ck.put(key, got, [])
                rows += got
        print(f"  done: {name}")
    ck.report()
    df = pd.DataFrame(rows)
    if df.empty:
        print("[exp6] no usable configurations"); return df
    df.to_csv(os.path.join(results_dir, "exp6_crossover.csv"), index=False)

    piv = df.pivot_table(index="n_minority", columns="Method", values="AUC")
    print("\nDetection AUC by minority-class size:")
    print(piv.round(4).to_string())
    _amend_col = next((c for c in piv.columns if c.startswith("AMEND")), None)
    if _amend_col is not None and "CL" in piv.columns:
        gap = (piv[_amend_col] - piv["CL"]).round(4)
        print("\nAMEND - CL:"); print(gap.to_string())
        win = gap[gap > 0]
        print("\n  " + (f"AMEND leads CL at n_minority = {list(win.index)} "
                        "-> that regime is the paper's claim; confirm it holds "
                        "per dataset before writing it up."
                        if len(win) else
                        "CL leads at every minority size tested."))
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in piv.columns:
            ax.plot(piv.index, piv[m], marker="o", label=m,
                    linewidth=(2.5 if m.startswith("AMEND") else 1.4))
        ax.axhline(0.5, ls="--", c="k", lw=1)
        ax.set_xlabel("minority-class size (samples)")
        ax.set_ylabel("detection ROC-AUC")
        ax.set_title("Detection quality as the rare class shrinks")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "exp6_crossover.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  plot skipped ({e})")
    print("[exp6] saved exp6_crossover.csv (+ plot)")
    return df


# ============================================================================
# MAIN
# ============================================================================
def _make_checkpoints():
    """One resume log per experiment, each with its own protocol fingerprint."""
    if not RESUME:
        print("Resume: DISABLED (RESUME=False) — an interrupted run starts over")
        return {e: _NullCheckpoint()
                for e in ("exp1", "exp2", "exp3", "exp4", "exp5", "exp6")}
    cdir = CHECKPOINT_DIR or os.path.join(RESULTS_DIR, "_checkpoints")
    base = {"n_splits": N_SPLITS,
            "use_sample_weights": USE_SAMPLE_WEIGHTS,
            "flag_threshold": FLAG_WEIGHT_THRESHOLD}
    print(f"Resume: ENABLED — log dir {cdir}")
    cfgs = {
        "exp1": {**base, "classifiers": list(EXP1_CLASSIFIERS),
                 "noise": NOISE_RATE_ABLATION, "scheme": ABLATION_SCHEME,
                 "ablation_default": dict(ABLATION_DEFAULT)},
        "exp2": {**base, "classifiers": list(EXP2_CLASSIFIERS)},
        "exp3": {**base, "classifiers": list(EXP3_CLASSIFIERS),
                 "noise": NOISE_RATE_ABLATION, "scheme": ABLATION_SCHEME},
        "exp4": {"seed": SEEDS[0], "d_fixed": EXP4_D_FIXED,
                 "n_fixed": EXP4_N_FIXED},
        "exp5": {"n_splits": N_SPLITS},
        "exp6": {"n_splits": N_SPLITS, "grid": list(EXP6_MIN_GRID)},
    }
    return {name: Checkpoint(name, cfg, cdir, enabled=True, fresh=FRESH_START)
            for name, cfg in cfgs.items()}


def main():
    if _IN_COLAB:
        drive.mount("/content/drive", force_remount=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 74)
    print("AMEND STANDALONE CLEANER — EXPERIMENT SUITE v2 (AMEND ONLY)")
    print(f"Data:    {FOLDER_PATH}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Eval:    {len(SEEDS)} seeds x {N_SPLITS}-fold CV, "
          "per-fold median imputation")
    print(f"Positive class: minority class, resolved per dataset")
    print(f"Exp3 classifiers: {', '.join(EXP3_CLASSIFIERS)}")
    print(f"Soft sample weights -> classifiers: {USE_SAMPLE_WEIGHTS}")
    print(f"Baselines available (imblearn): {'YES' if _IMBLEARN else 'NO'}")
    print("=" * 74)

    binary_datasets = load_binary_datasets(FOLDER_PATH)
    print(f"\nTotal binary datasets: {len(binary_datasets)}")
    if not binary_datasets:
        raise RuntimeError(
            "No binary datasets found — check FOLDER_PATH / LABEL_OVERRIDES.")

    cks = _make_checkpoints()
    try:
        if RUN_EXP1:
            run_exp1(_subset(binary_datasets, ABLATION_DATASETS), SEEDS,
                     N_SPLITS, NOISE_RATE_ABLATION, ABLATION_SCHEME,
                     RESULTS_DIR, EXP1_CLASSIFIERS, ck=cks["exp1"])

        if RUN_EXP2:
            run_exp2(_subset(binary_datasets, ROBUSTNESS_DATASETS), SEEDS,
                     N_SPLITS, available_cleaners(EXP2_CLEANERS),
                     EXP2_RATES, EXP2_SCHEMES,
                     RESULTS_DIR, EXP2_CLASSIFIERS, ck=cks["exp2"])

        if RUN_EXP3:
            run_exp3(_subset(binary_datasets, BASELINE_DATASETS), SEEDS,
                     N_SPLITS, available_cleaners(EXP3_CLEANERS),
                     NOISE_RATE_ABLATION,
                     ABLATION_SCHEME, RESULTS_DIR, EXP3_CLASSIFIERS,
                     ck=cks["exp3"])

        if RUN_EXP4:
            run_exp4(available_cleaners(EXP4_CLEANERS), EXP4_N_GRID, EXP4_D_GRID,
                     EXP4_D_FIXED, EXP4_N_FIXED, SEEDS[0], RESULTS_DIR,
                     ck=cks["exp4"])

        if RUN_EXP5:
            run_exp5(_subset(binary_datasets, EXP5_DATASETS), SEEDS, N_SPLITS,
                     EXP5_METHODS, NOISE_RATE_ABLATION, ABLATION_SCHEME,
                     RESULTS_DIR, ck=cks["exp5"])

        if RUN_EXP6:
            run_exp6(_subset(binary_datasets, EXP6_DATASETS), SEEDS, N_SPLITS,
                     EXP6_METHODS, EXP6_MIN_GRID, NOISE_RATE_ABLATION,
                     ABLATION_SCHEME, RESULTS_DIR, ck=cks["exp6"])
    except KeyboardInterrupt:
        print("\n" + "!" * 74)
        print("INTERRUPTED — completed units are on disk. Re-run this cell to")
        print("resume from the last finished unit; nothing already done repeats.")
        print("!" * 74)
        raise
    finally:
        for c in cks.values():
            c.close()

    if _DATASET_FAILURES:
        print("\n" + "!" * 74)
        print("DATASETS THAT FAILED AND WERE SKIPPED — results below are "
              "INCOMPLETE:")
        for k, v in sorted(_DATASET_FAILURES.items()):
            print(f"  {k}: {v}")
        print("!" * 74)

    if _CLEANER_FAILURES or _CLASSIFIER_FAILURES:
        print("\n" + "!" * 74)
        print("SILENT FAILURES CAUGHT (check before reporting):")
        for k, v in sorted(_CLEANER_FAILURES.items()):
            print(f"  cleaner    {k}  x{v}  (fold fell back to uncleaned data)")
        for k, v in sorted(_CLASSIFIER_FAILURES.items()):
            print(f"  classifier {k}  x{v}  (fold dropped from the mean)")
        print("!" * 74)

    print("\n" + "=" * 74)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Outputs written to: {RESULTS_DIR}")
    print("=" * 74)



# ============================================================================
# AMEND — ADD-ON EXPERIMENT SUITE  (Exp 7-11)
#
# HOW TO RUN
# ----------------------------------------------------------------------------
# Paste this in a NEW Colab cell, BELOW the cell holding your existing suite
# (the one that ends with `main()`). It reuses the classes and helpers already
# defined there: AMENDCleaner, ConfidentLearningCleaner, AUMCleaner,
# build_cleaner, build_classifier, Checkpoint, inject_label_noise,
# load_binary_datasets, _friedman_wilcoxon, cliffs_delta, ... Then call:
#
#     main_extra()
#
# WHY IT IS A SEPARATE FILE
# ----------------------------------------------------------------------------
# Checkpoint archives its log whenever the experiment's config fingerprint
# changes. Editing the cfgs of exp1/exp2/exp3 would therefore throw away every
# completed unit and re-run the whole study. Nothing here modifies those three
# configs; all new work lives under NEW checkpoint names (exp7 ... exp11), and
# all new cleaners are built by wrapping AMENDCleaner rather than editing it.
# Your existing results stay valid and reusable.
#
# WHAT IT ADDS
# ----------------------------------------------------------------------------
#   Exp 7  CONTROLS — is AMEND a cleaner or a resampler?
#            RUS-matched      random majority undersampling, per fold matched
#                             to the number of points AMEND removed in that
#                             same fold (the decisive control)
#            RUS@0.083        fixed-rate random majority undersampling
#            AMEND-RemOnly    AMEND's removal, weights NOT passed
#            AMEND-WeightsOnly AMEND's weights, NOTHING removed
#            AMEND-FullySym   pooled threshold AND removal allowed in both
#                             classes (see note below)
#          vs AMEND and the no-clean control, under uniform and
#          minority->majority noise.
#
#   Exp 8  AGREE-WEIGHT SWEEP — w from 0 to 1 inside AMEND's own pipeline,
#          projection and k unchanged. w=0 is pure subspace coherence, w=1 is
#          pure neighbour-label disagreement. Reports end-task metrics AND the
#          ranking ROC-AUC/AP of NS at every w, so the subspace term is tested
#          against disagreement in a MATCHED feature space and a matched k.
#          This is the experiment that answers the paper's central question.
#
#   Exp 9  CL WITH A MATCHED BASE MODEL — confident learning run with each
#          downstream classifier as its own probability model, instead of
#          logistic regression for all five.
#
#   Exp 10 INSTANCE-DEPENDENT NOISE — flips the lowest-margin samples rather
#          than random ones, which is the realistic noise model and the one
#          where local geometry should behave differently.
#
#   Exp 11 INTERNAL-STATE AUDIT — logs realized k, r, r-cap hits, projection
#          width, tau, the weight distribution and the effective class balance
#          (sum of weights per class) before and after cleaning. Classifier-free
#          and fast. This is the reporting your saved runs cannot provide,
#          because the current logs store only end-task and detection metrics.
#
#   Plus:  export_exp3_pairwise() writes Cliff's delta and Holm-adjusted p per
#          comparison to CSV from your EXISTING exp3 files. No re-run needed.
#
# A FINDING FROM READING YOUR CODE, BEFORE YOU RUN ANYTHING
# ----------------------------------------------------------------------------
# In SoftCleaner.clean, the class_symmetric=True branch still does
#     remove = hard & (~is_min);  quarantine = hard & is_min
# so the "symmetric" ablation row NEVER removes minority samples either. That
# row therefore tests THRESHOLD asymmetry only (pooled quantile vs per-class
# quantiles with tau/2 for the minority). The removal asymmetry — the thing the
# paper calls AMEND's core design and credits in Section VI-B — is not ablated
# anywhere in the current suite. AMEND-FullySym in Exp 7 is the missing row.
# Whatever you report, Table IX's caption has to say which of the two the
# "class-symmetric thresholds" row changes.
# ============================================================================

import os
import re
import json
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_drive():
    """Mount Drive if the data folder is not reachable (main() normally does it)."""
    try:
        if os.path.exists(FOLDER_PATH):
            return
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass





# ============================================================================
# CONFIG FOR THE ADD-ON
# ============================================================================
# Conditions used by Exp 7 / 9 / 11: (scheme, rate). The first mirrors the
# paper's headline setting (Table III); the second is the regime AMEND is
# designed for, at the middle of the sweep.
EXTRA_CONDITIONS = [("uniform", 0.15), ("minority_to_maj", 0.20)]

# Exp 7
EXP7_CLEANERS = ["None", "AMEND", "AMEND-RemOnly", "AMEND-WeightsOnly",
                 "AMEND-FullySym", "RUS-matched", "RUS@0.083"]
RUS_FIXED_RATE = 0.083          # AMEND's measured mean removal rate (Table V)

# Exp 8 — the agree-weight sweep. w=0.3 is the paper's default.
EXP8_W_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
EXP8_CONDITION = ("uniform", 0.15)

# Exp 9 — CL with a matched base model
EXP9_BASE_CLASSIFIERS = ["RF", "SVM", "kNN", "LR", "DT"]

# Exp 10 — instance-dependent noise
EXP10_SCHEMES = ["instance_dep", "instance_dep_min"]
EXP10_RATES = [0.05, 0.10, 0.20, 0.30]
EXP10_CLEANERS = ["None", "AMEND", "CL"]

# Exp 11 — internal-state audit (no classifiers, cheap)
EXP11_CONDITIONS = [("uniform", 0.0), ("uniform", 0.15),
                    ("minority_to_maj", 0.20), ("maj_to_minority", 0.20)]

RUN_EXP7, RUN_EXP8, RUN_EXP9, RUN_EXP10, RUN_EXP11 = True, True, True, True, True
RUN_POSTHOC_CLIFFS = True

EXTRA_DATASETS = None            # None = all 13


# ============================================================================
# NEW CLEANERS  (all wrap AMENDCleaner; the core classes are untouched)
# ============================================================================
def _amend_kwargs(**overrides):
    kw = dict(AMEND_DEFAULTS)
    kw.update(overrides)
    return kw


class AMENDWeightsOnlyCleaner(BaseEstimator):
    """AMEND's soft weights with NOTHING removed.

    Samples AMEND would have deleted keep the quarantine weight instead, which
    is the same treatment AMEND already gives suspect minority samples. Pair
    with AMEND-RemOnly to split the end-task effect into its two halves.
    """

    def __init__(self, quarantine_weight=0.1, random_state=42, amend_kw=None):
        self.quarantine_weight = quarantine_weight
        self.random_state = random_state
        self.amend_kw = amend_kw

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y); n = len(y)
        inner = AMENDCleaner(**_amend_kwargs(random_state=self.random_state,
                                             **(self.amend_kw or {})))
        inner.fit_resample(X, y)
        self.inner_ = inner
        self.skipped_ = bool(getattr(inner, "skipped_", False))

        w = np.ones(n)
        km = np.asarray(getattr(inner, "keep_mask_", np.ones(n, bool)), dtype=bool)
        sw = np.asarray(getattr(inner, "sample_weight_", np.ones(int(km.sum()))))
        if len(sw) == int(km.sum()):
            w[km] = sw
        w[~km] = self.quarantine_weight

        self.keep_mask_ = np.ones(n, dtype=bool)
        self.sample_weight_ = w
        self.noise_scores_ = getattr(inner, "noise_scores_", np.zeros(n))
        self.tau_ = float(getattr(inner, "tau_", 0.0))
        self.n_removed_ = 0
        return X, y

    def fit(self, X, y):
        self.fit_resample(X, y); return self


class AMENDFullySymmetricCleaner(BaseEstimator):
    """Pooled threshold AND removal permitted in BOTH classes.

    This is the row missing from Table IX. SoftCleaner's class_symmetric branch
    symmetrises the THRESHOLD but still refuses to delete minority samples, so
    it does not test the removal asymmetry. Here a single pooled quantile is
    applied and anything above it is deleted regardless of its label, with a
    floor of two retained samples per class so a fold is never destroyed.
    """

    def __init__(self, min_per_class=2, random_state=42, amend_kw=None):
        self.min_per_class = min_per_class
        self.random_state = random_state
        self.amend_kw = amend_kw

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y); n = len(y)
        inner = AMENDCleaner(**_amend_kwargs(random_state=self.random_state,
                                             **(self.amend_kw or {})))
        inner.fit_resample(X, y)
        self.inner_ = inner
        ns = np.asarray(getattr(inner, "noise_scores_", np.zeros(n)), dtype=float)
        tau = float(getattr(inner, "tau_", 0.0))

        if getattr(inner, "skipped_", False) or tau <= 0.0 or len(ns) != n:
            self.keep_mask_ = np.ones(n, dtype=bool)
            self.sample_weight_ = np.ones(n)
            self.noise_scores_ = ns; self.tau_ = tau
            self.n_removed_ = 0; self.skipped_ = True
            return X, y
        self.skipped_ = False

        th_hard = float(np.quantile(ns, 1 - tau / 2))
        th_soft = float(np.quantile(ns, 1 - 2 * tau))
        if th_soft >= th_hard:
            th_soft = th_hard * 0.8

        hard = ns > th_hard
        soft = (ns > th_soft) & (~hard)
        w = np.ones(n)
        if soft.any():
            w[soft] = np.clip(1 - (ns[soft] - th_soft) / max(th_hard - th_soft, 1e-12),
                              0.1, 1.0)

        remove = hard.copy()
        for c in np.unique(y):                      # never wipe out a class
            idx = np.where(y == c)[0]
            short = self.min_per_class - int((~remove[idx]).sum())
            if short > 0:
                back = idx[remove[idx]]
                remove[back[np.argsort(ns[back])][:short]] = False

        keep = ~remove
        self.keep_mask_ = keep
        self.sample_weight_ = w[keep]
        self.noise_scores_ = ns
        self.tau_ = tau
        self.n_removed_ = int(remove.sum())
        return X[keep], y[keep]

    def fit(self, X, y):
        self.fit_resample(X, y); return self


class RandomUndersampleCleaner(BaseEstimator):
    """Random removal of majority-labelled samples — the rebalancing control.

    rate=None  -> per fold, remove exactly as many majority points as AMEND
                  removed in that same fold (matched control; costs one extra
                  AMEND call, ~0.2 s).
    rate=float -> remove round(rate * n) majority points.

    Removal is restricted to the majority-labelled class, exactly as AMEND's
    is, so the only difference from AMEND is WHICH points go.
    """

    def __init__(self, rate=None, min_per_class=2, random_state=42,
                 amend_kw=None):
        self.rate = rate
        self.min_per_class = min_per_class
        self.random_state = random_state
        self.amend_kw = amend_kw

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y); n = len(y)
        cc = Counter(y)
        min_cls = min(cc, key=cc.get)
        maj_idx = np.where(y != min_cls)[0]

        if self.rate is None:
            inner = AMENDCleaner(**_amend_kwargs(random_state=self.random_state,
                                                 **(self.amend_kw or {})))
            inner.fit_resample(X, y)
            self.matched_to_ = int(getattr(inner, "n_removed_", 0))
            self.tau_ = float(getattr(inner, "tau_", 0.0))
            n_rm = self.matched_to_
        else:
            self.matched_to_ = None
            self.tau_ = float("nan")
            n_rm = int(round(float(self.rate) * n))

        n_rm = int(max(0, min(n_rm, len(maj_idx) - self.min_per_class)))
        rng = np.random.RandomState(self.random_state)
        drop = rng.choice(maj_idx, size=n_rm, replace=False) if n_rm > 0 \
            else np.array([], dtype=int)

        keep = np.ones(n, dtype=bool); keep[drop] = False
        self.keep_mask_ = keep
        self.sample_weight_ = np.ones(int(keep.sum()))
        self.n_removed_ = int(n_rm)
        self.skipped_ = False
        return X[keep], y[keep]

    def fit(self, X, y):
        self.fit_resample(X, y); return self


def build_cl_base(name, seed):
    """Downstream classifier reused as CL's probability model.

    SVC needs probability=True for predict_proba; the other four already have
    it. Everything else matches build_classifier exactly, so CL sees the same
    model the end task will use.
    """
    if name == "SVM":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", SVC(C=1.0, kernel="rbf", gamma="scale",
                                     class_weight="balanced", probability=True,
                                     random_state=seed))])
    return build_classifier(name, seed)


# ============================================================================
# CLEANER REGISTRY FOR THE ADD-ON
# ============================================================================
def build_cleaner_ext(cname, seed):
    """Extends build_cleaner with the new controls. Falls through for old names.

    `_use_weights` decides whether the fold runner passes sample weights to the
    classifier, matching the convention in AMEND_LADDER.
    """
    if cname == "AMEND-RemOnly":                      # removal, weights OFF
        cl = AMENDCleaner(**_amend_kwargs(random_state=seed))
        cl._use_weights = False
        return cl
    if cname == "AMEND-WeightsOnly":                  # weights, removal OFF
        cl = AMENDWeightsOnlyCleaner(random_state=seed)
        cl._use_weights = True
        return cl
    if cname == "AMEND-FullySym":                     # symmetric threshold+removal
        cl = AMENDFullySymmetricCleaner(random_state=seed)
        cl._use_weights = True
        return cl
    if cname == "RUS-matched":
        cl = RandomUndersampleCleaner(rate=None, random_state=seed)
        cl._use_weights = False
        return cl
    if cname.startswith("RUS@"):
        cl = RandomUndersampleCleaner(rate=float(cname.split("@")[1]),
                                      random_state=seed)
        cl._use_weights = False
        return cl
    if cname.startswith("AMEND-w"):                   # agree-weight sweep
        w = float(cname.split("AMEND-w")[1])
        cl = AMENDCleaner(**_amend_kwargs(agree_weight=w, random_state=seed))
        cl._use_weights = True
        return cl
    if cname.startswith("CL-"):                       # matched base model
        base = cname.split("CL-")[1]
        cl = ConfidentLearningCleaner(base_estimator=build_cl_base(base, seed),
                                      random_state=seed)
        cl._use_weights = False
        return cl
    return build_cleaner(cname, seed)


def cleaner_tag_ext(cname):
    """Checkpoint identity. New names are already self-describing."""
    if (cname in ("AMEND-RemOnly", "AMEND-WeightsOnly", "AMEND-FullySym",
                  "RUS-matched") or cname.startswith(("RUS@", "AMEND-w", "CL-"))):
        return cname
    return cleaner_tag(cname)


def classifiers_for(cname, classifiers):
    """CL-<CLF> is only meaningful with its own classifier."""
    if cname.startswith("CL-"):
        base = cname.split("CL-")[1]
        return [base] if base in classifiers else []
    return list(classifiers)


# ============================================================================
# INSTANCE-DEPENDENT NOISE
# ============================================================================
_MARGIN_CACHE = {}
_MARGIN_CACHE_MAX = 4096


def _fold_fingerprint(X, y, seed):
    """Cheap exact-enough identity for one (fold, seed) so the margin model is
    fitted ONCE, not once per cleaner. Without this, Exp 10 would refit the
    logistic regression three times per fold on 22 283 features for nothing."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    Xs = X if X.nbytes <= 50_000_000 else X[::3, ::7]
    h.update(np.ascontiguousarray(Xs).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    h.update(str((X.shape, int(seed))).encode())
    return h.hexdigest()


def _own_class_probability(X, y, seed):
    """Out-of-fold probability of each sample's own (clean) label.

    Used only to DEFINE the noise, never by the cleaner or the classifier, and
    computed on the training fold alone, so no test information leaks.
    """
    key = _fold_fingerprint(X, y, seed)
    if key in _MARGIN_CACHE:
        return _MARGIN_CACHE[key]
    p = _own_class_probability_uncached(X, y, seed)
    if len(_MARGIN_CACHE) >= _MARGIN_CACHE_MAX:
        _MARGIN_CACHE.clear()
    _MARGIN_CACHE[key] = p
    return p


def _own_class_probability_uncached(X, y, seed):
    classes = np.unique(y)
    yi = np.searchsorted(classes, y)
    n_min = int(min(Counter(yi).values()))
    est = Pipeline([("sc", StandardScaler()),
                    ("c", LogisticRegression(max_iter=1000,
                                             class_weight="balanced",
                                             random_state=seed))])
    if n_min >= 2:
        k = int(max(2, min(3, n_min)))
        try:
            P = cross_val_predict(est, X, yi, method="predict_proba",
                                  cv=StratifiedKFold(n_splits=k, shuffle=True,
                                                     random_state=seed))
            return P[np.arange(len(yi)), yi]
        except Exception:
            pass
    try:
        est.fit(X, yi)
        return est.predict_proba(X)[np.arange(len(yi)), yi]
    except Exception:
        return np.full(len(yi), 0.5)


def inject_label_noise_ext(X, y, rate, scheme="uniform", random_state=0):
    """inject_label_noise plus two instance-dependent schemes.

    instance_dep      flip the round(rate*n) samples the model is least
                      confident about, in either class — the hardest-first
                      analogue of uniform noise.
    instance_dep_min  the same, restricted to minority samples, so it is
                      directly comparable with minority_to_maj.

    Boundary-adjacent flips are the realistic case and the one where local
    subspaces of the two classes overlap most, which is why SCS should behave
    differently here than under random flips.
    """
    y = np.asarray(y).copy()
    if scheme not in ("instance_dep", "instance_dep_min"):
        return inject_label_noise(y, rate, scheme, random_state=random_state)

    n = len(y)
    cc = Counter(y)
    min_cls = min(cc, key=cc.get); maj_cls = max(cc, key=cc.get)
    flip = np.zeros(n, dtype=bool)
    if rate <= 0:
        return y.copy(), flip

    p_own = _own_class_probability(X, y, random_state)
    rng = np.random.RandomState(random_state)
    jitter = rng.rand(n) * 1e-9                      # deterministic tie-break

    if scheme == "instance_dep":
        cand = np.arange(n)
        budget = int(round(rate * n))
    else:
        cand = np.where(y == min_cls)[0]
        budget = int(round(rate * len(cand)))
    budget = int(max(0, min(budget, len(cand) - 1)))
    if budget == 0:
        return y.copy(), flip

    order = cand[np.argsort(p_own[cand] + jitter[cand])]   # least confident first
    flip[order[:budget]] = True

    y_noisy = y.copy()
    y_noisy[flip] = np.where(y_noisy[flip] == min_cls, maj_cls, min_cls)
    return y_noisy, flip


# ============================================================================
# INTERNAL-STATE LOGGING
# ============================================================================
STATE_KEYS = ["k_used", "r_used", "r_cap", "r_at_cap", "proj_dim", "tau",
              "n_removed", "removal_rate", "w_mean", "w_min", "frac_w_eq1",
              "frac_w_le05", "sum_w_min", "sum_w_maj", "eff_IR_before",
              "eff_IR_after", "n_min_obs", "n_maj_obs", "skipped",
              "RankAUC", "RankAP"]


def cleaner_state(cleaner, y_noisy, flipped=None):
    """Everything the current logs do not record.

    Reports the realized structural parameters, how often r saturates at its
    k-1 cap (if it usually does, the local subspace spans the whole
    neighbourhood and SCS is measuring neighbourhood overlap), the weight
    distribution, and the effective class balance sum(omega) per class before
    and after cleaning — the number that shows how much of AMEND is a
    resampler.
    """
    out = {k: np.nan for k in STATE_KEYS}
    if cleaner is None:
        return out
    n = len(y_noisy)
    cc = Counter(y_noisy)
    min_cls = min(cc, key=cc.get)
    is_min = (np.asarray(y_noisy) == min_cls)
    n_min, n_maj = int(is_min.sum()), int((~is_min).sum())

    inner = getattr(cleaner, "inner_", cleaner)
    k_used = getattr(inner, "k_used_", np.nan)
    r_used = getattr(inner, "r_used_", np.nan)
    out["k_used"] = float(k_used) if k_used == k_used else np.nan
    out["r_used"] = float(r_used) if r_used == r_used else np.nan
    if out["k_used"] == out["k_used"]:
        out["r_cap"] = out["k_used"] - 1
        out["r_at_cap"] = float(out["r_used"] >= out["r_cap"] - 1e-9)
    out["proj_dim"] = float(getattr(inner, "proj_dim_used_", np.nan))
    out["tau"] = float(getattr(inner, "tau_", np.nan))
    out["skipped"] = float(bool(getattr(inner, "skipped_", False)))
    out["n_min_obs"], out["n_maj_obs"] = float(n_min), float(n_maj)

    km = getattr(cleaner, "keep_mask_", None)
    sw = getattr(cleaner, "sample_weight_", None)
    if km is None:
        return out
    km = np.asarray(km, dtype=bool)
    w_full = np.zeros(n)
    if sw is not None and len(sw) == int(km.sum()):
        w_full[km] = np.asarray(sw, dtype=float)
    else:
        w_full[km] = 1.0

    out["n_removed"] = float((~km).sum())
    out["removal_rate"] = float((~km).sum() / max(n, 1))
    kept_w = w_full[km]
    if len(kept_w):
        out["w_mean"] = float(kept_w.mean())
        out["w_min"] = float(kept_w.min())
        out["frac_w_eq1"] = float(np.mean(kept_w >= 1.0 - 1e-9))
        out["frac_w_le05"] = float(np.mean(kept_w <= 0.5))
    out["sum_w_min"] = float(w_full[is_min].sum())
    out["sum_w_maj"] = float(w_full[~is_min].sum())
    out["eff_IR_before"] = float(n_maj / max(n_min, 1))
    out["eff_IR_after"] = float(out["sum_w_maj"] / max(out["sum_w_min"], 1e-9))

    ns = getattr(cleaner, "noise_scores_", None)
    if ns is not None and flipped is not None and len(ns) == n:
        f = np.asarray(flipped, dtype=int)
        if 0 < f.sum() < n and np.isfinite(ns).all():
            try:
                out["RankAUC"] = float(roc_auc_score(f, ns))
                out["RankAP"] = float(average_precision_score(f, ns))
            except Exception:
                pass
    return out


# ============================================================================
# FOLD RUNNER  (own copy: X-aware noise + state logging, old one untouched)
# ============================================================================
def fold_run_ext(X, y, cleaner_factory, seed, n_splits, fold, noise_rate,
                 scheme, classifiers, pos_label, cname, want_state=True,
                 want_classifiers=True):
    if pos_label is None:
        pos_label = minority_label(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    if fold >= len(splits):
        return [], []
    tr, te = splits[fold]

    imp = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
    ytr, yte = y[tr].copy(), y[te].copy()

    flipped = np.zeros(len(ytr), dtype=bool)
    if noise_rate > 0:
        ytr, flipped = inject_label_noise_ext(
            Xtr, ytr, noise_rate, scheme, random_state=seed * 1000 + fold)

    cleaner = cleaner_factory(seed)
    Xc, yc, keep, w, dt = apply_cleaner(cleaner, Xtr, ytr, cname)

    det_rows = []
    d = detection_scores(flipped, keep, w) if noise_rate > 0 \
        else {k: np.nan for k in DET_KEYS}
    d["clean_seconds"] = dt
    d["n_after"] = len(yc)
    d["kept_frac"] = len(yc) / max(len(ytr), 1)
    if want_state:
        d.update(cleaner_state(cleaner, ytr, flipped if noise_rate > 0 else None))
    d["SkipRate"] = float(bool(getattr(cleaner, "skipped_", False)))
    det_rows.append(d)

    end_rows = []
    if not want_classifiers or len(np.unique(yc)) < 2:
        return end_rows, det_rows

    sw = None
    if (USE_SAMPLE_WEIGHTS and getattr(cleaner, "_use_weights", True)
            and w is not None and len(w) == len(yc)
            and not np.allclose(w, 1.0)):
        sw = w

    for cl_name in classifiers:
        try:
            clf = build_classifier(cl_name, seed)
            if sw is not None and cl_name in _SUPPORTS_WEIGHT:
                clf.fit(Xc, yc, clf__sample_weight=sw)
            else:
                clf.fit(Xc, yc)
            row = endtask_scores(yte, clf.predict(Xte),
                                 _pos_scores(clf, Xte, pos_label), pos_label)
        except Exception as e:
            _CLASSIFIER_FAILURES[f"{cl_name}: {type(e).__name__}"] += 1
            continue
        row["Classifier"] = cl_name
        row["clean_seconds"] = dt
        row["n_after"] = len(yc)
        end_rows.append(row)
    return end_rows, det_rows


def multi_seed_ext(X, y, fac, seeds, n_splits, rate, scheme, classifiers,
                   pos_label, cname, ck, key_parts, want_classifiers=True):
    """One unit per (seed, fold), as in the original harness."""
    e_all, d_all = [], []
    for s in seeds:
        for fold in range(n_splits):
            k = Checkpoint.key(*key_parts, f"seed{s}", f"fold{fold}")
            if ck.has(k):
                e, d = ck.get(k)
            else:
                e, d = fold_run_ext(X, y, fac, s, n_splits, fold, rate, scheme,
                                    classifiers, pos_label, cname,
                                    want_classifiers=want_classifiers)
                ck.put(k, e, d)
            e_all += e; d_all += d
    return e_all, d_all


def _rows_from(dataset, meta, end_rows, det_rows, classifiers):
    det = _r4(agg(det_rows)) if det_rows else {}
    per_clf = agg(end_rows, by="Classifier")
    out = []
    for cl in classifiers:
        row = {"Dataset": dataset}; row.update(meta); row["Classifier"] = cl
        if len(per_clf) and (per_clf["Classifier"] == cl).any():
            e = per_clf[per_clf["Classifier"] == cl].iloc[0].to_dict()
            e.pop("Classifier", None)
            row.update(_r4(e))
        row.update(det)
        out.append(row)
    return out


def _pairwise_stats_df(piv, ref, lower_is_better=False, **tags):
    """Holm-adjusted Wilcoxon + Cliff's delta as a DataFrame, not just text."""
    from scipy.stats import wilcoxon
    piv = piv.dropna()
    if ref not in piv.columns or piv.shape[0] < 3:
        return pd.DataFrame()
    prs = []
    for a in [c for c in piv.columns if c != ref]:
        d = piv[ref].values - piv[a].values
        eff = cliffs_delta(piv[ref].values, piv[a].values)
        pv = np.nan
        if int((d != 0).sum()) >= 5:
            _, pv = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        wins = int((d > 0).sum()) if not lower_is_better else int((d < 0).sum())
        prs.append([a, float(np.mean(d)), float(pv), float(eff), wins,
                    int(len(d) - wins - int((d == 0).sum()))])
    testable = sorted([p for p in prs if p[2] == p[2]], key=lambda t: t[2])
    m = len(testable); run = 0.0; holm = {}
    for i, p in enumerate(testable):
        adj = max(min(p[2] * (m - i), 1.0), run); run = adj; holm[p[0]] = adj
    rows = []
    for a, dm, pv, eff, wins, losses in prs:
        r = dict(tags)
        r.update({"Comparator": a, "dmean": round(dm, 4),
                  "p_raw": round(pv, 5) if pv == pv else np.nan,
                  "p_holm": round(holm.get(a, np.nan), 5),
                  "cliffs_delta": round(eff, 3),
                  "Wins": wins, "Losses": losses})
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================
# EXPERIMENT 7 — CONTROLS: CLEANER OR RESAMPLER?
# ============================================================================
def run_exp7(datasets, seeds, n_splits, results_dir, classifiers, ck):
    print("\n" + "=" * 74)
    print("EXP 7 — CONTROLS (RUS-matched / removal-only / weights-only /")
    print("        fully symmetric) vs AMEND and the no-clean control")
    print("=" * 74)
    rows, stats = [], []
    for name, X, y in datasets:
        try:
            pos = minority_label(y)
            for scheme, rate in EXTRA_CONDITIONS:
                for cname in EXP7_CLEANERS:
                    fac = (lambda cn: (lambda s: build_cleaner_ext(cn, s)))(cname)
                    e, d = multi_seed_ext(
                        X, y, fac, seeds, n_splits, rate, scheme, classifiers,
                        pos, cname, ck,
                        (name, scheme, rate, cleaner_tag_ext(cname)))
                    rows += _rows_from(name, {"Scheme": scheme, "NoiseRate": rate,
                                              "Cleaner": cname},
                                       e, d, classifiers)
            print(f"  done: {name}")
        except Exception as ex:
            _DATASET_FAILURES[f"exp7/{name}"] = f"{type(ex).__name__}: {ex}"
            print(f"  !! exp7: '{name}' FAILED ({type(ex).__name__}: {ex})")
        finally:
            __import__("gc").collect()
    ck.report()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp7_controls.csv"), index=False)

    txt = ["EXP 7 — CONTROLS", f"seeds={list(seeds)} folds={n_splits}", ""]
    for scheme, rate in EXTRA_CONDITIONS:
        sub = df[(df.Scheme == scheme) & (df.NoiseRate == rate)]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="Cleaner", columns="Classifier", values="F1")
        piv["MEAN"] = piv.mean(axis=1)
        piv = piv.round(4).sort_values("MEAN", ascending=False)
        head = f"--- {scheme} @ rho={rate} : end-task F1 ---"
        print("\n" + head); print(piv.to_string())
        txt += [head, piv.to_string(), ""]
        for cl in classifiers:
            for metric in ["F1", "GMean", "Recall", "FPR", "AUC"]:
                p = sub[sub.Classifier == cl].pivot_table(
                    index="Dataset", columns="Cleaner", values=metric)
                txt.append(_friedman_wilcoxon(
                    p, f"{scheme}@{rate} / {cl} / {metric}", ref="AMEND",
                    lower_is_better=(metric == "FPR")))
                s = _pairwise_stats_df(p, "AMEND",
                                       lower_is_better=(metric == "FPR"),
                                       Experiment="exp7", Scheme=scheme,
                                       NoiseRate=rate, Classifier=cl,
                                       Metric=metric)
                if len(s):
                    stats.append(s)
    with open(os.path.join(results_dir, "exp7_stats.txt"), "w") as fh:
        fh.write("\n".join(txt))
    if stats:
        pd.concat(stats, ignore_index=True).to_csv(
            os.path.join(results_dir, "exp7_pairwise_stats.csv"), index=False)

    # The headline read: AMEND minus each control, per classifier.
    try:
        sub = df[(df.Scheme == EXTRA_CONDITIONS[0][0]) &
                 (df.NoiseRate == EXTRA_CONDITIONS[0][1])]
        piv = sub.pivot_table(index="Cleaner", columns="Classifier", values="F1")
        base = piv.loc["AMEND"]
        delta = (piv - base).drop(index="AMEND").round(4)
        print("\nAMEND minus each control (F1; positive = AMEND ahead):")
        print(delta.to_string())
        print("\n  If RUS-matched is close to zero, the end-task gain is "
              "rebalancing, not noise removal.")
        print("  AMEND-RemOnly vs AMEND-WeightsOnly splits the effect into "
              "deletion and reweighting.")
    except Exception:
        pass
    print("[exp7] saved exp7_controls.csv, exp7_stats.txt, "
          "exp7_pairwise_stats.csv")
    return df


# ============================================================================
# EXPERIMENT 8 — AGREE-WEIGHT SWEEP  (the central question)
# ============================================================================
def run_exp8(datasets, seeds, n_splits, results_dir, classifiers, ck):
    scheme, rate = EXP8_CONDITION
    print("\n" + "=" * 74)
    print("EXP 8 — AGREE-WEIGHT SWEEP w=0..1 (subspace vs label disagreement,")
    print("        matched projection and matched k)")
    print("=" * 74)
    rows = []
    for name, X, y in datasets:
        try:
            pos = minority_label(y)
            for w in EXP8_W_GRID:
                cname = f"AMEND-w{w:g}"
                fac = (lambda cn: (lambda s: build_cleaner_ext(cn, s)))(cname)
                e, d = multi_seed_ext(
                    X, y, fac, seeds, n_splits, rate, scheme, classifiers,
                    pos, cname, ck, (name, scheme, rate, cname))
                rows += _rows_from(name, {"w": w, "Cleaner": cname}, e, d,
                                   classifiers)
            print(f"  done: {name}")
        except Exception as ex:
            _DATASET_FAILURES[f"exp8/{name}"] = f"{type(ex).__name__}: {ex}"
            print(f"  !! exp8: '{name}' FAILED ({type(ex).__name__}: {ex})")
        finally:
            __import__("gc").collect()
    ck.report()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp8_w_sweep.csv"), index=False)

    cols = [c for c in ["F1", "GMean", "AUC", "RankAUC", "RankAP",
                        "NoisePrec", "NoiseRec", "FRR", "tau", "removal_rate"]
            if c in df.columns]
    summ = df.groupby("w")[cols].mean().round(4)
    print("\nMean over datasets and classifiers, by agree-weight w:")
    print(summ.to_string())
    print("\n  w=0 is pure subspace coherence, w=1 pure neighbour-label")
    print("  disagreement, w=0.3 the paper's default. RankAUC is the")
    print("  threshold-free ranking quality of NS against the injected flips.")
    print("  If RankAUC rises monotonically with w, the subspace term dilutes")
    print("  the ranking and the comparison is now matched in space and in k,")
    print("  so it supports the claim the kNN-disagree comparison could not.")

    txt = ["EXP 8 — AGREE-WEIGHT SWEEP", f"{scheme} @ rho={rate}",
           f"seeds={list(seeds)} folds={n_splits}", "", summ.to_string(), ""]
    stats = []
    per_ds = df.groupby(["Dataset", "w"])[cols].mean().reset_index()
    for metric in [m for m in ["RankAUC", "RankAP", "F1"] if m in cols]:
        piv = per_ds.pivot_table(index="Dataset", columns="w", values=metric)
        piv.columns = [f"w={c:g}" for c in piv.columns]
        ref = "w=0.3"
        txt.append(_friedman_wilcoxon(piv, f"{metric} across w", ref=ref))
        s = _pairwise_stats_df(piv, ref, Experiment="exp8", Metric=metric)
        if len(s):
            stats.append(s)
    with open(os.path.join(results_dir, "exp8_stats.txt"), "w") as fh:
        fh.write("\n".join(txt))
    if stats:
        pd.concat(stats, ignore_index=True).to_csv(
            os.path.join(results_dir, "exp8_pairwise_stats.csv"), index=False)

    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        if "RankAUC" in summ:
            ax[0].plot(summ.index, summ.RankAUC, marker="o", lw=2.4)
            ax[0].axhline(0.5, ls="--", c="k", lw=1)
            ax[0].set_ylabel("ranking ROC-AUC of NS")
        ax[1].plot(summ.index, summ.F1, marker="o", lw=2.4, color="C1")
        ax[1].set_ylabel("end-task F1")
        for a in ax:
            a.set_xlabel("agree-weight w  (0 = subspace only, 1 = disagreement only)")
            a.grid(alpha=0.3)
        fig.suptitle("Does the subspace term add signal? (matched projection and k)")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "exp8_w_sweep.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  plot skipped ({e})")
    print("[exp8] saved exp8_w_sweep.csv, exp8_stats.txt (+ plot)")
    return df


# ============================================================================
# EXPERIMENT 9 — CL WITH A MATCHED BASE MODEL
# ============================================================================
def run_exp9(datasets, seeds, n_splits, results_dir, classifiers, ck):
    print("\n" + "=" * 74)
    print("EXP 9 — CONFIDENT LEARNING WITH EACH CLASSIFIER AS ITS OWN BASE MODEL")
    print("=" * 74)
    cleaners = ["None", "AMEND", "CL"] + \
               [f"CL-{c}" for c in EXP9_BASE_CLASSIFIERS if c in classifiers]
    rows = []
    for name, X, y in datasets:
        try:
            pos = minority_label(y)
            for scheme, rate in EXTRA_CONDITIONS:
                for cname in cleaners:
                    clfs = classifiers_for(cname, classifiers)
                    if not clfs:
                        continue
                    fac = (lambda cn: (lambda s: build_cleaner_ext(cn, s)))(cname)
                    e, d = multi_seed_ext(
                        X, y, fac, seeds, n_splits, rate, scheme, clfs, pos,
                        cname, ck, (name, scheme, rate, cleaner_tag_ext(cname)))
                    rows += _rows_from(name, {"Scheme": scheme, "NoiseRate": rate,
                                              "Cleaner": cname}, e, d, clfs)
            print(f"  done: {name}")
        except Exception as ex:
            _DATASET_FAILURES[f"exp9/{name}"] = f"{type(ex).__name__}: {ex}"
            print(f"  !! exp9: '{name}' FAILED ({type(ex).__name__}: {ex})")
        finally:
            __import__("gc").collect()
    ck.report()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp9_cl_matched.csv"), index=False)

    # Collapse CL-<CLF> into one "CL_matched" column, one value per classifier.
    df["CleanerX"] = np.where(df.Cleaner.str.startswith("CL-"), "CL_matched",
                              df.Cleaner)
    txt = ["EXP 9 — CL WITH A MATCHED BASE MODEL", ""]
    stats = []
    for scheme, rate in EXTRA_CONDITIONS:
        sub = df[(df.Scheme == scheme) & (df.NoiseRate == rate)]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="CleanerX", columns="Classifier", values="F1")
        piv["MEAN"] = piv.mean(axis=1)
        piv = piv.round(4).sort_values("MEAN", ascending=False)
        head = f"--- {scheme} @ rho={rate} : end-task F1 ---"
        print("\n" + head); print(piv.to_string())
        txt += [head, piv.to_string(), ""]
        for cl in classifiers:
            for metric in ["F1", "GMean", "AUC"]:
                p = sub[sub.Classifier == cl].pivot_table(
                    index="Dataset", columns="CleanerX", values=metric)
                txt.append(_friedman_wilcoxon(
                    p, f"{scheme}@{rate} / {cl} / {metric}", ref="AMEND"))
                s = _pairwise_stats_df(p, "AMEND", Experiment="exp9",
                                       Scheme=scheme, NoiseRate=rate,
                                       Classifier=cl, Metric=metric)
                if len(s):
                    stats.append(s)
    with open(os.path.join(results_dir, "exp9_stats.txt"), "w") as fh:
        fh.write("\n".join(txt))
    if stats:
        pd.concat(stats, ignore_index=True).to_csv(
            os.path.join(results_dir, "exp9_pairwise_stats.csv"), index=False)
    print("\n  'CL' is the paper's logistic-regression version; 'CL_matched'")
    print("  gives CL the same model the end task uses. If CL_matched widens")
    print("  the gap, the paper's CL comparison was conservative in CL's "
          "disfavour.")
    print("[exp9] saved exp9_cl_matched.csv, exp9_stats.txt")
    return df


# ============================================================================
# EXPERIMENT 10 — INSTANCE-DEPENDENT NOISE
# ============================================================================
def run_exp10(datasets, seeds, n_splits, results_dir, classifiers, ck):
    print("\n" + "=" * 74)
    print("EXP 10 — INSTANCE-DEPENDENT NOISE (lowest-margin samples flipped)")
    print("=" * 74)
    rows = []
    for name, X, y in datasets:
        try:
            pos = minority_label(y)
            for scheme in EXP10_SCHEMES:
                for rate in EXP10_RATES:
                    for cname in EXP10_CLEANERS:
                        fac = (lambda cn: (lambda s: build_cleaner_ext(cn, s)))(cname)
                        e, d = multi_seed_ext(
                            X, y, fac, seeds, n_splits, rate, scheme,
                            classifiers, pos, cname, ck,
                            (name, scheme, rate, cleaner_tag_ext(cname)))
                        rows += _rows_from(name, {"Scheme": scheme,
                                                  "NoiseRate": rate,
                                                  "Cleaner": cname},
                                           e, d, classifiers)
                print(f"  {name} / {scheme} done")
        except Exception as ex:
            _DATASET_FAILURES[f"exp10/{name}"] = f"{type(ex).__name__}: {ex}"
            print(f"  !! exp10: '{name}' FAILED ({type(ex).__name__}: {ex})")
        finally:
            __import__("gc").collect()
    ck.report()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp10_instance_noise.csv"), index=False)

    txt = ["EXP 10 — INSTANCE-DEPENDENT NOISE", ""]
    stats = []
    for scheme in EXP10_SCHEMES:
        sub = df[df.Scheme == scheme]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="NoiseRate", columns="Cleaner", values="F1")
        head = f"--- {scheme} : end-task F1 by rate ---"
        print("\n" + head); print(piv.round(4).to_string())
        txt += [head, piv.round(4).to_string(), ""]
        if "RankAUC" in sub.columns:
            rk = sub[sub.Cleaner == "AMEND"].pivot_table(
                index="NoiseRate", values=["RankAUC", "NoiseRec", "FRR"])
            txt += ["AMEND ranking/detection under this scheme:",
                    rk.round(4).to_string(), ""]
        for rate in EXP10_RATES:
            s2 = sub[sub.NoiseRate == rate]
            for cl in classifiers:
                p = s2[s2.Classifier == cl].pivot_table(
                    index="Dataset", columns="Cleaner", values="F1")
                txt.append(_friedman_wilcoxon(
                    p, f"{scheme}@{rate} / {cl} / F1", ref="AMEND"))
                st = _pairwise_stats_df(p, "AMEND", Experiment="exp10",
                                        Scheme=scheme, NoiseRate=rate,
                                        Classifier=cl, Metric="F1")
                if len(st):
                    stats.append(st)
    with open(os.path.join(results_dir, "exp10_stats.txt"), "w") as fh:
        fh.write("\n".join(txt))
    if stats:
        pd.concat(stats, ignore_index=True).to_csv(
            os.path.join(results_dir, "exp10_pairwise_stats.csv"), index=False)

    try:
        for scheme in EXP10_SCHEMES:
            sub = df[df.Scheme == scheme]
            piv = sub.pivot_table(index="NoiseRate", columns="Cleaner", values="F1")
            if piv.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for c in piv.columns:
                ax.plot(piv.index, piv[c], marker="o", label=c,
                        lw=(2.5 if c.startswith("AMEND") else 1.3))
            ax.set_xlabel(f"injected rate ({scheme})")
            ax.set_ylabel("end-task F1"); ax.legend(); ax.grid(alpha=0.3)
            ax.set_title("Instance-dependent label noise")
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir, f"exp10_{scheme}.png"), dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"  plot skipped ({e})")
    print("[exp10] saved exp10_instance_noise.csv, exp10_stats.txt (+ plots)")
    return df


# ============================================================================
# EXPERIMENT 11 — INTERNAL-STATE AUDIT  (no classifiers; fast)
# ============================================================================
def run_exp11(datasets, seeds, n_splits, results_dir, ck):
    print("\n" + "=" * 74)
    print("EXP 11 — INTERNAL-STATE AUDIT (k, r, r-cap, p, tau, weights,")
    print("         effective class balance). No classifiers are trained.")
    print("=" * 74)
    rows = []
    for name, X, y in datasets:
        try:
            pos = minority_label(y)
            for scheme, rate in EXP11_CONDITIONS:
                fac = lambda s: build_cleaner_ext("AMEND", s)
                _, d = multi_seed_ext(
                    X, y, fac, seeds, n_splits, rate, scheme, [], pos,
                    "AMEND", ck, (name, "state", scheme, rate),
                    want_classifiers=False)
                for r in d:
                    row = {"Dataset": name, "Scheme": scheme, "NoiseRate": rate}
                    row.update(r)
                    rows.append(row)
            print(f"  done: {name}")
        except Exception as ex:
            _DATASET_FAILURES[f"exp11/{name}"] = f"{type(ex).__name__}: {ex}"
            print(f"  !! exp11: '{name}' FAILED ({type(ex).__name__}: {ex})")
        finally:
            __import__("gc").collect()
    ck.report()

    df = pd.DataFrame(rows)
    if df.empty:
        print("[exp11] no rows"); return df
    df.to_csv(os.path.join(results_dir, "exp11_internal_state_folds.csv"),
              index=False)

    cols = [c for c in STATE_KEYS if c in df.columns]
    per_ds = df.groupby(["Dataset", "Scheme", "NoiseRate"])[cols].mean().round(4)
    per_ds.to_csv(os.path.join(results_dir, "exp11_internal_state.csv"))

    ref = df[(df.Scheme == "uniform") & (df.NoiseRate == 0.15)]
    if len(ref):
        show = [c for c in ["k_used", "r_used", "r_cap", "r_at_cap", "proj_dim",
                            "tau", "removal_rate", "frac_w_eq1", "frac_w_le05",
                            "eff_IR_before", "eff_IR_after"] if c in ref.columns]
        print("\nPer dataset at 15% uniform noise:")
        print(ref.groupby("Dataset")[show].mean().round(3).to_string())
        print("\nOverall means:")
        print(ref[show].mean().round(4).to_string())
        if "r_at_cap" in ref:
            print(f"\n  r sits at its k-1 cap in {ref.r_at_cap.mean():.1%} of "
                  f"folds. If that is most of them, the local subspace spans "
                  f"the whole neighbourhood and SCS is largely measuring "
                  f"neighbourhood overlap.")
        if {"eff_IR_before", "eff_IR_after"} <= set(ref.columns):
            print(f"  Effective imbalance ratio {ref.eff_IR_before.mean():.2f} "
                  f"-> {ref.eff_IR_after.mean():.2f} after cleaning. That shift "
                  f"is the size of AMEND's rebalancing effect.")
        if "frac_w_eq1" in ref:
            print(f"  Only {ref.frac_w_eq1.mean():.1%} of retained samples keep "
                  f"weight 1; the rest are down-weighted.")
    print("[exp11] saved exp11_internal_state.csv (+ per-fold file)")
    return df


# ============================================================================
# POST-HOC: CLIFF'S DELTA FROM THE EXISTING EXP3 FILES  (no re-run)
# ============================================================================
def export_exp3_pairwise(results_dir):
    """Machine-readable effect sizes for Table IV, from results you already have."""
    b = os.path.join(results_dir, "exp3_baselines.csv")
    dpath = os.path.join(results_dir, "exp3_detection.csv")
    if not os.path.exists(b):
        print("[posthoc] exp3_baselines.csv not found — skipped"); return
    df = pd.read_csv(b)
    df["Cleaner"] = df["Cleaner"].fillna("None")
    out = []
    for cl in df.Classifier.dropna().unique():
        sub = df[df.Classifier == cl]
        for metric in ["Recall", "Precision", "F1", "GMean", "FPR", "AUC"]:
            if metric not in sub.columns:
                continue
            piv = sub.pivot_table(index="Dataset", columns="Cleaner", values=metric)
            s = _pairwise_stats_df(piv, "AMEND",
                                   lower_is_better=(metric == "FPR"),
                                   Experiment="exp3", Classifier=cl,
                                   Metric=metric)
            if len(s):
                out.append(s)
    if os.path.exists(dpath):
        dd = pd.read_csv(dpath)
        dd["Cleaner"] = dd["Cleaner"].fillna("None")
        dd = dd[dd.Cleaner != "None"]
        for metric, lower in [("NoisePrec", False), ("NoiseRec", False),
                              ("NoiseF1", False), ("FRR", True),
                              ("FlagF1", False)]:
            if metric not in dd.columns:
                continue
            piv = dd.pivot_table(index="Dataset", columns="Cleaner", values=metric)
            s = _pairwise_stats_df(piv, "AMEND", lower_is_better=lower,
                                   Experiment="exp3", Classifier="-",
                                   Metric=metric)
            if len(s):
                out.append(s)
    if not out:
        print("[posthoc] nothing to write"); return
    res = pd.concat(out, ignore_index=True)
    p = os.path.join(results_dir, "exp3_pairwise_stats.csv")
    res.to_csv(p, index=False)
    print(f"[posthoc] wrote {p} — dmean, Holm p and Cliff's delta for every "
          f"comparison in Table IV ({len(res)} rows)")
    return res


# ============================================================================
# MAIN
# ============================================================================
def _make_extra_checkpoints():
    if not RESUME:
        return {e: _NullCheckpoint() for e in
                ("exp7", "exp8", "exp9", "exp10", "exp11")}
    cdir = CHECKPOINT_DIR or os.path.join(RESULTS_DIR, "_checkpoints")
    base = {"n_splits": N_SPLITS, "use_sample_weights": USE_SAMPLE_WEIGHTS,
            "flag_threshold": FLAG_WEIGHT_THRESHOLD,
            "amend_defaults": dict(AMEND_DEFAULTS)}
    cfgs = {
        "exp7": {**base, "classifiers": list(EXP3_CLASSIFIERS),
                 "cleaners": list(EXP7_CLEANERS),
                 "conditions": [list(c) for c in EXTRA_CONDITIONS],
                 "rus_fixed": RUS_FIXED_RATE},
        "exp8": {**base, "classifiers": list(EXP3_CLASSIFIERS),
                 "w_grid": list(EXP8_W_GRID),
                 "condition": list(EXP8_CONDITION)},
        "exp9": {**base, "classifiers": list(EXP3_CLASSIFIERS),
                 "bases": list(EXP9_BASE_CLASSIFIERS),
                 "conditions": [list(c) for c in EXTRA_CONDITIONS]},
        "exp10": {**base, "classifiers": list(EXP3_CLASSIFIERS),
                  "schemes": list(EXP10_SCHEMES), "rates": list(EXP10_RATES),
                  "cleaners": list(EXP10_CLEANERS)},
        "exp11": {**base, "conditions": [list(c) for c in EXP11_CONDITIONS]},
    }
    print(f"Resume (add-on): ENABLED — log dir {cdir}")
    return {n: Checkpoint(n, c, cdir, enabled=True, fresh=FRESH_START)
            for n, c in cfgs.items()}


def main_extra():

    _ensure_drive()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 74)
    print("AMEND ADD-ON SUITE — Exp 7-11")
    print(f"Results: {RESULTS_DIR}")
    print(f"Eval:    {len(SEEDS)} seeds x {N_SPLITS}-fold CV")
    print("Existing exp1/exp2/exp3 checkpoints are NOT touched.")
    print("=" * 74)

    datasets = _subset(load_binary_datasets(FOLDER_PATH), EXTRA_DATASETS)
    if not datasets:
        raise RuntimeError("No binary datasets found — check FOLDER_PATH.")
    print(f"\nTotal binary datasets: {len(datasets)}")

    cks = _make_extra_checkpoints()
    try:
        if RUN_POSTHOC_CLIFFS:
            export_exp3_pairwise(RESULTS_DIR)
        if RUN_EXP11:      # cheapest and most informative — run it first
            run_exp11(datasets, SEEDS, N_SPLITS, RESULTS_DIR, cks["exp11"])
        if RUN_EXP8:
            run_exp8(datasets, SEEDS, N_SPLITS, RESULTS_DIR,
                     EXP3_CLASSIFIERS, cks["exp8"])
        if RUN_EXP7:
            run_exp7(datasets, SEEDS, N_SPLITS, RESULTS_DIR,
                     EXP3_CLASSIFIERS, cks["exp7"])
        if RUN_EXP9:
            run_exp9(datasets, SEEDS, N_SPLITS, RESULTS_DIR,
                     EXP3_CLASSIFIERS, cks["exp9"])
        if RUN_EXP10:
            run_exp10(datasets, SEEDS, N_SPLITS, RESULTS_DIR,
                      EXP3_CLASSIFIERS, cks["exp10"])
    except KeyboardInterrupt:
        print("\nINTERRUPTED — completed units are on disk; re-run to resume.")
        raise
    finally:
        for c in cks.values():
            c.close()

    if _DATASET_FAILURES:
        print("\n" + "!" * 74)
        print("DATASETS THAT FAILED AND WERE SKIPPED:")
        for k, v in sorted(_DATASET_FAILURES.items()):
            print(f"  {k}: {v}")
        print("!" * 74)
    if _CLEANER_FAILURES or _CLASSIFIER_FAILURES:
        print("\n" + "!" * 74)
        print("SILENT FAILURES CAUGHT (check before reporting):")
        for k, v in sorted(_CLEANER_FAILURES.items()):
            print(f"  cleaner    {k}  x{v}")
        for k, v in sorted(_CLASSIFIER_FAILURES.items()):
            print(f"  classifier {k}  x{v}")
        print("!" * 74)

    print("\n" + "=" * 74)
    print("ADD-ON EXPERIMENTS COMPLETE")
    print(f"Outputs written to: {RESULTS_DIR}")
    print("=" * 74)




def run_amend():
    """Run the selected experiments from the unified AMEND suite."""
    main()
    main_extra()


if __name__ == "__main__":
    run_amend()