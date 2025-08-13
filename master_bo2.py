#!/usr/bin/env python3
"""
Unified Multi-Objective Bayesian Optimization Script (Optimized Version)

This script integrates three models (logistic, random forest, decision tree) and supports:
  - One GP implementation using BOtorch (with GPU support when available)
  - Candidate generation methods (e.g. "dycors", "dycors_org", "sobol", etc.)
  - Candidate selection methods (e.g. "pareto", "ehvi", "parego")
  
Reproducibility is enforced via fixed seeds and by placing all tensor operations on the proper device.
"""

import random
import numpy as np
import torch
import os, time, warnings
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.inspection import permutation_importance

# Data & Fairness
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric

# Models and Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# GP (Botorch & GPyTorch)
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling import SobolQMCNormalSampler, SobolEngine
from botorch.optim import optimize_acqf_discrete
from botorch.acquisition.multi_objective.parego import qLogNParEGO, TAU_MAX, TAU_RELU
import gpytorch

# For LHS sampling (skopt)
from skopt.space import Space, Real, Integer, Categorical
from skopt.sampler import Lhs

warnings.filterwarnings("ignore")

# ========================
# Global settings for reproducibility and device management
# ========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Define device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    # For additional reproducibility on GPU:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Using device: {device}")
if device.type == "cuda":
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


# ========================
# DATASET LOADING
# ========================
def load_compas_dataset():
    dataset = CompasDataset()
    X = dataset.features
    y = dataset.labels.ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    return dataset, X_train, X_test, y_train, y_test

# ========================
# BOtorch GP FUNCTIONS (GPU-enabled)
# ========================
def gp_fit_botorch(X, Y):
    models = []
    # Convert training data to torch tensors on the proper device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    for i in range(Y.shape[1]):
        Y_i = torch.tensor(Y[:, i], dtype=torch.float32, device=device).unsqueeze(-1)
        gp = SingleTaskGP(X_tensor, Y_i).to(device)
        # Ensure mean and covariance modules are on the device
        gp.mean_module = gp.mean_module.to(device)
        gp.covar_module = gp.covar_module.to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll_torch(mll)
        models.append(gp)
    return ModelListGP(*models).to(device)

def gp_predict_botorch(models, X_pred):
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32, device=device)
    means = []
    # Use mixed precision if on GPU
    context = torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad()
    with context:
        for model in models.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_pred_tensor)  # Returns a MultivariateNormal
            means.append(pred.mean.detach().cpu().numpy().flatten())
    return np.vstack(means).T

# ========================
# FEATURE IMPORTANCE VIA PERMUTATION (Method 1)
# ========================
from sklearn.base import BaseEstimator, RegressorMixin
class GPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, gp_model):
        self.gp_model = gp_model
    def fit(self, X, y=None):
        # Already fitted
        return self
    def predict(self, X):
        self.gp_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            pred = self.gp_model(X_tensor).mean
        return pred.detach().cpu().numpy()

def calculate_feature_importance_1(gp_models, decimals=3, epsilon=1e-6, n_repeats=5):
    from sklearn.inspection import permutation_importance
    importance_list = []
    for model in gp_models.models:
        # Retrieve training data (ensure data is on CPU as required by scikit-learn)
        X_tr = model.train_inputs[0].detach().cpu().numpy()
        y_tr = model.train_targets.detach().cpu().numpy()
        wrapped = GPWrapper(model)
        def rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred)**2))
        # The scorer expects estimator, X, y:
        scoring = lambda est, X, y, sample_weight=None: -rmse(y, est.predict(X))
        result = permutation_importance(wrapped, X_tr, y_tr,
                                        scoring=scoring, n_repeats=n_repeats,
                                        random_state=SEED, n_jobs=-1)
        imp = 1.0 / (result.importances_mean + epsilon)
        importance_list.append(imp)
    importance_matrix = np.vstack(importance_list).T  # shape (d, num_objectives)
    return np.round(importance_matrix, decimals=decimals)

# ========================
# FEATURE IMPORTANCE VIA PERMUTATION (Method 2)
# ========================
def calculate_feature_importance_2(gp_models, decimals=3, epsilon=1e-6, n_repeats=5):
    rng = np.random.RandomState(SEED)
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))
    importance_list = []
    for model in gp_models.models:
        X_tr = model.train_inputs[0].detach().cpu().numpy()
        y_tr = model.train_targets.detach().cpu().numpy()
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_tr, dtype=torch.float32, device=device)
            y_pred = model(X_tensor).mean.detach().cpu().numpy()
        baseline = rmse(y_tr, y_pred)
        d = X_tr.shape[1]
        perm_importance = np.zeros(d)
        for i in range(d):
            scores = []
            for _ in range(n_repeats):
                X_perm = X_tr.copy()
                rng.shuffle(X_perm[:, i])
                with torch.no_grad():
                    Xp_tensor = torch.tensor(X_perm, dtype=torch.float32, device=device)
                    y_perm_pred = model(Xp_tensor).mean.detach().cpu().numpy()
                scores.append(rmse(y_tr, y_perm_pred))
            increase = np.mean(scores) - baseline
            perm_importance[i] = increase
        imp = 1.0 / (perm_importance + epsilon)
        importance_list.append(imp)
    importance_matrix = np.vstack(importance_list).T
    return np.round(importance_matrix, decimals=decimals)

# ========================
# PARETO FRONTIER FUNCTIONS
# ========================
def identify_pareto_front(Y):
    pareto_idx = []
    num_points = Y.shape[0]
    for i in range(num_points):
        dominated = False
        for j in range(num_points):
            if all(Y[j] <= Y[i]) and any(Y[j] < Y[i]):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    return pareto_idx

def select_center_by_auc_reduction(Y, X):
    # Compute Pareto frontier indices based on Y
    pareto_idx = identify_pareto_front(Y)
    pareto_Y = Y[pareto_idx]
    pareto_X = X[pareto_idx]
    pareto_Y_clipped = np.clip(pareto_Y, 0, 1)
    sorted_idx = np.argsort(pareto_Y_clipped[:, 0])
    pareto_Y_sorted = pareto_Y_clipped[sorted_idx]
    full_auc = np.trapz(pareto_Y_sorted[:, 1], pareto_Y_sorted[:, 0])
    auc_reductions = []
    for i in range(len(pareto_Y_sorted)):
        reduced = np.delete(pareto_Y_sorted, i, axis=0)
        if reduced.shape[0] < 2:
            auc_reductions.append(0)
        else:
            new_auc = np.trapz(reduced[:, 1], reduced[:, 0])
            auc_reductions.append(full_auc - new_auc)
    auc_reductions = np.array(auc_reductions)
    best_idx_in_sorted = np.argmax(auc_reductions)
    best_center = pareto_X[sorted_idx[best_idx_in_sorted]]
    return best_center

def select_top2_centers_by_auc_reduction(Y, X):
    pareto_idx = identify_pareto_front(Y)
    pareto_Y = Y[pareto_idx]
    pareto_X = X[pareto_idx]
    pareto_Y_clipped = np.clip(pareto_Y, 0, 1)
    sorted_idx = np.argsort(pareto_Y_clipped[:, 0])
    pareto_Y_sorted = pareto_Y_clipped[sorted_idx]
    full_auc = np.trapz(pareto_Y_sorted[:, 1], pareto_Y_sorted[:, 0])
    auc_reductions = []
    for i in range(len(pareto_Y_sorted)):
        reduced = np.delete(pareto_Y_sorted, i, axis=0)
        if reduced.shape[0] < 2:
            auc_reductions.append(0)
        else:
            new_auc = np.trapz(reduced[:, 1], reduced[:, 0])
            auc_reductions.append(full_auc - new_auc)
    auc_reductions = np.array(auc_reductions)
    if len(auc_reductions) == 0:
        raise ValueError("No Pareto points found to select centers.")
    elif len(auc_reductions) == 1:
        top_indices = [0, 0]
    else:
        top_indices = np.argsort(auc_reductions)[-2:][::-1]
    best_centers = [pareto_X[sorted_idx[idx]] for idx in top_indices]
    return best_centers

# ========================
# CANDIDATE GENERATION FUNCTIONS
# ========================
def dycors(xbest, num_cand, lb, up, scalefactor, d):
    subset = np.arange(d)
    prob_perturb = 1  # constant perturbation probability
    ar = np.random.rand(num_cand, len(subset)) < prob_perturb
    ind = np.where(np.sum(ar, axis=1) == 0)[0]
    if len(ind) > 0:
        ar[ind, np.random.randint(0, len(subset), size=len(ind))] = 1
    cand = np.tile(xbest, (num_cand, 1))
    for i in subset:
        lower, upper, sigma = lb[i], up[i], scalefactor
        indices = np.where(ar[:, i] == 1)[0]
        if len(indices) > 0:
            p = truncnorm.rvs(a=(lower - xbest[i]) / sigma,
                              b=(upper - xbest[i]) / sigma,
                              loc=xbest[i],
                              scale=sigma,
                              size=len(indices))
            cand[indices, i] = p
    return np.array(cand)

def dycors_org(xbest, num_cand, lb, up, scalefactor, d, num_evals, max_evals):
    min_prob = min(1.0, 1.0/d)
    prob_perturb = min(20.0/d, 1.0) * (1.0 - (np.log(num_evals) / np.log(max_evals)))
    prob_perturb = max(prob_perturb, min_prob)
    ar = np.random.rand(num_cand, d) < prob_perturb
    ind = np.where(np.sum(ar, axis=1) == 0)[0]
    if len(ind) > 0:
        ar[ind, np.random.randint(0, d, size=len(ind))] = 1
    cand = np.tile(xbest, (num_cand, 1))
    for i in range(d):
        lower, upper, sigma = lb[i], up[i], scalefactor
        indices = np.where(ar[:, i] == 1)[0]
        if len(indices) > 0:
            p = truncnorm.rvs(a=(lower - xbest[i]) / sigma,
                              b=(upper - xbest[i]) / sigma,
                              loc=xbest[i],
                              scale=sigma,
                              size=len(indices))
            cand[indices, i] = p
    return cand

def dycors_fi1(xbest, num_cand, lb, up, scalefactor, d, gp_models):
    # Compute feature importance using permutation importance (Method 1)
    importance_matrix = calculate_feature_importance_1(gp_models, decimals=3)
    pareto_idx = identify_pareto_front(importance_matrix)
    fi_pareto_set = set(pareto_idx)
    ar = np.zeros((num_cand, d), dtype=bool)
    for i in range(d):
        ar[:, i] = True if i in fi_pareto_set else False
    ind = np.where(np.sum(ar, axis=1) == 0)[0]
    if len(ind) > 0:
        ar[ind, np.random.randint(0, d, size=len(ind))] = True
    cand = np.tile(xbest, (num_cand, 1))
    for i in range(d):
        if np.any(ar[:, i]):
            lower, upper, sigma = lb[i], up[i], scalefactor
            indices = np.where(ar[:, i])[0]
            p = truncnorm.rvs(a=(lower - xbest[i]) / sigma,
                              b=(upper - xbest[i]) / sigma,
                              loc=xbest[i],
                              scale=sigma,
                              size=len(indices))
            cand[indices, i] = p
    return np.array(cand)

def dycors_fi2(xbest, num_cand, lb, up, scalefactor, d, gp_models):
    # Compute feature importance using permutation importance (Method 2)
    importance_matrix = calculate_feature_importance_2(gp_models, decimals=3)
    pareto_idx = identify_pareto_front(importance_matrix)
    fi_pareto_set = set(pareto_idx)
    ar = np.zeros((num_cand, d), dtype=bool)
    for i in range(d):
        ar[:, i] = True if i in fi_pareto_set else False
    ind = np.where(np.sum(ar, axis=1) == 0)[0]
    if len(ind) > 0:
        ar[ind, np.random.randint(0, d, size=len(ind))] = True
    cand = np.tile(xbest, (num_cand, 1))
    for i in range(d):
        if np.any(ar[:, i]):
            lower, upper, sigma = lb[i], up[i], scalefactor
            indices = np.where(ar[:, i])[0]
            p = truncnorm.rvs(a=(lower - xbest[i]) / sigma,
                              b=(upper - xbest[i]) / sigma,
                              loc=xbest[i],
                              scale=sigma,
                              size=len(indices))
            cand[indices, i] = p
    return np.array(cand)

def generate_candidates_sobol(lb, up, pool_size, d, seed=SEED):
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    candidates = sobol.draw(pool_size).numpy()
    candidates = lb + (up - lb) * candidates
    return candidates

# ========================
# CANDIDATE SELECTION FUNCTIONS
# ========================
def select_candidates_pareto(Y_pred, candidates, batch_size):
    idx = identify_pareto_front(Y_pred)
    selected = candidates[idx]
    if selected.shape[0] >= batch_size:
        indices = np.random.choice(selected.shape[0], batch_size, replace=False)
        return selected[indices]
    else:
        return selected

def select_candidates_ehvi(models_botorch, candidates, batch_size, Y_obs):
    candidates_torch = torch.tensor(candidates, dtype=torch.float32, device=device)
    ref_point = np.min(Y_obs, axis=0) - 1e-4
    ref_point_tensor = torch.tensor(ref_point, dtype=torch.float32, device=device)
    Y_obs_tensor = torch.tensor(Y_obs, dtype=torch.float32, device=device)
    partitioning = NondominatedPartitioning(ref_point=ref_point_tensor, Y=Y_obs_tensor)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1000]))
    acq_func = qExpectedHypervolumeImprovement(
        model=models_botorch,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler=sampler
    )
    selected, _ = optimize_acqf_discrete(
        acq_function=acq_func,
        q=batch_size,
        choices=candidates_torch,
        unique=True
    )
    return selected.detach().cpu().numpy()

def select_candidates_qlognparego(models_botorch, candidates, batch_size, Y_obs, X_baseline,
                                  scalarization_weights=None, sampler=None,
                                  objective=None, constraints=None, X_pending=None,
                                  eta=1e-3, fat=True, prune_baseline=False,
                                  cache_root=True, tau_relu=TAU_RELU, tau_max=TAU_MAX):
    candidates_torch = torch.tensor(candidates, dtype=torch.float32, device=device)
    if sampler is None:
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1000]))
    acq_func = qLogNParEGO(
        model=models_botorch,
        X_baseline=torch.tensor(X_baseline, dtype=torch.float32, device=device),
        scalarization_weights=scalarization_weights,
        sampler=sampler,
        objective=objective,
        constraints=constraints,
        X_pending=X_pending,
        eta=eta,
        fat=fat,
        prune_baseline=prune_baseline,
        cache_root=cache_root,
        tau_relu=tau_relu,
        tau_max=tau_max,
    )
    selected, _ = optimize_acqf_discrete(
        acq_function=acq_func,
        q=batch_size,
        choices=candidates_torch,
        unique=True
    )
    return selected.detach().cpu().numpy()

# ========================
# ROUNDING FUNCTIONS
# ========================
def rounding_logistic(x):
    x_rounded = x.copy()
    x_rounded[:, 1] = np.round(x[:, 1]).astype(int)
    x_rounded[:, 1] = np.clip(x_rounded[:, 1], 0, 3)
    x_rounded[:, 2] = np.round(x[:, 2]).astype(int)
    return x_rounded

def rounding_rf(x):
    return np.round(x).astype(int)

def rounding_rf2(x):
    x_rounded = x.copy()
    # Round the integer hyperparameters:
    x_rounded[:, 0] = np.round(x[:, 0]).astype(int)   # max_depth
    x_rounded[:, 1] = np.round(x[:, 1]).astype(int)   # min_samples_split
    x_rounded[:, 2] = np.round(x[:, 2]).astype(int)   # max_leaf_nodes
    x_rounded[:, 3] = np.round(x[:, 3]).astype(int)   # min_samples_leaf
    x_rounded[:, 4] = np.round(x[:, 4]).astype(int)   # n_estimators
    # For the fractional hyperparameters, round to 3 decimals:
    x_rounded[:, 5] = np.round(x[:, 5], 3)            # max_samples
    x_rounded[:, 6] = np.round(x[:, 6], 3)            # max_features
    return x_rounded


def rounding_dt(x):
    return np.round(x).astype(int)

# ========================
# MAPPING FUNCTION (for logistic regression)
# ========================
def map_index_to_solver(index):
    solvers = ['lbfgs', 'liblinear', 'sag', 'saga']
    if 0 <= index < len(solvers):
        return solvers[index]
    else:
        raise ValueError("Invalid solver index")

# ========================
# EVALUATION FUNCTIONS
# ========================
def evaluate_logistic(hp, X_train, X_test, y_train, y_test, dataset):
    try:
        solver_str = map_index_to_solver(int(hp[1]))
        hp_dict = {'C': hp[0], 'solver': solver_str, 'max_iter': int(hp[2])}
        if hp_dict['max_iter'] <= 0:
            raise ValueError("Invalid hyperparameters.")
        model = LogisticRegression(**hp_dict)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        error_rate = 1 - f1
        df = pd.DataFrame(X_test, columns=dataset.feature_names)
        df['two_year_recid'] = y_pred
        test_ds = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                     df=df, label_names=['two_year_recid'],
                                     protected_attribute_names=['race'])
        metric = BinaryLabelDatasetMetric(test_ds,
                                          privileged_groups=[{'race': 1}],
                                          unprivileged_groups=[{'race': 0}])
        stat_parity = metric.statistical_parity_difference()
        return np.array([error_rate, stat_parity])
    except Exception as e:
        with open("evaluation_errors.log", "a") as f:
            f.write(f"Logistic error with hp {hp}: {e}\n")
        return np.array([1e6, 1e6])

def evaluate_rf(hp, X_train, X_test, y_train, y_test, dataset):
    try:
        hp_dict = {'n_estimators': int(hp[0]),
                   'max_depth': int(hp[1]),
                   'min_samples_split': int(hp[2]),
                   'random_state': SEED}
        if hp_dict['n_estimators'] < 1 or hp_dict['max_depth'] < 1 or hp_dict['min_samples_split'] < 2:
            raise ValueError("Invalid hyperparameters.")
        model = RandomForestClassifier(**hp_dict)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        error_rate = 1 - f1
        df = pd.DataFrame(X_test, columns=dataset.feature_names)
        df['two_year_recid'] = y_pred
        test_ds = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                     df=df, label_names=['two_year_recid'],
                                     protected_attribute_names=['race'])
        metric = BinaryLabelDatasetMetric(test_ds,
                                          privileged_groups=[{'race': 1}],
                                          unprivileged_groups=[{'race': 0}])
        stat_parity = metric.statistical_parity_difference()
        return np.array([error_rate, stat_parity])
    except Exception as e:
        with open("evaluation_errors.log", "a") as f:
            f.write(f"RF error with hp {hp}: {e}\n")
        return np.array([1e6, 1e6])

def evaluate_dt(hp, X_train, X_test, y_train, y_test, dataset):
    try:
        hp_dict = {'max_depth': int(hp[0]),
                   'min_samples_split': int(hp[1]),
                   'min_samples_leaf': int(hp[2]),
                   'random_state': SEED}
        if hp_dict['max_depth'] < 1 or hp_dict['min_samples_split'] < 2 or hp_dict['min_samples_leaf'] < 1:
            raise ValueError("Invalid hyperparameters.")
        model = DecisionTreeClassifier(**hp_dict)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        error_rate = 1 - f1
        df = pd.DataFrame(X_test, columns=dataset.feature_names)
        df['two_year_recid'] = y_pred
        test_ds = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                     df=df, label_names=['two_year_recid'],
                                     protected_attribute_names=['race'])
        metric = BinaryLabelDatasetMetric(test_ds,
                                          privileged_groups=[{'race': 1}],
                                          unprivileged_groups=[{'race': 0}])
        stat_parity = metric.statistical_parity_difference()
        return np.array([error_rate, stat_parity])
    except Exception as e:
        with open("evaluation_errors.log", "a") as f:
            f.write(f"DT error with hp {hp}: {e}\n")
        return np.array([1e6, 1e6])
    


def evaluate_random_forest2(hp, X_train, X_test, y_train, y_test, dataset):
    try:
        # hp order: [max_depth, min_samples_split, max_leaf_nodes, min_samples_leaf, n_estimators, max_samples, max_features]
        hp_dict = {
            'max_depth': int(hp[0]),
            'min_samples_split': int(hp[1]),
            'max_leaf_nodes': int(hp[2]),
            'min_samples_leaf': int(hp[3]),
            'n_estimators': int(hp[4]),
            'max_samples': float(hp[5]),    # Fraction of samples used for bootstrap
            'max_features': float(hp[6]),     # Fraction of features to consider
            'bootstrap': True,
            'random_state': SEED
        }
        # Validate hyperparameters (optional)
        if hp_dict['max_depth'] < 1 or hp_dict['min_samples_split'] < 2 or hp_dict['min_samples_leaf'] < 1 or hp_dict['n_estimators'] < 1:
            raise ValueError("Invalid hyperparameters.")
        model = RandomForestClassifier(**hp_dict)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        error_rate = 1 - f1
        df = pd.DataFrame(X_test, columns=dataset.feature_names)
        df['two_year_recid'] = y_pred
        test_ds = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                     df=df, label_names=['two_year_recid'],
                                     protected_attribute_names=['race'])
        metric = BinaryLabelDatasetMetric(test_ds,
                                          privileged_groups=[{'race': 1}],
                                          unprivileged_groups=[{'race': 0}])
        stat_parity = metric.statistical_parity_difference()
        return np.array([error_rate, stat_parity])
    except Exception as e:
        with open("evaluation_errors.log", "a") as f:
            f.write(f"RF2 error with hp {hp}: {e}\n")
        return np.array([1e6, 1e6])

# ========================
# MODEL CONFIGURATION
# ========================
def get_model_config(model_type):
    config = {}
    if model_type == "logistic":
        config["evaluate"] = evaluate_logistic
        config["initial_pool_filename"] = "pool_initial_lr.csv"
        config["bounds"] = (np.array([0.001, 0, 100]), np.array([1000, 1, 500]))
        config["round_func"] = rounding_logistic
        config["lhs_dim"] = 3
        def generate_lhs_lr(dim, lb, ub, output_dir):
            space_dims = [
                Categorical([0.001, 0.01, 0.1, 1, 10, 100, 1000]),
                Real(lb[1], ub[1]),
                Real(lb[2], ub[2])
            ]
            space = Space(space_dims)
            NN = 2 * (dim + 1)
            lhs_sampler = Lhs(criterion="maximin", iterations=10000)
            samples = np.array(lhs_sampler.generate(space.dimensions, NN, random_state=SEED), dtype=object)
            os.makedirs(output_dir, exist_ok=True)
            df = pd.DataFrame(samples, columns=[f"Param_{i+1}" for i in range(dim)])
            pool_path = os.path.join(output_dir, config["initial_pool_filename"])
            df.to_csv(pool_path, index=False)
            return samples
        config["generate_lhs"] = generate_lhs_lr
        def process_lr_samples(X_cont):
            X_proc = X_cont.copy()
            X_proc[:, 1] = [int(np.clip(np.floor(val * 4), 0, 3)) for val in X_cont[:, 1]]
            X_proc[:, 2] = np.round(X_cont[:, 2]).astype(int)
            return X_proc
        config["process_lhs"] = process_lr_samples
        config["evaluate_wrapper"] = lambda hp, X_tr, X_te, y_tr, y_te, ds: evaluate_logistic(hp, X_tr, X_te, y_tr, y_te, ds)
    elif model_type == "random_forest":
        config["evaluate"] = evaluate_rf
        config["initial_pool_filename"] = "pool_initial_rf.csv"
        config["bounds"] = (np.array([10, 2, 2]), np.array([500, 50, 10]))
        config["round_func"] = rounding_rf
        config["lhs_dim"] = 3
        def generate_lhs_rf(dim, lb, ub, output_dir):
            space = Space([Integer(lb[i], ub[i]) for i in range(dim)])
            NN = 2 * (dim + 1)
            lhs_sampler = Lhs(criterion="maximin", iterations=10000)
            samples = np.array(lhs_sampler.generate(space.dimensions, NN, random_state=SEED))
            os.makedirs(output_dir, exist_ok=True)
            df = pd.DataFrame(samples, columns=[f"Param_{i+1}" for i in range(dim)])
            pool_path = os.path.join(output_dir, config["initial_pool_filename"])
            df.to_csv(pool_path, index=False)
            return samples
        config["generate_lhs"] = generate_lhs_rf
        config["process_lhs"] = lambda X: np.round(X).astype(int)
        config["evaluate_wrapper"] = lambda hp, X_tr, X_te, y_tr, y_te, ds: evaluate_rf(hp, X_tr, X_te, y_tr, y_te, ds)
    elif model_type == "random_forest2":
        config["evaluate"] = evaluate_random_forest2
        config["initial_pool_filename"] = "pool_initial_rf2.csv"
        lb = np.array([5, 2, 10, 1, 50, 0.5, 0.1])
        up = np.array([100, 20, 100, 20, 1000, 1.0, 1.0])
        config["bounds"] = (lb, up)
        config["lhs_dim"] = 7

        def generate_lhs_rf2(dim, lb, up, output_dir):
            from skopt.space import Space, Integer, Real
            space = Space([
                Integer(lb[0], up[0]),   # max_depth
                Integer(lb[1], up[1]),   # min_samples_split
                Integer(lb[2], up[2]),   # max_leaf_nodes
                Integer(lb[3], up[3]),   # min_samples_leaf
                Integer(lb[4], up[4]),   # n_estimators
                Real(lb[5], up[5]),      # max_samples
                Real(lb[6], up[6])       # max_features
            ])
            NN = 2 * (dim + 1)
            lhs_sampler = Lhs(criterion="maximin", iterations=10000)
            samples = np.array(lhs_sampler.generate(space.dimensions, NN, random_state=SEED))
            os.makedirs(output_dir, exist_ok=True)
            df = pd.DataFrame(samples, columns=[f"Param_{i+1}" for i in range(dim)])
            pool_path = os.path.join(output_dir, config["initial_pool_filename"])
            df.to_csv(pool_path, index=False)
            return samples

        def process_rf2_samples(X_cont):
            X_proc = X_cont.copy()
            X_proc[:, 0] = np.round(X_cont[:, 0]).astype(int)
            X_proc[:, 1] = np.round(X_cont[:, 1]).astype(int)
            X_proc[:, 2] = np.round(X_cont[:, 2]).astype(int)
            X_proc[:, 3] = np.round(X_cont[:, 3]).astype(int)
            X_proc[:, 4] = np.round(X_cont[:, 4]).astype(int)
            X_proc[:, 5] = np.round(X_cont[:, 5], 3)
            X_proc[:, 6] = np.round(X_cont[:, 6], 3)
            return X_proc

        config["generate_lhs"] = generate_lhs_rf2
        config["process_lhs"] = process_rf2_samples
        config["evaluate_wrapper"] = lambda hp, X_tr, X_te, y_tr, y_te, ds: evaluate_random_forest2(hp, X_tr, X_te, y_tr, y_te, ds)
        config["round_func"] = rounding_rf2  # This is the new line you must add.
    elif model_type == "decision_tree":
        config["evaluate"] = evaluate_dt
        config["initial_pool_filename"] = "pool_initial_dt.csv"
        config["bounds"] = (np.array([2, 2, 1]), np.array([50, 10, 10]))
        config["round_func"] = rounding_dt
        config["lhs_dim"] = 3
        def generate_lhs_dt(dim, lb, ub, output_dir):
            space = Space([Integer(lb[i], ub[i]) for i in range(dim)])
            NN = 2 * (dim + 1)
            lhs_sampler = Lhs(criterion="maximin", iterations=10000)
            samples = np.array(lhs_sampler.generate(space.dimensions, NN, random_state=SEED))
            os.makedirs(output_dir, exist_ok=True)
            df = pd.DataFrame(samples, columns=[f"Param_{i+1}" for i in range(dim)])
            pool_path = os.path.join(output_dir, config["initial_pool_filename"])
            df.to_csv(pool_path, index=False)
            return samples
        config["generate_lhs"] = generate_lhs_dt
        config["process_lhs"] = lambda X: np.round(X).astype(int)
        config["evaluate_wrapper"] = lambda hp, X_tr, X_te, y_tr, y_te, ds: evaluate_dt(hp, X_tr, X_te, y_tr, y_te, ds)
    else:
        raise ValueError("Unknown model type")
    return config

# ========================
# UNIFIED BAYESIAN OPTIMIZATION CLASS
# ========================
class UnifiedBO:
    def __init__(self, budget, d, eval_func, X_init, Y_init, cand_size, batch_size,
                 lb, up, round_func, cand_gen, cand_sel,
                 X_train, X_test, y_train, y_test, dataset, total_evals):
        self.Budget = budget
        self.d = d
        self.eval_func = eval_func
        self.X = X_init.copy()
        self.Y = Y_init.copy()
        self.cand_size = cand_size
        self.batch_size = batch_size
        self.lb = lb
        self.up = up
        self.round_func = round_func
        self.cand_gen = cand_gen
        self.cand_sel = cand_sel
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset = dataset
        self.total_evals = total_evals
        self.iteration = 0

    def fit_surrogate(self):
        self.gp_models = gp_fit_botorch(self.X, self.Y)
        self.predict_func = lambda X_pred: gp_predict_botorch(self.gp_models, X_pred)

    def generate_candidates(self):
        if self.cand_gen == "dycors":
            best_f1_idx = np.argmin(self.Y[:, 0])
            best_f2_idx = np.argmin(self.Y[:, 1])
            best_f1 = self.X[best_f1_idx]
            best_f2 = self.X[best_f2_idx]
            cand1 = dycors(best_f1, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d)
            cand2 = dycors(best_f2, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d)
            candidates = np.vstack([cand1, cand2])
            candidates = np.clip(candidates, self.lb, self.up)
        elif self.cand_gen == "dycors_org":
            num_evals = self.X.shape[0]
            best_f1_idx = np.argmin(self.Y[:, 0])
            best_f2_idx = np.argmin(self.Y[:, 1])
            best_f1 = self.X[best_f1_idx]
            best_f2 = self.X[best_f2_idx]
            cand1 = dycors_org(best_f1, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d,
                               num_evals=num_evals, max_evals=self.total_evals)
            cand2 = dycors_org(best_f2, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d,
                               num_evals=num_evals, max_evals=self.total_evals)
            candidates = np.vstack([cand1, cand2])
            candidates = np.clip(candidates, self.lb, self.up)
        elif self.cand_gen == "dycors_fi1":
            best_f1_idx = np.argmin(self.Y[:, 0])
            best_f2_idx = np.argmin(self.Y[:, 1])
            best_f1 = self.X[best_f1_idx]
            best_f2 = self.X[best_f2_idx]
            cand1 = dycors_fi1(best_f1, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d, gp_models=self.gp_models)
            cand2 = dycors_fi1(best_f2, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d, gp_models=self.gp_models)
            candidates = np.vstack([cand1, cand2])
            candidates = np.clip(candidates, self.lb, self.up)
        elif self.cand_gen == "dycors_fi2":
            best_f1_idx = np.argmin(self.Y[:, 0])
            best_f2_idx = np.argmin(self.Y[:, 1])
            best_f1 = self.X[best_f1_idx]
            best_f2 = self.X[best_f2_idx]
            cand1 = dycors_fi2(best_f1, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d, gp_models=self.gp_models)
            cand2 = dycors_fi2(best_f2, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d, gp_models=self.gp_models)
            candidates = np.vstack([cand1, cand2])
            candidates = np.clip(candidates, self.lb, self.up)
        elif self.cand_gen == "dycors_fi_auc1":
            best_center = select_center_by_auc_reduction(self.Y, self.X)
            cand1 = dycors_fi2(best_center, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d, gp_models=self.gp_models)
            cand2 = dycors_fi2(best_center, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d, gp_models=self.gp_models)
            candidates = np.vstack([cand1, cand2])
            candidates = np.clip(candidates, self.lb, self.up)
        elif self.cand_gen == "dycors_fi_auc2":
            top_centers = select_top2_centers_by_auc_reduction(self.Y, self.X)
            cand1 = dycors_fi2(top_centers[0], self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d, gp_models=self.gp_models)
            cand2 = dycors_fi2(top_centers[1], self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d, gp_models=self.gp_models)
            candidates = np.vstack([cand1, cand2])
            candidates = np.clip(candidates, self.lb, self.up)
        elif self.cand_gen == "dycors_auc1":
            best_center = select_center_by_auc_reduction(self.Y, self.X)
            cand1 = dycors(best_center, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d)
            cand2 = dycors(best_center, self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d)
            candidates = np.vstack([cand1, cand2])
            candidates = np.clip(candidates, self.lb, self.up)
        elif self.cand_gen == "dycors_auc2":
                top_centers = select_top2_centers_by_auc_reduction(self.Y, self.X)
                cand1 = dycors(top_centers[0], self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d)
                cand2 = dycors(top_centers[1], self.cand_size, self.lb, self.up, scalefactor=0.3, d=self.d)
                candidates = np.vstack([cand1, cand2])
                candidates = np.clip(candidates, self.lb, self.up)
        elif self.cand_gen == "sobol":
            candidates = generate_candidates_sobol(self.lb, self.up, self.cand_size, self.d)
        else:
            raise ValueError("Unknown candidate generation method")
        return candidates

    def select_candidates(self, candidates):
        if self.cand_sel == "pareto":
            Y_pred = self.predict_func(candidates)
            selected = select_candidates_pareto(Y_pred, candidates, self.batch_size)
        elif self.cand_sel == "ehvi":
            selected = select_candidates_ehvi(self.gp_models, candidates, self.batch_size, self.Y)
        elif self.cand_sel == "parego":
            selected = select_candidates_qlognparego(
                self.gp_models,
                candidates,
                self.batch_size,
                self.Y,
                X_baseline=self.X
            )
        else:
            raise ValueError("Unknown candidate selection method")
        return selected

    def run(self):
        start_time = time.time()
        auc_history = []  # to store AUC history for each iteration
        while self.Budget > 0:
            self.iteration += 1

            # Fit the surrogate model so that self.gp_models is defined.
            self.fit_surrogate()
            
            # Calculate feature importance from GP models.
            fi_matrix = calculate_feature_importance_1(self.gp_models, decimals=3)
            fi_pareto = identify_pareto_front(fi_matrix)
            fi_pareto_size = len(fi_pareto)
            
            # Generate candidate points (the unevaluated set)
            candidates = self.generate_candidates()
            candidate_pool_size = candidates.shape[0]
            
            # Select a subset of candidates for evaluation
            selected = self.select_candidates(candidates)
            selected_size = selected.shape[0]
            uneval_size = candidate_pool_size - selected_size
            
            # (The evaluated set so far is self.X)
            evaluated_set_size = self.X.shape[0]

            # Build a structured log message
            log_message = (
                f"\nIteration {self.iteration} | Scenario: [model={self.model_type}, "
                f"cand_gen={self.cand_gen}, cand_sel={self.cand_sel}, batch_size={self.batch_size}]\n"
                f"  - Feature Importance Pareto set size: {fi_pareto_size}\n"
                f"  - Budget remaining: {self.Budget}\n"
                f"  - Evaluated set size (self.X): {evaluated_set_size}\n"
                f"  - Candidate pool size: {candidate_pool_size}\n"
                f"  - Selected points size: {selected_size}\n"
                f"  - Unevaluated set size: {uneval_size}\n"
            )
            print(log_message)
            
            # Round and evaluate the selected points.
            selected = self.round_func(selected)
            new_objs = []
            for hp in selected:
                obj = self.eval_func(hp, self.X_train, self.X_test, self.y_train, self.y_test, self.dataset)
                new_objs.append(obj)
            new_objs = np.array(new_objs)
            
            # Update the evaluated set.
            self.X = np.vstack([self.X, selected])
            self.Y = np.vstack([self.Y, new_objs])
            self.Budget -= selected.shape[0]

            # Compute the current Pareto frontier and its AUC.
            pareto_idx = identify_pareto_front(self.Y)
            pareto_Y = self.Y[pareto_idx]
            pareto_Y_clipped = np.clip(pareto_Y, 0, 1)
            pareto_Y_sorted = pareto_Y_clipped[pareto_Y_clipped[:, 0].argsort()]
            try: 
                current_auc = np.trapz(pareto_Y_sorted[:, 1], pareto_Y_sorted[:, 0])
            except Exception as e:
                current_auc = np.nan
            auc_history.append(current_auc)
            print(f"  -> New candidates evaluated: {selected_size}; Budget now: {self.Budget}; Current AUC: {current_auc:.4f}\n")
        duration = time.time() - start_time
        print(f"Optimization completed in {duration:.2f} seconds.")
        return self.X, self.Y, duration, auc_history


# ========================
# MAIN SCRIPT FUNCTION
# ========================
def run_BO_scenario(model_type, cand_gen, cand_sel, batch_size, cand_size, total_budget, output_dir):
    # DATASET
    dataset, X_train, X_test, y_train, y_test = load_compas_dataset()
    # MODEL CONFIGURATION
    config = get_model_config(model_type)
    lb, up = config["bounds"]
    d = config["lhs_dim"]
    # INITIAL HYPERPARAMETER SAMPLES
    pool_file = os.path.join(output_dir, config["initial_pool_filename"])
    if not os.path.exists(pool_file):
        print("Generating initial LHS samples...")
        samples_cont = config["generate_lhs"](d, lb, up, output_dir)
    else:
        print("Initial LHS samples already exist.")
    df_pool = pd.read_csv(pool_file)
    X_init_cont = df_pool.values
    X_init = config["process_lhs"](X_init_cont)
    # EVALUATE INITIAL SAMPLES
    Y_init = []
    for hp in X_init:
        obj = config["evaluate_wrapper"](hp, X_train, X_test, y_train, y_test, dataset)
        Y_init.append(obj)
    Y_init = np.array(Y_init)
    print("Initial samples evaluated.")
    remaining_budget = total_budget - X_init.shape[0]
    total_evals = X_init.shape[0] + remaining_budget
    bo = UnifiedBO(remaining_budget, d, config["evaluate_wrapper"], X_init, Y_init,
                   cand_size, batch_size, lb, up, config["round_func"],
                   cand_gen, cand_sel, X_train, X_test, y_train, y_test, dataset,
                   total_evals)
    bo.model_type = model_type
    bo.cand_gen = cand_gen
    bo.cand_sel = cand_sel
    bo.batch_size = batch_size
    final_X, final_Y, duration, auc_history = bo.run()
    pareto_idx = identify_pareto_front(final_Y)
    labels = np.zeros(final_Y.shape[0], dtype=int)
    labels[pareto_idx] = 1
    if model_type == "logistic":
        colnames = ["C", "solver", "max_iter"]
    elif model_type == "random_forest":
        colnames = ["n_estimators", "max_depth", "min_samples_split"]
    elif model_type == "random_forest2":
        colnames = ["max_depth", "min_samples_split", "max_leaf_nodes", "min_samples_leaf", "n_estimators", "max_samples", "max_features"]
    elif model_type == "decision_tree":
        colnames = ["max_depth", "min_samples_leaf", "min_samples_split"]
    else:
        colnames = [f"Param_{i+1}" for i in range(d)]
    df_points = pd.DataFrame(final_X, columns=colnames)
    df_points["error_rate"] = final_Y[:, 0]
    df_points["stat_parity"] = final_Y[:, 1]
    df_points["pareto"] = labels
    auc_df = pd.DataFrame({"iteration": np.arange(1, len(auc_history) + 1),
                           "auc": auc_history})
    return {
        "points_df": df_points,
        "auc_df": auc_df,
        "final_X": final_X,
        "final_Y": final_Y,
        "duration": duration,
        "auc_history": auc_history
    }

# Uncomment to run directly:
# if __name__ == "__main__":
#     res = run_BO_scenario("logistic", "dycors_fi_auc1", "pareto", batch_size=1, cand_size=1500,
#                           total_budget=150, output_dir="./fairpilot_results")
#     print("Results:", res)
