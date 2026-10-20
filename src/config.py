"""
Configuration module for SNTP Assignment 2:
Recommendation & Information Diffusion in a Social Network.
Student Roll Number: 23L-2550
"""

import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "epinions"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure output directories exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Data File Paths
RATINGS_FILE = DATA_DIR / "ratings_data.txt"
TRUST_FILE = DATA_DIR / "trust_data.txt"

# Random Seeds for Reproducibility
RANDOM_SEED = 42

# Data Subsampling / Core Extraction Parameters
# Epinions has 40,163 users and 139,738 items. For reproducible, memory-safe,
# and high-speed execution, we extract a dense induced core of active users
# who have both sufficient ratings and social trust connections.
CORE_CONFIG = {
    "min_ratings_per_user": 2,          # Include cold-start users (>= 2 ratings)
    "min_ratings_per_item": 2,
    "min_social_degree": 1,
    "max_core_users": 600,       # Selected active core for deep experimentation
    "demo_core_users": 150       # Fast demo core for quick verification (<30s)
}

# Recommendation Hyperparameters
RECOMMENDER_CONFIG = {
    "test_size": 0.20,           # 80/20 train/test split
    "relevance_threshold": 4.0,  # Rating >= 4 is considered a relevant (positive) signal
    "top_k": 10,                 # Top-K recommendations
    "user_knn_k": 20,            # K neighbors for User-KNN
    "item_knn_k": 20,            # K neighbors for Item-KNN
    "svd_factors": 20,           # Latent dimension for SVD
    "svd_lr": 0.005,             # Learning rate for SGD
    "svd_reg": 0.02,             # L2 Regularization
    "svd_epochs": 25,            # SGD epochs
    "social_svd_beta": 0.05,     # Social regularization strength
    "social_knn_betas": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] # Blending parameter beta
}

# Information Diffusion Hyperparameters
DIFFUSION_CONFIG = {
    "ic_propagation_prob": 0.12, # Base activation probability for IC
    "sir_beta": 0.18,            # Transmission rate (S -> I)
    "sir_gamma": 0.08,           # Recovery rate (I -> R)
    "max_steps": 25,             # Maximum simulation time steps
    "mc_simulations": 15,        # Monte Carlo trials for expected spread
    "demo_mc_simulations": 5     # Fast trials for demo mode
}

# Influence Maximization Parameters
INFLUENCE_CONFIG = {
    "budgets": [2, 5, 8, 12, 15, 20],
    "default_budget": 10,
    "demo_budget": 5
}

# Group Recommendation Parameters
GROUP_CONFIG = {
    "group_sizes": [3, 5, 8, 10],
    "default_group_size": 5,
    "num_groups_to_test": 5
}

# Intervention Settings
INTERVENTION_CONFIG = {
    "max_out_link_caps": [10, 5, 2, 1], # Degree capping interventions
    "prob_damping_factors": [1.0, 0.75, 0.5, 0.25] # Transmission damping
}
