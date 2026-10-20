"""
Experiment 1: Recommendations and Social Relationship Integration.
Addresses Requirements 1, 2, and 3:
1. Represent the social environment and inspect user-item & trust structures.
2. Evaluate two substantially different recommendation approaches (User-KNN vs. Item-KNN vs. SVD).
   Investigate cold-start user/item behavior and rating pattern differences.
3. Incorporate social relationships (Social-Constrained User-KNN across beta in [0, 1]).
   Isolate when social information improves recommendations vs. when it degrades them.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TABLES_DIR, PLOTS_DIR, CORE_CONFIG, RECOMMENDER_CONFIG, RANDOM_SEED
)
from src.data_loader import load_epinions_data
from src.recommender import (
    UserKNNRecommender, ItemKNNRecommender, SocialUserKNNRecommender, MatrixFactorizationRecommender
)
from src.evaluation import Evaluator

def run_recommendation_experiments(max_users: int = 300, seed: int = RANDOM_SEED):
    print("=" * 75)
    print(" [EXPERIMENT 1] Requirements 1, 2 & 3: Recommender Systems & Social Context")
    print("=" * 75)
    
    # 1. Load Data & Represent Social Environment (Requirement 1)
    env = load_epinions_data(max_users=max_users, seed=seed)
    evaluator = Evaluator(env, relevance_threshold=RECOMMENDER_CONFIG["relevance_threshold"])
    
    # 2. Train Models (Requirement 2)
    print("\n[*] Training Recommender Models from scratch...")
    user_cf = UserKNNRecommender(env, k_neighbors=RECOMMENDER_CONFIG["user_knn_k"])
    item_cf = ItemKNNRecommender(env, k_neighbors=RECOMMENDER_CONFIG["item_knn_k"])
    svd = MatrixFactorizationRecommender(
        env,
        n_factors=RECOMMENDER_CONFIG["svd_factors"],
        lr=RECOMMENDER_CONFIG["svd_lr"],
        reg=RECOMMENDER_CONFIG["svd_reg"],
        epochs=RECOMMENDER_CONFIG["svd_epochs"],
        social_beta=0.0,
        seed=seed
    )
    svd.fit(verbose=False)
    
    # Evaluate Base Models for Model Comparison Table
    base_models = {
        "User-KNN (Memory User-CF)": user_cf,
        "Item-KNN (Memory Item-CF)": item_cf,
        "SVD (Model-Based Latent MF)": svd
    }
    
    perf_records = []
    for name, model in base_models.items():
        print(f"    - Evaluating {name}...")
        pred_metrics = evaluator.evaluate_rating_prediction(model)
        topk_metrics = evaluator.evaluate_top_k(model, k=RECOMMENDER_CONFIG["top_k"])
        
        row = {
            "Model": name,
            "MAE": round(pred_metrics["MAE"], 4),
            "RMSE": round(pred_metrics["RMSE"], 4),
            "Classification_Acc": round(pred_metrics["Classification_Accuracy"], 4),
            "Precision@10": round(topk_metrics["Precision@10"], 4),
            "Recall@10": round(topk_metrics["Recall@10"], 4),
            "F1@10": round(topk_metrics["F1@10"], 4),
            "NDCG@10": round(topk_metrics["NDCG@10"], 4),
            "Spearman_Rho": round(topk_metrics["Spearman_Rho"], 4),
            "Catalog_Coverage": round(topk_metrics["Catalog_Coverage"], 4)
        }
        perf_records.append(row)
        
    df_perf = pd.DataFrame(perf_records)
    df_perf.to_csv(TABLES_DIR / "req2_model_comparison.csv", index=False)
    print("\n[+] Requirement 2: Model Performance Table Saved:")
    print(df_perf.to_string(index=False))

    # Evaluate Models across Cold-Start Tiers (including Social-CF to demonstrate social rescue of cold users)
    soc_user_cf = SocialUserKNNRecommender(env, beta=1.0, k_neighbors=RECOMMENDER_CONFIG["user_knn_k"])
    cold_eval_models = {
        "User-Based CF (no social, beta=0)": user_cf,
        "Item-Based CF": item_cf,
        "Social User-Based CF (beta=1.0)": soc_user_cf,
        "SVD (Model-Based Latent MF)": svd
    }
    cold_start_records = []
    for name, model in cold_eval_models.items():
        cold_metrics = evaluator.evaluate_cold_start_segments(model)
        for segment, seg_vals in cold_metrics.items():
            cold_start_records.append({
                "Model": name,
                "Segment": segment,
                "MAE": round(seg_vals["MAE"], 4),
                "RMSE": round(seg_vals["RMSE"], 4),
                "Test_Count": seg_vals["Count"]
            })
            
    df_cold = pd.DataFrame(cold_start_records)
    df_cold.to_csv(TABLES_DIR / "req2_cold_start_analysis.csv", index=False)
    
    # 3. Requirement 3: Incorporate Social Relationships across Beta
    print("\n[*] Requirement 3: Evaluating Social Relationships with parameter beta in [0, 1]...")
    betas = RECOMMENDER_CONFIG["social_knn_betas"]
    social_records = []
    
    for b in betas:
        soc_model = SocialUserKNNRecommender(env, beta=b, k_neighbors=RECOMMENDER_CONFIG["user_knn_k"])
        p_res = evaluator.evaluate_rating_prediction(soc_model)
        t_res = evaluator.evaluate_top_k(soc_model, k=10)
        social_records.append({
            "Beta (Social Weight)": b,
            "MAE": round(p_res["MAE"], 4),
            "RMSE": round(p_res["RMSE"], 4),
            "Precision@10": round(t_res["Precision@10"], 4),
            "Recall@10": round(t_res["Recall@10"], 4),
            "NDCG@10": round(t_res["NDCG@10"], 4)
        })
        print(f"    - Beta = {b:3.1f} | RMSE: {p_res['RMSE']:.4f} | Precision@10: {t_res['Precision@10']:.4f} | NDCG@10: {t_res['NDCG@10']:.4f}")
        
    df_social = pd.DataFrame(social_records)
    df_social.to_csv(TABLES_DIR / "req3_social_beta_sweep.csv", index=False)
    
    # Also evaluate Social SVD
    soc_svd = MatrixFactorizationRecommender(
        env,
        n_factors=RECOMMENDER_CONFIG["svd_factors"],
        lr=RECOMMENDER_CONFIG["svd_lr"],
        reg=RECOMMENDER_CONFIG["svd_reg"],
        epochs=RECOMMENDER_CONFIG["svd_epochs"],
        social_beta=RECOMMENDER_CONFIG["social_svd_beta"],
        seed=seed
    )
    soc_svd.fit()
    soc_svd_res = evaluator.evaluate_rating_prediction(soc_svd)
    print(f"    - Social Regularized SVD (beta={RECOMMENDER_CONFIG['social_svd_beta']}) | RMSE: {soc_svd_res['RMSE']:.4f}")
    
    # =========================================================================
    # PLOTS
    # =========================================================================
    print("\n[*] Generating Requirement 2 & 3 figures...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Model Comparison (RMSE & NDCG@10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=df_perf, x="Model", y="RMSE", hue="Model", ax=axes[0], palette="Blues_d", legend=False)
    axes[0].set_title("Rating Prediction Error (RMSE - Lower is Better)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis='x', rotation=15)
    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:
            axes[0].annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height / 2),
                             ha='center', va='center', color='white', fontweight='bold')

    sns.barplot(data=df_perf, x="Model", y="NDCG@10", hue="Model", ax=axes[1], palette="Greens_d", legend=False)
    axes[1].set_title("Ranking Quality (NDCG@10 - Higher is Better)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("NDCG@10")
    axes[1].tick_params(axis='x', rotation=15)
    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height / 2),
                             ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req2_model_performance.png", dpi=300)
    plt.close()

    # Plot 2: Cold-Start Behavior across User Activity Segments
    plt.figure(figsize=(10, 5.5))
    ax_cold = sns.barplot(data=df_cold, x="Segment", y="RMSE", hue="Model", palette="tab10", edgecolor="black", linewidth=0.8)
    plt.title("Cold-Start Behavior: Prediction Error Across User History Density", fontsize=13, fontweight='bold')
    plt.ylabel("Test RMSE (Lower is Better)", fontsize=11, fontweight='bold')
    plt.xlabel("User Interaction History Segment", fontsize=11, fontweight='bold')
    plt.legend(title="Model", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req2_cold_start_analysis.png", dpi=300)
    plt.close()

    # Plot 3: Social Relationship Contribution (Beta Sweep)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel("Social Weight Parameter Beta (0 = Pure Rating Sim, 1 = Pure Friends Trust)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("RMSE (Lower is Better)", color=color, fontsize=11, fontweight='bold')
    line1 = ax1.plot(df_social["Beta (Social Weight)"], df_social["RMSE"], color=color, marker='o', linewidth=2.5, label="RMSE")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel("NDCG@10 (Higher is Better)", color=color, fontsize=11, fontweight='bold')
    line2 = ax2.plot(df_social["Beta (Social Weight)"], df_social["NDCG@10"], color=color, marker='s', linewidth=2.5, linestyle='--', label="NDCG@10")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Requirement 3: Effect of Social Information Weight (Beta)", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req3_social_beta_effect.png", dpi=300)
    plt.close()

    print("[+] All Experiment 1 tables and plots generated successfully in results/")
    return df_perf, df_social

if __name__ == "__main__":
    run_recommendation_experiments()
