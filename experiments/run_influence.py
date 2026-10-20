"""
Experiment 3: Influence Maximization under Limited Promotion Budget.
Addresses Requirement 5:
- Platform can initially promote an item to only a limited budget k of users.
- Implements Greedy Seed Selection (SMM-ch7 Slides 22-29) maximizing marginal expected spread.
- Compares against Random, Degree Centrality, and PageRank under identical budgets k in [2, 5, 8, 12, 15, 20].
- Explicitly investigates and quantifies overlapping audiences.
- Accounts for uncertainty with Monte Carlo confidence intervals.
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
    TABLES_DIR, PLOTS_DIR, CORE_CONFIG, INFLUENCE_CONFIG, DIFFUSION_CONFIG, RANDOM_SEED
)
from src.data_loader import load_epinions_data
from src.diffusion import DiffusionSimulator
from src.influence import InfluenceMaximizer

def run_influence_experiments(max_users: int = 300, seed: int = RANDOM_SEED):
    print("=" * 75)
    print(" [EXPERIMENT 3] Requirement 5: Influence Maximization & Seed Selection")
    print("=" * 75)
    
    # 1. Load Data
    env = load_epinions_data(max_users=max_users, seed=seed)
    inf_max = InfluenceMaximizer(
        env,
        base_p=DIFFUSION_CONFIG["ic_propagation_prob"],
        mc_trials=DIFFUSION_CONFIG["mc_simulations"]
    )
    simulator = DiffusionSimulator(env)
    
    budgets = INFLUENCE_CONFIG["budgets"]
    # Candidate pool: top 80 active nodes by degree for fast responsive greedy hill climbing
    top_candidates = sorted(range(env.num_users), key=lambda u: len(env.trust_out.get(u, set())), reverse=True)[:80]
    
    # Pre-select seeds for each method at max budget
    max_k = max(budgets)
    print(f"\n[*] Executing Seed Selection for Budgets {budgets} (Max Budget k = {max_k})...")
    
    # 1. Greedy / CELF (submodular hill-climbing)
    greedy_seeds_full, spread_hist, g_time, g_evals = inf_max.select_celf(k=max_k, candidate_pool=top_candidates, seed=seed)
    greedy_seed_list = list(greedy_seeds_full)
    
    # 2. Degree Centrality
    degree_seeds_full = inf_max.select_degree(k=max_k)
    degree_seed_list = list(degree_seeds_full)
    
    # 3. PageRank
    pagerank_seeds_full = inf_max.select_pagerank(k=max_k)
    pagerank_seed_list = list(pagerank_seeds_full)
    
    # 4. Random Selection
    random_seeds_full = inf_max.select_random(k=max_k, seed=seed)
    random_seed_list = list(random_seeds_full)
    
    # Now evaluate all methods across budgets using independent Monte Carlo simulation
    influence_results = []
    
    for k in budgets:
        print(f"    - Testing budget k = {k}...")
        sub_methods = {
            "Greedy (Marginal Gain)": set(greedy_seed_list[:k]),
            "Degree Centrality": set(degree_seed_list[:k]),
            "PageRank Centrality": set(pagerank_seed_list[:k]),
            "Random Selection": set(random_seed_list[:k])
        }
        
        for name, seeds in sub_methods.items():
            mc_eval = simulator.run_monte_carlo_ic(
                seed_set=seeds,
                num_trials=DIFFUSION_CONFIG["mc_simulations"],
                base_p=DIFFUSION_CONFIG["ic_propagation_prob"],
                seed=seed + k * 13
            )
            overlap = inf_max.compute_audience_overlap(seeds)
            
            influence_results.append({
                "Budget_k": k,
                "Strategy": name,
                "Expected_Spread": round(mc_eval["mean_spread"], 2),
                "Std_Spread": round(mc_eval["std_spread"], 2),
                "Min_Spread": mc_eval["min_spread"],
                "Max_Spread": mc_eval["max_spread"],
                "Audience_Overlap": round(overlap, 4),
                "Expected_Exposure": round(mc_eval["mean_exposure"], 2)
            })
            
    df_inf = pd.DataFrame(influence_results)
    df_inf.to_csv(TABLES_DIR / "req5_influence_maximization_results.csv", index=False)
    
    print("\n[+] Influence Maximization Summary Table:")
    pivot_summary = df_inf.pivot(index="Budget_k", columns="Strategy", values="Expected_Spread")
    print(pivot_summary.to_string())
    
    # =========================================================================
    # PLOTS
    # =========================================================================
    print("\n[*] Generating Requirement 5 figures...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Spread vs Budget with Confidence Intervals
    plt.figure(figsize=(9, 5.5))
    strategies = df_inf["Strategy"].unique()
    colors = {"Greedy (Marginal Gain)": "crimson", "Degree Centrality": "darkorange",
              "PageRank Centrality": "forestgreen", "Random Selection": "gray"}
    markers = {"Greedy (Marginal Gain)": "o", "Degree Centrality": "s",
               "PageRank Centrality": "^", "Random Selection": "x"}
    
    for strat in strategies:
        sub = df_inf[df_inf["Strategy"] == strat]
        plt.errorbar(sub["Budget_k"], sub["Expected_Spread"], yerr=sub["Std_Spread"],
                     fmt=f'-{markers[strat]}', color=colors[strat], linewidth=2.2, capsize=4, label=strat)
        
    plt.title("Requirement 5: Cascade Spread vs Promotion Budget k", fontsize=12, fontweight='bold')
    plt.xlabel("Promotion Budget k (Initial Seeds)", fontsize=11, fontweight='bold')
    plt.ylabel("Total Expected Adoption Spread (Nodes)", fontsize=11, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req5_spread_vs_budget.png", dpi=300)
    plt.close()
    
    # Plot 2: Audience Overlap across Strategies
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_inf, x="Budget_k", y="Audience_Overlap", hue="Strategy", palette=colors)
    plt.title("Requirement 5: Audience Redundancy (Average Pairwise Jaccard Overlap)", fontsize=12, fontweight='bold')
    plt.xlabel("Promotion Budget k", fontsize=11, fontweight='bold')
    plt.ylabel("Audience Overlap (Jaccard Index)", fontsize=11, fontweight='bold')
    plt.legend(title="Selection Strategy")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req5_audience_overlap.png", dpi=300)
    plt.close()
    
    print("[+] Experiment 3 tables and plots saved successfully in results/")
    return df_inf

if __name__ == "__main__":
    run_influence_experiments()
