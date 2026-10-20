"""
Experiment 2: Adoption, Diffusion, and Group Recommendation Dynamics.
Addresses Requirements 4 & 6:
4. Model adoption and information spread using Independent Cascade (IC) and SIR Epidemic models.
   Distinguish receiving recommendation (seed), exposed through neighbor, and actually adopting.
   Investigate effect of influence strength, network structure, and user preferences on cascade size/speed.
6. Group recommendations: Compare Average Satisfaction, Least Misery, and Most Pleasure.
   Analyze whether the item preferred by group strategy also spreads most widely when seeded by the group.
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
    TABLES_DIR, PLOTS_DIR, CORE_CONFIG, DIFFUSION_CONFIG, GROUP_CONFIG, RANDOM_SEED
)
from src.data_loader import load_epinions_data
from src.recommender import UserKNNRecommender
from src.diffusion import DiffusionSimulator
from src.group_rec import GroupRecommender

def run_diffusion_experiments(max_users: int = 300, seed: int = RANDOM_SEED):
    print("=" * 75)
    print(" [EXPERIMENT 2] Requirements 4 & 6: Information Diffusion & Group Dynamics")
    print("=" * 75)
    
    # 1. Load Data
    env = load_epinions_data(max_users=max_users, seed=seed)
    simulator = DiffusionSimulator(env)
    recommender = UserKNNRecommender(env, k_neighbors=20)
    
    # 2. Requirement 4: IC vs. SIR Model Comparison
    print("\n[*] Requirement 4: Comparing Independent Cascade (IC) vs. SIR Epidemic Models...")
    # Pick a high-influence seed set (e.g., top 5 out-degree nodes)
    sorted_nodes = sorted(range(env.num_users), key=lambda u: len(env.trust_out.get(u, set())), reverse=True)
    seed_set = set(sorted_nodes[:5])
    
    # Run repeated IC simulations
    ic_runs = []
    sir_runs = []
    
    rng = np.random.default_rng(seed)
    ic_res = simulator.simulate_ic(seed_set=seed_set, base_p=DIFFUSION_CONFIG["ic_propagation_prob"], rng=rng)
    sir_res = simulator.simulate_sir(seed_set=seed_set, beta=DIFFUSION_CONFIG["sir_beta"], gamma=DIFFUSION_CONFIG["sir_gamma"], rng=rng)
    
    print(f"    - IC Model: Final Adopted = {ic_res['final_adopted']} ({ic_res['adoption_fraction']*100:.1f}%), Total Exposed = {ic_res['total_exposed']}")
    print(f"    - SIR Model: Total Ever Infected = {sir_res['final_ever_infected']} ({sir_res['adoption_fraction']*100:.1f}%), Peak Infected = {sir_res['peak_infected']}")
    
    # Sweep Influence Probabilities (p for IC, beta for SIR)
    print("\n[*] Evaluating diffusion sensitivity across varying influence strengths...")
    probs = [0.04, 0.08, 0.12, 0.16, 0.20, 0.25]
    prob_records = []
    
    for p in probs:
        # 15 Monte Carlo trials per probability
        ic_stats = simulator.run_monte_carlo_ic(seed_set=seed_set, num_trials=DIFFUSION_CONFIG["mc_simulations"], base_p=p, seed=seed)
        sir_spreads = []
        for trial in range(DIFFUSION_CONFIG["mc_simulations"]):
            r = simulator.simulate_sir(seed_set=seed_set, beta=p * 1.2, gamma=0.08, rng=np.random.default_rng(seed + trial * 7))
            sir_spreads.append(r["final_ever_infected"])
            
        prob_records.append({
            "Influence_Prob": p,
            "IC_Mean_Spread": round(ic_stats["mean_spread"], 2),
            "IC_Std_Spread": round(ic_stats["std_spread"], 2),
            "IC_Mean_Exposure": round(ic_stats["mean_exposure"], 2),
            "SIR_Mean_Spread": round(float(np.mean(sir_spreads)), 2),
            "SIR_Std_Spread": round(float(np.std(sir_spreads)), 2)
        })
        
    df_prob = pd.DataFrame(prob_records)
    df_prob.to_csv(TABLES_DIR / "req4_diffusion_probabilities.csv", index=False)
    print(df_prob.to_string(index=False))
    
    # 3. Requirement 6: Group Recommendation & Group-Seeded Diffusion
    print("\n[*] Requirement 6: Group Recommendation and Group Diffusion Evaluation...")
    group_rec = GroupRecommender(env, recommender)
    
    # Test groups of varying sizes and conflict levels
    group_results_list = []
    group_sizes = GROUP_CONFIG["group_sizes"]
    
    for g_size in group_sizes:
        # Form conflicting group (diverse tastes)
        group = group_rec.form_group(size=g_size, group_type="conflicting", seed=seed + g_size)
        res = group_rec.evaluate_group_diffusion(
            group=group,
            mc_trials=DIFFUSION_CONFIG["mc_simulations"],
            base_p=DIFFUSION_CONFIG["ic_propagation_prob"],
            seed=seed
        )
        
        for strategy, metrics in res.items():
            group_results_list.append({
                "Group_Size": g_size,
                "Group_Type": "Conflicting",
                "Strategy": strategy,
                "Item_ID": metrics["best_item"],
                "Strategy_Score": round(metrics["strategy_score"], 3),
                "Group_Avg_Satisfaction": round(metrics["group_avg_sat"], 3),
                "Group_Min_Satisfaction": round(metrics["group_min_sat"], 3),
                "Group_Variance": round(metrics["group_variance_sat"], 4),
                "Expected_Spread": round(metrics["expected_spread"], 2),
                "Std_Spread": round(metrics["std_spread"], 2),
                "Expected_Exposure": round(metrics["expected_exposure"], 2)
            })
            
    df_group = pd.DataFrame(group_results_list)
    df_group.to_csv(TABLES_DIR / "req6_group_recommendation_diffusion.csv", index=False)
    print("\n[+] Group Strategy vs Spread Summary (Group Size = 5):")
    print(df_group[df_group["Group_Size"] == 5][["Strategy", "Group_Avg_Satisfaction", "Group_Min_Satisfaction", "Expected_Spread", "Std_Spread"]].to_string(index=False))
    
    # =========================================================================
    # PLOTS
    # =========================================================================
    print("\n[*] Generating Requirement 4 & 6 figures...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: IC Cascade vs SIR Epidemic Dynamics over Time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # IC trajectory
    axes[0].plot(ic_res["adoption_history"], marker='o', color='crimson', linewidth=2.5, label="Cumulative Adopted")
    axes[0].bar(range(len(ic_res["newly_active_history"])), ic_res["newly_active_history"], alpha=0.4, color='salmon', label="Newly Active per Step")
    axes[0].set_title("Independent Cascade (IC) Progression", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Time Step (t)")
    axes[0].set_ylabel("Number of Users")
    axes[0].legend()
    
    # SIR trajectories
    axes[1].plot(sir_res["s_history"], color='steelblue', linestyle='--', linewidth=2, label="Susceptible (S)")
    axes[1].plot(sir_res["i_history"], color='crimson', marker='^', linewidth=2.5, label="Infected / Active (I)")
    axes[1].plot(sir_res["r_history"], color='forestgreen', marker='s', linewidth=2, label="Recovered (R)")
    axes[1].set_title("SIR Epidemic Model Compartments", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Time Step (t)")
    axes[1].set_ylabel("Number of Users")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req4_diffusion_model_dynamics.png", dpi=300)
    plt.close()
    
    # Plot 2: Spread as a function of Influence Strength (IC vs SIR)
    plt.figure(figsize=(8, 5))
    plt.errorbar(df_prob["Influence_Prob"], df_prob["IC_Mean_Spread"], yerr=df_prob["IC_Std_Spread"],
                 fmt='-o', color='crimson', linewidth=2, capsize=4, label="Independent Cascade (IC)")
    plt.errorbar(df_prob["Influence_Prob"], df_prob["SIR_Mean_Spread"], yerr=df_prob["SIR_Std_Spread"],
                 fmt='--s', color='navy', linewidth=2, capsize=4, label="SIR Epidemic")
    plt.title("Requirement 4: Expected Spread vs Influence Probability", fontsize=12, fontweight='bold')
    plt.xlabel("Transmission Probability / Influence Strength")
    plt.ylabel("Total Expected Adoption Spread (Nodes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req4_spread_vs_probability.png", dpi=300)
    plt.close()
    
    # Plot 3: Group Recommendation - Satisfaction vs Widespread Adoption
    df_g5 = df_group[df_group["Group_Size"] == 5]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_g5))
    width = 0.35
    
    ax1.bar(x - width/2, df_g5["Group_Min_Satisfaction"], width, label="Least Misery (Min Sat)", color='lightcoral')
    ax1.bar(x - width/2, df_g5["Group_Avg_Satisfaction"] - df_g5["Group_Min_Satisfaction"], width,
            bottom=df_g5["Group_Min_Satisfaction"], label="Average Satisfaction", color='teal', alpha=0.7)
    ax1.set_ylabel("Group Satisfaction Score", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 5.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_g5["Strategy"], fontsize=11, fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.errorbar(x + width/2, df_g5["Expected_Spread"], yerr=df_g5["Std_Spread"],
                 fmt='o', color='purple', linewidth=2.5, markersize=8, capsize=5, label="Expected Network Spread")
    ax2.set_ylabel("Expected Network Adoption Spread", color='purple', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title("Requirement 6: Group Satisfaction vs Network Spread Trade-off", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req6_group_satisfaction_vs_spread.png", dpi=300)
    plt.close()
    
    print("[+] Experiment 2 tables and plots saved successfully in results/")
    return df_prob, df_group

if __name__ == "__main__":
    run_diffusion_experiments()
