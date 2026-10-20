"""
Experiment 4: Intervention Strategies & Computational Cost Optimization.
Addresses Requirement 8:
- Introduce interventions limiting diffusion: Degree Capping (out-link limits) and Transmission Damping.
- Examine joint effects on cascade spread and user satisfaction.
- Identify the computational bottleneck: Repeated Monte Carlo evaluations in Greedy Influence Maximization.
- Implement and benchmark CELF (Cost-Effective Lazy Forward) submodular optimization against Standard Greedy.
"""

import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TABLES_DIR, PLOTS_DIR, CORE_CONFIG, DIFFUSION_CONFIG, INTERVENTION_CONFIG, RANDOM_SEED
)
from src.data_loader import load_epinions_data, SocialEnvironment
from src.recommender import UserKNNRecommender
from src.diffusion import DiffusionSimulator
from src.influence import InfluenceMaximizer

def run_intervention_experiments(max_users: int = 250, seed: int = RANDOM_SEED):
    print("=" * 75)
    print(" [EXPERIMENT 4] Requirement 8: Diffusion Intervention & Computational Cost")
    print("=" * 75)
    
    # 1. Load Data & Base Models
    env = load_epinions_data(max_users=max_users, seed=seed)
    simulator = DiffusionSimulator(env)
    recommender = UserKNNRecommender(env, k_neighbors=15)
    
    # Select high-influence seed set
    sorted_nodes = sorted(range(env.num_users), key=lambda u: len(env.trust_out.get(u, set())), reverse=True)
    seed_set = set(sorted_nodes[:5])
    test_item = 10 # representative item
    
    # =========================================================================
    # Part 1: Diffusion Interventions (Degree Capping & Damping)
    # =========================================================================
    print("\n[*] Evaluating Diffusion Interventions: Out-Link Degree Capping & Damping...")
    degree_caps = [None, 10, 5, 2, 1] # None = Uncapped
    damping_factors = [1.0, 0.75, 0.50, 0.25]
    
    intervention_records = []
    
    for cap in degree_caps:
        # Create a capped copy of trust edges
        original_trust_out = env.trust_out
        if cap is not None:
            capped_trust = {}
            for u, nbrs in original_trust_out.items():
                capped_trust[u] = set(list(nbrs)[:cap])
            env.trust_out = capped_trust
            cap_label = f"Cap={cap}"
        else:
            cap_label = "Uncapped"
            
        for damp in damping_factors:
            base_p = DIFFUSION_CONFIG["ic_propagation_prob"] * damp
            
            # Simulate 10 MC trials
            spreads = []
            satisfactions = []
            
            for trial in range(10):
                rng = np.random.default_rng(seed + trial * 37)
                res = simulator.simulate_ic(
                    seed_set=seed_set,
                    item_id=test_item,
                    recommender=recommender,
                    base_p=base_p,
                    pref_weighted=True,
                    rng=rng
                )
                spreads.append(res["final_adopted"])
                
                # Compute average satisfaction of adopters
                adopted = list(res["active_node_set"])
                if adopted:
                    sat = np.mean([recommender.predict(u, test_item) for u in adopted])
                else:
                    sat = 3.5
                satisfactions.append(sat)
                
            intervention_records.append({
                "Out_Degree_Cap": cap_label,
                "Damping_Factor": damp,
                "Effective_P": round(base_p, 4),
                "Mean_Spread": round(float(np.mean(spreads)), 2),
                "Std_Spread": round(float(np.std(spreads)), 2),
                "Mean_Adopter_Satisfaction": round(float(np.mean(satisfactions)), 3)
            })
            
        # Restore original trust graph
        env.trust_out = original_trust_out
        
    df_inter = pd.DataFrame(intervention_records)
    df_inter.to_csv(TABLES_DIR / "req8_diffusion_intervention_results.csv", index=False)
    print("\n[+] Intervention Results Summary:")
    print(df_inter[df_inter["Damping_Factor"] == 1.0][["Out_Degree_Cap", "Mean_Spread", "Mean_Adopter_Satisfaction"]].to_string(index=False))
    
    # =========================================================================
    # Part 2: Computational Bottleneck & CELF Optimization Benchmark
    # =========================================================================
    print("\n[*] Benchmarking Computational Bottleneck: Standard Greedy vs CELF Optimization...")
    inf_max = InfluenceMaximizer(env, base_p=DIFFUSION_CONFIG["ic_propagation_prob"], mc_trials=5)
    
    candidate_subset = sorted_nodes[:35] # Controlled candidate set for rigorous runtime benchmark
    test_budgets = [2, 4, 6, 8, 10]
    
    benchmark_records = []
    
    for b in test_budgets:
        print(f"    - Benchmarking budget k = {b}...")
        # Standard Greedy
        t0 = time.time()
        s_greedy, hist_g, time_greedy = inf_max.select_greedy(k=b, candidate_pool=candidate_subset, seed=seed)
        greedy_evals = b * len(candidate_subset) # Number of MC calls in standard greedy
        
        # CELF (Lazy Forward)
        t1 = time.time()
        s_celf, hist_c, time_celf, celf_evals = inf_max.select_celf(k=b, candidate_pool=candidate_subset, seed=seed)
        
        # Verify solution quality
        spread_g = hist_g[-1] if hist_g else 0.0
        spread_c = hist_c[-1] if hist_c else 0.0
        speedup = time_greedy / max(1e-5, time_celf)
        eval_reduction = ((greedy_evals - celf_evals) / greedy_evals) * 100.0 if greedy_evals > 0 else 0.0
        
        benchmark_records.append({
            "Budget_k": b,
            "Greedy_Time_sec": round(time_greedy, 4),
            "CELF_Time_sec": round(time_celf, 4),
            "Speedup_Ratio": round(speedup, 2),
            "Greedy_Evaluations": greedy_evals,
            "CELF_Evaluations": celf_evals,
            "Evaluation_Reduction_Pct": round(eval_reduction, 1),
            "Greedy_Spread": round(spread_g, 2),
            "CELF_Spread": round(spread_c, 2)
        })
        print(f"      Greedy: {time_greedy:.2f}s ({greedy_evals} evals) | CELF: {time_celf:.2f}s ({celf_evals} evals) | Speedup: {speedup:.1f}x")
        
    df_bench = pd.DataFrame(benchmark_records)
    df_bench.to_csv(TABLES_DIR / "req8_celf_optimization_benchmark.csv", index=False)
    print("\n[+] Optimization Benchmark Table:")
    print(df_bench[["Budget_k", "Greedy_Time_sec", "CELF_Time_sec", "Speedup_Ratio", "Greedy_Evaluations", "CELF_Evaluations"]].to_string(index=False))
    
    # =========================================================================
    # PLOTS
    # =========================================================================
    print("\n[*] Generating Requirement 8 figures...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Intervention Trade-off: Spread vs Satisfaction
    plt.figure(figsize=(8, 5))
    df_uncapped = df_inter[df_inter["Damping_Factor"] == 1.0]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'crimson'
    ax1.set_xlabel("Degree Capping Level (Maximum Out-Links Allowed)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Expected Network Adoption Spread", color=color, fontsize=11, fontweight='bold')
    ax1.plot(df_uncapped["Out_Degree_Cap"], df_uncapped["Mean_Spread"], color=color, marker='o', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'teal'
    ax2.set_ylabel("Mean Adopter Satisfaction (Predicted Rating)", color=color, fontsize=11, fontweight='bold')
    ax2.plot(df_uncapped["Out_Degree_Cap"], df_uncapped["Mean_Adopter_Satisfaction"], color=color, marker='s', linewidth=2.5, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Requirement 8: Impact of Degree Capping on Spread vs Satisfaction", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req8_intervention_spread_vs_satisfaction.png", dpi=300)
    plt.close()
    
    # Plot 2: CELF vs Greedy Runtime & Function Evaluation Speedup
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(df_bench["Budget_k"], df_bench["Greedy_Time_sec"], marker='o', color='crimson', linewidth=2, label="Standard Greedy")
    axes[0].plot(df_bench["Budget_k"], df_bench["CELF_Time_sec"], marker='s', color='navy', linewidth=2, label="CELF (Submodular Lazy Forward)")
    axes[0].set_title("Runtime Comparison (Lower is Faster)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Budget k", fontsize=11)
    axes[0].set_ylabel("Runtime (seconds)", fontsize=11)
    axes[0].legend()
    
    axes[1].bar(df_bench["Budget_k"] - 0.2, df_bench["Greedy_Evaluations"], width=0.4, label="Standard Greedy", color='lightcoral')
    axes[1].bar(df_bench["Budget_k"] + 0.2, df_bench["CELF_Evaluations"], width=0.4, label="CELF (Lazy Evaluations)", color='royalblue')
    axes[1].set_title("Monte Carlo Function Evaluations Needed", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Budget k", fontsize=11)
    axes[1].set_ylabel("Total Evaluations", fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "req8_celf_vs_greedy_runtime.png", dpi=300)
    plt.close()
    
    print("[+] Experiment 4 tables and plots saved successfully in results/")
    return df_inter, df_bench

if __name__ == "__main__":
    run_intervention_experiments()
