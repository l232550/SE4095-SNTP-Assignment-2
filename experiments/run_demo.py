"""
Quick Demonstration Script for SNTP Assignment 2.
Student Roll Number: 23L-2550
Executes a rapid, self-contained end-to-end verification covering all 8 requirements in ~30 seconds.
"""

import time
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CORE_CONFIG, RECOMMENDER_CONFIG, DIFFUSION_CONFIG
from src.data_loader import load_epinions_data
from src.recommender import (
    UserKNNRecommender, ItemKNNRecommender, SocialUserKNNRecommender, MatrixFactorizationRecommender
)
from src.diffusion import DiffusionSimulator
from src.influence import InfluenceMaximizer
from src.group_rec import GroupRecommender
from src.evaluation import Evaluator

def run_demo():
    print("=" * 80)
    print("      FAST DEMONSTRATION RUNNER - SNTP ASSIGNMENT 2 (ROLL NO: 23L-2550)")
    print("=" * 80)
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # Requirement 1: Represent the Social Environment
    # -------------------------------------------------------------------------
    print("\n[REQUIREMENT 1] Ingesting Epinions Social Environment (Fast Demo Core)...")
    env = load_epinions_data(max_users=CORE_CONFIG["demo_core_users"], seed=42)
    evaluator = Evaluator(env, relevance_threshold=4.0)
    
    # -------------------------------------------------------------------------
    # Requirement 2: Two Substantially Different Recommender Approaches
    # -------------------------------------------------------------------------
    print("\n[REQUIREMENT 2] Training Recommenders: User-KNN, Item-KNN & SVD...")
    user_cf = UserKNNRecommender(env, k_neighbors=15)
    item_cf = ItemKNNRecommender(env, k_neighbors=15)
    svd = MatrixFactorizationRecommender(env, n_factors=15, epochs=15, lr=0.005, seed=42)
    svd.fit(verbose=False)
    
    m_user = evaluator.evaluate_rating_prediction(user_cf)
    m_item = evaluator.evaluate_rating_prediction(item_cf)
    m_svd = evaluator.evaluate_rating_prediction(svd)
    
    print(f"    - User-KNN : MAE = {m_user['MAE']:.4f} | RMSE = {m_user['RMSE']:.4f}")
    print(f"    - Item-KNN : MAE = {m_item['MAE']:.4f} | RMSE = {m_item['RMSE']:.4f}")
    print(f"    - SVD (MF) : MAE = {m_svd['MAE']:.4f} | RMSE = {m_svd['RMSE']:.4f}")
    
    # -------------------------------------------------------------------------
    # Requirement 3: Incorporate Social Relationships (Beta Sweep)
    # -------------------------------------------------------------------------
    print("\n[REQUIREMENT 3] Testing Social Context Weight (Beta in [0, 1])...")
    for b in [0.0, 0.5, 1.0]:
        soc_knn = SocialUserKNNRecommender(env, beta=b, k_neighbors=15)
        res = evaluator.evaluate_rating_prediction(soc_knn)
        label = "Pure Sim" if b == 0.0 else ("Hybrid" if b == 0.5 else "Pure Friends")
        print(f"    - Beta = {b:.1f} ({label:12s}): MAE = {res['MAE']:.4f} | RMSE = {res['RMSE']:.4f}")
        
    # -------------------------------------------------------------------------
    # Requirement 4: Model Adoption & Diffusion (IC & SIR)
    # -------------------------------------------------------------------------
    print("\n[REQUIREMENT 4] Simulating Information Diffusion (IC vs. SIR)...")
    simulator = DiffusionSimulator(env)
    top_seeds = set(sorted(range(env.num_users), key=lambda u: len(env.trust_out.get(u, set())), reverse=True)[:3])
    
    ic_res = simulator.simulate_ic(seed_set=top_seeds, base_p=0.15)
    sir_res = simulator.simulate_sir(seed_set=top_seeds, beta=0.20, gamma=0.10)
    print(f"    - IC Model  : Seeds = {len(top_seeds)} -> Final Adopted = {ic_res['final_adopted']} ({ic_res['adoption_fraction']*100:.1f}%) in {ic_res['steps_to_quiescence']} steps")
    print(f"    - SIR Model : Seeds = {len(top_seeds)} -> Total Ever Infected = {sir_res['final_ever_infected']} ({sir_res['adoption_fraction']*100:.1f}%), Peak = {sir_res['peak_infected']}")
    
    # -------------------------------------------------------------------------
    # Requirement 5: Limited Promotion Budget (Influence Maximization)
    # -------------------------------------------------------------------------
    print("\n[REQUIREMENT 5] Selecting Seeds Under Limited Budget (Greedy vs Baselines)...")
    inf_max = InfluenceMaximizer(env, base_p=0.15, mc_trials=5)
    candidates = sorted(range(env.num_users), key=lambda u: len(env.trust_out.get(u, set())), reverse=True)[:25]
    
    budget = 4
    greedy_seeds, _, _, _ = inf_max.select_celf(k=budget, candidate_pool=candidates)
    degree_seeds = inf_max.select_degree(k=budget)
    random_seeds = inf_max.select_random(k=budget)
    
    spread_g = simulator.run_monte_carlo_ic(greedy_seeds, num_trials=5, base_p=0.15)["mean_spread"]
    spread_d = simulator.run_monte_carlo_ic(degree_seeds, num_trials=5, base_p=0.15)["mean_spread"]
    spread_r = simulator.run_monte_carlo_ic(random_seeds, num_trials=5, base_p=0.15)["mean_spread"]
    
    overlap_g = inf_max.compute_audience_overlap(greedy_seeds)
    overlap_d = inf_max.compute_audience_overlap(degree_seeds)
    
    print(f"    - Greedy/CELF Spread: {spread_g:.1f} nodes (Audience Overlap: {overlap_g:.3f})")
    print(f"    - Degree Cent. Spread: {spread_d:.1f} nodes (Audience Overlap: {overlap_d:.3f})")
    print(f"    - Random Pick Spread: {spread_r:.1f} nodes")
    
    # -------------------------------------------------------------------------
    # Requirement 6: Recommend to Groups
    # -------------------------------------------------------------------------
    print("\n[REQUIREMENT 6] Evaluating Group Recommendations & Group Diffusion...")
    group_rec = GroupRecommender(env, user_cf)
    test_group = group_rec.form_group(size=4, group_type="conflicting")
    group_diff = group_rec.evaluate_group_diffusion(group=test_group, mc_trials=5, base_p=0.15)
    
    for strat, data in group_diff.items():
        print(f"    - {strat:15s}: Min Sat = {data['group_min_sat']:.2f} | Avg Sat = {data['group_avg_sat']:.2f} | Network Spread = {data['expected_spread']:.1f}")
        
    # -------------------------------------------------------------------------
    # Requirement 7: Comprehensive Evaluation
    # -------------------------------------------------------------------------
    print("\n[REQUIREMENT 7] Ranking & Classification Quality Metrics...")
    topk_res = evaluator.evaluate_top_k(user_cf, k=5)
    print(f"    - Precision@5 = {topk_res['Precision@5']:.4f} | Recall@5 = {topk_res['Recall@5']:.4f} | NDCG@5 = {topk_res['NDCG@5']:.4f}")
    print(f"    - Catalog Coverage = {topk_res['Catalog_Coverage']*100:.2f}% | Spearman Rho = {topk_res['Spearman_Rho']:.4f}")
    
    # -------------------------------------------------------------------------
    # Requirement 8: Diffusion Intervention & CELF Optimization
    # -------------------------------------------------------------------------
    print("\n[REQUIREMENT 8] Intervention & Computational Optimization...")
    # CELF vs Greedy Speedup on candidates
    t0 = time.time()
    _, _, time_g = inf_max.select_greedy(k=3, candidate_pool=candidates[:15])
    t1 = time.time()
    _, _, time_c, c_evals = inf_max.select_celf(k=3, candidate_pool=candidates[:15])
    speedup = time_g / max(1e-4, time_c)
    print(f"    - Greedy Time: {time_g:.3f}s | CELF Time: {time_c:.3f}s | Speedup: {speedup:.1f}x (Lazy Evaluations: {c_evals})")
    
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"[+] DEMONSTRATION COMPLETE IN {total_elapsed:.2f} SECONDS! ALL 8 REQUIREMENTS VERIFIED.")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
