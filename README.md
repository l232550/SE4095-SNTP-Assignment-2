# SNTP Assignment 2: Recommendation & Information Diffusion in a Social Network

**Course Title:** Social Network & Text Processing (SNTP)  
**Institution:** National University of Computer and Emerging Sciences (FAST-NUCES), Chiniot-Faisalabad Campus  
**Department:** Department of Software Engineering  
**Student Name:** Tooba Nadeem  
**Roll Number:** 23L-2550  
**Due Date:** Tuesday, 20 October 2026, 10:00 PM  

---

## 1. Project Overview & Architecture

This repository contains a complete, rigorous, and reproducible implementation of **Joint Personalized Recommendation, Social Influence, and Information Diffusion** built from first principles strictly following the course lecture slides:
- **Recommendation Systems 1:** Explicit vs. implicit ratings, memory-based User-KNN (Slide 23-27), Item-KNN (Slide 21, 28-29), Group Recommendation aggregation (Slide 37).
- **Social Media Mining (SMM) Chapter 9:** Socially-constrained recommendations (Slide 10), Social regularized matrix factorization (Slide 9), evaluation metrics (MAE, RMSE, Precision@K, Recall@K, NDCG@K, Spearman's $\rho$, Slide 32-40).
- **Social Media Mining (SMM) Chapter 7:** Independent Cascade (IC) Model (Slide 17-19), Epidemic SIR Model (Slide 64-67), Greedy Influence Maximization under budget (Slide 22-29), Submodularity & CELF optimization (Slide 28), and network interventions (Slide 33 & 78).

### Overall System Architecture & Data Flow
```
+---------------------------------------------------------------------------------+
|                                Observed Data                                    |
|   1. User-Item Ratings (Epinions: 1-5 explicit)                                 |
|   2. User-User Trust Graph (Directed edges: source trusts target)               |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                            Data Preparation Layer                               |
|   - Leakage-free 80/20 train-test splitting per user                            |
|   - Mean user / item baselines computed strictly on training data               |
|   - Active induced core extraction (ratings + social overlap)                   |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                             Recommender Layer                                   |
|   - Approach A: Memory-Based User-KNN (co-rated Pearson correlation)            |
|   - Approach B: Memory-Based Item-KNN (adjusted cosine similarity)              |
|   - Approach C: Latent Factor Model (SVD with manual SGD update rules)          |
|   - Requirement 3: Social-Constrained User-KNN (blending parameter β ∈ [0, 1]) |
|   - Social Regularized SVD (social taste penalty on trusted friends)            |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                       Adoption & Information Diffusion                          |
|   - Three Stages: Receiving Recommendation -> Exposed -> Adopting               |
|   - Independent Cascade (IC) Model (discrete one-shot activation)               |
|   - SIR Epidemic Model (continuous transmission with recovery compartments)     |
|   - Preference-Aware Adoption: P(adopt | exposed) modulated by predicted rating|
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                      Influence Maximization & Budget Layer                      |
|   - Seed selection under budget k ∈ {2, 5, 8, 12, 15, 20}                       |
|   - Marginal spread evaluation considering overlapping audiences                |
|   - Baselines: Degree Centrality, PageRank, Random Selection                    |
|   - Requirement 8 Optimization: CELF (Lazy Forward) submodular hill-climbing    |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                         Group Layer & Interventions                             |
|   - Group Aggregation: Average Satisfaction, Least Misery, Most Pleasure        |
|   - Group-seeded cascade diffusion (Satisfaction vs. Virality trade-off)       |
|   - Interventions: Degree capping (out-link limits) and transmission damping    |
+---------------------------------------------------------------------------------+
```

---

## 2. Directory Structure & Repository Map

```
23L2550_SNTP_Assignment2/
├── README.md                              # This root documentation file
├── requirements.txt                       # Dependency list with pinned versions
├── generate_report.py                     # Script to compile publication-ready PDF report
├── 23L2550_SNTP_Assignment2_Report.pdf    # Searchable PDF written report (Req 4)
├── data/
│   └── epinions/
│       ├── ratings_data.txt               # Epinions user-item explicit ratings (664,825 rows)
│       └── trust_data.txt                 # Epinions directed trust edges (487,184 rows)
├── src/
│   ├── __init__.py
│   ├── config.py                          # Global paths, seeds, hyperparameters, simulation config
│   ├── data_loader.py                     # Dataset loading, active core extraction, synthetic generator
│   ├── recommender.py                     # User-KNN, Item-KNN, Social-KNN (beta), and SVD (manual SGD)
│   ├── diffusion.py                       # IC Model, SIR Model, preference-aware adoption simulator
│   ├── influence.py                       # Greedy Influence Maximizer, CELF optimization, Baselines
│   ├── group_rec.py                       # Average, Least Misery, Most Pleasure & group cascade diffusion
│   └── evaluation.py                      # MAE, RMSE, Precision@K, Recall@K, NDCG@K, Spearman, Coverage
├── experiments/
│   ├── __init__.py
│   ├── run_demo.py                        # Rapid end-to-end demo covering all 8 requirements (<15s)
│   ├── run_all.py                         # Master runner: executes all 4 experiment suites + PDF report
│   ├── run_recs.py                        # Experiment 1: Requirements 1, 2, 3 (Recommenders & Social beta)
│   ├── run_diffusion.py                   # Experiment 2: Requirements 4, 6 (IC vs SIR & Group Diffusion)
│   ├── run_influence.py                   # Experiment 3: Requirement 5 (Influence Maximization & Budgets)
│   └── run_intervention.py                # Experiment 4: Requirement 8 (Interventions & CELF Speedup)
└── results/
    ├── tables/                            # Saved experimental results (CSV format)
    │   ├── req2_model_comparison.csv
    │   ├── req2_cold_start_analysis.csv
    │   ├── req3_social_beta_sweep.csv
    │   ├── req4_diffusion_probabilities.csv
    │   ├── req5_influence_maximization_results.csv
    │   ├── req6_group_recommendation_diffusion.csv
    │   ├── req8_diffusion_intervention_results.csv
    │   └── req8_celf_optimization_benchmark.csv
    └── plots/                             # High-resolution figures (300 DPI PNG)
        ├── req2_model_performance.png
        ├── req2_cold_start_analysis.png
        ├── req3_social_beta_effect.png
        ├── req4_diffusion_model_dynamics.png
        ├── req4_spread_vs_probability.png
        ├── req5_spread_vs_budget.png
        ├── req5_audience_overlap.png
        ├── req6_group_satisfaction_vs_spread.png
        ├── req8_intervention_spread_vs_satisfaction.png
        └── req8_celf_vs_greedy_runtime.png
```

---

## 3. Dataset Information & Acquisition Instructions

The assignment utilizes the benchmark **Epinions Social Network & Recommendation Dataset**:
1. **Included in Submission:**
   - Raw ratings: `data/epinions/ratings_data.txt` (664,825 explicit ratings on a 1–5 scale).
   - Social trust: `data/epinions/trust_data.txt` (487,184 directed trust relationships where source user trusts target user).
2. **Public Dataset Source & Citation:**
   - Dataset repository: Trustlet Epinions Dataset ([http://www.trustlet.org/extended_epinions.html](http://www.trustlet.org/extended_epinions.html)) and Stanford SNAP ([https://snap.stanford.edu/data/soc-Epinions1.html](https://snap.stanford.edu/data/soc-Epinions1.html)).
   - Citation: P. Massa and P. Avesani, *"Trust-aware Collaborative Filtering for Recommender Systems"*, ACM RecSys, 2007.
3. **Data Format:**
   - `ratings_data.txt`: `<user_id> <item_id> <rating_value>` (space-delimited integers/floats).
   - `trust_data.txt`: `<source_user_id> <target_user_id> <trust_weight>` (directed binary trust where weight=1).
4. **Synthetic Network Fallback Generator:**
   - In environments where raw Epinions files are absent, `src/data_loader.py` includes `generate_synthetic_social_environment(num_users=200, num_items=500, ...)` implementing a scale-free Barabási-Albert network with latent Dirichlet preferences for zero-dependency test runs.

---

## 4. Software Environment & Installation Instructions

### Prerequisites
- **Operating System:** Windows, Linux, or macOS
- **Python Version:** Python 3.10+ (tested on Python 3.12.9)
- **RAM Recommended:** 4 GB - 8 GB
- **Hardware Requirement:** Standard CPU (no GPU needed)

### Installation Steps
1. Navigate to the outer submission folder:
   ```bash
   cd 23L2550_SNTP_Assignment2
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - On Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Execution Commands & Expected Runtimes

### A. Quick Demonstration (< 15 seconds)
To run a fast, self-contained demonstration verifying all 8 requirements end-to-end:
```bash
python experiments/run_demo.py
```
- **Expected Runtime:** ~7 - 12 seconds
- **What it does:** Extracts a fast core, trains User-KNN, Item-KNN, and SVD, evaluates social $\beta$ weighting, simulates IC and SIR cascades, runs CELF influence maximization, tests group recommendation strategies, evaluates metrics, and benchmarks CELF vs. Greedy speedup.

### B. Reproducing the Full Experimental Suite (~ 3 - 5 minutes)
To execute all experimental suites end-to-end, regenerate all CSV data tables, regenerate all 300 DPI plots, and compile the final searchable PDF report:
```bash
python experiments/run_all.py
```
- **Expected Runtime:** ~3 to 5 minutes on a standard laptop CPU.
- **Output:** All CSV files in `results/tables/`, all charts in `results/plots/`, and `23L2550_SNTP_Assignment2_Report.pdf` in the root folder.

### C. Running Individual Experiment Modules
You can also run any specific requirement module independently:
- **Requirements 1, 2, 3 (Recommenders & Social Beta):**
  ```bash
  python experiments/run_recs.py
  ```
- **Requirements 4 & 6 (Diffusion & Group Dynamics):**
  ```bash
  python experiments/run_diffusion.py
  ```
- **Requirement 5 (Influence Maximization & Budgets):**
  ```bash
  python experiments/run_influence.py
  ```
- **Requirement 8 (Intervention & CELF Optimization):**
  ```bash
  python experiments/run_intervention.py
  ```
- **Compile / Recompile PDF Report Only:**
  ```bash
  python generate_report.py
  ```

---

## 5. Requirement Mapping to Code & Report

| Requirement | Description | Source File(s) & Core Function(s) | Report Section |
| :--- | :--- | :--- | :--- |
| **Req 1: Social Environment** | Epinions ingestion, directed trust graph, sparse rating matrix, leakage-free 80/20 train-test split, synthetic network generator | `src/data_loader.py` (`load_epinions_data`, `SocialEnvironment`, `generate_synthetic_social_environment`) | Section 1 |
| **Req 2: Preference Learning (Two Methods)** | User-KNN (Pearson correlation on co-rated items) and Item-KNN (adjusted cosine), plus SVD with manual SGD. Cold-start user segment evaluation | `src/recommender.py` (`UserKNNRecommender`, `ItemKNNRecommender`, `MatrixFactorizationRecommender`), `src/evaluation.py` (`evaluate_cold_start_segments`) | Section 2 |
| **Req 3: Social Relationships** | Social-Constrained User-KNN with convex blending parameter $\beta \in [0, 1]$, Social Regularized SVD, isolating when social context improves vs. hurts | `src/recommender.py` (`SocialUserKNNRecommender`, `MatrixFactorizationRecommender`), `experiments/run_recs.py` | Section 3 |
| **Req 4: Adoption & Information Spread** | Three-stage model (recommended -> exposed -> adopted), Independent Cascade (IC) and Epidemic SIR models from scratch, preference-aware probability modulation | `src/diffusion.py` (`DiffusionSimulator.simulate_ic`, `simulate_sir`, `run_monte_carlo_ic`), `experiments/run_diffusion.py` | Section 4 |
| **Req 5: Limited Promotion Budget** | Influence Maximization under budget $k \in \{2, 5, 8, 12, 15, 20\}$, marginal gain greedy selection, audience overlap quantification (Jaccard), Degree/PageRank/Random baselines, Monte Carlo confidence bounds | `src/influence.py` (`InfluenceMaximizer.select_greedy`, `select_degree`, `select_pagerank`, `compute_audience_overlap`), `experiments/run_influence.py` | Section 5 |
| **Req 6: Group Recommendation** | Average Satisfaction, Least Misery, Most Pleasure aggregations (Slide 37); downstream cascade diffusion seeded by group; individual satisfaction vs. widespread adoption trade-off | `src/group_rec.py` (`GroupRecommender.rank_items_for_group`, `evaluate_group_diffusion`), `experiments/run_diffusion.py` | Section 6 |
| **Req 7: Complete Evaluation** | Leakage prevention, MAE, RMSE, Precision@K, Recall@K, NDCG@K, Spearman's $\rho$, Catalog Coverage, classification accuracy (positive $r \ge 4$ vs negative $r < 4$), Monte Carlo variability reporting | `src/evaluation.py` (`Evaluator.evaluate_rating_prediction`, `evaluate_top_k`), `experiments/run_recs.py`, `experiments/run_diffusion.py` | Section 7 |
| **Req 8: Intervention & Computational Cost** | Degree capping (out-link limits) and transmission damping interventions; bottleneck identification; CELF (Cost-Effective Lazy Forward) submodular optimization benchmark | `src/influence.py` (`InfluenceMaximizer.select_celf`), `experiments/run_intervention.py` | Section 8 & 9 |

---

## 6. Key Scientific Findings & Discussion

### A. When Does Social Influence Help vs. Create Trade-offs?
1. **Cold Users & Sparsity (High Benefit):** For users with limited interaction history ($\le 5$ ratings), collaborative filtering fails due to sparse co-rated items. Social trust edges act as an effective prior, rescuing cold users and improving recommendation coverage.
2. **Dense Users & Taste Divergence (Trade-off):** For active users with established tastes, setting $\beta \to 1.0$ degrades RMSE. Trust does not imply identical taste across all item genres. Imposing social constraints introduces taste bias and reduces intra-list diversity.
3. **Consensus vs. Virality Trade-off in Groups:** *Least Misery* prevents dissatisfaction and maximizes group harmony, but selects conservative, lower-arousal items that produce modest downstream cascades. *Most Pleasure* and *Average Satisfaction* pick higher-variance, polarized items that achieve significantly broader viral spread through the social network.

### B. Influence Maximization & Audience Redundancy
- **Degree Centrality Failure:** Out-Degree centrality selects the most popular nodes, but in clustered social networks, top hubs have high audience overlap (Jaccard index > 0.60). As a result, degree-based promotion wastes budget repeatedly targeting the same mutual friends.
- **Greedy / CELF Superiority:** By evaluating marginal spread $\sigma(S \cup \{v\}) - \sigma(S)$, Greedy explicitly penalizes audience overlap, attaining higher coverage and spread per unit of budget.

### C. Computational Cost & CELF Optimization
- Standard Greedy requires $\mathcal{O}(k \cdot |V| \cdot R)$ Monte Carlo simulations.
- By exploiting **submodularity** (diminishing marginal returns), **CELF** re-evaluates candidates lazily via a priority heap, reducing Monte Carlo function calls by **over 92%** while discovering the **identically optimal** seed set.

---

## 7. Known Limitations
1. **Static Trust Assumptions:** The Epinions trust network is static; real social platforms exhibit dynamic relationship formation and decay over time.
2. **Absence of Explicit Distrust:** The model treats absent edges as lack of trust rather than explicit distrust or negative influence.
3. **Discrete Diffusion Granularity:** Real user adoption delays follow continuous-time Poisson processes rather than uniform discrete simulation rounds.
