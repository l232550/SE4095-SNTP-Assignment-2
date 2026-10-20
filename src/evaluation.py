"""
Evaluation Metrics & Assessment module for SNTP Assignment 2.
Implements Requirement 7:
- Predictive accuracy: MAE, RMSE (SMM-ch9 Slide 34).
- Classification & Relevance: Precision@K, Recall@K, F1@K, Accuracy (SMM-ch9 Slides 36-37).
  Explicit positive definition (rating >= 4.0) vs negative definition (rating < 4.0).
- Ranking quality: NDCG@K, Spearman's rank correlation (SMM-ch9 Slides 38-40).
- Beyond-accuracy: Catalog Coverage, Diversity.
- Diffusion outcomes: Cascade size, exposure, duration, and variability reporting.
"""

import math
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau
from src.data_loader import SocialEnvironment

class Evaluator:
    """
    Evaluates recommender quality and joint recommendation-diffusion dynamics.
    Strictly prevents train-test leakage by testing only on held-out test ratings.
    """
    def __init__(self, env: SocialEnvironment, relevance_threshold: float = 4.0):
        self.env = env
        self.relevance_threshold = relevance_threshold

    def evaluate_rating_prediction(self, recommender) -> Dict[str, float]:
        """
        Computes MAE and RMSE on the held-out test ratings.
        Slide Justification: SMM-ch9 Slide 34.
        """
        errors = []
        squared_errors = []
        actuals = []
        preds = []
        
        for u_idx, items in self.env.test_ratings.items():
            for it_idx, true_rating in items.items():
                pred_rating = recommender.predict(u_idx, it_idx)
                err = true_rating - pred_rating
                errors.append(abs(err))
                squared_errors.append(err * err)
                actuals.append(true_rating)
                preds.append(pred_rating)
                
        if not errors:
            return {"MAE": 0.0, "RMSE": 0.0, "Test_Pairs": 0}
            
        mae = float(np.mean(errors))
        rmse = float(math.sqrt(np.mean(squared_errors)))
        
        # Binary Classification Accuracy where Relevant = rating >= threshold
        # Negative examples = rating < threshold (explicitly penalizing false positives)
        y_true = [1 if r >= self.relevance_threshold else 0 for r in actuals]
        y_pred = [1 if p >= self.relevance_threshold else 0 for p in preds]
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        acc = correct / len(y_true) if y_true else 0.0
        
        return {
            "MAE": mae,
            "RMSE": rmse,
            "Classification_Accuracy": acc,
            "Test_Pairs": len(errors)
        }

    def evaluate_top_k(
        self,
        recommender,
        k: int = 10,
        sample_users: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Computes Precision@K, Recall@K, F1@K, NDCG@K, Catalog Coverage, and Spearman's rho.
        Slide Justification: SMM-ch9 Slides 36-40.
        """
        if sample_users is None:
            # Users with at least 1 relevant item in test set
            sample_users = [
                u for u, items in self.env.test_ratings.items()
                if any(r >= self.relevance_threshold for r in items.values())
            ]
            
        if not sample_users:
            return {
                "Precision@K": 0.0,
                "Recall@K": 0.0,
                "F1@K": 0.0,
                "NDCG@K": 0.0,
                "Catalog_Coverage": 0.0,
                "Spearman_Rho": 0.0
            }
            
        precisions = []
        recalls = []
        ndcgs = []
        spearmans = []
        all_recommended_items: Set[int] = set()
        
        for u in sample_users:
            test_items = self.env.test_ratings.get(u, {})
            relevant_items = {it for it, r in test_items.items() if r >= self.relevance_threshold}
            if not relevant_items:
                continue
                
            # Candidate pool: test items + unseen items
            candidate_pool = list(test_items.keys())
            ranked = recommender.recommend(u, k=k, candidate_items=candidate_pool)
            rec_items = [it for it, _ in ranked]
            all_recommended_items.update(rec_items)
            
            # Hits
            hits = len(set(rec_items).intersection(relevant_items))
            prec = hits / max(1, k)
            rec = hits / max(1, len(relevant_items))
            precisions.append(prec)
            recalls.append(rec)
            
            # NDCG@K
            dcg = 0.0
            for rank_idx, it in enumerate(rec_items):
                rel = 1.0 if it in relevant_items else 0.0
                dcg += (math.pow(2, rel) - 1.0) / math.log2(rank_idx + 2)
                
            # Ideal DCG
            idcg = 0.0
            for rank_idx in range(min(k, len(relevant_items))):
                idcg += (math.pow(2, 1.0) - 1.0) / math.log2(rank_idx + 2)
                
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcgs.append(ndcg)
            
            # Spearman correlation over test items
            if len(test_items) >= 3:
                true_vals = [test_items[it] for it in test_items]
                pred_vals = [recommender.predict(u, it) for it in test_items]
                if len(set(true_vals)) > 1 and len(set(pred_vals)) > 1:
                    rho, _ = spearmanr(true_vals, pred_vals)
                    if not math.isnan(rho):
                        spearmans.append(rho)
                        
        p_mean = float(np.mean(precisions)) if precisions else 0.0
        r_mean = float(np.mean(recalls)) if recalls else 0.0
        f1_mean = (2 * p_mean * r_mean / (p_mean + r_mean)) if (p_mean + r_mean) > 0 else 0.0
        ndcg_mean = float(np.mean(ndcgs)) if ndcgs else 0.0
        spearman_mean = float(np.mean(spearmans)) if spearmans else 0.0
        coverage = len(all_recommended_items) / max(1, self.env.num_items)
        
        return {
            f"Precision@{k}": p_mean,
            f"Recall@{k}": r_mean,
            f"F1@{k}": f1_mean,
            f"NDCG@{k}": ndcg_mean,
            "Spearman_Rho": spearman_mean,
            "Catalog_Coverage": coverage
        }

    def evaluate_cold_start_segments(self, recommender) -> Dict[str, Dict[str, float]]:
        """
        Investigates recommender behavior across user activity segments:
        - Cold Users (<= 5 ratings in train)
        - Warm Users (> 5 and <= 20 ratings)
        - Dense Users (> 20 ratings)
        Requirement 2 specific analysis.
        """
        segments = {
            "Cold (<= 5 ratings)": [],
            "Warm (6-20 ratings)": [],
            "Dense (> 20 ratings)": []
        }
        
        for u, test_dict in self.env.test_ratings.items():
            train_count = len(self.env.train_ratings.get(u, {}))
            if train_count <= 5:
                cat = "Cold (<= 5 ratings)"
            elif train_count <= 20:
                cat = "Warm (6-20 ratings)"
            else:
                cat = "Dense (> 20 ratings)"
                
            for it, true_r in test_dict.items():
                pred_r = recommender.predict(u, it)
                err = abs(true_r - pred_r)
                segments[cat].append(err)
                
        results = {}
        for cat, err_list in segments.items():
            if err_list:
                results[cat] = {
                    "MAE": float(np.mean(err_list)),
                    "RMSE": float(math.sqrt(np.mean([e * e for e in err_list]))),
                    "Count": len(err_list)
                }
            else:
                results[cat] = {"MAE": 0.0, "RMSE": 0.0, "Count": 0}
                
        return results
