"""
Group Recommendation & Group Diffusion module for SNTP Assignment 2.
Implements Requirement 6:
- Group aggregation strategies: Average Satisfaction, Least Misery, Most Pleasure.
  (Slide Justification: Recommendation System-1, Slide 37)
- Group formation (cohesive friend clusters vs conflicting taste groups).
- Simulates diffusion of group-recommended items through the network when introduced by the group.
- Investigates the alignment / trade-off between group satisfaction and widespread adoption.
"""

import numpy as np
import random
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from src.data_loader import SocialEnvironment
from src.diffusion import DiffusionSimulator

class GroupRecommender:
    """
    Implements Group Recommendation aggregation strategies:
    1. Average Satisfaction: Agg_avg(g, i) = (1/|g|) * sum_{u in g} r_hat(u, i)
    2. Least Misery:        Agg_min(g, i) = min_{u in g} r_hat(u, i)
    3. Most Pleasure:       Agg_max(g, i) = max_{u in g} r_hat(u, i)
    """
    def __init__(self, env: SocialEnvironment, recommender):
        self.env = env
        self.recommender = recommender
        self.simulator = DiffusionSimulator(env)

    def form_group(self, size: int = 5, group_type: str = "conflicting", seed: int = 42) -> List[int]:
        """
        Forms a group of users:
        - 'conflicting': users with divergent rating histories or low similarity.
        - 'cohesive': a connected clique or high-trust friend community.
        - 'random': uniformly sampled users.
        """
        rng = random.Random(seed)
        all_users = list(range(self.env.num_users))
        
        if group_type == "cohesive":
            # Pick a core user with high degree and their trusted friends
            sorted_users = sorted(all_users, key=lambda u: len(self.env.trust_out.get(u, set())), reverse=True)
            for center in sorted_users:
                friends = list(self.env.trust_out.get(center, set()))
                if len(friends) >= size - 1:
                    return [center] + rng.sample(friends, size - 1)
            return rng.sample(all_users, min(size, len(all_users)))
            
        elif group_type == "conflicting":
            # Pick users with contrasting mean ratings or sparse common items
            sorted_by_mean = sorted(all_users, key=lambda u: self.env.user_means.get(u, 3.5))
            half = size // 2
            # Pick lowest mean raters and highest mean raters
            low_raters = sorted_by_mean[:half]
            high_raters = sorted_by_mean[- (size - half):]
            return low_raters + high_raters
            
        else: # random
            return rng.sample(all_users, min(size, len(all_users)))

    def rank_items_for_group(
        self,
        group: List[int],
        candidate_items: Optional[List[int]] = None
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Ranks unseen candidate items for the group using:
        - Average Satisfaction
        - Least Misery
        - Most Pleasure
        """
        if candidate_items is None:
            # Union of items not fully rated by all members, but with at least some interest
            candidate_items = list(range(min(200, self.env.num_items)))
            
        avg_scores = []
        min_scores = []
        max_scores = []
        
        for it in candidate_items:
            preds = [self.recommender.predict(u, it) for u in group]
            avg_score = float(np.mean(preds))
            min_score = float(np.min(preds))
            max_score = float(np.max(preds))
            
            avg_scores.append((it, avg_score))
            min_scores.append((it, min_score))
            max_scores.append((it, max_score))
            
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        min_scores.sort(key=lambda x: x[1], reverse=True)
        max_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "Average": avg_scores,
            "Least Misery": min_scores,
            "Most Pleasure": max_scores
        }

    def evaluate_group_diffusion(
        self,
        group: List[int],
        candidate_items: Optional[List[int]] = None,
        mc_trials: int = 10,
        base_p: float = 0.12,
        seed: int = 42
    ) -> Dict[str, any]:
        """
        Recommends the best item under each strategy, then tests whether the
        item preferred by the group strategy also spreads most widely when
        introduced through that group.
        """
        ranked = self.rank_items_for_group(group, candidate_items)
        seed_set = set(group)
        
        results = {}
        for strategy, items in ranked.items():
            best_item, group_predicted_score = items[0]
            
            # Individual satisfaction across members
            member_preds = [self.recommender.predict(u, best_item) for u in group]
            avg_sat = float(np.mean(member_preds))
            min_sat = float(np.min(member_preds))
            max_sat = float(np.max(member_preds))
            variance_sat = float(np.var(member_preds))
            
            # Simulate preference-aware diffusion initiated by the group
            diffusion_stats = self.simulator.run_monte_carlo_ic(
                seed_set=seed_set,
                item_id=best_item,
                recommender=self.recommender,
                base_p=base_p,
                pref_weighted=True,
                num_trials=mc_trials,
                seed=seed
            )
            
            results[strategy] = {
                "best_item": best_item,
                "strategy_score": group_predicted_score,
                "group_avg_sat": avg_sat,
                "group_min_sat": min_sat,
                "group_max_sat": max_sat,
                "group_variance_sat": variance_sat,
                "expected_spread": diffusion_stats["mean_spread"],
                "std_spread": diffusion_stats["std_spread"],
                "expected_exposure": diffusion_stats["mean_exposure"]
            }
            
        return results
