"""
Influence Maximization & Seed Selection module for SNTP Assignment 2.
Implements Requirement 5 (Select users under limited budget, accounting for audience overlap and uncertainty)
and Requirement 8 Optimization (CELF: Cost-Effective Lazy Forward algorithm exploiting submodularity).

Slide Justification:
- SMM-ch7 Slides 22-29: Maximizing the Spread of Cascades.
- Objective function f(S) = expected spread is monotone submodular (Slide 28).
- Greedy hill-climbing achieves (1 - 1/e) ~ 63% approximation guarantee.
- Baselines: Random and Degree Centrality. Degree fails when top nodes have overlapping audiences!
"""

import time
import heapq
import random
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from src.data_loader import SocialEnvironment
from src.diffusion import DiffusionSimulator

class InfluenceMaximizer:
    """
    Implements Influence Maximization seed selection under budget k:
    1. Standard Greedy Influence Maximization
    2. CELF (Cost-Effective Lazy Forward) Submodular Optimization
    3. Degree Centrality Baseline
    4. PageRank Baseline
    5. Random Baseline
    """
    def __init__(self, env: SocialEnvironment, base_p: float = 0.12, mc_trials: int = 10):
        self.env = env
        self.base_p = base_p
        self.mc_trials = mc_trials
        self.simulator = DiffusionSimulator(env)
        
    def _eval_spread(self, seed_set: Set[int], seed: int = 42) -> float:
        """Evaluates expected spread using Monte Carlo IC simulations."""
        if not seed_set:
            return 0.0
        res = self.simulator.run_monte_carlo_ic(
            seed_set=seed_set,
            num_trials=self.mc_trials,
            base_p=self.base_p,
            seed=seed
        )
        return res["mean_spread"]

    def select_random(self, k: int, seed: int = 42) -> Set[int]:
        """Baseline: Selects k random nodes."""
        rng = random.Random(seed)
        all_nodes = list(range(self.env.num_users))
        return set(rng.sample(all_nodes, min(k, len(all_nodes))))

    def select_degree(self, k: int) -> Set[int]:
        """
        Baseline: Selects top-k nodes by out-degree (popularity).
        Limitation: Ignores audience overlap!
        """
        degree_list = [(u, len(self.env.trust_out.get(u, set()))) for u in range(self.env.num_users)]
        degree_list.sort(key=lambda x: x[1], reverse=True)
        return set([u for u, _ in degree_list[:k]])

    def select_pagerank(self, k: int) -> Set[int]:
        """Baseline: Selects top-k nodes by network PageRank."""
        if self.env.networkx_graph is None or len(self.env.networkx_graph.nodes) == 0:
            return self.select_degree(k)
        try:
            pr = nx.pagerank(self.env.networkx_graph, alpha=0.85)
            sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            return set([u for u, _ in sorted_nodes[:k]])
        except Exception:
            return self.select_degree(k)

    def select_greedy(self, k: int, candidate_pool: Optional[List[int]] = None, seed: int = 42) -> Tuple[Set[int], List[float], float]:
        """
        Standard Greedy Algorithm (Slide Justification: SMM-ch7 Slide 29).
        Repeatedly selects candidate node v that maximizes marginal gain:
        v* = argmax_{v not in S} [ sigma(S U {v}) - sigma(S) ]
        
        Returns:
        - seed_set: selected k seeds
        - spread_history: spread after adding each seed
        - elapsed_time: runtime in seconds
        """
        start_time = time.time()
        if candidate_pool is None:
            # For speed on large graphs, candidate pool can be top degree nodes or all nodes
            candidate_pool = list(range(self.env.num_users))
            
        S: Set[int] = set()
        spread_history: List[float] = []
        current_spread = 0.0
        
        for step in range(k):
            best_v = None
            best_gain = -1.0
            
            for v in candidate_pool:
                if v in S:
                    continue
                spread_with_v = self._eval_spread(S.union({v}), seed=seed + step * 31)
                gain = spread_with_v - current_spread
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                    
            if best_v is not None:
                S.add(best_v)
                current_spread += max(0.0, best_gain)
                spread_history.append(current_spread)
            else:
                break
                
        elapsed = time.time() - start_time
        return S, spread_history, elapsed

    def select_celf(self, k: int, candidate_pool: Optional[List[int]] = None, seed: int = 42) -> Tuple[Set[int], List[float], float, int]:
        """
        CELF (Cost-Effective Lazy Forward) Algorithm (Requirement 8 Optimization).
        Exploits submodularity: marginal gains are monotonically non-increasing as the seed set grows.
        f(S U {v}) - f(S) >= f(T U {v}) - f(T) for S subseteq T.
        
        Uses a priority queue (max-heap) storing (cached_marginal_gain, node, iteration_updated).
        Avoids recalculating marginal gain for all nodes at every iteration!
        
        Returns:
        - seed_set: selected k seeds
        - spread_history: spread curve
        - elapsed_time: runtime in seconds
        - evaluations_count: number of Monte Carlo evaluations performed
        """
        start_time = time.time()
        if candidate_pool is None:
            candidate_pool = list(range(self.env.num_users))
            
        eval_count = 0
        # Priority queue stores (-marginal_gain, node, iteration_index)
        heap = []
        
        # Step 1: Initial evaluation of singletons
        for v in candidate_pool:
            gain = self._eval_spread({v}, seed=seed)
            eval_count += 1
            heapq.heappush(heap, (-gain, v, 0))
            
        S: Set[int] = set()
        spread_history: List[float] = []
        current_spread = 0.0
        curr_iter = 0
        
        # Step 2: Lazy forward hill-climbing
        while len(S) < k and heap:
            neg_gain, u, iter_evaluated = heapq.heappop(heap)
            
            # If this node was already updated in the current iteration,
            # by submodularity its gain is guaranteed to be >= all other nodes in heap!
            if iter_evaluated == curr_iter:
                S.add(u)
                current_spread += (-neg_gain)
                spread_history.append(current_spread)
                curr_iter += 1
            else:
                # Re-evaluate marginal gain for this node with current seed set S
                new_spread = self._eval_spread(S.union({u}), seed=seed + curr_iter * 31)
                eval_count += 1
                new_gain = max(0.0, new_spread - current_spread)
                heapq.heappush(heap, (-new_gain, u, curr_iter))
                
        elapsed = time.time() - start_time
        return S, spread_history, elapsed, eval_count

    def compute_audience_overlap(self, seed_set: Set[int]) -> float:
        """
        Quantifies the audience overlap between seeds:
        Average pairwise Jaccard index between 1-hop social audiences.
        Higher overlap means redundant reach.
        """
        if len(seed_set) <= 1:
            return 0.0
            
        seeds = list(seed_set)
        jaccard_sum = 0.0
        pairs = 0
        
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                u, v = seeds[i], seeds[j]
                nbrs_u = self.env.trust_out.get(u, set())
                nbrs_v = self.env.trust_out.get(v, set())
                union = nbrs_u.union(nbrs_v)
                inter = nbrs_u.intersection(nbrs_v)
                jaccard = len(inter) / max(1, len(union))
                jaccard_sum += jaccard
                pairs += 1
                
        return jaccard_sum / max(1, pairs)
