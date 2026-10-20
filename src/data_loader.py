"""
Data Loader & Environment Representation module for SNTP Assignment 2.
Handles Requirement 1: Represent the Social Environment.
- Ingests Epinions ratings and trust graph.
- Distinguishes observed data from inferred signals.
- Implements induced core extraction & train-test split without leakage.
- Provides a synthetic social network generator as an alternative/reproducible testbed.
"""

import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import networkx as nx

from src.config import (
    RATINGS_FILE, TRUST_FILE, RANDOM_SEED, CORE_CONFIG, RECOMMENDER_CONFIG
)

class SocialEnvironment:
    """
    Encapsulates the complete Social Environment:
    1. User-Item Interaction Matrix (Explicit Ratings 1-5)
    2. Directed Social Trust Network G = (V, E)
    """
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        trust_df: pd.DataFrame,
        name: str = "Epinions"
    ):
        self.name = name
        self.ratings_df = ratings_df
        self.trust_df = trust_df
        
        # Mappings: Original IDs -> Contiguous Index IDs
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        
        # Core structures
        self.num_users: int = 0
        self.num_items: int = 0
        self.train_ratings: Dict[int, Dict[int, float]] = defaultdict(dict) # user_idx -> {item_idx: rating}
        self.test_ratings: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.item_users_train: Dict[int, Dict[int, float]] = defaultdict(dict) # item_idx -> {user_idx: rating}
        
        # Trust Network: Directed graph where edge u -> v means u trusts v
        # In diffusion/influence: information or recommendations can flow along trust edges
        self.trust_out: Dict[int, Set[int]] = defaultdict(set) # u trusts whom: u -> [v1, v2]
        self.trust_in: Dict[int, Set[int]] = defaultdict(set)  # who trusts u: [w1, w2] -> u
        self.networkx_graph: Optional[nx.DiGraph] = None
        
        # Precomputed user & item statistics (computed strictly on train set)
        self.user_means: Dict[int, float] = {}
        self.item_means: Dict[int, float] = {}
        self.global_mean: float = 3.5

    def build(self, test_size: float = 0.20, seed: int = RANDOM_SEED):
        """
        Builds contiguous mappings, train/test splits, and directed trust graph.
        Strictly prevents train-test leakage.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        unique_users = sorted(list(set(self.ratings_df['user_id']).union(set(self.trust_df['source'])).union(set(self.trust_df['target']))))
        unique_items = sorted(list(set(self.ratings_df['item_id'])))
        
        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {it: i for i, it in enumerate(unique_items)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        
        # Train / Test split per user to ensure fair evaluation of unobserved items
        user_grouped = self.ratings_df.groupby('user_id')
        train_rows = []
        test_rows = []
        
        for u_id, group in user_grouped:
            u_idx = self.user_to_idx[u_id]
            records = group.to_dict('records')
            if len(records) >= 2:
                # Random split without leakage: ensure at least 1 train and 1 test item
                random.shuffle(records)
                n_test = max(1, int(len(records) * test_size))
                if n_test >= len(records):
                    n_test = len(records) - 1
                test_recs = records[:n_test]
                train_recs = records[n_test:]
            else:
                train_recs = records
                test_recs = []
                
            for r in train_recs:
                it_idx = self.item_to_idx[r['item_id']]
                val = float(r['rating'])
                self.train_ratings[u_idx][it_idx] = val
                self.item_users_train[it_idx][u_idx] = val
                train_rows.append((u_idx, it_idx, val))
                
            for r in test_recs:
                it_idx = self.item_to_idx[r['item_id']]
                val = float(r['rating'])
                self.test_ratings[u_idx][it_idx] = val
                test_rows.append((u_idx, it_idx, val))
                
        # Build Trust Graph
        nx_g = nx.DiGraph()
        nx_g.add_nodes_from(range(self.num_users))
        
        for _, row in self.trust_df.iterrows():
            src = row['source']
            tgt = row['target']
            if src in self.user_to_idx and tgt in self.user_to_idx:
                u_src = self.user_to_idx[src]
                u_tgt = self.user_to_idx[tgt]
                if u_src != u_tgt:
                    self.trust_out[u_src].add(u_tgt)
                    self.trust_in[u_tgt].add(u_src)
                    nx_g.add_edge(u_src, u_tgt)
                    
        self.networkx_graph = nx_g
        
        # Compute baseline statistics strictly from Train set
        all_train_ratings = [val for u_dict in self.train_ratings.values() for val in u_dict.values()]
        self.global_mean = float(np.mean(all_train_ratings)) if all_train_ratings else 3.5
        
        for u_idx in range(self.num_users):
            if u_idx in self.train_ratings and len(self.train_ratings[u_idx]) > 0:
                self.user_means[u_idx] = float(np.mean(list(self.train_ratings[u_idx].values())))
            else:
                self.user_means[u_idx] = self.global_mean
                
        for it_idx in range(self.num_items):
            if it_idx in self.item_users_train and len(self.item_users_train[it_idx]) > 0:
                self.item_means[it_idx] = float(np.mean(list(self.item_users_train[it_idx].values())))
            else:
                self.item_means[it_idx] = self.global_mean

    def get_summary_statistics(self) -> Dict[str, any]:
        """Returns structural statistics of the social environment."""
        num_ratings_train = sum(len(v) for v in self.train_ratings.values())
        num_ratings_test = sum(len(v) for v in self.test_ratings.values())
        num_edges = sum(len(v) for v in self.trust_out.values())
        
        density_ratings = (num_ratings_train + num_ratings_test) / (self.num_users * max(1, self.num_items))
        avg_out_degree = num_edges / max(1, self.num_users)
        
        return {
            "Environment Name": self.name,
            "Total Users": self.num_users,
            "Total Items": self.num_items,
            "Train Ratings": num_ratings_train,
            "Test Ratings": num_ratings_test,
            "Total Social Edges": num_edges,
            "Rating Matrix Sparsity": f"{(1.0 - density_ratings) * 100:.2f}%",
            "Avg Social Out-Degree": f"{avg_out_degree:.2f}",
            "Global Mean Rating": f"{self.global_mean:.2f}"
        }


def load_epinions_data(
    ratings_path: str = str(RATINGS_FILE),
    trust_path: str = str(TRUST_FILE),
    max_users: Optional[int] = CORE_CONFIG["max_core_users"],
    min_ratings: int = CORE_CONFIG["min_ratings_per_user"],
    min_degree: int = CORE_CONFIG["min_social_degree"],
    seed: int = RANDOM_SEED
) -> SocialEnvironment:
    """
    Loads Epinions dataset and extracts a dense, active induced core.
    
    Slide Justification (Recommendation System-1 Slide 18-19, SMM-ch9 Slide 3):
    Explicit ratings (1-5) provide preference signals. Unrated items are missing (unobserved).
    Trust is a directed edge: source user trusts target user.
    To rigorously study the interplay between collaborative filtering and social diffusion,
    we extract the active core of users present in both networks.
    """
    print(f"[*] Ingesting Epinions data from:\n    Ratings: {ratings_path}\n    Trust: {trust_path}")
    
    # Read raw files
    ratings_df = pd.read_csv(
        ratings_path,
        sep=r'\s+',
        header=None,
        names=['user_id', 'item_id', 'rating'],
        dtype={'user_id': int, 'item_id': int, 'rating': float}
    )
    
    trust_df = pd.read_csv(
        trust_path,
        sep=r'\s+',
        header=None,
        names=['source', 'target', 'trust'],
        dtype={'source': int, 'target': int, 'trust': int}
    )
    
    if max_users is not None:
        print(f"[*] Extracting induced active core (Target: ~{max_users} users)...")
        # Find users who have both sufficient ratings and social links
        user_rating_counts = ratings_df['user_id'].value_counts()
        active_rating_users = set(user_rating_counts[user_rating_counts >= min_ratings].index)
        
        trust_degree = trust_df['source'].value_counts() + trust_df['target'].value_counts()
        trust_degree = trust_degree.fillna(0)
        active_social_users = set(trust_degree[trust_degree >= min_degree].index)
        
        # Intersect
        core_candidate_users = list(active_rating_users.intersection(active_social_users))
        
        # Activity score combining ratings count and trust degree
        activity_score = {
            u: user_rating_counts.get(u, 0) + trust_degree.get(u, 0)
            for u in core_candidate_users
        }
        
        # Stratified selection to ensure cold, warm, and dense users are all present
        n_dense = int(max_users * 0.60)
        n_warm = int(max_users * 0.20)
        n_cold = max_users - n_dense - n_warm
        
        dense_cand = sorted([u for u in core_candidate_users if user_rating_counts[u] > 20], key=lambda u: activity_score.get(u, 0), reverse=True)
        selected_dense = set(dense_cand[:n_dense])
        dense_items = set(ratings_df[ratings_df['user_id'].isin(selected_dense)]['item_id'])
        
        warm_cand = sorted([u for u in core_candidate_users if 6 <= user_rating_counts[u] <= 20], key=lambda u: activity_score.get(u, 0), reverse=True)
        selected_warm = set(warm_cand[:n_warm])
        
        selected_so_far = selected_dense.union(selected_warm)
        edges_involving = trust_df[(trust_df['source'].isin(selected_so_far)) | (trust_df['target'].isin(selected_so_far))]
        cold_connected = set(edges_involving['source']).union(set(edges_involving['target']))
        
        cold_overlap = ratings_df[ratings_df['user_id'].isin(cold_connected) & ratings_df['item_id'].isin(dense_items)]
        cold_with_overlap = set(cold_overlap['user_id'].value_counts()[lambda x: x >= 2].index)
        cold_cand = [u for u in core_candidate_users if 2 <= user_rating_counts[u] <= 6 and u in cold_with_overlap]
        if len(cold_cand) < n_cold:
            cold_cand = [u for u in core_candidate_users if 2 <= user_rating_counts[u] <= 6]
        selected_cold = set(sorted(cold_cand, key=lambda u: activity_score.get(u, 0), reverse=True)[:n_cold])
        
        selected_core_users = selected_dense.union(selected_warm).union(selected_cold)
        
        # Fallback if any tier had fewer candidates
        if len(selected_core_users) < max_users:
            remaining = [u for u in sorted(core_candidate_users, key=lambda u: activity_score.get(u, 0), reverse=True) if u not in selected_core_users]
            selected_core_users.update(remaining[:max_users - len(selected_core_users)])
        
        # Filter ratings and trust to the selected core
        ratings_filtered = ratings_df[ratings_df['user_id'].isin(selected_core_users)].copy()
        
        # Keep items with at least 2 ratings in the core to ensure collaborative overlaps
        item_counts = ratings_filtered['item_id'].value_counts()
        valid_items = set(item_counts[item_counts >= 2].index)
        ratings_filtered = ratings_filtered[ratings_filtered['item_id'].isin(valid_items)].copy()
        
        trust_filtered = trust_df[
            trust_df['source'].isin(selected_core_users) & 
            trust_df['target'].isin(selected_core_users)
        ].copy()
    else:
        ratings_filtered = ratings_df
        trust_filtered = trust_df
        
    env = SocialEnvironment(ratings_filtered, trust_filtered, name="Epinions-ActiveCore")
    env.build(test_size=RECOMMENDER_CONFIG["test_size"], seed=seed)
    
    stats = env.get_summary_statistics()
    print("[+] Social Environment successfully constructed:")
    for k, v in stats.items():
        print(f"    - {k}: {v}")
        
    return env


def generate_synthetic_social_environment(
    num_users: int = 120,
    num_items: int = 80,
    num_communities: int = 3,
    avg_degree: int = 6,
    sparsity: float = 0.88,
    seed: int = RANDOM_SEED
) -> SocialEnvironment:
    """
    Generates a synthetic social platform environment with homophily.
    Users belong to interest communities. Connected users share preferences with higher probability.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Generate Social Network with Community Homophily (Stochastic Block Model)
    community_sizes = [num_users // num_communities] * num_communities
    p_in = avg_degree / (num_users / num_communities)
    p_out = p_in * 0.15 # Higher probability of trusting within community
    
    pref_matrix = [[p_in if i == j else p_out for j in range(num_communities)] for i in range(num_communities)]
    sbm = nx.stochastic_block_model(community_sizes, pref_matrix, directed=True, seed=seed)
    
    trust_records = []
    for u, v in sbm.edges():
        trust_records.append({'source': u, 'target': v, 'trust': 1})
    trust_df = pd.DataFrame(trust_records)
    
    # 2. Generate Item Categories and User Ratings reflecting Homophily
    # Items belong to categories matching communities
    item_categories = np.random.randint(0, num_communities, size=num_items)
    
    # User community assignments
    user_communities = {}
    idx = 0
    for c_id, size in enumerate(community_sizes):
        for _ in range(size):
            user_communities[idx] = c_id
            idx += 1
            
    rating_records = []
    for u in range(num_users):
        u_comm = user_communities[u]
        # Number of items rated
        num_ratings = max(5, int((1.0 - sparsity) * num_items * np.random.uniform(0.7, 1.3)))
        rated_items = np.random.choice(num_items, size=min(num_ratings, num_items), replace=False)
        for it in rated_items:
            it_cat = item_categories[it]
            # Higher rating if item category matches user's community
            if it_cat == u_comm:
                rating = float(np.random.choice([4.0, 5.0], p=[0.35, 0.65]))
            else:
                rating = float(np.random.choice([1.0, 2.0, 3.0], p=[0.4, 0.4, 0.2]))
            rating_records.append({'user_id': u, 'item_id': it, 'rating': rating})
            
    ratings_df = pd.DataFrame(rating_records)
    
    env = SocialEnvironment(ratings_df, trust_df, name="Synthetic-HomophilicNetwork")
    env.build(test_size=RECOMMENDER_CONFIG["test_size"], seed=seed)
    return env
