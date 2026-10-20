"""
Recommender Systems module for SNTP Assignment 2.
Implements Requirement 2 (Two Substantially Different Recommendation Approaches from scratch)
and Requirement 3 (Incorporate Social Relationships & isolate social parameter beta).

Slide Justifications:
- Approach A: Memory-Based User-KNN (Recommendation System-1, Slides 23-27)
  Exact formula: r_hat(u,i) = r_bar_u + sum_v(sim(u,v)*(r_v,i - r_bar_v)) / sum_v(|sim(u,v)|)
- Approach B: Memory-Based Item-KNN (Recommendation System-1, Slides 21, 28-29)
  Exact formula: r_hat(u,i) = r_bar_i + sum_j(sim(i,j)*(r_u,j - r_bar_j)) / sum_j(|sim(i,j)|)
- Approach C: Model-Based Latent Factor Model (SVD / Funk-SVD with manual SGD update rules)
- Social Extension: Social-Constrained User-KNN (SMM-ch9 Slide 10) & Social Regularized SVD (SMM-ch9 Slide 9)
  Controllable parameter beta in [0, 1] to isolate whether social friends provide better signals than unconnected similar users.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from src.data_loader import SocialEnvironment

class BaseRecommender:
    """Base class for recommendation models."""
    def __init__(self, env: SocialEnvironment):
        self.env = env
        
    def fit(self):
        raise NotImplementedError
        
    def predict(self, u_idx: int, it_idx: int) -> float:
        raise NotImplementedError
        
    def recommend(
        self,
        u_idx: int,
        k: int = 10,
        candidate_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generates Top-K ranked recommendations for unseen items.
        Filters out items the user has already rated in training.
        """
        rated_items = set(self.env.train_ratings.get(u_idx, {}).keys())
        
        if candidate_items is None:
            candidate_pool = [it for it in range(self.env.num_items) if it not in rated_items]
        else:
            candidate_pool = [it for it in candidate_items if it not in rated_items]
            
        scores = []
        for it in candidate_pool:
            score = self.predict(u_idx, it)
            scores.append((it, score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ============================================================================
# Approach A: Memory-Based User-KNN (User Collaborative Filtering)
# ============================================================================
class UserKNNRecommender(BaseRecommender):
    """
    User-based Collaborative Filtering implemented strictly from scratch.
    Computes Pearson / Cosine similarity between users over co-rated items.
    """
    def __init__(self, env: SocialEnvironment, k_neighbors: int = 20, min_co_rated: int = 2):
        super().__init__(env)
        self.k_neighbors = k_neighbors
        self.min_co_rated = min_co_rated
        # Cache for user similarities: (u, v) -> similarity
        self.sim_cache: Dict[Tuple[int, int], float] = {}

    def compute_similarity(self, u: int, v: int) -> float:
        """
        Computes Pearson correlation (mean-centered cosine) on co-rated items.
        Slide Justification: Recommendation System-1 Slide 23.
        """
        if u == v:
            return 1.0
        cache_key = (min(u, v), max(u, v))
        if cache_key in self.sim_cache:
            return self.sim_cache[cache_key]
            
        ratings_u = self.env.train_ratings.get(u, {})
        ratings_v = self.env.train_ratings.get(v, {})
        
        common_items = set(ratings_u.keys()).intersection(set(ratings_v.keys()))
        if len(common_items) < self.min_co_rated:
            self.sim_cache[cache_key] = 0.0
            return 0.0
            
        mean_u = self.env.user_means.get(u, self.env.global_mean)
        mean_v = self.env.user_means.get(v, self.env.global_mean)
        
        num = 0.0
        den_u = 0.0
        den_v = 0.0
        for it in common_items:
            diff_u = ratings_u[it] - mean_u
            diff_v = ratings_v[it] - mean_v
            num += diff_u * diff_v
            den_u += diff_u * diff_u
            den_v += diff_v * diff_v
            
        if den_u <= 1e-9 or den_v <= 1e-9:
            sim = 0.0
        else:
            sim = num / (math.sqrt(den_u) * math.sqrt(den_v))
            # Clip to [-1, 1]
            sim = max(-1.0, min(1.0, sim))
            
        self.sim_cache[cache_key] = sim
        return sim

    def fit(self):
        """Precomputes nothing heavy eagerly, builds lazily or on-demand for efficiency."""
        pass

    def predict(self, u_idx: int, it_idx: int) -> float:
        """
        Exact weighted average formula from Recommendation System-1 Slide 25:
        r_hat(u,i) = r_bar_u + sum_{v in N_i(u)} (sim(u,v) * (r_v,i - r_bar_v)) / sum_{v in N_i(u)} |sim(u,v)|
        """
        u_mean = self.env.user_means.get(u_idx, self.env.global_mean)
        users_who_rated = self.env.item_users_train.get(it_idx, {})
        
        if not users_who_rated:
            # Fallback to user mean or item mean
            return u_mean
            
        # Collect candidate neighbors who rated item i
        candidates = []
        for v_idx, r_vi in users_who_rated.items():
            if v_idx == u_idx:
                continue
            sim = self.compute_similarity(u_idx, v_idx)
            if sim > 0: # positive correlation
                candidates.append((v_idx, sim, r_vi))
                
        if not candidates:
            return u_mean
            
        # Top-K most similar neighbors
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k_neighbors = candidates[:self.k_neighbors]
        
        num = 0.0
        den = 0.0
        for v_idx, sim, r_vi in top_k_neighbors:
            v_mean = self.env.user_means.get(v_idx, self.env.global_mean)
            num += sim * (r_vi - v_mean)
            den += abs(sim)
            
        if den <= 1e-9:
            pred = u_mean
        else:
            pred = u_mean + (num / den)
            
        # Clip to valid explicit rating bounds [1.0, 5.0]
        return float(np.clip(pred, 1.0, 5.0))


# ============================================================================
# Approach B: Memory-Based Item-KNN (Item Collaborative Filtering)
# ============================================================================
class ItemKNNRecommender(BaseRecommender):
    """
    Item-based Collaborative Filtering implemented strictly from scratch.
    Slide Justification: Recommendation System-1 Slide 21 & 28-29.
    Predicts rating based on user's ratings for items similar to target item i:
    r_hat(u,i) = r_bar_i + sum_{j in N_u(i)} (sim(i,j) * (r_u,j - r_bar_j)) / sum_{j in N_u(i)} |sim(i,j)|
    """
    def __init__(self, env: SocialEnvironment, k_neighbors: int = 20, min_co_users: int = 2):
        super().__init__(env)
        self.k_neighbors = k_neighbors
        self.min_co_users = min_co_users
        self.sim_cache: Dict[Tuple[int, int], float] = {}

    def compute_similarity(self, i: int, j: int) -> float:
        """Computes Adjusted Cosine similarity between item i and item j."""
        if i == j:
            return 1.0
        cache_key = (min(i, j), max(i, j))
        if cache_key in self.sim_cache:
            return self.sim_cache[cache_key]
            
        users_i = self.env.item_users_train.get(i, {})
        users_j = self.env.item_users_train.get(j, {})
        
        common_users = set(users_i.keys()).intersection(set(users_j.keys()))
        if len(common_users) < self.min_co_users:
            self.sim_cache[cache_key] = 0.0
            return 0.0
            
        num = 0.0
        den_i = 0.0
        den_j = 0.0
        for u in common_users:
            u_mean = self.env.user_means.get(u, self.env.global_mean)
            diff_i = users_i[u] - u_mean
            diff_j = users_j[u] - u_mean
            num += diff_i * diff_j
            den_i += diff_i * diff_i
            den_j += diff_j * diff_j
            
        if den_i <= 1e-9 or den_j <= 1e-9:
            sim = 0.0
        else:
            sim = num / (math.sqrt(den_i) * math.sqrt(den_j))
            sim = max(-1.0, min(1.0, sim))
            
        self.sim_cache[cache_key] = sim
        return sim

    def fit(self):
        pass

    def predict(self, u_idx: int, it_idx: int) -> float:
        it_mean = self.env.item_means.get(it_idx, self.env.global_mean)
        items_rated_by_u = self.env.train_ratings.get(u_idx, {})
        
        if not items_rated_by_u:
            return it_mean
            
        candidates = []
        for j_idx, r_uj in items_rated_by_u.items():
            if j_idx == it_idx:
                continue
            sim = self.compute_similarity(it_idx, j_idx)
            if sim > 0:
                candidates.append((j_idx, sim, r_uj))
                
        if not candidates:
            return self.env.user_means.get(u_idx, it_mean)
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k_items = candidates[:self.k_neighbors]
        
        num = 0.0
        den = 0.0
        for j_idx, sim, r_uj in top_k_items:
            j_mean = self.env.item_means.get(j_idx, self.env.global_mean)
            num += sim * (r_uj - j_mean)
            den += abs(sim)
            
        if den <= 1e-9:
            pred = it_mean
        else:
            pred = it_mean + (num / den)
            
        return float(np.clip(pred, 1.0, 5.0))


# ============================================================================
# Requirement 3: Social-Constrained User-KNN (Blending Rating Sim & Social Network)
# ============================================================================
class SocialUserKNNRecommender(BaseRecommender):
    """
    Extends User-KNN to incorporate the Social Network.
    Slide Justification: SMM-ch9 Slide 10 ("Recommendation Constrained by Social Context").
    
    Weights neighbors by blending Rating Similarity and Social Trust:
    w(u, v) = (1 - beta) * sim_rating(u, v) + beta * Trust(u, v)
    
    When beta = 0.0: Pure Rating-based Collaborative Filtering (unconnected similar users).
    When beta = 1.0: Pure Social Trust Recommendation (strictly constrained to trusted friends).
    By varying beta in [0, 1], we can experimentally isolate and demonstrate when social
    ties help vs. when they make recommendations worse.
    """
    def __init__(
        self,
        env: SocialEnvironment,
        beta: float = 0.5,
        k_neighbors: int = 20,
        min_co_rated: int = 2
    ):
        super().__init__(env)
        self.beta = float(beta)
        self.k_neighbors = k_neighbors
        self.min_co_rated = min_co_rated
        self.base_knn = UserKNNRecommender(env, k_neighbors=k_neighbors, min_co_rated=min_co_rated)

    def fit(self):
        pass

    def predict(self, u_idx: int, it_idx: int) -> float:
        u_mean = self.env.user_means.get(u_idx, self.env.global_mean)
        users_who_rated = self.env.item_users_train.get(it_idx, {})
        
        if not users_who_rated:
            return u_mean
            
        trusted_friends = self.env.trust_out.get(u_idx, set())
        
        candidates = []
        for v_idx, r_vi in users_who_rated.items():
            if v_idx == u_idx:
                continue
                
            sim_rating = self.base_knn.compute_similarity(u_idx, v_idx)
            is_friend = 1.0 if v_idx in trusted_friends else 0.0
            
            # Blended neighbor weight
            # When beta = 0: relies purely on sim_rating
            # When beta = 1: relies purely on friends who rated the item
            blended_weight = (1.0 - self.beta) * max(0.0, sim_rating) + self.beta * is_friend
            
            if blended_weight > 0:
                candidates.append((v_idx, blended_weight, r_vi))
                
        if not candidates:
            return u_mean
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = candidates[:self.k_neighbors]
        
        num = 0.0
        den = 0.0
        for v_idx, weight, r_vi in top_k:
            v_mean = self.env.user_means.get(v_idx, self.env.global_mean)
            num += weight * (r_vi - v_mean)
            den += weight
            
        if den <= 1e-9:
            pred = u_mean
        else:
            pred = u_mean + (num / den)
            
        return float(np.clip(pred, 1.0, 5.0))


# ============================================================================
# Approach C: Latent Factor Model (SVD & Social Regularized SVD via manual SGD)
# ============================================================================
class MatrixFactorizationRecommender(BaseRecommender):
    """
    Model-based Latent Factor Model (SVD / Funk-SVD with manual SGD update rules).
    Also supports Social Regularization (SMM-ch9 Slide 9).
    
    Prediction:
    r_hat(u, i) = mu + b_u + b_i + P_u . Q_i^T
    
    Manual SGD Update Rules (No external blackbox library):
    e_{u,i} = r_{u,i} - r_hat(u,i)
    b_u <- b_u + eta * (e_{u,i} - lambda * b_u)
    b_i <- b_i + eta * (e_{u,i} - lambda * b_i)
    P_u <- P_u + eta * (e_{u,i} * Q_i - lambda * P_u - beta_social * sum_{v in Friends(u)}(P_u - P_v))
    Q_i <- Q_i + eta * (e_{u,i} * P_u - lambda * Q_i)
    """
    def __init__(
        self,
        env: SocialEnvironment,
        n_factors: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        epochs: int = 25,
        social_beta: float = 0.0, # 0.0 = Standard SVD, >0.0 = Social SVD
        seed: int = 42
    ):
        super().__init__(env)
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.social_beta = social_beta
        self.seed = seed
        
        # Parameters
        self.mu: float = 0.0
        self.b_u = np.zeros(self.env.num_users)
        self.b_i = np.zeros(self.env.num_items)
        self.P = np.zeros((self.env.num_users, self.n_factors))
        self.Q = np.zeros((self.env.num_items, self.n_factors))

    def fit(self, verbose: bool = False):
        rng = np.random.default_rng(self.seed)
        self.mu = self.env.global_mean
        self.b_u = np.zeros(self.env.num_users)
        self.b_i = np.zeros(self.env.num_items)
        
        # Initialize latent factor matrices with small Gaussian noise
        self.P = rng.normal(0, 0.1, (self.env.num_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (self.env.num_items, self.n_factors))
        
        # Collect all train triplets
        samples = []
        for u_idx, items in self.env.train_ratings.items():
            for it_idx, r in items.items():
                samples.append((u_idx, it_idx, r))
                
        for epoch in range(self.epochs):
            rng.shuffle(samples)
            total_loss = 0.0
            
            for u, i, r in samples:
                # Prediction
                dot_prod = np.dot(self.P[u], self.Q[i])
                pred = self.mu + self.b_u[u] + self.b_i[i] + dot_prod
                err = r - pred
                total_loss += err * err
                
                # Update Biases
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                
                # Social Regularization Gradient term: sum_{v in Friends(u)} (P_u - P_v)
                social_grad = np.zeros(self.n_factors)
                if self.social_beta > 0:
                    friends = self.env.trust_out.get(u, set())
                    if friends:
                        for v in friends:
                            social_grad += (self.P[u] - self.P[v])
                        social_grad = social_grad / len(friends)
                        
                # Update Latent Factors
                p_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u] - self.social_beta * social_grad)
                self.Q[i] += self.lr * (err * p_u_old - self.reg * self.Q[i])
                
            if verbose and (epoch + 1) % 5 == 0:
                rmse = math.sqrt(total_loss / len(samples))
                print(f"      [SVD Epoch {epoch+1}/{self.epochs}] Train RMSE: {rmse:.4f}")

    def predict(self, u_idx: int, it_idx: int) -> float:
        if u_idx >= self.env.num_users or it_idx >= self.env.num_items:
            return self.env.global_mean
            
        pred = self.mu + self.b_u[u_idx] + self.b_i[it_idx] + np.dot(self.P[u_idx], self.Q[it_idx])
        return float(np.clip(pred, 1.0, 5.0))
