"""
Information Diffusion and Epidemic Adoption module for SNTP Assignment 2.
Implements Requirement 4:
- Distinguishes Receiving Recommendation (Seed), Becoming Exposed, and Actually Adopting.
- Independent Cascade (IC) Model (SMM-ch7, Slides 17-19)
- Epidemic SIR Model (SMM-ch7, Slides 64-67)
- Integrates Recommender Preference Signals into Adoption Decisions.
- Repeated Monte Carlo simulations reporting mean and variability.
"""

import random
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from src.data_loader import SocialEnvironment

class DiffusionSimulator:
    """
    Simulates information spread, social influence, and item adoption on the trust graph.
    """
    def __init__(self, env: SocialEnvironment):
        self.env = env
        
    def simulate_ic(
        self,
        seed_set: Set[int],
        item_id: Optional[int] = None,
        recommender = None,
        base_p: float = 0.12,
        pref_weighted: bool = False,
        pref_threshold: float = 3.5,
        max_steps: int = 30,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, any]:
        """
        Independent Cascade (IC) Model (Slide Justification: SMM-ch7 Slides 17-19).
        
        States:
        - Inactive: has not adopted.
        - Exposed: received message/recommendation from active friend.
        - Active (Adopted): has adopted item and attempts to activate neighbors once.
        
        Distinction:
        - Seeds: Received direct recommendation from the platform.
        - Exposed: Unactive nodes contacted by newly active neighbors.
        - Adopted: Contacted nodes who actually adopt (based on influence probability
                   and optionally preference threshold from recommender).
        """
        if rng is None:
            rng = np.random.default_rng()
            
        active_nodes: Set[int] = set(seed_set)
        newly_active: Set[int] = set(seed_set)
        exposed_nodes: Set[int] = set(seed_set)
        
        adoption_history: List[int] = [len(active_nodes)]
        newly_active_history: List[int] = [len(seed_set)]
        step = 0
        
        while newly_active and step < max_steps:
            step += 1
            next_active: Set[int] = set()
            
            for u in newly_active:
                # Information flows along trust edges: u is trusted by / influences its out-neighbors
                # In Epinions: trust_out[u] are users whom u trusts.
                # In social cascades, a user's action influences people who trust them (trust_in[u])
                # or their direct trust contacts. We consider outgoing influence edges.
                neighbors = self.env.trust_out.get(u, set())
                
                for v in neighbors:
                    if v not in active_nodes:
                        exposed_nodes.add(v)
                        
                        # Transmission probability
                        p = base_p
                        
                        # Preference weighting if recommender is attached
                        if pref_weighted and recommender is not None and item_id is not None:
                            pred_rating = recommender.predict(v, item_id)
                            if pred_rating < pref_threshold:
                                # Low preference reduces adoption likelihood
                                p *= max(0.1, (pred_rating / 5.0))
                            else:
                                # High preference boosts adoption likelihood
                                p = min(1.0, p * (pred_rating / 3.0))
                                
                        if rng.random() < p:
                            next_active.add(v)
                            
            newly_active = next_active
            active_nodes.update(next_active)
            adoption_history.append(len(active_nodes))
            newly_active_history.append(len(next_active))
            
        return {
            "model": "Independent Cascade",
            "seed_size": len(seed_set),
            "final_adopted": len(active_nodes),
            "total_exposed": len(exposed_nodes),
            "adoption_fraction": len(active_nodes) / max(1, self.env.num_users),
            "exposure_fraction": len(exposed_nodes) / max(1, self.env.num_users),
            "steps_to_quiescence": step,
            "adoption_history": adoption_history,
            "newly_active_history": newly_active_history,
            "active_node_set": active_nodes
        }

    def simulate_sir(
        self,
        seed_set: Set[int],
        item_id: Optional[int] = None,
        recommender = None,
        beta: float = 0.18,      # Infection rate (S -> I)
        gamma: float = 0.08,     # Recovery rate (I -> R)
        pref_weighted: bool = False,
        pref_threshold: float = 3.5,
        max_steps: int = 30,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, any]:
        """
        SIR Epidemic Model (Slide Justification: SMM-ch7 Slides 64-67).
        
        States:
        - S (Susceptible): Unaware or has not adopted, open to infection/adoption.
        - I (Infected/Adopted): Currently using/promoting the item. Repeatedly attempts
                               to infect susceptible contacts at every step.
        - R (Recovered/Removed): Ceased active promotion/usage (bored/churned), immune to reinfection.
        """
        if rng is None:
            rng = np.random.default_rng()
            
        all_nodes = set(range(self.env.num_users))
        I: Set[int] = set(seed_set)
        R: Set[int] = set()
        S: Set[int] = all_nodes - I
        
        exposed_nodes: Set[int] = set(seed_set)
        
        s_history = [len(S)]
        i_history = [len(I)]
        r_history = [len(R)]
        
        step = 0
        while I and step < max_steps:
            step += 1
            new_infections: Set[int] = set()
            new_recoveries: Set[int] = set()
            
            # 1. Infection phase
            for u in I:
                neighbors = self.env.trust_out.get(u, set())
                for v in neighbors:
                    if v in S:
                        exposed_nodes.add(v)
                        inf_prob = beta
                        if pref_weighted and recommender is not None and item_id is not None:
                            pred_rating = recommender.predict(v, item_id)
                            if pred_rating < pref_threshold:
                                inf_prob *= max(0.1, (pred_rating / 5.0))
                            else:
                                inf_prob = min(1.0, inf_prob * (pred_rating / 3.0))
                                
                        if rng.random() < inf_prob:
                            new_infections.add(v)
                            
            # 2. Recovery phase
            for u in I:
                if rng.random() < gamma:
                    new_recoveries.add(u)
                    
            # Update compartments
            S -= new_infections
            I = (I - new_recoveries).union(new_infections)
            R.update(new_recoveries)
            
            s_history.append(len(S))
            i_history.append(len(I))
            r_history.append(len(R))
            
        total_ever_infected = len(I) + len(R)
        return {
            "model": "SIR Epidemic",
            "seed_size": len(seed_set),
            "final_ever_infected": total_ever_infected,
            "final_recovered": len(R),
            "total_exposed": len(exposed_nodes),
            "adoption_fraction": total_ever_infected / max(1, self.env.num_users),
            "exposure_fraction": len(exposed_nodes) / max(1, self.env.num_users),
            "peak_infected": max(i_history),
            "time_to_peak": int(np.argmax(i_history)),
            "s_history": s_history,
            "i_history": i_history,
            "r_history": r_history
        }

    def run_monte_carlo_ic(
        self,
        seed_set: Set[int],
        num_trials: int = 15,
        base_p: float = 0.12,
        item_id: Optional[int] = None,
        recommender = None,
        pref_weighted: bool = False,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Runs repeated Monte Carlo trials of IC to estimate expected spread and variability.
        Prevents relying on a single simulation run.
        """
        spreads = []
        exposures = []
        steps_list = []
        
        for trial in range(num_trials):
            rng = np.random.default_rng(seed + trial * 1007)
            res = self.simulate_ic(
                seed_set=seed_set,
                item_id=item_id,
                recommender=recommender,
                base_p=base_p,
                pref_weighted=pref_weighted,
                rng=rng
            )
            spreads.append(res["final_adopted"])
            exposures.append(res["total_exposed"])
            steps_list.append(res["steps_to_quiescence"])
            
        return {
            "mean_spread": float(np.mean(spreads)),
            "std_spread": float(np.std(spreads)),
            "mean_exposure": float(np.mean(exposures)),
            "std_exposure": float(np.std(exposures)),
            "mean_steps": float(np.mean(steps_list)),
            "min_spread": int(np.min(spreads)),
            "max_spread": int(np.max(spreads))
        }
