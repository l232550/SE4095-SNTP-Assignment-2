"""
Master Experiment Runner & PDF Report Builder.
SNTP Assignment 2: Recommendation & Information Diffusion in a Social Network.
Student Roll Number: 23L-2550

Executes all 4 experiment modules sequentially:
1. run_recs.py (Req 1, 2, 3)
2. run_diffusion.py (Req 4, 6)
3. run_influence.py (Req 5)
4. run_intervention.py (Req 8)
Followed by compiling the complete searchable PDF report!
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from experiments.run_recs import run_recommendation_experiments
from experiments.run_diffusion import run_diffusion_experiments
from experiments.run_influence import run_influence_experiments
from experiments.run_intervention import run_intervention_experiments
from generate_report import build_pdf_report

def run_all_experiments():
    start_total = time.time()
    print("=" * 80)
    print("      MASTER EXPERIMENT SUITE - SNTP ASSIGNMENT 2 (ROLL NO: 23L-2550)")
    print("=" * 80)
    
    # Experiment 1
    t0 = time.time()
    print("\n>>> Launching Experiment 1 (Requirements 1, 2, 3)...")
    run_recommendation_experiments(max_users=300)
    print(f">>> Experiment 1 completed in {time.time() - t0:.2f} seconds.")
    
    # Experiment 2
    t1 = time.time()
    print("\n>>> Launching Experiment 2 (Requirements 4, 6)...")
    run_diffusion_experiments(max_users=300)
    print(f">>> Experiment 2 completed in {time.time() - t1:.2f} seconds.")
    
    # Experiment 3
    t2 = time.time()
    print("\n>>> Launching Experiment 3 (Requirement 5)...")
    run_influence_experiments(max_users=300)
    print(f">>> Experiment 3 completed in {time.time() - t2:.2f} seconds.")
    
    # Experiment 4
    t3 = time.time()
    print("\n>>> Launching Experiment 4 (Requirement 8)...")
    run_intervention_experiments(max_users=250)
    print(f">>> Experiment 4 completed in {time.time() - t3:.2f} seconds.")
    
    # Compile PDF Report
    print("\n>>> Generating Final Academic Searchable PDF Report...")
    build_pdf_report()
    
    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"[+] ALL EXPERIMENTS & PDF REPORT COMPLETED IN {total_time:.2f} SECONDS (~{total_time/60:.1f} MIN)!")
    print(f"[+] Results saved in: {BASE_DIR / 'results'}")
    print(f"[+] Final PDF Report: {BASE_DIR / '23L2550_SNTP_Assignment2_Report.pdf'}")
    print("=" * 80)

if __name__ == "__main__":
    run_all_experiments()
