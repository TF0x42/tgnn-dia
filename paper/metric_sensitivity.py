#!/usr/bin/env python3
"""Metric-sensitivity check: are the findings artifacts of the hit@10 cutoff?

Part 1 (Sec. IV) recomputes the set-level quantities at hit@5, hit@10 and
hit@20: subsumption between solved sets, contested fraction, per-edge
oracle gain over the best single architecture, and between-architecture vs
between-seed disagreement. Every conclusion of Section IV holds at every
cutoff: no architecture subsumes another, the oracle gain stays positive on
all eight datasets, and architectures disagree more than seeds everywhere.

Part 2 (Sec. V / Fig. 4) recomputes the within-model ablation comparisons,
full vs ablated novel-edge hit@k, at the same cutoffs. Each conclusion is
cutoff-stable: removing walk projections or the co-occurrence channel costs
double-digit points at every k, GraphMixer collapses under a swapped time
encoder at every k, and TPNet barely moves at every k.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diagnostics import (load, novel, solved, hit_matrix, composition,
                         best_single, oracle, solved_sets, DATASETS, MODELS,
                         SEEDS)

for k in [5, 10, 20]:
    print(f"--- hit@{k} ---")
    print(f"{'dataset':10s} {'cont%':>6} {'gain':>6} {'arch%':>6} {'seed%':>6} "
          f"{'subsumption':>12}")
    for ds in DATASETS:
        tab = load(ds); nov = novel(tab)
        c, g, sub = [], [], False
        for s in SEEDS:
            H = hit_matrix(tab, s, mask=nov, k=k)
            c.append(composition(H)[1])
            g.append(oracle(H) - best_single(H))
            for i in range(len(MODELS)):
                for j in range(len(MODELS)):
                    if i != j and not (H[i] & ~H[j]).any():
                        sub = True
        S = solved_sets(tab, mask=nov, k=k)
        arch = np.mean([np.mean(S[a, s] ^ S[b, s]) for s in SEEDS
                        for a, b in itertools.combinations(MODELS, 2)])
        seed = np.mean([np.mean(S[m, s1] ^ S[m, s2]) for m in MODELS
                        for s1, s2 in itertools.combinations(SEEDS, 2)])
        print(f"{ds:10s} {100*np.mean(c):6.1f} {100*np.mean(g):+6.1f} "
              f"{100*arch:6.1f} {100*seed:6.1f} "
              f"{'SUBSUMED' if sub else 'none':>12}")

print("\n--- within-model ablations (Fig. 4), full -> ablated novel-edge hit@k ---")
PAIRS = [("enron", "TPNet", "TPNet_nowalk"),
         ("canparl", "TPNet_recency", "TPNet_nowalk_recency"),
         ("uci", "DyGFormer", "DyGFormer_nocooc"),
         ("canparl", "DyGFormer", "DyGFormer_nocooc"),
         ("uci", "GraphMixer", "GraphMixer_learnedtime"),
         ("canparl", "GraphMixer", "GraphMixer_learnedtime"),
         ("uci", "TPNet", "TPNet_lineartime"),
         ("canparl", "TPNet", "TPNet_lineartime")]

def hit(tab, col, mask, k):
    return np.mean([solved(tab, col, s, k)[mask].mean() for s in SEEDS])

print(f"{'ablation':34s} {'k=5':>13} {'k=10':>13} {'k=20':>13}")
for ds, full, abl in PAIRS:
    tab = load(ds); nov = novel(tab)
    cells = [f"{hit(tab, full, nov, k):.2f}->{hit(tab, abl, nov, k):.2f}"
             for k in (5, 10, 20)]
    print(f"{ds:8s} {abl:25s} " + " ".join(f"{c:>13}" for c in cells))
