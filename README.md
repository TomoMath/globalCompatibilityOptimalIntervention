# Replication code: "Global Compatibility of Optimal Interventions" (revision)

Python 3 with numpy, scipy, matplotlib.

| File | Produces |
|---|---|
| `tri.py` | Core routines: three-agent network, welfare matrix, exact solver of the planner's problem (regular and hard case), tracking of the lowest principal component, zeros of the status-quo projection and critical radii |
| `table_cstar.py` | Thresholds by loop radius and heterogeneity (Figure 2a, Section 7.4) -> `cstar_table.json` |
| `cost.py` | Cost of continuity, lower bound and constructed rule (Sections 2 and 7.5) -> `cost.json` |
| `figs.py` | Figures 1 and 2 (`fig_path.pdf`, `fig_threshold_cost.pdf`); run after `table_cstar.py` |
| `check_loop.py` | Exact loop threshold with three neutral networks (Theorem 5.6) |
| `audit74.py` | Stress tests of Proposition 7.4 (random status quos, large common level, near the stability boundary) |
| `checks2.py` | Asymptotic threshold formula (Proposition 7.4) and robustness to asymmetric links (Section 8) |

Run order: `python3 table_cstar.py; python3 cost.py; python3 figs.py; python3 check_loop.py; python3 checks2.py`.
`figs.py` writes the figures to the parent folder.
