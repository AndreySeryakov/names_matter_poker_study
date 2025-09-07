#!/usr/bin/env python3
from __future__ import annotations
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

ERRORBAR_MULTIPLIER = 1.0
OUTPUT_ROOT = Path(__file__).resolve().parent / "strategy_results"

@dataclass
class PlayerStats:
    name: str
    n_hands: int
    n_folds: int
    n_calls: int
    n_raises: int
    n_raises_ge_thresh: int
    color: Optional[str] = None  # optional (e.g. 'red' or '#FF0000')


ROW_REGEX = re.compile(
    r"""^\s*
        (?P<name>[^,]+?)\s*,?\s*      # name (comma after name optional)
        (?P<n_hands>\d+)\s*,\s*
        (?P<n_folds>\d+)\s*,\s*
        (?P<n_calls>\d+)\s*,\s*
        (?P<n_raises>\d+)\s*,\s*
        (?P<n_bigraises>\d+)
        (?:\s*,\s*(?P<color>\#[0-9A-Fa-f]{6}|[A-Za-z]+))?   # optional color
        $""",
    re.VERBOSE
)

def _logit_delta_se_p(k: float, n: float) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p_hat = k / n
    k_adj = k + 0.5
    n_adj = n + 1.0
    if k_adj <= 0:
        k_adj = 1e-9
    if n_adj - k_adj <= 0:
        n_adj = k_adj + 1e-9
    se_logit = (1.0 / k_adj + 1.0 / (n_adj - k_adj)) ** 0.5
    se_p = p_hat * (1.0 - p_hat) * se_logit
    return p_hat, se_p

def parse_row_to_stats(row_str: str, ln: int) -> PlayerStats:
    m = ROW_REGEX.match(row_str.strip())
    if not m:
        raise ValueError(
            f"Line {ln}: could not parse row. Expected 'name, N_hands, N_folds, N_calls, N_raises, N_raises_ge_threshold, color'. Got: {row_str!r}"
        )
    name = m.group("name").strip().rstrip(",")
    n_hands = int(m.group("n_hands"))
    n_folds = int(m.group("n_folds"))
    n_calls = int(m.group("n_calls"))
    n_raises = int(m.group("n_raises"))
    n_big = int(m.group("n_bigraises"))
    color = m.group("color")
    return PlayerStats(name, n_hands, n_folds, n_calls, n_raises, n_big, color=color)

def parse_config_file(path: Path) -> Tuple[int, List[PlayerStats], str, Optional[PlayerStats], Optional[PlayerStats], Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    
    threshold_bb: Optional[int] = None
    title_suffix: str = ""
    baseline: Optional[PlayerStats] = None
    baseline2: Optional[PlayerStats] = None
    players: List[PlayerStats] = []
    ylims: Dict[str, Tuple[float, float]] = {}  # per-plot limits for probability plots
    zlims: Dict[str, Tuple[float, float]] = {}  # per-plot limits for z-score plots
    
    with path.open("r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip().lower()
                val = val.strip()
                if key in ("threshold_bb", "threshold", "thresh", "t"):
                    try:
                        threshold_bb = int(val)
                    except ValueError:
                        raise ValueError(f"Line {ln}: invalid threshold integer: {val!r}")
                    continue
                if key in ("baseline", "base"):
                    baseline = parse_row_to_stats(val, ln)
                    continue
                if key in ("baseline2", "base2"):
                    baseline2 = parse_row_to_stats(val, ln)
                    continue
                if key in ("title_suffix", "title"):
                    title_suffix = val
                    continue
                    
                # Handle probability plot ylims
                if key in ("ylim_folds", "ylim_calls", "ylim_raises", "ylim_raises_ge"):
                    parts = [p for p in re.split(r"[,\s]+", val) if p]
                    if len(parts) != 2:
                        raise ValueError(f"Line {ln}: {key.upper()} expects two numbers, got: {val!r}")
                    try:
                        lo, hi = float(parts[0]), float(parts[1])
                    except ValueError:
                        raise ValueError(f"Line {ln}: {key.upper()} values must be numeric, got: {val!r}")
                    if lo > hi:
                        lo, hi = hi, lo
                    lo = max(0.0, lo); hi = min(1.0, hi)

                    if key.endswith("ylim_folds") or key == "ylim_folds":
                        ylims["folds"] = (lo, hi)
                    elif key.endswith("ylim_calls") or key == "ylim_calls":
                        ylims["calls"] = (lo, hi)
                    elif key.endswith("ylim_raises") or key == "ylim_raises":
                        ylims["raises"] = (lo, hi)
                    else:  # ylim_raises_ge
                        ylims["raises_ge"] = (lo, hi)
                    continue
                    
                # Handle z-score plot ylims
                if key in ("ylim_zscore_folds", "ylim_zscore_calls", "ylim_zscore_raises", "ylim_zscore_raises_ge"):
                    parts = [p for p in re.split(r"[,\s]+", val) if p]
                    if len(parts) != 2:
                        raise ValueError(f"Line {ln}: {key.upper()} expects two numbers, got: {val!r}")
                    try:
                        lo, hi = float(parts[0]), float(parts[1])
                    except ValueError:
                        raise ValueError(f"Line {ln}: {key.upper()} values must be numeric, got: {val!r}")
                    if lo > hi:
                        lo, hi = hi, lo
                    
                    if "ylim_zscore_folds" in key:
                        zlims["folds"] = (lo, hi)
                    elif "ylim_zscore_calls" in key:
                        zlims["calls"] = (lo, hi)
                    elif "ylim_zscore_raises_ge" in key:
                        zlims["raises_ge"] = (lo, hi)
                    elif "ylim_zscore_raises" in key:
                        zlims["raises"] = (lo, hi)
                    continue
                    
            players.append(parse_row_to_stats(line, ln))

    if threshold_bb is None:
        raise ValueError("No THRESHOLD_BB provided in the config file.")
    if not players:
        raise ValueError("No player rows found in the config file.")
    return threshold_bb, players, title_suffix, baseline, baseline2, ylims, zlims

def compute_baseline_metrics(baseline: Optional[PlayerStats], threshold_bb: int):
    if baseline is None:
        return None
    metrics = {}
    for label, k in [
        ("folds", baseline.n_folds),
        ("calls", baseline.n_calls),
        ("raises", baseline.n_raises),
        (f"raises_ge_{threshold_bb}bb", baseline.n_raises_ge_thresh),
    ]:
        p_hat, se = _logit_delta_se_p(k, baseline.n_hands)
        metrics[label] = {"p": p_hat, "se": se}
    return metrics

def compute_metrics(players: List[PlayerStats], threshold_bb: int):
    results = []
    for pl in players:
        metrics = {}
        for label, k in [
            ("folds", pl.n_folds),
            ("calls", pl.n_calls),
            ("raises", pl.n_raises),
            (f"raises_ge_{threshold_bb}bb", pl.n_raises_ge_thresh),
        ]:
            p_hat, se = _logit_delta_se_p(k, pl.n_hands)
            metrics[label] = {"p": p_hat, "se": se}
        results.append((pl, metrics))
    return results

def compute_z_scores(results, baseline_metrics, threshold_bb):
    """
    Compute z-scores for each player and metric relative to baseline.
    Returns a dictionary mapping metric labels to lists of z-scores.
    """
    if baseline_metrics is None:
        return None
    
    z_scores = {}
    for label in ["folds", "calls", "raises", f"raises_ge_{threshold_bb}bb"]:
        z_list = []
        p_base = baseline_metrics[label]["p"]
        se_base = baseline_metrics[label]["se"]
        
        for pl, metrics in results:
            p_player = metrics[label]["p"]
            se_player = metrics[label]["se"]
            
            # Combined standard error
            se_combined = np.sqrt(se_player**2 + se_base**2)
            
            # Z-score
            if se_combined > 0:
                z = (p_player - p_base) / se_combined
            else:
                z = 0.0
            
            z_list.append(z)
        
        z_scores[label] = z_list
    
    return z_scores

def save_csv(results, out_dir: Path, threshold_bb: int, title_suffix: str, z_scores=None):
    out_csv = out_dir / "metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "name", "n_hands", "n_folds", "n_calls", "n_raises", f"n_raises_ge_{threshold_bb}bb",
            "p_folds", "se_folds",
            "p_calls", "se_calls",
            "p_raises", "se_raises",
            f"p_raises_ge_{threshold_bb}bb", f"se_raises_ge_{threshold_bb}bb",
        ]
        
        # Add z-score columns if available
        if z_scores is not None:
            header.extend([
                "zscore_folds", "zscore_calls", "zscore_raises", f"zscore_raises_ge_{threshold_bb}bb"
            ])
        
        header.extend(["threshold_bb", "title_suffix", "timestamp"])
        w.writerow(header)
        
        timestamp = datetime.now().isoformat(timespec="seconds")
        for idx, (pl, m) in enumerate(results):
            row = [
                pl.name, pl.n_hands, pl.n_folds, pl.n_calls, pl.n_raises, pl.n_raises_ge_thresh,
                m["folds"]["p"], m["folds"]["se"],
                m["calls"]["p"], m["calls"]["se"],
                m["raises"]["p"], m["raises"]["se"],
                m[f"raises_ge_{threshold_bb}bb"]["p"], m[f"raises_ge_{threshold_bb}bb"]["se"],
            ]
            
            # Add z-scores if available
            if z_scores is not None:
                row.extend([
                    z_scores["folds"][idx],
                    z_scores["calls"][idx],
                    z_scores["raises"][idx],
                    z_scores[f"raises_ge_{threshold_bb}bb"][idx],
                ])
            
            row.extend([threshold_bb, title_suffix, timestamp])
            w.writerow(row)
    return out_csv

def make_plot(names, ps, ses, title_prefix: str, title_suffix: str, out_path: Path,
              baseline_metrics=None, metric_key: str = "", threshold_bb: int = 0,
              ylim: Optional[Tuple[float, float]] = None, colors: Optional[List[Optional[str]]] = None):    
    
    # Map each unique name to a single x position
    name_to_x: dict[str, int] = {}
    unique_names: list[str] = []
    for nm in names:
        if nm not in name_to_x:
            name_to_x[nm] = len(unique_names)
            unique_names.append(nm)

    x_positions = [name_to_x[nm] for nm in names]

    plt.figure(figsize=(8, 8))

    if colors is None:
        colors = [None] * len(names)

    for xi, pi, sei, ci in zip(x_positions, ps, ses, colors):
        col = ci if ci else "C0"
        plt.errorbar([xi], [pi], yerr=ERRORBAR_MULTIPLIER * sei,
                    fmt='o', capsize=4, linewidth=1.5, color=col)

    plt.xticks(range(len(unique_names)), unique_names, rotation=0, 
    fontsize=12)
    plt.xlim(-0.5, len(unique_names) - 0.5)
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    else:
        plt.ylim(0, 1)
    plt.ylabel("Probability p", fontsize=16)
    plt.yticks(fontsize=14)
    plt.title(f"{title_prefix}{title_suffix}", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.4)

    # Baseline lines
    if baseline_metrics is not None:
        if metric_key in baseline_metrics:
            m = baseline_metrics[metric_key]
            p = m["p"]
            se = m["se"] * ERRORBAR_MULTIPLIER
            plt.axhline(y=p, color="blue", linewidth=2.0)
            plt.axhline(y=max(0.0, p - se), color="blue", linewidth=1.0, linestyle="--")
            plt.axhline(y=min(1.0, p + se), color="blue", linewidth=1.0, linestyle="--")

        # Second baseline (if provided)
        key2 = metric_key + "_2"
        if key2 in baseline_metrics:
            m2 = baseline_metrics[key2]
            p2 = m2["p"]
            se2 = m2["se"] * ERRORBAR_MULTIPLIER
            plt.axhline(y=p2, color="red", linewidth=2.0)
            plt.axhline(y=max(0.0, p2 - se2), color="red", linewidth=1.0, linestyle=":")
            plt.axhline(y=min(1.0, p2 + se2), color="red", linewidth=1.0, linestyle=":")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def make_zscore_plot(names, z_scores, title_prefix: str, title_suffix: str, out_path: Path,
                     ylim: Optional[Tuple[float, float]] = None, 
                     colors: Optional[List[Optional[str]]] = None):
    """
    Create a z-score plot showing standardized differences from baseline.
    """
    # Map each unique name to a single x position
    name_to_x: dict[str, int] = {}
    unique_names: list[str] = []
    for nm in names:
        if nm not in name_to_x:
            name_to_x[nm] = len(unique_names)
            unique_names.append(nm)

    x_positions = [name_to_x[nm] for nm in names]

    plt.figure(figsize=(8, 8))

    if colors is None:
        colors = [None] * len(names)

    # Plot z-scores as points (no error bars since z-score already incorporates uncertainty)
    for xi, zi, ci in zip(x_positions, z_scores, colors):
        col = ci if ci else "C0"
        plt.plot([xi], [zi], 'o', markersize=8, color=col)

    plt.xticks(range(len(unique_names)), unique_names, rotation=0, fontsize=12)
    plt.xlim(-0.5, len(unique_names) - 0.5)
    
    # Set y-axis limits
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    else:
        # Auto-scale with minimum range
        max_abs_z = max(abs(min(z_scores)), abs(max(z_scores))) if z_scores else 3
        y_range = max(3, max_abs_z * 1.2)
        plt.ylim(-y_range, y_range)
    
    plt.ylabel("Standardized Difference (z-score)", fontsize=16)
    plt.yticks(fontsize=14)
    plt.title(f"{title_prefix} (Z-scores){title_suffix}", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.4)
    
    # Add horizontal line at y=0 (baseline)
    plt.axhline(y=0, color="black", linewidth=1.5, linestyle="-")
    
    # Add significance reference lines
    plt.axhline(y=1.96, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    plt.axhline(y=-1.96, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    plt.axhline(y=2.58, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    plt.axhline(y=-2.58, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    
    # Add labels for reference lines on the right side
    ax = plt.gca()
    x_right = ax.get_xlim()[1]
    plt.text(x_right * 0.98, 1.96 + 0.01, "95% CI", ha="right", va="bottom", color="gray", fontsize=15)
    plt.text(x_right * 0.98, -1.96 + 0.01, "95% CI", ha="right", va="bottom", color="gray", fontsize=15)
    plt.text(x_right * 0.98, 2.58 + 0.01, "99% CI", ha="right", va="bottom", color="gray", fontsize=15)
    plt.text(x_right * 0.98, -2.58 + 0.01, "99% CI", ha="right", va="bottom", color="gray", fontsize=15)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def choose_ylim(
    ylims: Dict[str, Tuple[float, float]],
    metric_label: str,
    default: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float]]:
    """
    metric_label is one of: 'folds', 'calls', 'raises', or f'raises_ge_{threshold_bb}bb'.
    Return a per-plot (lo, hi) if configured; otherwise return `default` (usually None).
    """
    key = "raises_ge" if metric_label.startswith("raises_ge_") else metric_label
    return ylims.get(key, default)


def plot_all(results, out_dir: Path, threshold_bb: int, title_suffix: str, 
             baseline_metrics, ylims: Dict[str, Tuple[float, float]], 
             z_scores, zlims: Dict[str, Tuple[float, float]]):
    names = [pl.name for pl, _ in results]

    for label, pretty in [
        ("folds", "Fold probability"),
        ("calls", "Call probability"),
        ("raises", "Raise probability"),
        (f"raises_ge_{threshold_bb}bb", f"Raise ≥ {threshold_bb}bb probability"),
    ]:
        ps  = [m[label]["p"]  for _, m in results]
        ses = [m[label]["se"] for _, m in results]
        colors = [pl.color for pl, _ in results]
        
        # Original probability plot
        filename = label.replace("≥", "ge").replace(" ", "_") + ".png"
        out_path = out_dir / filename
        per_plot_ylim = choose_ylim(ylims, label)
        make_plot(names, ps, ses, pretty, title_suffix, out_path,
                baseline_metrics, label, threshold_bb, per_plot_ylim, colors=colors)
        
        # Z-score plot (only if we have z_scores)
        if z_scores is not None and label in z_scores:
            z_filename = label.replace("≥", "ge").replace(" ", "_") + "_zscore.png"
            z_out_path = out_dir / z_filename
            z_plot_ylim = choose_ylim(zlims, label)
            pretty_z = pretty.replace("probability", "").strip()
            make_zscore_plot(names, z_scores[label], pretty_z, title_suffix, z_out_path,
                           z_plot_ylim, colors=colors)

def main():
    if len(sys.argv) != 2:
        print("Usage: python betting_strategy_viz.py file.txt")
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)
    
    # Parse config with zlims support
    threshold_bb, players, title_suffix, baseline, baseline2, ylims, zlims = parse_config_file(cfg_path)   
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    results = compute_metrics(players, threshold_bb)
    baseline_metrics = compute_baseline_metrics(baseline, threshold_bb)
    
    # Note: We ignore baseline2 for z-scores as requested
    baseline2_metrics = compute_baseline_metrics(baseline2, threshold_bb)
    if baseline2_metrics:
        if baseline_metrics is None:
            baseline_metrics = {}
        # Still merge for probability plots
        for k, v in baseline2_metrics.items():
            baseline_metrics[k + "_2"] = v
    
    # Compute z-scores (only if baseline exists, only using first baseline)
    z_scores = None
    if baseline is not None:
        base1_metrics = compute_baseline_metrics(baseline, threshold_bb)
        z_scores = compute_z_scores(results, base1_metrics, threshold_bb)
    
    # Save CSV with z-scores
    csv_path = save_csv(results, out_dir, threshold_bb, title_suffix, z_scores)
    
    # Create all plots
    plot_all(results, out_dir, threshold_bb, title_suffix, baseline_metrics, ylims, z_scores, zlims)
    
    print(f"Saved results to: {out_dir.resolve()}")
    print(f"CSV: {csv_path.name}")
    print("PNG plots: folds.png, calls.png, raises.png, raises_ge_*bb.png")
    if z_scores is not None:
        print("Z-score plots: folds_zscore.png, calls_zscore.png, raises_zscore.png, raises_ge_*bb_zscore.png")
    else:
        print("Z-score plots skipped (no baseline specified)")

if __name__ == "__main__":
    main()