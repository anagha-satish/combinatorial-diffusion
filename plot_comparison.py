#!/usr/bin/env python3
"""
Plot comparison between DPMD-RF and random policy trajectories.

Usage:
    python plot_comparison.py --dpmd results/trajectories_ep10.csv --random results/Chlamydia_T25_B1_seed0_2026-01-07_22-35-34/trajectories.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_aggregate_trajectories(csv_path):
    """
    Load trajectories CSV and compute mean/std of detection curves.

    Returns:
        mean_x: mean fraction tested at each step
        mean_y: mean fraction detected at each step
        std_y: std of fraction detected at each step
    """
    df = pd.read_csv(csv_path)

    # Group by episode and step to get trajectories
    episodes = df['episode'].unique()
    max_steps = df.groupby('episode')['step'].max().max() + 1

    # Initialize arrays for all episodes
    all_frac_tested = []
    all_frac_detected = []

    for ep in episodes:
        ep_df = df[df['episode'] == ep].sort_values('step')
        frac_tested = ep_df['frac_tested'].values
        frac_detected = ep_df['frac_detected'].values

        # Pad to max_steps if needed
        if len(frac_tested) < max_steps:
            pad_len = max_steps - len(frac_tested)
            frac_tested = np.concatenate([frac_tested, [frac_tested[-1]] * pad_len])
            frac_detected = np.concatenate([frac_detected, [frac_detected[-1]] * pad_len])

        all_frac_tested.append(frac_tested)
        all_frac_detected.append(frac_detected)

    # Convert to arrays and compute statistics
    all_frac_tested = np.array(all_frac_tested)
    all_frac_detected = np.array(all_frac_detected)

    mean_x = all_frac_tested.mean(axis=0)
    mean_y = all_frac_detected.mean(axis=0)
    std_y = all_frac_detected.std(axis=0)

    return mean_x, mean_y, std_y


def main():
    parser = argparse.ArgumentParser(
        description="Compare DPMD-RF and random policy detection curves"
    )
    parser.add_argument(
        "--dpmd",
        type=str,
        required=True,
        help="Path to DPMD-RF trajectories CSV",
    )
    parser.add_argument(
        "--random",
        type=str,
        required=True,
        help="Path to random policy trajectories CSV",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: comparison_plot.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Detection Curve Comparison",
        help="Plot title",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_path = "comparison_plot.png"
    else:
        output_path = args.output

    # Load trajectories
    print(f"Loading DPMD-RF trajectories from: {args.dpmd}")
    dpmd_x, dpmd_y, dpmd_std = load_and_aggregate_trajectories(args.dpmd)

    print(f"Loading random trajectories from: {args.random}")
    random_x, random_y, random_std = load_and_aggregate_trajectories(args.random)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot DPMD-RF
    ax.plot(dpmd_x, dpmd_y, linestyle="-", color="tab:blue", linewidth=2, label="DPMD-RF")
    ax.fill_between(
        dpmd_x,
        dpmd_y - dpmd_std,
        dpmd_y + dpmd_std,
        color="tab:blue",
        alpha=0.25,
    )

    # Plot random
    ax.plot(random_x, random_y, linestyle="--", color="tab:orange", linewidth=2, label="Random")
    ax.fill_between(
        random_x,
        random_y - random_std,
        random_y + random_std,
        color="tab:orange",
        alpha=0.25,
    )

    # Add reference line at 50% tested
    ax.axvline(x=0.5, linestyle=":", color="gray", alpha=0.7, label="50% tested")

    # Formatting
    ax.set_xlabel("Fraction of population tested", fontsize=12)
    ax.set_ylabel("Fraction of positive cases detected", fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    # Find detection rates at 50% tested
    idx_50_dpmd = np.argmin(np.abs(dpmd_x - 0.5))
    idx_50_random = np.argmin(np.abs(random_x - 0.5))

    print(f"\nAt 50% population tested:")
    print(f"  DPMD-RF:  {dpmd_y[idx_50_dpmd]:.3f} ± {dpmd_std[idx_50_dpmd]:.3f}")
    print(f"  Random:   {random_y[idx_50_random]:.3f} ± {random_std[idx_50_random]:.3f}")
    print(f"  Improvement: {dpmd_y[idx_50_dpmd] - random_y[idx_50_random]:.3f} ({100*(dpmd_y[idx_50_dpmd]/random_y[idx_50_random] - 1):.1f}%)")

    # Find final detection rates
    print(f"\nFinal detection rate:")
    print(f"  DPMD-RF:  {dpmd_y[-1]:.3f} ± {dpmd_std[-1]:.3f}")
    print(f"  Random:   {random_y[-1]:.3f} ± {random_std[-1]:.3f}")

    plt.close()


if __name__ == "__main__":
    main()
