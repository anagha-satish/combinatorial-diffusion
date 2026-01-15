#!/usr/bin/env python3
"""
Plot comparison between DPMD-RF, random, greedy, and optionally SRL trajectories.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_aggregate_trajectories(csv_path: str):
    """
    Load trajectories CSV and compute mean/std of detection curves.

    Returns:
        mean_x: mean fraction tested at each step
        mean_y: mean fraction detected at each step
        std_y:  std of fraction detected at each step
        n_eps:  number of episodes found in csv
        max_steps: number of steps (after padding)
    """
    df = pd.read_csv(csv_path)

    required = {"episode", "step", "frac_tested", "frac_detected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    episodes = np.sort(df["episode"].unique())
    if episodes.size == 0:
        raise ValueError(f"{csv_path} has no episodes")

    max_steps = int(df.groupby("episode")["step"].max().max() + 1)

    all_frac_tested = []
    all_frac_detected = []

    for ep in episodes:
        ep_df = df[df["episode"] == ep].sort_values("step")
        frac_tested = ep_df["frac_tested"].to_numpy(dtype=float)
        frac_detected = ep_df["frac_detected"].to_numpy(dtype=float)

        # Pad to max_steps if needed
        if len(frac_tested) < max_steps:
            pad_len = max_steps - len(frac_tested)
            frac_tested = np.concatenate([frac_tested, np.repeat(frac_tested[-1], pad_len)])
            frac_detected = np.concatenate([frac_detected, np.repeat(frac_detected[-1], pad_len)])

        all_frac_tested.append(frac_tested)
        all_frac_detected.append(frac_detected)

    all_frac_tested = np.asarray(all_frac_tested, dtype=float)
    all_frac_detected = np.asarray(all_frac_detected, dtype=float)

    mean_x = all_frac_tested.mean(axis=0)
    mean_y = all_frac_detected.mean(axis=0)
    std_y = all_frac_detected.std(axis=0)

    return mean_x, mean_y, std_y, int(episodes.size), int(max_steps)


def y_at_x(mean_x, mean_y, std_y, x_target: float):
    idx = int(np.argmin(np.abs(mean_x - x_target)))
    return float(mean_x[idx]), float(mean_y[idx]), float(std_y[idx]), idx


def main():
    parser = argparse.ArgumentParser(description="Detection curve comparison (3-way or 4-way)")
    parser.add_argument("--dpmd", required=True, help="Path to DPMD-RF trajectories CSV")
    parser.add_argument("--random", required=True, help="Path to random trajectories CSV")
    parser.add_argument("--greedy", required=True, help="Path to greedy trajectories CSV")
    parser.add_argument("--srl", default=None, help="(Optional) Path to SRL trajectories CSV")

    parser.add_argument("-o", "--output", default="comparison.png", help="Output PNG path")
    parser.add_argument("--title", default="Detection Curve Comparison", help="Plot title")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument("--x_ref", type=float, default=0.5, help="Reference x line (default 0.5)")

    # Optional label tweaks
    parser.add_argument("--label_dpmd", default="DPMD-RF")
    parser.add_argument("--label_random", default="Random")
    parser.add_argument("--label_greedy", default="Greedy + Positive Neighbors")
    parser.add_argument("--label_srl", default="SRL")

    args = parser.parse_args()

    series = []

    print(f"Loading DPMD-RF: {args.dpmd}")
    x_d, y_d, s_d, n_d, T_d = load_and_aggregate_trajectories(args.dpmd)
    series.append(("dpmd", args.label_dpmd, x_d, y_d, s_d, n_d, T_d))

    print(f"Loading Random: {args.random}")
    x_r, y_r, s_r, n_r, T_r = load_and_aggregate_trajectories(args.random)
    series.append(("random", args.label_random, x_r, y_r, s_r, n_r, T_r))

    print(f"Loading Greedy: {args.greedy}")
    x_g, y_g, s_g, n_g, T_g = load_and_aggregate_trajectories(args.greedy)
    series.append(("greedy", args.label_greedy, x_g, y_g, s_g, n_g, T_g))

    if args.srl is not None:
        print(f"Loading SRL: {args.srl}")
        x_s, y_s, s_s, n_s, T_s = load_and_aggregate_trajectories(args.srl)
        series.append(("srl", args.label_srl, x_s, y_s, s_s, n_s, T_s))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 6))

    # Keep linestyles consistent with your original
    for key, label, x, y, s, n_eps, T in series:
        if key == "dpmd":
            ax.plot(x, y, linewidth=2, label=label)
            ax.fill_between(x, y - s, y + s, alpha=0.20)
        elif key == "random":
            ax.plot(x, y, linestyle="--", linewidth=2, label=label)
            ax.fill_between(x, y - s, y + s, alpha=0.20)
        elif key == "greedy":
            ax.plot(x, y, linestyle=":", linewidth=2, label=label)
            ax.fill_between(x, y - s, y + s, alpha=0.20)
        elif key == "srl":
            ax.plot(x, y, linestyle="-.", linewidth=2, label=label)
            ax.fill_between(x, y - s, y + s, alpha=0.20)

    ax.axvline(x=args.x_ref, linestyle="-.", alpha=0.7, label=f"{int(args.x_ref*100)}% tested")

    ax.set_xlabel("Fraction of population tested", fontsize=12)
    ax.set_ylabel("Fraction of positive cases detected", fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {args.output}")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("Summary at reference fraction tested")
    print("=" * 70)
    print(f"Reference x = {args.x_ref:.2f}\n")

    for key, label, x, y, s, n_eps, T in series:
        xx, yy, ss, idx = y_at_x(x, y, s, args.x_ref)
        print(f"{label:26s}: {yy:.3f} ± {ss:.3f} (x≈{xx:.3f}, idx={idx}, eps={n_eps}, steps={T})")

    print("\nFinal detection rate:")
    for key, label, x, y, s, n_eps, T in series:
        print(f"{label:26s}: {y[-1]:.3f} ± {s[-1]:.3f}")

    # Improvements vs baselines (if present)
    # DPMD vs others
    def get_series(keyname: str):
        for key, label, x, y, s, n_eps, T in series:
            if key == keyname:
                return (label, x, y, s)
        return None

    dpmd = get_series("dpmd")
    if dpmd is not None:
        _, x0, y0, s0 = dpmd
        xd, yd, sd, _ = y_at_x(x0, y0, s0, args.x_ref)

        rand = get_series("random")
        greedy = get_series("greedy")
        srl = get_series("srl")

        print("\nImprovements at reference (using y @ x_ref):")
        if rand is not None:
            _, xr, yr, sr = rand
            _, yrr, _, _ = y_at_x(xr, yr, sr, args.x_ref)
            print(f"  {args.label_dpmd} − {args.label_random}: {yd - yrr:+.3f}")
        if greedy is not None:
            _, xg, yg, sg = greedy
            _, ygg, _, _ = y_at_x(xg, yg, sg, args.x_ref)
            print(f"  {args.label_dpmd} − {args.label_greedy}: {yd - ygg:+.3f}")
        if srl is not None:
            _, xs, ys, ss = srl
            _, yss, _, _ = y_at_x(xs, ys, ss, args.x_ref)
            print(f"  {args.label_dpmd} − {args.label_srl}: {yd - yss:+.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
