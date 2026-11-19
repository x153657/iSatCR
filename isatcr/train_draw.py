"""Plot training reward curves over the configured seeds for each algorithm."""

import os
import re
from datetime import datetime
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training_process_data", "v1")
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")

# 图像参数：修改后保存并重新运行本脚本。尺寸单位英寸，字号单位pt。
FIGURE_SIZE = (3.4, 2.5)
OUTPUT_DPI = 600
OUTPUT_FORMATS = ("png", "pdf")
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 10
AXIS_LABEL_SIZE =9    # X/Y轴标题字号
X_TICK_SIZE = 8
Y_TICK_SIZE = 8
TITLE_SIZE = 10
LEGEND_SIZE = 10
LINE_WIDTH = 1.4
STD_ALPHA = 0.20        # 标准差阴影透明度
AXIS_WIDTH = 0.8
GRID_STYLE = "--"
GRID_WIDTH = 0.45
GRID_ALPHA = 0.45
AXIS_LABEL_PAD = 1.5
X_TICK_INTERVAL = 200
Y_TICK_INTERVAL = 0.3
X_LEFT_MARGIN = 30
Y_BOTTOM_MARGIN = 0.05
LEGEND_LOCATION = "lower right"
LEGEND_COLUMNS = 1
LEGEND_HANDLE_LENGTH = 2.2
LEGEND_COLUMN_SPACING = 2.0
LEGEND_LABEL_SPACING = 0.65
LEGEND_TEXT_PAD = 0.5
LEGEND_AXES_PAD = 0.2
SUBPLOT_MARGINS = dict(left=0.17, right=0.99, bottom=0.15, top=0.99)
SAVE_PAD_INCHES = 0.02

# 原始种子诊断图的独立尺寸和分辨率。
RAW_COLUMNS = 3
RAW_FIGURE_WIDTH = 10.0
RAW_ROW_HEIGHT = 2.8
RAW_OUTPUT_DPI = 400

EXPECTED_SEEDS = 5
INCLUDED_SEEDS = {0, 1, 2, 3, 4}
SMOOTHING_WINDOW = 30
OUTPUT_BASENAME = "training"
RAW_DIAGNOSTIC_BASENAME = "training_reward_raw_seeds_diagnostic"
PLOT_Y_MAX = 0.9
PLOT_Y_MIN = -0.9

PLOT_X_MAX = 1000

ALGORITHMS = {
    "DDQN": {
        "directory": "DDQN",
        "file_prefix": "train_DDQN",
        "color": "#7f7f7f",
        "linestyle": ":",
    },
    "PPO": {
        "directory": "ppo",
        "file_prefix": "train_PPO",
        "color": "#17becf",
        "linestyle": "--",
        "zorder": 4,
    },
    "D3QN": {
        "directory": "D3QN",
        "file_prefix": "train_D3QN",
        "color": "#bcbd22",
        "linestyle": "-",
    },
    "iSatCR": {
        "directory": "ISACR_self_elimination",
        "file_prefix": "train_ISACR_fault_self_elimination",
        "color": "#1f77b4",
        "linestyle": "-.",
    },
}


def parse_training_log(file_path):
    """Read one log while preserving its recorded training-step positions."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    steps = []
    rewards = []
    headers = list(
        re.finditer(r"^====== step\s+(\d+)\s+======\s*$", content, flags=re.MULTILINE)
    )
    for index, header in enumerate(headers):
        block_end = headers[index + 1].start() if index + 1 < len(headers) else len(content)
        block_text = content[header.end() : block_end]
        reward_match = re.search(
            r"Average ending reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|None)",
            block_text,
        )
        if not reward_match:
            continue

        steps.append(float(header.group(1)))
        reward_text = reward_match.group(1)
        rewards.append(np.nan if reward_text == "None" else float(reward_text))

    if not steps:
        raise ValueError(f"No training rewards found in {file_path}")
    return np.asarray(steps), np.asarray(rewards, dtype=float)


def smooth_series(values, window=SMOOTHING_WINDOW):
    """Apply a centered moving average over the recorded training rounds."""
    if window <= 1:
        return values.copy()

    smoothed = (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    smoothed[~np.isfinite(values)] = np.nan
    return smoothed


def discover_logs(
    directory,
    file_prefix,
    included_seeds=INCLUDED_SEEDS,
    expected_seeds=EXPECTED_SEEDS,
):
    """Find and numerically sort the seed logs for one algorithm."""
    pattern = os.path.join(MODEL_DIR, directory, f"{file_prefix}_seed*_*.txt")
    candidates = []
    for file_path in glob(pattern):
        match = re.search(r"_seed(\d+)(?:_|\.)", os.path.basename(file_path))
        if match:
            seed = int(match.group(1))
            if seed in included_seeds:
                candidates.append((seed, file_path))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        print(f"Warning: no seed logs found for {file_prefix}; skipping it.")
        return []
    if len(candidates) != expected_seeds:
        print(
            f"Warning: expected {expected_seeds} seed logs for {file_prefix}, "
            f"found {len(candidates)}; using all available seeds."
        )
    return candidates


def align_runs(runs):
    """Return one common step grid and each smoothed run sampled on that grid."""
    if all(np.array_equal(runs[0][0], steps) for steps, _ in runs[1:]):
        common_steps = runs[0][0]
        aligned = np.vstack([values for _, values in runs])
        return common_steps, aligned

    valid_ranges = [
        (steps[np.isfinite(values)][0], steps[np.isfinite(values)][-1])
        for steps, values in runs
        if np.any(np.isfinite(values))
    ]
    start = max(start for start, _ in valid_ranges)
    end = min(end for _, end in valid_ranges)
    if start > end:
        raise ValueError("Seed logs do not have an overlapping training-step range")

    common_steps = np.unique(
        np.concatenate([steps[(steps >= start) & (steps <= end)] for steps, _ in runs])
    )
    aligned = []
    for steps, values in runs:
        valid = np.isfinite(values) & (steps >= start) & (steps <= end)
        if valid.sum() < 2:
            raise ValueError("A seed log has fewer than two valid rewards in the aligned range")
        aligned.append(np.interp(common_steps, steps[valid], values[valid]))
    return common_steps, np.vstack(aligned)


def mean_and_std(values):
    """Calculate per-step mean and sample standard deviation, ignoring NaNs."""
    frame = pd.DataFrame(values)
    mean = frame.mean(axis=0, skipna=True).to_numpy()
    std = frame.std(axis=0, skipna=True, ddof=1).to_numpy()
    # A partial experiment may contain only one available seed.  Its sample
    # standard deviation is undefined, but the mean curve is still plottable.
    valid_counts = frame.notna().sum(axis=0).to_numpy()
    std[valid_counts == 1] = 0.0
    return mean, std


def load_algorithm_data(algorithm_config):
    runs = []
    seed_logs = discover_logs(
        algorithm_config["directory"],
        algorithm_config["file_prefix"],
        algorithm_config.get("included_seeds", INCLUDED_SEEDS),
        algorithm_config.get("expected_seeds", EXPECTED_SEEDS),
    )
    for seed, file_path in seed_logs:
        try:
            steps, rewards = parse_training_log(file_path)
        except ValueError as exc:
            print(f"Skipping seed{seed}: {exc}")
            continue
        if np.isfinite(rewards).sum() < 2:
            print(f"Skipping seed{seed}: fewer than two recorded rewards in {file_path}")
            continue
        runs.append((steps, smooth_series(rewards)))
        print(f"Loaded {algorithm_config['file_prefix']} seed{seed}: {file_path}")

    if not runs:
        return None
    steps, aligned_values = align_runs(runs)
    mean, std = mean_and_std(aligned_values)
    if steps[-1] < PLOT_X_MAX:
        print(
            f"Warning: {algorithm_config['file_prefix']} only reaches "
            f"step {steps[-1]:g}."
        )
    return steps, mean, std


def plot_raw_seed_diagnostics():
    """Plot each algorithm's selected un-smoothed seed curves for boundary diagnosis."""
    columns = RAW_COLUMNS
    rows = (len(ALGORITHMS) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(RAW_FIGURE_WIDTH, RAW_ROW_HEIGHT * rows), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (name, config) in zip(axes, ALGORITHMS.items()):
        seed_logs = discover_logs(
            config["directory"],
            config["file_prefix"],
            config.get("included_seeds", INCLUDED_SEEDS),
            config.get("expected_seeds", EXPECTED_SEEDS),
        )
        if not seed_logs:
            ax.set_visible(False)
            continue
        for seed, file_path in seed_logs:
            try:
                steps, rewards = parse_training_log(file_path)
            except ValueError as exc:
                print(f"Skipping seed{seed}: {exc}")
                continue
            if np.isfinite(rewards).sum() < 2:
                print(f"Skipping seed{seed}: fewer than two recorded rewards in {file_path}")
                continue
            ax.plot(
                steps,
                rewards,
                linewidth=LINE_WIDTH,
                linestyle=config["linestyle"],
                alpha=0.85,
                label=f"seed{seed}",
            )
        ax.set_title(f"{name} (n={len(seed_logs)})")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
        ax.set_xlim(0, PLOT_X_MAX)
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Average Ending Reward")

    for ax in axes[len(ALGORITHMS):]:
        ax.set_visible(False)
    legend_handles = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend_handles.setdefault(label, handle)
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles),
        loc="upper center",
        ncol=min(5, len(legend_handles)),
        frameon=False,
    )
    fig.tight_layout(rect=(0.04, 0.05, 1.0, 0.88), pad=0.6)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates = (
        os.path.join(OUTPUT_DIR, f"{RAW_DIAGNOSTIC_BASENAME}.png"),
        os.path.join(PROJECT_ROOT, f"{RAW_DIAGNOSTIC_BASENAME}.png"),
        os.path.join(PROJECT_ROOT, f"{RAW_DIAGNOSTIC_BASENAME}_{timestamp}.png"),
    )
    for output_path in candidates:
        try:
            fig.savefig(
                output_path,
                format="png",
                dpi=RAW_OUTPUT_DPI,
                bbox_inches="tight",
                pad_inches=SAVE_PAD_INCHES,
            )
            break
        except (OSError, PermissionError) as exc:
            print(f"Cannot save {output_path}: {exc}")
    else:
        raise OSError("Failed to save raw diagnostic figure to all output paths")
    plt.close(fig)
    print(f"Saved raw diagnostic: {output_path}")


def plot_data():
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE if AXIS_LABEL_SIZE is None else AXIS_LABEL_SIZE,
            "xtick.labelsize": FONT_SIZE if X_TICK_SIZE is None else X_TICK_SIZE,
            "ytick.labelsize": FONT_SIZE if Y_TICK_SIZE is None else Y_TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.unicode_minus": False,
            "axes.linewidth": AXIS_WIDTH,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": AXIS_WIDTH,
            "ytick.major.width": AXIS_WIDTH,
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=OUTPUT_DPI)

    loaded_data = []
    for name, config in ALGORITHMS.items():
        algorithm_data = load_algorithm_data(config)
        if algorithm_data is None:
            continue
        loaded_data.append((name, config, algorithm_data))

    plot_y_limits = (PLOT_Y_MIN, PLOT_Y_MAX)

    plotted_names = []
    for name, config, algorithm_data in loaded_data:
        steps, mean, std = algorithm_data
        line_zorder = config.get("zorder", 2)
        in_range = (
            np.isfinite(mean)
            & np.isfinite(std)
            & (mean >= plot_y_limits[0])
            & (mean <= plot_y_limits[1])
        )
        lower = np.clip(mean - std, *plot_y_limits)
        upper = np.clip(mean + std, *plot_y_limits)

        ax.fill_between(
            steps,
            lower,
            upper,
            where=in_range,
            color=config["color"],
            alpha=STD_ALPHA,
            linewidth=0,
            zorder=line_zorder - 1,
        )
        ax.plot(
            steps,
            np.where(in_range, mean, np.nan),
            color=config["color"],
            linewidth=LINE_WIDTH,
            linestyle=config["linestyle"],
            label=name,
            zorder=line_zorder,
        )
        plotted_names.append(name)

    ax.set_xlabel("Training Epochs", labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("Average Ending Reward", labelpad=AXIS_LABEL_PAD)
    # Offset the first ticks from the frame to leave space at the lower left.
    ax.set_xlim(-X_LEFT_MARGIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Y_MIN - Y_BOTTOM_MARGIN, PLOT_Y_MAX)
    ax.set_xticks(np.arange(0, PLOT_X_MAX + 1, X_TICK_INTERVAL))
    ax.set_yticks(np.arange(PLOT_Y_MIN, PLOT_Y_MAX + 1e-9, Y_TICK_INTERVAL))
    ax.minorticks_off()
    ax.grid(True, which="major", linestyle=GRID_STYLE, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    # Match the reference draw_train.py frame: four thin axis spines and
    # Matplotlib's default rounded, light-gray legend box.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_WIDTH)
    legend = ax.legend(
        ncol=LEGEND_COLUMNS,
        loc=LEGEND_LOCATION,
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
        framealpha=0.8,
        fancybox=True,
        handlelength=LEGEND_HANDLE_LENGTH,
        columnspacing=LEGEND_COLUMN_SPACING,
        labelspacing=LEGEND_LABEL_SPACING,
        handletextpad=LEGEND_TEXT_PAD,
        borderaxespad=LEGEND_AXES_PAD,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(**SUBPLOT_MARGINS)
    output_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for extension in OUTPUT_FORMATS:
        candidates = (
            os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.{extension}"),
            os.path.join(PROJECT_ROOT, f"{OUTPUT_BASENAME}.{extension}"),
            os.path.join(PROJECT_ROOT, f"{OUTPUT_BASENAME}_{timestamp}.{extension}"),
        )
        for output_path in candidates:
            try:
                fig.savefig(
                    output_path,
                    format=extension,
                    dpi=OUTPUT_DPI,
                    bbox_inches="tight",
                    pad_inches=SAVE_PAD_INCHES,
                )
                output_paths.append(output_path)
                break
            except (OSError, PermissionError) as exc:
                print(f"Cannot save {output_path}: {exc}")
        else:
            raise OSError(f"Failed to save {extension} figure to all output paths")
    plt.close(fig)

    for output_path in output_paths:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_raw_seed_diagnostics()
    plot_data()
