"""Exp1: Comprehensive error diagnosis and optimal estimation.

This script implements the workflow described in the lab specification:
1. Load the measurement matrix (20 groups × 100 repetitions) from ``data.csv``.
2. Perform basic statistics and visualization.
3. Detect and mitigate gross errors.
4. Diagnose and correct systematic errors (constant/linear/periodic).
5. Assess precision class (equal vs. unequal) and produce optimal estimate.
6. Export required plots (fig1–fig4) and ``results.txt`` summary.

Usage:
    python experiment/exp1.py

The script is deterministic (random seed fixed). Exceptions are raised when the
input file is missing or malformed so that the failure mode is obvious to the
user executing the script.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from numpy.fft import rfft, rfftfreq

# =========================
# 全局设置 (可根据报告需要在此处说明)
# =========================
np.random.seed(12345)  # 复现性
ALPHA_Z = 0.01  # 恒值偏差显著性水平 (双侧)
ALPHA_SLOPE = 0.01  # 线性趋势显著性水平
PERIOD_ENERGY_RATIO = 0.2  # 周期性判定阈值
TREND_GROUPS_FOR_PLOT = (0, 4, 9, 19)  # 对应 1、5、10、20 组
THRESH_EQUAL_PRECISION_RATIO = 2.0  # 等精度判定阈值 max(s)/min(s) < 2
SIGMA_THRESHOLD = 3.0  # 粗大误差 3σ 判据阈值
K_LIMIT = 3.0  # 极限误差倍率
MAX_OUTLIER_ITER = 10


@dataclass
class OutlierResult:
    mask_good: np.ndarray
    cleaned: np.ndarray
    display: np.ndarray


@dataclass
class SystemCorrectionResult:
    corrected: np.ndarray
    constant_bias_groups: List[int]
    linear_trend_groups: List[int]
    periodic_groups: List[int]
    before_after_pairs: List[Tuple[np.ndarray, np.ndarray]]


class DataShapeError(ValueError):
    """Raised when the input data does not have the expected shape."""


def load_data(csv_path: str = "data.csv") -> np.ndarray:
    """Load measurement data.

    Parameters
    ----------
    csv_path:
        Path to the CSV file. The file must contain at least 20 rows and
        100 columns. Extra columns are allowed; only the first 100 will be
        used to comply with the experiment specification.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Required data file not found: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", dtype=float)
    if data.ndim != 2:
        raise DataShapeError("data.csv must be a 2D array")
    if data.shape[0] < 20 or data.shape[1] < 100:
        raise DataShapeError(
            "data.csv must contain at least 20 rows and 100 columns"
        )
    # Trim to 20×100 if more data is provided, to avoid surprises.
    data = data[:20, :100]
    return data


def basic_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and (unbiased) standard deviation for each group."""

    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1, ddof=1)
    return means, stds


def detect_and_remove_outliers(row: np.ndarray) -> OutlierResult:
    """Iteratively remove outliers using the 3σ rule.

    Returns
    -------
    OutlierResult
        ``cleaned`` contains the retained samples (variable length).
        ``display`` replaces rejected samples via neighbourhood median for
        visualization purposes.
    """

    if row.size == 0:
        return OutlierResult(np.zeros(0, dtype=bool), row.copy(), row.copy())

    mask_good = np.ones_like(row, dtype=bool)
    for _ in range(MAX_OUTLIER_ITER):
        candidate = row[mask_good]
        if candidate.size <= 1:
            break
        mu = candidate.mean()
        sd = candidate.std(ddof=1)
        if sd == 0:
            break
        z_scores = np.abs((row - mu) / sd)
        new_bad = (z_scores > SIGMA_THRESHOLD) & mask_good
        if not np.any(new_bad):
            break
        mask_good[new_bad] = False

    cleaned = row[mask_good]
    display = row.copy()
    bad_indices = np.where(~mask_good)[0]
    for idx in bad_indices:
        left = max(0, idx - 2)
        right = min(len(row), idx + 3)
        neighbourhood = [j for j in range(left, right) if mask_good[j]]
        if neighbourhood:
            display[idx] = np.median(row[neighbourhood])
        else:
            # Fallback to global median of retained points
            display[idx] = np.median(cleaned) if cleaned.size > 0 else row[idx]

    return OutlierResult(mask_good, cleaned, display)


def z_test_mean(
    group_mean: float,
    group_std: float,
    n: int,
    global_mean: float,
    alpha: float = ALPHA_Z,
) -> bool:
    """Two-sided z-test for constant bias."""

    safe_std = max(group_std, 1e-12)
    if n <= 1:
        return False
    z_val = (group_mean - global_mean) / (safe_std / math.sqrt(n))
    crit = stats.norm.ppf(1 - alpha / 2)
    if np.isnan(z_val):
        return False
    return abs(z_val) > crit


def linear_trend_test(y: np.ndarray, alpha: float = ALPHA_SLOPE):
    """Test for linear trend using linear regression t-test."""

    n = y.size
    if n <= 2:
        return False, 0.0, 0.0, y.copy()
    x = np.arange(1, n + 1)
    slope, intercept, _, p_value, _ = stats.linregress(x, y)
    has_trend = bool(p_value < alpha)
    if has_trend:
        detrended = y - (intercept + slope * x)
    else:
        detrended = y.copy()
    return has_trend, slope, intercept, detrended


def detect_periodic(y: np.ndarray, ratio_thresh: float = PERIOD_ENERGY_RATIO):
    """Detect dominant periodic component via FFT energy ratio."""

    n = y.size
    if n == 0:
        return False, 0.0, np.zeros_like(y), y.copy()
    demeaned = y - y.mean()
    spectrum = rfft(demeaned)
    power = np.abs(spectrum) ** 2
    if power.size <= 1:
        return False, 0.0, np.zeros_like(y), y.copy()

    idx = np.arange(1, power.size)
    dominant_index = idx[np.argmax(power[1:])]
    total_power = power.sum()
    dom_power = power[dominant_index]
    ratio = dom_power / total_power if total_power > 0 else 0.0
    has_period = ratio > ratio_thresh

    freq = rfftfreq(n, d=1.0)[dominant_index]
    if has_period and freq != 0:
        k = np.arange(n)
        design = np.column_stack(
            [np.sin(2 * np.pi * freq * k), np.cos(2 * np.pi * freq * k), np.ones(n)]
        )
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coeffs
        corrected = y - fitted
    else:
        fitted = np.zeros_like(y)
        corrected = y.copy()

    return has_period, freq, fitted, corrected


def apply_system_corrections(data: np.ndarray) -> SystemCorrectionResult:
    """Diagnose and correct systematic errors for each group."""

    group_means, group_stds = basic_stats(data)
    weights = 1.0 / np.maximum(group_stds, 1e-12) ** 2
    global_mean = np.average(group_means, weights=weights)

    corrected = np.zeros_like(data)
    t1: List[int] = []
    t2: List[int] = []
    t3: List[int] = []
    before_after: List[Tuple[np.ndarray, np.ndarray]] = []

    for idx, series in enumerate(data):
        n = series.size
        # Constant bias check
        has_bias = z_test_mean(group_means[idx], group_stds[idx], n, global_mean)
        bias_corrected = series.copy()
        if has_bias:
            offset = group_means[idx] - global_mean
            bias_corrected = series - offset
            t1.append(idx + 1)

        # Linear trend check
        has_trend, slope, intercept, detrended = linear_trend_test(
            bias_corrected
        )
        trend_corrected = bias_corrected.copy()
        if has_trend:
            trend_corrected = detrended
            t2.append(idx + 1)

        # Periodic check
        has_period, _, fitted, periodic_removed = detect_periodic(trend_corrected)
        final_series = trend_corrected.copy()
        if has_period:
            final_series = periodic_removed
            t3.append(idx + 1)

        corrected[idx] = final_series
        before_after.append((series.copy(), final_series))

    return SystemCorrectionResult(corrected, t1, t2, t3, before_after)


def weighted_average(means: np.ndarray, stds: np.ndarray) -> Tuple[float, float]:
    """Weighted mean and limit error (3σ convention)."""

    safe_stds = np.maximum(stds, 1e-12)
    weights = 1.0 / safe_stds**2
    weighted_mean = np.sum(weights * means) / np.sum(weights)
    sigma_mean = math.sqrt(1.0 / np.sum(weights))
    limit_error = K_LIMIT * sigma_mean
    return weighted_mean, limit_error


def fmt_float_list(values: Sequence[float]) -> str:
    return ",".join(f"{float(v):.2f}" for v in values)


def fmt_int_list(values: Iterable[int]) -> str:
    return ",".join(str(int(v)) for v in values)


def save_results(
    raw_means: np.ndarray,
    raw_stds: np.ndarray,
    outlier_groups: Sequence[int],
    corr_result: SystemCorrectionResult,
    final_means: np.ndarray,
    final_stds: np.ndarray,
    optimal_mean: float,
    limit_error: float,
    equal_precision: bool,
) -> None:
    """Write ``results.txt`` according to the specification."""

    summary = [
        f"粗大误差检测采用迭代3σ法，阈值{SIGMA_THRESHOLD:.1f}σ，共识别{len(outlier_groups)}组含异常点。",
        (
            "恒值偏差使用z检验(alpha={:.2f})，线性趋势使用斜率t检验(alpha={:.2f})，"
            "周期性依据FFT主频能量占比阈值{:.2f}。"
        ).format(ALPHA_Z, ALPHA_SLOPE, PERIOD_ENERGY_RATIO),
        (
            "系统误差校正后判定为{}精度，并据此计算最优估计；"
            "最终结果为{:.2f}±{:.2f}(3σ极限误差)。"
        ).format("等" if equal_precision else "不等", optimal_mean, limit_error),
        "进一步分析请见报告，对阈值敏感性、过拟合风险及AI协助均作说明。",
    ]

    lines = [
        fmt_float_list(raw_means),
        fmt_float_list(raw_stds),
        fmt_int_list(outlier_groups),
        fmt_int_list(corr_result.constant_bias_groups),
        fmt_int_list(corr_result.linear_trend_groups),
        fmt_int_list(corr_result.periodic_groups),
        f"{optimal_mean:.2f}±{limit_error:.2f}",
        " ".join(summary),
    ]

    with open("results.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _subplot_layout(count: int) -> Tuple[plt.Figure, np.ndarray, int, int]:
    """Utility to build a compact subplot grid for ``count`` panels."""

    if count <= 0:
        raise ValueError("subplot count must be positive")

    cols = 2 if count > 1 else 1
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(cols * 4.5, rows * 3.2))
    axes_array = np.atleast_1d(axes).reshape(-1)
    return fig, axes_array, rows, cols


def plot_raw_groups(raw: np.ndarray, indices: Sequence[int], path: str) -> None:
    fig, axes, rows, cols = _subplot_layout(len(indices))
    n = raw.shape[1]
    x_axis = np.arange(1, n + 1)

    for ax, idx in zip(axes, indices):
        ax.plot(x_axis, raw[idx], label="原始曲线", color="#1f77b4")
        ax.set_title(f"第{idx + 1}组")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for ax in axes[len(indices):]:
        ax.axis("off")

    for i, ax in enumerate(axes):
        if i >= len(indices):
            continue
        row = i // cols
        if row == rows - 1:
            ax.set_xlabel("测量序号")
        ax.set_ylabel("测量值")

    fig.suptitle("典型4组原始测量曲线", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_outlier_comparison(
    before: np.ndarray, after: np.ndarray, indices: Sequence[int], path: str
) -> None:
    fig, axes, rows, cols = _subplot_layout(len(indices))
    n = before.shape[1]
    x_axis = np.arange(1, n + 1)

    for ax, idx in zip(axes, indices):
        ax.plot(x_axis, before[idx], alpha=0.7, label="剔除前", color="#d62728")
        ax.plot(
            x_axis,
            after[idx],
            linestyle="--",
            label="剔除后(替代)",
            color="#2ca02c",
        )
        ax.set_title(f"第{idx + 1}组")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for ax in axes[len(indices):]:
        ax.axis("off")

    for i, ax in enumerate(axes):
        if i >= len(indices):
            continue
        row = i // cols
        if row == rows - 1:
            ax.set_xlabel("测量序号")
        ax.set_ylabel("测量值")

    fig.suptitle("粗大误差剔除前后对比", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_system_correction(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    indices: Sequence[int],
    path: str,
) -> None:
    if not pairs:
        raise ValueError("No data provided for system correction plot")

    fig, axes, rows, cols = _subplot_layout(len(indices))
    n = pairs[0][0].size
    x_axis = np.arange(1, n + 1)

    for ax, idx in zip(axes, indices):
        before, after = pairs[idx]
        ax.plot(x_axis, before, alpha=0.7, label="校正前", color="#9467bd")
        ax.plot(x_axis, after, linestyle="--", label="校正后", color="#ff7f0e")
        ax.set_title(f"第{idx + 1}组")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for ax in axes[len(indices):]:
        ax.axis("off")

    for i, ax in enumerate(axes):
        if i >= len(indices):
            continue
        row = i // cols
        if row == rows - 1:
            ax.set_xlabel("测量序号")
        ax.set_ylabel("测量值")

    fig.suptitle("系统误差校正前后对比", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_means_stds(
    means: np.ndarray, stds: np.ndarray, path: str, title: str
) -> None:
    fig, ax1 = plt.subplots()
    groups = np.arange(1, means.size + 1)
    ax1.set_xlabel("Group ID")
    ax1.set_ylabel("Mean")
    line1 = ax1.plot(groups, means, marker="o", label="Group mean")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Std dev")
    line2 = ax2.plot(groups, stds, marker="s", linestyle="--", label="Group std")

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    data = load_data("data.csv")
    num_groups, n_measurements = data.shape

    # Step 1: basic statistics
    raw_means, raw_stds = basic_stats(data)
    plot_raw_groups(data, TREND_GROUPS_FOR_PLOT, "fig1.png")

    # Step 2: outlier detection
    outlier_groups: List[int] = []
    display_rows: List[np.ndarray] = []
    for idx in range(num_groups):
        row = data[idx]
        outlier_result = detect_and_remove_outliers(row)
        if np.any(~outlier_result.mask_good):
            outlier_groups.append(idx + 1)
        display_rows.append(outlier_result.display)

    display_array = np.vstack(display_rows)
    plot_outlier_comparison(data, display_array, TREND_GROUPS_FOR_PLOT, "fig2.png")

    # Step 3: systematic error correction
    system_result = apply_system_corrections(display_array)
    corrected = system_result.corrected
    plot_system_correction(system_result.before_after_pairs, TREND_GROUPS_FOR_PLOT, "fig3.png")

    # Step 4: optimal estimate after correction
    final_means, final_stds = basic_stats(corrected)
    s_max = float(np.max(final_stds))
    s_min = float(np.max([np.min(final_stds), 1e-12]))
    equal_precision = s_max / s_min < THRESH_EQUAL_PRECISION_RATIO

    if equal_precision:
        optimal_mean = float(np.mean(final_means))
        if final_means.size > 1:
            sigma_groups = float(np.std(final_means, ddof=1))
            sigma_opt = sigma_groups / math.sqrt(final_means.size)
        else:
            sigma_opt = 0.0
        limit_error = K_LIMIT * sigma_opt
    else:
        optimal_mean, limit_error = weighted_average(final_means, final_stds)

    plot_means_stds(
        final_means,
        final_stds,
        "fig4.png",
        "Group means and std devs after correction",
    )

    save_results(
        raw_means,
        raw_stds,
        outlier_groups,
        system_result,
        final_means,
        final_stds,
        optimal_mean,
        limit_error,
        equal_precision,
    )

    print(
        "Done: generated fig1.png, fig2.png, fig3.png, fig4.png, results.txt."
    )


if __name__ == "__main__":
    main()
