from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from numpy.fft import rfft, rfftfreq

"""
Exp1: Comprehensive error diagnosis and optimal estimation.

Deterministic pipeline:
1. Load data matrix (20×100) from BASE_DIR/data.csv
2. Basic stats + fig1.png
3. Gross error detection/remedy + fig2.png
4. Systematic error diagnosis/correction + fig3.png
5. Precision class, optimal estimate, fig4.png
6. Write results.txt

All outputs (fig1.png ... results.txt) are saved into BASE_DIR.
"""

# =========================================================
# 固定的实验工作目录（← 这是你指定的最终目录）
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 确保目录存在（如果路径打错，这里会创建一个新空目录，同样可见）
os.makedirs(BASE_DIR, exist_ok=True)
print("实验工作目录 (读写位置):", BASE_DIR)

# =========================================================
# 全局参数（写报告时可以引用这些数值）
# =========================================================
np.random.seed(12345)            # 复现性
ALPHA_Z = 0.01                   # 恒值偏差显著性水平 (双侧)
ALPHA_SLOPE = 0.01               # 线性趋势显著性水平
PERIOD_ENERGY_RATIO = 0.2        # 周期性判定阈值 (主频能量占比)
TREND_GROUPS_FOR_PLOT = (0, 4, 9, 19)  # 对应要求的 1/5/10/20 组
THRESH_EQUAL_PRECISION_RATIO = 2.0     # 等精度阈值：max(s)/min(s) < 2
SIGMA_THRESHOLD = 3.0            # 粗大误差 3σ 判据
K_LIMIT = 3.0                    # 极限误差系数 (3σ)
MAX_OUTLIER_ITER = 10            # 粗大误差迭代上限


# =========================================================
# 数据结构
# =========================================================
@dataclass
class OutlierResult:
    mask_good: np.ndarray     # bool数组，同长度，True=保留点
    cleaned: np.ndarray       # 剔除粗大误差后的点（变长）
    display: np.ndarray       # 仅用于画图：异常点用邻域中位数替代


@dataclass
class SystemCorrectionResult:
    corrected: np.ndarray                       # 系统误差校正后的序列 (20×N)
    constant_bias_groups: List[int]             # t1
    linear_trend_groups: List[int]              # t2
    periodic_groups: List[int]                  # t3
    before_after_pairs: List[Tuple[np.ndarray, np.ndarray]]  # (before, after)


class DataShapeError(ValueError):
    """Raised when data.csv does not have expected shape."""


# =========================================================
# 读数据：强制从 BASE_DIR/data.csv 读取
# =========================================================
def load_data() -> np.ndarray:
    """
    读取测量数据矩阵 (20组 × 100次测量).
    要求: BASE_DIR/data.csv 必须存在.
    允许 data.csv 比 20×100 更大；我们只取前20行、前100列。
    """
    csv_path = os.path.join(BASE_DIR, "data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[FATAL] data.csv not found at {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", dtype=float)

    if data.ndim != 2:
        raise DataShapeError("data.csv must be a 2D numeric array")
    if data.shape[0] < 20 or data.shape[1] < 100:
        raise DataShapeError(
            f"data.csv shape {data.shape} is too small; need at least 20 rows × 100 cols"
        )

    # 裁剪到 20×100
    data = data[:20, :100]
    return data


# =========================================================
# 基础统计
# =========================================================
def basic_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1, ddof=1)
    return means, stds


# =========================================================
# 粗大误差检测（迭代3σ法）
# =========================================================
def detect_and_remove_outliers(row: np.ndarray) -> OutlierResult:
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

        z = np.abs((row - mu) / sd)
        new_bad = (z > SIGMA_THRESHOLD) & mask_good
        if not np.any(new_bad):
            break
        mask_good[new_bad] = False

    cleaned = row[mask_good]

    # 用邻域中位数/全局中位数替换，只是为了画“after”曲线
    display = row.copy()
    bad_idx = np.where(~mask_good)[0]
    for bi in bad_idx:
        L = max(0, bi - 2)
        R = min(len(row), bi + 3)
        neigh = [j for j in range(L, R) if mask_good[j]]
        if len(neigh) > 0:
            display[bi] = np.median(row[neigh])
        else:
            display[bi] = np.median(cleaned) if cleaned.size > 0 else row[bi]

    return OutlierResult(mask_good, cleaned, display)


# =========================================================
# 系统误差判定工具
# =========================================================
def z_test_mean(
    group_mean: float,
    group_std: float,
    n: int,
    global_mean: float,
    alpha: float = ALPHA_Z,
) -> bool:
    """恒值系统误差(z检验)."""
    if n <= 1:
        return False
    safe_std = max(group_std, 1e-12)
    z_val = (group_mean - global_mean) / (safe_std / math.sqrt(n))
    if np.isnan(z_val):
        return False
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return abs(z_val) > z_crit


def linear_trend_test(y: np.ndarray, alpha: float = ALPHA_SLOPE):
    """线性趋势检验: 对 y(k)=a+b*k 回归，检验 b 是否显著非0。"""
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
    """
    周期性检验: FFT 主频能量占总能量比例是否超过阈值.
    同时拟合主频正弦+余弦并扣除，以得到去周期序列.
    """
    n = y.size
    if n == 0:
        return False, 0.0, np.zeros_like(y), y.copy()

    demeaned = y - y.mean()
    spectrum = rfft(demeaned)
    power = np.abs(spectrum) ** 2
    if power.size <= 1:
        return False, 0.0, np.zeros_like(y), y.copy()

    idx_nonzero = np.arange(1, power.size)
    dom_idx = idx_nonzero[np.argmax(power[1:])]
    dom_power = power[dom_idx]
    total_power = power.sum()
    ratio = dom_power / total_power if total_power > 0 else 0.0
    has_period = ratio > ratio_thresh

    freq_arr = rfftfreq(n, d=1.0)
    f = freq_arr[dom_idx]

    if has_period and f != 0:
        k = np.arange(n)
        design = np.column_stack([
            np.sin(2 * np.pi * f * k),
            np.cos(2 * np.pi * f * k),
            np.ones(n)
        ])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef
        corrected = y - fitted
    else:
        fitted = np.zeros_like(y)
        corrected = y.copy()

    return has_period, f, fitted, corrected


# =========================================================
# 系统误差诊断+校正
# =========================================================
def apply_system_corrections(data: np.ndarray) -> SystemCorrectionResult:
    """
    对每组序列：
    - 恒值偏差检测/平移校正
    - 线性趋势检测/去趋势
    - 周期成分检测/去周期
    返回校正后数据、以及每类系统误差对应的组号列表。
    """

    group_means, group_stds = basic_stats(data)
    # 全局均值用加权平均（权重 = 1/sigma^2）
    weights = 1.0 / (np.maximum(group_stds, 1e-12) ** 2)
    global_mean = np.average(group_means, weights=weights)

    corrected = np.zeros_like(data)
    t1_list: List[int] = []
    t2_list: List[int] = []
    t3_list: List[int] = []
    before_after_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for i, series in enumerate(data):
        n = series.size

        # 1. 恒值偏差
        has_bias = z_test_mean(group_means[i], group_stds[i], n, global_mean)
        bias_corrected = series.copy()
        if has_bias:
            offset = group_means[i] - global_mean
            bias_corrected = series - offset
            t1_list.append(i + 1)  # 组号从1开始

        # 2. 线性趋势
        has_trend, slope, intercept, detrended = linear_trend_test(
            bias_corrected
        )
        trend_corrected = bias_corrected.copy()
        if has_trend:
            trend_corrected = detrended
            t2_list.append(i + 1)

        # 3. 周期成分
        has_period, dom_f, fitted, periodic_removed = detect_periodic(
            trend_corrected
        )
        final_series = trend_corrected.copy()
        if has_period:
            final_series = periodic_removed
            t3_list.append(i + 1)

        corrected[i] = final_series
        before_after_pairs.append((series.copy(), final_series))

    return SystemCorrectionResult(
        corrected=corrected,
        constant_bias_groups=t1_list,
        linear_trend_groups=t2_list,
        periodic_groups=t3_list,
        before_after_pairs=before_after_pairs,
    )


# =========================================================
# 加权平均 (不等精度情况)
# =========================================================
def weighted_average(means: np.ndarray, stds: np.ndarray) -> Tuple[float, float]:
    safe_stds = np.maximum(stds, 1e-12)
    w = 1.0 / (safe_stds ** 2)
    a = float(np.sum(w * means) / np.sum(w))
    sigma_a = math.sqrt(1.0 / np.sum(w))
    limit_err = K_LIMIT * sigma_a
    return a, float(limit_err)


# =========================================================
# 文本输出格式化
# =========================================================
def fmt_float_list(values: Sequence[float]) -> str:
    return ",".join(f"{float(v):.2f}" for v in values)

def fmt_int_list(values: Iterable[int]) -> str:
    return ",".join(str(int(v)) for v in values)


# =========================================================
# 保存 results.txt —— 现在写到 BASE_DIR 下
# =========================================================
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

    summary_lines = [
        f"粗大误差检测采用迭代3σ法(阈值={SIGMA_THRESHOLD:.1f}σ)，共识别{len(outlier_groups)}组含异常点。",
        (
            "恒值偏差用z检验(alpha={:.2f})，线性趋势用斜率t检验(alpha={:.2f})，"
            "周期成分用FFT主频能量占比阈值{:.2f}。"
        ).format(ALPHA_Z, ALPHA_SLOPE, PERIOD_ENERGY_RATIO),
        (
            "系统误差校正后判定为{}精度，并据此计算最优估计；"
            "最终结果为{:.2f}±{:.2f}(3σ极限误差)。"
        ).format("等" if equal_precision else "不等", optimal_mean, limit_error),
        "后续报告将讨论阈值敏感性、过拟合风险以及AI协助范围。",
    ]

    lines = [
        fmt_float_list(raw_means),                                 # 1行 m1
        fmt_float_list(raw_stds),                                  # 2行 s1
        fmt_int_list(outlier_groups),                              # 3行 p
        fmt_int_list(corr_result.constant_bias_groups),           # 4行 t1
        fmt_int_list(corr_result.linear_trend_groups),            # 5行 t2
        fmt_int_list(corr_result.periodic_groups),                # 6行 t3
        f"{optimal_mean:.2f}±{limit_error:.2f}",                  # 7行 a
        " ".join(summary_lines),                                  # 8行 说明
    ]

    results_path = os.path.join(BASE_DIR, "results.txt")
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# =========================================================
# 画图函数（现在全部写到 BASE_DIR 下）
# =========================================================
def plot_raw_groups(raw: np.ndarray, indices: Sequence[int], filename: str) -> None:
    plt.figure()
    n = raw.shape[1]
    x_axis = np.arange(1, n + 1)
    for idx in indices:
        plt.plot(x_axis, raw[idx], label=f"Group {idx + 1} raw")
    plt.xlabel("Measurement index")
    plt.ylabel("Value")
    plt.title("Raw measurements (Groups 1,5,10,20)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_outlier_comparison(before: np.ndarray,
                            after: np.ndarray,
                            indices: Sequence[int],
                            filename: str) -> None:
    plt.figure()
    n = before.shape[1]
    x_axis = np.arange(1, n + 1)
    for idx in indices:
        plt.plot(x_axis, before[idx], alpha=0.7, label=f"G{idx + 1} before")
        plt.plot(x_axis, after[idx], linestyle="--", label=f"G{idx + 1} after")
    plt.xlabel("Measurement index")
    plt.ylabel("Value")
    plt.title("Outlier removal comparison")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_system_correction(pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
                           indices: Sequence[int],
                           filename: str) -> None:
    plt.figure()
    if not pairs:
        n = 0
        x_axis = np.arange(1, n + 1)
    else:
        n = pairs[0][0].size
        x_axis = np.arange(1, n + 1)

    for idx in indices:
        before, after = pairs[idx]
        plt.plot(x_axis, before, alpha=0.7, label=f"G{idx + 1} before")
        plt.plot(x_axis, after, linestyle="--", label=f"G{idx + 1} after")
    plt.xlabel("Measurement index")
    plt.ylabel("Value")
    plt.title("Systematic error correction")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_means_stds(means: np.ndarray,
                    stds: np.ndarray,
                    filename: str,
                    title: str) -> None:
    fig, ax1 = plt.subplots()
    groups = np.arange(1, means.size + 1)

    ax1.set_xlabel("Group ID")
    ax1.set_ylabel("Mean")
    l1 = ax1.plot(groups, means, marker="o", label="Group mean")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Std dev")
    l2 = ax2.plot(groups, stds, marker="s", linestyle="--", label="Group std")

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title(title)
    fig.tight_layout()

    out_path = os.path.join(BASE_DIR, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =========================================================
# 主流程
# =========================================================
def main() -> None:
    # Step 0: 读数据
    data = load_data()  # 强制从 BASE_DIR/data.csv
    num_groups, n_meas = data.shape

    # Step 1: 基础统计 + fig1
    raw_means, raw_stds = basic_stats(data)
    plot_raw_groups(data, TREND_GROUPS_FOR_PLOT, "fig1.png")

    # Step 2: 粗大误差识别 + fig2
    outlier_groups: List[int] = []
    display_rows: List[np.ndarray] = []
    for gi in range(num_groups):
        res = detect_and_remove_outliers(data[gi])
        if np.any(~res.mask_good):
            outlier_groups.append(gi + 1)  # 1-based
        display_rows.append(res.display)
    display_arr = np.vstack(display_rows)
    plot_outlier_comparison(data, display_arr, TREND_GROUPS_FOR_PLOT, "fig2.png")

    # Step 3: 系统误差诊断/校正 + fig3
    sys_result = apply_system_corrections(display_arr)
    corrected = sys_result.corrected
    plot_system_correction(sys_result.before_after_pairs,
                           TREND_GROUPS_FOR_PLOT,
                           "fig3.png")

    # Step 4: 等/不等精度 + 最优估计 + fig4
    final_means, final_stds = basic_stats(corrected)
    s_max = float(np.max(final_stds))
    s_min = float(max(np.min(final_stds), 1e-12))
    equal_precision = (s_max / s_min) < THRESH_EQUAL_PRECISION_RATIO

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

    # Step 5: results.txt
    save_results(
        raw_means,
        raw_stds,
        outlier_groups,
        sys_result,
        final_means,
        final_stds,
        float(optimal_mean),
        float(limit_error),
        equal_precision,
    )

    print("完成 ✅ 已生成以下文件到:", BASE_DIR)
    print(" - fig1.png, fig2.png, fig3.png, fig4.png")
    print(" - results.txt")


# =========================================================
# 入口
# =========================================================
if __name__ == "__main__":
    main()
