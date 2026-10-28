# E-Proposed MOO (proposed algorithm; ablation/sensitivity switches in USER SETTINGS) for the ZCAT problems with any number of objectives M (paper: M = 3 and M = 5).
# Usage: python E-Proposed_MOO.py <PROBLEM_ID> <M>   (or set PROBLEM_ID and M in USER SETTINGS and press F5).
# E-Proposed_MOO_M3.py  --  self-contained file (no other file needed).
# Run in Spyder with F5. Set PROBLEM_ID (or PROBLEM_LIST) in USER SETTINGS below.
# Results are saved to results_csv\ next to this file; use stats_analysis.py afterwards.

import os
import random
import time
import warnings
from multiprocessing import Pool, cpu_count
from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from pymoo.indicators.hv import HV
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

PI = np.pi
E = np.e

# ==========================================================
# 0. GLOBAL CONSTANTS FROM THE PAPER
# ==========================================================
K_DISCONNECT = 5
ZCAT19_A = 5
BIAS_GAMMA = 0.05


# ==========================================================
# 1. DECISION-SPACE NORMALIZATION (batched over the whole population)
# ==========================================================
def bounds_for(n):
    idx = np.arange(1, n + 1, dtype=np.float64)
    return np.stack([-idx / 2.0, idx / 2.0], axis=1)


def x_to_y_batch(X):
    """X: (pop_size, n) in Omega -> Y: (pop_size, n) in [0,1]^n."""
    n = X.shape[1]
    idx = np.arange(1, n + 1, dtype=np.float64)
    a = -idx / 2.0
    b = idx / 2.0
    return (X - a[None, :]) / (b - a)[None, :]


def _theta(m, n):
    j = np.arange(m + 1, n + 1, dtype=np.float64)
    return 2.0 * PI * (j - (m + 1)) / n


# ==========================================================
# 2. g FUNCTIONS (Table 3), batched: YI (pop,m) -> (pop, n-m)
# ==========================================================
def g0_b(YI, m, n):
    return np.full((YI.shape[0], n - m), 0.2210)


def g1_b(YI, m, n):
    th = _theta(m, n)
    s = np.sin(1.5 * PI * YI[:, :, None] + th[None, None, :])
    return s.sum(axis=1) / (2 * m) + 0.5


def g2_b(YI, m, n):
    th = _theta(m, n)
    s = (YI[:, :, None] ** 2) * np.sin(4.5 * PI * YI[:, :, None] + th[None, None, :])
    return s.sum(axis=1) / (2 * m) + 0.5


def g3_b(YI, m, n):
    th = _theta(m, n)
    c = np.cos(PI * YI[:, :, None] + th[None, None, :]) ** 2
    return c.sum(axis=1) / m


def g4_b(YI, m, n):
    th = _theta(m, n)
    sYI = YI.sum(axis=1)
    S = (4.0 * PI / m) * sYI
    cosval = np.cos(S[:, None] + th[None, :])
    return (sYI[:, None] * cosval) / (2 * m) + 0.5


def g5_b(YI, m, n):
    th = _theta(m, n)
    s = np.sin(2.0 * PI * YI[:, :, None] - 1.0 + th[None, None, :]) ** 3
    return s.sum(axis=1) / (2 * m) + 0.5


def g6_b(YI, m, n):
    th = _theta(m, n)
    mean_sq = np.mean(YI ** 2, axis=1)
    term1 = -10.0 * np.exp(-0.4 * np.sqrt(mean_sq))
    meancos = np.mean(np.cos(11.0 * PI * YI[:, :, None] + th[None, None, :]), axis=1)
    num = term1[:, None] + 10.0 + E - np.exp(meancos ** 3)
    denom = -10.0 * np.exp(-0.4) - np.exp(-1.0) + 10.0 + E
    return num / denom


def g7_b(YI, m, n):
    th = _theta(m, n)
    meany = np.mean(YI, axis=1)
    term = np.exp(np.sin(7.0 * PI * meany[:, None] - PI / 2.0 + th[None, :]))
    return (meany[:, None] + term - np.exp(-1.0)) / (1.0 + E - np.exp(-1.0))


def g8_b(YI, m, n):
    th = _theta(m, n)
    s = np.abs(np.sin(2.5 * PI * (YI[:, :, None] - 0.5) + th[None, None, :]))
    return s.mean(axis=1)


def g9_b(YI, m, n):
    th = _theta(m, n)
    part2 = np.abs(np.sin(2.5 * PI * YI[:, :, None] - PI / 2.0 + th[None, None, :])).mean(axis=1)
    return 0.5 * np.mean(YI, axis=1)[:, None] - 0.5 * part2 + 0.5


def g10_b(YI, m, n):
    th = _theta(m, n)
    s = np.sin((4.0 * YI[:, :, None] - 2.0) * PI + th[None, None, :]).sum(axis=1)
    return (s ** 3) / (2.0 * m ** 3) + 0.5


G_FUNCS_B = {0: g0_b, 1: g1_b, 2: g2_b, 3: g3_b, 4: g4_b, 5: g5_b,
             6: g6_b, 7: g7_b, 8: g8_b, 9: g9_b, 10: g10_b}

COMPLICATED_G = {
    1: 4, 2: 5, 3: 2, 4: 7, 5: 9, 6: 4, 7: 5, 8: 2, 9: 7, 10: 9,
    11: 3, 12: 10, 13: 1, 14: 6, 15: 8, 16: 10, 17: 1, 18: 8, 19: 6, 20: 3,
}


def g_function_batch(problem_id, YI, m, n, complicated_ps):
    idx = COMPLICATED_G[problem_id] if complicated_ps else 0
    return G_FUNCS_B[idx](YI, m, n)


# ==========================================================
# 3. Z FUNCTIONS (Table 4), batched: W (pop,k) -> (pop,)
# ==========================================================
def Z1_b(W):
    return 10.0 * np.mean(W ** 2, axis=1)


def Z2_b(W):
    return 10.0 * np.max(np.abs(W), axis=1)


def Z3_b(W):
    return 10.0 * np.mean((W ** 2 - np.cos((2 * K_DISCONNECT - 1) * PI * W) + 1.0) / 3.0, axis=1)


def Z4_b(W):
    a = np.exp(np.sqrt(np.max(np.abs(W), axis=1)))
    b = np.exp(np.mean(0.5 * (np.cos((2 * K_DISCONNECT - 1) * PI * W) + 1.0), axis=1))
    return (10.0 / (2.0 * E - 2.0)) * (a - b - 1.0 + E)


def Z5_b(W):
    return -0.7 * Z3_b(W) + 10.0 * np.mean(np.abs(W) ** 0.002, axis=1)


def Z6_b(W):
    return -0.7 * Z4_b(W) + 10.0 * (np.mean(np.abs(W), axis=1)) ** 0.002


Z_FUNCS_B = {1: Z1_b, 2: Z2_b, 3: Z3_b, 4: Z4_b, 5: Z5_b, 6: Z6_b}


def Ji_indices(i, m, n, M):
    """Eq. 8: J_i = {j : (j-m-i) mod M == 0, j = m+1..n} (1-indexed j)."""
    j = np.arange(m + 1, n + 1)
    mask = ((j - m - i) % M) == 0
    return j[mask] - 1  # 0-indexed positions into y


def _disc_FM_b(y1, K=K_DISCONNECT):
    return (np.cos((2 * K - 1) * y1 * PI) + 2 * y1 + 4 * K * (1 - y1) - 1) / (4.0 * K)


# ==========================================================
# 4. F FUNCTIONS (Table 2), batched: YI (pop,m) -> F (pop, M)
# ==========================================================
def _prod_sin_b(YI, k):
    if k <= 0:
        return np.ones(YI.shape[0])
    return np.prod(np.sin(YI[:, :k] * PI / 2), axis=1)


def _prod_1mcos_b(YI, k):
    if k <= 0:
        return np.ones(YI.shape[0])
    return np.prod(1 - np.cos(YI[:, :k] * PI / 2), axis=1)


def _prod_1my_b(YI, k):
    if k <= 0:
        return np.ones(YI.shape[0])
    return np.prod(1 - YI[:, :k], axis=1)


def _prod_1msin_b(YI, k):
    if k <= 0:
        return np.ones(YI.shape[0])
    return np.prod(1 - np.sin(YI[:, :k] * PI / 2), axis=1)


def F_zcat1_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = _prod_sin_b(YI, M - 1)
    for j in range(2, M):
        F[:, j - 1] = _prod_sin_b(YI, M - j) * np.cos(YI[:, M - j] * PI / 2)
    F[:, M - 1] = 1 - np.sin(YI[:, 0] * PI / 2)
    return F


def F_zcat2_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = _prod_1mcos_b(YI, M - 1)
    for j in range(2, M):
        F[:, j - 1] = _prod_1mcos_b(YI, M - j) * (1 - np.sin(YI[:, M - j] * PI / 2))
    F[:, M - 1] = 1 - np.sin(YI[:, 0] * PI / 2)
    return F


def F_zcat3_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = np.mean(YI, axis=1)
    for j in range(2, M):
        k = M - j
        F[:, j - 1] = (np.sum(YI[:, :k], axis=1) + (1 - YI[:, k])) / (M - j + 1)
    F[:, M - 1] = 1 - YI[:, 0]
    return F


def F_zcat4_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, :M - 1] = YI
    F[:, M - 1] = 1 - np.mean(YI, axis=1)
    return F


def F_zcat5_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, :M - 1] = YI
    F[:, M - 1] = (np.exp(np.mean(1 - YI, axis=1)) ** 8 - 1) / (np.exp(1.0) ** 8 - 1)
    return F


def F_zcat6_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, :M - 1] = YI
    mu = np.mean(YI, axis=1)
    kappa, rho = 40.0, 0.05
    num = (1 + np.exp(2 * kappa * mu - kappa)) ** -1 - rho * mu - (1 + np.exp(kappa)) ** -1 + rho
    den = (1 + np.exp(-kappa)) ** -1 - (1 + np.exp(kappa)) ** -1 + rho
    F[:, M - 1] = num / den
    return F


def F_zcat7_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, :M - 1] = YI
    F[:, M - 1] = np.sum((0.5 - YI) ** 5, axis=1) / (2 * (M - 1) * 0.5 ** 5) + 0.5
    return F


def F_zcat8_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = 1 - _prod_1msin_b(YI, M - 1)
    for j in range(2, M):
        F[:, j - 1] = 1 - _prod_1msin_b(YI, M - j) * (1 - np.cos(YI[:, M - j] * PI / 2))
    F[:, M - 1] = np.cos(YI[:, 0] * PI / 2)
    return F


def F_zcat9_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = np.mean(np.sin(YI * PI / 2), axis=1)
    for j in range(2, M):
        k = M - j
        F[:, j - 1] = (np.sum(np.sin(YI[:, :k] * PI / 2), axis=1) + np.cos(YI[:, k] * PI / 2)) / (M - j + 1)
    F[:, M - 1] = np.cos(YI[:, 0] * PI / 2)
    return F


def F_zcat10_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, :M - 1] = YI
    rho = 0.02
    num = rho ** -1 - (np.mean(1 - YI, axis=1) + rho) ** -1
    den = rho ** -1 - (1 + rho) ** -1
    F[:, M - 1] = num / den
    return F


def F_zcat11_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = np.mean(YI, axis=1)
    for j in range(2, M):
        k = M - j
        F[:, j - 1] = (np.sum(YI[:, :k], axis=1) + (1 - YI[:, k])) / (M - j + 1)
    F[:, M - 1] = _disc_FM_b(YI[:, 0])
    return F


def F_zcat12_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = 1 - _prod_1my_b(YI, M - 1)
    for j in range(2, M):
        k = M - j
        F[:, j - 1] = 1 - _prod_1my_b(YI, k) * YI[:, k]
    F[:, M - 1] = _disc_FM_b(YI[:, 0])
    return F


def F_zcat13_b(YI, M):
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = 1 - np.mean(np.sin(YI * PI / 2), axis=1)
    for j in range(2, M):
        k = M - j
        F[:, j - 1] = 1 - (np.sum(np.sin(YI[:, :k] * PI / 2), axis=1) + np.cos(YI[:, k] * PI / 2)) / (M - j + 1)
    F[:, M - 1] = 1 - _disc_FM_b(YI[:, 0])
    return F


def F_zcat14_b(YI, M):
    y1 = YI[:, 0]
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = np.sin(y1 * PI / 2) ** 2
    for j in range(2, M - 1):
        F[:, j - 1] = np.sin(y1 * PI / 2) ** (2 + (j - 1) / (M - 2))
    if M > 2:
        F[:, M - 2] = 0.5 * (1 + np.sin(6 * y1 * PI / 2 - PI / 2))
    F[:, M - 1] = np.cos(y1 * PI / 2)
    return F


def F_zcat15_b(YI, M):
    y1 = YI[:, 0]
    F = np.zeros((YI.shape[0], M))
    for j in range(1, M):
        F[:, j - 1] = y1 ** (1 + (j - 1) / (4.0 * M))
    F[:, M - 1] = _disc_FM_b(y1)
    return F


def F_zcat16_b(YI, M):
    y1 = YI[:, 0]
    F = np.zeros((YI.shape[0], M))
    F[:, 0] = np.sin(y1 * PI / 2)
    for j in range(2, M - 1):
        F[:, j - 1] = np.sin(y1 * PI / 2) ** (1 + (j - 1) / (M - 2))
    if M > 2:
        F[:, M - 2] = 0.5 * (1 + np.sin(10 * y1 * PI / 2 - PI / 2))
    F[:, M - 1] = _disc_FM_b(y1)
    return F


def F_zcat17_b(YI, M):
    y0 = YI[:, 0]
    cond = np.all(YI <= 0.5, axis=1)
    F = np.zeros((YI.shape[0], M))
    F[:, :M - 1] = np.where(cond[:, None], y0[:, None], YI)
    wedge_last = (np.exp(1 - y0) ** 8 - 1) / (np.exp(1.0) ** 8 - 1)
    nonwedge_last = (np.exp(np.mean(1 - YI, axis=1)) ** 8 - 1) / (np.exp(1.0) ** 8 - 1)
    F[:, M - 1] = np.where(cond, wedge_last, nonwedge_last)
    return F


def F_zcat18_b(YI, M):
    y0 = YI[:, 0]
    cond = np.all(YI <= 0.4, axis=1) | np.all(YI >= 0.6, axis=1)
    F = np.zeros((YI.shape[0], M))
    F[:, :M - 1] = np.where(cond[:, None], y0[:, None], YI)
    wedge_last = ((0.5 - y0) ** 5 + 0.5 ** 5) / (2 * 0.5 ** 5)
    nonwedge_last = np.sum((0.5 - YI) ** 5, axis=1) / (2 * (M - 1) * 0.5 ** 5) + 0.5
    F[:, M - 1] = np.where(cond, wedge_last, nonwedge_last)
    return F


F_STANDARD_B = {
    1: F_zcat1_b, 2: F_zcat2_b, 3: F_zcat3_b, 4: F_zcat4_b, 5: F_zcat5_b, 6: F_zcat6_b,
    7: F_zcat7_b, 8: F_zcat8_b, 9: F_zcat9_b, 10: F_zcat10_b, 11: F_zcat11_b, 12: F_zcat12_b,
    13: F_zcat13_b, 17: F_zcat17_b, 18: F_zcat18_b,
}
F_DEGENERATE_M1_B = {14: F_zcat14_b, 15: F_zcat15_b, 16: F_zcat16_b}
DYNAMIC_M_PROBLEMS = {19, 20}


def F_zcat19_true_b(Y, M, A=ZCAT19_A):
    """Y: (pop, M-1) random y_I samples -> F: (pop, M), for true-front sampling."""
    y1 = Y[:, 0]
    mask = ((y1 >= 0.0) & (y1 <= 0.2)) | ((y1 >= 0.4) & (y1 <= 0.6))
    F_m1 = np.zeros((Y.shape[0], M))
    F_m1[:, :M - 1] = y1[:, None]
    F_m1[:, M - 1] = 1 - y1 - np.cos(2 * A * PI * y1 + PI / 2) / (2 * A * PI)
    meany = np.mean(Y, axis=1)
    F_mM1 = np.zeros((Y.shape[0], M))
    F_mM1[:, :M - 1] = Y
    F_mM1[:, M - 1] = 1 - meany - np.cos(2 * A * PI * meany + PI / 2) / (2 * A * PI)
    return np.where(mask[:, None], F_m1, F_mM1)


def F_zcat20_true_b(Y, M):
    y1 = Y[:, 0]
    mask = ((y1 >= 0.1) & (y1 <= 0.4)) | ((y1 >= 0.6) & (y1 <= 0.9))
    F_m1 = np.zeros((Y.shape[0], M))
    F_m1[:, :M - 1] = y1[:, None]
    F_m1[:, M - 1] = ((0.5 - y1) ** 5 + 0.5 ** 5) / (2 * 0.5 ** 5)
    F_mM1 = np.zeros((Y.shape[0], M))
    F_mM1[:, :M - 1] = Y
    F_mM1[:, M - 1] = np.sum((0.5 - Y) ** 5, axis=1) / (2 * (M - 1) * 0.5 ** 5) + 0.5
    return np.where(mask[:, None], F_m1, F_mM1)


# ==========================================================
# 5. FULL OBJECTIVE EVALUATION (batched): f = alpha + beta
# ==========================================================
def _alpha_beta_fixed_m(problem_id, Y, m, M, n, complicated_ps, level, bias, imbalance, F):
    idx = np.arange(1, M + 1, dtype=np.float64)
    alpha = (idx[None, :] ** 2) * F

    YI = Y[:, :m]
    YII = Y[:, m:n]
    if YII.shape[1] == 0:
        return alpha

    gvals = g_function_batch(problem_id, YI, m, n, complicated_ps)
    Z = YII - gvals
    W_full = np.abs(Z) ** BIAS_GAMMA if bias else Z

    pop = Y.shape[0]
    beta = np.zeros((pop, M))
    for i in range(1, M + 1):
        Ji = Ji_indices(i, m, n, M)
        if len(Ji) == 0:
            continue
        w = W_full[:, Ji - m]
        if imbalance:
            zf = Z4_b if (i % 2 == 0) else Z1_b
        else:
            zf = Z_FUNCS_B[level]
        beta[:, i - 1] = (i ** 2) * zf(w)

    return alpha + beta


def _evaluate_dynamic_m(problem_id, Y, M, n, complicated_ps, level, bias, imbalance, A=ZCAT19_A):
    """
    ZCAT19/20: m is 1 for individuals whose y1 falls in certain ranges, else M-1.
    Computed for BOTH branches fully vectorized, then selected per-individual
    with a boolean mask -- no per-row Python loop needed.
    """
    y1 = Y[:, 0]
    if problem_id == 19:
        mask = ((y1 >= 0.0) & (y1 <= 0.2)) | ((y1 >= 0.4) & (y1 <= 0.6))
    else:
        mask = ((y1 >= 0.1) & (y1 <= 0.4)) | ((y1 >= 0.6) & (y1 <= 0.9))

    pop = Y.shape[0]

    # branch m = 1
    m1 = 1
    F1 = np.zeros((pop, M))
    if problem_id == 19:
        F1[:, :M - 1] = y1[:, None]
        F1[:, M - 1] = 1 - y1 - np.cos(2 * A * PI * y1 + PI / 2) / (2 * A * PI)
    else:
        F1[:, :M - 1] = y1[:, None]
        F1[:, M - 1] = ((0.5 - y1) ** 5 + 0.5 ** 5) / (2 * 0.5 ** 5)
    obj_m1 = _alpha_beta_fixed_m(problem_id, Y, m1, M, n, complicated_ps, level, bias, imbalance, F1)

    # branch m = M-1
    mM1 = M - 1
    YI_full = Y[:, :mM1]
    FMm1 = np.zeros((pop, M))
    if problem_id == 19:
        meany = np.mean(YI_full, axis=1)
        FMm1[:, :M - 1] = YI_full
        FMm1[:, M - 1] = 1 - meany - np.cos(2 * A * PI * meany + PI / 2) / (2 * A * PI)
    else:
        FMm1[:, :M - 1] = YI_full
        FMm1[:, M - 1] = np.sum((0.5 - YI_full) ** 5, axis=1) / (2 * (M - 1) * 0.5 ** 5) + 0.5
    obj_mM1 = _alpha_beta_fixed_m(problem_id, Y, mM1, M, n, complicated_ps, level, bias, imbalance, FMm1)

    return np.where(mask[:, None], obj_m1, obj_mM1)


def evaluate_population(problem_id, X, M, n, complicated_ps=False, level=1,
                         bias=False, imbalance=False):
    """X: (pop_size, n) in Omega -> F: (pop_size, M). Fully vectorized."""
    Y = x_to_y_batch(X)

    if problem_id in F_STANDARD_B:
        m = M - 1
        F = F_STANDARD_B[problem_id](Y[:, :m], M)
        return _alpha_beta_fixed_m(problem_id, Y, m, M, n, complicated_ps, level, bias, imbalance, F)

    if problem_id in F_DEGENERATE_M1_B:
        m = 1
        F = F_DEGENERATE_M1_B[problem_id](Y[:, :m], M)
        return _alpha_beta_fixed_m(problem_id, Y, m, M, n, complicated_ps, level, bias, imbalance, F)

    if problem_id in DYNAMIC_M_PROBLEMS:
        return _evaluate_dynamic_m(problem_id, Y, M, n, complicated_ps, level, bias, imbalance)

    raise ValueError(f"Unknown problem id {problem_id}")

# ==========================================================
# 6. TRUE PARETO FRONT SAMPLING (beta = 0 by construction), batched
# ==========================================================
def _balanced_factor_pair(n):
    """Return (r, c) with r*c == n and r, c as close to each other (i.e.
    to sqrt(n)) as possible -- used to build a deterministic rectangular
    grid of exactly n points instead of a random sample."""
    best = (1, n)
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            best = (i, n // i)
    return best


def generate_true_front(problem_id, M, n_pts=40, seed=0):
    # Isolated random number generator for safety
    rng = np.random.default_rng(seed)
    
    # Calculate sample counts safely for scaling dimensions
    n_samples = n_pts ** min(M - 1, 3) if M - 1 <= 3 else n_pts * 200
    n_samples = max(n_samples, 500)

    # Handle degenerate ZCAT problem types
    if problem_id in F_DEGENERATE_M1_B:
        y1_grid = np.linspace(0, 1, n_samples)[:, None]
        F = F_DEGENERATE_M1_B[problem_id](y1_grid, M)
        return F * (np.arange(1, M + 1) ** 2)[None, :]

    # Handle dynamic ZCAT problem types (Problems 19 and 20)
    if problem_id in DYNAMIC_M_PROBLEMS:
        # Use Latin Hypercube Sampling instead of uniform random for clean space-filling
        sampler = qmc.LatinHypercube(d=M - 1, seed=seed)
        Y = sampler.random(n=n_samples)
        
        F = F_zcat19_true_b(Y, M) if problem_id == 19 else F_zcat20_true_b(Y, M)
        return F * (np.arange(1, M + 1) ** 2)[None, :]

    # Handle standard ZCAT problems
    if M - 1 == 1:
        # 1D: always a deterministic, evenly-spaced line -- use exactly
        # max(n_pts, n_samples) points so the floor is honored without
        # ever falling back to random sampling.
        grid_pts = max(n_pts, n_samples)
        Y = np.linspace(0, 1, grid_pts)[:, None]
    elif M - 1 == 2:
        if n_pts ** 2 >= n_samples:
            r = c = n_pts
        else:
            # Deterministic rectangular grid with r*c == n_samples exactly,
            # r and c chosen as close to each other (i.e. as close to a
            # square grid) as possible -- keeps the true front visually
            # evenly spread (like the plain n_pts x n_pts mesh) instead of
            # falling back to Latin Hypercube Sampling, which looks clumpy
            # at low-to-moderate point counts.
            r, c = _balanced_factor_pair(n_samples)
        grid_r = np.linspace(0, 1, r)
        grid_c = np.linspace(0, 1, c)
        mesh = np.meshgrid(grid_r, grid_c)
        Y = np.stack([g.ravel() for g in mesh], axis=1)
    else:
        # Higher dimensions use Latin Hypercube Sampling to avoid the curse of dimensionality
        sampler = qmc.LatinHypercube(d=M - 1, seed=seed)
        Y = sampler.random(n=n_samples)
        
    F = F_STANDARD_B[problem_id](Y, M)
    return F * (np.arange(1, M + 1) ** 2)[None, :]


# ==========================================================
# 7. VECTORIZED ARCHIVE MECHANICS (non-dominated sort + crowding distance)
# ==========================================================
def fast_non_dominated_indices(F):
    """F: (n, M). Returns indices of the non-dominated set. Fully vectorized."""
    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    Fi = F[:, None, :]
    Fj = F[None, :, :]
    le = np.all(Fj <= Fi, axis=2)
    lt = np.any(Fj < Fi, axis=2)
    dom = le & lt
    np.fill_diagonal(dom, False)
    dominated_count = dom.sum(axis=1)
    return np.where(dominated_count == 0)[0]


def crowding_distance(F):
    """Fully vectorized crowding distance (no per-solution Python loop)."""
    n_sols, n_objs = F.shape
    crowding = np.zeros(n_sols)
    if n_sols <= 2:
        crowding[:] = np.inf
        return crowding
    for m_ in range(n_objs):
        order = np.argsort(F[:, m_])
        f_sorted = F[order, m_]
        crowding[order[0]] = np.inf
        crowding[order[-1]] = np.inf
        f_min, f_max = f_sorted[0], f_sorted[-1]
        if f_max - f_min < 1e-12:
            continue
        diffs = (f_sorted[2:] - f_sorted[:-2]) / (f_max - f_min)
        crowding[order[1:-1]] += diffs
    return crowding


# ==========================================================
# 7.5 REFERENCE-DIRECTION / PBI ARCHIVE TRUNCATION
#
#     Crowding-distance truncation gives infinite priority to
#     boundary/extreme non-dominated points every generation, so a
#     poorly-converged extreme solution can survive in the archive
#     indefinitely once it is topologically "outermost" -- and because
#     rao_update() draws its exploitation target (x_best) as a *random*
#     archive member, that residual convergence error keeps feeding back
#     into the population. Crowding distance has no term that rewards
#     getting *closer* to the front, only spread.
#
#     Reference-direction niching (Das & Dennis simplex-lattice
#     directions, as in NSGA-III) fixes this: each archive slot is tied
#     to a fixed direction in normalized objective space (so diversity is
#     still enforced, independent of population size), and the survivor
#     within each niche is chosen by PBI (Penalty-based Boundary
#     Intersection: d_parallel + theta * d_perpendicular) instead of by
#     "furthest from its neighbours". d_parallel gives real convergence
#     pressure -- the one thing crowding distance was missing.
# ==========================================================
def _das_dennis(M, H):
    """Das & Dennis (1998) simplex-lattice reference directions: all
    M-tuples of non-negative multiples of 1/H that sum to 1."""
    def rec(m, left, cur, out):
        if m == 1:
            out.append(cur + [left])
            return
        for i in range(left + 1):
            rec(m - 1, left - i, cur + [i], out)
    out = []
    rec(M, H, [], out)
    return np.array(out, dtype=np.float64) / H


def generate_reference_directions(M, n_target):
    """Smallest Das-Dennis lattice (by depth H) with at least n_target
    directions for M objectives. Cached per (M, n_target) by the caller."""
    H = 1
    while len(_das_dennis(M, H + 1)) <= n_target:
        H += 1
    return _das_dennis(M, H)


def pbi_truncate(F, nd_idx, size, ref_dirs, theta=1.0):
    """Reference-direction niching with PBI (d_parallel + theta *
    d_perpendicular) as the within-niche survivor criterion.

    F        : (n, M) full combined objective matrix
    nd_idx   : indices (into F) of the non-dominated set to select from
    size     : number of survivors to keep (== archive_size)
    ref_dirs : (n_ref, M) unit-normalized reference directions
    theta    : PBI penalty weight on the perpendicular (diversity) term;
               theta=1 validated as robust across the ZCAT suite (theta
               in [0.5, 2] all within run-to-run noise of each other).
    """
    Fn = F[nd_idx]
    lo = Fn.min(axis=0)
    span = Fn.max(axis=0) - lo
    span[span <= 1e-12] = 1.0
    A = (Fn - lo) / span                      # normalize each objective to [0,1]
    A_norm = np.linalg.norm(A, axis=1)
    A_unit = A / (A_norm[:, None] + 1e-12)
    # Das-Dennis directions sum to 1 but are NOT unit-norm (their L2 norm
    # varies with how many components are nonzero) -- they must be
    # re-normalized before use as the cosine's right-hand side, or `cos`
    # below isn't a real cosine and the whole niche assignment breaks.
    ref_unit = ref_dirs / (np.linalg.norm(ref_dirs, axis=1, keepdims=True) + 1e-12)
    cos = A_unit @ ref_unit.T                 # (n_nd, n_ref)
    assoc = cos.argmax(axis=1)
    cos_a = cos[np.arange(len(A)), assoc]
    d_perp = A_norm * np.sqrt(np.maximum(0.0, 1.0 - cos_a ** 2))
    d_par = A_norm * cos_a
    pbi = d_par + theta * d_perp

    chosen = []
    niche_count = np.zeros(len(ref_dirs), dtype=int)
    remaining = np.ones(len(A), dtype=bool)
    while len(chosen) < size and remaining.any():
        active = np.unique(assoc[remaining])
        # fill least-represented niches first, exactly like NSGA-III's
        # niche-preservation step
        for nz in active[np.argsort(niche_count[active], kind='mergesort')]:
            if len(chosen) >= size:
                break
            cand = np.where(remaining & (assoc == nz))[0]
            if len(cand) == 0:
                continue
            pick = cand[np.argmin(pbi[cand])]
            chosen.append(pick)
            remaining[pick] = False
            niche_count[nz] += 1
    return nd_idx[np.array(chosen, dtype=int)]


def truncate_archive(problem_id, combined_F, nd_idx, archive_size, ref_dirs_cache,
                     use_pbi=True, theta=1.0):
    """Reference-direction/PBI niching for archive truncation, replacing
    plain crowding-distance truncation (see module comment above).
    use_pbi=False (ablation) falls back to crowding-distance truncation,
    exactly as in the baseline Proposed MOO."""
    if len(nd_idx) <= archive_size:
        return nd_idx
    if not use_pbi:
        cd = crowding_distance(combined_F[nd_idx])
        return nd_idx[np.argsort(-cd)[:archive_size]]
    M = combined_F.shape[1]
    key = (M, archive_size)
    if key not in ref_dirs_cache:
        ref_dirs_cache[key] = generate_reference_directions(M, archive_size)
    return pbi_truncate(combined_F, nd_idx, archive_size, ref_dirs_cache[key], theta=theta)


# ==========================================================
# 8. MO-RAO ALGORITHM (vectorized position update)
# ==========================================================
def rao_update(X, bounds, archive_X, rng, cf=1.0):
    """cf: annealing factor in (0, 1], applied to the exploratory
    (X - x_rand) term. Without it this term keeps injecting
    full-strength perturbation regardless of how converged the
    population already is, which stalls decision variables that
    need to settle on an exact value (e.g. ZCAT g0 = 0.2210).
    Decaying cf towards 0 lets exploitation (the x_best term)
    dominate late in the run."""
    pop_size = X.shape[0]
    if len(archive_X) > 0:
        best_idx = rng.integers(0, len(archive_X), size=pop_size)
        x_best = archive_X[best_idx]
    else:
        x_best = X
    rand_idx = rng.integers(0, pop_size, size=pop_size)
    x_rand = X[rand_idx]
    r1 = rng.random((pop_size, 1))
    r2 = rng.random((pop_size, 1))
    new_val = X + r1 * (x_best - X) + cf * r2 * (X - x_rand)
    return np.clip(new_val, bounds[:, 0], bounds[:, 1])


def boundary_directed_sampling(archive_X, bounds, rng, k_elites=5, n_samples=2):
    """Generic, benchmark-agnostic directional sampling: draws elites
    directly from the current archive, and for each elite samples
    `n_samples` points along the segment toward the lower bound L and
    `n_samples` toward the upper bound U, each at a random fraction rho
    in [0,1) of the way there -- candidates that stay strictly inside
    the current bounds while explicitly covering low-density boundary
    regions, cheaply (O(k_elites * n_samples) extra evaluations per
    iteration, independent of n)."""
    n_arch, n_vars = archive_X.shape
    if n_arch == 0:
        return np.empty((0, n_vars))
    idx = rng.integers(0, n_arch, size=min(k_elites, n_arch))
    elites = archive_X[idx][:, None, :]                      # (k, 1, n)
    L = bounds[:, 0][None, None, :]                           # (1, 1, n)
    U = bounds[:, 1][None, None, :]
    rho = rng.random((elites.shape[0], n_samples, 1))         # (k, ns, 1)
    cand_L = elites + rho * (L - elites)                      # (k, ns, n)
    cand_U = elites + rho * (U - elites)
    cand = np.concatenate([cand_L.reshape(-1, n_vars), cand_U.reshape(-1, n_vars)], axis=0)
    return np.clip(cand, bounds[:, 0], bounds[:, 1])


def multi_objective_rao(problem_id, M, n, population_size, max_iterations,
                         archive_size, complicated_ps=False, level=1,
                         bias=False, imbalance=False, metric_interval=100,
                         seed=None, verbose=False,
                         bd_sampling=True, bd_k_elites=5, bd_n_samples=2,
                         use_pbi=True, use_annealing=True, pbi_theta=1.0, cf_floor=0.5):
    rng = np.random.default_rng(seed)
    bounds = bounds_for(n)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], (population_size, n))
    F = evaluate_population(problem_id, X, M, n, complicated_ps, level, bias, imbalance)

    archive_X = np.empty((0, n))
    archive_F = np.empty((0, M))
    ref_dirs_cache = {}

    metric_iterations, archives_at_metrics = [], []

    for it in range(max_iterations):
        combined_X = np.vstack([archive_X, X]) if len(archive_X) else X
        combined_F = np.vstack([archive_F, F]) if len(archive_F) else F

        # Boundary-directed sampling from the current archive (see
        # boundary_directed_sampling docstring). Adds a small, fixed
        # number of extra evaluations per iteration.
        if bd_sampling and len(archive_X) > 0:
            bd_X = boundary_directed_sampling(archive_X, bounds, rng,
                                               k_elites=bd_k_elites, n_samples=bd_n_samples)
            bd_F = evaluate_population(problem_id, bd_X, M, n, complicated_ps, level, bias, imbalance)
            combined_X = np.vstack([combined_X, bd_X])
            combined_F = np.vstack([combined_F, bd_F])

        nd_idx = fast_non_dominated_indices(combined_F)
        sel = truncate_archive(problem_id, combined_F, nd_idx, archive_size, ref_dirs_cache,
                               use_pbi=use_pbi, theta=pbi_theta)

        archive_X, archive_F = combined_X[sel], combined_F[sel]

        if it % metric_interval == 0 or it == max_iterations - 1:
            metric_iterations.append(it)
            archives_at_metrics.append(archive_F.copy())
            if verbose:
                print(f"    Iter {it+1}/{max_iterations}: archive size = {len(archive_F)}")

        # Linear annealing of the exploratory term, floor 0.5: cf runs
        # from 1.0 down to 0.5 over the course of the run. This avoids
        # the long-budget plateau seen with no annealing (ZCAT g0=0.2210
        # distance variables stalling at |y-g|~0.06) while not starving
        # short-budget runs of exploration too early.
        progress = (it + 1) / max_iterations
        # use_annealing=False (ablation) keeps c_f = 1; cf_floor is the end value (default 0.5)
        cf = (1.0 - (1.0 - cf_floor) * progress) if use_annealing else 1.0
        X = rao_update(X, bounds, archive_X, rng, cf=cf)
        F = evaluate_population(problem_id, X, M, n, complicated_ps, level, bias, imbalance)

    return archive_X, archive_F, metric_iterations, archives_at_metrics


# ==========================================================
# 9. METRICS (GD, GD+, IGD, IGD+, HV, Spacing, Spread) -- M-agnostic
# ==========================================================
def calculate_metrics_unified(true_pf, algo_pf):
    keys = ['GD', 'GD+', 'IGD', 'IGD+', 'HV', 'Spacing', 'Spread']
    if len(algo_pf) == 0 or len(true_pf) == 0:
        return {k: np.nan for k in keys}

    algo_np, true_np = np.asarray(algo_pf), np.asarray(true_pf)
    min_v, max_v = true_np.min(0), true_np.max(0)
    rng = max_v - min_v
    rng[rng == 0] = 1.0
    a, t = (algo_np - min_v) / rng, (true_np - min_v) / rng

    d_at = a[:, None, :] - t[None, :, :]
    dist_at = np.sqrt((d_at ** 2).sum(-1))
    d_ta = t[:, None, :] - a[None, :, :]
    dist_ta = np.sqrt((d_ta ** 2).sum(-1))

    # Plain GD / IGD (Euclidean, normalized space)
    gd = dist_at.min(1).mean()
    igd = dist_ta.min(1).mean()

    # Dominance-compliant distance of Ishibuchi et al. (2015):
    #   d+(a, z) = sqrt( sum_k max(a_k - z_k, 0)^2 )   (minimization)
    # GD+  : for every obtained solution a, take the MIN d+(a, z) over all
    #        reference points z, then average over the solutions.
    # IGD+ : for every reference point z, take the MIN d+(a, z) over all
    #        obtained solutions a, then average over the reference points.
    # The minimum is taken over d+ itself (not over the Euclidean nearest point).
    dplus_at = np.sqrt((np.maximum(d_at, 0.0) ** 2).sum(-1))   # (|A|, |T|), d_at = a - t
    dplus_ta = np.sqrt((np.maximum(-d_ta, 0.0) ** 2).sum(-1))  # (|T|, |A|), -d_ta = a - t
    gd_plus = dplus_at.min(1).mean()
    igd_plus = dplus_ta.min(1).mean()

    try:
        hv = float(HV(ref_point=np.array([1.1] * true_np.shape[1]))(a))
    except Exception:
        hv = np.nan

    spacing = 0.0
    if len(a) > 1:
        dm = np.sqrt(((a[:, None] - a) ** 2).sum(-1))
        np.fill_diagonal(dm, np.inf)
        spacing = np.std(dm.min(1))

    spread = 0.0
    if len(a) > 1:
        extremes = []
        for k in range(a.shape[1]):
            ref = np.zeros(a.shape[1]); ref[k] = 1.0
            extremes.append(np.sqrt(((a - ref) ** 2).sum(1)).min())
        df_dl = sum(extremes)
        order = np.argsort(a[:, 0])
        sa = a[order]
        cons = np.sqrt(((sa[1:] - sa[:-1]) ** 2).sum(1))
        d_mean = cons.mean() if len(cons) else 0
        num = df_dl + np.sum(np.abs(cons - d_mean))
        den = df_dl + len(cons) * d_mean
        spread = num / (den + 1e-10)

    return {'GD': gd, 'GD+': gd_plus, 'IGD': igd, 'IGD+': igd_plus,
            'HV': hv, 'Spacing': spacing, 'Spread': spread}

# ==========================================================
# 10. PARALLEL RUN WRAPPER (same seeding scheme: run_id*9876+123)
# ==========================================================
def run_single_optimization(args):
    (run_id, problem_id, M, n, pop_size, max_iter, arch_size,
     complicated_ps, level, bias, imbalance, met_interval, true_pf,
     bd_sampling, bd_k_elites, bd_n_samples) = args[:16]
    # ablation / sensitivity switches (defaults = full algorithm)
    use_pbi, use_annealing, pbi_theta, cf_floor = (list(args[16:]) + [True, True, 1.0, 0.5][len(args[16:]):])

    # 1. Deterministic seed calculation based on run_id for absolute reproducibility
    seed = run_id * 9876 + 123
    
    # 2. Force local process states to decouple from parent process fork/spawn states
    np.random.seed(seed)
    random.seed(seed)

    t0 = time.time()
    # 3. Execute the algorithm passing the explicit seed to its internal scopes
    archive_X, archive_F, metric_iters, archives_at_metrics = multi_objective_rao(
        problem_id, M, n, pop_size, max_iter, arch_size,
        complicated_ps, level, bias, imbalance, met_interval, seed=seed, verbose=False,
        bd_sampling=bd_sampling, bd_k_elites=bd_k_elites, bd_n_samples=bd_n_samples,
        use_pbi=use_pbi, use_annealing=use_annealing, pbi_theta=pbi_theta, cf_floor=cf_floor)
    run_time = time.time() - t0

    run_metrics = {k: [] for k in ["GD", "GD+", "IGD", "IGD+", "HV", "Spacing", "Spread"]}
    
    # 4. Snapshot evaluation loop safely guarding against empty results
    for snap in archives_at_metrics:
        # Check if the optimization snapshot is empty or invalid to prevent a metric crash
        if snap is None or len(snap) == 0:
            for k in run_metrics:
                run_metrics[k].append(np.nan)
            continue
            
        m = calculate_metrics_unified(true_pf, snap)
        for k in run_metrics:
            run_metrics[k].append(m[k])

    return {
        'run_id': run_id, 
        'run_time': run_time, 
        'run_metrics': run_metrics,
        'final_pareto_front': archive_F, 
        'metric_iterations': metric_iters,
    }

# ==========================================================
# 11. PLOTTING -- journal-ready, 300 DPI, saved as PNG + PDF
# ==========================================================
def _apply_plot_style():
    """Journal-ready styling: bold, legible text/lines even at 300 DPI,
    sized for the larger (12x6-style) figures used below."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.labelweight': 'bold',
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 17,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.5,
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
        'figure.dpi': 100,        # on-screen preview
        'savefig.dpi': 300,       # exported files
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
    })


def _save_and_close(out_dir, stem):
    """Save PNG + PDF (both at 300 DPI) into out_dir, show, then close the
    current figure to free memory."""
    plt.savefig(os.path.join(out_dir, f'{stem}.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, f'{stem}.pdf'), dpi=300)
    plt.show()
    plt.close()


def generate_plots(problem_id, M, avg_metrics, evals, final_pf, viz_pf,
                    out_dir='plots'):
    """
    Produces four journal-ready figure types -- convergence metrics,
    hypervolume, Pareto front (2D/3D/parallel-coordinates depending on M),
    and saves each at 300 DPI as both .png and .pdf into `out_dir`
    (created if needed). Every figure is closed after saving so repeated
    calls (e.g. across problems) don't accumulate open figures in memory.
    """
    os.makedirs(out_dir, exist_ok=True)
    _apply_plot_style()

    # 1. Convergence Metrics Plot
    plt.figure(figsize=(12, 6))
    for key, style in [('GD', 'b-o'), ('GD+', 'c--o'), ('IGD', 'g-s'),
                        ('IGD+', 'y--s'), ('Spacing', 'm-^'), ('Spread', 'k-v')]:
        if key in avg_metrics:
            plt.plot(evals, avg_metrics[key], style, label=key, linewidth=1.8,
                      markevery=max(1, len(evals) // 10))
    plt.title(f'ZCAT{problem_id} Convergence Modified-Proposed MOO',fontsize=16, fontweight='bold')
    plt.xlabel('Function Evaluations', fontsize=16)
    plt.ylabel('Metric Value', fontsize=16)
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)
    plt.grid(False)
    plt.legend(frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9,
               loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=14)
    # plt.legend(frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
    plt.tight_layout()
    _save_and_close(out_dir, f'ZCAT{problem_id}_Convergence')

    # 2. Hypervolume Plot
    plt.figure(figsize=(12, 6))
    plt.plot(evals, avg_metrics['HV'], 'r-D', label='Hypervolume', linewidth=2,
              markevery=max(1, len(evals) // 10))
    plt.title(f'ZCAT{problem_id} Hypervolume Convergence Modified-Proposed MOO',fontsize=16, fontweight='bold')
    plt.xlabel('Function Evaluations', fontsize=16)
    plt.ylabel('Hypervolume', fontsize=16)
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)
    plt.grid(False)
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9, fontsize=14)
    # plt.legend(frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
    plt.tight_layout()
    _save_and_close(out_dir, f'ZCAT{problem_id}_HV')

    # 3. Pareto Front Visualization based on Objectives (M)
    if M == 2:
        plt.figure(figsize=(8, 6.5))
        plt.scatter(viz_pf[:, 0], viz_pf[:, 1], c='#0072B2', s=18, alpha=0.5, label='True PF', zorder=1)
        plt.scatter(final_pf[:, 0], final_pf[:, 1], c='#D55E00', s=40, edgecolor='black',
                    linewidth=0.6, label='Modified-Proposed PF', zorder=2)
        plt.xlabel(r'$f_1$')
        plt.ylabel(r'$f_2$')
        plt.title(f'ZCAT{problem_id} Pareto Front (M=2)', fontweight='bold')
        plt.grid(False)
        plt.legend(frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
        plt.tight_layout()
        _save_and_close(out_dir, f'ZCAT{problem_id}_PF_2D')

    elif M == 3:
        fig = plt.figure(figsize=(10.5, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(viz_pf[:, 0], viz_pf[:, 1], viz_pf[:, 2], c='royalblue', alpha=0.8, s=12, label='True PF',zorder=1)
        ax.scatter(final_pf[:, 0], final_pf[:, 1], final_pf[:, 2], c='crimson', s=40,
                   edgecolor='black', linewidth=0.5, alpha=0.95, label='Modified-Proposed PF',zorder=2)
        ax.zaxis.set_rotate_label(False)
        ax.set_xlabel(r'$f_1$', labelpad=12, fontsize=18, fontweight='bold')
        ax.set_ylabel(r'$f_2$', labelpad=12, fontsize=18, fontweight='bold')
        ax.set_zlabel(r'$f_3$', labelpad=-4, fontsize=18, fontweight='bold', rotation=0, ha="right", va="center")
        ax.set_title(f'ZCAT{problem_id} Pareto Front Modified-Proposed MOO', fontsize=16, fontweight='bold')
        ax.view_init(elev=25, azim=45)
        ax.tick_params(labelsize=15)
        leg =ax.legend(frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9, prop={'size': 16})
        leg.legend_handles[0].set_sizes([12.0])  # Enlarges the tiny blue dot to size 40
        leg.legend_handles[0].set_alpha(1.0)     # Makes the blue dot fully solid
        ax.grid(False)
        plt.tight_layout()
        # fig.subplots_adjust(left=0.15)
        _save_and_close(out_dir, f'ZCAT{problem_id}_PF_3D')

    else:
        plt.figure(figsize=(12, 6))
        for row in viz_pf[:min(len(viz_pf), 300)]:
            plt.plot(range(1, M + 1), row, color='#0072B2', alpha=0.08, linewidth=1)
        plt.plot([], [], color='#0072B2', alpha=0.6, linewidth=1.8, label='True PF')
        for row in final_pf:
            plt.plot(range(1, M + 1), row, color='#D55E00', alpha=0.4, linewidth=1.4)
        plt.plot([], [], color='#D55E00', alpha=0.8, linewidth=1.8, label='Modified-Proposed PF')
        plt.xticks(range(1, M + 1), [f'$f_{{{i}}}$' for i in range(1, M + 1)])
        plt.title(f'ZCAT{problem_id} Parallel Coordinates (M={M})', fontweight='bold')
        plt.xlabel('Objective Index')
        plt.ylabel('Objective Value')
        plt.grid(False)
        plt.legend(frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
        plt.tight_layout()
        _save_and_close(out_dir, f'ZCAT{problem_id}_Parallel_Coords')


# ==========================================================
# 12. MAIN
# ==========================================================

# ==========================================================
# SAVING EVERY RUN (built in -- no other file needed)
# Results go to a folder "results_csv" next to this file:
#   results_csv\<ALGORITHM>\ZCAT<id>_M<M>_final.csv        final value of every indicator, 30 runs
#   results_csv\<ALGORITHM>\ZCAT<id>_M<M>_convergence.csv  every checkpoint of every run
#   results_csv\<ALGORITHM>\ZCAT<id>_M<M>_fronts.npz       final fronts of the 30 runs
# stats_analysis.py finds these folders automatically.
# ==========================================================
def save_run_results(alg, problem_id, M, results, evals):
    import csv
    metrics = ["GD", "GD+", "IGD", "IGD+", "HV", "Spacing", "Spread"]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_csv", alg)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"ZCAT{problem_id}_M{M}"
    with open(os.path.join(out_dir, stem + "_final.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "problem", "M", "run", "run_time_s"] + metrics)
        for r in results:
            w.writerow([alg, problem_id, M, r["run_id"], f"{r['run_time']:.4f}"]
                       + [repr(float(r["run_metrics"][k][-1])) for k in metrics])
    with open(os.path.join(out_dir, stem + "_convergence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "problem", "M", "run", "iteration", "evaluations"] + metrics)
        for r in results:
            for i, it in enumerate(r["metric_iterations"]):
                w.writerow([alg, problem_id, M, r["run_id"], it, evals[i]]
                           + [repr(float(r["run_metrics"][k][i])) for k in metrics])
    np.savez_compressed(os.path.join(out_dir, stem + "_fronts.npz"),
                        **{f"run{r['run_id']}": np.asarray(r["final_pareto_front"], dtype=float) for r in results})
    print(f"\nAll 30 runs saved to {out_dir}\\{stem}_*")


if __name__ == '__main__':

    # ---------------- USER SETTINGS ----------------
    PROBLEM_ID     = 1          # <-- change this: 1..20
    M              = 3          # number of objectives: 3 or 5 (paper); also python <this file> <PROBLEM_ID> <M>
    # Optional command-line override (used by run_all_experiments.py):
    #   python <this file> <PROBLEM_ID> [M]
    import sys
    if len(sys.argv) > 1:
        PROBLEM_ID = int(sys.argv[1])
    if len(sys.argv) > 2:
        M = int(sys.argv[2])
    # ---------------- RUN SEVERAL PROBLEMS IN ONE GO (optional) ----------------
    # Leave empty [] to run only PROBLEM_ID above. Example for all problems:
    #   PROBLEM_LIST = list(range(1, 21))
    # Problems whose results already exist in results_csv are skipped.
    PROBLEM_LIST = []
    if PROBLEM_LIST and len(sys.argv) == 1:
        import subprocess
        _me = os.path.abspath(__file__)
        for _pid in PROBLEM_LIST:
            print(f"\n==================== ZCAT{_pid} (M={M}) ====================", flush=True)
            _r = subprocess.run([sys.executable, _me, str(_pid), str(M)], capture_output=True, text=True,
                                env=dict(os.environ, MPLBACKEND="Agg"), cwd=os.path.dirname(_me))
            print(_r.stdout[-2500:], flush=True)
            if _r.returncode != 0:
                print("ERROR in ZCAT", _pid, ":\n", _r.stderr[-3000:], flush=True)
        print("\nAll problems in PROBLEM_LIST finished.")
    # --------------------------------------------------------------------------
    if not (PROBLEM_LIST and len(sys.argv) == 1):
        N_VARS         = 10 * M     # number of decision variables (paper default n = 10*M)
        COMPLICATED_PS = False
        LEVEL          = 1          # 1..6 (ignored if IMBALANCE=True)
        BIAS           = False
        IMBALANCE      = False

        # POPULATION_SIZE and ARCHIVE_SIZE are decoupled (they used to both
        # equal 210): shrinking the population while keeping a larger archive
        # buys more iterations per generation for the same evaluation budget,
        # and pairs with PBI archive truncation above to hold onto a well-
        # spread, converging non-dominated set even with a small population.
        POPULATION_SIZE = 100 if M == 2 else 210         # 210 for M = 3 and M = 5 (paper)
        ARCHIVE_SIZE_OVERRIDE = None if M == 2 else 210  # None -> falls back to POPULATION_SIZE

        # Boundary-directed sampling (see boundary_directed_sampling
        # docstring): adds 2 * BD_K_ELITES * BD_N_SAMPLES extra evaluations
        # per iteration, independent of n. Helps close the HV gap between
        # small and large numbers of decision variables. Set BD_SAMPLING=False
        # to disable it.
        BD_SAMPLING  = True
        BD_K_ELITES  = 5
        BD_N_SAMPLES = 2

        # ---------------- ABLATION / SENSITIVITY SWITCHES ----------------
        # Defaults reproduce the full algorithm. Change ONE of them for an ablation run:
        #   USE_PBI = False        -> crowding-distance archive truncation (as in Proposed MOO)
        #   USE_ANNEALING = False  -> no annealing (c_f = 1 throughout)
        #   BD_SAMPLING = False    -> no boundary-directed sampling (switch above)
        # Sensitivity study: PBI_THETA in {0.5, 1, 2, 5}, CF_FLOOR in {0.3, 0.5, 0.7},
        # BD_K_ELITES in {3, 5, 10}. Non-default runs are saved under their own name
        # (e.g. results_csv\ModifiedProposed_noPBI\, plots_ModifiedProposed_noPBI\).
        USE_PBI       = True
        USE_ANNEALING = True
        PBI_THETA     = 1.0
        CF_FLOOR      = 0.5
        # optional command-line overrides, e.g.:  python <file> 1 3 USE_PBI=0 PBI_THETA=2
        for _kv in sys.argv[3:]:
            _k, _v = _kv.split('=', 1)
            _old = globals()[_k]
            globals()[_k] = (_v.lower() in ('1', 'true', 'yes')) if isinstance(_old, bool) else type(_old)(float(_v))
        _tags = ([] if USE_PBI else ['noPBI']) + ([] if USE_ANNEALING else ['noAnneal']) \
            + ([] if BD_SAMPLING else ['noBD']) + ([] if PBI_THETA == 1.0 else [f'theta{PBI_THETA:g}']) \
            + ([] if CF_FLOOR == 0.5 else [f'cf{CF_FLOOR:g}']) + ([] if BD_K_ELITES == 5 else [f'k{BD_K_ELITES}'])
        VARIANT = 'ModifiedProposed' + ('_' + '_'.join(_tags) if _tags else '')
        # ------------------------------------------------------------------
        EVALS_PER_ITER = POPULATION_SIZE + (2 * BD_K_ELITES * BD_N_SAMPLES if BD_SAMPLING else 0)

        # Same total budget as every competitor (2000 generations x 210 = 420,000
        # evaluations). Each iteration here costs EVALS_PER_ITER (= 230 with
        # boundary-directed sampling), so the iteration count is derived from the
        # budget instead of fixed at 2000 (which would give 460,000 evaluations).
        MAX_ITERATIONS  = 420_000 // EVALS_PER_ITER    # = 1826 for M=3  (1826*230 = 419,980)
        ARCHIVE_SIZE    = ARCHIVE_SIZE_OVERRIDE if ARCHIVE_SIZE_OVERRIDE is not None else POPULATION_SIZE
        NUM_RUNS        = 30
        METRIC_INTERVAL = 100
        # -------------------------------------------------

        print("=" * 70)
        print(f"ZCAT{PROBLEM_ID} OPTIMIZATION USING {VARIANT}  (M={M}, n={N_VARS})")
        print(f"Complicated_PS={COMPLICATED_PS}  Level={LEVEL}  Bias={BIAS}  Imbalance={IMBALANCE}")
        print("=" * 70)

        print("\nPre-computing true Pareto front...")
        true_pf = generate_true_front(PROBLEM_ID, M, n_pts=20)
        print(f"  True PF size: {len(true_pf)}")

        args_list = [
            (run, PROBLEM_ID, M, N_VARS, POPULATION_SIZE, MAX_ITERATIONS, ARCHIVE_SIZE,
             COMPLICATED_PS, LEVEL, BIAS, IMBALANCE, METRIC_INTERVAL, true_pf,
             BD_SAMPLING, BD_K_ELITES, BD_N_SAMPLES,
             USE_PBI, USE_ANNEALING, PBI_THETA, CF_FLOOR)
            for run in range(NUM_RUNS)
        ]

        print(f"\nStarting {NUM_RUNS} parallel runs on {cpu_count()} cores...")
        start = time.time()
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(run_single_optimization, args_list)
        total_time = time.time() - start
        print(f"Done in {total_time/60:.2f} min ({total_time/NUM_RUNS:.2f}s/run avg)")

        n_points = len(results[0]['metric_iterations'])
        metric_keys = results[0]['run_metrics'].keys()
        agg = {k: [[] for _ in range(n_points)] for k in metric_keys}
        for res in results:
            for k in metric_keys:
                for i in range(n_points):
                    v = res['run_metrics'][k][i]
                    if not np.isnan(v):
                        agg[k][i].append(v)
        avg_metrics = {k: [np.mean(v) if v else np.nan for v in vals] for k, vals in agg.items()}
        iters = results[0]['metric_iterations']
        evals = [(i + 1) * EVALS_PER_ITER for i in iters]

        print(f"\nFINAL METRICS ({NUM_RUNS} runs):")
        print(f"{'Metric':<10}| {'Mean':<13}| {'Std':<13}| {'Best':<13}")
        print("-" * 50)
        for k in metric_keys:
            finals = [r['run_metrics'][k][-1] for r in results]
            finals = [v for v in finals if not np.isnan(v)]
            if finals:
                best = np.max(finals) if k == "HV" else np.min(finals)
                print(f"{k:<10}| {np.mean(finals):<13.6f}| {np.std(finals):<13.6f}| {best:<13.6f}")

        # Save every run (final metrics, convergence, fronts) for stats_analysis.py
        save_run_results(VARIANT, PROBLEM_ID, M, results, evals)

        # Extract Best Pareto Front
        best_idx = max(
            range(NUM_RUNS),
            key=lambda i: results[i]['run_metrics']['HV'][-1]
            if not np.isnan(results[i]['run_metrics']['HV'][-1]) else -np.inf
        )
        final_pf = results[best_idx]['final_pareto_front']
        viz_pf = generate_true_front(PROBLEM_ID, M, n_pts=20)

        # ---------------- PLOTS (journal-ready, 300 DPI) ----------------
        generate_plots(PROBLEM_ID, M, avg_metrics, evals, final_pf, viz_pf,
                       out_dir='plots' if VARIANT == 'ModifiedProposed' else f'plots_{VARIANT}')
        print("\nDone. Figures saved to ./plots/ as PNG + PDF at 300 DPI.")
        print("Change PROBLEM_ID (1-20) at the top of the __main__ block to switch problems.")
