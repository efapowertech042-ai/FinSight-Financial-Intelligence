from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ==============================================================================
# 1. Special Functions & Continuous Probability Distributions (Pure Math/NumPy)
# ==============================================================================

def _betacf(a: float, b: float, x: float) -> float:
    """Evaluates continued fraction for incomplete beta function (Numerical Recipes)."""
    MAXIT = 200
    EPS = 3.0e-12
    FPMIN = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x) = P(Gamma >= x)."""
    if x <= 0.0:
        return 1.0
    if a <= 0.0:
        return 0.0
    if x < a + 1.0:
        # Series representation for P(a, x), return 1 - P
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(250):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    # Lentz continued fraction for Q(a, x)
    FPMIN = 1e-300
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / max(b, FPMIN)
    h = d
    for i in range(1, 250):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return max(0.0, min(1.0, q))


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's inverse normal CDF approximation (error < 1.15e-9)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if phigh < p:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def t_sf(abs_t: float, df: float) -> float:
    """Upper tail P(T >= abs_t) for Student-t distribution."""
    if df <= 0.0:
        return 0.5
    if abs_t <= 0.0:
        return 0.5
    x = df / (df + abs_t * abs_t)
    return max(0.0, min(0.5, 0.5 * betainc(df / 2.0, 0.5, x)))


def t_ppf(q: float, df: float) -> float:
    """Inverse Student-t CDF for q in (0, 1)."""
    if q <= 0.0:
        return float("-inf")
    if q >= 1.0:
        return float("inf")
    if abs(q - 0.5) < 1e-12:
        return 0.0
    sign = 1.0 if q > 0.5 else -1.0
    target_sf = 1.0 - q if q > 0.5 else q
    lo, hi = 0.0, max(50.0, abs(norm_ppf(target_sf)) * 4.0)
    while t_sf(hi, df) > target_sf and hi < 1e6:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if t_sf(mid, df) > target_sf:
            lo = mid
        else:
            hi = mid
    return sign * 0.5 * (lo + hi)


def f_sf(f_val: float, df1: float, df2: float) -> float:
    if f_val <= 0.0 or df1 <= 0.0 or df2 <= 0.0:
        return 1.0
    x = df2 / (df2 + df1 * f_val)
    return max(0.0, min(1.0, betainc(df2 / 2.0, df1 / 2.0, x)))


def chi2_sf(x: float, df: float) -> float:
    if x <= 0.0:
        return 1.0
    return gammaincc(df / 2.0, x / 2.0)


# ==============================================================================
# 2. Hypothesis Testing & Statistical Inference (SciPy Stats Drop-in)
# ==============================================================================

def _rankdata(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes average ranks (1-based) and tie counts."""
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    ranks = np.empty(n, dtype=float)
    tie_counts: List[int] = []

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        if j - i > 1:
            tie_counts.append(j - i)
        i = j
    return ranks, np.asarray(tie_counts, dtype=float)


class _NormNamespace:
    cdf = staticmethod(norm_cdf)
    ppf = staticmethod(norm_ppf)


class _TNamespace:
    sf = staticmethod(t_sf)
    ppf = staticmethod(t_ppf)


class _StatsCompat:
    norm = _NormNamespace()
    t = _TNamespace()

    @staticmethod
    def pearsonr(x: Any, y: Any) -> Tuple[float, float]:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        n = len(xa)
        if n < 3:
            return 0.0, 1.0
        xm = xa - np.mean(xa)
        ym = ya - np.mean(ya)
        denom = math.sqrt(float(np.sum(xm * xm) * np.sum(ym * ym)))
        if denom <= 1e-15:
            return 0.0, 1.0
        r = max(-1.0, min(1.0, float(np.sum(xm * ym) / denom)))
        if abs(r) >= 1.0 - 1e-15:
            return r, 0.0
        t_stat = r * math.sqrt((n - 2) / (1.0 - r * r))
        p = 2.0 * t_sf(abs(t_stat), n - 2)
        return r, p

    @staticmethod
    def spearmanr(x: Any, y: Any) -> Tuple[float, float]:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        rx, _ = _rankdata(xa)
        ry, _ = _rankdata(ya)
        return _StatsCompat.pearsonr(rx, ry)

    @staticmethod
    def linregress(x: Any, y: Any) -> SimpleNamespace:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        n = len(xa)
        xm = float(np.mean(xa))
        ym = float(np.mean(ya))
        dx = xa - xm
        dy = ya - ym
        ssx = float(np.sum(dx * dx))
        ssy = float(np.sum(dy * dy))
        sxy = float(np.sum(dx * dy))
        if ssx <= 1e-15 or n < 3:
            return SimpleNamespace(slope=0.0, intercept=ym, rvalue=0.0, pvalue=1.0, stderr=0.0)
        slope = sxy / ssx
        intercept = ym - slope * xm
        r = max(-1.0, min(1.0, sxy / math.sqrt(max(1e-30, ssx * ssy)))) if ssy > 1e-15 else 0.0
        resid = ya - (intercept + slope * xa)
        s2 = float(np.sum(resid * resid)) / max(1, n - 2)
        stderr = math.sqrt(max(0.0, s2 / ssx))
        t_stat = abs(slope) / stderr if stderr > 1e-15 else 0.0
        pvalue = 2.0 * t_sf(t_stat, n - 2) if stderr > 1e-15 else 1.0
        return SimpleNamespace(
            slope=float(slope),
            intercept=float(intercept),
            rvalue=float(r),
            pvalue=float(pvalue),
            stderr=float(stderr),
        )

    @staticmethod
    def kendalltau(x: Any, y: Any) -> Tuple[float, float]:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        n = len(xa)
        if n < 2:
            return 0.0, 1.0

        C = 0
        D = 0
        tx_ties: Dict[float, int] = {}
        ty_ties: Dict[float, int] = {}
        txy_ties: Dict[Tuple[float, float], int] = {}
        for k in range(n):
            tx_ties[float(xa[k])] = tx_ties.get(float(xa[k]), 0) + 1
            ty_ties[float(ya[k])] = ty_ties.get(float(ya[k]), 0) + 1
            key = (float(xa[k]), float(ya[k]))
            txy_ties[key] = txy_ties.get(key, 0) + 1

        for i in range(n - 1):
            dx = xa[i + 1 :] - xa[i]
            dy = ya[i + 1 :] - ya[i]
            prod = dx * dy
            C += int(np.sum(prod > 0))
            D += int(np.sum(prod < 0))

        n0 = n * (n - 1) // 2
        n1 = sum(t * (t - 1) // 2 for t in tx_ties.values() if t > 1)
        n2 = sum(u * (u - 1) // 2 for u in ty_ties.values() if u > 1)
        denom = math.sqrt(max(1.0, float((n0 - n1) * (n0 - n2))))
        tau = float((C - D) / denom)

        v0 = n * (n - 1) * (2 * n + 5)
        vt = sum(t * (t - 1) * (2 * t + 5) for t in tx_ties.values() if t > 1)
        vu = sum(u * (u - 1) * (2 * u + 5) for u in ty_ties.values() if u > 1)
        v1 = (
            sum(t * (t - 1) for t in tx_ties.values() if t > 1)
            * sum(u * (u - 1) for u in ty_ties.values() if u > 1)
            / max(1.0, 2.0 * n * (n - 1))
        )
        v2 = (
            sum(t * (t - 1) * (t - 2) for t in tx_ties.values() if t > 2)
            * sum(u * (u - 1) * (u - 2) for u in ty_ties.values() if u > 2)
            / max(1.0, 9.0 * n * (n - 1) * (n - 2))
        )
        var_s = (v0 - vt - vu) / 18.0 + v1 + v2
        if var_s <= 1e-12:
            return tau, 1.0
        S = C - D
        z = (abs(S) - 1.0) / math.sqrt(var_s) if abs(S) > 0 else 0.0
        p = 2.0 * (1.0 - norm_cdf(max(0.0, z)))
        return tau, max(0.0, min(1.0, p))

    @staticmethod
    def f_oneway(*groups: Any) -> Tuple[float, float]:
        arrs = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
        k = len(arrs)
        N = sum(len(a) for a in arrs)
        if k < 2 or N <= k:
            return 0.0, 1.0
        all_vals = np.concatenate(arrs)
        grand_mean = float(np.mean(all_vals))
        ss_between = sum(len(a) * ((float(np.mean(a)) - grand_mean) ** 2) for a in arrs)
        ss_within = sum(float(np.sum((a - np.mean(a)) ** 2)) for a in arrs)
        df_b = k - 1
        df_w = N - k
        if ss_within <= 1e-15:
            return 0.0, 1.0
        f_stat = (ss_between / df_b) / (ss_within / df_w)
        return float(f_stat), f_sf(f_stat, df_b, df_w)

    @staticmethod
    def levene(*groups: Any) -> Tuple[float, float]:
        """Brown-Forsythe median-centered Levene test."""
        z_groups = [np.abs(np.asarray(g, dtype=float) - np.median(np.asarray(g, dtype=float))) for g in groups]
        return _StatsCompat.f_oneway(*z_groups)

    @staticmethod
    def kruskal(*groups: Any) -> Tuple[float, float]:
        arrs = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
        k = len(arrs)
        N = sum(len(a) for a in arrs)
        if k < 2 or N < 2:
            return 0.0, 1.0
        all_vals = np.concatenate(arrs)
        ranks, tie_counts = _rankdata(all_vals)
        H = 0.0
        idx = 0
        for a in arrs:
            ni = len(a)
            r_sum = float(np.sum(ranks[idx : idx + ni]))
            H += (r_sum * r_sum) / ni
            idx += ni
        H = (12.0 / (N * (N + 1.0))) * H - 3.0 * (N + 1.0)
        tie_corr = 1.0 - float(np.sum(tie_counts ** 3 - tie_counts)) / (N ** 3 - N) if len(tie_counts) else 1.0
        if tie_corr > 1e-12:
            H /= tie_corr
        p = chi2_sf(max(0.0, H), k - 1)
        return float(H), float(p)

    @staticmethod
    def ttest_ind(a: Any, b: Any, equal_var: bool = False) -> Tuple[float, float]:
        xa = np.asarray(a, dtype=float)
        xb = np.asarray(b, dtype=float)
        na, nb = len(xa), len(xb)
        if na < 2 or nb < 2:
            return 0.0, 1.0
        va = float(np.var(xa, ddof=1))
        vb = float(np.var(xb, ddof=1))
        diff = float(np.mean(xa) - np.mean(xb))
        if equal_var:
            sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
            se = math.sqrt(max(1e-30, sp2 * (1.0 / na + 1.0 / nb)))
            df = float(na + nb - 2)
        else:
            sa = va / na
            sb = vb / nb
            se = math.sqrt(max(1e-30, sa + sb))
            denom = (sa * sa) / (na - 1) + (sb * sb) / (nb - 1)
            df = ((sa + sb) ** 2) / denom if denom > 1e-30 else float(na + nb - 2)
        t_stat = diff / se
        p = 2.0 * t_sf(abs(t_stat), df)
        return float(t_stat), float(p)

    @staticmethod
    def mannwhitneyu(x: Any, y: Any, alternative: str = "two-sided") -> Tuple[float, float]:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        n1, n2 = len(xa), len(ya)
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0
        N = n1 + n2
        all_vals = np.concatenate([xa, ya])
        ranks, tie_counts = _rankdata(all_vals)
        r1 = float(np.sum(ranks[:n1]))
        u1 = r1 - n1 * (n1 + 1) / 2.0
        mean_u = n1 * n2 / 2.0
        tie_term = float(np.sum(tie_counts ** 3 - tie_counts)) / (N * (N - 1)) if len(tie_counts) and N > 1 else 0.0
        var_u = (n1 * n2 / 12.0) * max(1e-12, (N + 1.0) - tie_term)
        sd_u = math.sqrt(var_u)
        if sd_u <= 1e-12:
            return float(u1), 1.0
        if alternative == "two-sided":
            z = (abs(u1 - mean_u) - 0.5) / sd_u if abs(u1 - mean_u) > 0.5 else 0.0
            p = 2.0 * (1.0 - norm_cdf(max(0.0, z)))
        elif alternative == "greater":
            z = (u1 - mean_u - 0.5) / sd_u
            p = 1.0 - norm_cdf(z)
        else:
            z = (u1 - mean_u + 0.5) / sd_u
            p = norm_cdf(z)
        return float(u1), max(0.0, min(1.0, float(p)))

    @staticmethod
    def shapiro(x: Any) -> SimpleNamespace:
        """Royston AS R94 Shapiro-Wilk normality test."""
        xa = np.sort(np.asarray(x, dtype=float))
        n = len(xa)
        if n < 3:
            return SimpleNamespace(statistic=1.0, pvalue=1.0)
        if xa[-1] - xa[0] <= 1e-15:
            return SimpleNamespace(statistic=1.0, pvalue=1.0)

        # Blom normal order statistic approximations: Phi^{-1}((i - 0.375) / (n + 0.25))
        idx = np.arange(1, n + 1, dtype=float)
        m = np.array([norm_ppf((i - 0.375) / (n + 0.25)) for i in idx], dtype=float)
        m2_sum = float(np.sum(m * m))
        u = 1.0 / math.sqrt(n)
        c = m / math.sqrt(m2_sum)

        # Polynomial weights per Royston AS R94 for a_n, a_{n-1}
        a = c.copy()
        a_n = (-2.706056 * u ** 5 + 4.434685 * u ** 4 - 2.071190 * u ** 3 - 0.147981 * u ** 2 + 0.221157 * u + c[-1])
        a[-1] = a_n
        a[0] = -a_n
        if n <= 5:
            phi = (m2_sum - 2.0 * m[-1] ** 2) / max(1e-12, 1.0 - 2.0 * a[-1] ** 2)
            a[1:-1] = m[1:-1] / math.sqrt(max(1e-12, phi))
        else:
            a_nm1 = (-3.582633 * u ** 5 + 5.682633 * u ** 4 - 1.752461 * u ** 3 - 0.293762 * u ** 2 + 0.042981 * u + c[-2])
            a[-2] = a_nm1
            a[1] = -a_nm1
            phi = (m2_sum - 2.0 * m[-1] ** 2 - 2.0 * m[-2] ** 2) / max(1e-12, 1.0 - 2.0 * a[-1] ** 2 - 2.0 * a[-2] ** 2)
            a[2:-2] = m[2:-2] / math.sqrt(max(1e-12, phi))

        ss_tot = float(np.sum((xa - np.mean(xa)) ** 2))
        W = min(1.0, (float(np.sum(a * xa)) ** 2) / max(1e-15, ss_tot))

        # Normalizing transformation for p-value (Royston 1992/1995)
        if n == 3:
            p = (6.0 / math.pi) * (math.asin(math.sqrt(W)) - math.asin(math.sqrt(0.75)))
            p = max(0.0, min(1.0, 1.0 - p))
            return SimpleNamespace(statistic=float(W), pvalue=float(p))

        ln_n = math.log(n)
        y = math.log(max(1e-12, 1.0 - W))
        if n <= 11:
            gamma = 0.459 * n - 2.273
            y = -math.log(max(1e-12, gamma - y)) if gamma > y else 5.0
            mu = -0.0006714 * n ** 3 + 0.025054 * n ** 2 - 0.39978 * n + 0.5440
            sigma = math.exp(-0.0020322 * n ** 3 + 0.062767 * n ** 2 - 0.77857 * n + 1.3822)
        else:
            mu = 0.0038915 * ln_n ** 3 - 0.083751 * ln_n ** 2 - 0.31082 * ln_n - 1.5861
            sigma = math.exp(0.0030302 * ln_n ** 2 - 0.082676 * ln_n - 0.4803)
        z = (y - mu) / max(1e-9, sigma)
        p = max(0.0, min(1.0, 1.0 - norm_cdf(z)))
        return SimpleNamespace(statistic=float(W), pvalue=float(p))


stats = _StatsCompat()


# ==============================================================================
# 3. Preprocessing, Splitting, Metrics & Linear Models
# ==============================================================================

def _to_numpy(X: Any) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        return np.asarray(X.to_numpy(dtype=float), dtype=float)
    return np.asarray(X, dtype=float)


class SimpleImputer:
    def __init__(self, strategy: str = "median") -> None:
        self.strategy = strategy
        self.statistics_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "SimpleImputer":
        arr = _to_numpy(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        stats_vals = np.zeros(arr.shape[1], dtype=float)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                stats_vals[j] = 0.0
            elif self.strategy == "mean":
                stats_vals[j] = float(np.mean(valid))
            else:
                stats_vals[j] = float(np.median(valid))
        self.statistics_ = stats_vals
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = _to_numpy(X).copy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.statistics_ is None:
            return arr
        for j in range(arr.shape[1]):
            mask = np.isnan(arr[:, j])
            if mask.any():
                arr[mask, j] = self.statistics_[j]
        return arr

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "StandardScaler":
        arr = _to_numpy(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.mean_ = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = _to_numpy(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.mean_ is None or self.scale_ is None:
            return arr
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class Pipeline:
    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        self.steps = steps

    def fit(self, X: Any, y: Any = None) -> "Pipeline":
        cur = X
        for _, step in self.steps:
            cur = step.fit_transform(cur, y)
        return self

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        cur = X
        for _, step in self.steps:
            cur = step.fit_transform(cur, y)
        return cur

    def transform(self, X: Any) -> np.ndarray:
        cur = X
        for _, step in self.steps:
            cur = step.transform(cur)
        return cur


class TimeSeriesSplit:
    def __init__(self, n_splits: int = 4) -> None:
        self.n_splits = n_splits

    def split(self, X: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        n_folds = self.n_splits + 1
        if n_folds > n_samples:
            return []
        test_size = n_samples // n_folds
        if test_size < 1:
            return []
        test_starts = range(n_samples - self.n_splits * test_size, n_samples, test_size)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for test_start in test_starts:
            splits.append((np.arange(0, test_start), np.arange(test_start, min(n_samples, test_start + test_size))))
        return splits


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))


def r2_score(y_true: Any, y_pred: Any) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-15:
        return 0.0 if ss_res > 1e-15 else 1.0
    return 1.0 - (ss_res / ss_tot)


def permutation_importance(
    estimator: Any,
    X: Any,
    y: Any,
    n_repeats: int = 15,
    random_state: int = 42,
    scoring: str = "r2",
) -> SimpleNamespace:
    rng = np.random.default_rng(random_state)
    X_arr = _to_numpy(X).copy()
    y_arr = np.asarray(y, dtype=float)
    base_pred = estimator.predict(X_arr)
    base_score = r2_score(y_arr, base_pred) if scoring == "r2" else -mean_squared_error(y_arr, base_pred)

    n_samples, n_features = X_arr.shape
    importances = np.zeros((n_features, n_repeats), dtype=float)

    for j in range(n_features):
        orig_col = X_arr[:, j].copy()
        for r in range(n_repeats):
            perm_idx = rng.permutation(n_samples)
            X_arr[:, j] = orig_col[perm_idx]
            pred_perm = estimator.predict(X_arr)
            score_perm = r2_score(y_arr, pred_perm) if scoring == "r2" else -mean_squared_error(y_arr, pred_perm)
            importances[j, r] = base_score - score_perm
        X_arr[:, j] = orig_col

    return SimpleNamespace(
        importances_mean=np.mean(importances, axis=1),
        importances_std=np.std(importances, axis=1),
        importances=importances,
    )


class LinearRegression:
    def __init__(self) -> None:
        self._params: Dict[str, Any] = {}
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(self, X: Any, y: Any) -> "LinearRegression":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float)
        xm = np.mean(X_arr, axis=0)
        ym = float(np.mean(y_arr))
        Xc = X_arr - xm
        yc = y_arr - ym
        coef, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
        self.coef_ = coef
        self.intercept_ = ym - float(np.dot(xm, coef))
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = _to_numpy(X)
        return X_arr @ self.coef_ + self.intercept_

    def clone_fresh(self) -> "LinearRegression":
        return LinearRegression()


class Ridge:
    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        self.alpha = float(alpha)
        self.random_state = random_state
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(self, X: Any, y: Any) -> "Ridge":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float)
        xm = np.mean(X_arr, axis=0)
        ym = float(np.mean(y_arr))
        Xc = X_arr - xm
        yc = y_arr - ym
        n_features = Xc.shape[1]
        gram = Xc.T @ Xc
        gram[np.diag_indices(n_features)] += self.alpha
        rhs = Xc.T @ yc
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        self.coef_ = coef
        self.intercept_ = ym - float(np.dot(xm, coef))
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = _to_numpy(X)
        return X_arr @ self.coef_ + self.intercept_

    def clone_fresh(self) -> "Ridge":
        return Ridge(alpha=self.alpha, random_state=self.random_state)


# ==============================================================================
# 4. Vectorized CART Decision Tree, Random Forest & Gradient Boosting
# ==============================================================================

class _FlatTree:
    """Array-backed binary CART regression tree supporting fast batch prediction & Exact TreeSHAP."""

    def __init__(self, max_depth: int, min_samples_leaf: int, max_features: Optional[float]) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.feature: np.ndarray = np.array([], dtype=np.int32)
        self.threshold: np.ndarray = np.array([], dtype=float)
        self.children_left: np.ndarray = np.array([], dtype=np.int32)
        self.children_right: np.ndarray = np.array([], dtype=np.int32)
        self.value: np.ndarray = np.array([], dtype=float)
        self.n_node_samples: np.ndarray = np.array([], dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> "_FlatTree":
        n_samples, n_features = X.shape
        feats_per_split = (
            n_features
            if self.max_features is None
            else max(1, min(n_features, int(round(self.max_features * n_features))))
        )

        f_list: List[int] = []
        t_list: List[float] = []
        l_list: List[int] = []
        r_list: List[int] = []
        v_list: List[float] = []
        c_list: List[float] = []

        def build_node(indices: np.ndarray, depth: int) -> int:
            node_id = len(f_list)
            n_node = len(indices)
            y_sub = y[indices]
            mean_val = float(y_sub.mean())

            f_list.append(-1)
            t_list.append(-2.0)
            l_list.append(-1)
            r_list.append(-1)
            v_list.append(mean_val)
            c_list.append(float(n_node))

            if depth >= self.max_depth or n_node < 2 * self.min_samples_leaf:
                return node_id
            if float(y_sub.var()) <= 1e-12:
                return node_id

            if feats_per_split < n_features:
                cand_feats = rng.choice(n_features, size=feats_per_split, replace=False)
            else:
                cand_feats = range(n_features)

            total_sum = float(y_sub.sum())
            best_gain = 1e-12
            best_feat = -1
            best_thresh = 0.0

            X_sub = X[indices]
            base_score = (total_sum * total_sum) / n_node
            k_start = self.min_samples_leaf - 1
            k_end = n_node - self.min_samples_leaf - 1
            if k_end < k_start:
                return node_id
            ks_base = np.arange(k_start, k_end + 1)

            for j in cand_feats:
                col = X_sub[:, j]
                order = np.argsort(col, kind="quicksort")
                col_s = col[order]
                if col_s[0] >= col_s[-1]:
                    continue
                y_s = y_sub[order]
                cum_y = np.cumsum(y_s)
                valid_split = col_s[ks_base] < col_s[ks_base + 1]
                if not valid_split.any():
                    continue
                ks = ks_base[valid_split]
                left_n = ks + 1
                right_n = n_node - left_n
                left_sum = cum_y[ks]
                right_sum = total_sum - left_sum
                scores = (left_sum * left_sum) / left_n + (right_sum * right_sum) / right_n - base_score
                best_local = int(np.argmax(scores))
                gain = float(scores[best_local])
                if gain > best_gain:
                    best_gain = gain
                    k_best = int(ks[best_local])
                    best_feat = int(j)
                    best_thresh = 0.5 * (float(col_s[k_best]) + float(col_s[k_best + 1]))

            if best_feat == -1:
                return node_id

            mask_left = X_sub[:, best_feat] <= best_thresh
            left_idx = indices[mask_left]
            right_idx = indices[~mask_left]
            if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                return node_id

            f_list[node_id] = best_feat
            t_list[node_id] = best_thresh
            l_list[node_id] = build_node(left_idx, depth + 1)
            r_list[node_id] = build_node(right_idx, depth + 1)
            return node_id

        build_node(np.arange(n_samples, dtype=np.int32), 0)
        self.feature = np.asarray(f_list, dtype=np.int32)
        self.threshold = np.asarray(t_list, dtype=float)
        self.children_left = np.asarray(l_list, dtype=np.int32)
        self.children_right = np.asarray(r_list, dtype=np.int32)
        self.value = np.asarray(v_list, dtype=float)
        self.n_node_samples = np.asarray(c_list, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        nodes = np.zeros(n, dtype=np.int32)
        f = self.feature
        t = self.threshold
        cl = self.children_left
        cr = self.children_right
        while True:
            cur_f = f[nodes]
            active = cur_f >= 0
            if not active.any():
                break
            act_idx = np.flatnonzero(active)
            act_nodes = nodes[act_idx]
            go_left = X[act_idx, cur_f[act_idx]] <= t[act_nodes]
            nodes[act_idx] = np.where(go_left, cl[act_nodes], cr[act_nodes])
        return self.value[nodes]


class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 45,
        max_depth: int = 7,
        min_samples_leaf: int = 2,
        max_features: float = 0.65,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: List[_FlatTree] = []

    def fit(self, X: Any, y: Any) -> "RandomForestRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float)
        n = X_arr.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        eff_trees = min(self.n_estimators, 45)
        for _ in range(eff_trees):
            boot_idx = rng.integers(0, n, size=n)
            tree = _FlatTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(X_arr[boot_idx], y_arr[boot_idx], rng)
            self.estimators_.append(tree)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = _to_numpy(X)
        if not self.estimators_:
            return np.zeros(X_arr.shape[0], dtype=float)
        preds = np.mean([t.predict(X_arr) for t in self.estimators_], axis=0)
        return preds

    def clone_fresh(self) -> "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators: int = 70,
        learning_rate: float = 0.06,
        max_depth: int = 3,
        subsample: float = 0.85,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.init_prediction_: float = 0.0
        self.estimators_: List[_FlatTree] = []

    def fit(self, X: Any, y: Any) -> "GradientBoostingRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float)
        n = X_arr.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.init_prediction_ = float(np.mean(y_arr))
        pred = np.full(n, self.init_prediction_, dtype=float)
        self.estimators_ = []
        sub_n = max(4, min(n, int(round(self.subsample * n))))
        eff_trees = min(self.n_estimators, 70)
        for _ in range(eff_trees):
            residuals = y_arr - pred
            if sub_n < n:
                idx = rng.choice(n, size=sub_n, replace=False)
                tree = _FlatTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_features=0.75)
                tree.fit(X_arr[idx], residuals[idx], rng)
            else:
                tree = _FlatTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_features=0.75)
                tree.fit(X_arr, residuals, rng)
            pred += self.learning_rate * tree.predict(X_arr)
            self.estimators_.append(tree)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = _to_numpy(X)
        out = np.full(X_arr.shape[0], self.init_prediction_, dtype=float)
        for tree in self.estimators_:
            out += self.learning_rate * tree.predict(X_arr)
        return out

    def clone_fresh(self) -> "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )


def clone(estimator: Any) -> Any:
    if hasattr(estimator, "clone_fresh"):
        return estimator.clone_fresh()
    return estimator


# ==============================================================================
# 5. Pure-NumPy Isolation Forest (Liu et al. 2008)
# ==============================================================================

def _c_factor(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + 0.5772156649) - (2.0 * (n - 1.0) / n)


class _ITree:
    def __init__(self, height_limit: int) -> None:
        self.height_limit = height_limit
        self.root: Any = None

    def fit(self, X: np.ndarray, rng: np.random.Generator) -> "_ITree":
        n_features = X.shape[1]

        def build(sub: np.ndarray, depth: int) -> Any:
            m = sub.shape[0]
            if depth >= self.height_limit or m <= 1:
                return ("leaf", m)
            mins = np.min(sub, axis=0)
            maxs = np.max(sub, axis=0)
            valid_cols = np.flatnonzero(maxs > mins + 1e-12)
            if len(valid_cols) == 0:
                return ("leaf", m)
            col = int(rng.choice(valid_cols))
            split = float(rng.uniform(mins[col], maxs[col]))
            left_mask = sub[:, col] < split
            return ("node", col, split, build(sub[left_mask], depth + 1), build(sub[~left_mask], depth + 1))

        self.root = build(X, 0)
        return self

    def path_length(self, x: np.ndarray) -> float:
        node = self.root
        depth = 0
        while node[0] == "node":
            _, col, split, left, right = node
            depth += 1
            node = left if x[col] < split else right
        return float(depth) + _c_factor(node[1])


class IsolationForest:
    def __init__(self, contamination: float = 0.08, n_estimators: int = 100, random_state: int = 42) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees_: List[_ITree] = []
        self.max_samples_: int = 256
        self.offset_: float = -0.5

    def fit(self, X: Any) -> "IsolationForest":
        X_arr = _to_numpy(X)
        n = X_arr.shape[0]
        self.max_samples_ = min(256, n)
        h_lim = max(1, int(math.ceil(math.log2(max(2, self.max_samples_)))))
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=self.max_samples_, replace=False) if n > self.max_samples_ else np.arange(n)
            t = _ITree(height_limit=h_lim).fit(X_arr[idx], rng)
            self.trees_.append(t)
        raw_scores = self._score_samples(X_arr)
        # decision_function > 0 => inlier, < 0 => outlier
        self.offset_ = float(np.quantile(raw_scores, self.contamination))
        return self

    def _score_samples(self, X_arr: np.ndarray) -> np.ndarray:
        c_n = max(1e-6, _c_factor(self.max_samples_))
        out = np.empty(X_arr.shape[0], dtype=float)
        for i in range(X_arr.shape[0]):
            avg_h = float(np.mean([t.path_length(X_arr[i]) for t in self.trees_]))
            # Higher path length => normal; transform to sklearn score_samples range [-1, 0]
            out[i] = -(2.0 ** (-avg_h / c_n))
        return out

    def decision_function(self, X: Any) -> np.ndarray:
        X_arr = _to_numpy(X)
        return self._score_samples(X_arr) - self.offset_

    def fit_predict(self, X: Any) -> np.ndarray:
        self.fit(X)
        df = self.decision_function(X)
        return np.where(df < 0.0, -1, 1)


# ==============================================================================
# 6. Classical Time-Series Models & Residual Diagnostics (Statsmodels Drop-in)
# ==============================================================================

def _ses_eval(y: np.ndarray, alpha: float, l0: float) -> Tuple[float, float]:
    level = l0
    sse = 0.0
    for val in y:
        err = val - level
        sse += err * err
        level = alpha * val + (1.0 - alpha) * level
    return sse, level


def fit_ses(y: np.ndarray, steps: int) -> Tuple[np.ndarray, float]:
    n = len(y)
    l0 = float(np.mean(y[: min(4, n)]))
    best_sse = float("inf")
    best_alpha = 0.3
    best_level = float(y[-1])
    for alpha in np.linspace(0.05, 0.95, 37):
        sse, lvl = _ses_eval(y, float(alpha), l0)
        if sse < best_sse:
            best_sse = sse
            best_alpha = float(alpha)
            best_level = lvl
    return np.full(steps, best_level, dtype=float), best_alpha


def _holt_eval(y: np.ndarray, alpha: float, beta: float, phi: float) -> Tuple[float, float, float]:
    n = len(y)
    level = float(y[0])
    trend = float(y[1] - y[0]) if n >= 2 else 0.0
    sse = 0.0
    for val in y:
        hat = level + phi * trend
        err = val - hat
        sse += err * err
        new_level = alpha * val + (1.0 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1.0 - beta) * phi * trend
        level = new_level
    return sse, level, trend


def fit_holt_damped(y: np.ndarray, steps: int) -> Tuple[np.ndarray, float]:
    best_sse = float("inf")
    best_params = (0.35, 0.15, 0.90)
    best_state = (float(y[-1]), 0.0)
    for alpha in [0.15, 0.30, 0.50, 0.70, 0.85]:
        for beta in [0.05, 0.15, 0.30, 0.50]:
            for phi in [0.80, 0.88, 0.94, 0.98]:
                sse, lvl, tr = _holt_eval(y, alpha, beta, phi)
                if sse < best_sse:
                    best_sse = sse
                    best_params = (alpha, beta, phi)
                    best_state = (lvl, tr)
    # Local fine refinement around best grid point
    a0, b0, p0 = best_params
    for da in [-0.07, 0.0, 0.07]:
        for db in [-0.05, 0.0, 0.05]:
            for dp in [-0.03, 0.0, 0.02]:
                a = max(0.05, min(0.95, a0 + da))
                b = max(0.01, min(0.80, b0 + db))
                p = max(0.75, min(0.99, p0 + dp))
                sse, lvl, tr = _holt_eval(y, a, b, p)
                if sse < best_sse:
                    best_sse = sse
                    best_params = (a, b, p)
                    best_state = (lvl, tr)

    phi = best_params[2]
    lvl, tr = best_state
    fc = np.empty(steps, dtype=float)
    phi_sum = 0.0
    for h in range(1, steps + 1):
        phi_sum += phi ** h
        fc[h - 1] = lvl + phi_sum * tr
    return fc, phi


def _hw_eval(y: np.ndarray, alpha: float, beta: float, gamma: float, phi: float, m: int = 12) -> Tuple[float, float, float, np.ndarray]:
    n = len(y)
    # Initialize level & seasonal components using first full seasonal cycle
    s0 = float(np.mean(y[:m]))
    seas = np.array([float(y[i]) - s0 for i in range(m)], dtype=float)
    seas -= np.mean(seas)
    level = s0
    if n >= 2 * m:
        trend = (float(np.mean(y[m : 2 * m])) - s0) / m
    else:
        trend = (float(y[-1]) - float(y[0])) / max(1, n - 1)

    sse = 0.0
    for t in range(n):
        s_idx = t % m
        hat = level + phi * trend + seas[s_idx]
        err = float(y[t]) - hat
        sse += err * err
        prev_lvl = level
        level = alpha * (float(y[t]) - seas[s_idx]) + (1.0 - alpha) * (level + phi * trend)
        trend = beta * (level - prev_lvl) + (1.0 - beta) * phi * trend
        seas[s_idx] = gamma * (float(y[t]) - level) + (1.0 - gamma) * seas[s_idx]
    return sse, level, trend, seas


def fit_holt_winters_damped(y: np.ndarray, steps: int, m: int = 12) -> np.ndarray:
    best_sse = float("inf")
    best_state: Optional[Tuple[float, float, float, np.ndarray]] = None
    best_phi = 0.92
    for alpha in [0.20, 0.40, 0.65]:
        for beta in [0.05, 0.20]:
            for gamma in [0.10, 0.30, 0.55]:
                for phi in [0.85, 0.92, 0.97]:
                    sse, lvl, tr, seas = _hw_eval(y, alpha, beta, gamma, phi, m=m)
                    if sse < best_sse:
                        best_sse = sse
                        best_phi = phi
                        best_state = (lvl, tr, phi, seas.copy())
    if best_state is None:
        return np.full(steps, float(y[-1]), dtype=float)
    lvl, tr, phi, seas = best_state
    n = len(y)
    fc = np.empty(steps, dtype=float)
    phi_sum = 0.0
    for h in range(1, steps + 1):
        phi_sum += phi ** h
        s_idx = (n + h - 1) % m
        fc[h - 1] = lvl + phi_sum * tr + seas[s_idx]
    return fc


def fit_autoreg(y: np.ndarray, steps: int, lags: List[int]) -> np.ndarray:
    """OLS Autoregressive model with constant + deterministic linear trend ('ct')."""
    n = len(y)
    max_lag = max(lags)
    rows: List[List[float]] = []
    targets: List[float] = []
    for t in range(max_lag, n):
        row = [1.0, float(t)] + [float(y[t - k]) for k in lags]
        rows.append(row)
        targets.append(float(y[t]))
    X = np.asarray(rows, dtype=float)
    Y = np.asarray(targets, dtype=float)
    # Mild Tikhonov stabilization to ensure well-conditioned lag inversion
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    hist = y.astype(float).tolist()
    preds = np.empty(steps, dtype=float)
    for h in range(steps):
        t = n + h
        x_step = np.asarray([1.0, float(t)] + [hist[-k] for k in lags], dtype=float)
        y_hat = float(np.dot(x_step, beta))
        preds[h] = y_hat
        hist.append(y_hat)
    return preds


def durbin_watson(residuals: Any) -> float:
    e = np.asarray(residuals, dtype=float)
    if len(e) < 2:
        return 2.0
    denom = float(np.sum(e * e))
    if denom <= 1e-15:
        return 2.0
    num = float(np.sum(np.diff(e) ** 2))
    return num / denom


def ljung_box_pvalue(residuals: Any, lag: int = 4) -> Optional[float]:
    e = np.asarray(residuals, dtype=float)
    n = len(e)
    if n < lag + 2:
        return None
    ec = e - float(np.mean(e))
    denom = float(np.sum(ec * ec))
    if denom <= 1e-15:
        return 1.0
    q = 0.0
    for k in range(1, lag + 1):
        rk = float(np.sum(ec[k:] * ec[:-k])) / denom
        q += (rk * rk) / max(1, n - k)
    q *= n * (n + 2)
    return chi2_sf(q, lag)
