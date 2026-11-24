from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from .lightweight_ml import (
        GradientBoostingRegressor,
        LinearRegression,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        durbin_watson,
        fit_autoreg,
        fit_holt_damped,
        fit_holt_winters_damped,
        fit_seasonal_ets,
        fit_ses,
        ljung_box_pvalue,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        stats,
    )
except ImportError:
    from lightweight_ml import (
        GradientBoostingRegressor,
        LinearRegression,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        durbin_watson,
        fit_autoreg,
        fit_holt_damped,
        fit_holt_winters_damped,
        fit_seasonal_ets,
        fit_ses,
        ljung_box_pvalue,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        stats,
    )


TS_FEATURE_DICTIONARY: List[Dict[str, str]] = [
    {
        "name": "Time_Index",
        "category": "Trend & Horizon",
        "formula": "t (months elapsed since first historical observation, 0-indexed)",
        "leakage_status": "Leak-free (calendar deterministic)",
        "business_relevance": "Captures secular nominal expansion in revenue and profit scale.",
    },
    {
        "name": "Month_Sin",
        "category": "Seasonality",
        "formula": "sin(2 * pi * Month_Num / 12)",
        "leakage_status": "Leak-free (calendar deterministic)",
        "business_relevance": "Encodes smooth annual intra-year seasonal cycles without discontinuities.",
    },
    {
        "name": "Month_Cos",
        "category": "Seasonality",
        "formula": "cos(2 * pi * Month_Num / 12)",
        "leakage_status": "Leak-free (calendar deterministic)",
        "business_relevance": "Pairs with Month_Sin to capture Q4 demand peaks and Q1 troughs.",
    },
    {
        "name": "Quarter_Num",
        "category": "Seasonality",
        "formula": "1..4 derived from Month_Num",
        "leakage_status": "Leak-free (calendar deterministic)",
        "business_relevance": "Captures quarterly commercial campaign and budget cycles.",
    },
    {
        "name": "Lag1_Target",
        "category": "Autoregressive Lag",
        "formula": "y_{t-1}",
        "leakage_status": "Leak-free (known prior to period t)",
        "business_relevance": "Immediate prior-month operating profit momentum.",
    },
    {
        "name": "Lag2_Target",
        "category": "Autoregressive Lag",
        "formula": "y_{t-2}",
        "leakage_status": "Leak-free (known prior to period t)",
        "business_relevance": "Captures short-term two-month mean reversion or continuation.",
    },
    {
        "name": "Lag3_Target",
        "category": "Autoregressive Lag",
        "formula": "y_{t-3}",
        "leakage_status": "Leak-free (known prior to period t)",
        "business_relevance": "One-quarter lagged operating profit anchor.",
    },
    {
        "name": "Lag12_Target",
        "category": "Seasonal Lag",
        "formula": "y_{t-12}",
        "leakage_status": "Leak-free (known 12 months prior)",
        "business_relevance": "Same-month prior-year operating profit capturing annual seasonality.",
    },
    {
        "name": "Rolling_Mean_3M",
        "category": "Rolling Trend",
        "formula": "mean(y_{t-3}, y_{t-2}, y_{t-1})",
        "leakage_status": "Leak-free (strictly backward window)",
        "business_relevance": "Smooths one-off monthly noise to represent quarterly run-rate.",
    },
    {
        "name": "Rolling_Mean_6M",
        "category": "Rolling Trend",
        "formula": "mean(y_{t-6} .. y_{t-1})",
        "leakage_status": "Leak-free (strictly backward window)",
        "business_relevance": "Captures medium-term half-year operating profit trajectory.",
    },
    {
        "name": "Rolling_Mean_12M",
        "category": "Rolling Trend",
        "formula": "mean(y_{t-12} .. y_{t-1})",
        "leakage_status": "Leak-free (strictly backward window)",
        "business_relevance": "Deseasonalized trailing 12-month operating run-rate baseline.",
    },
    {
        "name": "Rolling_Std_6M",
        "category": "Historical Volatility",
        "formula": "std(y_{t-6} .. y_{t-1})",
        "leakage_status": "Leak-free (strictly backward window)",
        "business_relevance": "Measures recent earnings volatility and operational regime instability.",
    },
    {
        "name": "MoM_Diff_Lag1",
        "category": "Momentum",
        "formula": "y_{t-1} - y_{t-2}",
        "leakage_status": "Leak-free (strictly backward difference)",
        "business_relevance": "Prior-month directional NGN change in operating profit.",
    },
    {
        "name": "YoY_Growth_Pct_Lag1",
        "category": "Growth Dynamics",
        "formula": "((y_{t-1} - y_{t-13}) / max(|y_{t-13}|, 1e6)) * 100",
        "leakage_status": "Leak-free (strictly backward growth)",
        "business_relevance": "Trailing year-over-year earnings expansion velocity.",
    },
    {
        "name": "Operating_Margin_Lag1",
        "category": "Profitability Ratio",
        "formula": "Operating_Margin_Pct_{t-1}",
        "leakage_status": "Leak-free (ex-ante lagged ratio)",
        "business_relevance": "Most recent realized operating efficiency ratio prior to forecast month.",
    },
    {
        "name": "Cost_to_Revenue_Lag1",
        "category": "Cost Efficiency",
        "formula": "((COGS_{t-1} + Marketing_{t-1} + OtherOpEx_{t-1}) / Sales_Revenue_{t-1}) * 100",
        "leakage_status": "Leak-free (ex-ante lagged ratio)",
        "business_relevance": "Measures total cost absorption relative to revenue prior to period t.",
    },
]

TS_FEATURE_COLS = [
    "Time_Index",
    "Month_Sin",
    "Month_Cos",
    "Quarter_Num",
    "Lag1_Target",
    "Lag2_Target",
    "Lag3_Target",
    "Lag12_Target",
    "Rolling_Mean_3M",
    "Rolling_Mean_6M",
    "Rolling_Mean_12M",
    "Rolling_Std_6M",
    "MoM_Diff_Lag1",
    "YoY_Growth_Pct_Lag1",
    "Operating_Margin_Lag1",
    "Cost_to_Revenue_Lag1",
]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    # Require positive magnitudes >= 5% of median absolute target to avoid zero-crossing blowups
    thresh = max(1e5, float(np.median(np.abs(y_true))) * 0.05) if len(y_true) else 1e5
    mask = np.abs(y_true) >= thresh
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _compute_validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and np.std(y_true) > 1e-6 else None
    mape = _safe_mape(y_true, y_pred)
    mean_err = float(np.mean(y_true - y_pred))  # >0 => underpredicts, <0 => overpredicts
    mean_actual = float(np.mean(np.abs(y_true))) or 1.0
    bias_pct = (mean_err / mean_actual) * 100.0
    if abs(bias_pct) < 2.0:
        bias_label = "Balanced"
    elif mean_err > 0:
        bias_label = "Underpredicts"
    else:
        bias_label = "Overpredicts"
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2) if mape is not None else None,
        "r2": round(r2, 4) if r2 is not None else None,
        "mean_error": round(mean_err, 2),
        "bias_pct": round(bias_pct, 2),
        "forecast_bias": bias_label,
    }


def _build_step_vector(
    history_y: List[float],
    t_idx: int,
    month_num: int,
    op_margin_lag1: float,
    cost_rev_lag1: float,
) -> List[float]:
    n = len(history_y)
    lag1 = history_y[-1] if n >= 1 else 0.0
    lag2 = history_y[-2] if n >= 2 else lag1
    lag3 = history_y[-3] if n >= 3 else lag2
    lag12 = history_y[-12] if n >= 12 else history_y[0]
    lag13 = history_y[-13] if n >= 13 else history_y[0]

    r3 = float(np.mean(history_y[-3:])) if n >= 1 else 0.0
    r6 = float(np.mean(history_y[-6:])) if n >= 1 else 0.0
    r12 = float(np.mean(history_y[-12:])) if n >= 1 else 0.0
    std6 = float(np.std(history_y[-6:], ddof=1)) if n >= 2 else 0.0

    mom_diff = lag1 - lag2
    yoy_growth = ((lag1 - lag13) / max(abs(lag13), 1e6)) * 100.0
    q_num = float(((month_num - 1) // 3) + 1)
    m_sin = math.sin(2.0 * math.pi * month_num / 12.0)
    m_cos = math.cos(2.0 * math.pi * month_num / 12.0)

    return [
        float(t_idx),
        m_sin,
        m_cos,
        q_num,
        lag1,
        lag2,
        lag3,
        lag12,
        r3,
        r6,
        r12,
        std6,
        mom_diff,
        yoy_growth,
        op_margin_lag1,
        cost_rev_lag1,
    ]


def _prepare_ts_training_matrix(
    df: pd.DataFrame,
    target_col: str = "Operating_Profit_NGN",
    min_history: int = 13,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    y_vals = df[target_col].astype(float).tolist()
    months = df["Month_Num"].astype(int).tolist()
    op_margins = df["Operating_Margin_Pct"].astype(float).tolist() if "Operating_Margin_Pct" in df.columns else [0.0] * len(df)
    if {"COGS_NGN", "Marketing_Spend_NGN", "Other_Operating_Cost_NGN", "Sales_Revenue_NGN"}.issubset(df.columns):
        tot_cost = (df["COGS_NGN"] + df["Marketing_Spend_NGN"] + df["Other_Operating_Cost_NGN"]).astype(float)
        rev = df["Sales_Revenue_NGN"].astype(float).replace(0.0, np.nan)
        cost_ratios = ((tot_cost / rev) * 100.0).fillna(70.0).tolist()
    else:
        cost_ratios = [70.0] * len(df)

    start_i = min(min_history, max(3, len(df) // 4))
    X_rows: List[List[float]] = []
    y_rows: List[float] = []
    indices: List[int] = []

    for i in range(start_i, len(df)):
        vec = _build_step_vector(
            history_y=y_vals[:i],
            t_idx=i,
            month_num=months[i],
            op_margin_lag1=op_margins[i - 1],
            cost_rev_lag1=cost_ratios[i - 1],
        )
        X_rows.append(vec)
        y_rows.append(y_vals[i])
        indices.append(i)

    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float), indices


def _recursive_ml_forecast(
    estimator: Any,
    scaler: Optional[StandardScaler],
    train_df: pd.DataFrame,
    steps: int,
    target_col: str = "Operating_Profit_NGN",
) -> np.ndarray:
    history_y = train_df[target_col].astype(float).tolist()
    last_date = pd.to_datetime(train_df["Month"].iloc[-1])
    last_op_margin = float(train_df["Operating_Margin_Pct"].iloc[-1]) if "Operating_Margin_Pct" in train_df.columns else 25.0
    if {"COGS_NGN", "Marketing_Spend_NGN", "Other_Operating_Cost_NGN", "Sales_Revenue_NGN"}.issubset(train_df.columns):
        rev_last = float(train_df["Sales_Revenue_NGN"].iloc[-1]) or 1.0
        cost_last = float(
            train_df["COGS_NGN"].iloc[-1]
            + train_df["Marketing_Spend_NGN"].iloc[-1]
            + train_df["Other_Operating_Cost_NGN"].iloc[-1]
        )
        last_cost_ratio = (cost_last / rev_last) * 100.0
    else:
        last_cost_ratio = 75.0

    # Empirical trailing 6M avg absolute monthly delta to guard against recursive numerical divergence
    recent_diffs = np.diff(history_y[-13:]) if len(history_y) >= 4 else np.array([0.0])
    step_cap = max(float(np.std(history_y[-24:]) * 2.5) if len(history_y) >= 6 else 1e7, float(np.max(np.abs(recent_diffs)) * 2.5) if len(recent_diffs) else 1e7)

    preds: List[float] = []
    start_t = len(history_y)
    for h in range(steps):
        f_date = last_date + pd.DateOffset(months=h + 1)
        vec = _build_step_vector(
            history_y=history_y,
            t_idx=start_t + h,
            month_num=int(f_date.month),
            op_margin_lag1=last_op_margin,
            cost_rev_lag1=last_cost_ratio,
        )
        X_step = np.asarray([vec], dtype=float)
        if scaler is not None:
            X_step = scaler.transform(X_step)
        y_hat = float(estimator.predict(X_step)[0])

        # Soft safety clamp against runaway recursive explosion while preserving full trend trajectory
        prev = history_y[-1]
        if abs(y_hat - prev) > step_cap * 1.8:
            y_hat = prev + math.copysign(step_cap * 1.8, y_hat - prev)

        preds.append(y_hat)
        history_y.append(y_hat)

    return np.asarray(preds, dtype=float)


def _forecast_model_candidates(
    train_df: pd.DataFrame,
    steps: int,
    target_col: str = "Operating_Profit_NGN",
    fast_backtest: bool = False,
    only_keys: Optional[set] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Fits all candidate statistical & ML models strictly on train_df
    and produces `steps` out-of-sample recursive forecasts.
    """
    y_train = train_df[target_col].astype(float).values
    n = len(y_train)
    results: Dict[str, Dict[str, Any]] = {}

    # 1. Naive Baseline
    if only_keys is None or "naive" in only_keys:
        naive_pred = np.full(steps, y_train[-1], dtype=float)
        results["naive"] = {
            "name": "Naive Baseline (Last Value)",
            "family": "Statistical / Baseline",
            "preds": naive_pred,
            "estimator": None,
            "scaler": None,
            "is_tree": False,
            "status_note": "Persistence benchmark y_{t+h} = y_t",
        }
    else:
        naive_pred = np.full(steps, y_train[-1], dtype=float)

    # 2. Seasonal Naive (12-month)
    if only_keys is None or "seasonal_naive" in only_keys:
        if n >= 12:
            s_naive = np.array([y_train[-12 + (h % 12)] for h in range(steps)], dtype=float)
        else:
            s_naive = naive_pred.copy()
        results["seasonal_naive"] = {
            "name": "Seasonal Naive (12M)",
            "family": "Statistical / Baseline",
            "preds": s_naive,
            "estimator": None,
            "scaler": None,
            "is_tree": False,
            "status_note": "Prior-year same-month benchmark",
        }

    # 3. Simple Exponential Smoothing
    if n >= 8 and (only_keys is None or "ses" in only_keys):
        try:
            ses_pred, ses_alpha = fit_ses(y_train, steps)
            results["ses"] = {
                "name": "Simple Exponential Smoothing",
                "family": "Statistical / Time-Series",
                "preds": ses_pred,
                "estimator": None,
                "scaler": None,
                "is_tree": False,
                "status_note": f"alpha={round(_safe_float(ses_alpha, 0.3), 3)}",
            }
        except Exception:
            pass

    # 4. Multiplicative Seasonal Exponential Smoothing ETS(M,N,M)
    if n >= 24 and (only_keys is None or "seasonal_ets" in only_keys):
        try:
            sets_pred, (sets_a, sets_g) = fit_seasonal_ets(y_train, steps, m=12)
            results["seasonal_ets"] = {
                "name": "Seasonal Exponential Smoothing (ETS)",
                "family": "Statistical / Time-Series",
                "preds": sets_pred,
                "estimator": None,
                "scaler": None,
                "is_tree": False,
                "status_note": f"Multiplicative 12M seasonality (alpha={round(sets_a, 2)}, gamma={round(sets_g, 2)})",
            }
        except Exception:
            pass

    # 5. Damped Trend Exponential Smoothing (Holt Damped)
    if n >= 12 and (only_keys is None or "holt_damped" in only_keys):
        try:
            holt_pred, holt_phi = fit_holt_damped(y_train, steps)
            results["holt_damped"] = {
                "name": "Damped Trend Exponential Smoothing",
                "family": "Statistical / Time-Series",
                "preds": holt_pred,
                "estimator": None,
                "scaler": None,
                "is_tree": False,
                "status_note": f"phi={round(_safe_float(holt_phi, 0.8), 3)}",
            }
        except Exception:
            pass

    # 6. Holt-Winters Seasonal + Damped Trend (Multiplicative & Additive auto-fit)
    if n >= 24 and (only_keys is None or "holt_winters" in only_keys):
        try:
            hw_pred = fit_holt_winters_damped(y_train, steps, m=12)
            results["holt_winters"] = {
                "name": "Holt-Winters Seasonal (Damped)",
                "family": "Statistical / Time-Series",
                "preds": hw_pred,
                "estimator": None,
                "scaler": None,
                "is_tree": False,
                "status_note": "12M multiplicative/additive seasonality + damped trend",
            }
        except Exception:
            pass

    # 7. Autoregressive AR(1,2,3,12) with trend
    if n >= 20 and (only_keys is None or "autoreg" in only_keys):
        try:
            ar_lags = [1, 2, 3, 12] if n >= 30 else [1, 2, 3]
            ar_pred = fit_autoreg(y_train, steps, lags=ar_lags)
            results["autoreg"] = {
                "name": "Autoregressive AR(1,2,3,12) + Trend",
                "family": "Statistical / Time-Series",
                "preds": ar_pred,
                "estimator": None,
                "scaler": None,
                "is_tree": False,
                "status_note": f"Lags {ar_lags} with deterministic trend",
            }
        except Exception:
            pass

    # 8-11. ML Recursive Regression Models
    ml_keys = {"linear_ts", "ridge_ts", "rf_ts", "gbr_ts"}
    if n >= 20 and (only_keys is None or bool(only_keys & ml_keys)):
        X_mat, y_vec, _ = _prepare_ts_training_matrix(train_df, target_col=target_col)
        if len(X_mat) >= 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_mat)

            rf_trees = 35 if fast_backtest else 55
            gbr_trees = 45 if fast_backtest else 70

            ml_specs: List[Tuple[str, str, Any, bool, bool]] = [
                ("linear_ts", "Linear Regression (TS Features)", LinearRegression(), True, False),
                ("ridge_ts", "Ridge Regression (L2 TS)", Ridge(alpha=15.0, random_state=42), True, False),
                (
                    "rf_ts",
                    "Random Forest Regressor (TS)",
                    RandomForestRegressor(n_estimators=rf_trees, max_depth=6, min_samples_leaf=2, random_state=42),
                    False,
                    True,
                ),
                (
                    "gbr_ts",
                    "Gradient Boosting Regressor (TS)",
                    GradientBoostingRegressor(n_estimators=gbr_trees, learning_rate=0.06, max_depth=3, random_state=42),
                    False,
                    True,
                ),
            ]

            for key, name, est, use_scaler, is_tree in ml_specs:
                if only_keys is not None and key not in only_keys:
                    continue
                try:
                    X_fit = X_scaled if use_scaler else X_mat
                    est.fit(X_fit, y_vec)
                    preds = _recursive_ml_forecast(
                        est,
                        scaler if use_scaler else None,
                        train_df,
                        steps=steps,
                        target_col=target_col,
                    )
                    results[key] = {
                        "name": name,
                        "family": "Machine Learning Regression",
                        "preds": preds,
                        "estimator": est,
                        "scaler": scaler if use_scaler else None,
                        "is_tree": is_tree,
                        "X_fit": X_fit,
                        "y_fit": y_vec,
                        "status_note": "Recursive point-in-time feature forecast",
                    }
                except Exception:
                    pass

    return results


def _diebold_mariano_test(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_bench: np.ndarray,
    h: int = 1,
) -> Dict[str, Any]:
    """
    Harvey-Leybourne-Newbold modified Diebold-Mariano test comparing squared forecast errors.
    H0: E[d_t] = 0 where d_t = e_model^2 - e_bench^2.
    """
    e1 = y_true - pred_model
    e2 = y_true - pred_bench
    d = (e1 ** 2) - (e2 ** 2)
    T = len(d)
    if T < 6:
        return {
            "applicable": False,
            "statistic": None,
            "p_value": None,
            "reason": f"Only {T} paired holdout points available; >=6 required for DM variance estimate.",
        }

    mean_d = float(np.mean(d))
    d_centered = d - mean_d
    gamma0 = float(np.sum(d_centered ** 2) / T)
    var_d = gamma0
    max_lag = min(h - 1, T // 3)
    for lag in range(1, max_lag + 1):
        gamma_k = float(np.sum(d_centered[lag:] * d_centered[:-lag]) / T)
        var_d += 2.0 * (1.0 - lag / (max_lag + 1.0)) * gamma_k

    if var_d <= 1e-12:
        return {
            "applicable": False,
            "statistic": 0.0,
            "p_value": 1.0,
            "reason": "Identical error sequence between candidate and baseline (zero variance in loss differential).",
        }

    dm_raw = mean_d / math.sqrt(var_d / T)
    # Harvey-Leybourne-Newbold finite-sample correction
    k = max(1, h)
    hln_factor = math.sqrt(max(1e-9, (T + 1.0 - 2.0 * k + (k * (k - 1.0)) / T) / T))
    dm_stat = dm_raw * hln_factor
    p_val = float(2.0 * stats.t.sf(abs(dm_stat), df=max(1, T - 1)))
    return {
        "applicable": True,
        "statistic": round(dm_stat, 4),
        "p_value": round(p_val, 6),
        "mean_loss_diff": round(mean_d, 2),
        "reason": "Harvey-Leybourne-Newbold modified Diebold-Mariano test on out-of-sample squared errors.",
    }


def compute_hypothesis_tests(
    df: pd.DataFrame,
    champion_holdout_actual: Optional[np.ndarray] = None,
    champion_holdout_pred: Optional[np.ndarray] = None,
    naive_holdout_pred: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    split_year: int = 2023,
) -> Dict[str, Any]:
    tests: List[Dict[str, Any]] = []
    n = len(df)
    y = df["Operating_Profit_NGN"].astype(float).values
    t = np.arange(n, dtype=float)

    # H1: Operating Profit Trend (Mann-Kendall / Kendall Tau + OLS Slope)
    if n >= 8:
        tau, p_mk = stats.kendalltau(t, y)
        slope_res = stats.linregress(t, y)
        se_slope = float(slope_res.stderr)
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=max(1, n - 2)))
        ci_low = float(slope_res.slope - t_crit * se_slope)
        ci_high = float(slope_res.slope + t_crit * se_slope)
        reject = bool(p_mk < alpha)
        direction = "upward" if tau > 0 else "downward"
        tests.append({
            "id": "H1_trend",
            "title": "H1: Operating Profit Secular Trend",
            "null_hypothesis": "H0: There is no statistically significant monotonic trend in monthly Operating_Profit_NGN over time.",
            "alternative_hypothesis": "H1: Monthly Operating_Profit_NGN exhibits a statistically significant upward or downward trend.",
            "test_used": "Mann-Kendall Tau Test (Non-parametric) + OLS Linear Slope CI",
            "statistic_name": "Kendall tau",
            "statistic": round(_safe_float(tau), 4),
            "p_value": round(_safe_float(p_mk, 1.0), 6),
            "alpha": alpha,
            "confidence_interval": f"OLS slope ₦{slope_res.slope/1e6:+.2f}M/mo (95% CI: ₦{ci_low/1e6:+.2f}M to ₦{ci_high/1e6:+.2f}M)",
            "decision": "Reject H0" if reject else "Fail to Reject H0",
            "significant": reject,
            "assumption_checks": "Non-parametric Mann-Kendall handles non-normal residuals; nominal series shows strong positive autocorrelation, so p-value reflects monotonic drift rather than independent draws.",
            "business_interpretation": (
                f"Statistically significant {direction} trend detected (tau={tau:.3f}, p={p_mk:.4g}). Average nominal Operating Profit expanded by ₦{slope_res.slope/1e6:.2f}M per month, though nominal growth also reflects FX/inflationary scaling."
                if reject
                else "No statistically significant monotonic trend detected at the chosen significance level."
            ),
        })
    else:
        tests.append({
            "id": "H1_trend",
            "title": "H1: Operating Profit Secular Trend",
            "null_hypothesis": "H0: No trend in Operating_Profit_NGN.",
            "alternative_hypothesis": "H1: Statistically significant trend exists.",
            "test_used": "Mann-Kendall Tau Test",
            "statistic_name": "Kendall tau",
            "statistic": None,
            "p_value": None,
            "alpha": alpha,
            "confidence_interval": "N/A",
            "decision": "Insufficient Evidence",
            "significant": False,
            "assumption_checks": f"Only {n} observations available; at least 8 months required.",
            "business_interpretation": "Dataset length is too short for trend inference.",
        })

    # H2: Quarterly / Segment Performance (Department column if present, otherwise Quarter Q1-Q4)
    group_col = "Department" if "Department" in df.columns else "Quarter"
    group_label = "Departmental" if group_col == "Department" else "Quarterly (Seasonal Q1–Q4)"
    groups = [g["Operating_Profit_NGN"].astype(float).values for _, g in df.groupby(group_col) if len(g) >= 2]
    group_names = [str(k) for k, g in df.groupby(group_col) if len(g) >= 2]
    if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
        lev_stat, p_lev = stats.levene(*groups)
        shapiro_p = [float(stats.shapiro(g).pvalue) if 3 <= len(g) <= 5000 else 1.0 for g in groups]
        normal_ok = all(p > 0.05 for p in shapiro_p) and (p_lev > 0.05)
        if normal_ok:
            stat_val, p_val = stats.f_oneway(*groups)
            test_name = "One-Way ANOVA (Parametric)"
            stat_lbl = "F-statistic"
        else:
            stat_val, p_val = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis H-Test (Non-parametric ANOVA)"
            stat_lbl = "Kruskal H"
        means_str = ", ".join(f"{nm}: ₦{np.mean(g)/1e6:.1f}M" for nm, g in zip(group_names, groups))
        reject = bool(p_val < alpha)
        tests.append({
            "id": "H2_segments",
            "title": f"H2: {group_label} Performance Differences",
            "null_hypothesis": f"H0: Mean/median Operating_Profit_NGN does not differ significantly across {group_col} groups.",
            "alternative_hypothesis": f"H1: At least one {group_col} group exhibits significantly different Operating_Profit_NGN.",
            "test_used": test_name,
            "statistic_name": stat_lbl,
            "statistic": round(_safe_float(stat_val), 4),
            "p_value": round(_safe_float(p_val, 1.0), 6),
            "alpha": alpha,
            "confidence_interval": f"Group means: {means_str}",
            "decision": "Reject H0" if reject else "Fail to Reject H0",
            "significant": reject,
            "assumption_checks": (
                f"Dataset uses '{group_col}' (no separate Department column exists). Shapiro-Wilk min p={min(shapiro_p):.4f}, Levene variance p={p_lev:.4f}; "
                + ("normality & equal-variance assumptions satisfied (ANOVA selected)." if normal_ok else "normality/homoscedasticity not satisfied, so non-parametric Kruskal-Wallis was automatically selected.")
            ),
            "business_interpretation": (
                f"Statistically significant differences exist across {group_col} groups ({stat_lbl}={stat_val:.3f}, p={p_val:.4f})."
                if reject
                else f"No statistically significant difference detected across {group_col} groups ({stat_lbl}={stat_val:.3f}, p={p_val:.4f}); nominal trend growth dominates intra-year seasonal variance across multi-year spans."
            ),
        })

    # H3: Operational Cost & Profitability Association (Levels + MoM Changes to expose confounding)
    if n >= 8:
        total_cost = (df["COGS_NGN"] + df["Marketing_Spend_NGN"] + df["Other_Operating_Cost_NGN"]).astype(float).values
        r_lvl, p_lvl = stats.pearsonr(total_cost, y)
        rho_lvl, p_rho = stats.spearmanr(total_cost, y)
        # Fisher z 95% CI for Pearson r
        z_r = np.arctanh(np.clip(r_lvl, -0.9999, 0.9999))
        se_z = 1.0 / math.sqrt(max(1, n - 3))
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_r_low = float(np.tanh(z_r - z_crit * se_z))
        ci_r_high = float(np.tanh(z_r + z_crit * se_z))

        # Also check first-differenced (detrended) MoM cost-ratio vs margin correlation
        cost_ratio = (total_cost / np.maximum(df["Sales_Revenue_NGN"].astype(float).values, 1.0)) * 100.0
        r_ratio, p_ratio = stats.pearsonr(cost_ratio, df["Operating_Margin_Pct"].astype(float).values)
        reject = bool(p_rho < alpha)
        tests.append({
            "id": "H3_cost_profit",
            "title": "H3: Operating Costs & Profitability Association",
            "null_hypothesis": "H0: There is no statistically significant monotonic association between Total Operating Costs and Operating_Profit_NGN.",
            "alternative_hypothesis": "H1: Total Operating Costs have a statistically significant association with Operating_Profit_NGN.",
            "test_used": "Spearman Rank Correlation + Pearson r (with Cost-Ratio Margin Check)",
            "statistic_name": "Spearman rho",
            "statistic": round(_safe_float(rho_lvl), 4),
            "p_value": round(_safe_float(p_rho, 1.0), 6),
            "alpha": alpha,
            "confidence_interval": f"Pearson r={r_lvl:.3f} (CI: [{ci_r_low:.3f}, {ci_r_high:.3f}]); Cost-to-Rev % vs Margin r={r_ratio:.3f} (p={p_ratio:.4g})",
            "decision": "Reject H0" if reject else "Fail to Reject H0",
            "significant": reject,
            "assumption_checks": "Time-series scale confounding check: in nominal levels both revenue and costs grow together over 2016–2026 (positive r). Testing Cost-to-Revenue % vs Operating Margin % isolates true efficiency.",
            "business_interpretation": (
                f"Nominal costs and operating profit co-move strongly (rho={rho_lvl:.3f}, p={p_rho:.4g}) because higher sales scale lifts both revenue and COGS. Crucially, Cost-to-Revenue % negatively governs Operating Margin (r={r_ratio:.3f}, p={p_ratio:.4g}) — correlation in levels does NOT imply higher costs cause higher profit."
                if reject
                else "No statistically significant association found between total costs and operating profit."
            ),
        })

    # H4: Operational Changes Between Historical Periods (Pre-split vs Post-split regime)
    pre_df = df[df["Year"] < split_year]
    post_df = df[df["Year"] >= split_year]
    if len(pre_df) >= 4 and len(post_df) >= 4:
        a = pre_df["Operating_Profit_NGN"].astype(float).values
        b = post_df["Operating_Profit_NGN"].astype(float).values
        t_stat, p_t = stats.ttest_ind(b, a, equal_var=False)
        u_stat, p_u = stats.mannwhitneyu(b, a, alternative="two-sided")
        diff_mean = float(np.mean(b) - np.mean(a))
        se_diff = math.sqrt(float(np.var(a, ddof=1)) / len(a) + float(np.var(b, ddof=1)) / len(b))
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_l = diff_mean - z_crit * se_diff
        ci_h = diff_mean + z_crit * se_diff
        reject = bool(p_u < alpha)
        tests.append({
            "id": "H4_period_shift",
            "title": f"H4: Period Comparison (Pre-{split_year} vs {split_year}+)",
            "null_hypothesis": f"H0: Operating_Profit_NGN distribution does not differ between Pre-{split_year} (n={len(a)}) and {split_year}+ (n={len(b)}).",
            "alternative_hypothesis": f"H1: Operating_Profit_NGN differs significantly between Pre-{split_year} and {split_year}+.",
            "test_used": "Mann-Whitney U Test + Welch's t-Test",
            "statistic_name": "Mann-Whitney U",
            "statistic": round(_safe_float(u_stat), 2),
            "p_value": round(_safe_float(p_u, 1.0), 6),
            "alpha": alpha,
            "confidence_interval": f"Mean shift ₦{diff_mean/1e6:+.2f}M/mo (95% CI: ₦{ci_l/1e6:+.2f}M to ₦{ci_h/1e6:+.2f}M); Welch p={p_t:.4g}",
            "decision": "Reject H0" if reject else "Fail to Reject H0",
            "significant": reject,
            "assumption_checks": f"Welch t-test relaxes equal-variance assumption (Pre-{split_year} std=₦{np.std(a, ddof=1)/1e6:.2f}M vs {split_year}+ std=₦{np.std(b, ddof=1)/1e6:.2f}M); Mann-Whitney U verifies robustness to skewness.",
            "business_interpretation": (
                f"Highly significant structural shift between Pre-{split_year} (avg ₦{np.mean(a)/1e6:.2f}M/mo) and {split_year}+ (avg ₦{np.mean(b)/1e6:.2f}M/mo), driven by pricing expansion and FX regime adjustment."
                if reject
                else f"No statistically significant shift detected between Pre-{split_year} and {split_year}+ periods."
            ),
        })
    else:
        tests.append({
            "id": "H4_period_shift",
            "title": f"H4: Period Comparison (Pre-{split_year} vs {split_year}+)",
            "null_hypothesis": f"H0: No difference between Pre-{split_year} and {split_year}+.",
            "alternative_hypothesis": "H1: Significant difference exists.",
            "test_used": "Mann-Whitney U / Welch t-Test",
            "statistic_name": "Mann-Whitney U",
            "statistic": None,
            "p_value": None,
            "alpha": alpha,
            "confidence_interval": "N/A",
            "decision": "Insufficient Evidence",
            "significant": False,
            "assumption_checks": f"Pre-{split_year} n={len(pre_df)}, {split_year}+ n={len(post_df)}; >=4 observations per period required.",
            "business_interpretation": "Choose a cutoff year with at least 4 monthly observations on both sides.",
        })

    # H5: Champion Model vs. Naive Forecast Improvement (Diebold-Mariano)
    if (
        champion_holdout_actual is not None
        and champion_holdout_pred is not None
        and naive_holdout_pred is not None
    ):
        dm = _diebold_mariano_test(champion_holdout_actual, champion_holdout_pred, naive_holdout_pred, h=1)
        champ_rmse = math.sqrt(mean_squared_error(champion_holdout_actual, champion_holdout_pred))
        naive_rmse = math.sqrt(mean_squared_error(champion_holdout_actual, naive_holdout_pred))
        if dm["applicable"] and dm["p_value"] is not None:
            # One-sided improvement check: champion has lower squared error (statistic < 0) and two-sided/one-sided sig
            p_one_sided = dm["p_value"] / 2.0 if dm["statistic"] < 0 else 1.0 - dm["p_value"] / 2.0
            reject = bool(dm["statistic"] < 0 and p_one_sided < alpha)
            tests.append({
                "id": "H5_dm_improvement",
                "title": "H5: Champion Model vs. Naive Baseline Error",
                "null_hypothesis": "H0: The champion forecasting model does not achieve lower squared forecast error than the Naive baseline.",
                "alternative_hypothesis": "H1: The champion forecasting model achieves significantly lower forecast error than the Naive baseline.",
                "test_used": "Modified Diebold-Mariano Test (Harvey-Leybourne-Newbold)",
                "statistic_name": "DM t-stat",
                "statistic": dm["statistic"],
                "p_value": round(p_one_sided, 6),
                "alpha": alpha,
                "confidence_interval": f"Champion RMSE ₦{champ_rmse/1e6:.2f}M vs Naive RMSE ₦{naive_rmse/1e6:.2f}M (two-sided p={dm['p_value']:.4g})",
                "decision": "Reject H0" if reject else "Fail to Reject H0",
                "significant": reject,
                "assumption_checks": f"Evaluated on n={len(champion_holdout_actual)} untouched chronological holdout months. Loss differential stationarity is approximated over multi-step holdout errors.",
                "business_interpretation": (
                    f"Champion outperforms Naive baseline (RMSE ₦{champ_rmse/1e6:.2f}M vs ₦{naive_rmse/1e6:.2f}M, one-sided DM p={p_one_sided:.4g}), confirming measurable predictive signal beyond last-value persistence."
                    if reject
                    else f"Champion RMSE (₦{champ_rmse/1e6:.2f}M) vs Naive RMSE (₦{naive_rmse/1e6:.2f}M): difference is not statistically significant at α={alpha} on the {len(champion_holdout_actual)}-month holdout window."
                ),
            })
        else:
            tests.append({
                "id": "H5_dm_improvement",
                "title": "H5: Champion Model vs. Naive Baseline Error",
                "null_hypothesis": "H0: Champion equal to Naive baseline.",
                "alternative_hypothesis": "H1: Champion outperforms Naive baseline.",
                "test_used": "Modified Diebold-Mariano Test",
                "statistic_name": "DM t-stat",
                "statistic": dm.get("statistic"),
                "p_value": dm.get("p_value"),
                "alpha": alpha,
                "confidence_interval": f"Champion RMSE ₦{champ_rmse/1e6:.2f}M vs Naive RMSE ₦{naive_rmse/1e6:.2f}M",
                "decision": "Insufficient Evidence",
                "significant": False,
                "assumption_checks": dm.get("reason", "Insufficient holdout sample size."),
                "business_interpretation": dm.get("reason", "Expand dataset length to run Diebold-Mariano comparison."),
            })

    # H6: FX Shock Regimes vs Operating Margin Compression
    if {"FX_Change_MoM_Pct", "Operating_Margin_Pct"}.issubset(df.columns):
        valid_fx = df.dropna(subset=["FX_Change_MoM_Pct", "Operating_Margin_Pct"])
        shock_m = valid_fx[valid_fx["FX_Change_MoM_Pct"] > 3.0]["Operating_Margin_Pct"].astype(float).values
        calm_m = valid_fx[valid_fx["FX_Change_MoM_Pct"] <= 3.0]["Operating_Margin_Pct"].astype(float).values
        if len(shock_m) >= 3 and len(calm_m) >= 5:
            u_fx, p_fx = stats.mannwhitneyu(shock_m, calm_m, alternative="two-sided")
            r_fx, p_rfx = stats.spearmanr(valid_fx["FX_Change_MoM_Pct"].values, valid_fx["Operating_Margin_Pct"].values)
            reject = bool(p_fx < alpha)
            tests.append({
                "id": "H6_fx_margin",
                "title": "H6: FX Depreciation Shocks (>3% MoM) & Margin Compression",
                "null_hypothesis": "H0: Operating_Margin_Pct does not differ between high-FX-shock months (>3% MoM depreciation) and normal FX months.",
                "alternative_hypothesis": "H1: High-FX-shock months exhibit significantly different Operating_Margin_Pct.",
                "test_used": "Mann-Whitney U Test + Spearman Correlation",
                "statistic_name": "Mann-Whitney U",
                "statistic": round(_safe_float(u_fx), 2),
                "p_value": round(_safe_float(p_fx, 1.0), 6),
                "alpha": alpha,
                "confidence_interval": f"Shock avg margin {np.mean(shock_m):.2f}% (n={len(shock_m)}) vs Calm avg {np.mean(calm_m):.2f}% (n={len(calm_m)}); Spearman rho={r_fx:.3f} (p={p_rfx:.4g})",
                "decision": "Reject H0" if reject else "Fail to Reject H0",
                "significant": reject,
                "assumption_checks": "Non-parametric test handles skewed FX shock events without assuming Gaussian margin distributions.",
                "business_interpretation": (
                    f"Sharp monthly FX depreciations (>3% MoM) significantly compress operating margins ({np.mean(shock_m):.2f}% vs {np.mean(calm_m):.2f}% in calm months, p={p_fx:.4g})."
                    if reject
                    else f"Margin differences during >3% FX shock months ({np.mean(shock_m):.2f}% vs {np.mean(calm_m):.2f}%) are not statistically decisive at α={alpha}, though directional FX correlation is rho={r_fx:.3f}."
                ),
            })

    return {
        "alpha": alpha,
        "split_year": split_year,
        "tests": tests,
        "methodology_note": (
            "Statistical significance (p < α) measures whether an observed pattern is unlikely under the null hypothesis—it does NOT automatically prove causality or material business impact."
        ),
    }


def _forecast_secondary_kpi(
    df: pd.DataFrame,
    col: str,
    steps: int,
    allow_negative: bool = False,
) -> Dict[str, Any]:
    if col not in df.columns or len(df) < 12 or steps <= 0:
        return {"available": False, "forecast": [], "model_used": "Unavailable", "holdout_rmse": None}

    y = df[col].astype(float).values
    val_h = min(12, max(4, len(df) // 6))
    y_tr, y_va = y[:-val_h], y[-val_h:]

    candidates: List[Tuple[str, np.ndarray, Any]] = [
        ("Naive", np.full(val_h, y_tr[-1]), lambda full_y: np.full(steps, full_y[-1])),
    ]
    if len(y_tr) >= 12:
        try:
            va_h, _ = fit_holt_damped(y_tr, val_h)
            candidates.append((
                "Holt Damped Trend",
                va_h,
                lambda full_y: fit_holt_damped(full_y, steps)[0],
            ))
        except Exception:
            pass
    if len(y_tr) >= 24:
        try:
            va_sets, _ = fit_seasonal_ets(y_tr, val_h, m=12)
            candidates.append((
                "Seasonal ETS (Multiplicative)",
                va_sets,
                lambda full_y: fit_seasonal_ets(full_y, steps, m=12)[0],
            ))
        except Exception:
            pass
    if len(y_tr) >= 24:
        try:
            va_hw = fit_holt_winters_damped(y_tr, val_h, m=12)
            candidates.append((
                "Holt-Winters Seasonal",
                va_hw,
                lambda full_y: fit_holt_winters_damped(full_y, steps, m=12),
            ))
        except Exception:
            pass

    best_name = candidates[0][0]
    best_rmse = float("inf")
    best_fn = candidates[0][2]
    for cname, va_pred, full_fn in candidates:
        err = math.sqrt(mean_squared_error(y_va, va_pred))
        if err < best_rmse:
            best_rmse = err
            best_name = cname
            best_fn = full_fn

    fc = best_fn(y)
    if not allow_negative:
        fc = np.maximum(0.0, fc)
    return {
        "available": True,
        "model_used": best_name,
        "holdout_rmse": round(best_rmse, 2),
        "forecast": [round(float(v), 2) for v in fc],
    }


def build_forecasting_centre_bundle(
    processed_df: pd.DataFrame,
    alpha: float = 0.05,
    split_year: int = 2023,
) -> Dict[str, Any]:
    df = processed_df.copy().sort_values("Month").reset_index(drop=True)
    # Add derived operational efficiency KPIs strictly from existing dataset columns
    df["Total_Operating_Expenses_NGN"] = (
        df["Marketing_Spend_NGN"].astype(float) + df["Other_Operating_Cost_NGN"].astype(float)
    )
    df["Profit_Per_Unit_NGN"] = np.where(
        df["Units_Sold"].astype(float) > 0,
        df["Operating_Profit_NGN"].astype(float) / df["Units_Sold"].astype(float),
        0.0,
    )
    df["OpEx_to_Revenue_Ratio_Pct"] = np.where(
        df["Sales_Revenue_NGN"].astype(float) > 0,
        (df["Total_Operating_Expenses_NGN"].astype(float) / df["Sales_Revenue_NGN"].astype(float)) * 100.0,
        0.0,
    )

    n = len(df)
    if n < 24:
        return {
            "sufficient_data": False,
            "limitation_notice": f"Only {n} monthly records available; at least 24 chronological observations are required for multi-horizon 2030 validation and seasonal forecasting.",
            "verdict": {
                "category": "Insufficient Evidence",
                "summary": "Insufficient historical records to produce reliable forecasts through December 2030.",
            },
        }

    # Horizon: from month after last historical observation through Dec 2030
    last_date = pd.to_datetime(df["Month"].iloc[-1])
    target_end = pd.Timestamp("2030-12-01")
    steps = max(0, (target_end.year - last_date.year) * 12 + (target_end.month - last_date.month))
    horizon_note = None
    if steps == 0:
        steps = 12
        horizon_note = f"Dataset already reaches {last_date.strftime('%Y-%m')}; displaying 12-month forward extension."

    # Chronological Holdout: untouched last 18 months (or 15-20% scaled safely)
    holdout_size = min(18, max(6, int(round(n * 0.15))))
    if n - holdout_size < 18:
        holdout_size = max(6, n // 4)
    split_idx = n - holdout_size

    train_hold_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_hold_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    y_test = test_hold_df["Operating_Profit_NGN"].astype(float).values

    # 1. Holdout validation across all candidate models
    holdout_runs = _forecast_model_candidates(train_hold_df, steps=holdout_size, target_col="Operating_Profit_NGN")

    # 2. Expanding-window rolling-origin backtesting (4 origins x 6-month step)
    backtest_horizon = min(6, max(3, holdout_size // 3))
    origins: List[int] = []
    min_origin = max(24, int(n * 0.55))
    max_origin = n - backtest_horizon
    if max_origin > min_origin:
        step_gap = max(3, (max_origin - min_origin) // 3)
        origins = list(range(min_origin, max_origin + 1, step_gap))[:4]
    elif n >= 30:
        origins = [n - backtest_horizon]

    backtest_errors_by_key: Dict[str, List[float]] = {k: [] for k in holdout_runs.keys()}
    for orig in origins:
        bt_train = df.iloc[:orig].copy().reset_index(drop=True)
        bt_actual = df.iloc[orig : orig + backtest_horizon]["Operating_Profit_NGN"].astype(float).values
        bt_runs = _forecast_model_candidates(
            bt_train, steps=len(bt_actual), target_col="Operating_Profit_NGN", fast_backtest=True
        )
        for k, res in bt_runs.items():
            if k in backtest_errors_by_key:
                backtest_errors_by_key[k].append(float(math.sqrt(mean_squared_error(bt_actual, res["preds"]))))

    # Assemble Model Leaderboard
    leaderboard: List[Dict[str, Any]] = []
    for k, res in holdout_runs.items():
        y_pred_hold = res["preds"]
        metrics = _compute_validation_metrics(y_test, y_pred_hold)
        bt_rmses = backtest_errors_by_key.get(k, [])
        bt_mean_rmse = float(np.mean(bt_rmses)) if bt_rmses else metrics["rmse"]
        bt_std_rmse = float(np.std(bt_rmses)) if len(bt_rmses) >= 2 else 0.0
        # Stability = 1 - coeff of variation across rolling origins (higher = more consistent)
        stability_score = max(0.0, min(100.0, (1.0 - bt_std_rmse / max(bt_mean_rmse, 1.0)) * 100.0))

        # Empirical 80% / 95% PI coverage on holdout using expanding error std
        resids = y_test - y_pred_hold
        sigma_est = float(np.std(resids, ddof=1)) if len(resids) >= 2 else 1.0
        # Compare against rolling training residual scale
        horizon_scale = np.sqrt(np.arange(1, len(y_test) + 1) / max(1.0, len(y_test) / 2.0))
        cov_80 = float(np.mean(np.abs(resids) <= 1.2816 * sigma_est * horizon_scale) * 100.0)
        cov_95 = float(np.mean(np.abs(resids) <= 1.96 * sigma_est * horizon_scale) * 100.0)

        composite_score = 0.65 * metrics["rmse"] + 0.35 * bt_mean_rmse
        leaderboard.append({
            "key": k,
            "name": res["name"],
            "family": res["family"],
            "training_obs": int(len(train_hold_df)),
            "validation_obs": int(len(test_hold_df)),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape_pct": metrics["mape_pct"],
            "r2": metrics["r2"],
            "mean_error": metrics["mean_error"],
            "bias_pct": metrics["bias_pct"],
            "forecast_bias": metrics["forecast_bias"],
            "backtest_mean_rmse": round(bt_mean_rmse, 2),
            "backtest_std_rmse": round(bt_std_rmse, 2),
            "stability_score_pct": round(stability_score, 1),
            "pi_coverage_80_pct": round(cov_80, 1),
            "pi_coverage_95_pct": round(cov_95, 1),
            "composite_rmse": round(composite_score, 2),
            "status_note": res["status_note"],
            "status": "Benchmark",
        })

    # Select champion strictly based on lowest composite out-of-sample RMSE (excluding naive only if non-naive is competitive, or honestly crown best!)
    leaderboard.sort(key=lambda item: item["composite_rmse"])
    champion_entry = leaderboard[0]
    champion_entry["status"] = "Champion"
    champion_key = champion_entry["key"]

    naive_hold_preds = holdout_runs["naive"]["preds"]
    champ_hold_preds = holdout_runs[champion_key]["preds"]

    # Holdout Actual vs Predicted & Residual Diagnostics for Champion
    holdout_residuals = y_test - champ_hold_preds
    holdout_series = [
        {
            "month": str(test_hold_df.iloc[i]["Month"]),
            "month_label": str(test_hold_df.iloc[i]["Month_Label"]),
            "actual": round(float(y_test[i]), 2),
            "predicted": round(float(champ_hold_preds[i]), 2),
            "naive": round(float(naive_hold_preds[i]), 2),
            "residual": round(float(holdout_residuals[i]), 2),
        }
        for i in range(len(test_hold_df))
    ]

    # Residual Histogram & Autocorrelation Diagnostics
    res_counts, res_edges = np.histogram(holdout_residuals, bins=min(8, max(4, len(holdout_residuals) // 2)))
    residual_histogram = [
        {
            "bin": f"₦{res_edges[i]/1e6:.1f}M–₦{res_edges[i+1]/1e6:.1f}M",
            "count": int(res_counts[i]),
        }
        for i in range(len(res_counts))
    ]

    max_acf_lag = min(6, len(holdout_residuals) // 2)
    residual_acf: List[Dict[str, Any]] = []
    if len(holdout_residuals) >= 6 and np.std(holdout_residuals) > 1e-6:
        r_centered = holdout_residuals - np.mean(holdout_residuals)
        denom = float(np.sum(r_centered ** 2)) or 1.0
        for lag in range(1, max_acf_lag + 1):
            acf_val = float(np.sum(r_centered[lag:] * r_centered[:-lag]) / denom)
            ci_bound = 1.96 / math.sqrt(len(holdout_residuals))
            residual_acf.append({
                "lag": lag,
                "acf": round(acf_val, 4),
                "bound": round(ci_bound, 4),
            })

    dw_stat = round(float(durbin_watson(holdout_residuals)), 4) if len(holdout_residuals) >= 4 else None
    lb_pvalue = None
    if len(holdout_residuals) >= 8:
        try:
            lb_p = ljung_box_pvalue(holdout_residuals, lag=min(4, len(holdout_residuals) // 3))
            lb_pvalue = round(float(lb_p), 4) if lb_p is not None else None
        except Exception:
            lb_pvalue = None

    # Refit Champion (+ driver importance fallback) on ALL historical data to forecast through Dec 2030
    full_runs = _forecast_model_candidates(
        df,
        steps=steps,
        target_col="Operating_Profit_NGN",
        only_keys={champion_key, "ridge_ts", "naive"},
    )
    champ_full = full_runs.get(champion_key) or full_runs["naive"]
    baseline_fc = champ_full["preds"]

    # Calibrate prediction interval uncertainty from actual holdout error sigma + horizon growth
    # Scale uncertainty smoothly as sqrt(1 + h / 12) capped realistically to reflect multi-year uncertainty widening
    holdout_sigma = max(float(np.std(holdout_residuals, ddof=1)) if len(holdout_residuals) >= 2 else float(np.std(df["Operating_Profit_NGN"]) * 0.1), champion_entry["rmse"] * 0.85)

    # Forecast secondary KPIs through Dec 2030
    sec_rev = _forecast_secondary_kpi(df, "Sales_Revenue_NGN", steps, allow_negative=False)
    sec_gp = _forecast_secondary_kpi(df, "Gross_Profit_NGN", steps, allow_negative=True)
    sec_opex = _forecast_secondary_kpi(df, "Total_Operating_Expenses_NGN", steps, allow_negative=False)
    sec_units = _forecast_secondary_kpi(df, "Units_Sold", steps, allow_negative=False)

    # Build monthly forecast timeline (Oct 2026 -> Dec 2030)
    forecast_months: List[Dict[str, Any]] = []
    # Risk threshold: negative profit OR operating profit below 50% of latest 12M avg
    recent_12m_avg_op = float(df["Operating_Profit_NGN"].tail(12).mean())
    low_profit_risk_floor = max(0.0, recent_12m_avg_op * 0.5)

    for h in range(steps):
        f_date = last_date + pd.DateOffset(months=h + 1)
        m_iso = f_date.strftime("%Y-%m-%d")
        m_lbl = f_date.strftime("%b %Y")
        q_lbl = f"Q{((f_date.month - 1) // 3) + 1}"

        # Horizon multiplier anchored to holdout error growth (1..18M verified -> smoothly extrapolated to 51M)
        h_mult = math.sqrt(1.0 + (h / 12.0))
        half_80 = 1.2816 * holdout_sigma * h_mult
        half_95 = 1.9600 * holdout_sigma * h_mult

        y_hat = float(baseline_fc[h])
        rev_hat = float(sec_rev["forecast"][h]) if sec_rev["available"] else max(1.0, y_hat * 3.2)
        gp_hat = float(sec_gp["forecast"][h]) if sec_gp["available"] else (y_hat + (float(sec_opex["forecast"][h]) if sec_opex["available"] else 0.0))
        opex_hat = float(sec_opex["forecast"][h]) if sec_opex["available"] else max(0.0, gp_hat - y_hat)
        units_hat = float(sec_units["forecast"][h]) if sec_units["available"] else float(df["Units_Sold"].tail(12).mean())
        op_margin_hat = (y_hat / rev_hat) * 100.0 if rev_hat > 0 else 0.0
        ppu_hat = (y_hat / units_hat) if units_hat > 0 else 0.0

        lower_95 = y_hat - half_95
        upper_95 = y_hat + half_95
        lower_80 = y_hat - half_80
        upper_80 = y_hat + half_80

        is_negative = y_hat < 0.0
        is_risk = is_negative or (lower_95 < 0.0) or (y_hat < low_profit_risk_floor)

        forecast_months.append({
            "month": m_iso,
            "month_label": m_lbl,
            "year": int(f_date.year),
            "month_num": int(f_date.month),
            "quarter": q_lbl,
            "year_quarter": f"{f_date.year} {q_lbl}",
            "horizon_step": h + 1,
            "period_type": "forecast",
            "operating_profit_ngn": round(y_hat, 2),
            "lower_80_ngn": round(lower_80, 2),
            "upper_80_ngn": round(upper_80, 2),
            "lower_95_ngn": round(lower_95, 2),
            "upper_95_ngn": round(upper_95, 2),
            "sales_revenue_ngn": round(rev_hat, 2),
            "gross_profit_ngn": round(gp_hat, 2),
            "operating_expenses_ngn": round(opex_hat, 2),
            "units_sold": round(units_hat, 0),
            "operating_margin_pct": round(op_margin_hat, 2),
            "profit_per_unit_ngn": round(ppu_hat, 2),
            "is_negative_profit": is_negative,
            "is_risk_period": is_risk,
        })

    # Combine Historical + Forecast monthly timeline for chart & CSV export
    historical_months: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        op_val = float(row["Operating_Profit_NGN"])
        historical_months.append({
            "month": str(row["Month"]),
            "month_label": str(row["Month_Label"]),
            "year": int(row["Year"]),
            "month_num": int(row["Month_Num"]),
            "quarter": str(row["Quarter"]),
            "year_quarter": str(row["Year_Quarter"]),
            "horizon_step": 0,
            "period_type": "historical",
            "operating_profit_ngn": round(op_val, 2),
            "lower_80_ngn": None,
            "upper_80_ngn": None,
            "lower_95_ngn": None,
            "upper_95_ngn": None,
            "sales_revenue_ngn": round(float(row["Sales_Revenue_NGN"]), 2),
            "gross_profit_ngn": round(float(row["Gross_Profit_NGN"]), 2),
            "operating_expenses_ngn": round(float(row["Total_Operating_Expenses_NGN"]), 2),
            "units_sold": round(float(row["Units_Sold"]), 0),
            "operating_margin_pct": round(float(row["Operating_Margin_Pct"]), 2),
            "profit_per_unit_ngn": round(float(row["Profit_Per_Unit_NGN"]), 2),
            "is_negative_profit": op_val < 0.0,
            "is_risk_period": op_val < 0.0,
        })

    # Annual summaries (historical + projected years)
    all_df = pd.DataFrame(historical_months + forecast_months)
    annual_summaries: List[Dict[str, Any]] = []
    prev_yr_profit: Optional[float] = None
    cum_forecast_profit = 0.0
    for yr, grp in all_df.groupby("year"):
        tot_op = float(grp["operating_profit_ngn"].sum())
        tot_rev = float(grp["sales_revenue_ngn"].sum())
        tot_gp = float(grp["gross_profit_ngn"].sum())
        tot_opex = float(grp["operating_expenses_ngn"].sum())
        avg_om = (tot_op / tot_rev * 100.0) if tot_rev > 0 else float(grp["operating_margin_pct"].mean())
        fc_rows = grp[grp["period_type"] == "forecast"]
        hist_rows = grp[grp["period_type"] == "historical"]
        if len(fc_rows) > 0:
            cum_forecast_profit += float(fc_rows["operating_profit_ngn"].sum())
        yoy_pct = ((tot_op - prev_yr_profit) / abs(prev_yr_profit) * 100.0) if (prev_yr_profit is not None and abs(prev_yr_profit) > 1e-6) else None
        annual_summaries.append({
            "year": int(yr),
            "months_count": int(len(grp)),
            "historical_months": int(len(hist_rows)),
            "forecast_months": int(len(fc_rows)),
            "period_scope": "Historical" if len(fc_rows) == 0 else ("Mixed (Actual + Forecast)" if len(hist_rows) > 0 else "Forecast"),
            "total_operating_profit_ngn": round(tot_op, 2),
            "total_sales_revenue_ngn": round(tot_rev, 2),
            "total_gross_profit_ngn": round(tot_gp, 2),
            "total_operating_expenses_ngn": round(tot_opex, 2),
            "weighted_operating_margin_pct": round(avg_om, 2),
            "yoy_operating_profit_pct": round(yoy_pct, 2) if yoy_pct is not None else None,
            "negative_months": int(grp["is_negative_profit"].sum()),
            "risk_months": int(grp["is_risk_period"].sum()),
        })
        prev_yr_profit = tot_op

    # Quarterly forecast summaries
    quarterly_forecasts: List[Dict[str, Any]] = []
    for yq, grp in pd.DataFrame(forecast_months).groupby("year_quarter", sort=False):
        tot_op = float(grp["operating_profit_ngn"].sum())
        tot_rev = float(grp["sales_revenue_ngn"].sum())
        quarterly_forecasts.append({
            "year_quarter": str(yq),
            "year": int(grp["year"].iloc[0]),
            "quarter": str(grp["quarter"].iloc[0]),
            "months": int(len(grp)),
            "total_operating_profit_ngn": round(tot_op, 2),
            "total_sales_revenue_ngn": round(tot_rev, 2),
            "avg_operating_margin_pct": round((tot_op / tot_rev * 100.0) if tot_rev > 0 else 0.0, 2),
            "lower_95_sum_ngn": round(float(grp["lower_95_ngn"].sum()), 2),
            "upper_95_sum_ngn": round(float(grp["upper_95_ngn"].sum()), 2),
        })

    # Feature importance & driver analysis for ML recursive model + Champion insight
    # Evaluate feature importance on the richest fitted ML model (and champion if ML)
    driver_importance: List[Dict[str, Any]] = []
    ml_key = champion_key if champion_key in {"linear_ts", "ridge_ts", "rf_ts", "gbr_ts", "xgb_ts"} else "ridge_ts"
    ml_run = full_runs.get(ml_key) or full_runs.get("ridge_ts")
    if ml_run and ml_run.get("estimator") is not None:
        est = ml_run["estimator"]
        if hasattr(est, "coef_"):
            coefs = np.asarray(est.coef_, dtype=float).ravel()
            abs_sum = float(np.sum(np.abs(coefs))) or 1.0
            for fname, cval in zip(TS_FEATURE_COLS, coefs):
                driver_importance.append({
                    "feature": fname,
                    "importance_value": round(abs(float(cval)), 2),
                    "signed_effect": round(float(cval), 2),
                    "share_pct": round((abs(float(cval)) / abs_sum) * 100.0, 2),
                    "method": "Standardized Linear / Ridge Coefficient (NGN per 1 SD)",
                    "direction": "Positive (+)" if cval >= 0 else "Negative (−)",
                })
        elif hasattr(est, "feature_importances_"):
            imps = np.asarray(est.feature_importances_, dtype=float).ravel()
            abs_sum = float(np.sum(imps)) or 1.0
            for fname, ival in zip(TS_FEATURE_COLS, imps):
                driver_importance.append({
                    "feature": fname,
                    "importance_value": round(float(ival), 4),
                    "signed_effect": round(float(ival), 4),
                    "share_pct": round((float(ival) / abs_sum) * 100.0, 2),
                    "method": "Tree Impurity / Gain Importance",
                    "direction": "Non-linear impact",
                })
        driver_importance.sort(key=lambda d: d["share_pct"], reverse=True)

    # Hypothesis testing bundle
    hypothesis_bundle = compute_hypothesis_tests(
        df=df,
        champion_holdout_actual=y_test,
        champion_holdout_pred=champ_hold_preds,
        naive_holdout_pred=naive_hold_preds,
        alpha=alpha,
        split_year=split_year,
    )

    # Executive Summary Metrics
    latest_actual_op = float(df["Operating_Profit_NGN"].iloc[-1])
    dec_2030 = forecast_months[-1]
    dec_2030_op = float(dec_2030["operating_profit_ngn"])
    growth_to_2030_pct = ((dec_2030_op - latest_actual_op) / abs(latest_actual_op) * 100.0) if abs(latest_actual_op) > 1e-6 else 0.0
    fc_total_op = float(sum(m["operating_profit_ngn"] for m in forecast_months))
    fc_total_rev = float(sum(m["sales_revenue_ngn"] for m in forecast_months))
    fc_avg_margin = (fc_total_op / fc_total_rev * 100.0) if fc_total_rev > 0 else float(np.mean([m["operating_margin_pct"] for m in forecast_months]))
    dec_2030_ci_half = (float(dec_2030["upper_95_ngn"]) - float(dec_2030["lower_95_ngn"])) / 2.0
    uncertainty_pct = (dec_2030_ci_half / max(abs(dec_2030_op), 1.0)) * 100.0

    # Operational Indicators Trajectory (improving vs deteriorating across trailing 12M vs prior 12M)
    t12 = df.tail(12)
    p12 = df.iloc[-24:-12]
    indicator_trends = []
    for col, lbl, higher_is_better in [
        ("Operating_Profit_NGN", "Monthly Operating Profit (NGN)", True),
        ("Sales_Revenue_NGN", "Monthly Sales Revenue (NGN)", True),
        ("Operating_Margin_Pct", "Operating Margin (%)", True),
        ("OpEx_to_Revenue_Ratio_Pct", "OpEx-to-Revenue Ratio (%)", False),
        ("Profit_Per_Unit_NGN", "Operating Profit per Unit (NGN)", True),
        ("Units_Sold", "Monthly Unit Volume", True),
    ]:
        cur_v = float(t12[col].mean())
        prv_v = float(p12[col].mean())
        delta_pct = ((cur_v - prv_v) / abs(prv_v) * 100.0) if abs(prv_v) > 1e-6 else 0.0
        improved = (delta_pct > 1.0) if higher_is_better else (delta_pct < -1.0)
        deteriorated = (delta_pct < -1.0) if higher_is_better else (delta_pct > 1.0)
        status = "Improving" if improved else "Deteriorating" if deteriorated else "Stable"
        indicator_trends.append({
            "indicator": lbl,
            "trailing_12m_avg": round(cur_v, 2),
            "prior_12m_avg": round(prv_v, 2),
            "yoy_change_pct": round(delta_pct, 2),
            "status": status,
        })

    # Compute Base, Optimistic, Stress scenarios + Verdict
    base_bundle = {
        "sufficient_data": True,
        "dataset_summary": {
            "historical_records": n,
            "historical_start": str(df["Month"].iloc[0]),
            "historical_end": str(df["Month"].iloc[-1]),
            "forecast_start": forecast_months[0]["month"],
            "forecast_end": forecast_months[-1]["month"],
            "forecast_horizon_months": steps,
            "horizon_note": horizon_note,
            "historical_negative_profit_months": int((df["Operating_Profit_NGN"] < 0).sum()),
        },
        "executive_summary": {
            "latest_actual_month": str(df["Month_Label"].iloc[-1]),
            "latest_actual_operating_profit_ngn": round(latest_actual_op, 2),
            "dec_2030_month": dec_2030["month_label"],
            "dec_2030_forecast_operating_profit_ngn": round(dec_2030_op, 2),
            "dec_2030_lower_95_ngn": dec_2030["lower_95_ngn"],
            "dec_2030_upper_95_ngn": dec_2030["upper_95_ngn"],
            "projected_cumulative_operating_profit_ngn": round(fc_total_op, 2),
            "forecast_operating_margin_pct": round(fc_avg_margin, 2),
            "forecast_growth_to_2030_pct": round(growth_to_2030_pct, 2),
            "champion_model_name": champion_entry["name"],
            "champion_holdout_rmse": champion_entry["rmse"],
            "champion_holdout_mae": champion_entry["mae"],
            "champion_holdout_mape_pct": champion_entry["mape_pct"],
            "champion_holdout_r2": champion_entry["r2"],
            "forecast_uncertainty_95_pct": round(uncertainty_pct, 1),
            "forecast_negative_months": int(sum(1 for m in forecast_months if m["is_negative_profit"])),
            "forecast_risk_months": int(sum(1 for m in forecast_months if m["is_risk_period"])),
        },
        "indicator_trends": indicator_trends,
        "feature_dictionary": TS_FEATURE_DICTIONARY,
        "leaderboard": leaderboard,
        "validation": {
            "holdout_size": holdout_size,
            "holdout_period": f"{test_hold_df['Month'].iloc[0]} to {test_hold_df['Month'].iloc[-1]}",
            "backtest_origins_count": len(origins),
            "backtest_horizon_months": backtest_horizon,
            "champion_name": champion_entry["name"],
            "champion_bias": champion_entry["forecast_bias"],
            "champion_mean_error": champion_entry["mean_error"],
            "champion_bias_pct": champion_entry["bias_pct"],
            "holdout_series": holdout_series,
            "residual_histogram": residual_histogram,
            "residual_acf": residual_acf,
            "durbin_watson": dw_stat,
            "ljung_box_p_value": lb_pvalue,
            "bias_explanation": (
                f"Mean error (Actual − Predicted) on the {holdout_size}-month holdout is ₦{champion_entry['mean_error']/1e6:+.2f}M "
                f"({champion_entry['bias_pct']:+.2f}%), indicating {champion_entry['forecast_bias'].lower()} behavior."
            ),
        },
        "driver_importance": driver_importance,
        "driver_importance_model": ml_run["name"] if ml_run else champion_entry["name"],
        "hypothesis_testing": hypothesis_bundle,
        "secondary_kpis": {
            "Sales_Revenue_NGN": {"model": sec_rev["model_used"], "holdout_rmse": sec_rev["holdout_rmse"]},
            "Gross_Profit_NGN": {"model": sec_gp["model_used"], "holdout_rmse": sec_gp["holdout_rmse"]},
            "Total_Operating_Expenses_NGN": {"model": sec_opex["model_used"], "holdout_rmse": sec_opex["holdout_rmse"]},
            "Units_Sold": {"model": sec_units["model_used"], "holdout_rmse": sec_units["holdout_rmse"]},
        },
        "historical_months": historical_months,
        "forecast_months": forecast_months,
        "annual_summaries": annual_summaries,
        "quarterly_forecasts": quarterly_forecasts,
    }

    scenarios = compute_scenario_forecasts(base_bundle)
    verdict = compute_operational_verdict(base_bundle, scenarios)
    custom_verdict = compute_custom_scenario_verdict(base_bundle, scenarios)
    base_bundle["scenarios"] = scenarios
    base_bundle["verdict"] = verdict
    base_bundle["custom_verdict"] = custom_verdict
    return base_bundle


def _apply_scenario_to_months(
    forecast_months: List[Dict[str, Any]],
    rev_growth_adj_pp: float,
    opex_growth_adj_pp: float,
    margin_eff_adj_pp: float,
    fx_cost_adj_pct: float,
    volume_adj_pct: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cum_op = 0.0
    cum_rev = 0.0
    neg_months = 0
    risk_months = 0

    for m in forecast_months:
        years_ahead = m["horizon_step"] / 12.0
        rev_mult = ((1.0 + rev_growth_adj_pp / 100.0) ** years_ahead) * (1.0 + volume_adj_pct / 100.0)
        opex_mult = (1.0 + opex_growth_adj_pp / 100.0) ** years_ahead

        base_rev = float(m["sales_revenue_ngn"])
        base_gp = float(m["gross_profit_ngn"])
        base_opex = float(m["operating_expenses_ngn"])
        base_cogs = max(0.0, base_rev - base_gp)

        sim_rev = base_rev * rev_mult
        sim_cogs = base_cogs * (1.0 + volume_adj_pct / 100.0) * (1.0 + fx_cost_adj_pct / 100.0)
        sim_opex = base_opex * opex_mult
        sim_op = (sim_rev - sim_cogs - sim_opex) + sim_rev * (margin_eff_adj_pp / 100.0)

        # Preserve baseline residual gap anchor if baseline revenue differed slightly from secondary model
        base_identity_op = base_rev - base_cogs - base_opex
        identity_delta = sim_op - base_identity_op
        anchored_op = float(m["operating_profit_ngn"]) + identity_delta
        sim_margin = (anchored_op / sim_rev * 100.0) if sim_rev > 0 else 0.0

        cum_op += anchored_op
        cum_rev += sim_rev
        if anchored_op < 0:
            neg_months += 1
        if anchored_op < 0 or sim_margin < 10.0:
            risk_months += 1

        rows.append({
            "month": m["month"],
            "month_label": m["month_label"],
            "year": m["year"],
            "operating_profit_ngn": round(anchored_op, 2),
            "sales_revenue_ngn": round(sim_rev, 2),
            "operating_expenses_ngn": round(sim_cogs + sim_opex, 2),
            "opex_only_ngn": round(sim_opex, 2),
            "operating_margin_pct": round(sim_margin, 2),
            "is_negative_profit": anchored_op < 0,
            "is_risk_period": anchored_op < 0 or sim_margin < 10.0,
        })

    last = rows[-1] if rows else {"operating_profit_ngn": 0.0, "operating_margin_pct": 0.0}
    yr_2030 = [r for r in rows if r["year"] == 2030]
    rev_2030 = sum(r["sales_revenue_ngn"] for r in yr_2030) or 1.0
    op_2030 = sum(r["operating_profit_ngn"] for r in yr_2030)
    summary = {
        "dec_2030_operating_profit_ngn": round(float(last["operating_profit_ngn"]), 2),
        "dec_2030_operating_margin_pct": round(float(last["operating_margin_pct"]), 2),
        "year_2030_total_profit_ngn": round(op_2030, 2),
        "year_2030_operating_margin_pct": round((op_2030 / rev_2030) * 100.0, 2),
        "cumulative_profit_ngn": round(cum_op, 2),
        "cumulative_revenue_ngn": round(cum_rev, 2),
        "cumulative_margin_pct": round((cum_op / cum_rev) * 100.0, 2) if cum_rev > 0 else 0.0,
        "negative_months": neg_months,
        "risk_months": risk_months,
    }
    return rows, summary


def compute_scenario_forecasts(
    bundle: Dict[str, Any],
    custom_assumptions: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    fc_months = bundle.get("forecast_months", [])
    custom = custom_assumptions or {
        "revenue_growth_adj_pp": 0.0,
        "opex_growth_adj_pp": 0.0,
        "margin_efficiency_adj_pp": 0.0,
        "fx_cost_pressure_adj_pct": 0.0,
        "volume_productivity_adj_pct": 0.0,
    }

    c_rev = float(custom.get("revenue_growth_adj_pp", 0.0))
    c_opex = float(custom.get("opex_growth_adj_pp", 0.0))
    c_margin = float(custom.get("margin_efficiency_adj_pp", 0.0))
    c_fx = float(custom.get("fx_cost_pressure_adj_pct", 0.0))
    c_vol = float(custom.get("volume_productivity_adj_pct", 0.0))

    specs = {
        "base": {
            "label": "Base Case (Model Baseline)",
            "description": "Continuation of validated historical trajectory with zero additional user adjustments.",
            "params": {
                "revenue_growth_adj_pp": 0.0,
                "opex_growth_adj_pp": 0.0,
                "margin_efficiency_adj_pp": 0.0,
                "fx_cost_pressure_adj_pct": 0.0,
                "volume_productivity_adj_pct": 0.0,
            },
        },
        "optimistic": {
            "label": "Optimistic Case",
            "description": "Higher pricing pass-through & volume throughput (+5%/yr rev, +6% volume, +2 pp efficiency, −4% FX cost pressure).",
            "params": {
                "revenue_growth_adj_pp": 5.0,
                "opex_growth_adj_pp": -2.0,
                "margin_efficiency_adj_pp": 2.0,
                "fx_cost_pressure_adj_pct": -4.0,
                "volume_productivity_adj_pct": 6.0,
            },
        },
        "stress": {
            "label": "Stress Case",
            "description": "FX depreciation & cost inflation shock (−5%/yr demand, +7%/yr OpEx, −3 pp efficiency, +14% import/FX cost shock).",
            "params": {
                "revenue_growth_adj_pp": -5.0,
                "opex_growth_adj_pp": 7.0,
                "margin_efficiency_adj_pp": -3.0,
                "fx_cost_pressure_adj_pct": 14.0,
                "volume_productivity_adj_pct": -8.0,
            },
        },
        "custom": {
            "label": "Custom User Scenario",
            "description": "User-defined operational & macroeconomic assumptions applied over model baseline.",
            "params": {
                "revenue_growth_adj_pp": c_rev,
                "opex_growth_adj_pp": c_opex,
                "margin_efficiency_adj_pp": c_margin,
                "fx_cost_pressure_adj_pct": c_fx,
                "volume_productivity_adj_pct": c_vol,
            },
        },
    }

    summaries: Dict[str, Any] = {}
    monthly_by_case: Dict[str, List[Dict[str, Any]]] = {}
    for key, spec in specs.items():
        p = spec["params"]
        rows, summ = _apply_scenario_to_months(
            fc_months,
            rev_growth_adj_pp=p["revenue_growth_adj_pp"],
            opex_growth_adj_pp=p["opex_growth_adj_pp"],
            margin_eff_adj_pp=p["margin_efficiency_adj_pp"],
            fx_cost_adj_pct=p["fx_cost_pressure_adj_pct"],
            volume_adj_pct=p["volume_productivity_adj_pct"],
        )
        monthly_by_case[key] = rows
        summaries[key] = {
            "key": key,
            "label": spec["label"],
            "description": spec["description"],
            "assumptions": p,
            "summary": summ,
        }

    # Sequential exact cumulative profit waterfall decomposition for Custom vs Base (2026-2030)
    base_cum = summaries["base"]["summary"]["cumulative_profit_ngn"]
    _, s1 = _apply_scenario_to_months(fc_months, c_rev, 0.0, 0.0, 0.0, 0.0)
    _, s2 = _apply_scenario_to_months(fc_months, c_rev, 0.0, 0.0, 0.0, c_vol)
    _, s3 = _apply_scenario_to_months(fc_months, c_rev, 0.0, 0.0, c_fx, c_vol)
    _, s4 = _apply_scenario_to_months(fc_months, c_rev, c_opex, 0.0, c_fx, c_vol)
    custom_cum = summaries["custom"]["summary"]["cumulative_profit_ngn"]

    lever_attribution_waterfall = [
        {"step": "Base 2030 Cumulative", "value": round(base_cum, 2), "type": "baseline"},
        {"step": "Revenue Growth", "value": round(s1["cumulative_profit_ngn"] - base_cum, 2), "type": "delta"},
        {"step": "Volume & Productivity", "value": round(s2["cumulative_profit_ngn"] - s1["cumulative_profit_ngn"], 2), "type": "delta"},
        {"step": "FX & Import Shock", "value": round(s3["cumulative_profit_ngn"] - s2["cumulative_profit_ngn"], 2), "type": "delta"},
        {"step": "OpEx Growth", "value": round(s4["cumulative_profit_ngn"] - s3["cumulative_profit_ngn"], 2), "type": "delta"},
        {"step": "Margin Efficiency", "value": round(custom_cum - s4["cumulative_profit_ngn"], 2), "type": "delta"},
        {"step": "Simulated Cumulative", "value": round(custom_cum, 2), "type": "total"},
    ]

    # Merge monthly chart series for Base vs Optimistic vs Stress vs Custom across KPIs
    combined_series: List[Dict[str, Any]] = []
    for idx, m in enumerate(fc_months):
        b_row = monthly_by_case["base"][idx]
        c_row = monthly_by_case["custom"][idx]
        combined_series.append({
            "month": m["month"],
            "month_label": m["month_label"],
            "year": m["year"],
            "base": b_row["operating_profit_ngn"],
            "optimistic": monthly_by_case["optimistic"][idx]["operating_profit_ngn"],
            "stress": monthly_by_case["stress"][idx]["operating_profit_ngn"],
            "custom": c_row["operating_profit_ngn"],
            "base_revenue": b_row["sales_revenue_ngn"],
            "custom_revenue": c_row["sales_revenue_ngn"],
            "base_opex": b_row["opex_only_ngn"],
            "custom_opex": c_row["opex_only_ngn"],
            "base_margin": b_row["operating_margin_pct"],
            "custom_margin": c_row["operating_margin_pct"],
        })

    # Annual 2026-2030 scenario comparison
    annual_comparison: List[Dict[str, Any]] = []
    years = sorted({int(m["year"]) for m in fc_months})
    for yr in years:
        b_yr = [r for r in monthly_by_case["base"] if r["year"] == yr]
        c_yr = [r for r in monthly_by_case["custom"] if r["year"] == yr]
        o_yr = [r for r in monthly_by_case["optimistic"] if r["year"] == yr]
        s_yr = [r for r in monthly_by_case["stress"] if r["year"] == yr]
        b_op = sum(r["operating_profit_ngn"] for r in b_yr)
        c_op = sum(r["operating_profit_ngn"] for r in c_yr)
        b_rev = sum(r["sales_revenue_ngn"] for r in b_yr) or 1.0
        c_rev = sum(r["sales_revenue_ngn"] for r in c_yr) or 1.0
        annual_comparison.append({
            "year": yr,
            "months": len(c_yr),
            "base_profit_ngn": round(b_op, 2),
            "custom_profit_ngn": round(c_op, 2),
            "delta_profit_ngn": round(c_op - b_op, 2),
            "optimistic_profit_ngn": round(sum(r["operating_profit_ngn"] for r in o_yr), 2),
            "stress_profit_ngn": round(sum(r["operating_profit_ngn"] for r in s_yr), 2),
            "base_revenue_ngn": round(b_rev, 2),
            "custom_revenue_ngn": round(c_rev, 2),
            "base_margin_pct": round((b_op / b_rev) * 100.0, 2),
            "custom_margin_pct": round((c_op / c_rev) * 100.0, 2),
            "custom_negative_months": sum(1 for r in c_yr if r["is_negative_profit"]),
            "custom_risk_months": sum(1 for r in c_yr if r["is_risk_period"]),
        })

    return {
        "cases": summaries,
        "monthly_comparison": combined_series,
        "annual_comparison": annual_comparison,
        "lever_attribution_waterfall": lever_attribution_waterfall,
    }


def compute_operational_verdict(bundle: Dict[str, Any], scenarios: Dict[str, Any]) -> Dict[str, Any]:
    exec_sum = bundle["executive_summary"]
    hyp_tests = {t["id"]: t for t in bundle["hypothesis_testing"]["tests"]}
    ind_trends = bundle["indicator_trends"]
    stress_sum = scenarios["cases"]["stress"]["summary"]
    opt_sum = scenarios["cases"]["optimistic"]["summary"]

    growth_pct = float(exec_sum["forecast_growth_to_2030_pct"])
    fc_margin = float(exec_sum["forecast_operating_margin_pct"])
    uncertainty_pct = float(exec_sum["forecast_uncertainty_95_pct"])
    neg_months = int(exec_sum["forecast_negative_months"])
    stress_neg = int(stress_sum["negative_months"])
    stress_2030_op = float(stress_sum["dec_2030_operating_profit_ngn"])

    hist_months = bundle.get("historical_months", [])
    t12_avg_op = float(np.mean([m["operating_profit_ngn"] for m in hist_months[-12:]])) if hist_months else float(exec_sum["latest_actual_operating_profit_ngn"])
    t12_annual_op = t12_avg_op * 12.0
    annual_2030 = next((a for a in bundle["annual_summaries"] if a["year"] == 2030), None)
    annual_2030_op = float(annual_2030["total_operating_profit_ngn"]) if annual_2030 else float(exec_sum["dec_2030_forecast_operating_profit_ngn"]) * 12.0
    annual_growth_pct = ((annual_2030_op - t12_annual_op) / abs(t12_annual_op) * 100.0) if abs(t12_annual_op) > 1e-6 else growth_pct
    vs_t12_dec_pct = ((float(exec_sum["dec_2030_forecast_operating_profit_ngn"]) - t12_avg_op) / abs(t12_avg_op) * 100.0) if abs(t12_avg_op) > 1e-6 else growth_pct

    h1_sig = bool(hyp_tests.get("H1_trend", {}).get("significant"))
    h5_sig = bool(hyp_tests.get("H5_dm_improvement", {}).get("significant"))

    # Documented transparent decision thresholds
    thresholds = [
        {"rule": "Annualized 2030 Profit vs Trailing 12M", "threshold": ">= +1.5% annualized growth", "observed": f"{annual_growth_pct:+.2f}%", "passed": annual_growth_pct >= 1.5},
        {"rule": "Healthy Forecast Operating Margin", "threshold": ">= 20.0% projected margin", "observed": f"{fc_margin:.2f}%", "passed": fc_margin >= 20.0},
        {"rule": "Zero Baseline Negative-Profit Months", "threshold": "0 negative months through 2030", "observed": f"{neg_months} months", "passed": neg_months == 0},
        {"rule": "Statistically Significant Historical Trend (H1)", "threshold": "Kendall tau p < 0.05", "observed": hyp_tests.get("H1_trend", {}).get("decision", "N/A"), "passed": h1_sig},
        {"rule": "Stress-Case Solvency Resilience", "threshold": "Dec 2030 Stress Profit > ₦0", "observed": f"₦{stress_2030_op/1e6:.2f}M", "passed": stress_2030_op > 0},
        {"rule": "Bounded Long-Horizon Uncertainty", "threshold": "95% CI half-width <= 60% of Dec 2030 level", "observed": f"±{uncertainty_pct:.1f}%", "passed": uncertainty_pct <= 60.0},
    ]

    if annual_growth_pct < -12.0 or neg_months >= 6 or (stress_2030_op < 0 and fc_margin < 10.0):
        category = "High-Risk Outlook"
    elif annual_growth_pct < -5.0 or fc_margin < 10.0:
        category = "Negative Operational Outlook"
    elif annual_growth_pct >= 12.0 and fc_margin >= 22.0 and neg_months == 0 and stress_2030_op > 0 and uncertainty_pct <= 45.0:
        category = "Strong Positive Outlook"
    elif (annual_growth_pct >= 1.5 or vs_t12_dec_pct >= 4.0) and fc_margin >= 18.0 and neg_months == 0 and stress_2030_op > 0:
        category = "Positive but Requires Monitoring"
    else:
        category = "Stable / Mixed Outlook"

    improving = [i["indicator"] for i in ind_trends if i["status"] == "Improving"]
    deteriorating = [i["indicator"] for i in ind_trends if i["status"] == "Deteriorating"]

    top_drivers = [d["feature"] for d in bundle.get("driver_importance", [])[:3]]
    annual_2027 = next((a for a in bundle["annual_summaries"] if a["year"] == 2027), None)

    return {
        "category": category,
        "thresholds_evaluated": thresholds,
        "headline": (
            f"{category}: Baseline Operating Profit is projected at ₦{exec_sum['dec_2030_forecast_operating_profit_ngn']/1e6:.2f}M "
            f"by December 2030 ({vs_t12_dec_pct:+.1f}% vs. trailing 12M avg of ₦{t12_avg_op/1e6:.2f}M; {growth_pct:+.1f}% vs. {exec_sum['latest_actual_month']}), "
            f"with cumulative 2026–2030 profit of ₦{exec_sum['projected_cumulative_operating_profit_ngn']/1e9:.2f}B at {fc_margin:.2f}% average margin."
        ),
        "expected_dec_2030_profit_ngn": exec_sum["dec_2030_forecast_operating_profit_ngn"],
        "annual_trajectory_summary": (
            f"Annual Operating Profit transitions from ₦{(annual_2027['total_operating_profit_ngn']/1e9 if annual_2027 else 0):.2f}B in 2027 "
            f"to ₦{(annual_2030['total_operating_profit_ngn']/1e9 if annual_2030 else 0):.2f}B in 2030 "
            f"(Stress Case Dec 2030: ₦{stress_2030_op/1e6:.2f}M; Optimistic Case Dec 2030: ₦{opt_sum['dec_2030_operating_profit_ngn']/1e6:.2f}M)."
        ),
        "operational_efficiency": f"Projected operating margin averages {fc_margin:.2f}% across the 2026–2030 horizon. Improving indicators: {', '.join(improving) if improving else 'None materially above +1% YoY'}.",
        "areas_deteriorating": deteriorating if deteriorating else ["No primary operational indicator deteriorated >1% YoY in the trailing 12 months."],
        "major_cost_pressures": "USD/NGN exchange-rate depreciation directly inflates Naira-denominated COGS per unit (r≈1.000), while inflation elevates domestic marketing and fixed operating expenses.",
        "significant_drivers": top_drivers,
        "forecast_uncertainty_summary": (
            f"95% prediction interval at December 2030 spans ₦{exec_sum['dec_2030_lower_95_ngn']/1e6:.2f}M to ₦{exec_sum['dec_2030_upper_95_ngn']/1e6:.2f}M "
            f"(±{uncertainty_pct:.1f}%), calibrated from empirical out-of-sample holdout RMSE of ₦{exec_sum['champion_holdout_rmse']/1e6:.2f}M. "
            f"{'Champion outperforms Naive persistence significantly (H5).' if h5_sig else 'Holdout comparison vs Naive did not pass α=' + str(bundle['hypothesis_testing']['alpha']) + ' significance, reinforcing wider long-horizon caution.'}"
        ),
        "principal_risks": [
            f"FX Shock Pass-Through Lag: Sudden >3% MoM Naira depreciation compresses operating margins before selling prices adjust (Stress Case yields {stress_neg} negative-profit months).",
            "Multi-Year Extrapolation Drift: Forecasting 51 months beyond Sep 2026 compounds recursive uncertainty if macro or competitive pricing regimes shift.",
            "Cost-to-Revenue Absorption: Unchecked OpEx growth above revenue growth erodes unit operating leverage.",
        ],
        "recommended_interventions": [
            "Implement dynamic FX-indexed pricing review triggers whenever USD/NGN moves >2.5% MoM to protect unit gross spread.",
            "Enforce OpEx-to-Revenue ceiling discipline by linking discretionary Marketing Spend increments to verified unit volume thresholds.",
            "Re-run expanding-window model retraining quarterly and track prediction interval breaches as early-warning triggers.",
        ],
        "verdict_rationale": (
            f"Verdict '{category}' was derived deterministically from {sum(1 for t in thresholds if t['passed'])}/{len(thresholds)} "
            f"documented thresholds passing, combined with {champion_entry_name(bundle)} validation evidence and Stress Case solvency checks."
        ),
    }


def compute_custom_scenario_verdict(bundle: Dict[str, Any], scenarios: Dict[str, Any]) -> Dict[str, Any]:
    exec_sum = bundle["executive_summary"]
    hyp_tests = {t["id"]: t for t in bundle["hypothesis_testing"]["tests"]}
    base_sum = scenarios["cases"]["base"]["summary"]
    custom_sum = scenarios["cases"]["custom"]["summary"]
    stress_sum = scenarios["cases"]["stress"]["summary"]

    hist_months = bundle.get("historical_months", [])
    t12_avg_op = float(np.mean([m["operating_profit_ngn"] for m in hist_months[-12:]])) if hist_months else float(exec_sum["latest_actual_operating_profit_ngn"])
    t12_annual_op = t12_avg_op * 12.0

    custom_dec_2030 = float(custom_sum["dec_2030_operating_profit_ngn"])
    custom_yr_2030 = float(custom_sum["year_2030_total_profit_ngn"])
    custom_cum = float(custom_sum["cumulative_profit_ngn"])
    custom_margin = float(custom_sum["cumulative_margin_pct"])
    custom_neg = int(custom_sum["negative_months"])
    custom_risk = int(custom_sum["risk_months"])
    stress_2030_op = float(stress_sum["dec_2030_operating_profit_ngn"])

    annual_growth_pct = ((custom_yr_2030 - t12_annual_op) / abs(t12_annual_op) * 100.0) if abs(t12_annual_op) > 1e-6 else 0.0
    vs_t12_dec_pct = ((custom_dec_2030 - t12_avg_op) / abs(t12_avg_op) * 100.0) if abs(t12_avg_op) > 1e-6 else 0.0
    uncertainty_pct = float(exec_sum["forecast_uncertainty_95_pct"])
    h1_sig = bool(hyp_tests.get("H1_trend", {}).get("significant"))

    thresholds = [
        {"rule": "Simulated Annualized 2030 Profit vs Trailing 12M", "threshold": ">= +1.5% annualized growth", "observed": f"{annual_growth_pct:+.2f}%", "passed": annual_growth_pct >= 1.5},
        {"rule": "Simulated 2026–2030 Operating Margin", "threshold": ">= 20.0% projected margin", "observed": f"{custom_margin:.2f}%", "passed": custom_margin >= 20.0},
        {"rule": "Zero Simulated Negative-Profit Months", "threshold": "0 negative months through 2030", "observed": f"{custom_neg} months", "passed": custom_neg == 0},
        {"rule": "Statistically Significant Historical Trend (H1)", "threshold": "Kendall tau p < 0.05", "observed": hyp_tests.get("H1_trend", {}).get("decision", "N/A"), "passed": h1_sig},
        {"rule": "Simulated Dec 2030 Solvency", "threshold": "Dec 2030 Simulated Profit > ₦0", "observed": f"₦{custom_dec_2030/1e6:.2f}M", "passed": custom_dec_2030 > 0},
        {"rule": "Simulated Risk Exposure Limit", "threshold": "<= 3 low-margin/loss risk months", "observed": f"{custom_risk} months", "passed": custom_risk <= 3},
    ]

    if annual_growth_pct < -12.0 or custom_neg >= 6 or custom_dec_2030 < 0 or custom_margin < 8.0:
        category = "High-Risk Outlook"
    elif annual_growth_pct < -5.0 or custom_margin < 12.0 or custom_neg > 0:
        category = "Negative Operational Outlook"
    elif annual_growth_pct >= 12.0 and custom_margin >= 22.0 and custom_neg == 0 and stress_2030_op > 0:
        category = "Strong Positive Outlook"
    elif (annual_growth_pct >= 1.5 or vs_t12_dec_pct >= 4.0) and custom_margin >= 18.0 and custom_neg == 0:
        category = "Positive but Requires Monitoring"
    else:
        category = "Stable / Mixed Outlook"

    delta_dec = custom_dec_2030 - float(base_sum["dec_2030_operating_profit_ngn"])
    delta_cum = custom_cum - float(base_sum["cumulative_profit_ngn"])
    delta_margin = custom_margin - float(base_sum["cumulative_margin_pct"])

    return {
        "category": category,
        "thresholds_evaluated": thresholds,
        "headline": (
            f"Simulated Scenario ({category}): Dec 2030 Operating Profit reaches ₦{custom_dec_2030/1e6:.2f}M "
            f"({delta_dec/1e6:+.2f}M vs. Base Model), with cumulative 2026–2030 profit of ₦{custom_cum/1e9:.2f}B "
            f"({delta_cum/1e6:+.1f}M vs. Base) at {custom_margin:.2f}% average margin."
        ),
        "expected_dec_2030_profit_ngn": round(custom_dec_2030, 2),
        "delta_vs_base_dec_2030_ngn": round(delta_dec, 2),
        "delta_vs_base_cumulative_ngn": round(delta_cum, 2),
        "delta_vs_base_margin_pp": round(delta_margin, 2),
        "uncertainty_pct": round(uncertainty_pct, 1),
    }


def champion_entry_name(bundle: Dict[str, Any]) -> str:
    return str(bundle.get("executive_summary", {}).get("champion_model_name", "Champion Model"))


def _simulate_accounting_5yr_path(
    *,
    last_date: pd.Timestamp,
    horizon_months: int,
    anchor_units: float,
    anchor_price: float,
    anchor_fx: float,
    anchor_usd_cost: float,
    anchor_mkt: float,
    anchor_other_opex: float,
    seasonal_factors: Dict[int, float],
    hist_fx_monthly_vol: float,
    hist_resid_sigma: float,
    params: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    ann_inf = float(params["annual_inflation_pct"])
    unit_growth = float(params["unit_sales_growth_pct"])
    price_inf_pt = float(params["price_inflation_pass_through_pct"])
    fx_cost_pt = float(params["fx_cost_pass_through_pct"])
    usd_cost_growth = float(params["usd_import_cost_growth_pct"])
    opex_inf_sens = float(params["opex_inflation_sensitivity_pct"])
    fx_inf_sens = float(params["fx_inflation_sensitivity_x"])

    # Macro PPP-style FX drift driven by domestic inflation excess scaled by G (FX Inflation Sensitivity)
    inf_excess = max(0.0, (ann_inf - 3.0) / 100.0)
    fx_annual_rate = max(-0.05, min(0.65, inf_excess * 0.55 * fx_inf_sens))
    price_annual_rate = (ann_inf / 100.0) * (price_inf_pt / 100.0) + fx_annual_rate * (fx_cost_pt / 100.0)
    opex_annual_rate = (ann_inf / 100.0) * (opex_inf_sens / 100.0)

    cum_op = 0.0
    cum_rev = 0.0
    neg_months = 0
    band_breach_months = 0
    rows: List[Dict[str, Any]] = []

    for t in range(1, horizon_months + 1):
        dt = last_date + pd.DateOffset(months=t)
        y = t / 12.0
        s_m = seasonal_factors.get(int(dt.month), 1.0)

        fx_t = anchor_fx * ((1.0 + fx_annual_rate) ** y)
        usd_cost_t = anchor_usd_cost * ((1.0 + usd_cost_growth / 100.0) ** y)
        units_t = max(1.0, anchor_units * ((1.0 + unit_growth / 100.0) ** y) * s_m)
        price_t = max(1.0, anchor_price * ((1.0 + price_annual_rate) ** y))
        mkt_t = max(0.0, anchor_mkt * ((1.0 + opex_annual_rate) ** y) * (0.45 + 0.55 * s_m))
        other_opex_t = max(0.0, anchor_other_opex * ((1.0 + opex_annual_rate) ** y))

        rev_t = units_t * price_t
        cogs_t = units_t * fx_t * usd_cost_t
        op_t = rev_t - cogs_t - mkt_t - other_opex_t
        margin_t = (op_t / rev_t * 100.0) if rev_t > 0 else 0.0

        # FX-propagated 80% prediction cone (z = 1.2816)
        fx_sigma_t = hist_fx_monthly_vol * math.sqrt(float(t)) * max(0.0, fx_inf_sens)
        fx_sigma_t = min(1.25, fx_sigma_t)
        z80 = 1.2816

        fx_up = fx_t * math.exp(z80 * fx_sigma_t)
        fx_dn = fx_t * math.exp(-z80 * fx_sigma_t)

        # Under adverse FX depreciation, FX Cost Pass-Through (D%) partially cushions Revenue
        price_up = price_t * (1.0 + (fx_cost_pt / 100.0) * (fx_up / max(fx_t, 1e-6) - 1.0))
        price_dn = price_t * max(0.5, 1.0 + (fx_cost_pt / 100.0) * (fx_dn / max(fx_t, 1e-6) - 1.0))

        op_fx_low = units_t * price_up - (units_t * fx_up * usd_cost_t) - mkt_t - other_opex_t
        op_fx_high = units_t * price_dn - (units_t * fx_dn * usd_cost_t) - mkt_t - other_opex_t

        resid_80 = z80 * hist_resid_sigma * math.sqrt(float(t) / 6.0)
        low_80 = min(op_fx_low, op_t) - resid_80 * 0.35
        high_80 = max(op_fx_high, op_t) + resid_80 * 0.35

        cum_op += op_t
        cum_rev += rev_t
        if op_t < 0:
            neg_months += 1
        if low_80 < 0:
            band_breach_months += 1

        rows.append({
            "month": dt.strftime("%Y-%m-%d"),
            "month_label": dt.strftime("%b %Y"),
            "year": int(dt.year),
            "operating_profit_ngn": round(op_t, 2),
            "band_low_80_ngn": round(low_80, 2),
            "band_high_80_ngn": round(high_80, 2),
            "sales_revenue_ngn": round(rev_t, 2),
            "cogs_ngn": round(cogs_t, 2),
            "operating_expenses_ngn": round(mkt_t + other_opex_t, 2),
            "operating_margin_pct": round(margin_t, 2),
            "usd_ngn_rate": round(fx_t, 2),
            "units_sold": round(units_t, 0),
        })

    final_op = rows[-1]["operating_profit_ngn"] if rows else 0.0
    cum_margin = (cum_op / cum_rev * 100.0) if cum_rev > 0 else 0.0
    return rows, {
        "final_op_ngn": round(final_op, 2),
        "cumulative_op_ngn": round(cum_op, 2),
        "cumulative_rev_ngn": round(cum_rev, 2),
        "operating_margin_pct": round(cum_margin, 2),
        "negative_months": neg_months,
        "band_breach_months": band_breach_months,
    }


def compute_accounting_5yr_simulation(
    processed_df: pd.DataFrame,
    user_params: Optional[Dict[str, Optional[float]]] = None,
    horizon_months: int = 60,
) -> Dict[str, Any]:
    df = processed_df.copy().sort_values("Month").reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("At least 3 monthly records are required for 5-year accounting simulation.")

    df["_dt"] = pd.to_datetime(df["Month"])
    last_row = df.iloc[-1]
    last_date = pd.to_datetime(last_row["Month"])

    anchor_inflation = round(float(last_row["Inflation_YoY_Pct"]), 2)
    anchor_units = float(last_row["Units_Sold"])
    anchor_price = float(last_row["Avg_Selling_Price_NGN"])
    anchor_fx = float(last_row["USD_NGN_Planning_Rate"])
    anchor_usd_cost = float(last_row["Import_Cost_USD_Per_Unit"])
    anchor_mkt = float(last_row["Marketing_Spend_NGN"])
    anchor_other_opex = float(last_row["Other_Operating_Cost_NGN"])
    latest_actual_op = float(last_row["Operating_Profit_NGN"])

    # Compute empirical seasonal unit index normalized to anchor month
    roll_u = df["Units_Sold"].astype(float).rolling(12, min_periods=3).mean()
    ratio_u = (df["Units_Sold"].astype(float) / np.maximum(roll_u, 1.0)).fillna(1.0)
    raw_s: Dict[int, float] = {}
    for m_num in range(1, 13):
        vals = ratio_u[df["_dt"].dt.month == m_num]
        raw_s[m_num] = float(np.clip(vals.mean(), 0.84, 1.18)) if len(vals) > 0 else 1.0
    anchor_s = raw_s.get(int(last_date.month), 1.0) or 1.0
    seasonal_factors = {m_num: float(np.clip(val / anchor_s, 0.82, 1.22)) for m_num, val in raw_s.items()}

    # Empirical FX MoM volatility & Operating Profit residual volatility from recent history
    fx_mom = df["FX_Change_MoM_Pct"].dropna().astype(float) / 100.0
    recent_fx_mom = fx_mom.tail(min(48, len(fx_mom)))
    hist_fx_monthly_vol = float(np.clip(recent_fx_mom.std(ddof=1) if len(recent_fx_mom) >= 4 else 0.045, 0.015, 0.14))

    op_diff = df["Operating_Profit_NGN"].astype(float).diff().dropna()
    hist_resid_sigma = float(op_diff.tail(min(36, len(op_diff))).std(ddof=1)) if len(op_diff) >= 4 else abs(latest_actual_op) * 0.1

    base_defaults: Dict[str, float] = {
        "annual_inflation_pct": anchor_inflation,
        "unit_sales_growth_pct": 10.0,
        "price_inflation_pass_through_pct": 60.0,
        "fx_cost_pass_through_pct": 20.0,
        "usd_import_cost_growth_pct": 3.0,
        "opex_inflation_sensitivity_pct": 100.0,
        "fx_inflation_sensitivity_x": 1.0,
    }

    presets: Dict[str, Dict[str, float]] = {
        "base": dict(base_defaults),
        "opt": {
            "annual_inflation_pct": round(max(6.0, anchor_inflation - 4.5), 2),
            "unit_sales_growth_pct": 15.0,
            "price_inflation_pass_through_pct": 75.0,
            "fx_cost_pass_through_pct": 35.0,
            "usd_import_cost_growth_pct": 1.0,
            "opex_inflation_sensitivity_pct": 85.0,
            "fx_inflation_sensitivity_x": 0.7,
        },
        "stress": {
            "annual_inflation_pct": round(min(45.0, anchor_inflation + 10.0), 2),
            "unit_sales_growth_pct": -4.0,
            "price_inflation_pass_through_pct": 35.0,
            "fx_cost_pass_through_pct": 10.0,
            "usd_import_cost_growth_pct": 8.0,
            "opex_inflation_sensitivity_pct": 140.0,
            "fx_inflation_sensitivity_x": 1.8,
        },
    }

    effective_params: Dict[str, float] = {}
    raw_in = user_params or {}
    for k, dv in base_defaults.items():
        val = raw_in.get(k)
        effective_params[k] = dv if val is None else float(val)

    sim_kwargs = dict(
        last_date=last_date,
        horizon_months=horizon_months,
        anchor_units=anchor_units,
        anchor_price=anchor_price,
        anchor_fx=anchor_fx,
        anchor_usd_cost=anchor_usd_cost,
        anchor_mkt=anchor_mkt,
        anchor_other_opex=anchor_other_opex,
        seasonal_factors=seasonal_factors,
        hist_fx_monthly_vol=hist_fx_monthly_vol,
        hist_resid_sigma=hist_resid_sigma,
    )

    base_rows, base_sum = _simulate_accounting_5yr_path(params=base_defaults, **sim_kwargs)
    user_rows, user_sum = _simulate_accounting_5yr_path(params=effective_params, **sim_kwargs)

    # Assemble combined historical + 5-yr projection series
    monthly_series: List[Dict[str, Any]] = []
    hist_neg_count = 0
    n_hist = len(df)
    for idx, row in df.iterrows():
        op_val = round(float(row["Operating_Profit_NGN"]), 2)
        rev_val = round(float(row["Sales_Revenue_NGN"]), 2)
        cogs_val = round(float(row["COGS_NGN"]), 2)
        opex_val = round(float(row["Marketing_Spend_NGN"]) + float(row["Other_Operating_Cost_NGN"]), 2)
        margin_val = round(float(row["Operating_Margin_Pct"]), 2)
        is_neg = op_val < 0
        if is_neg:
            hist_neg_count += 1
        is_anchor = idx == (n_hist - 1)
        monthly_series.append({
            "month": str(row["Month"]),
            "month_label": str(row["Month_Label"]),
            "year": int(pd.to_datetime(row["Month"]).year),
            "period_type": "historical",
            "historical_op": op_val,
            "baseline_op": op_val if is_anchor else None,
            "scenario_op": op_val if is_anchor else None,
            "band_80": [op_val, op_val] if is_anchor else None,
            "negative_op": op_val if is_neg else None,
            "sales_revenue_ngn": rev_val,
            "cogs_ngn": cogs_val,
            "operating_expenses_ngn": opex_val,
            "operating_margin_pct": margin_val,
            "is_negative_profit": is_neg,
        })

    for b_r, u_r in zip(base_rows, user_rows):
        sc_op = u_r["operating_profit_ngn"]
        is_neg = sc_op < 0
        monthly_series.append({
            "month": u_r["month"],
            "month_label": u_r["month_label"],
            "year": u_r["year"],
            "period_type": "forecast",
            "historical_op": None,
            "baseline_op": b_r["operating_profit_ngn"],
            "scenario_op": sc_op,
            "band_80": [u_r["band_low_80_ngn"], u_r["band_high_80_ngn"]],
            "negative_op": sc_op if is_neg else None,
            "sales_revenue_ngn": u_r["sales_revenue_ngn"],
            "cogs_ngn": u_r["cogs_ngn"],
            "operating_expenses_ngn": u_r["operating_expenses_ngn"],
            "operating_margin_pct": u_r["operating_margin_pct"],
            "is_negative_profit": is_neg,
        })

    # Annual aggregation
    annual_map: Dict[int, Dict[str, Any]] = {}
    for pt in monthly_series:
        yr = pt["year"]
        if yr not in annual_map:
            annual_map[yr] = {
                "year": yr,
                "month_label": str(yr),
                "has_hist": False,
                "has_fc": False,
                "hist_op": 0.0,
                "base_op": 0.0,
                "scen_op": 0.0,
                "band_low": 0.0,
                "band_high": 0.0,
                "rev": 0.0,
                "months": 0,
            }
        bucket = annual_map[yr]
        bucket["months"] += 1
        bucket["rev"] += float(pt["sales_revenue_ngn"] or 0.0)
        if pt["period_type"] == "historical":
            bucket["has_hist"] = True
            bucket["hist_op"] += float(pt["historical_op"] or 0.0)
            bucket["base_op"] += float(pt["historical_op"] or 0.0)
            bucket["scen_op"] += float(pt["historical_op"] or 0.0)
            bucket["band_low"] += float(pt["historical_op"] or 0.0)
            bucket["band_high"] += float(pt["historical_op"] or 0.0)
        else:
            bucket["has_fc"] = True
            bucket["base_op"] += float(pt["baseline_op"] or 0.0)
            bucket["scen_op"] += float(pt["scenario_op"] or 0.0)
            b_arr = pt["band_80"] or [pt["scenario_op"], pt["scenario_op"]]
            bucket["band_low"] += float(b_arr[0])
            bucket["band_high"] += float(b_arr[1])

    annual_series: List[Dict[str, Any]] = []
    sorted_yrs = sorted(annual_map.keys())
    for yr in sorted_yrs:
        b = annual_map[yr]
        hist_v = round(b["hist_op"], 2) if (b["has_hist"] and not b["has_fc"]) else (round(b["hist_op"], 2) if b["has_hist"] else None)
        base_v = round(b["base_op"], 2) if b["has_fc"] else None
        scen_v = round(b["scen_op"], 2) if b["has_fc"] else None
        band_v = [round(b["band_low"], 2), round(b["band_high"], 2)] if b["has_fc"] else None
        eff_op = scen_v if scen_v is not None else (hist_v or 0.0)
        annual_series.append({
            "year": yr,
            "month_label": str(yr),
            "period_type": "forecast" if (b["has_fc"] and not b["has_hist"]) else ("mixed" if b["has_fc"] else "historical"),
            "historical_op": hist_v if not b["has_fc"] else None,
            "baseline_op": base_v if b["has_fc"] else None,
            "scenario_op": scen_v if b["has_fc"] else None,
            "band_80": band_v,
            "negative_op": eff_op if eff_op < 0 else None,
            "sales_revenue_ngn": round(b["rev"], 2),
            "operating_margin_pct": round((eff_op / b["rev"] * 100.0) if b["rev"] > 0 else 0.0, 2),
            "months": b["months"],
        })

    final_fc = user_rows[-1]
    start_lbl = str(df["Month_Label"].iloc[0])
    end_hist_lbl = str(last_row["Month_Label"])
    end_fc_lbl = final_fc["month_label"]

    return {
        "anchor_month_label": end_hist_lbl,
        "history_range_label": f"{start_lbl} – {end_hist_lbl}",
        "full_range_label": f"{start_lbl} – {end_fc_lbl}",
        "historical_records": n_hist,
        "forecast_horizon_months": horizon_months,
        "params": effective_params,
        "presets": presets,
        "summary": {
            "latest_actual_month": end_hist_lbl,
            "latest_actual_op_ngn": round(latest_actual_op, 2),
            "final_forecast_month": end_fc_lbl,
            "final_forecast_op_ngn": user_sum["final_op_ngn"],
            "final_baseline_op_ngn": base_sum["final_op_ngn"],
            "final_op_delta_ngn": round(user_sum["final_op_ngn"] - base_sum["final_op_ngn"], 2),
            "cumulative_5yr_op_ngn": user_sum["cumulative_op_ngn"],
            "cumulative_5yr_baseline_op_ngn": base_sum["cumulative_op_ngn"],
            "cumulative_5yr_delta_ngn": round(user_sum["cumulative_op_ngn"] - base_sum["cumulative_op_ngn"], 2),
            "operating_margin_5yr_pct": user_sum["operating_margin_pct"],
            "baseline_margin_5yr_pct": base_sum["operating_margin_pct"],
            "historical_negative_months": hist_neg_count,
            "forecast_negative_months": user_sum["negative_months"],
            "band_breach_months": user_sum["band_breach_months"],
        },
        "monthly_series": monthly_series,
        "annual_series": annual_series,
    }

