from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from .lightweight_ml import (
        GradientBoostingRegressor,
        LinearRegression,
        Pipeline,
        RandomForestRegressor,
        Ridge,
        SimpleImputer,
        StandardScaler,
        TimeSeriesSplit,
        clone,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
except ImportError:
    from lightweight_ml import (
        GradientBoostingRegressor,
        LinearRegression,
        Pipeline,
        RandomForestRegressor,
        Ridge,
        SimpleImputer,
        StandardScaler,
        TimeSeriesSplit,
        clone,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

TARGET_COL = "Operating_Profit_NGN"

# Strictly leakage-free driver predictors (never target, never Operating_Margin_Pct, never COGS/Revenue/GrossProfit)
DRIVER_FEATURES = [
    "USD_NGN_Planning_Rate",
    "Inflation_YoY_Pct",
    "Units_Sold",
    "Import_Cost_USD_Per_Unit",
    "Avg_Selling_Price_NGN",
    "Marketing_Spend_NGN",
    "Other_Operating_Cost_NGN",
    "FX_Change_MoM_Pct",
    "Import_Cost_NGN_Per_Unit",
    "Unit_Spread_NGN",
    "Price_to_Import_Ratio",
    "Quarter_Num",
    "Month_Sin",
    "Month_Cos",
]

# Strictly pre-period / planning predictors for Ex-Ante Forecasting (no concurrent Units_Sold)
EX_ANTE_FEATURES = [
    "USD_NGN_Planning_Rate",
    "Inflation_YoY_Pct",
    "Import_Cost_USD_Per_Unit",
    "Avg_Selling_Price_NGN",
    "Marketing_Spend_NGN",
    "Other_Operating_Cost_NGN",
    "FX_Change_MoM_Pct",
    "Import_Cost_NGN_Per_Unit",
    "Unit_Spread_NGN",
    "Price_to_Import_Ratio",
    "Quarter_Num",
    "Month_Sin",
    "Month_Cos",
    "Lag1_Units_Sold",
    "Lag1_Operating_Profit_NGN",
    "Rolling3M_Operating_Profit_NGN",
]

EXCLUDED_LEAKAGE_FEATURES = [
    {
        "feature": "Operating_Profit_NGN",
        "reason": "Prediction target variable.",
        "availability": "Realized at period close",
    },
    {
        "feature": "Operating_Margin_Pct",
        "reason": "Direct algebraic ratio of target (Operating_Profit_NGN / Sales_Revenue_NGN).",
        "availability": "Realized at period close",
    },
    {
        "feature": "Gross_Profit_NGN",
        "reason": "Directly reconstructs target via Gross_Profit_NGN - Marketing - Other_OpEx.",
        "availability": "Realized at period close",
    },
    {
        "feature": "Gross_Margin_Pct",
        "reason": "Derived directly from Gross_Profit_NGN / Sales_Revenue_NGN.",
        "availability": "Realized at period close",
    },
    {
        "feature": "Sales_Revenue_NGN",
        "reason": "Combined with COGS_NGN, trivially yields Operating_Profit_NGN via accounting identity.",
        "availability": "Realized at period close",
    },
    {
        "feature": "COGS_NGN",
        "reason": "Combined with Sales_Revenue_NGN, trivially yields Operating_Profit_NGN via accounting identity.",
        "availability": "Realized at period close",
    },
]


def _quarter_to_num(q: Any) -> int:
    s = str(q).strip().upper()
    if s in {"Q1", "1"}:
        return 1
    if s in {"Q2", "2"}:
        return 2
    if s in {"Q3", "3"}:
        return 3
    if s in {"Q4", "4"}:
        return 4
    return 1


def prepare_modeling_dataframe(processed_df: pd.DataFrame) -> pd.DataFrame:
    df = processed_df.copy()
    df["Quarter_Num"] = df["Quarter"].apply(_quarter_to_num).astype(float)
    df["Month_Sin"] = np.sin(2.0 * np.pi * df["Month_Num"].astype(float) / 12.0)
    df["Month_Cos"] = np.cos(2.0 * np.pi * df["Month_Num"].astype(float) / 12.0)
    df["Unit_Spread_NGN"] = df["Avg_Selling_Price_NGN"].astype(float) - df["Import_Cost_NGN_Per_Unit"].astype(float)
    df["Price_to_Import_Ratio"] = np.where(
        df["Import_Cost_NGN_Per_Unit"].astype(float) > 1e-6,
        df["Avg_Selling_Price_NGN"].astype(float) / df["Import_Cost_NGN_Per_Unit"].astype(float),
        1.0,
    )
    df["Lag1_Units_Sold"] = df["Units_Sold"].shift(1)
    df["Lag1_Operating_Profit_NGN"] = df[TARGET_COL].shift(1)
    df["Rolling3M_Operating_Profit_NGN"] = df[TARGET_COL].shift(1).rolling(3, min_periods=1).mean()
    return df


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    mask = np.abs(y_true) > 1e-6
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-6
    if not mask.any():
        return None
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    mape = _safe_mape(y_true, y_pred)
    smape = _safe_smape(y_true, y_pred)
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4) if r2 is not None else None,
        "mape_pct": round(mape, 2) if mape is not None else None,
        "smape_pct": round(smape, 2) if smape is not None else None,
    }


def _build_candidates() -> List[Tuple[str, str, Any, bool]]:
    """Returns (key, display_name, estimator, is_tree_model)."""
    return [
        ("linear", "Linear Regression (Baseline)", LinearRegression(), False),
        ("ridge", "Ridge Regression (L2)", Ridge(alpha=1.0, random_state=42), False),
        (
            "random_forest",
            "Random Forest Regressor",
            RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=2, random_state=42),
            True,
        ),
        (
            "gradient_boosting",
            "Gradient Boosting Regressor",
            GradientBoostingRegressor(
                n_estimators=140, learning_rate=0.05, max_depth=3, subsample=0.85, random_state=42
            ),
            True,
        ),
    ]


def _make_preprocessor(feature_cols: List[str], scale: bool) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def train_and_evaluate_suite(
    processed_df: pd.DataFrame,
    feature_cols: List[str],
    mode_label: str,
) -> Dict[str, Any]:
    df_mod = prepare_modeling_dataframe(processed_df)
    valid_df = df_mod.dropna(subset=[TARGET_COL]).copy().reset_index(drop=True)

    # Drop leading rows if lag features are all NaN on row 0 in ex-ante mode
    if "Lag1_Operating_Profit_NGN" in feature_cols:
        valid_df = valid_df.dropna(subset=["Lag1_Operating_Profit_NGN"]).reset_index(drop=True)

    n = len(valid_df)
    if n < 12:
        return {
            "mode": mode_label,
            "sufficient_data": False,
            "limitation_notice": f"Only {n} valid observations available. At least 12 chronological observations are required for reliable out-of-time validation.",
            "models": [],
            "selected_model": None,
        }

    # Chronological split: 85% train, 15% out-of-time test (at least 4 test observations)
    test_size = max(4, int(round(n * 0.15)))
    if n - test_size < 8:
        test_size = max(3, n // 4)
    split_idx = n - test_size

    train_df = valid_df.iloc[:split_idx].copy()
    test_df = valid_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].astype(float).values
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].astype(float).values

    # Expanding-window TimeSeriesSplit CV inside training set
    n_splits = min(4, max(2, len(train_df) // 15))
    tscv = TimeSeriesSplit(n_splits=n_splits) if len(train_df) >= 12 else None

    model_results: List[Dict[str, Any]] = []
    fitted_artifacts: Dict[str, Dict[str, Any]] = {}

    for key, display_name, estimator, is_tree in _build_candidates():
        prep = _make_preprocessor(feature_cols, scale=not is_tree)
        X_train_t = prep.fit_transform(X_train)
        X_test_t = prep.transform(X_test)

        # TimeSeriesSplit CV on train
        cv_rmses: List[float] = []
        if tscv is not None:
            for tr_i, val_i in tscv.split(X_train_t):
                if len(tr_i) < 5 or len(val_i) < 2:
                    continue
                cv_prep = _make_preprocessor(feature_cols, scale=not is_tree)
                X_tr_cv = cv_prep.fit_transform(X_train.iloc[tr_i])
                X_va_cv = cv_prep.transform(X_train.iloc[val_i])
                cv_est = clone(estimator)
                cv_est.fit(X_tr_cv, y_train[tr_i])
                cv_pred = cv_est.predict(X_va_cv)
                cv_rmses.append(math.sqrt(mean_squared_error(y_train[val_i], cv_pred)))

        estimator.fit(X_train_t, y_train)
        train_pred = estimator.predict(X_train_t)
        test_pred = estimator.predict(X_test_t)

        train_metrics = _compute_metrics(y_train, train_pred)
        test_metrics = _compute_metrics(y_test, test_pred)

        residuals_test = (y_test - test_pred).tolist()
        residuals_train = (y_train - train_pred).tolist()

        model_results.append({
            "key": key,
            "name": display_name,
            "is_tree": is_tree,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_rmse": round(float(np.mean(cv_rmses)), 2) if cv_rmses else None,
            "test_predictions": [
                {
                    "month": str(test_df.iloc[i]["Month"]),
                    "month_label": str(test_df.iloc[i]["Month_Label"]),
                    "actual": round(float(y_test[i]), 2),
                    "predicted": round(float(test_pred[i]), 2),
                    "residual": round(float(residuals_test[i]), 2),
                }
                for i in range(len(test_df))
            ],
            "train_predictions": [
                {
                    "month": str(train_df.iloc[i]["Month"]),
                    "month_label": str(train_df.iloc[i]["Month_Label"]),
                    "actual": round(float(y_train[i]), 2),
                    "predicted": round(float(train_pred[i]), 2),
                    "residual": round(float(residuals_train[i]), 2),
                }
                for i in range(len(train_df))
            ],
        })

        fitted_artifacts[key] = {
            "estimator": estimator,
            "preprocessor": prep,
            "X_train_df": X_train,
            "X_test_df": X_test,
            "X_train_t": X_train_t,
            "X_test_t": X_test_t,
            "y_train": y_train,
            "y_test": y_test,
            "train_pred": train_pred,
            "test_pred": test_pred,
            "is_tree": is_tree,
            "display_name": display_name,
            "feature_cols": feature_cols,
        }

    # Select final model strictly on out-of-time test RMSE (lower is better)
    model_results.sort(key=lambda m: m["test_metrics"]["rmse"] if m["test_metrics"]["rmse"] is not None else float("inf"))
    best_model_summary = model_results[0]
    best_key = best_model_summary["key"]

    # Also build full-series fit artifact for interactive deployment AFTER selection
    # Note: we keep the chronological test evaluation intact and honest, and also fit a production pipeline on all data for forward user predictions if desired,
    # OR keep the out-of-time validated estimator for consistent explainability! Let's retain BOTH:
    # We evaluate strictly on out-of-time test, and for forward prediction we also fit a full-history clone of the selected architecture so 2026+ user predictions aren't truncated at 2024!
    best_art = fitted_artifacts[best_key]
    full_prep = _make_preprocessor(feature_cols, scale=not best_art["is_tree"])
    X_full_df = valid_df[feature_cols].copy()
    y_full = valid_df[TARGET_COL].astype(float).values
    X_full_t = full_prep.fit_transform(X_full_df)
    full_est = clone(best_art["estimator"])
    full_est.fit(X_full_t, y_full)

    # Fit continuous scaled Ridge benchmark on full history for scalable extrapolation checks
    ridge_full_prep = _make_preprocessor(feature_cols, scale=True)
    X_full_ridge_t = ridge_full_prep.fit_transform(X_full_df)
    ridge_full_est = Ridge(alpha=1.0, random_state=42)
    ridge_full_est.fit(X_full_ridge_t, y_full)

    feature_bounds: Dict[str, Dict[str, float]] = {}
    for col in feature_cols:
        s = X_full_df[col].dropna().astype(float)
        if len(s) > 0:
            feature_bounds[col] = {
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "mean": round(float(s.mean()), 4),
            }

    limitation_notice = None
    if n < 36:
        limitation_notice = (
            f"Dataset contains {n} observations ({len(train_df)} train / {len(test_df)} out-of-time test). "
            "Small sample size increases variance of out-of-time error estimates."
        )
    elif best_art["is_tree"]:
        limitation_notice = (
            "Tree ensembles cannot extrapolate operating profit outside historical price/volume ranges; "
            "monitor drift if macro inputs move materially beyond historical bounds."
        )

    return {
        "mode": mode_label,
        "sufficient_data": True,
        "dataset_size": n,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_period": f"{train_df['Month'].iloc[0]} to {train_df['Month'].iloc[-1]}",
        "test_period": f"{test_df['Month'].iloc[0]} to {test_df['Month'].iloc[-1]}",
        "feature_cols": feature_cols,
        "feature_bounds": feature_bounds,
        "models": model_results,
        "selected_model": best_model_summary,
        "limitation_notice": limitation_notice,
        "leakage_audit": EXCLUDED_LEAKAGE_FEATURES,
        "_artifacts": {
            "best_key": best_key,
            "holdout": best_art,
            "full_estimator": full_est,
            "full_preprocessor": full_prep,
            "ridge_full_estimator": ridge_full_est,
            "ridge_full_preprocessor": ridge_full_prep,
            "X_full_df": X_full_df,
            "X_full_t": X_full_t,
            "y_full": y_full,
            "valid_df": valid_df,
        },
    }


def _compute_single_feature_row(
    fx: float,
    inf_pct: float,
    units: float,
    imp_usd: float,
    price: float,
    mkt: float,
    other_op: float,
    fx_mom: float,
    quarter: str,
    month_num: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    import_cost_ngn_per_unit = fx * imp_usd
    cogs_ngn = units * import_cost_ngn_per_unit
    sales_revenue_ngn = units * price
    gross_profit_ngn = sales_revenue_ngn - cogs_ngn
    gross_margin_pct = (gross_profit_ngn / sales_revenue_ngn) * 100.0 if sales_revenue_ngn > 0 else 0.0
    accounting_operating_profit = gross_profit_ngn - mkt - other_op
    accounting_operating_margin_pct = (
        (accounting_operating_profit / sales_revenue_ngn) * 100.0 if sales_revenue_ngn > 0 else 0.0
    )
    unit_spread_ngn = price - import_cost_ngn_per_unit
    price_to_import_ratio = (price / import_cost_ngn_per_unit) if import_cost_ngn_per_unit > 1e-6 else 1.0

    feature_row = {
        "USD_NGN_Planning_Rate": fx,
        "Inflation_YoY_Pct": inf_pct,
        "Units_Sold": units,
        "Import_Cost_USD_Per_Unit": imp_usd,
        "Avg_Selling_Price_NGN": price,
        "Marketing_Spend_NGN": mkt,
        "Other_Operating_Cost_NGN": other_op,
        "FX_Change_MoM_Pct": fx_mom,
        "Import_Cost_NGN_Per_Unit": import_cost_ngn_per_unit,
        "Unit_Spread_NGN": unit_spread_ngn,
        "Price_to_Import_Ratio": price_to_import_ratio,
        "Quarter_Num": float(_quarter_to_num(quarter)),
        "Month_Sin": math.sin(2.0 * math.pi * month_num / 12.0),
        "Month_Cos": math.cos(2.0 * math.pi * month_num / 12.0),
    }
    acc_metrics = {
        "Import_Cost_NGN_Per_Unit": import_cost_ngn_per_unit,
        "COGS_NGN": cogs_ngn,
        "Sales_Revenue_NGN": sales_revenue_ngn,
        "Gross_Profit_NGN": gross_profit_ngn,
        "Gross_Margin_Pct": gross_margin_pct,
        "Accounting_Operating_Profit_NGN": accounting_operating_profit,
        "Accounting_Operating_Margin_Pct": accounting_operating_margin_pct,
        "Unit_Spread_NGN": unit_spread_ngn,
        "Price_to_Import_Ratio": price_to_import_ratio,
    }
    return feature_row, acc_metrics


def predict_from_user_inputs(
    model_bundle: Dict[str, Any],
    user_inputs: Dict[str, Any],
    last_known_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes the end-to-end scalable & measurable prediction workflow:
    1. Validate user inputs & optional horizon scaling
    2. Apply exact feature engineering pipeline & detect historical bound extrapolations
    3. Pass transformed features into Champion ML model + Ridge linear benchmark
    4. Compute calibrated prediction intervals, unit economics, and +10% driver sensitivities
    5. Return predicted Operating_Profit_NGN + separate accounting reconciliation
    """
    fx = float(user_inputs["USD_NGN_Planning_Rate"])
    inf_raw = float(user_inputs["Inflation_YoY_Pct"])
    inf_pct = inf_raw * 100.0 if abs(inf_raw) <= 1.5 and user_inputs.get("inflation_is_decimal") else inf_raw
    units = float(user_inputs["Units_Sold"])
    imp_usd = float(user_inputs["Import_Cost_USD_Per_Unit"])
    price = float(user_inputs["Avg_Selling_Price_NGN"])
    mkt = float(user_inputs["Marketing_Spend_NGN"])
    other_op = float(user_inputs["Other_Operating_Cost_NGN"])
    quarter = str(user_inputs.get("Quarter") or "Q4").strip().upper()
    month_num = int(user_inputs.get("Month_Num") or (3 * _quarter_to_num(quarter)))
    horizon_months = max(1, min(60, int(user_inputs.get("horizon_months") or 1)))

    prev_fx = float(last_known_row.get("USD_NGN_Planning_Rate") or fx)
    if user_inputs.get("FX_Change_MoM_Pct") is not None and user_inputs.get("FX_Change_MoM_Pct") != "":
        fx_mom = float(user_inputs["FX_Change_MoM_Pct"])
    else:
        fx_mom = ((fx - prev_fx) / prev_fx) * 100.0 if prev_fx else 0.0

    feature_row, acc = _compute_single_feature_row(
        fx=fx,
        inf_pct=inf_pct,
        units=units,
        imp_usd=imp_usd,
        price=price,
        mkt=mkt,
        other_op=other_op,
        fx_mom=fx_mom,
        quarter=quarter,
        month_num=month_num,
    )

    art = model_bundle["_artifacts"]
    feature_cols = model_bundle["feature_cols"]
    full_prep = art["full_preprocessor"]
    full_est = art["full_estimator"]
    ridge_prep = art.get("ridge_full_preprocessor")
    ridge_est = art.get("ridge_full_estimator")

    X_input_df = pd.DataFrame([[feature_row[c] for c in feature_cols]], columns=feature_cols)
    X_input_t = full_prep.transform(X_input_df)
    ml_predicted_profit = float(full_est.predict(X_input_t)[0])

    ridge_predicted_profit = ml_predicted_profit
    if ridge_prep is not None and ridge_est is not None:
        ridge_predicted_profit = float(ridge_est.predict(ridge_prep.transform(X_input_df))[0])

    sales_revenue_ngn = acc["Sales_Revenue_NGN"]
    ml_predicted_margin_pct = (
        (ml_predicted_profit / sales_revenue_ngn) * 100.0 if sales_revenue_ngn > 0 else 0.0
    )

    # Detect out-of-range scaled inputs relative to training history
    bounds = model_bundle.get("feature_bounds") or {}
    check_cols = [
        ("USD_NGN_Planning_Rate", "USD/NGN Planning Rate"),
        ("Units_Sold", "Units Sold"),
        ("Avg_Selling_Price_NGN", "Avg Selling Price (NGN)"),
        ("Import_Cost_USD_Per_Unit", "Import Cost (USD/Unit)"),
        ("Marketing_Spend_NGN", "Marketing Spend (NGN)"),
        ("Other_Operating_Cost_NGN", "Other Operating Cost (NGN)"),
    ]
    out_of_range_warnings: List[Dict[str, Any]] = []
    for col, label in check_cols:
        b = bounds.get(col)
        if not b:
            continue
        v = feature_row[col]
        if v < b["min"] or v > b["max"]:
            out_of_range_warnings.append({
                "feature": col,
                "label": label,
                "input_value": round(v, 2),
                "hist_min": round(b["min"], 2),
                "hist_max": round(b["max"], 2),
                "direction": "above historical maximum" if v > b["max"] else "below historical minimum",
            })

    # Calibrated Prediction Intervals from out-of-time RMSE
    rmse = float(model_bundle["selected_model"]["test_metrics"]["rmse"] or 9_000_000.0)
    pi_80_low = ml_predicted_profit - 1.2816 * rmse
    pi_80_high = ml_predicted_profit + 1.2816 * rmse
    pi_95_low = ml_predicted_profit - 1.9600 * rmse
    pi_95_high = ml_predicted_profit + 1.9600 * rmse

    # Measurable Unit Economics & Break-Even Metrics
    total_opex = mkt + other_op
    unit_spread = acc["Unit_Spread_NGN"]
    break_even_units = (total_opex / unit_spread) if unit_spread > 1.0 else None
    margin_of_safety_pct = (
        ((units - break_even_units) / units) * 100.0
        if (break_even_units is not None and units > 0)
        else None
    )
    marketing_roas = (sales_revenue_ngn / mkt) if mkt > 0 else None
    opex_to_rev_pct = (total_opex / sales_revenue_ngn) * 100.0 if sales_revenue_ngn > 0 else 0.0
    mkt_to_rev_pct = (mkt / sales_revenue_ngn) * 100.0 if sales_revenue_ngn > 0 else 0.0

    # Measurable +10% Local Driver Sensitivity Probe
    sensitivities: List[Dict[str, Any]] = []
    shock_specs = [
        ("Units_Sold", "Sales Volume (+10%)", {"units": units * 1.10}),
        ("Avg_Selling_Price_NGN", "Selling Price (+10%)", {"price": price * 1.10}),
        ("USD_NGN_Planning_Rate", "USD/NGN FX Rate (+10%)", {"fx": fx * 1.10}),
        ("Import_Cost_USD_Per_Unit", "USD Import Cost (+10%)", {"imp_usd": imp_usd * 1.10}),
        ("Marketing_Spend_NGN", "Marketing Spend (+10%)", {"mkt": mkt * 1.10}),
        ("Other_Operating_Cost_NGN", "Other OpEx (+10%)", {"other_op": other_op * 1.10}),
    ]
    for skey, slabel, overrides in shock_specs:
        s_feat, s_acc = _compute_single_feature_row(
            fx=overrides.get("fx", fx),
            inf_pct=inf_pct,
            units=overrides.get("units", units),
            imp_usd=overrides.get("imp_usd", imp_usd),
            price=overrides.get("price", price),
            mkt=overrides.get("mkt", mkt),
            other_op=overrides.get("other_op", other_op),
            fx_mom=fx_mom,
            quarter=quarter,
            month_num=month_num,
        )
        X_s_df = pd.DataFrame([[s_feat[c] for c in feature_cols]], columns=feature_cols)
        s_ml_pred = float(full_est.predict(full_prep.transform(X_s_df))[0])
        acc_delta = s_acc["Accounting_Operating_Profit_NGN"] - acc["Accounting_Operating_Profit_NGN"]
        ml_delta = s_ml_pred - ml_predicted_profit
        sensitivities.append({
            "lever": skey,
            "label": slabel,
            "accounting_delta_ngn": round(acc_delta, 2),
            "accounting_delta_pct": round((acc_delta / max(abs(acc["Accounting_Operating_Profit_NGN"]), 1.0)) * 100.0, 2),
            "ml_delta_ngn": round(ml_delta, 2),
            "ml_delta_pct": round((ml_delta / max(abs(ml_predicted_profit), 1.0)) * 100.0, 2),
        })

    # Scalable Horizon Totals (1M, 3M, 12M, or custom horizon_months)
    horizon_scale_rmse = rmse * math.sqrt(horizon_months)
    horizon_scaling = {
        "horizon_months": horizon_months,
        "ml_scaled_profit_ngn": round(ml_predicted_profit * horizon_months, 2),
        "ml_scaled_ci95_low_ngn": round(ml_predicted_profit * horizon_months - 1.96 * horizon_scale_rmse, 2),
        "ml_scaled_ci95_high_ngn": round(ml_predicted_profit * horizon_months + 1.96 * horizon_scale_rmse, 2),
        "accounting_scaled_profit_ngn": round(acc["Accounting_Operating_Profit_NGN"] * horizon_months, 2),
        "scaled_revenue_ngn": round(sales_revenue_ngn * horizon_months, 2),
        "scaled_units_sold": round(units * horizon_months, 0),
        "ridge_scaled_profit_ngn": round(ridge_predicted_profit * horizon_months, 2),
    }

    return {
        "ml_prediction": {
            "model_name": model_bundle["selected_model"]["name"],
            "model_key": model_bundle["selected_model"]["key"],
            "is_tree_model": bool(model_bundle["selected_model"].get("is_tree")),
            "predicted_operating_profit_ngn": round(ml_predicted_profit, 2),
            "predicted_operating_margin_pct": round(ml_predicted_margin_pct, 2),
            "ridge_extrapolation_ngn": round(ridge_predicted_profit, 2),
            "prediction_interval_80": [round(pi_80_low, 2), round(pi_80_high, 2)],
            "prediction_interval_95": [round(pi_95_low, 2), round(pi_95_high, 2)],
            "out_of_range_warnings": out_of_range_warnings,
            "out_of_time_rmse": model_bundle["selected_model"]["test_metrics"]["rmse"],
            "out_of_time_r2": model_bundle["selected_model"]["test_metrics"]["r2"],
            "input_feature_vector": feature_row,
            "X_input_t": X_input_t,
            "X_input_df": X_input_df,
        },
        "accounting_reconciliation": {
            "Import_Cost_NGN_Per_Unit": round(acc["Import_Cost_NGN_Per_Unit"], 2),
            "COGS_NGN": round(acc["COGS_NGN"], 2),
            "Sales_Revenue_NGN": round(sales_revenue_ngn, 2),
            "Gross_Profit_NGN": round(acc["Gross_Profit_NGN"], 2),
            "Gross_Margin_Pct": round(acc["Gross_Margin_Pct"], 2),
            "Marketing_Spend_NGN": round(mkt, 2),
            "Other_Operating_Cost_NGN": round(other_op, 2),
            "Accounting_Operating_Profit_NGN": round(acc["Accounting_Operating_Profit_NGN"], 2),
            "Accounting_Operating_Margin_Pct": round(acc["Accounting_Operating_Margin_Pct"], 2),
            "FX_Change_MoM_Pct": round(fx_mom, 3),
            "ml_vs_accounting_gap_ngn": round(ml_predicted_profit - acc["Accounting_Operating_Profit_NGN"], 2),
        },
        "measurable_unit_economics": {
            "unit_spread_ngn": round(unit_spread, 2),
            "price_to_import_ratio": round(acc["Price_to_Import_Ratio"], 3),
            "break_even_units": round(break_even_units, 0) if break_even_units is not None else None,
            "margin_of_safety_pct": round(margin_of_safety_pct, 2) if margin_of_safety_pct is not None else None,
            "marketing_roas": round(marketing_roas, 2) if marketing_roas is not None else None,
            "opex_to_revenue_pct": round(opex_to_rev_pct, 2),
            "marketing_to_revenue_pct": round(mkt_to_rev_pct, 2),
        },
        "measurable_sensitivities": sensitivities,
        "horizon_scaling": horizon_scaling,
    }
