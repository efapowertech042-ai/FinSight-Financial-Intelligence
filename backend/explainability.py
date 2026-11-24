from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from backend.lightweight_ml import permutation_importance


def _exact_tree_shap_single(tree: Any, x: np.ndarray) -> np.ndarray:
    """
    Exact TreeSHAP (Lundberg, Erion & Lee 2018, Algorithm 2) over _FlatTree arrays.
    Satisfies: tree.value[0] + sum(phi) == tree.predict(x) to machine precision.
    """
    n_features = x.shape[0]
    phi = np.zeros(n_features, dtype=float)

    f = tree.feature
    t = tree.threshold
    cl = tree.children_left
    cr = tree.children_right
    val = tree.value
    r_count = tree.n_node_samples

    # Path tuple entries: [feature_idx, zero_fraction, one_fraction, pweight]
    def extend(
        m: List[List[float]],
        pz: float,
        po: float,
        pi: int,
    ) -> List[List[float]]:
        l_len = len(m)
        m_new = [list(row) for row in m]
        m_new.append([float(pi), pz, po, 1.0 if l_len == 0 else 0.0])
        for i in range(l_len - 1, -1, -1):
            m_new[i + 1][3] += po * m_new[i][3] * (i + 1) / (l_len + 1)
            m_new[i][3] = pz * m_new[i][3] * (l_len - i) / (l_len + 1)
        return m_new

    def unwind(m: List[List[float]], idx: int) -> List[List[float]]:
        l_len = len(m) - 1
        n_val = m[l_len][3]
        m_new = [list(row) for row in m[:l_len]]
        po = m[idx][2]
        pz = m[idx][1]
        for i in range(l_len - 1, -1, -1):
            if po != 0.0:
                tmp = m_new[i][3]
                m_new[i][3] = n_val * (l_len + 1) / ((i + 1) * po)
                n_val = tmp - m_new[i][3] * pz * (l_len - i) / (l_len + 1)
            else:
                m_new[i][3] = (m_new[i][3] * (l_len + 1)) / (pz * (l_len - i))
        for i in range(idx, l_len):
            m_new[i][0] = m[i + 1][0]
            m_new[i][1] = m[i + 1][1]
            m_new[i][2] = m[i + 1][2]
        return m_new

    def unwound_sum(m: List[List[float]], idx: int) -> float:
        l_len = len(m) - 1
        total = 0.0
        n_val = m[l_len][3]
        po = m[idx][2]
        pz = m[idx][1]
        for i in range(l_len - 1, -1, -1):
            if po != 0.0:
                tmp = n_val * (l_len + 1) / ((i + 1) * po)
                total += tmp
                n_val = m[i][3] - tmp * pz * (l_len - i) / (l_len + 1)
            else:
                total += (m[i][3] * (l_len + 1)) / (pz * (l_len - i))
        return total

    def recurse(
        j: int,
        m: List[List[float]],
        pz: float,
        po: float,
        pi: int,
    ) -> None:
        m = extend(m, pz, po, pi)
        feat = int(f[j])
        if feat < 0:
            # Leaf node
            for i in range(1, len(m)):
                w = unwound_sum(m, i)
                fi = int(m[i][0])
                phi[fi] += w * (m[i][2] - m[i][1]) * float(val[j])
            return

        hot = int(cl[j]) if x[feat] <= t[j] else int(cr[j])
        cold = int(cr[j]) if hot == int(cl[j]) else int(cl[j])
        r_j = max(1.0, float(r_count[j]))
        r_hot = float(r_count[hot])
        r_cold = float(r_count[cold])

        iz = 1.0
        io = 1.0
        existing_k = -1
        for k in range(1, len(m)):
            if int(m[k][0]) == feat:
                existing_k = k
                break
        if existing_k != -1:
            iz = m[existing_k][1]
            io = m[existing_k][2]
            m = unwind(m, existing_k)

        recurse(hot, m, iz * (r_hot / r_j), io, feat)
        recurse(cold, m, iz * (r_cold / r_j), 0.0, feat)

    # Start recursion cleanly from root
    feat0 = int(f[0])
    if feat0 < 0:
        return phi
    hot0 = int(cl[0]) if x[feat0] <= t[0] else int(cr[0])
    cold0 = int(cr[0]) if hot0 == int(cl[0]) else int(cl[0])
    r0 = max(1.0, float(r_count[0]))
    recurse(hot0, [[0.0, 1.0, 1.0, 1.0]], float(r_count[hot0]) / r0, 1.0, feat0)
    recurse(cold0, [[0.0, 1.0, 1.0, 1.0]], float(r_count[cold0]) / r0, 0.0, feat0)
    return phi


def _compute_shap_matrix(estimator: Any, X_bg_t: np.ndarray, X_eval_t: np.ndarray, is_tree: bool) -> Tuple[np.ndarray, float, Any]:
    X_bg = np.asarray(X_bg_t, dtype=float)
    X_ev = np.asarray(X_eval_t, dtype=float)

    if not is_tree and hasattr(estimator, "coef_"):
        # Exact LinearSHAP under independent/mean background
        bg_mean = np.mean(X_bg, axis=0)
        coef = np.asarray(estimator.coef_, dtype=float)
        shap_arr = (X_ev - bg_mean) * coef
        base_val = float(np.mean(estimator.predict(X_bg)))
        return shap_arr, base_val, None

    if is_tree and hasattr(estimator, "estimators_") and estimator.estimators_:
        trees = estimator.estimators_
        n_eval, n_feat = X_ev.shape
        shap_arr = np.zeros((n_eval, n_feat), dtype=float)
        if hasattr(estimator, "init_prediction_"):
            # GradientBoostingRegressor: pred = init_prediction_ + lr * sum(trees)
            lr = float(getattr(estimator, "learning_rate", 0.05))
            base_val = float(estimator.init_prediction_) + lr * sum(float(t.value[0]) for t in trees)
            for i in range(n_eval):
                acc = np.zeros(n_feat, dtype=float)
                for t in trees:
                    acc += lr * _exact_tree_shap_single(t, X_ev[i])
                shap_arr[i] = acc
        else:
            # RandomForestRegressor: pred = mean(trees)
            base_val = float(np.mean([float(t.value[0]) for t in trees]))
            for i in range(n_eval):
                acc = np.zeros(n_feat, dtype=float)
                for t in trees:
                    acc += _exact_tree_shap_single(t, X_ev[i])
                shap_arr[i] = acc / len(trees)
        return shap_arr, base_val, None

    # Safe empirical sensitivity decomposition fallback
    bg_mean = np.mean(X_bg, axis=0)
    base_val = float(estimator.predict(bg_mean.reshape(1, -1))[0])
    shap_arr = np.zeros_like(X_ev, dtype=float)
    for i in range(X_ev.shape[0]):
        row = X_ev[i]
        pred_full = float(estimator.predict(row.reshape(1, -1))[0])
        raw = np.zeros(X_ev.shape[1], dtype=float)
        for j in range(X_ev.shape[1]):
            alt = row.copy()
            alt[j] = bg_mean[j]
            raw[j] = pred_full - float(estimator.predict(alt.reshape(1, -1))[0])
        diff = (pred_full - base_val) - float(np.sum(raw))
        if abs(np.sum(np.abs(raw))) > 1e-12:
            raw += diff * (np.abs(raw) / np.sum(np.abs(raw)))
        shap_arr[i] = raw
    return shap_arr, base_val, None


def compute_global_explainability(model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    if not model_bundle.get("sufficient_data"):
        return {"available": False, "reason": model_bundle.get("limitation_notice")}

    art = model_bundle["_artifacts"]
    holdout = art["holdout"]
    estimator = holdout["estimator"]
    X_test_t = holdout["X_test_t"]
    y_test = holdout["y_test"]
    is_tree = holdout["is_tree"]
    feature_cols = model_bundle["feature_cols"]

    # Evaluate SHAP on full-history model over all rows for rich summary & dependence curves
    full_est = art["full_estimator"]
    X_full_t = art["X_full_t"]
    X_full_df = art["X_full_df"]
    valid_df = art["valid_df"]

    shap_arr, base_val, _ = _compute_shap_matrix(full_est, X_full_t, X_full_t, is_tree)

    # Compute permutation importance strictly on out-of-time test set!
    perm = permutation_importance(
        estimator,
        X_test_t,
        y_test,
        n_repeats=15,
        random_state=42,
        scoring="r2",
    )

    global_importance: List[Dict[str, Any]] = []
    beeswarm_points: List[Dict[str, Any]] = []

    for j, fname in enumerate(feature_cols):
        col_vals = X_full_df[fname].astype(float).values
        col_shap = shap_arr[:, j]
        mean_abs_shap = float(np.mean(np.abs(col_shap)))
        mean_signed_shap = float(np.mean(col_shap))

        # Directionality: correlation between feature value and SHAP contribution
        if np.std(col_vals) > 1e-9 and np.std(col_shap) > 1e-9:
            corr = float(np.corrcoef(col_vals, col_shap)[0, 1])
            if math.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        direction = (
            "Positive (higher value raises predicted profit)"
            if corr > 0.15
            else "Negative (higher value lowers predicted profit)"
            if corr < -0.15
            else "Non-linear / Mixed effect"
        )

        global_importance.append({
            "feature": fname,
            "mean_abs_shap": round(mean_abs_shap, 2),
            "mean_signed_shap": round(mean_signed_shap, 2),
            "value_shap_correlation": round(corr, 4),
            "direction": direction,
            "permutation_importance_mean": round(float(perm.importances_mean[j]), 4),
            "permutation_importance_std": round(float(perm.importances_std[j]), 4),
        })

        # Normalized feature value [0, 1] for beeswarm coloring
        f_min = float(np.min(col_vals))
        f_max = float(np.max(col_vals))
        span = f_max - f_min if f_max > f_min else 1.0
        for i in range(len(col_vals)):
            beeswarm_points.append({
                "feature": fname,
                "month": str(valid_df.iloc[i]["Month"]),
                "month_label": str(valid_df.iloc[i]["Month_Label"]),
                "feature_value": round(float(col_vals[i]), 4),
                "normalized_value": round(float((col_vals[i] - f_min) / span), 4),
                "shap_value": round(float(col_shap[i]), 2),
            })

    global_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    total_abs = sum(item["mean_abs_shap"] for item in global_importance) or 1.0
    for item in global_importance:
        item["share_pct"] = round((item["mean_abs_shap"] / total_abs) * 100.0, 2)

    # Dependence plots for key business drivers
    dependence_features = [
        "USD_NGN_Planning_Rate",
        "Import_Cost_NGN_Per_Unit",
        "Units_Sold",
        "Marketing_Spend_NGN",
        "Inflation_YoY_Pct",
        "Avg_Selling_Price_NGN",
    ]
    dependence_plots: Dict[str, List[Dict[str, Any]]] = {}
    for df_name in dependence_features:
        if df_name not in feature_cols:
            continue
        j = feature_cols.index(df_name)
        pts = []
        for i in range(len(X_full_df)):
            pts.append({
                "month": str(valid_df.iloc[i]["Month"]),
                "month_label": str(valid_df.iloc[i]["Month_Label"]),
                "x": round(float(X_full_df.iloc[i][df_name]), 4),
                "shap_value": round(float(shap_arr[i, j]), 2),
                "fx_rate": round(float(X_full_df.iloc[i]["USD_NGN_Planning_Rate"]), 2),
                "actual_profit": round(float(valid_df.iloc[i]["Operating_Profit_NGN"]), 2),
            })
        dependence_plots[df_name] = pts

    return {
        "available": True,
        "model_name": model_bundle["selected_model"]["name"],
        "base_value_ngn": round(base_val, 2),
        "global_importance": global_importance,
        "beeswarm_points": beeswarm_points,
        "dependence_plots": dependence_plots,
        "methodology_note": (
            "SHAP attributions measure how much each variable shifts the model's prediction relative to the baseline expectation. "
            "High SHAP importance reflects predictive association in the trained model—not standalone causal proof."
        ),
    }


def explain_single_prediction(
    model_bundle: Dict[str, Any],
    X_input_t: np.ndarray,
    X_input_df: pd.DataFrame,
) -> Dict[str, Any]:
    art = model_bundle["_artifacts"]
    full_est = art["full_estimator"]
    X_full_t = art["X_full_t"]
    is_tree = art["holdout"]["is_tree"]
    feature_cols = model_bundle["feature_cols"]

    shap_arr, base_val, _ = _compute_shap_matrix(full_est, X_full_t, X_input_t, is_tree)
    row_shap = shap_arr[0]
    predicted_val = float(full_est.predict(X_input_t)[0])

    contributions: List[Dict[str, Any]] = []
    for j, fname in enumerate(feature_cols):
        val = float(X_input_df.iloc[0][fname])
        contrib = float(row_shap[j])
        contributions.append({
            "feature": fname,
            "input_value": round(val, 4),
            "shap_contribution_ngn": round(contrib, 2),
            "abs_contribution": round(abs(contrib), 2),
            "direction": "increases" if contrib >= 0 else "decreases",
        })

    contributions.sort(key=lambda c: c["abs_contribution"], reverse=True)
    positive_drivers = [c for c in contributions if c["shap_contribution_ngn"] > 0]
    negative_drivers = [c for c in contributions if c["shap_contribution_ngn"] < 0]

    return {
        "baseline_prediction_ngn": round(base_val, 2),
        "final_predicted_operating_profit_ngn": round(predicted_val, 2),
        "shap_sum_check_ngn": round(base_val + float(np.sum(row_shap)), 2),
        "contributions": contributions,
        "top_increasing_drivers": positive_drivers[:4],
        "top_decreasing_drivers": negative_drivers[:4],
    }
