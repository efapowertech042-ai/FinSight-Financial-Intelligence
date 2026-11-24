from __future__ import annotations

import hashlib
import io
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from .anomaly import detect_anomalies_and_findings
    from .data_engineering import (
        compute_engineered_row,
        process_and_validate_dataset,
    )
    from .explainability import (
        compute_global_explainability,
        explain_single_prediction,
    )
    from .modeling import (
        DRIVER_FEATURES,
        EX_ANTE_FEATURES,
        predict_from_user_inputs,
        train_and_evaluate_suite,
    )
    from .rag_engine import run_graphical_rag_query
    from .forecasting_centre import (
        build_forecasting_centre_bundle,
        compute_accounting_5yr_simulation,
        compute_custom_scenario_verdict,
        compute_hypothesis_tests,
        compute_operational_verdict,
        compute_scenario_forecasts,
    )
except ImportError:
    from anomaly import detect_anomalies_and_findings
    from data_engineering import (
        compute_engineered_row,
        process_and_validate_dataset,
    )
    from explainability import (
        compute_global_explainability,
        explain_single_prediction,
    )
    from modeling import (
        DRIVER_FEATURES,
        EX_ANTE_FEATURES,
        predict_from_user_inputs,
        train_and_evaluate_suite,
    )
    from rag_engine import run_graphical_rag_query
    from forecasting_centre import (
        build_forecasting_centre_bundle,
        compute_accounting_5yr_simulation,
        compute_custom_scenario_verdict,
        compute_hypothesis_tests,
        compute_operational_verdict,
        compute_scenario_forecasts,
    )

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "default_financial_data.csv"

app = FastAPI(title="FinSight Financial Intelligence Python ML API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


import threading


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.dataset_name: str = "default_financial_data.csv"
        self.dataset_sha256: str = ""
        self.df: pd.DataFrame = pd.DataFrame()
        self.records: List[Dict[str, Any]] = []
        self.quality_report: Dict[str, Any] = {}
        self.feature_intelligence: List[Dict[str, Any]] = []
        self._driver_model_bundle: Optional[Dict[str, Any]] = None
        self._ex_ante_model_bundle: Optional[Dict[str, Any]] = None
        self._explainability: Optional[Dict[str, Any]] = None
        self._anomalies: Optional[Dict[str, Any]] = None
        self._forecasting_centre: Optional[Dict[str, Any]] = None

    def load_dataframe(self, raw_df: pd.DataFrame, dataset_name: str, raw_bytes: Optional[bytes] = None) -> None:
        with self._lock:
            processed = process_and_validate_dataset(raw_df)
            self.dataset_name = dataset_name
            if raw_bytes is not None:
                self.dataset_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            else:
                self.dataset_sha256 = hashlib.sha256(raw_df.to_csv(index=False).encode("utf-8")).hexdigest()
            self.df = processed["df"]
            self.records = processed["records"]
            self.quality_report = processed["quality_report"]
            self.feature_intelligence = processed["feature_intelligence"]
            # Clear cached ML/forecasting artifacts so they build lazily on demand
            self._driver_model_bundle = None
            self._ex_ante_model_bundle = None
            self._explainability = None
            self._anomalies = None
            self._forecasting_centre = None

    @property
    def driver_model_bundle(self) -> Dict[str, Any]:
        if self._driver_model_bundle is None:
            with self._lock:
                if self._driver_model_bundle is None and len(self.df) > 0:
                    self._driver_model_bundle = train_and_evaluate_suite(
                        self.df, DRIVER_FEATURES, "Driver-Based Regression (Concurrent Macro & Commercial Inputs)"
                    )
        return self._driver_model_bundle or {}

    @property
    def ex_ante_model_bundle(self) -> Dict[str, Any]:
        if self._ex_ante_model_bundle is None:
            with self._lock:
                if self._ex_ante_model_bundle is None and len(self.df) > 0:
                    self._ex_ante_model_bundle = train_and_evaluate_suite(
                        self.df, EX_ANTE_FEATURES, "Ex-Ante Pre-Period Forecast (Lagged Operations + Macro Plan)"
                    )
        return self._ex_ante_model_bundle or {}

    @property
    def explainability(self) -> Dict[str, Any]:
        if self._explainability is None:
            drv = self.driver_model_bundle
            with self._lock:
                if self._explainability is None:
                    self._explainability = compute_global_explainability(drv)
        return self._explainability or {}

    @property
    def anomalies(self) -> Dict[str, Any]:
        if self._anomalies is None:
            drv = self.driver_model_bundle
            with self._lock:
                if self._anomalies is None:
                    self._anomalies = detect_anomalies_and_findings(self.df, drv)
        return self._anomalies or {}

    @property
    def forecasting_centre(self) -> Dict[str, Any]:
        if self._forecasting_centre is None:
            with self._lock:
                if self._forecasting_centre is None and len(self.df) > 0:
                    self._forecasting_centre = build_forecasting_centre_bundle(self.df)
        return self._forecasting_centre or {}

    @forecasting_centre.setter
    def forecasting_centre(self, val: Dict[str, Any]) -> None:
        self._forecasting_centre = val


STATE = AppState()


def _init_default_dataset() -> None:
    if DEFAULT_DATA_PATH.exists():
        raw_bytes = DEFAULT_DATA_PATH.read_bytes()
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))
        STATE.load_dataframe(raw_df, "default_financial_data.csv", raw_bytes=raw_bytes)


_init_default_dataset()


def _filter_df(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    quarter: Optional[str] = None,
) -> pd.DataFrame:
    df = STATE.df.copy()
    if start_date:
        start_ym = start_date[:7]
        df = df[df["Month"].astype(str).str.slice(0, 7) >= start_ym]
    if end_date:
        end_ym = end_date[:7]
        df = df[df["Month"].astype(str).str.slice(0, 7) <= end_ym]
    if quarter and quarter.upper() != "ALL":
        df = df[df["Quarter"].str.upper() == quarter.upper()]
    return df.reset_index(drop=True)


def _compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 0:
        return {
            "total_sales_revenue": 0.0,
            "total_operating_profit": 0.0,
            "avg_operating_margin_pct": 0.0,
            "total_gross_profit": 0.0,
            "gross_margin_pct": 0.0,
            "total_units_sold": 0.0,
            "total_marketing_spend": 0.0,
            "total_import_costs": 0.0,
            "avg_usd_ngn_rate": 0.0,
            "avg_inflation_pct": 0.0,
            "period_count": 0,
            "deltas": {},
        }

    total_rev = float(df["Sales_Revenue_NGN"].sum())
    total_op = float(df["Operating_Profit_NGN"].sum())
    total_gp = float(df["Gross_Profit_NGN"].sum())
    total_cogs = float(df["COGS_NGN"].sum())
    total_units = float(df["Units_Sold"].sum())
    total_mkt = float(df["Marketing_Spend_NGN"].sum())
    total_other_op = float(df["Other_Operating_Cost_NGN"].sum())

    avg_op_margin = float(df["Operating_Margin_Pct"].mean())
    weighted_gm = (total_gp / total_rev * 100.0) if total_rev > 0 else 0.0
    weighted_om = (total_op / total_rev * 100.0) if total_rev > 0 else 0.0

    # Compare second half of filtered window vs first half for genuine period-over-period delta
    deltas: Dict[str, Optional[float]] = {}
    if len(df) >= 4:
        mid = len(df) // 2
        first = df.iloc[:mid]
        second = df.iloc[mid:]

        def _pct_delta(a: float, b: float) -> Optional[float]:
            if abs(a) < 1e-6:
                return None
            return round((b - a) / abs(a) * 100.0, 2)

        deltas = {
            "revenue_pct": _pct_delta(float(first["Sales_Revenue_NGN"].mean()), float(second["Sales_Revenue_NGN"].mean())),
            "operating_profit_pct": _pct_delta(float(first["Operating_Profit_NGN"].mean()), float(second["Operating_Profit_NGN"].mean())),
            "gross_profit_pct": _pct_delta(float(first["Gross_Profit_NGN"].mean()), float(second["Gross_Profit_NGN"].mean())),
            "units_pct": _pct_delta(float(first["Units_Sold"].mean()), float(second["Units_Sold"].mean())),
            "gross_margin_pp": round(float(second["Gross_Margin_Pct"].mean() - first["Gross_Margin_Pct"].mean()), 2),
            "operating_margin_pp": round(float(second["Operating_Margin_Pct"].mean() - first["Operating_Margin_Pct"].mean()), 2),
            "marketing_pct": _pct_delta(float(first["Marketing_Spend_NGN"].mean()), float(second["Marketing_Spend_NGN"].mean())),
            "import_cost_pct": _pct_delta(float(first["COGS_NGN"].mean()), float(second["COGS_NGN"].mean())),
        }

    return {
        "total_sales_revenue": round(total_rev, 2),
        "total_operating_profit": round(total_op, 2),
        "avg_operating_margin_pct": round(avg_op_margin, 2),
        "weighted_operating_margin_pct": round(weighted_om, 2),
        "total_gross_profit": round(total_gp, 2),
        "gross_margin_pct": round(weighted_gm, 2),
        "total_units_sold": round(total_units, 0),
        "total_marketing_spend": round(total_mkt, 2),
        "total_other_operating_cost": round(total_other_op, 2),
        "total_import_costs": round(total_cogs, 2),
        "avg_usd_ngn_rate": round(float(df["USD_NGN_Planning_Rate"].mean()), 2),
        "avg_inflation_pct": round(float(df["Inflation_YoY_Pct"].mean()), 2),
        "period_count": int(len(df)),
        "deltas": deltas,
    }


def _compute_quarterly_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if len(df) == 0:
        return []
    out = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df[df["Quarter"] == q]
        if len(sub) == 0:
            continue
        out.append({
            "quarter": q,
            "months_count": int(len(sub)),
            "total_revenue": round(float(sub["Sales_Revenue_NGN"].sum()), 2),
            "avg_revenue": round(float(sub["Sales_Revenue_NGN"].mean()), 2),
            "total_gross_profit": round(float(sub["Gross_Profit_NGN"].sum()), 2),
            "total_operating_profit": round(float(sub["Operating_Profit_NGN"].sum()), 2),
            "avg_operating_profit": round(float(sub["Operating_Profit_NGN"].mean()), 2),
            "avg_operating_margin_pct": round(float(sub["Operating_Margin_Pct"].mean()), 2),
            "avg_gross_margin_pct": round(float(sub["Gross_Margin_Pct"].mean()), 2),
            "total_units": int(sub["Units_Sold"].sum()),
            "total_marketing": round(float(sub["Marketing_Spend_NGN"].sum()), 2),
        })
    return out


@app.get("/api/py/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "dataset_name": STATE.dataset_name,
        "dataset_sha256_short": STATE.dataset_sha256[:12] if STATE.dataset_sha256 else "",
        "records": len(STATE.df),
    }


@app.get("/api/py/dataset/state")
def get_dataset_state(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    quarter: Optional[str] = Query(None),
) -> Dict[str, Any]:
    df = _filter_df(start_date, end_date, quarter)
    kpis = _compute_kpis(df)
    quarterly = _compute_quarterly_summary(df)

    # Financial performance distribution (Operating Profit histogram)
    dist_bins: List[Dict[str, Any]] = []
    if len(df) >= 3:
        vals = df["Operating_Profit_NGN"].dropna().values
        counts, edges = np.histogram(vals, bins=min(8, max(3, len(vals) // 8)))
        dist_bins = [
            {
                "range": f"₦{edges[i]/1e6:.1f}M–₦{edges[i+1]/1e6:.1f}M",
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]

    # Factor correlations with Operating Profit & Revenue
    factor_impacts = []
    for col, label in [
        ("USD_NGN_Planning_Rate", "FX Rate (USD/NGN)"),
        ("Inflation_YoY_Pct", "Inflation YoY %"),
        ("Marketing_Spend_NGN", "Marketing Spend"),
        ("Import_Cost_NGN_Per_Unit", "Import Cost (NGN/Unit)"),
        ("Units_Sold", "Units Sold"),
        ("Avg_Selling_Price_NGN", "Avg Selling Price"),
    ]:
        if len(df) > 2:
            c_op = float(df[[col, "Operating_Profit_NGN"]].corr().iloc[0, 1])
            c_rev = float(df[[col, "Sales_Revenue_NGN"]].corr().iloc[0, 1])
            factor_impacts.append({
                "factor": label,
                "column": col,
                "corr_operating_profit": round(0.0 if math.isnan(c_op) else c_op, 4),
                "corr_revenue": round(0.0 if math.isnan(c_rev) else c_rev, 4),
                "avg_value": round(float(df[col].mean()), 2),
            })

    return {
        "dataset_name": STATE.dataset_name,
        "available_date_range": STATE.quality_report.get("date_range"),
        "kpis": kpis,
        "quarterly_summary": quarterly,
        "profit_distribution": dist_bins,
        "factor_impacts": factor_impacts,
        "records": df.replace({np.nan: None}).to_dict(orient="records"),
        "quality_summary": {
            "records_processed": STATE.quality_report.get("records_processed"),
            "missing_values_total": STATE.quality_report.get("missing_values_total"),
            "duplicate_records": STATE.quality_report.get("duplicate_records"),
            "material_discrepancies_count": STATE.quality_report.get("material_discrepancies_count"),
            "rounding_variances_count": STATE.quality_report.get("rounding_variances_count"),
            "sha256_short": STATE.dataset_sha256[:12] if STATE.dataset_sha256 else "",
        },
        "latest_record": STATE.records[-1] if STATE.records else None,
    }


@app.post("/api/py/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    filename = file.filename or "uploaded_dataset"
    lower = filename.lower()
    content = await file.read()
    try:
        if lower.endswith(".csv"):
            raw_df = pd.read_csv(io.BytesIO(content))
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            raw_df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv or .xlsx Excel file.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to parse uploaded file: {exc}")

    try:
        STATE.load_dataframe(raw_df, filename, raw_bytes=content)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {exc}")

    return {
        "status": "uploaded",
        "dataset_name": STATE.dataset_name,
        "quality_report": STATE.quality_report,
    }


@app.post("/api/py/dataset/reset")
def reset_dataset() -> Dict[str, Any]:
    _init_default_dataset()
    return {
        "status": "reset",
        "dataset_name": STATE.dataset_name,
        "records_processed": STATE.quality_report.get("records_processed"),
    }


@app.get("/api/py/features/intelligence")
def get_feature_intelligence() -> Dict[str, Any]:
    dependency_graph = {
        "nodes": [
            {"id": "fx", "label": "USD_NGN_Planning_Rate", "category": "Macroeconomic", "formula": "Exogenous FX rate (₦/$)"},
            {"id": "imp_usd", "label": "Import_Cost_USD_Per_Unit", "category": "Import & Cost", "formula": "Supplier USD cost/unit"},
            {"id": "imp_ngn", "label": "Import_Cost_NGN_Per_Unit", "category": "Import & Cost", "formula": "USD_NGN × Import_Cost_USD"},
            {"id": "units", "label": "Units_Sold", "category": "Sales & Commercial", "formula": "Monthly sales volume"},
            {"id": "price", "label": "Avg_Selling_Price_NGN", "category": "Sales & Commercial", "formula": "Realized unit price (₦)"},
            {"id": "cogs", "label": "COGS_NGN", "category": "Import & Cost", "formula": "Units_Sold × Import_Cost_NGN"},
            {"id": "rev", "label": "Sales_Revenue_NGN", "category": "Financial Performance", "formula": "Units_Sold × Avg_Selling_Price"},
            {"id": "gp", "label": "Gross_Profit_NGN", "category": "Financial Performance", "formula": "Sales_Revenue − COGS_NGN"},
            {"id": "mkt", "label": "Marketing_Spend_NGN", "category": "Sales & Commercial", "formula": "Promotional expenditure"},
            {"id": "opex", "label": "Other_Operating_Cost_NGN", "category": "Import & Cost", "formula": "Fixed/semi-fixed overhead"},
            {"id": "op", "label": "Operating_Profit_NGN", "category": "Financial Performance", "formula": "Gross_Profit − Marketing − Other_OpEx"},
        ],
        "edges": [
            {"source": "fx", "target": "imp_ngn", "label": "×"},
            {"source": "imp_usd", "target": "imp_ngn", "label": "×"},
            {"source": "imp_ngn", "target": "cogs", "label": "× Units"},
            {"source": "units", "target": "cogs", "label": "× Unit Cost"},
            {"source": "units", "target": "rev", "label": "× Price"},
            {"source": "price", "target": "rev", "label": "× Units"},
            {"source": "rev", "target": "gp", "label": "+"},
            {"source": "cogs", "target": "gp", "label": "−"},
            {"source": "gp", "target": "op", "label": "+"},
            {"source": "mkt", "target": "op", "label": "−"},
            {"source": "opex", "target": "op", "label": "−"},
        ],
    }
    return {
        "features": STATE.feature_intelligence,
        "quality_report": STATE.quality_report,
        "dependency_graph": dependency_graph,
    }


def _strip_artifacts(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in bundle.items() if k != "_artifacts"}


@app.get("/api/py/models/summary")
def get_models_summary() -> Dict[str, Any]:
    return {
        "driver_suite": _strip_artifacts(STATE.driver_model_bundle),
        "ex_ante_suite": _strip_artifacts(STATE.ex_ante_model_bundle),
        "latest_record": STATE.records[-1] if STATE.records else None,
    }


class PredictionRequest(BaseModel):
    USD_NGN_Planning_Rate: float
    Inflation_YoY_Pct: float
    Units_Sold: float
    Import_Cost_USD_Per_Unit: float
    Avg_Selling_Price_NGN: float
    Marketing_Spend_NGN: float
    Other_Operating_Cost_NGN: float
    FX_Change_MoM_Pct: Optional[float] = None
    Quarter: str = "Q4"
    Month_Num: Optional[int] = None
    Market_Notes: Optional[str] = ""
    inflation_is_decimal: bool = False
    horizon_months: int = 1


@app.post("/api/py/models/predict")
def predict_operating_profit(req: PredictionRequest) -> Dict[str, Any]:
    if not STATE.driver_model_bundle.get("sufficient_data"):
        raise HTTPException(status_code=400, detail="Insufficient data to train ML prediction model.")

    last_row = STATE.records[-1] if STATE.records else {}
    pred_out = predict_from_user_inputs(STATE.driver_model_bundle, req.model_dump(), last_row)
    shap_local = explain_single_prediction(
        STATE.driver_model_bundle,
        pred_out["ml_prediction"]["X_input_t"],
        pred_out["ml_prediction"]["X_input_df"],
    )

    # Remove non-JSON numpy arrays before returning
    pred_clean = {
        k: v for k, v in pred_out["ml_prediction"].items() if k not in {"X_input_t", "X_input_df"}
    }

    recent_history = [
        {
            "month": r["Month"],
            "month_label": r["Month_Label"],
            "operating_profit_ngn": r["Operating_Profit_NGN"],
            "sales_revenue_ngn": r["Sales_Revenue_NGN"],
            "operating_margin_pct": r["Operating_Margin_Pct"],
        }
        for r in STATE.records[-12:]
    ]

    return {
        "ml_prediction": pred_clean,
        "accounting_reconciliation": pred_out["accounting_reconciliation"],
        "measurable_unit_economics": pred_out["measurable_unit_economics"],
        "measurable_sensitivities": pred_out["measurable_sensitivities"],
        "horizon_scaling": pred_out["horizon_scaling"],
        "local_explanation": shap_local,
        "recent_history": recent_history,
        "market_notes": req.Market_Notes or "",
    }


@app.get("/api/py/explainability")
def get_explainability() -> Dict[str, Any]:
    return STATE.explainability


@app.get("/api/py/anomalies")
def get_anomalies() -> Dict[str, Any]:
    return STATE.anomalies


class ScenarioRequest(BaseModel):
    baseline: PredictionRequest
    simulated: PredictionRequest


@app.post("/api/py/scenario/simulate")
def simulate_scenario(req: ScenarioRequest) -> Dict[str, Any]:
    last_row = STATE.records[-1] if STATE.records else {}
    base_res = predict_from_user_inputs(STATE.driver_model_bundle, req.baseline.model_dump(), last_row)
    sim_res = predict_from_user_inputs(STATE.driver_model_bundle, req.simulated.model_dump(), last_row)

    b_acc = base_res["accounting_reconciliation"]
    s_acc = sim_res["accounting_reconciliation"]
    b_ml = base_res["ml_prediction"]
    s_ml = sim_res["ml_prediction"]

    # Waterfall bridge from Baseline Accounting Operating Profit to Simulated Accounting Operating Profit
    # Decompose Revenue change into Volume effect + Price effect, COGS change, Marketing change, Other OpEx change
    b_units = req.baseline.Units_Sold
    s_units = req.simulated.Units_Sold
    b_price = req.baseline.Avg_Selling_Price_NGN
    s_price = req.simulated.Avg_Selling_Price_NGN

    rev_volume_effect = (s_units - b_units) * b_price
    rev_price_effect = s_units * (s_price - b_price)
    cogs_effect = -(s_acc["COGS_NGN"] - b_acc["COGS_NGN"])
    mkt_effect = -(s_acc["Marketing_Spend_NGN"] - b_acc["Marketing_Spend_NGN"])
    opex_effect = -(s_acc["Other_Operating_Cost_NGN"] - b_acc["Other_Operating_Cost_NGN"])

    waterfall = [
        {"step": "Baseline Operating Profit", "value": round(b_acc["Accounting_Operating_Profit_NGN"], 2), "type": "baseline"},
        {"step": "Volume Impact (Units)", "value": round(rev_volume_effect, 2), "type": "delta"},
        {"step": "Pricing Impact (Price)", "value": round(rev_price_effect, 2), "type": "delta"},
        {"step": "COGS Impact (FX & Import)", "value": round(cogs_effect, 2), "type": "delta"},
        {"step": "Marketing Spend Impact", "value": round(mkt_effect, 2), "type": "delta"},
        {"step": "Other OpEx Impact", "value": round(opex_effect, 2), "type": "delta"},
        {"step": "Simulated Operating Profit", "value": round(s_acc["Accounting_Operating_Profit_NGN"], 2), "type": "total"},
    ]

    return {
        "baseline": {
            "accounting": b_acc,
            "ml_predicted_operating_profit_ngn": b_ml["predicted_operating_profit_ngn"],
            "ml_predicted_operating_margin_pct": b_ml["predicted_operating_margin_pct"],
        },
        "simulated": {
            "accounting": s_acc,
            "ml_predicted_operating_profit_ngn": s_ml["predicted_operating_profit_ngn"],
            "ml_predicted_operating_margin_pct": s_ml["predicted_operating_margin_pct"],
        },
        "deltas": {
            "revenue_delta": round(s_acc["Sales_Revenue_NGN"] - b_acc["Sales_Revenue_NGN"], 2),
            "cogs_delta": round(s_acc["COGS_NGN"] - b_acc["COGS_NGN"], 2),
            "gross_profit_delta": round(s_acc["Gross_Profit_NGN"] - b_acc["Gross_Profit_NGN"], 2),
            "accounting_profit_delta": round(s_acc["Accounting_Operating_Profit_NGN"] - b_acc["Accounting_Operating_Profit_NGN"], 2),
            "ml_predicted_profit_delta": round(s_ml["predicted_operating_profit_ngn"] - b_ml["predicted_operating_profit_ngn"], 2),
            "gross_margin_pp_delta": round(s_acc["Gross_Margin_Pct"] - b_acc["Gross_Margin_Pct"], 2),
            "operating_margin_pp_delta": round(s_acc["Accounting_Operating_Margin_Pct"] - b_acc["Accounting_Operating_Margin_Pct"], 2),
        },
        "waterfall": waterfall,
    }


class RagQueryRequest(BaseModel):
    question: str = ""
    lens: str = "profit_intelligence"
    selected_month: Optional[str] = None


@app.post("/api/py/rag/query")
def rag_query(req: RagQueryRequest) -> Dict[str, Any]:
    return run_graphical_rag_query(
        processed_df=STATE.df,
        model_bundle=STATE.driver_model_bundle,
        explainability_bundle=STATE.explainability,
        anomaly_bundle=STATE.anomalies,
        quality_report=STATE.quality_report,
        question=req.question,
        lens=req.lens,
        selected_month=req.selected_month,
        forecasting_bundle=STATE.forecasting_centre,
        ex_ante_bundle=STATE.ex_ante_model_bundle,
        dataset_name=STATE.dataset_name,
    )


@app.post("/api/rag/ask")
def rag_ask(req: RagQueryRequest) -> Dict[str, Any]:
    retrieved = run_graphical_rag_query(
        processed_df=STATE.df,
        model_bundle=STATE.driver_model_bundle,
        explainability_bundle=STATE.explainability,
        anomaly_bundle=STATE.anomalies,
        quality_report=STATE.quality_report,
        question=req.question,
        lens=req.lens,
        selected_month=req.selected_month,
        forecasting_bundle=STATE.forecasting_centre,
        ex_ante_bundle=STATE.ex_ante_model_bundle,
        dataset_name=STATE.dataset_name,
    )

    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    llm_synthesis: Optional[str] = None
    llm_grounded = False
    synthesis_mode = "Deterministic Retrieval-Grounded Explanation"

    if api_key and api_key != "MY_GEMINI_API_KEY" and len(api_key) > 10:
        try:
            grounded_prompt = "\n\n".join([
                "You are a senior financial intelligence analyst.",
                "Answer the user's analytical question strictly using ONLY the retrieved evidence, ledger records, and retrieval process trace below.",
                "Do NOT invent any numbers, months, or findings not present in the retrieved evidence.",
                "Provide 3-4 concise sentences explaining both the exact quantitative figures retrieved from the dataset and the decision implication.",
                f"User Question: {retrieved.get('question')}",
                f"Data Provenance: {json.dumps(retrieved.get('data_provenance'))}",
                f"Matched Dataset Ledger Rows: {json.dumps(retrieved.get('matched_records'))}",
                f"Retrieval Process Trace: {json.dumps(retrieved.get('retrieval_process'))}",
                f"Retrieved Evidence: {json.dumps(retrieved.get('evidence'))}",
                f"Baseline Summary: {retrieved.get('analytical_explanation')}",
                f"Business Implications: {retrieved.get('business_implications')}",
            ])
            payload = json.dumps({
                "contents": [{"parts": [{"text": grounded_prompt}]}]
            }).encode("utf-8")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            http_req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(http_req, timeout=15) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            candidates = resp_data.get("candidates") or []
            if candidates:
                parts = candidates[0].get("content", {}).get("parts") or []
                text_out = "".join(str(p.get("text", "")) for p in parts).strip()
                if text_out:
                    llm_synthesis = text_out
                    llm_grounded = True
                    synthesis_mode = "Retrieval-Grounded Gemini Synthesis"
        except Exception:
            pass

    return {
        **retrieved,
        "synthesis_mode": synthesis_mode,
        "llm_grounded": llm_grounded,
        "llm_narrative": llm_synthesis,
    }


@app.get("/api/py/monitoring")
def get_model_monitoring() -> Dict[str, Any]:
    sel = STATE.driver_model_bundle.get("selected_model") or {}
    test_preds = sel.get("test_predictions", [])
    train_preds = sel.get("train_predictions", [])

    # Compute feature drift between first 70% (reference) and last 30% (recent) using standardized mean shift (PSI-style z-shift)
    df = STATE.df
    drift_metrics: List[Dict[str, Any]] = []
    if len(df) >= 20:
        cut = int(len(df) * 0.7)
        ref_df = df.iloc[:cut]
        cur_df = df.iloc[cut:]
        for col in [
            "USD_NGN_Planning_Rate",
            "Inflation_YoY_Pct",
            "Units_Sold",
            "Import_Cost_USD_Per_Unit",
            "Avg_Selling_Price_NGN",
            "Marketing_Spend_NGN",
            "Other_Operating_Cost_NGN",
            "Import_Cost_NGN_Per_Unit",
        ]:
            ref_mean = float(ref_df[col].mean())
            ref_std = float(ref_df[col].std(ddof=1)) or 1.0
            cur_mean = float(cur_df[col].mean())
            z_shift = abs(cur_mean - ref_mean) / ref_std
            status = "High Drift" if z_shift >= 1.5 else "Moderate Drift" if z_shift >= 0.75 else "Stable"
            drift_metrics.append({
                "feature": col,
                "reference_mean": round(ref_mean, 2),
                "recent_mean": round(cur_mean, 2),
                "pct_change": round(((cur_mean - ref_mean) / abs(ref_mean) * 100.0) if abs(ref_mean) > 1e-6 else 0.0, 2),
                "standardized_shift": round(z_shift, 3),
                "status": status,
            })

    # Residual trend check
    test_res = [abs(p["residual"]) for p in test_preds]
    mean_test_abs_err = float(np.mean(test_res)) if test_res else 0.0
    retraining_needed = any(d["status"] == "High Drift" for d in drift_metrics) or (
        (sel.get("test_metrics", {}).get("r2") or 0) < 0.75
    )

    return {
        "selected_model": sel.get("name"),
        "train_metrics": sel.get("train_metrics"),
        "test_metrics": sel.get("test_metrics"),
        "residual_series": train_preds + test_preds,
        "feature_drift": drift_metrics,
        "data_quality": {
            "records_processed": STATE.quality_report.get("records_processed"),
            "missing_values_total": STATE.quality_report.get("missing_values_total"),
            "material_discrepancies": STATE.quality_report.get("material_discrepancies_count"),
            "duplicate_records": STATE.quality_report.get("duplicate_records"),
        },
        "retraining_assessment": {
            "recommended": retraining_needed,
            "mean_recent_abs_error_ngn": round(mean_test_abs_err, 2),
            "rationale": (
                "Significant structural feature shift detected between early historical window and recent periods (e.g., FX & pricing regime shift). Schedule periodic chronological retraining as new months close."
                if retraining_needed
                else "Feature distributions and out-of-time residuals are within nominal thresholds."
            ),
        },
    }


@app.get("/api/py/export/csv")
def export_processed_csv() -> StreamingResponse:
    buf = io.StringIO()
    STATE.df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=finsight_processed_{STATE.dataset_name.replace('.xlsx', '.csv')}"},
    )


class ForecastHypothesisRequest(BaseModel):
    alpha: float = 0.05
    split_year: int = 2023


class ForecastScenarioRequest(BaseModel):
    revenue_growth_adj_pp: float = 0.0
    opex_growth_adj_pp: float = 0.0
    margin_efficiency_adj_pp: float = 0.0
    fx_cost_pressure_adj_pct: float = 0.0
    volume_productivity_adj_pct: float = 0.0


@app.get("/api/py/forecasting/centre")
def get_forecasting_centre() -> Dict[str, Any]:
    if not STATE.forecasting_centre:
        STATE.forecasting_centre = build_forecasting_centre_bundle(STATE.df)
    return STATE.forecasting_centre


@app.post("/api/py/forecasting/hypotheses")
def run_forecast_hypotheses(req: ForecastHypothesisRequest) -> Dict[str, Any]:
    alpha = max(0.001, min(0.25, float(req.alpha)))
    split_year = int(req.split_year)
    holdout_series = STATE.forecasting_centre.get("validation", {}).get("holdout_series", [])
    if holdout_series:
        act = np.asarray([float(r["actual"]) for r in holdout_series], dtype=float)
        pred = np.asarray([float(r["predicted"]) for r in holdout_series], dtype=float)
        naive = np.asarray([float(r["naive"]) for r in holdout_series], dtype=float)
    else:
        act = pred = naive = None
    return compute_hypothesis_tests(
        df=STATE.df,
        champion_holdout_actual=act,
        champion_holdout_pred=pred,
        naive_holdout_pred=naive,
        alpha=alpha,
        split_year=split_year,
    )


@app.post("/api/py/forecasting/scenario")
def simulate_forecast_scenario(req: ForecastScenarioRequest) -> Dict[str, Any]:
    if not STATE.forecasting_centre or not STATE.forecasting_centre.get("sufficient_data"):
        raise HTTPException(status_code=400, detail="Insufficient data for 2030 scenario simulation.")
    scenarios = compute_scenario_forecasts(STATE.forecasting_centre, req.model_dump())
    verdict = compute_operational_verdict(STATE.forecasting_centre, scenarios)
    custom_verdict = compute_custom_scenario_verdict(STATE.forecasting_centre, scenarios)
    return {
        "scenarios": scenarios,
        "verdict": verdict,
        "custom_verdict": custom_verdict,
    }


@app.get("/api/py/forecasting/export/csv")
def export_forecast_csv() -> StreamingResponse:
    fc = STATE.forecasting_centre
    rows = (fc.get("historical_months") or []) + (fc.get("forecast_months") or [])
    export_df = pd.DataFrame(rows)
    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=finsight_2030_forecast_timeline.csv"},
    )


@app.get("/api/py/forecasting/export/validation-csv")
def export_validation_csv() -> StreamingResponse:
    fc = STATE.forecasting_centre
    rows = fc.get("leaderboard") or []
    export_df = pd.DataFrame(rows)
    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=finsight_2030_model_validation.csv"},
    )


class Accounting5YrScenarioRequest(BaseModel):
    annual_inflation_pct: Optional[float] = None
    unit_sales_growth_pct: Optional[float] = None
    price_inflation_pass_through_pct: Optional[float] = None
    fx_cost_pass_through_pct: Optional[float] = None
    usd_import_cost_growth_pct: Optional[float] = None
    opex_inflation_sensitivity_pct: Optional[float] = None
    fx_inflation_sensitivity_x: Optional[float] = None
    horizon_months: int = 60


@app.post("/api/py/scenario/accounting-5yr")
def run_accounting_5yr_scenario(req: Accounting5YrScenarioRequest) -> Dict[str, Any]:
    horizon = max(12, min(72, int(req.horizon_months)))
    try:
        return compute_accounting_5yr_simulation(
            STATE.df,
            user_params=req.model_dump(exclude={"horizon_months"}),
            horizon_months=horizon,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.get("/api/py/scenario/accounting-5yr/export")
def export_accounting_5yr_csv(
    annual_inflation_pct: Optional[float] = Query(None),
    unit_sales_growth_pct: Optional[float] = Query(None),
    price_inflation_pass_through_pct: Optional[float] = Query(None),
    fx_cost_pass_through_pct: Optional[float] = Query(None),
    usd_import_cost_growth_pct: Optional[float] = Query(None),
    opex_inflation_sensitivity_pct: Optional[float] = Query(None),
    fx_inflation_sensitivity_x: Optional[float] = Query(None),
    horizon_months: int = Query(60),
) -> StreamingResponse:
    res = compute_accounting_5yr_simulation(
        STATE.df,
        user_params={
            "annual_inflation_pct": annual_inflation_pct,
            "unit_sales_growth_pct": unit_sales_growth_pct,
            "price_inflation_pass_through_pct": price_inflation_pass_through_pct,
            "fx_cost_pass_through_pct": fx_cost_pass_through_pct,
            "usd_import_cost_growth_pct": usd_import_cost_growth_pct,
            "opex_inflation_sensitivity_pct": opex_inflation_sensitivity_pct,
            "fx_inflation_sensitivity_x": fx_inflation_sensitivity_x,
        },
        horizon_months=max(12, min(72, int(horizon_months))),
    )
    export_df = pd.DataFrame(res["monthly_series"])
    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=finsight_5yr_accounting_scenario.csv"},
    )

