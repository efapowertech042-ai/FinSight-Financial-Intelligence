from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "Month",
    "USD_NGN_Planning_Rate",
    "Inflation_YoY_Pct",
    "Units_Sold",
    "Import_Cost_USD_Per_Unit",
    "Avg_Selling_Price_NGN",
    "Marketing_Spend_NGN",
    "Other_Operating_Cost_NGN",
    "FX_Change_MoM_Pct",
    "Import_Cost_NGN_Per_Unit",
    "COGS_NGN",
    "Sales_Revenue_NGN",
    "Gross_Profit_NGN",
    "Gross_Margin_Pct",
    "Operating_Profit_NGN",
    "Operating_Margin_Pct",
    "Quarter",
    "Market_Notes",
]

REQUIRED_INPUT_COLUMNS = [
    "Month",
    "USD_NGN_Planning_Rate",
    "Inflation_YoY_Pct",
    "Units_Sold",
    "Import_Cost_USD_Per_Unit",
    "Avg_Selling_Price_NGN",
    "Marketing_Spend_NGN",
    "Other_Operating_Cost_NGN",
]

FEATURE_METADATA: List[Dict[str, Any]] = [
    {
        "name": "USD_NGN_Planning_Rate",
        "category": "Macroeconomic",
        "data_type": "Float (NGN / USD)",
        "formula": "Exogenous macro input (Planning exchange rate)",
        "business_definition": "Budgeted or prevailing NGN per 1 USD exchange rate used for import procurement costing.",
        "mathematical_role": "Direct multiplier on USD import unit cost; drives local unit cost and COGS.",
        "financial_interpretation": "Higher USD/NGN rates immediately inflate landed import cost per unit in Naira.",
        "profitability_influence": "Typically compresses margins unless selling prices are raised proportionally.",
        "forecast_availability": "Known / Assumed prior to period close (planning rate).",
    },
    {
        "name": "Inflation_YoY_Pct",
        "category": "Macroeconomic",
        "data_type": "Float (%)",
        "formula": "Exogenous macro input (Year-over-Year CPI %)",
        "business_definition": "Annualized consumer price inflation rate impacting domestic operating costs and demand elasticity.",
        "mathematical_role": "Macroeconomic covariate affecting pricing power, operating overhead, and volume.",
        "financial_interpretation": "Elevated inflation raises operating expenses and can dampen real unit demand.",
        "profitability_influence": "Mixed: coincides with pricing adjustments but erodes purchasing power and overhead efficiency.",
        "forecast_availability": "Lagged macro release / planning assumption.",
    },
    {
        "name": "FX_Change_MoM_Pct",
        "category": "Macroeconomic",
        "data_type": "Float (%)",
        "formula": "((USD_NGN_t - USD_NGN_{t-1}) / USD_NGN_{t-1}) * 100",
        "business_definition": "Month-over-month percentage depreciation or appreciation of Naira against USD.",
        "mathematical_role": "Derived rate-of-change feature capturing currency shock velocity.",
        "financial_interpretation": "Sharp positive spikes indicate sudden Naira depreciation shocks (e.g., June 2023).",
        "profitability_influence": "Abrupt FX shocks cause immediate margin compression before prices re-adjust.",
        "forecast_availability": "Derivable from planning FX rate vs prior month actual FX rate.",
    },
    {
        "name": "Units_Sold",
        "category": "Sales & Commercial",
        "data_type": "Integer / Float (Units)",
        "formula": "Primary commercial volume driver",
        "business_definition": "Total physical units sold within the calendar month.",
        "mathematical_role": "Volume multiplier for both Sales Revenue and COGS.",
        "financial_interpretation": "Captures seasonal peaks (Q4) and macro/pandemic demand slowdowns.",
        "profitability_influence": "Strong positive lever on operating profit whenever unit contribution margin is positive.",
        "forecast_availability": "Unknown ex-ante; must be forecasted or planned for scenario analysis.",
    },
    {
        "name": "Avg_Selling_Price_NGN",
        "category": "Sales & Commercial",
        "data_type": "Float (NGN / Unit)",
        "formula": "Primary commercial pricing driver",
        "business_definition": "Average realized selling price per unit in Nigerian Naira.",
        "mathematical_role": "Unit revenue multiplier in Sales_Revenue_NGN = Units_Sold * Avg_Selling_Price_NGN.",
        "financial_interpretation": "Reflects pricing pass-through strategy during FX and inflationary cycles.",
        "profitability_influence": "Direct 100% pass-through to gross and operating profit per unit sold.",
        "forecast_availability": "Controllable commercial lever / price list assumption.",
    },
    {
        "name": "Marketing_Spend_NGN",
        "category": "Sales & Commercial",
        "data_type": "Float (NGN)",
        "formula": "Operating expense commercial input",
        "business_definition": "Monthly promotional, advertising, and customer acquisition expenditure in NGN.",
        "mathematical_role": "Direct deduction from Gross Profit in Operating Profit equation.",
        "financial_interpretation": "Shows seasonal Q4 surges; must be evaluated against incremental gross profit.",
        "profitability_influence": "Reduces operating profit dollar-for-dollar unless offset by incremental unit volume.",
        "forecast_availability": "Budgeted / controllable pre-period input.",
    },
    {
        "name": "Import_Cost_USD_Per_Unit",
        "category": "Import & Cost",
        "data_type": "Float (USD / Unit)",
        "formula": "Primary supplier USD unit cost",
        "business_definition": "Landed unit procurement cost denominated in US Dollars.",
        "mathematical_role": "Multiplied by USD_NGN_Planning_Rate to compute local unit cost.",
        "financial_interpretation": "Tracks international supplier pricing, freight, and tariff dynamics.",
        "profitability_influence": "Increases unit COGS; magnifies FX depreciation impact.",
        "forecast_availability": "Known from supplier purchase orders / contracts.",
    },
    {
        "name": "Import_Cost_NGN_Per_Unit",
        "category": "Import & Cost",
        "data_type": "Float (NGN / Unit)",
        "formula": "USD_NGN_Planning_Rate * Import_Cost_USD_Per_Unit",
        "business_definition": "Landed unit cost converted into Nigerian Naira at the planning exchange rate.",
        "mathematical_role": "Engineered intermediate feature feeding total COGS.",
        "financial_interpretation": "Measures unit cost pressure in local reporting currency.",
        "profitability_influence": "Higher local unit cost narrows the unit spread (Price - Import Cost NGN).",
        "forecast_availability": "Calculable from planning FX rate and USD unit cost.",
    },
    {
        "name": "COGS_NGN",
        "category": "Import & Cost",
        "data_type": "Float (NGN)",
        "formula": "Units_Sold * Import_Cost_NGN_Per_Unit",
        "business_definition": "Total monthly Cost of Goods Sold in Naira.",
        "mathematical_role": "Engineered aggregate cost deducted from Sales Revenue to derive Gross Profit.",
        "financial_interpretation": "Scales linearly with both volume and FX-adjusted unit cost.",
        "profitability_influence": "Primary cost component; excluded from ML features when preventing accounting leakage.",
        "forecast_availability": "Realized only after Units_Sold is known (excluded from pure ex-ante ML).",
    },
    {
        "name": "Other_Operating_Cost_NGN",
        "category": "Import & Cost",
        "data_type": "Float (NGN)",
        "formula": "Fixed/semi-fixed overhead operating input",
        "business_definition": "Monthly administrative, logistics, rent, and general operating overhead in NGN.",
        "mathematical_role": "Direct deduction from Gross Profit in Operating Profit equation.",
        "financial_interpretation": "Tends to drift upward with inflation and FX-linked local costs.",
        "profitability_influence": "Creates operating leverage risk when sales volume drops.",
        "forecast_availability": "Largely predictable / budgeted pre-period.",
    },
    {
        "name": "Sales_Revenue_NGN",
        "category": "Financial Performance",
        "data_type": "Float (NGN)",
        "formula": "Units_Sold * Avg_Selling_Price_NGN",
        "business_definition": "Total monthly top-line sales revenue in Nigerian Naira.",
        "mathematical_role": "Engineered top-line figure; denominator for Gross and Operating Margin.",
        "financial_interpretation": "Combines volume growth and price inflation effects.",
        "profitability_influence": "Excluded from ML predictors to prevent trivial target reconstruction.",
        "forecast_availability": "Realized at period end (derived from Units_Sold and Price).",
    },
    {
        "name": "Gross_Profit_NGN",
        "category": "Financial Performance",
        "data_type": "Float (NGN)",
        "formula": "Sales_Revenue_NGN - COGS_NGN",
        "business_definition": "Monthly profit remaining after direct import cost of goods sold.",
        "mathematical_role": "Engineered intermediate profit metric before OpEx.",
        "financial_interpretation": "Captures core commercial spread before marketing and overhead.",
        "profitability_influence": "Excluded from ML predictors to prevent target leakage.",
        "forecast_availability": "Realized at period end.",
    },
    {
        "name": "Gross_Margin_Pct",
        "category": "Financial Performance",
        "data_type": "Float (%)",
        "formula": "(Gross_Profit_NGN / Sales_Revenue_NGN) * 100",
        "business_definition": "Percentage of sales revenue retained after COGS.",
        "mathematical_role": "Unit-economics efficiency ratio independent of scale.",
        "financial_interpretation": "Drops sharply when FX surges outpace selling price adjustments.",
        "profitability_influence": "Key structural health metric; excluded from ML predictors to prevent leakage.",
        "forecast_availability": "Realized at period end.",
    },
    {
        "name": "Operating_Profit_NGN",
        "category": "Financial Performance",
        "data_type": "Float (NGN) [Target]",
        "formula": "Gross_Profit_NGN - Marketing_Spend_NGN - Other_Operating_Cost_NGN",
        "business_definition": "Bottom-line monthly operating income (EBIT proxy) in NGN.",
        "mathematical_role": "Primary ML prediction target variable.",
        "financial_interpretation": "Measures net operational profitability after COGS, marketing, and overhead.",
        "profitability_influence": "Target variable.",
        "forecast_availability": "Target variable realized at period close.",
    },
    {
        "name": "Operating_Margin_Pct",
        "category": "Financial Performance",
        "data_type": "Float (%)",
        "formula": "(Operating_Profit_NGN / Sales_Revenue_NGN) * 100",
        "business_definition": "Operating profit expressed as a percentage of total sales revenue.",
        "mathematical_role": "Target-derived profitability ratio (NEVER used as ML predictor).",
        "financial_interpretation": "Indicates operating leverage and net margin resilience.",
        "profitability_influence": "Strictly prohibited from ML predictors (direct target leakage).",
        "forecast_availability": "Realized at period end; strictly excluded from ML inputs.",
    },
    {
        "name": "Month",
        "category": "Temporal & Contextual",
        "data_type": "Date (YYYY-MM-DD)",
        "formula": "Chronological period index",
        "business_definition": "First calendar day of the monthly reporting period.",
        "mathematical_role": "Enforces chronological ordering for MoM FX changes and time-series validation.",
        "financial_interpretation": "Prevents future data leakage during train/test splitting.",
        "profitability_influence": "Captures secular trend and cyclical seasonality via month-of-year.",
        "forecast_availability": "Always known prior to period.",
    },
    {
        "name": "Quarter",
        "category": "Temporal & Contextual",
        "data_type": "Categorical (Q1-Q4)",
        "formula": "Derived from Month if missing (Q1..Q4)",
        "business_definition": "Fiscal/calendar quarter label (Q1, Q2, Q3, Q4).",
        "mathematical_role": "Seasonal categorical feature capturing quarterly demand and promo cycles.",
        "financial_interpretation": "Q4 typically reflects heightened holiday demand and marketing spend.",
        "profitability_influence": "Captures recurring intra-year seasonality.",
        "forecast_availability": "Always known prior to period.",
    },
    {
        "name": "Market_Notes",
        "category": "Temporal & Contextual",
        "data_type": "Text (Qualitative)",
        "formula": "Contextual analyst annotation",
        "business_definition": "Qualitative notes documenting macro shocks, regime changes, or disruptions.",
        "mathematical_role": "Contextual evidence for Graphical RAG and anomaly interpretation.",
        "financial_interpretation": "Explains structural breaks (e.g., June 2023 FX market regime change).",
        "profitability_influence": "Provides qualitative explanation for outlier residuals.",
        "forecast_availability": "Recorded concurrently or retrospectively.",
    },
]


def _safe_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if pd.isna(val):
        return None
    try:
        s = str(val).strip().replace(",", "").replace("₦", "").replace("$", "")
        if s == "" or s.lower() in {"nan", "null", "none", "n/a", "-"}:
            return None
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    except Exception:
        return None


def _derive_quarter(dt: pd.Timestamp) -> str:
    return f"Q{((dt.month - 1) // 3) + 1}"


def compute_engineered_row(
    usd_ngn_rate: float,
    import_cost_usd: float,
    units_sold: float,
    avg_selling_price: float,
    marketing_spend: float,
    other_operating_cost: float,
    prev_usd_ngn_rate: Optional[float] = None,
    supplied_fx_change_pct: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Reusable hardcoded feature engineering formulas for a single observation."""
    import_cost_ngn_per_unit = usd_ngn_rate * import_cost_usd
    cogs_ngn = units_sold * import_cost_ngn_per_unit
    sales_revenue_ngn = units_sold * avg_selling_price
    gross_profit_ngn = sales_revenue_ngn - cogs_ngn
    gross_margin_pct = (
        (gross_profit_ngn / sales_revenue_ngn) * 100.0
        if sales_revenue_ngn and sales_revenue_ngn != 0
        else None
    )
    operating_profit_ngn = gross_profit_ngn - marketing_spend - other_operating_cost
    operating_margin_pct = (
        (operating_profit_ngn / sales_revenue_ngn) * 100.0
        if sales_revenue_ngn and sales_revenue_ngn != 0
        else None
    )
    if prev_usd_ngn_rate is not None and prev_usd_ngn_rate != 0:
        fx_change_mom_pct = ((usd_ngn_rate - prev_usd_ngn_rate) / prev_usd_ngn_rate) * 100.0
    else:
        fx_change_mom_pct = supplied_fx_change_pct

    return {
        "Import_Cost_NGN_Per_Unit": import_cost_ngn_per_unit,
        "COGS_NGN": cogs_ngn,
        "Sales_Revenue_NGN": sales_revenue_ngn,
        "Gross_Profit_NGN": gross_profit_ngn,
        "Gross_Margin_Pct": gross_margin_pct,
        "Operating_Profit_NGN": operating_profit_ngn,
        "Operating_Margin_Pct": operating_margin_pct,
        "FX_Change_MoM_Pct": fx_change_mom_pct,
    }


def process_and_validate_dataset(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates schema, enforces chronological order, recalculates all engineered features
    from underlying inputs, checks supplied vs recalculated discrepancies, and produces
    a comprehensive data quality & feature intelligence report.
    """
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    present_cols = list(df.columns)
    missing_required = [c for c in REQUIRED_INPUT_COLUMNS if c not in present_cols]
    missing_optional = [c for c in EXPECTED_COLUMNS if c not in present_cols and c not in REQUIRED_INPUT_COLUMNS]
    unexpected_cols = [c for c in present_cols if c not in EXPECTED_COLUMNS]

    if missing_required:
        raise ValueError(
            f"Missing required underlying input columns: {', '.join(missing_required)}"
        )

    # Parse Month
    df["_Month_Raw"] = df["Month"].astype(str)
    df["_Month_Parsed"] = pd.to_datetime(df["Month"], errors="coerce")
    invalid_dates_count = int(df["_Month_Parsed"].isna().sum())

    # Coerce numeric columns
    numeric_input_cols = [
        "USD_NGN_Planning_Rate",
        "Inflation_YoY_Pct",
        "Units_Sold",
        "Import_Cost_USD_Per_Unit",
        "Avg_Selling_Price_NGN",
        "Marketing_Spend_NGN",
        "Other_Operating_Cost_NGN",
    ]
    supplied_calc_cols = [
        "FX_Change_MoM_Pct",
        "Import_Cost_NGN_Per_Unit",
        "COGS_NGN",
        "Sales_Revenue_NGN",
        "Gross_Profit_NGN",
        "Gross_Margin_Pct",
        "Operating_Profit_NGN",
        "Operating_Margin_Pct",
    ]

    missing_values_by_col: Dict[str, int] = {}
    invalid_numeric_issues: List[Dict[str, Any]] = []

    for col in numeric_input_cols + [c for c in supplied_calc_cols if c in df.columns]:
        converted = df[col].apply(_safe_float)
        raw_non_null = df[col].notna() & (df[col].astype(str).str.strip() != "")
        bad_mask = raw_non_null & converted.isna()
        if bad_mask.any():
            for idx in df[bad_mask].index:
                invalid_numeric_issues.append({
                    "row": int(idx) + 1,
                    "month": str(df.loc[idx, "_Month_Raw"]),
                    "column": col,
                    "raw_value": str(df.loc[idx, col]),
                })
        df[col] = converted.astype(float)
        missing_values_by_col[col] = int(df[col].isna().sum())

    if "Market_Notes" not in df.columns:
        df["Market_Notes"] = ""
    df["Market_Notes"] = df["Market_Notes"].fillna("").astype(str).str.strip()

    # Check duplicate months before dedup/sorting
    valid_dates = df["_Month_Parsed"].dropna()
    duplicate_months_series = valid_dates[valid_dates.duplicated(keep=False)]
    duplicate_months = sorted({d.strftime("%Y-%m-%d") for d in duplicate_months_series})
    duplicate_records_count = int(valid_dates.duplicated().sum())

    # Chronological sorting
    df_clean = df[df["_Month_Parsed"].notna()].copy()
    df_clean = df_clean.sort_values("_Month_Parsed", kind="mergesort").reset_index(drop=True)

    # Detect missing monthly periods in sequence
    missing_periods: List[str] = []
    if len(df_clean) >= 2:
        unique_months = sorted(df_clean["_Month_Parsed"].dt.to_period("M").unique())
        for i in range(1, len(unique_months)):
            diff = (unique_months[i] - unique_months[i - 1]).n
            if diff > 1:
                for step in range(1, diff):
                    missing_periods.append(str(unique_months[i - 1] + step))

    # Derive Quarter if missing
    if "Quarter" not in df_clean.columns:
        df_clean["Quarter"] = df_clean["_Month_Parsed"].apply(_derive_quarter)
    else:
        df_clean["Quarter"] = df_clean.apply(
            lambda r: str(r["Quarter"]).strip()
            if pd.notna(r["Quarter"]) and str(r["Quarter"]).strip() != ""
            else _derive_quarter(r["_Month_Parsed"]),
            axis=1,
        )

    # Detect scale convention of supplied percentage columns (decimal ratio e.g. 0.3783 vs pct e.g. 37.83)
    pct_scale_info: Dict[str, str] = {}
    for col in ["Inflation_YoY_Pct", "FX_Change_MoM_Pct", "Gross_Margin_Pct", "Operating_Margin_Pct"]:
        if col in df_clean.columns and df_clean[col].dropna().shape[0] > 0:
            med_abs = float(df_clean[col].dropna().abs().median())
            if med_abs <= 1.5:
                pct_scale_info[col] = "decimal_ratio"
            else:
                pct_scale_info[col] = "percentage_points"
        else:
            pct_scale_info[col] = "not_supplied"

    # Normalize Inflation_YoY_Pct to percentage points (0-100 scale) for consistent display/analysis
    if pct_scale_info["Inflation_YoY_Pct"] == "decimal_ratio":
        df_clean["Inflation_YoY_Pct_Display"] = df_clean["Inflation_YoY_Pct"] * 100.0
    else:
        df_clean["Inflation_YoY_Pct_Display"] = df_clean["Inflation_YoY_Pct"]

    # Independent recalculation pipeline
    recalc_rows: List[Dict[str, Any]] = []
    discrepancies: List[Dict[str, Any]] = []
    negative_flags: List[Dict[str, Any]] = []
    inconsistent_relationships: List[Dict[str, Any]] = []

    prev_fx: Optional[float] = None
    for idx, row in df_clean.iterrows():
        month_iso = row["_Month_Parsed"].strftime("%Y-%m-%d")
        month_label = row["_Month_Parsed"].strftime("%b %Y")

        fx = _safe_float(row.get("USD_NGN_Planning_Rate"))
        imp_usd = _safe_float(row.get("Import_Cost_USD_Per_Unit"))
        units = _safe_float(row.get("Units_Sold"))
        price = _safe_float(row.get("Avg_Selling_Price_NGN"))
        mkt = _safe_float(row.get("Marketing_Spend_NGN"))
        other_op = _safe_float(row.get("Other_Operating_Cost_NGN"))
        inf_pct = _safe_float(row.get("Inflation_YoY_Pct_Display"))

        # Flag unexpected negative values on inputs (FX, Units, Price, Import USD, Marketing, Other OpEx, Inflation)
        non_neg_checks = {
            "USD_NGN_Planning_Rate": fx,
            "Units_Sold": units,
            "Import_Cost_USD_Per_Unit": imp_usd,
            "Avg_Selling_Price_NGN": price,
            "Marketing_Spend_NGN": mkt,
            "Other_Operating_Cost_NGN": other_op,
        }
        for cname, cval in non_neg_checks.items():
            if cval is not None and cval < 0:
                negative_flags.append({
                    "month": month_iso,
                    "column": cname,
                    "value": cval,
                    "severity": "High",
                    "reason": f"{cname} should not be negative.",
                })

        if None in (fx, imp_usd, units, price, mkt, other_op):
            # Preserve NaN rather than silently replacing missing inputs with 0
            recalc = {
                "Import_Cost_NGN_Per_Unit": None,
                "COGS_NGN": None,
                "Sales_Revenue_NGN": None,
                "Gross_Profit_NGN": None,
                "Gross_Margin_Pct": None,
                "Operating_Profit_NGN": None,
                "Operating_Margin_Pct": None,
                "FX_Change_MoM_Pct": ((fx - prev_fx) / prev_fx * 100.0) if (fx is not None and prev_fx) else None,
            }
        else:
            recalc = compute_engineered_row(
                usd_ngn_rate=fx,
                import_cost_usd=imp_usd,
                units_sold=units,
                avg_selling_price=price,
                marketing_spend=mkt,
                other_operating_cost=other_op,
                prev_usd_ngn_rate=prev_fx,
                supplied_fx_change_pct=None,
            )

        if fx is not None:
            prev_fx = fx

        # Check economic relationships
        if recalc["Gross_Profit_NGN"] is not None and recalc["Gross_Profit_NGN"] < 0:
            inconsistent_relationships.append({
                "month": month_iso,
                "issue": "Negative Gross Profit",
                "detail": f"Unit import cost (₦{recalc['Import_Cost_NGN_Per_Unit']:,.2f}) exceeded selling price (₦{price:,.2f}).",
            })
        if recalc["Operating_Profit_NGN"] is not None and recalc["Operating_Profit_NGN"] < 0:
            negative_flags.append({
                "month": month_iso,
                "column": "Operating_Profit_NGN",
                "value": round(recalc["Operating_Profit_NGN"], 2),
                "severity": "Warning",
                "reason": "Operating loss recorded (Gross Profit < Marketing + Other Operating Costs).",
            })

        # Compare supplied calculated columns against independently recalculated values
        comparisons = [
            ("Import_Cost_NGN_Per_Unit", recalc["Import_Cost_NGN_Per_Unit"], 0.05, 1.0, False),
            ("COGS_NGN", recalc["COGS_NGN"], 50.0, 1.0, False),
            ("Sales_Revenue_NGN", recalc["Sales_Revenue_NGN"], 50.0, 1.0, False),
            ("Gross_Profit_NGN", recalc["Gross_Profit_NGN"], 50.0, 1.0, False),
            ("Gross_Margin_Pct", recalc["Gross_Margin_Pct"], 0.05, 100.0 if pct_scale_info["Gross_Margin_Pct"] == "decimal_ratio" else 1.0, True),
            ("Operating_Profit_NGN", recalc["Operating_Profit_NGN"], 50.0, 1.0, False),
            ("Operating_Margin_Pct", recalc["Operating_Margin_Pct"], 0.05, 100.0 if pct_scale_info["Operating_Margin_Pct"] == "decimal_ratio" else 1.0, True),
            ("FX_Change_MoM_Pct", recalc["FX_Change_MoM_Pct"], 0.05, 100.0 if pct_scale_info["FX_Change_MoM_Pct"] == "decimal_ratio" else 1.0, True),
        ]

        row_discrepancy_count = 0
        for col_name, calc_val, tol, scale_mult, is_pct in comparisons:
            if col_name not in df_clean.columns or calc_val is None:
                continue
            sup_raw = _safe_float(row.get(col_name))
            if sup_raw is None:
                continue
            sup_val = sup_raw * scale_mult
            diff = abs(sup_val - calc_val)
            # Check relative difference as well for large NGN figures
            rel_diff_pct = (diff / abs(calc_val) * 100.0) if abs(calc_val) > 1e-6 else diff
            # Flag rounding vs material discrepancy
            if diff > tol and (is_pct or rel_diff_pct > 0.01):
                row_discrepancy_count += 1
                discrepancies.append({
                    "month": month_iso,
                    "column": col_name,
                    "supplied_raw": round(sup_raw, 6),
                    "supplied_normalized": round(sup_val, 4),
                    "recalculated": round(calc_val, 4),
                    "abs_difference": round(diff, 4),
                    "rel_difference_pct": round(rel_diff_pct, 4),
                    "classification": "Material Discrepancy" if (rel_diff_pct > 0.5 or (is_pct and diff > 0.25)) else "Rounding Variance",
                })

        recalc_rows.append({
            "Month": month_iso,
            "Month_Label": month_label,
            "Year": int(row["_Month_Parsed"].year),
            "Month_Num": int(row["_Month_Parsed"].month),
            "Quarter": row["Quarter"],
            "Year_Quarter": f"{row['_Month_Parsed'].year} {row['Quarter']}",
            "USD_NGN_Planning_Rate": fx,
            "Inflation_YoY_Pct": inf_pct,
            "Units_Sold": units,
            "Import_Cost_USD_Per_Unit": imp_usd,
            "Avg_Selling_Price_NGN": price,
            "Marketing_Spend_NGN": mkt,
            "Other_Operating_Cost_NGN": other_op,
            "FX_Change_MoM_Pct": round(recalc["FX_Change_MoM_Pct"], 4) if recalc["FX_Change_MoM_Pct"] is not None else None,
            "Import_Cost_NGN_Per_Unit": round(recalc["Import_Cost_NGN_Per_Unit"], 2) if recalc["Import_Cost_NGN_Per_Unit"] is not None else None,
            "COGS_NGN": round(recalc["COGS_NGN"], 2) if recalc["COGS_NGN"] is not None else None,
            "Sales_Revenue_NGN": round(recalc["Sales_Revenue_NGN"], 2) if recalc["Sales_Revenue_NGN"] is not None else None,
            "Gross_Profit_NGN": round(recalc["Gross_Profit_NGN"], 2) if recalc["Gross_Profit_NGN"] is not None else None,
            "Gross_Margin_Pct": round(recalc["Gross_Margin_Pct"], 4) if recalc["Gross_Margin_Pct"] is not None else None,
            "Operating_Profit_NGN": round(recalc["Operating_Profit_NGN"], 2) if recalc["Operating_Profit_NGN"] is not None else None,
            "Operating_Margin_Pct": round(recalc["Operating_Margin_Pct"], 4) if recalc["Operating_Margin_Pct"] is not None else None,
            "Supplied_Operating_Profit_NGN": _safe_float(row.get("Operating_Profit_NGN")),
            "Supplied_COGS_NGN": _safe_float(row.get("COGS_NGN")),
            "Market_Notes": row["Market_Notes"],
            "Row_Discrepancies": row_discrepancy_count,
        })

    processed_df = pd.DataFrame(recalc_rows)

    # Compute outlier flags (IQR) across core numeric features
    outlier_summary: List[Dict[str, Any]] = []
    feature_distributions: Dict[str, Dict[str, Any]] = {}
    numeric_feature_cols = [
        m["name"] for m in FEATURE_METADATA if m["name"] not in {"Month", "Quarter", "Market_Notes"}
    ]

    for col in numeric_feature_cols:
        if col not in processed_df.columns:
            continue
        series = processed_df[col].dropna().astype(float)
        if len(series) == 0:
            continue
        q1 = float(series.quantile(0.25))
        med = float(series.quantile(0.50))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outlier_count = int(((series < lower_fence) | (series > upper_fence)).sum())
        corr_op = float(processed_df[[col, "Operating_Profit_NGN"]].dropna().corr().iloc[0, 1]) if len(series) > 2 else 0.0
        if math.isnan(corr_op):
            corr_op = 0.0

        # 8-bin histogram for visual distribution
        counts, edges = np.histogram(series, bins=min(8, max(3, len(series) // 10)))
        hist = [
            {
                "bin": f"{edges[i]:,.1f}–{edges[i+1]:,.1f}",
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]

        feature_distributions[col] = {
            "count": int(len(series)),
            "missing": int(processed_df[col].isna().sum()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std(ddof=1)) if len(series) > 1 else 0.0, 4),
            "min": round(float(series.min()), 4),
            "q1": round(q1, 4),
            "median": round(med, 4),
            "q3": round(q3, 4),
            "max": round(float(series.max()), 4),
            "iqr_outliers": outlier_count,
            "corr_with_operating_profit": round(corr_op, 4),
            "histogram": hist,
        }
        if outlier_count > 0:
            outlier_summary.append({
                "column": col,
                "outlier_count": outlier_count,
                "lower_fence": round(lower_fence, 2),
                "upper_fence": round(upper_fence, 2),
            })

    # Enrich FEATURE_METADATA with computed stats and quality warnings
    enriched_features: List[Dict[str, Any]] = []
    for meta in FEATURE_METADATA:
        fname = meta["name"]
        stats = feature_distributions.get(fname)
        warnings: List[str] = []
        if fname in missing_values_by_col and missing_values_by_col[fname] > 0:
            warnings.append(f"{missing_values_by_col[fname]} missing value(s) in uploaded file.")
        if fname in pct_scale_info and pct_scale_info[fname] == "decimal_ratio":
            warnings.append("Supplied values are stored as decimal ratios (e.g., 0.3783); normalized to % (0–100) for reporting.")
        if stats and stats["iqr_outliers"] > 0:
            warnings.append(f"{stats['iqr_outliers']} statistical IQR outlier(s) detected.")
        if fname in {"Operating_Profit_NGN", "Operating_Margin_Pct", "Gross_Profit_NGN", "Gross_Margin_Pct", "Sales_Revenue_NGN", "COGS_NGN"}:
            warnings.append("Excluded from ML training predictors to prevent target/accounting leakage.")

        enriched_features.append({
            **meta,
            "stats": stats,
            "warnings": warnings,
        })

    # Count rounding vs material discrepancies
    material_discrepancies = [d for d in discrepancies if d["classification"] == "Material Discrepancy"]
    rounding_variances = [d for d in discrepancies if d["classification"] == "Rounding Variance"]

    total_missing = int(sum(missing_values_by_col.values()))
    quality_report = {
        "records_processed": int(len(processed_df)),
        "raw_rows": int(len(raw_df)),
        "invalid_dates": invalid_dates_count,
        "missing_values_total": total_missing,
        "missing_values_by_column": missing_values_by_col,
        "duplicate_records": duplicate_records_count,
        "duplicate_months": duplicate_months,
        "missing_periods": missing_periods,
        "calculation_discrepancies_total": len(discrepancies),
        "material_discrepancies_count": len(material_discrepancies),
        "rounding_variances_count": len(rounding_variances),
        "discrepancy_samples": discrepancies[:50],
        "negative_value_flags": negative_flags,
        "inconsistent_relationships": inconsistent_relationships,
        "invalid_numeric_issues": invalid_numeric_issues,
        "outlier_summary": outlier_summary,
        "scale_conventions_detected": pct_scale_info,
        "missing_optional_columns": missing_optional,
        "unexpected_columns": unexpected_cols,
        "date_range": {
            "start": processed_df["Month"].iloc[0] if len(processed_df) else None,
            "end": processed_df["Month"].iloc[-1] if len(processed_df) else None,
        },
    }

    return {
        "df": processed_df,
        "records": processed_df.replace({np.nan: None}).to_dict(orient="records"),
        "quality_report": quality_report,
        "feature_intelligence": enriched_features,
    }
