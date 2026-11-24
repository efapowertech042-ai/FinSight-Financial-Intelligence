export interface FinancialRecord {
  Month: string;
  Month_Label: string;
  Year: number;
  Month_Num: number;
  Quarter: string;
  Year_Quarter: string;
  USD_NGN_Planning_Rate: number;
  Inflation_YoY_Pct: number;
  Units_Sold: number;
  Import_Cost_USD_Per_Unit: number;
  Avg_Selling_Price_NGN: number;
  Marketing_Spend_NGN: number;
  Other_Operating_Cost_NGN: number;
  FX_Change_MoM_Pct: number | null;
  Import_Cost_NGN_Per_Unit: number;
  COGS_NGN: number;
  Sales_Revenue_NGN: number;
  Gross_Profit_NGN: number;
  Gross_Margin_Pct: number;
  Operating_Profit_NGN: number;
  Operating_Margin_Pct: number;
  Supplied_Operating_Profit_NGN?: number | null;
  Supplied_COGS_NGN?: number | null;
  Market_Notes: string;
  Row_Discrepancies: number;
}

export interface DashboardKPIs {
  total_sales_revenue: number;
  total_operating_profit: number;
  avg_operating_margin_pct: number;
  weighted_operating_margin_pct?: number;
  total_gross_profit: number;
  gross_margin_pct: number;
  total_units_sold: number;
  total_marketing_spend: number;
  total_other_operating_cost?: number;
  total_import_costs: number;
  avg_usd_ngn_rate: number;
  avg_inflation_pct: number;
  period_count: number;
  deltas: {
    revenue_pct?: number | null;
    operating_profit_pct?: number | null;
    gross_profit_pct?: number | null;
    units_pct?: number | null;
    gross_margin_pp?: number | null;
    operating_margin_pp?: number | null;
    marketing_pct?: number | null;
    import_cost_pct?: number | null;
  };
}

export interface QuarterlySummaryItem {
  quarter: string;
  months_count: number;
  total_revenue: number;
  avg_revenue: number;
  total_gross_profit: number;
  total_operating_profit: number;
  avg_operating_profit: number;
  avg_operating_margin_pct: number;
  avg_gross_margin_pct: number;
  total_units: number;
  total_marketing: number;
}

export interface FactorImpact {
  factor: string;
  column: string;
  corr_operating_profit: number;
  corr_revenue: number;
  avg_value: number;
}

export interface DatasetState {
  dataset_name: string;
  available_date_range: { start: string | null; end: string | null };
  kpis: DashboardKPIs;
  quarterly_summary: QuarterlySummaryItem[];
  profit_distribution: { range: string; count: number }[];
  factor_impacts: FactorImpact[];
  records: FinancialRecord[];
  quality_summary: {
    records_processed: number;
    missing_values_total: number;
    duplicate_records: number;
    material_discrepancies_count: number;
    rounding_variances_count: number;
    sha256_short?: string;
  };
  latest_record: FinancialRecord | null;
}

export interface FeatureStats {
  count: number;
  missing: number;
  mean: number;
  std: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  iqr_outliers: number;
  corr_with_operating_profit: number;
  histogram: { bin: string; count: number }[];
}

export interface FeatureIntelligenceItem {
  name: string;
  category: string;
  data_type: string;
  formula: string;
  business_definition: string;
  mathematical_role: string;
  financial_interpretation: string;
  profitability_influence: string;
  forecast_availability: string;
  stats?: FeatureStats;
  warnings: string[];
}

export interface DataQualityReport {
  records_processed: number;
  raw_rows: number;
  invalid_dates: number;
  missing_values_total: number;
  missing_values_by_column: Record<string, number>;
  duplicate_records: number;
  duplicate_months: string[];
  missing_periods: string[];
  calculation_discrepancies_total: number;
  material_discrepancies_count: number;
  rounding_variances_count: number;
  discrepancy_samples: {
    month: string;
    column: string;
    supplied_raw: number;
    supplied_normalized: number;
    recalculated: number;
    abs_difference: number;
    rel_difference_pct: number;
    classification: string;
  }[];
  negative_value_flags: {
    month: string;
    column: string;
    value: number;
    severity: string;
    reason: string;
  }[];
  inconsistent_relationships: { month: string; issue: string; detail: string }[];
  invalid_numeric_issues: { row: number; month: string; column: string; raw_value: string }[];
  outlier_summary: { column: string; outlier_count: number; lower_fence: number; upper_fence: number }[];
  scale_conventions_detected: Record<string, string>;
  date_range: { start: string | null; end: string | null };
}

export interface FeaturesIntelligenceResponse {
  features: FeatureIntelligenceItem[];
  quality_report: DataQualityReport;
  dependency_graph: {
    nodes: { id: string; label: string; category: string; formula: string }[];
    edges: { source: string; target: string; label: string }[];
  };
}

export interface ModelMetrics {
  mae: number;
  rmse: number;
  r2: number | null;
  mape_pct: number | null;
  smape_pct: number | null;
}

export interface EvaluatedModel {
  key: string;
  name: string;
  is_tree: boolean;
  train_metrics: ModelMetrics;
  test_metrics: ModelMetrics;
  cv_rmse: number | null;
  test_predictions: { month: string; month_label: string; actual: number; predicted: number; residual: number }[];
  train_predictions: { month: string; month_label: string; actual: number; predicted: number; residual: number }[];
}

export interface ModelSuite {
  mode: string;
  sufficient_data: boolean;
  dataset_size?: number;
  train_size?: number;
  test_size?: number;
  train_period?: string;
  test_period?: string;
  feature_cols?: string[];
  models: EvaluatedModel[];
  selected_model: EvaluatedModel | null;
  limitation_notice?: string | null;
  leakage_audit?: { feature: string; reason: string; availability: string }[];
}

export interface ModelsSummaryResponse {
  driver_suite: ModelSuite;
  ex_ante_suite: ModelSuite;
  latest_record: FinancialRecord | null;
}

export interface PredictionInputs {
  USD_NGN_Planning_Rate: number;
  Inflation_YoY_Pct: number;
  Units_Sold: number;
  Import_Cost_USD_Per_Unit: number;
  Avg_Selling_Price_NGN: number;
  Marketing_Spend_NGN: number;
  Other_Operating_Cost_NGN: number;
  FX_Change_MoM_Pct: number | null;
  Quarter: string;
  Market_Notes: string;
  horizon_months?: number;
}

export interface OutOfRangeWarning {
  feature: string;
  label: string;
  input_value: number;
  hist_min: number;
  hist_max: number;
  direction: string;
}

export interface PredictionSensitivityItem {
  lever: string;
  label: string;
  accounting_delta_ngn: number;
  accounting_delta_pct: number;
  ml_delta_ngn: number;
  ml_delta_pct: number;
}

export interface PredictionResponse {
  ml_prediction: {
    model_name: string;
    model_key: string;
    is_tree_model?: boolean;
    predicted_operating_profit_ngn: number;
    predicted_operating_margin_pct: number;
    ridge_extrapolation_ngn?: number;
    prediction_interval_80?: [number, number];
    prediction_interval_95?: [number, number];
    out_of_range_warnings?: OutOfRangeWarning[];
    out_of_time_rmse: number | null;
    out_of_time_r2: number | null;
    input_feature_vector: Record<string, number>;
  };
  accounting_reconciliation: {
    Import_Cost_NGN_Per_Unit: number;
    COGS_NGN: number;
    Sales_Revenue_NGN: number;
    Gross_Profit_NGN: number;
    Gross_Margin_Pct: number;
    Marketing_Spend_NGN: number;
    Other_Operating_Cost_NGN: number;
    Accounting_Operating_Profit_NGN: number;
    Accounting_Operating_Margin_Pct: number;
    FX_Change_MoM_Pct: number;
    ml_vs_accounting_gap_ngn: number;
  };
  measurable_unit_economics?: {
    unit_spread_ngn: number;
    price_to_import_ratio: number;
    break_even_units: number | null;
    margin_of_safety_pct: number | null;
    marketing_roas: number | null;
    opex_to_revenue_pct: number;
    marketing_to_revenue_pct: number;
  };
  measurable_sensitivities?: PredictionSensitivityItem[];
  horizon_scaling?: {
    horizon_months: number;
    ml_scaled_profit_ngn: number;
    ml_scaled_ci95_low_ngn: number;
    ml_scaled_ci95_high_ngn: number;
    accounting_scaled_profit_ngn: number;
    scaled_revenue_ngn: number;
    scaled_units_sold: number;
    ridge_scaled_profit_ngn: number;
  };
  local_explanation: {
    baseline_prediction_ngn: number;
    final_predicted_operating_profit_ngn: number;
    shap_sum_check_ngn: number;
    contributions: {
      feature: string;
      input_value: number;
      shap_contribution_ngn: number;
      abs_contribution: number;
      direction: "increases" | "decreases";
    }[];
    top_increasing_drivers: {
      feature: string;
      input_value: number;
      shap_contribution_ngn: number;
    }[];
    top_decreasing_drivers: {
      feature: string;
      input_value: number;
      shap_contribution_ngn: number;
    }[];
  };
  recent_history: {
    month: string;
    month_label: string;
    operating_profit_ngn: number;
    sales_revenue_ngn: number;
    operating_margin_pct: number;
  }[];
  market_notes: string;
}

export interface GlobalImportanceItem {
  feature: string;
  mean_abs_shap: number;
  mean_signed_shap: number;
  value_shap_correlation: number;
  direction: string;
  permutation_importance_mean: number;
  permutation_importance_std: number;
  share_pct: number;
}

export interface ExplainabilityResponse {
  available: boolean;
  reason?: string;
  model_name: string;
  base_value_ngn: number;
  global_importance: GlobalImportanceItem[];
  beeswarm_points: {
    feature: string;
    month: string;
    month_label: string;
    feature_value: number;
    normalized_value: number;
    shap_value: number;
  }[];
  dependence_plots: Record<
    string,
    {
      month: string;
      month_label: string;
      x: number;
      shap_value: number;
      fx_rate: number;
      actual_profit: number;
    }[]
  >;
  methodology_note: string;
}

export interface AnomalyItem {
  month: string;
  month_label: string;
  quarter: string;
  variable_affected: string;
  variable_label: string;
  observed_value: number;
  expected_value: number;
  deviation_abs: number;
  deviation_pct: number;
  severity: "High" | "Medium" | "Low";
  methods_triggered: string[];
  all_affected_variables: string[];
  supporting_variables: {
    USD_NGN_Planning_Rate: number;
    FX_Change_MoM_Pct: number;
    Inflation_YoY_Pct: number;
    Units_Sold: number;
    Avg_Selling_Price_NGN: number;
    Gross_Margin_Pct: number;
    Operating_Margin_Pct: number;
    Market_Notes: string;
  };
  interpretation: string;
  recommended_investigation: string;
  statistical_vs_business_note: string;
}

export interface AnomalyReport {
  anomalies: AnomalyItem[];
  time_series: {
    month: string;
    month_label: string;
    actual_operating_profit: number;
    expected_operating_profit: number;
    residual: number;
    is_anomaly: boolean;
    severity: string;
    isolation_score: number;
  }[];
  severity_distribution: { severity: string; count: number }[];
  feature_breakdown: { variable: string; count: number }[];
  analytical_findings: {
    id: string;
    question: string;
    finding: string;
    evidence_months: string[];
  }[];
  summary: {
    total_anomalies: number;
    high_severity: number;
    medium_severity: number;
    low_severity: number;
  };
}

export interface ScenarioResponse {
  baseline: {
    accounting: PredictionResponse["accounting_reconciliation"];
    ml_predicted_operating_profit_ngn: number;
    ml_predicted_operating_margin_pct: number;
  };
  simulated: {
    accounting: PredictionResponse["accounting_reconciliation"];
    ml_predicted_operating_profit_ngn: number;
    ml_predicted_operating_margin_pct: number;
  };
  deltas: {
    revenue_delta: number;
    cogs_delta: number;
    gross_profit_delta: number;
    accounting_profit_delta: number;
    ml_predicted_profit_delta: number;
    gross_margin_pp_delta: number;
    operating_margin_pp_delta: number;
  };
  waterfall: { step: string; value: number; type: "baseline" | "delta" | "total" }[];
}

export interface RagMatchedRecord {
  month: string;
  month_label: string;
  quarter: string;
  usd_ngn_rate: number;
  fx_change_mom_pct: number | null;
  units_sold: number;
  avg_selling_price_ngn: number;
  sales_revenue_ngn: number;
  cogs_ngn: number;
  operating_profit_ngn: number;
  operating_margin_pct: number;
  market_notes: string;
  match_reason: string;
}

export interface RagRetrievalStep {
  step: number;
  stage: string;
  detail: string;
}

export interface GraphicalRagResponse {
  intent: string;
  question: string;
  data_provenance?: {
    dataset_name: string;
    records_scanned: number;
    date_start: string;
    date_end: string;
    stores_consulted: string[];
  };
  retrieval_process?: RagRetrievalStep[];
  matched_records?: RagMatchedRecord[];
  chart: {
    title: string;
    x_key: string;
    series: { key: string; label: string; color: string; axis: "left" | "right" }[];
    data: Record<string, any>[];
  };
  graph: {
    nodes: { id: string; label: string; role: string; metric: string }[];
    edges: { source: string; target: string; label: string }[];
  };
  evidence: {
    type: string;
    month: string;
    month_label: string;
    headline: string;
    detail: string;
  }[];
  analytical_explanation: string;
  business_implications: string;
  limitations: string;
  synthesis_mode?: string;
  llm_grounded?: boolean;
  llm_narrative?: string | null;
}

export interface MonitoringReport {
  selected_model: string;
  train_metrics: ModelMetrics;
  test_metrics: ModelMetrics;
  residual_series: { month: string; month_label: string; actual: number; predicted: number; residual: number }[];
  feature_drift: {
    feature: string;
    reference_mean: number;
    recent_mean: number;
    pct_change: number;
    standardized_shift: number;
    status: "Stable" | "Moderate Drift" | "High Drift";
  }[];
  data_quality: {
    records_processed: number;
    missing_values_total: number;
    material_discrepancies: number;
    duplicate_records: number;
  };
  retraining_assessment: {
    recommended: boolean;
    mean_recent_abs_error_ngn: number;
    rationale: string;
  };
}

export interface TSFeatureDictionaryItem {
  name: string;
  category: string;
  formula: string;
  leakage_status: string;
  business_relevance: string;
}

export interface ForecastModelLeaderboardItem {
  key: string;
  name: string;
  family: string;
  training_obs: number;
  validation_obs: number;
  mae: number;
  rmse: number;
  mape_pct: number | null;
  r2: number | null;
  mean_error: number;
  bias_pct: number;
  forecast_bias: "Balanced" | "Underpredicts" | "Overpredicts";
  backtest_mean_rmse: number;
  backtest_std_rmse: number;
  stability_score_pct: number;
  pi_coverage_80_pct: number;
  pi_coverage_95_pct: number;
  composite_rmse: number;
  status_note: string;
  status: "Champion" | "Benchmark";
}

export interface HypothesisTestItem {
  id: string;
  title: string;
  null_hypothesis: string;
  alternative_hypothesis: string;
  test_used: string;
  statistic_name: string;
  statistic: number | null;
  p_value: number | null;
  alpha: number;
  confidence_interval: string;
  decision: "Reject H0" | "Fail to Reject H0" | "Insufficient Evidence";
  significant: boolean;
  assumption_checks: string;
  business_interpretation: string;
}

export interface HypothesisTestingBundle {
  alpha: number;
  split_year: number;
  tests: HypothesisTestItem[];
  methodology_note: string;
}

export interface ForecastMonthItem {
  month: string;
  month_label: string;
  year: number;
  month_num: number;
  quarter: string;
  year_quarter: string;
  horizon_step: number;
  period_type: "historical" | "forecast";
  operating_profit_ngn: number;
  lower_80_ngn: number | null;
  upper_80_ngn: number | null;
  lower_95_ngn: number | null;
  upper_95_ngn: number | null;
  sales_revenue_ngn: number;
  gross_profit_ngn: number;
  operating_expenses_ngn: number;
  units_sold: number;
  operating_margin_pct: number;
  profit_per_unit_ngn: number;
  is_negative_profit: boolean;
  is_risk_period: boolean;
}

export interface ForecastAnnualSummary {
  year: number;
  months_count: number;
  historical_months: number;
  forecast_months: number;
  period_scope: string;
  total_operating_profit_ngn: number;
  total_sales_revenue_ngn: number;
  total_gross_profit_ngn: number;
  total_operating_expenses_ngn: number;
  weighted_operating_margin_pct: number;
  yoy_operating_profit_pct: number | null;
  negative_months: number;
  risk_months: number;
}

export interface ForecastQuarterlyItem {
  year_quarter: string;
  year: number;
  quarter: string;
  months: number;
  total_operating_profit_ngn: number;
  total_sales_revenue_ngn: number;
  avg_operating_margin_pct: number;
  lower_95_sum_ngn: number;
  upper_95_sum_ngn: number;
}

export interface ForecastScenarioAssumptions {
  revenue_growth_adj_pp: number;
  opex_growth_adj_pp: number;
  margin_efficiency_adj_pp: number;
  fx_cost_pressure_adj_pct: number;
  volume_productivity_adj_pct: number;
}

export interface ForecastScenarioCase {
  key: "base" | "optimistic" | "stress" | "custom";
  label: string;
  description: string;
  assumptions: ForecastScenarioAssumptions;
  summary: {
    dec_2030_operating_profit_ngn: number;
    dec_2030_operating_margin_pct: number;
    year_2030_total_profit_ngn: number;
    year_2030_operating_margin_pct: number;
    cumulative_profit_ngn: number;
    cumulative_revenue_ngn?: number;
    cumulative_margin_pct?: number;
    negative_months: number;
    risk_months: number;
  };
}

export interface ForecastAnnualScenarioComparison {
  year: number;
  months: number;
  base_profit_ngn: number;
  custom_profit_ngn: number;
  delta_profit_ngn: number;
  optimistic_profit_ngn: number;
  stress_profit_ngn: number;
  base_revenue_ngn: number;
  custom_revenue_ngn: number;
  base_margin_pct: number;
  custom_margin_pct: number;
  custom_negative_months: number;
  custom_risk_months: number;
}

export interface ForecastLeverWaterfallStep {
  step: string;
  value: number;
  type: "baseline" | "delta" | "total";
}

export interface ForecastScenariosBundle {
  cases: Record<"base" | "optimistic" | "stress" | "custom", ForecastScenarioCase>;
  monthly_comparison: {
    month: string;
    month_label: string;
    year?: number;
    base: number;
    optimistic: number;
    stress: number;
    custom: number;
    base_revenue?: number;
    custom_revenue?: number;
    base_opex?: number;
    custom_opex?: number;
    base_margin?: number;
    custom_margin?: number;
  }[];
  annual_comparison?: ForecastAnnualScenarioComparison[];
  lever_attribution_waterfall?: ForecastLeverWaterfallStep[];
}

export interface CustomScenarioVerdict {
  category:
    | "Strong Positive Outlook"
    | "Positive but Requires Monitoring"
    | "Stable / Mixed Outlook"
    | "Negative Operational Outlook"
    | "High-Risk Outlook"
    | "Insufficient Evidence";
  thresholds_evaluated: {
    rule: string;
    threshold: string;
    observed: string;
    passed: boolean;
  }[];
  headline: string;
  expected_dec_2030_profit_ngn: number;
  delta_vs_base_dec_2030_ngn: number;
  delta_vs_base_cumulative_ngn: number;
  delta_vs_base_margin_pp: number;
  uncertainty_pct: number;
}

export interface OperationalVerdict {
  category:
    | "Strong Positive Outlook"
    | "Positive but Requires Monitoring"
    | "Stable / Mixed Outlook"
    | "Negative Operational Outlook"
    | "High-Risk Outlook"
    | "Insufficient Evidence";
  thresholds_evaluated: {
    rule: string;
    threshold: string;
    observed: string;
    passed: boolean;
  }[];
  headline: string;
  expected_dec_2030_profit_ngn: number;
  annual_trajectory_summary: string;
  operational_efficiency: string;
  areas_deteriorating: string[];
  major_cost_pressures: string;
  significant_drivers: string[];
  forecast_uncertainty_summary: string;
  principal_risks: string[];
  recommended_interventions: string[];
  verdict_rationale: string;
}

export interface IndicatorTrendItem {
  indicator: string;
  trailing_12m_avg: number;
  prior_12m_avg: number;
  yoy_change_pct: number;
  status: "Improving" | "Deteriorating" | "Stable";
}

export interface DriverImportanceItem {
  feature: string;
  importance_value: number;
  signed_effect: number;
  share_pct: number;
  method: string;
  direction: string;
}

export interface ForecastingCentreResponse {
  sufficient_data: boolean;
  limitation_notice?: string;
  dataset_summary?: {
    historical_records: number;
    historical_start: string;
    historical_end: string;
    forecast_start: string;
    forecast_end: string;
    forecast_horizon_months: number;
    horizon_note: string;
    historical_negative_profit_months: number;
  };
  executive_summary?: {
    latest_actual_month: string;
    latest_actual_operating_profit_ngn: number;
    dec_2030_month: string;
    dec_2030_forecast_operating_profit_ngn: number;
    dec_2030_lower_95_ngn: number;
    dec_2030_upper_95_ngn: number;
    projected_cumulative_operating_profit_ngn: number;
    forecast_operating_margin_pct: number;
    forecast_growth_to_2030_pct: number;
    champion_model_name: string;
    champion_holdout_rmse: number;
    champion_holdout_mae: number;
    champion_holdout_mape_pct: number | null;
    champion_holdout_r2: number | null;
    forecast_uncertainty_95_pct: number;
    forecast_negative_months: number;
    forecast_risk_months: number;
  };
  indicator_trends?: IndicatorTrendItem[];
  feature_dictionary?: TSFeatureDictionaryItem[];
  leaderboard?: ForecastModelLeaderboardItem[];
  validation?: {
    holdout_size: number;
    holdout_period: string;
    backtest_origins_count: number;
    backtest_horizon_months: number;
    champion_name: string;
    champion_bias: string;
    champion_mean_error: number;
    champion_bias_pct: number;
    holdout_series: {
      month: string;
      month_label: string;
      actual: number;
      predicted: number;
      naive: number;
      residual: number;
    }[];
    residual_histogram: { bin: string; count: number }[];
    residual_acf: { lag: number; acf: number; bound: number }[];
    durbin_watson: number | null;
    ljung_box_p_value: number | null;
    bias_explanation: string;
  };
  driver_importance?: DriverImportanceItem[];
  driver_importance_model?: string;
  hypothesis_testing?: HypothesisTestingBundle;
  secondary_kpis?: Record<string, { model: string; holdout_rmse: number | null }>;
  historical_months?: ForecastMonthItem[];
  forecast_months?: ForecastMonthItem[];
  annual_summaries?: ForecastAnnualSummary[];
  quarterly_forecasts?: ForecastQuarterlyItem[];
  scenarios?: ForecastScenariosBundle;
  verdict: OperationalVerdict;
  custom_verdict?: CustomScenarioVerdict;
}

export interface Accounting5YrParams {
  annual_inflation_pct: number;
  unit_sales_growth_pct: number;
  price_inflation_pass_through_pct: number;
  fx_cost_pass_through_pct: number;
  usd_import_cost_growth_pct: number;
  opex_inflation_sensitivity_pct: number;
  fx_inflation_sensitivity_x: number;
}

export interface Accounting5YrPoint {
  month?: string;
  year: number;
  month_label: string;
  period_type: "historical" | "forecast" | "mixed";
  historical_op: number | null;
  baseline_op: number | null;
  scenario_op: number | null;
  band_80: [number, number] | null;
  negative_op: number | null;
  sales_revenue_ngn: number;
  cogs_ngn?: number;
  operating_expenses_ngn?: number;
  operating_margin_pct: number;
  is_negative_profit?: boolean;
  months?: number;
}

export interface Accounting5YrResponse {
  anchor_month_label: string;
  history_range_label: string;
  full_range_label: string;
  historical_records: number;
  forecast_horizon_months: number;
  params: Accounting5YrParams;
  presets: Record<"base" | "opt" | "stress", Accounting5YrParams>;
  summary: {
    latest_actual_month: string;
    latest_actual_op_ngn: number;
    final_forecast_month: string;
    final_forecast_op_ngn: number;
    final_baseline_op_ngn: number;
    final_op_delta_ngn: number;
    cumulative_5yr_op_ngn: number;
    cumulative_5yr_baseline_op_ngn: number;
    cumulative_5yr_delta_ngn: number;
    operating_margin_5yr_pct: number;
    baseline_margin_5yr_pct: number;
    historical_negative_months: number;
    forecast_negative_months: number;
    band_breach_months: number;
  };
  monthly_series: Accounting5YrPoint[];
  annual_series: Accounting5YrPoint[];
}

