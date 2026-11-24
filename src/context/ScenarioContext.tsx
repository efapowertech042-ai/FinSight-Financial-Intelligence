import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import {
  CustomScenarioVerdict,
  FinancialRecord,
  ForecastScenarioAssumptions,
  ForecastScenariosBundle,
  OperationalVerdict,
  PredictionInputs,
  ScenarioResponse,
} from "../types/financial";

export type ScenarioPresetId =
  | "base"
  | "optimistic"
  | "fx_shock"
  | "fx_pass_through"
  | "opex_discipline"
  | "custom";

export interface DriverDeltaPercentages {
  fxPct: number;
  pricePct: number;
  unitsPct: number;
  impUsdPct: number;
  mktPct: number;
  opexPct: number;
  inflationDeltaPp: number;
}

export interface ScenarioPresetSpec {
  id: Exclude<ScenarioPresetId, "custom">;
  label: string;
  shortLabel: string;
  description: string;
  forecast: ForecastScenarioAssumptions;
  deltas: DriverDeltaPercentages;
}

export const SCENARIO_PRESETS: ScenarioPresetSpec[] = [
  {
    id: "base",
    label: "Base Model (Zero Shock)",
    shortLabel: "Base Model",
    description: "Validated statistical baseline with zero manual overrides.",
    forecast: {
      revenue_growth_adj_pp: 0,
      opex_growth_adj_pp: 0,
      margin_efficiency_adj_pp: 0,
      fx_cost_pressure_adj_pct: 0,
      volume_productivity_adj_pct: 0,
    },
    deltas: {
      fxPct: 0,
      pricePct: 0,
      unitsPct: 0,
      impUsdPct: 0,
      mktPct: 0,
      opexPct: 0,
      inflationDeltaPp: 0,
    },
  },
  {
    id: "optimistic",
    label: "Optimistic Growth & Margin Expansion",
    shortLabel: "Optimistic",
    description: "+5 pp/yr revenue growth, +6% volume, −4% FX import pressure, +2 pp margin.",
    forecast: {
      revenue_growth_adj_pp: 5.0,
      opex_growth_adj_pp: -2.0,
      margin_efficiency_adj_pp: 2.0,
      fx_cost_pressure_adj_pct: -4.0,
      volume_productivity_adj_pct: 6.0,
    },
    deltas: {
      fxPct: -4,
      pricePct: 5,
      unitsPct: 8,
      impUsdPct: -2,
      mktPct: 2,
      opexPct: -3,
      inflationDeltaPp: -2.0,
    },
  },
  {
    id: "fx_shock",
    label: "FX Depreciation Shock",
    shortLabel: "FX Stress Shock",
    description: "+14% FX/import shock, −5 pp/yr demand growth, +7 pp/yr OpEx inflation, −8% volume.",
    forecast: {
      revenue_growth_adj_pp: -5.0,
      opex_growth_adj_pp: 7.0,
      margin_efficiency_adj_pp: -3.0,
      fx_cost_pressure_adj_pct: 14.0,
      volume_productivity_adj_pct: -8.0,
    },
    deltas: {
      fxPct: 18,
      pricePct: 0,
      unitsPct: -6,
      impUsdPct: 3,
      mktPct: 5,
      opexPct: 8,
      inflationDeltaPp: 4.5,
    },
  },
  {
    id: "fx_pass_through",
    label: "FX Shock + Pricing Pass-Through",
    shortLabel: "FX + Price Hedge",
    description: "+14% FX/import pressure offset by +4 pp/yr pricing & +0.5 pp operating discipline.",
    forecast: {
      revenue_growth_adj_pp: 4.0,
      opex_growth_adj_pp: 3.0,
      margin_efficiency_adj_pp: 0.5,
      fx_cost_pressure_adj_pct: 14.0,
      volume_productivity_adj_pct: -3.0,
    },
    deltas: {
      fxPct: 18,
      pricePct: 14,
      unitsPct: -3,
      impUsdPct: 0,
      mktPct: 2,
      opexPct: 3,
      inflationDeltaPp: 3.0,
    },
  },
  {
    id: "opex_discipline",
    label: "Cost & OpEx Efficiency",
    shortLabel: "Cost Discipline",
    description: "−4 pp/yr OpEx growth, +2.5 pp margin efficiency, +2% throughput.",
    forecast: {
      revenue_growth_adj_pp: 1.0,
      opex_growth_adj_pp: -4.0,
      margin_efficiency_adj_pp: 2.5,
      fx_cost_pressure_adj_pct: 0.0,
      volume_productivity_adj_pct: 2.0,
    },
    deltas: {
      fxPct: 0,
      pricePct: 2,
      unitsPct: 3,
      impUsdPct: -4,
      mktPct: -10,
      opexPct: -12,
      inflationDeltaPp: -1.0,
    },
  },
];

export const BASE_FORECAST_ASSUMPTIONS: ForecastScenarioAssumptions = SCENARIO_PRESETS[0].forecast;
export const BASE_DRIVER_DELTAS: DriverDeltaPercentages = SCENARIO_PRESETS[0].deltas;

function buildInputsFromRecord(
  rec: FinancialRecord | null,
  deltas: DriverDeltaPercentages = BASE_DRIVER_DELTAS
): PredictionInputs {
  const baseFx = rec?.USD_NGN_Planning_Rate ?? 1564.13;
  const baseInf = rec?.Inflation_YoY_Pct ?? 20.67;
  const baseUnits = rec?.Units_Sold ?? 11700;
  const baseImpUsd = rec?.Import_Cost_USD_Per_Unit ?? 10.12;
  const basePrice = rec?.Avg_Selling_Price_NGN ?? 25370;
  const baseMkt = rec?.Marketing_Spend_NGN ?? 7689000;
  const baseOpex = rec?.Other_Operating_Cost_NGN ?? 10037000;

  return {
    USD_NGN_Planning_Rate: Number((baseFx * (1 + deltas.fxPct / 100)).toFixed(2)),
    Inflation_YoY_Pct: Number(Math.max(0, baseInf + deltas.inflationDeltaPp).toFixed(2)),
    Units_Sold: Math.max(100, Math.round(baseUnits * (1 + deltas.unitsPct / 100))),
    Import_Cost_USD_Per_Unit: Number(Math.max(0.5, baseImpUsd * (1 + deltas.impUsdPct / 100)).toFixed(2)),
    Avg_Selling_Price_NGN: Math.max(100, Math.round(basePrice * (1 + deltas.pricePct / 100))),
    Marketing_Spend_NGN: Math.max(0, Math.round(baseMkt * (1 + deltas.mktPct / 100))),
    Other_Operating_Cost_NGN: Math.max(0, Math.round(baseOpex * (1 + deltas.opexPct / 100))),
    FX_Change_MoM_Pct:
      deltas.fxPct !== 0
        ? Number(deltas.fxPct.toFixed(2))
        : rec?.FX_Change_MoM_Pct !== null && rec?.FX_Change_MoM_Pct !== undefined
          ? Number(rec.FX_Change_MoM_Pct.toFixed(2))
          : 0,
    Quarter: rec?.Quarter ?? "Q4",
    Market_Notes: rec?.Market_Notes || "",
  };
}

interface ScenarioContextValue {
  presetId: ScenarioPresetId;
  activePresetLabel: string;
  isScenarioActive: boolean;
  anchorRecord: FinancialRecord | null;
  setAnchorRecord: (rec: FinancialRecord | null) => void;
  forecastAssumptions: ForecastScenarioAssumptions;
  setForecastAssumptions: (next: ForecastScenarioAssumptions) => void;
  driverDeltas: DriverDeltaPercentages;
  setDriverDeltas: (next: DriverDeltaPercentages) => void;
  baselineInputs: PredictionInputs;
  simulatedInputs: PredictionInputs;
  setExactSimulatedInputs: (next: PredictionInputs) => void;
  applyPreset: (id: Exclude<ScenarioPresetId, "custom">) => void;
  resetScenario: () => void;
  forecastScenarios: ForecastScenariosBundle | null;
  baseVerdict: OperationalVerdict | null;
  customVerdict: CustomScenarioVerdict | null;
  singlePeriodResult: ScenarioResponse | null;
  forecastLoading: boolean;
  singleLoading: boolean;
  drawerOpen: boolean;
  setDrawerOpen: (open: boolean) => void;
  refreshSimulation: () => void;
}

const ScenarioContext = createContext<ScenarioContextValue | null>(null);

interface ScenarioProviderProps {
  latestRecord: FinancialRecord | null;
  datasetKey: string;
  children: React.ReactNode;
}

export const ScenarioProvider: React.FC<ScenarioProviderProps> = ({
  latestRecord,
  datasetKey,
  children,
}) => {
  const [presetId, setPresetId] = useState<ScenarioPresetId>("base");
  const [anchorRecord, setAnchorRecord] = useState<FinancialRecord | null>(latestRecord);
  const [forecastAssumptions, setForecastAssumptionsState] =
    useState<ForecastScenarioAssumptions>(BASE_FORECAST_ASSUMPTIONS);
  const [driverDeltas, setDriverDeltasState] =
    useState<DriverDeltaPercentages>(BASE_DRIVER_DELTAS);
  const [manualSimulatedInputs, setManualSimulatedInputs] = useState<PredictionInputs | null>(null);

  const [forecastScenarios, setForecastScenarios] = useState<ForecastScenariosBundle | null>(null);
  const [baseVerdict, setBaseVerdict] = useState<OperationalVerdict | null>(null);
  const [customVerdict, setCustomVerdict] = useState<CustomScenarioVerdict | null>(null);
  const [singlePeriodResult, setSinglePeriodResult] = useState<ScenarioResponse | null>(null);

  const [forecastLoading, setForecastLoading] = useState<boolean>(false);
  const [singleLoading, setSingleLoading] = useState<boolean>(false);
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);
  const [refreshTick, setRefreshTick] = useState<number>(0);

  useEffect(() => {
    setAnchorRecord(latestRecord);
    setManualSimulatedInputs(null);
  }, [latestRecord?.Month, datasetKey]);

  const baselineInputs = useMemo(() => buildInputsFromRecord(anchorRecord, BASE_DRIVER_DELTAS), [anchorRecord]);
  const simulatedInputs = useMemo(
    () => manualSimulatedInputs || buildInputsFromRecord(anchorRecord, driverDeltas),
    [manualSimulatedInputs, anchorRecord, driverDeltas]
  );

  const isScenarioActive = useMemo(() => {
    if (presetId !== "base") return true;
    const fValues = Object.values(forecastAssumptions);
    const dValues = Object.values(driverDeltas);
    return fValues.some((v) => Math.abs(v) > 1e-6) || dValues.some((v) => Math.abs(v) > 1e-6);
  }, [presetId, forecastAssumptions, driverDeltas]);

  const activePresetLabel = useMemo(() => {
    if (presetId === "custom") return "Custom User Scenario";
    return SCENARIO_PRESETS.find((p) => p.id === presetId)?.shortLabel || "Base Model";
  }, [presetId]);

  // Simulate 2030 multi-year scenario via backend /api/py/forecasting/scenario
  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(async () => {
      setForecastLoading(true);
      try {
        const res = await api.simulateForecastScenario(forecastAssumptions);
        if (!cancelled) {
          setForecastScenarios(res.scenarios);
          setBaseVerdict(res.verdict);
          setCustomVerdict(res.custom_verdict);
        }
      } catch {
        // Fallback when dataset has <24 months for 2030 horizon
      } finally {
        if (!cancelled) setForecastLoading(false);
      }
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [forecastAssumptions, datasetKey, refreshTick]);

  // Simulate single-period accounting + ML profit bridge via /api/py/scenario/simulate
  useEffect(() => {
    if (!anchorRecord) return;
    let cancelled = false;
    const timer = setTimeout(async () => {
      setSingleLoading(true);
      try {
        const res = await api.simulateScenario(baselineInputs, simulatedInputs);
        if (!cancelled) {
          setSinglePeriodResult(res);
        }
      } catch {
        // Ignore transient invalid input
      } finally {
        if (!cancelled) setSingleLoading(false);
      }
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [baselineInputs, simulatedInputs, anchorRecord, datasetKey, refreshTick]);

  const setForecastAssumptions = useCallback((next: ForecastScenarioAssumptions) => {
    setPresetId("custom");
    setForecastAssumptionsState(next);
  }, []);

  const setDriverDeltas = useCallback((next: DriverDeltaPercentages) => {
    setPresetId("custom");
    setManualSimulatedInputs(null);
    setDriverDeltasState(next);
  }, []);

  const setExactSimulatedInputs = useCallback(
    (next: PredictionInputs) => {
      setPresetId("custom");
      setManualSimulatedInputs(next);
      // Reflect percentage delta back onto driverDeltas so cross-page widgets remain synchronized
      const pct = (cur: number, base: number) =>
        Math.abs(base) > 1e-6 ? Number((((cur - base) / base) * 100).toFixed(1)) : 0;
      setDriverDeltasState({
        fxPct: pct(next.USD_NGN_Planning_Rate, baselineInputs.USD_NGN_Planning_Rate),
        pricePct: pct(next.Avg_Selling_Price_NGN, baselineInputs.Avg_Selling_Price_NGN),
        unitsPct: pct(next.Units_Sold, baselineInputs.Units_Sold),
        impUsdPct: pct(next.Import_Cost_USD_Per_Unit, baselineInputs.Import_Cost_USD_Per_Unit),
        mktPct: pct(next.Marketing_Spend_NGN, baselineInputs.Marketing_Spend_NGN),
        opexPct: pct(next.Other_Operating_Cost_NGN, baselineInputs.Other_Operating_Cost_NGN),
        inflationDeltaPp: Number((next.Inflation_YoY_Pct - baselineInputs.Inflation_YoY_Pct).toFixed(2)),
      });
    },
    [baselineInputs]
  );

  const applyPreset = useCallback((id: Exclude<ScenarioPresetId, "custom">) => {
    const spec = SCENARIO_PRESETS.find((p) => p.id === id) || SCENARIO_PRESETS[0];
    setPresetId(spec.id);
    setManualSimulatedInputs(null);
    setForecastAssumptionsState(spec.forecast);
    setDriverDeltasState(spec.deltas);
  }, []);

  const resetScenario = useCallback(() => {
    applyPreset("base");
  }, [applyPreset]);

  const refreshSimulation = useCallback(() => {
    setRefreshTick((t) => t + 1);
  }, []);

  const value: ScenarioContextValue = {
    presetId,
    activePresetLabel,
    isScenarioActive,
    anchorRecord,
    setAnchorRecord,
    forecastAssumptions,
    setForecastAssumptions,
    driverDeltas,
    setDriverDeltas,
    baselineInputs,
    simulatedInputs,
    setExactSimulatedInputs,
    applyPreset,
    resetScenario,
    forecastScenarios,
    baseVerdict,
    customVerdict,
    singlePeriodResult,
    forecastLoading,
    singleLoading,
    drawerOpen,
    setDrawerOpen,
    refreshSimulation,
  };

  return <ScenarioContext.Provider value={value}>{children}</ScenarioContext.Provider>;
};

export function useScenario(): ScenarioContextValue {
  const ctx = useContext(ScenarioContext);
  if (!ctx) {
    throw new Error("useScenario must be used within ScenarioProvider");
  }
  return ctx;
}
