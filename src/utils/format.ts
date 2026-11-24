export function formatNGN(value: number | null | undefined, compact = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (compact) {
    if (abs >= 1e9) {
      return `${sign}₦${(abs / 1e9).toFixed(2)}B`;
    }
    if (abs >= 1e6) {
      return `${sign}₦${(abs / 1e6).toFixed(1)}M`;
    }
    if (abs >= 1e3) {
      return `${sign}₦${(abs / 1e3).toFixed(1)}K`;
    }
  }
  return `${sign}₦${abs.toLocaleString("en-NG", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  })}`;
}

export function formatNGNExact(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  return `${sign}₦${abs.toLocaleString("en-NG", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

export function formatCompactNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1e6) {
    return `${sign}${(abs / 1e6).toFixed(2)}M`;
  }
  if (abs >= 1e3) {
    return `${sign}${(abs / 1e3).toFixed(1)}K`;
  }
  return `${sign}${abs.toLocaleString("en-US")}`;
}

export function formatPct(value: number | null | undefined, decimals = 2, signed = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(decimals)}%`;
}

export function formatFX(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return `₦${value.toLocaleString("en-NG", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}/$`;
}
