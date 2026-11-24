"""C: 異常検知アラート。

急騰・急落・出来高スパイク・統計的外れ値を検知して通知用テキスト/CSVを出力する。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ensure_output_dir, fetch_history, load_config


def detect_anomalies(ticker: str, df: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """1銘柄について異常イベントを検知する。"""
    price_thr = float(cfg["price_change_pct"])
    vol_ratio_thr = float(cfg["volume_spike_ratio"])
    vol_days = int(cfg["volume_ma_days"])
    z_window = int(cfg["zscore_window"])
    z_thr = float(cfg["zscore_threshold"])

    close = df["close"]
    volume = df["volume"]
    day_ret = close.pct_change() * 100.0
    vol_ma = volume.rolling(vol_days).mean()
    vol_ratio = volume / vol_ma
    ret_mean = day_ret.rolling(z_window).mean()
    ret_std = day_ret.rolling(z_window).std()
    zscore = (day_ret - ret_mean) / ret_std.replace(0, np.nan)

    i = -1
    date = df.index[i].strftime("%Y-%m-%d")
    close_last = float(close.iloc[i])
    ret_last = float(day_ret.iloc[i]) if pd.notna(day_ret.iloc[i]) else 0.0
    vol_r = float(vol_ratio.iloc[i]) if pd.notna(vol_ratio.iloc[i]) else None
    z_last = float(zscore.iloc[i]) if pd.notna(zscore.iloc[i]) else None

    events: list[dict[str, Any]] = []

    def base(kind: str, detail: str, severity: str) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "date": date,
            "kind": kind,
            "severity": severity,
            "detail": detail,
            "close": round(close_last, 4),
            "day_return_pct": round(ret_last, 3),
            "volume_ratio": round(vol_r, 3) if vol_r is not None else None,
            "zscore": round(z_last, 3) if z_last is not None else None,
        }

    if abs(ret_last) >= price_thr:
        direction = "急騰" if ret_last > 0 else "急落"
        events.append(
            base(
                "price_move",
                f"{direction}: 日次リターン {ret_last:+.2f}% (閾値 ±{price_thr}%)",
                "high" if abs(ret_last) >= price_thr * 1.5 else "medium",
            )
        )

    if vol_r is not None and vol_r >= vol_ratio_thr:
        events.append(
            base(
                "volume_spike",
                f"出来高スパイク: 平均比 {vol_r:.2f}x (閾値 {vol_ratio_thr}x)",
                "high" if vol_r >= vol_ratio_thr * 1.5 else "medium",
            )
        )

    if z_last is not None and abs(z_last) >= z_thr:
        events.append(
            base(
                "zscore",
                f"統計的異常: z={z_last:+.2f} (閾値 ±{z_thr})",
                "high" if abs(z_last) >= z_thr + 1 else "medium",
            )
        )

    return events


def run_anomaly(config: dict[str, Any] | None = None, config_path: str | None = None) -> pd.DataFrame:
    """全銘柄の異常検知を実行する。"""
    config = config or load_config(config_path)
    histories = fetch_history(config["tickers"], int(config.get("lookback_days", 120)))
    anomaly_cfg = config["anomaly"]

    events: list[dict[str, Any]] = []
    for ticker, df in histories.items():
        events.extend(detect_anomalies(ticker, df, anomaly_cfg))

    if not events:
        return pd.DataFrame()

    order = {"high": 0, "medium": 1, "low": 2}
    result = pd.DataFrame(events)
    result["_sev"] = result["severity"].map(order)
    result = result.sort_values(by=["_sev", "ticker"]).drop(columns=["_sev"])
    return result.reset_index(drop=True)


def format_alert_text(df: pd.DataFrame) -> str:
    """Slack等に貼れる簡易テキストを作る。"""
    if df.empty:
        return "【異常検知】本日のアラートはありません。"

    lines = ["【異常検知アラート】"]
    for _, row in df.iterrows():
        lines.append(
            f"- [{row['severity']}] {row['ticker']} {row['date']}: {row['detail']}"
        )
    return "\n".join(lines)


def save_anomaly_result(df: pd.DataFrame, output_dir: Path, alert_text: str) -> tuple[Path, Path]:
    """CSVとアラートテキストを保存する。"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    csv_path = output_dir / f"anomaly_{ts}.csv"
    txt_path = output_dir / f"anomaly_alert_{ts}.txt"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    txt_path.write_text(alert_text + "\n", encoding="utf-8")
    return csv_path, txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="異常検知アラートを実行する")
    parser.add_argument("--config", default=None, help="設定YAMLのパス")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_output_dir(config)
    df = run_anomaly(config)
    alert = format_alert_text(df)
    csv_path, txt_path = save_anomaly_result(df, out_dir, alert)

    print(alert)
    print(f"CSV: {csv_path}")
    print(f"通知文: {txt_path}")


if __name__ == "__main__":
    main()
