"""A: 日次スクリーニング。

条件を満たす銘柄をCSVに出力する。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    ensure_output_dir,
    fetch_history,
    latest_row_metrics,
    load_config,
)


def evaluate_ticker(ticker: str, df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """スクリーニング条件を評価し、通過なら結果dictを返す。"""
    ma_short = int(cfg["ma_short"])
    ma_long = int(cfg["ma_long"])
    vol_days = int(cfg["volume_ma_days"])
    vol_min = float(cfg["volume_min_ratio"])
    min_ret = float(cfg["min_return_pct"])

    metrics = latest_row_metrics(df, ma_short, ma_long, vol_days)
    short_key = f"ma{ma_short}"
    long_key = f"ma{ma_long}"
    ret_key = f"return_{ma_short}d_pct"

    close = metrics["close"]
    ma_s = metrics[short_key]
    ma_l = metrics[long_key]
    vol_ratio = metrics["volume_ratio"]
    ret_n = metrics[ret_key]

    if None in (ma_s, ma_l, vol_ratio, ret_n):
        return None

    # トレンド: 終値 > 短期MA > 長期MA
    trend_ok = close > ma_s > ma_l
    # 出来高: 直近平均比が閾値以上
    volume_ok = vol_ratio >= vol_min
    # リターン: 短期リターンが下限以上
    return_ok = ret_n >= min_ret

    passed = trend_ok and volume_ok and return_ok
    row = {
        "ticker": ticker,
        "passed": passed,
        "trend_ok": trend_ok,
        "volume_ok": volume_ok,
        "return_ok": return_ok,
        **metrics,
    }
    return row


def run_screen(config: dict[str, Any] | None = None, config_path: str | None = None) -> pd.DataFrame:
    """全銘柄をスクリーニングしてDataFrameを返す。"""
    config = config or load_config(config_path)
    histories = fetch_history(config["tickers"], int(config.get("lookback_days", 120)))
    screen_cfg = config["screen"]

    rows: list[dict[str, Any]] = []
    for ticker, df in histories.items():
        row = evaluate_ticker(ticker, df, screen_cfg)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values(
        by=["passed", "day_return_pct"],
        ascending=[False, False],
    )
    return result.reset_index(drop=True)


def save_screen_result(df: pd.DataFrame, output_dir: Path) -> Path:
    """結果をCSV保存する。"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = output_dir / f"screen_{ts}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="日次スクリーニングを実行する")
    parser.add_argument("--config", default=None, help="設定YAMLのパス")
    parser.add_argument(
        "--all",
        action="store_true",
        help="不合格銘柄もCSVに含める（デフォルトは合格のみ）",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_output_dir(config)
    df = run_screen(config)

    if df.empty:
        print("スクリーニング結果: 対象データなし")
        return

    passed = df[df["passed"]].copy()
    export = df if args.all else passed
    path = save_screen_result(export, out_dir)

    print(f"監視銘柄数: {len(df)}")
    print(f"合格銘柄数: {len(passed)}")
    if not passed.empty:
        print("--- 合格銘柄 ---")
        cols = ["ticker", "date", "close", "day_return_pct", "volume_ratio"]
        print(passed[cols].to_string(index=False))
    else:
        print("本日の合格銘柄はありません。")
    print(f"保存先: {path}")


if __name__ == "__main__":
    main()
