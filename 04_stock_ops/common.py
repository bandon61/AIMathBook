"""株価分析の共通処理（データ取得・設定読み込み）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """YAML設定を読み込む。"""
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(config: dict[str, Any]) -> Path:
    """出力ディレクトリを作成して返す。"""
    out = ROOT / config.get("output_dir", "output")
    out.mkdir(parents=True, exist_ok=True)
    return out


def fetch_history(
    tickers: list[str],
    lookback_days: int = 120,
) -> dict[str, pd.DataFrame]:
    """各銘柄の日足OHLCVを取得する。失敗した銘柄はスキップする。"""
    # 余裕を見てカレンダー日数を多めに取る（祝日・休場分）
    period_days = max(lookback_days + 40, 60)
    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                period=f"{period_days}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001 - 銘柄単位で継続したい
            print(f"[WARN] {ticker}: 取得失敗 ({exc})")
            continue

        if df is None or df.empty:
            print(f"[WARN] {ticker}: データなし")
            continue

        # MultiIndex列になる場合があるのでフラット化
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        needed = ["open", "high", "low", "close", "volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] {ticker}: 列不足 {missing}")
            continue

        df = df[needed].dropna().copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if len(df) < 30:
            print(f"[WARN] {ticker}: データ不足 ({len(df)}行)")
            continue

        result[ticker] = df

    return result


def latest_row_metrics(df: pd.DataFrame, ma_short: int, ma_long: int, vol_ma: int) -> dict[str, Any]:
    """最新営業日の指標を計算する。"""
    close = df["close"]
    volume = df["volume"]
    short = close.rolling(ma_short).mean()
    long = close.rolling(ma_long).mean()
    vol_avg = volume.rolling(vol_ma).mean()

    last = df.index[-1]
    close_last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else close_last
    day_ret = (close_last / prev - 1.0) * 100.0 if prev else 0.0

    short_last = float(short.iloc[-1]) if pd.notna(short.iloc[-1]) else None
    long_last = float(long.iloc[-1]) if pd.notna(long.iloc[-1]) else None
    vol_last = float(volume.iloc[-1])
    vol_avg_last = float(vol_avg.iloc[-1]) if pd.notna(vol_avg.iloc[-1]) else None
    vol_ratio = (vol_last / vol_avg_last) if vol_avg_last else None

    ret_n = None
    if len(close) > ma_short:
        ret_n = (close_last / float(close.iloc[-(ma_short + 1)]) - 1.0) * 100.0

    return {
        "date": last.strftime("%Y-%m-%d"),
        "close": round(close_last, 4),
        "day_return_pct": round(day_ret, 3),
        f"ma{ma_short}": round(short_last, 4) if short_last is not None else None,
        f"ma{ma_long}": round(long_last, 4) if long_last is not None else None,
        "volume": int(vol_last),
        "volume_ma": int(vol_avg_last) if vol_avg_last is not None else None,
        "volume_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
        f"return_{ma_short}d_pct": round(ret_n, 3) if ret_n is not None else None,
    }
