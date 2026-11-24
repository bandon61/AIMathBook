"""日次ジョブ: スクリーニング(A) + 異常検知(C) をまとめて実行する。

使い方:
  python run_daily.py
  python run_daily.py --config config.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from anomaly import format_alert_text, run_anomaly, save_anomaly_result
from common import ensure_output_dir, load_config
from screen import run_screen, save_screen_result


def main() -> None:
    parser = argparse.ArgumentParser(description="株価スクリーニングと異常検知を日次実行")
    parser.add_argument("--config", default=None, help="設定YAMLのパス")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_output_dir(config)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"=== 株価オペレーション日次実行 ({now}) ===")

    # A: スクリーニング
    screen_df = run_screen(config)
    if screen_df.empty:
        print("[A] スクリーニング: データなし")
    else:
        passed = screen_df[screen_df["passed"]]
        path = save_screen_result(passed, out_dir)
        print(f"[A] 合格 {len(passed)} / 監視 {len(screen_df)} → {path}")
        if not passed.empty:
            print(passed[["ticker", "close", "day_return_pct", "volume_ratio"]].to_string(index=False))

    print()

    # C: 異常検知
    anomaly_df = run_anomaly(config)
    alert = format_alert_text(anomaly_df)
    csv_path, txt_path = save_anomaly_result(anomaly_df, out_dir, alert)
    print("[C] 異常検知")
    print(alert)
    print(f"→ {csv_path}")
    print(f"→ {txt_path}")


if __name__ == "__main__":
    main()
