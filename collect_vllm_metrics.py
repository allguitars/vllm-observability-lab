#!/usr/bin/env python3

"""定期收集 vLLM Metrics，並將即時數值與 Token 速率寫入 CSV。

腳本每秒向 vLLM 的 /metrics endpoint 取樣一次，同時在終端顯示結果。
使用者可在工作開始前執行腳本，
並在工作完成後按 Ctrl+C 正常結束。
"""

import csv
import sys
import time
from datetime import datetime

import requests

METRICS_URL = "http://192.168.130.178:8000/metrics"
INTERVAL_SECONDS = 1
REQUEST_TIMEOUT_SECONDS = 3

TARGET_METRICS = {
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
}

PROMPT_TOKENS_METRIC = "vllm:prompt_tokens_total"
GENERATION_TOKENS_METRIC = "vllm:generation_tokens_total"
COUNTER_METRICS = {
    PROMPT_TOKENS_METRIC,
    GENERATION_TOKENS_METRIC,
}


def parse_metrics(metrics_text):
    """解析 vLLM Prometheus text format，回傳指定 Metrics 的數值。"""

    collected_values = {
        metric_name: []
        for metric_name in TARGET_METRICS
    }

    for line in metrics_text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        metric_with_labels = parts[0]
        metric_name = metric_with_labels.split("{", 1)[0]

        if metric_name not in TARGET_METRICS:
            continue

        try:
            value = float(parts[1])
        except ValueError:
            continue

        collected_values[metric_name].append(value)

    result = {}

    for metric_name, csv_column in TARGET_METRICS.items():
        values = collected_values[metric_name]

        if not values:
            raise ValueError(f"找不到指標：{metric_name}")

        # Counter 可能因 model 或 engine label 出現多條 series，
        # 因此需合計總值。
        if metric_name in COUNTER_METRICS:
            result[csv_column] = sum(values)
            continue

        # Gauge 的多條 series 沒有已確認的聚合規則，避免自行推測。
        if len(values) > 1:
            raise ValueError(
                f"指標出現多筆資料：{metric_name}，"
                f"目前值為 {values}"
            )

        result[csv_column] = values[0]

    return result


def fetch_metrics():
    """從 vLLM endpoint 取得並解析本次 Metrics。"""

    response = requests.get(
        METRICS_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    return parse_metrics(response.text)


def main():
    """持續收集 Metrics，輸出 CSV，直到使用者按下 Ctrl+C。"""

    output_file = f"vllm_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    columns = [
        "timestamp",
        "num_requests_running",
        "num_requests_waiting",
        "kv_cache_usage_perc",
        "prompt_tokens_total",
        "prompt_tokens_delta",
        "generation_tokens_total",
        "generation_tokens_delta",
        "interval_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    ]
    previous_prompt_tokens_total = None
    previous_generation_tokens_total = None
    previous_sample_time = None

    with open(
        output_file,
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=columns,
        )
        writer.writeheader()

        print(f"Metrics URL：{METRICS_URL}")
        print(f"輸出檔案：{output_file}")
        print("開始收集，每秒一次。按 Ctrl+C 結束。")

        try:
            while True:
                # Prompt 與 Generation Counter 共用同一個單調時鐘取樣點。
                sample_time = time.monotonic()
                timestamp = datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                )

                try:
                    metrics = fetch_metrics()

                    prompt_tokens_total = metrics["prompt_tokens_total"]
                    generation_tokens_total = metrics["generation_tokens_total"]
                    prompt_tokens_delta = None
                    generation_tokens_delta = None
                    interval_seconds = None
                    prompt_tokens_per_second = None
                    generation_tokens_per_second = None

                    if (
                        previous_prompt_tokens_total is not None
                        and previous_generation_tokens_total is not None
                        and previous_sample_time is not None
                    ):
                        # 使用實際經過時間計算速率，
                        # 不假設間隔恰好為一秒。
                        interval_seconds = sample_time - previous_sample_time

                        if prompt_tokens_total >= previous_prompt_tokens_total:
                            prompt_tokens_delta = (
                                prompt_tokens_total
                                - previous_prompt_tokens_total
                            )
                            prompt_tokens_per_second = (
                                prompt_tokens_delta / interval_seconds
                                if interval_seconds > 0
                                else None
                            )

                        if (
                            generation_tokens_total
                            >= previous_generation_tokens_total
                        ):
                            generation_tokens_delta = (
                                generation_tokens_total
                                - previous_generation_tokens_total
                            )
                            generation_tokens_per_second = (
                                generation_tokens_delta / interval_seconds
                                if interval_seconds > 0
                                else None
                            )

                    row = {
                        "timestamp": timestamp,
                        "num_requests_running": metrics["num_requests_running"],
                        "num_requests_waiting": metrics["num_requests_waiting"],
                        "kv_cache_usage_perc": round(
                            metrics["kv_cache_usage_perc"],
                            6,
                        ),
                        "prompt_tokens_total": round(prompt_tokens_total, 3),
                        "prompt_tokens_delta": (
                            round(prompt_tokens_delta, 3)
                            if prompt_tokens_delta is not None
                            else ""
                        ),
                        "generation_tokens_total": round(
                            generation_tokens_total,
                            3,
                        ),
                        "generation_tokens_delta": (
                            round(generation_tokens_delta, 3)
                            if generation_tokens_delta is not None
                            else ""
                        ),
                        "interval_seconds": (
                            round(interval_seconds, 3)
                            if interval_seconds is not None
                            else ""
                        ),
                        "prompt_tokens_per_second": (
                            round(prompt_tokens_per_second, 3)
                            if prompt_tokens_per_second is not None
                            else ""
                        ),
                        "generation_tokens_per_second": (
                            round(generation_tokens_per_second, 3)
                            if generation_tokens_per_second is not None
                            else ""
                        ),
                    }

                    writer.writerow(row)
                    csv_file.flush()

                    # Counter 變小通常代表 vLLM 重啟，
                    # 本筆不計算負增量。
                    if (
                        previous_prompt_tokens_total is not None
                        and prompt_tokens_total < previous_prompt_tokens_total
                    ):
                        print(
                            "偵測到 prompt_tokens_total 重設，"
                            "本筆不計算增量。"
                        )

                    if (
                        previous_generation_tokens_total is not None
                        and generation_tokens_total
                        < previous_generation_tokens_total
                    ):
                        print(
                            "偵測到 generation_tokens_total 重設，"
                            "本筆不計算增量。"
                        )

                    prompt_delta_text = (
                        row["prompt_tokens_delta"]
                        if row["prompt_tokens_delta"] != ""
                        else "-"
                    )
                    prompt_rate_text = (
                        f"{row['prompt_tokens_per_second']} tokens/s"
                        if row["prompt_tokens_per_second"] != ""
                        else "-"
                    )
                    generation_delta_text = (
                        row["generation_tokens_delta"]
                        if row["generation_tokens_delta"] != ""
                        else "-"
                    )
                    generation_rate_text = (
                        f"{row['generation_tokens_per_second']} tokens/s"
                        if row["generation_tokens_per_second"] != ""
                        else "-"
                    )

                    print(
                        f"{row['timestamp']} | "
                        f"running={row['num_requests_running']} | "
                        f"waiting={row['num_requests_waiting']} | "
                        f"kv_cache={row['kv_cache_usage_perc']} | "
                        f"prompt_total={row['prompt_tokens_total']} | "
                        f"prompt_delta={prompt_delta_text} | "
                        f"prompt_rate={prompt_rate_text} | "
                        f"generation_total={row['generation_tokens_total']} | "
                        f"generation_delta={generation_delta_text} | "
                        f"generation_rate={generation_rate_text}"
                    )

                    previous_prompt_tokens_total = prompt_tokens_total
                    previous_generation_tokens_total = generation_tokens_total
                    previous_sample_time = sample_time

                except (requests.RequestException, ValueError) as error:
                    print(
                        f"{timestamp} | 收集失敗：{error}",
                        file=sys.stderr,
                    )

                    # 保留失敗時間點，
                    # 方便後續在 Excel 對照資料缺口。
                    writer.writerow(
                        {
                            "timestamp": timestamp,
                            "num_requests_running": "",
                            "num_requests_waiting": "",
                            "kv_cache_usage_perc": "",
                            "prompt_tokens_total": "",
                            "prompt_tokens_delta": "",
                            "generation_tokens_total": "",
                            "generation_tokens_delta": "",
                            "interval_seconds": "",
                            "prompt_tokens_per_second": "",
                            "generation_tokens_per_second": "",
                        }
                    )
                    csv_file.flush()

                time.sleep(INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print(f"\n停止收集，CSV 已儲存：{output_file}")


if __name__ == "__main__":
    main()
