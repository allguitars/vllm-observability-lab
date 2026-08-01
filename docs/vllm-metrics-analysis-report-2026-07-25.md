# vLLM Metrics 收集與分析報告

- 日期：2026-07-25
- 執行環境：應用伺服器連線至 H200 GPU Server 的 vLLM `/metrics`
- 收集工具：`collect_vllm_metrics.py`
- 輸出格式：CSV，後續使用 Excel 分析

## 1. 目標

本次工作目標是建立一套簡單、可人工控制的 vLLM 監控流程：

1. 翻譯開始前啟動收集腳本。
2. 定期呼叫 vLLM `/metrics`。
3. 將取樣結果即時顯示於終端，並同步寫入 CSV。
4. 翻譯完成後按 `Ctrl+C` 停止。
5. 使用 Excel 分析 Token throughput、KV cache 使用率與 request 狀態。

此工具定位為用途單純的一次性監控腳本，不導入額外的監控框架或複雜抽象。

## 2. 收集的 Metrics

目前收集以下 vLLM Metrics：

| Metric | 類型 | 用途 |
| --- | --- | --- |
| `vllm:num_requests_running` | Gauge | 取樣當下正在執行的 request 數 |
| `vllm:num_requests_waiting` | Gauge | 取樣當下等待處理的 request 數 |
| `vllm:kv_cache_usage_perc` | Gauge | 取樣當下的 KV cache 使用比例 |
| `vllm:prompt_tokens_total` | Counter | vLLM 啟動後累積處理的 Prompt Token |
| `vllm:generation_tokens_total` | Counter | vLLM 啟動後累積產生的 Generation Token |

Gauge 若出現多條 series，目前沒有已確認的聚合規則，因此腳本會直接拋出錯誤，避免自行使用 `sum`、`average` 或 `max`。

Prompt 與 Generation Counter 可能因 model 或 engine label 出現多條 series，依本次設計將各 series 加總為 endpoint 的總值。

## 3. CSV 欄位

CSV 包含以下欄位：

```text
timestamp
num_requests_running
num_requests_waiting
kv_cache_usage_perc
prompt_tokens_total
prompt_tokens_delta
generation_tokens_total
generation_tokens_delta
interval_seconds
prompt_tokens_per_second
generation_tokens_per_second
```

`timestamp` 使用本地時區並保留毫秒；`interval_seconds` 使用 `time.monotonic()` 計算，避免系統時間調整影響區間長度。

## 4. Token 計算方式

### 4.1 Prompt Token 增量

```text
prompt_tokens_delta
= 本次 prompt_tokens_total - 上次 prompt_tokens_total
```

### 4.2 Generation Token 增量

```text
generation_tokens_delta
= 本次 generation_tokens_total - 上次 generation_tokens_total
```

### 4.3 共用取樣間隔

Prompt 與 Generation Metrics 來自同一次 `/metrics` 呼叫，因此共用：

```text
interval_seconds
= 本次取樣時間 - 上次取樣時間
```

### 4.4 每個區間的 Throughput

```text
prompt_tokens_per_second
= prompt_tokens_delta / interval_seconds
```

```text
generation_tokens_per_second
= generation_tokens_delta / interval_seconds
```

第一筆取樣沒有上一筆資料可比較，因此 delta、interval 與 throughput 留空。

若 Counter 變小，通常表示 vLLM 重啟或 Counter 重設。腳本會讓該 Counter 當次的 delta 與 throughput 留空，避免產生錯誤的負值。

## 5. 整體平均 Throughput

直接平均每列的 `tokens_per_second`，只有在每個取樣區間完全相同時才會等於整體平均。此次取樣間隔大多接近，因此普通平均與正式結果很接近，但正式計算仍採用：

```text
整體 Throughput = SUM(delta) / SUM(interval_seconds)
```

這相當於依每個區間的實際長度進行加權平均。

本次測量結果：

| 指標 | 結果 |
| --- | ---: |
| 平均 Prompt Throughput | 5,708.42 tokens/s |
| 平均 Generation Throughput | 405.51 tokens/s |

資料範圍應從有效工作區間的第一筆變化開始，到最後一筆變化結束。開始前與結束後的 idle 資料可以排除，但工作區間內的合法零值應保留。

## 6. 零值的解讀

CSV 中的 `0` 是有效資料，不代表腳本失敗：

| 欄位 | `0` 的意義 |
| --- | --- |
| `num_requests_running` | 取樣當下沒有執行中的 request |
| `num_requests_waiting` | 取樣當下沒有 request 排隊 |
| `kv_cache_usage_perc` | 取樣當下 KV cache 已釋放或接近未使用 |
| `prompt_tokens_delta` | 該區間沒有新增 Prompt Token |
| `generation_tokens_delta` | 該區間沒有新增 Generation Token |
| `tokens_per_second` | 對應 Counter 在該區間的 delta 為 0 |

可能出現 `num_requests_running = 0`，但 `generation_tokens_delta > 0`。這是因為 Gauge 表示取樣當下的瞬間狀態，而 delta 表示前一個取樣區間內的累積變化。Request 可能在取樣前剛完成。

## 7. KV Cache 分析

本次有效工作區間的 KV cache 統計結果：

| 統計量 | 結果 |
| --- | ---: |
| 平均值 | 1.55% |
| P95 | 2.89% |
| 最大值 | 3.91% |

Excel 的 P95 公式為：

```excel
=PERCENTILE.INC(資料範圍, 0.95)
```

結果顯示此次工作負載的 KV cache 使用率偏低。約 95% 的取樣點不超過 2.89%，觀測到的最高值為 3.91%。

取樣約每 1.25 秒一次，非常短暫且發生於兩次取樣之間的尖峰可能未被捕捉，因此上述數據應解讀為「觀測到的 KV cache 壓力」。

## 8. 為何 KV Cache 與 Token Counter 無法直接對帳

使用：

```text
KV cache token capacity × kv_cache_usage_perc
```

得到的數值較接近取樣當下已配置的 KV token slots，不等於某個區間處理的 Prompt 或 Generation Token 數量。

兩者無法直接對帳的主要原因包括：

- Token Counter 是 vLLM 啟動後的累積流量，KV cache usage 是瞬間存量。
- Request 完成後 KV blocks 會釋放，但 Token Counter 不會減少。
- KV cache 以 block 為單位配置，可能包含尚未填滿的空間。
- Prefix caching 可能重用既有 KV。
- 多個 request 可能同時處於 prefill、decode 或完成階段。
- Tensor Parallel、Pipeline Parallel、worker 或 engine 可能具有不同的資料範圍。
- `/metrics` 雖由一次 HTTP 呼叫取得，但各指標不一定是跨所有平行單元的原子快照。

因此，不應要求 Token Counter 與 KV cache capacity 精確相等，而應讓不同指標回答不同問題。

## 9. 建議的指標使用方式

| 分析問題 | 建議指標 |
| --- | --- |
| Prompt／Prefill 處理速度 | `prompt_tokens_per_second` |
| Generation／Decode 生成速度 | `generation_tokens_per_second` |
| KV cache 容量壓力 | `kv_cache_usage_perc` 的平均、P95、最大值 |
| 同時處理量 | `num_requests_running` 的平均與最大值 |
| 是否發生排隊 | `num_requests_waiting` 是否曾大於 0 |

適合觀察的關係包括：

- Running requests 增加時，KV cache 是否同步上升。
- KV cache 接近高水位時，Waiting requests 是否出現。
- Concurrency 增加時，Generation throughput 如何變化。
- 工作完成後，KV cache 多久回落至接近零。

這些關係適合用於趨勢與容量分析，不應被解讀為可互相精確換算。

## 10. 結論

本次腳本已能以單次 vLLM `/metrics` 呼叫，同步收集 request、KV cache、Prompt Token 與 Generation Token 資料，並產生可直接由 Excel 分析的 CSV。

本次觀測結果顯示：

- 平均 Prompt Throughput 為 5,708.42 tokens/s。
- 平均 Generation Throughput 為 405.51 tokens/s。
- KV cache 平均使用率為 1.55%，P95 為 2.89%，最大值為 3.91%。
- 從觀測數據看，此次工作負載未呈現明顯的 KV cache 容量壓力。

後續分析應持續聚焦於各指標本身能回答的問題，並使用時間序列觀察其趨勢關聯，不強求不同語意的 Metrics 彼此精確對帳。
