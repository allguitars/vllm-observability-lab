# vLLM GPU Prefix Cache 與 aiDAPTIVCache External KV 命中分析報告

- 報告日期：2026-08-05
- 實驗日期：2026-08-03
- 模型：Gemma 3 27B
- 對外模型名稱：`gemma3-27b-with-middleware`
- 測試序列：`A1 → A2 → B1 → A3`

## 技術摘要

本次實驗以兩個互不相同、各約 7,600～7,700 tokens 的長 Prompt，驗證
vLLM GPU Automatic Prefix Caching 與 aiDAPTIVLink external KV cache 的分層
命中行為。

結果顯示：A1 與 B1 都是冷啟動；A2 的 7,713-token Prompt 有 7,712 tokens
命中本機 GPU Prefix Cache；B1 進入 GPU 後使 Prefix A 的多數 blocks 被回收；
A3 再次送出 Prefix A 時，本機 GPU 命中 2,384 tokens，external cache 補回
5,328 tokens，合計重用 7,712 tokens，只剩最後 1 個 Prompt token 需要計算。

五份 Prometheus 快照計算出的本輪累積 local hit rate 為 32.833%，external
hit rate 為 17.327%；這與 middleware log 顯示的 32.8% 與 17.3% 一致。
API response 的 prefill latency 也支持相同判讀：A2 的 GPU hit 最快，A3 的
external hit 慢於 A2，但仍明顯快於 A1 冷算。

因此，本輪資料足以確認 external KV connector 在 A3 實際貢獻了 5,328 個
cached tokens。若要進一步把 external hit 嚴格等同於「實體 SSD read」，仍需
加入 SSD read bytes 或更底層的 aiDAPTIVLink I/O log。

## 1. 實驗背景與驗證問題

H200 上的 Gemma 3 27B 配置如下：

| 項目 | 數值 |
| --- | ---: |
| Available KV Cache | 4.74 GiB |
| KV block size | 16 tokens |
| GPU KV blocks | 626 |
| GPU token capacity | 10,016 tokens |
| `max_model_len` | 8,192 tokens |
| aiDAPTIVLink SSD KV cache | 100 GB |

Prometheus 的 `vllm:cache_config_info` 同時確認：

```text
block_size="16"
num_gpu_blocks="626"
enable_prefix_caching="True"
```

因此 GPU token capacity 可直接驗算為：

```text
16 × 626 = 10,016 tokens
```

Prefix A 與 Prefix B 合計超過 15,000 tokens，無法完整同時保留在只有
10,016-token capacity 的 GPU KV cache。實驗要回答三個問題：

1. 相同的 A2 是否能從 GPU Prefix Cache 命中？
2. B1 是否會使 Prefix A 的 GPU blocks 被部分淘汰？
3. A3 能否從 aiDAPTIVLink external cache 補回被淘汰的 blocks？

## 2. 實驗資料與 Request 序列

使用兩份固定 Prompt：

| Prompt | 檔案 | 實際 Prompt tokens |
| --- | --- | ---: |
| A | `prompts/prefix_a_distributed_systems.txt` | 7,713 |
| B | `prompts/prefix_b_financial_risk.txt` | 7,611 |

四個 Request 均序列執行，前一個完成後才送出下一個：

```text
A1：第一次送 A，建立 cache
A2：再次送 A，驗證 GPU local hit
B1：送不同的 B，對 GPU cache 製造容量壓力
A3：再次送 A，驗證 external cache hit
```

本報告使用三類資料交叉驗證：

1. `prefix-cache-test-AABA-response-payload.txt`
   - 四次 `send_prefix_prompt.py` 收到的完整 API response。
   - 包含 token usage、client elapsed time，以及 vLLM 回傳的 prefill、decode、
     inference 與 e2e latency。
2. `prefix-cache-test-AABA-middleware-log.txt`
   - 以 response ID 對齊四次 request。
   - 包含逐 request 的 GPU cache hit、aiDAPTIVCache hit 與 store 路徑資訊。
3. `metrics_M0_before_A1.prom.txt` 至 `metrics_M4_after_A3.prom.txt`
   - 五個完整的 vLLM `/metrics` Prometheus snapshots。
   - 用相鄰 snapshot 的 Counter 差值還原每次 request 的命中數。

## 3. 指標定義與計算方法

### 3.1 API response 欄位

| 欄位 | 意義 |
| --- | --- |
| `prompt_tokens` | 套用 chat template 後的 Prompt token 數 |
| `completion_tokens` | 本次實際生成的 token 數 |
| `elapsed_seconds` | Client 觀察到的完整 HTTP request 時間 |
| `metrics.prefill_time` | vLLM 處理 Prompt／Prefix 的時間 |
| `metrics.decode_time` | vLLM decode 階段時間 |
| `metrics.inference_time` | vLLM 推論時間 |
| `metrics.e2e_latency` | vLLM server 端端到端 request latency |

API response 的 `cached_tokens` 與 `cache_write_tokens` 都是 `null`，因此不能
單靠 OpenAI-compatible usage schema 判斷 cache 層級。Cache 判讀必須交叉使用
middleware log 與 vLLM Prometheus metrics。

### 3.2 Prefix Cache Prometheus 指標

| Metric | 類型 | 單位與用途 |
| --- | --- | --- |
| `vllm:prefix_cache_queries_total` | Counter | 本機 Prefix Cache queried tokens |
| `vllm:prefix_cache_hits_total` | Counter | 本機 GPU Prefix Cache hit tokens |
| `vllm:external_prefix_cache_queries_total` | Counter | KV Connector external queried tokens |
| `vllm:external_prefix_cache_hits_total` | Counter | External Cache hit tokens |
| `vllm:prompt_tokens_total` | Counter | 累積處理的 Prompt tokens |
| `vllm:generation_tokens_total` | Counter | 累積處理的 Generation tokens |
| `vllm:kv_cache_usage_perc` | Gauge | 取樣當下 active request 的 KV cache usage |

所有 `_created` series 只是 Counter 建立時間，不代表 Cache 建立或命中數量，
因此不納入分析。

### 3.3 Snapshot 差值

取樣順序為：

```text
M0 → A1 → M1 → A2 → M2 → B1 → M3 → A3 → M4
```

每個 request 的 Counter 增量使用相鄰快照相減：

```text
A1 = M1 - M0
A2 = M2 - M1
B1 = M3 - M2
A3 = M4 - M3
```

M0 並非全零，已包含 7 個 Prompt/query tokens 與 10 個 Generation tokens。
因此必須扣除 M0，不能把 M4 的絕對累積值直接當成本輪結果。

每次 request 的命中率定義為：

```text
Local hit rate
= Δprefix_cache_hits_total / Δprefix_cache_queries_total

External contribution rate
= Δexternal_prefix_cache_hits_total / Δprefix_cache_queries_total

Combined reuse rate
= (Δprefix_cache_hits_total + Δexternal_prefix_cache_hits_total)
  / Δprefix_cache_queries_total
```

## 4. API Response 顯示 A2 最快、A3 次之

四次 response 的 Prompt 與 Generation tokens 與預期一致：

| Request | Prompt tokens | Generation tokens | Client elapsed | Prefill | Decode | Inference | Server e2e |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 7,713 | 2 | 4.087 s | 1.651 s | 0.036 s | 1.686 s | 1.692 s |
| A2 | 7,713 | 2 | 0.241 s | 0.155 s | 0.034 s | 0.188 s | 0.199 s |
| B1 | 7,611 | 3 | 1.740 s | 1.616 s | 0.071 s | 1.686 s | 1.695 s |
| A3 | 7,713 | 2 | 0.583 s | 0.490 s | 0.034 s | 0.524 s | 0.532 s |

延遲排序為：

```text
A2 GPU hit < A3 external hit < A1/B1 cold prefill
```

A2 的 prefill 比 A1 減少 90.634%；A3 比 A1 減少 70.313%，但 A3 的
prefill 約為 A2 的 3.170 倍。Decode time 在 A1、A2 與 A3 間接近，顯示主要
差異集中在 Prefix／Prefill 階段，而不是 Decode 階段。

A1 的 client elapsed 明顯高於 server e2e，可能包含首次連線、模板初始化或
middleware 路徑的額外成本，因此跨 request 比較以 server-side prefill 與 e2e
latency 為主，client elapsed 僅作旁證。

## 5. Prometheus Metrics 精確還原每次 Cache 命中

### 5.1 原始累積 Counter

| Snapshot | Prefix queries | Local hits | External queries | External hits | Prompt tokens | Generation tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 7 | 0 | 7 | 0 | 7 | 10 |
| M1 | 7,720 | 0 | 7,720 | 0 | 7,720 | 12 |
| M2 | 15,433 | 7,712 | 15,433 | 0 | 15,433 | 14 |
| M3 | 23,044 | 7,712 | 23,044 | 0 | 23,044 | 17 |
| M4 | 30,757 | 10,096 | 30,757 | 5,328 | 30,757 | 19 |

五份 snapshots 的 `engine="0"`、`model_name="gemma3-27b-with-middleware"`
以及四個 Counter 的 `_created` 時間均一致，期間沒有觀察到 vLLM restart 或
Counter reset。

### 5.2 相鄰快照差值

| Request | Prefix queries | Local hits | Local hit rate | External queries | External hits | External contribution | Combined reuse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 7,713 | 0 | 0.000% | 7,713 | 0 | 0.000% | 0.000% |
| A2 | 7,713 | 7,712 | 99.987% | 7,713 | 0 | 0.000% | 99.987% |
| B1 | 7,611 | 0 | 0.000% | 7,611 | 0 | 0.000% | 0.000% |
| A3 | 7,713 | 2,384 | 30.909% | 7,713 | 5,328 | 69.078% | 99.987% |

這組結果給出明確的 cache 分層：

- **A1：**local 與 external 都是 0，為冷啟動。
- **A2：**7,712 tokens 從 GPU local cache 命中，external hit 為 0。
- **B1：**local 與 external 都是 0，為不同 Prefix 的冷啟動。
- **A3：**GPU local 命中 2,384 tokens，external cache 補回 5,328 tokens。

A3 的可重用 Prefix 可精確對帳：

```text
GPU local hit       2,384
External hit        5,328
                    -----
Combined reuse      7,712
Prompt tokens       7,713
Recompute               1
```

這與 16-token block size 相符：7,712 tokens 構成完整可重用 blocks，最後
1 token 位於未完成 block，需要重新處理以取得 logits。

## 6. Middleware Log 與 Prometheus 結果一致

Middleware 逐 request 記錄：

| Request | Middleware GPU hit | Middleware aiDAPTIVCache hit | Store 訊息 |
| --- | ---: | ---: | --- |
| A1 | 0 | 0 | Store 7,713 / 7,713，`skip_leading_tokens=0` |
| A2 | 7,712 | 7,712 | Store 7,713 / 7,713，`skip_leading_tokens=0` |
| B1 | 0 | 0 | Store 7,611 / 7,611，`skip_leading_tokens=0` |
| A3 | 2,384 | 7,712 | Store 1 / 7,713，`skip_leading_tokens=7712` |

實驗期間的 Prometheus Counter 增量為：

```text
總 queried tokens       30,750
Local hit tokens        10,096 → 32.833%
External hit tokens      5,328 → 17.327%
```

Middleware 最後顯示：

```text
Prefix cache hit rate: 32.8%
External prefix cache hit rate: 17.3%
```

兩組結果在顯示精度內完全一致，證明 Prometheus local/external hit counters
正確反映本次 request 序列。

## 7. 幾個容易誤讀的欄位

### 7.1 External query 不等於 SSD read

`external_prefix_cache_queries_total` 在 A1、A2、B1、A3 都增加完整 Prompt
token 數，包括 GPU 已完整命中的 A2。這代表 connector 對整個 Prompt 進行
external 查詢或統計，不代表對應 tokens 都從 SSD 讀取。

真正用於本次 external 貢獻判讀的是
`external_prefix_cache_hits_total`：A2 增量為 0，A3 增量為 5,328。

### 7.2 Middleware 的 aiDAPTIVCache hit 不等於實際 SSD 貢獻

A2 的 middleware 顯示 `aiDAPTIVCache hit:7712`，但 Prometheus external hit
增量為 0，且 A2 已由 GPU 命中 7,712 tokens。因此該 middleware 欄位較可能
表示 external cache 可匹配的完整 Prefix 或 middleware 的合併判定，不能直接
當成實際 SSD read tokens。

A3 的實際 external 貢獻應以互斥命中數計算：

```text
7,712 reusable tokens - 2,384 local tokens = 5,328 external tokens
```

此結果與 Prometheus external hit Counter 的 5,328 完全一致。

### 7.3 Store 訊息不等於實際 SSD write bytes

四個 request 都進入 middleware 的 `Storing KV cache` 路徑，包括 GPU 完整
命中的 A2。這證明 middleware 每次 request 完成後都會執行 store／同步流程，
但不能單靠該 log 判定 SSD 是否真的重新寫入相同數量的 bytes。實際 write-through、
refresh、deduplication 或 metadata update 行為仍需 SSD write counter 或更底層
log 驗證。

### 7.4 `kv_cache_usage_perc = 0` 不代表 GPU Prefix Cache 為空

五個 snapshots 的 `vllm:kv_cache_usage_perc` 都是 0。這些取樣發生在 request
完成後，`Running: 0 reqs`；該 Gauge 反映當下 active request 使用的 KV blocks，
不代表 idle Prefix Cache 是否仍保留可重用 hash 與 KV data。

A2 的 7,712-token GPU hit 已直接證明：即使 idle 時 usage 顯示 0，Prefix A
仍能從 GPU cache 重用。

### 7.5 Generation tokens 也會占用 GPU KV cache

本次 A1、A2、B1、A3 分別生成 2、2、3、2 tokens。Generated tokens 在 request
執行期間也會占用 GPU KV cache；但相較於約 7,700-token Prompt，其影響很小，
且本次生成量可落在 Prompt 尾端尚未填滿的 block 中。

下一次重新送出相同 A 時，先前生成的 `READY` 並不是新 Prompt 的一部分，
因此不會增加共同 Prefix 命中數；合理預期仍是命中 7,712，而不是 7,715。

## 8. A3 的 GPU 與 External KV 載入流程

A3 並不是把 local hit 與 external hit 當作數字相加後直接推論，而是由兩個
來源共同提供實際 KV tensors：

```text
1. vLLM 先找到 GPU 上的 2,384-token local Prefix。
2. KV Connector 找到其後 5,328 tokens 的 external KV。
3. vLLM 為 external KV 配置 GPU blocks，middleware 將其載入 GPU KV buffer。
4. Attention 使用前，所需 KV 必須已在 GPU；connector 也可能逐 layer pipeline 載入。
5. 完整 7,712-token Prefix 就緒後，只計算最後 1 個 Prompt token，再進入 decode。
```

因此，A3 在 request 執行期間會把被淘汰的 A blocks 回補至 GPU，而不是重新
計算全部 7,713 tokens。A3 完成後，這些 exact external blocks 通常也可成為
後續 local Prefix Cache；可用緊接著送出 A4 並觀察 7,712-token GPU hit 來驗證。

## 9. 限制與可信度

### 已驗證

- API response、middleware request ID 與 Prometheus Counter 增量可互相對齊。
- Prompt／Generation token Counter 增量與四次 API usage 完全一致。
- A2 local hit、A3 local/external 混合命中均有獨立證據支持。
- Prometheus 累積 hit rate 與 middleware 顯示值在精度內一致。
- 五份 snapshots 的 labels 與 Counter 建立時間一致，沒有觀察到 reset。

### 尚未驗證

- External hit 是否每次都對應實體 SSD read，而非其他 external tier 或記憶體層。
- `Storing KV cache` 是否實際產生 SSD write bytes，以及是否有 deduplication。
- A3 載回的 KV blocks 在 request 完成後是否完整留在 local GPU Prefix Cache。
- Request 執行期間的瞬時 `kv_cache_usage_perc` 峰值。

原始擷取檔未納入本 repo；本報告使用的是本次分析工作中已讀取並核對的
response payload、middleware log 與五份 Prometheus snapshots。若要讓其他人
完整重現計算，後續應把去除敏感資訊的原始擷取檔保存至受控的實驗資料位置。

整體可信度評估為 **Ready to share with stated caveats**：Prefix Cache 分層命中
結論已被多個來源驗證；只有「external hit 是否等同物理 SSD I/O」仍需額外證據。

## 10. 結論與後續測試

本次 `A1 → A2 → B1 → A3` 實驗成功驗證以下行為：

1. A1 冷算後建立可重用的 Prefix KV。
2. A2 從 GPU Prefix Cache 命中 7,712 / 7,713 tokens。
3. B1 迫使 Prefix A 的大部分 GPU blocks 被回收，只留下 2,384-token local Prefix。
4. A3 從 external cache 補回 5,328 tokens，合計重用 7,712 tokens。
5. A3 的 prefill 明顯快於冷算，但慢於純 GPU hit，符合 external KV 載入成本。

建議下一輪補充兩項驗證：

1. 在 A3 後立刻送出 A4，確認 A3 載回的 blocks 是否成為完整 local GPU hit。
2. 同步收集 SSD read/write bytes 或 aiDAPTIVLink 底層 I/O event，區分
   external connector hit 與實際 SSD 存取。
