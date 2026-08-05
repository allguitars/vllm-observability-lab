# vLLM Metrics Collection and Prefix Cache Experiments

這個專案用於收集 vLLM metrics，並以固定的長 prompt 驗證 Automatic
Prefix Caching 與外部 KV cache（例如 aiDAPTIVCache SSD offload）的行為。

主要目標是驗證下列序列：

```text
A1 → A2 → B1 → C1 → A3
```

- `A1`：冷啟動，計算 Prefix A 並寫入 cache。
- `A2`：重送完全相同的 A，確認 local/GPU prefix cache hit。
- `B1`、`C1`：送入不同長 prefix，對有限 GPU KV cache 製造壓力。
- `A3`：重新送 A，確認 local hit、external KV hit 或重算的結果。

## 專案結構

```text
scripts/
  collect_vllm_metrics.py   # 定期收集 vLLM /metrics 並輸出 CSV
  run_test_flow.py          # 執行可設定的測試流程並輸出 JSONL
  send_prefix_prompt.py     # 將一個 prefix 檔送至 OpenAI-compatible API
flows/
  *.flow.json               # Prompt 與 metrics snapshot 的執行順序
prompts/
  generate_long_prompts.py  # 以 Gemma 3 27B tokenizer 產生固定長 prompt
  prefix_a_*.txt            # 分散式系統 prompt
  prefix_b_*.txt            # 金融風險 prompt
  prefix_c_*.txt            # 生物醫學 prompt
tokenizers/
  gemma-3-27b-it/
    tokenizer.json          # 可重現 Gemma 3 27B IT token 計數的最小 runtime 檔案
docs/
  vllm-metrics-analysis-report-2026-07-25.md
```

## 安裝

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

`.env` 含有可能的 API key，已被 Git 忽略，請勿提交。

`tokenizers/gemma-3-27b-it/tokenizer.json` 會由 Git 追蹤，因此 clone 後安裝
`requirements.txt` 即可重現 Prompt token 計數，不需要另外下載模型權重。

## API 呼叫設定

`scripts/send_prefix_prompt.py` 讀取 repo 根目錄的 `.env`。命令列的
`--base-url` 與 `--model` 可以覆寫 `.env`；shell 環境變數優先於 `.env`。

| 變數 | 用途 | OpenAI 測試值 | H200 vLLM 值 |
|---|---|---|---|
| `LLM_BASE_URL` | API 根網址，必須包含 `/v1` | `https://api.openai.com/v1` | `http://<H200_HOST>:8000/v1` |
| `VLLM_METRICS_URL` | Prometheus metrics endpoint | 不適用 | `http://<VLLM_HOST>:8000/metrics` |
| `MODEL_NAME` | 對外 model ID | `gpt-5.6-luna` | `/v1/models` 回傳的 `data[].id` |
| `TOKEN_LIMIT_PARAMETER_NAME` | request 的輸出 token 欄位名稱 | `max_completion_tokens` | `max_tokens` |
| `REASONING_EFFORT` | 有值時才傳送 `reasoning_effort` | `none` | 留空 |
| `OPENAI_API_KEY` | Bearer 認證 key | OpenAI API key | vLLM `--api-key` 的值；未啟用驗證時留空 |

OpenAI 設定範例：

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-5.6-luna
TOKEN_LIMIT_PARAMETER_NAME=max_completion_tokens
REASONING_EFFORT=none
OPENAI_API_KEY=sk-...
```

H200 vLLM 設定範例：

```dotenv
LLM_BASE_URL=http://<H200_HOST>:8000/v1
VLLM_METRICS_URL=http://<VLLM_HOST>:8000/metrics
MODEL_NAME=<served-model-name>
TOKEN_LIMIT_PARAMETER_NAME=max_tokens
REASONING_EFFORT=
OPENAI_API_KEY=
```

若 vLLM 以 `--served-model-name gemma3-27b` 啟動，`MODEL_NAME` 應填
`gemma3-27b`。最終仍以 `/v1/models` 回傳的 `data[].id` 為準。

## 送出長 Prefix

三份測試資料採用 Gemma 3 27B tokenizer 加上單一 user turn 的 chat
template 計算：

| 檔案 | 主題 | Chat tokens |
|---|---|---:|
| `prompts/prefix_a_distributed_systems.txt` | 分散式系統 | 7,713 |
| `prompts/prefix_b_financial_risk.txt` | 金融風險 | 7,611 |
| `prompts/prefix_c_biomedical_research.txt` | 生物醫學研究 | 7,704 |

腳本固定將輸出 token 上限設為 16。這足以讓模型完成 `READY` 回應，也能把
generation 對 KV cache 的影響降到很低。

```bash
python scripts/send_prefix_prompt.py \
  prompts/prefix_a_distributed_systems.txt
```

依序執行完整實驗：

```bash
python scripts/send_prefix_prompt.py prompts/prefix_a_distributed_systems.txt
python scripts/send_prefix_prompt.py prompts/prefix_a_distributed_systems.txt
python scripts/send_prefix_prompt.py prompts/prefix_b_financial_risk.txt
python scripts/send_prefix_prompt.py prompts/prefix_c_biomedical_research.txt
python scripts/send_prefix_prompt.py prompts/prefix_a_distributed_systems.txt
```

輸出中的欄位：

- `prompt_tokens`：實際服務端 tokenizer 計算的 prompt tokens。
- `completion_tokens`：模型實際生成的 token 數。
- `elapsed_seconds`：完整 HTTP request 時間，包含網路、排程與推論。
- `cached_tokens`、`cache_write_tokens`：OpenAI API 若提供，顯示其託管
  prompt cache accounting；vLLM 不一定回傳這些欄位。

## 收集 vLLM Metrics

### 執行可設定的測試流程

`flows/*.flow.json` 可交替執行 `capture_metrics` 與 `send_prompt`。例如：

```bash
python scripts/run_test_flow.py flows/prefix-cache-aaba.flow.json
```

每次執行會建立 `runs/run_<run-id>.jsonl`。其中包含完整的 Prometheus text、
`send_prefix_prompt.py` JSON response，以及每個步驟的開始、完成或失敗事件。
第一版不解析、不篩選或聚合 metrics；流程遇到失敗會停止，但已寫入的事件仍會保留。

可用命令列暫時覆蓋 metrics endpoint：

```bash
python scripts/run_test_flow.py \
  flows/prefix-cache-aaba.flow.json \
  --metrics-url http://<VLLM_HOST>:8000/metrics
```

### 持續收集 CSV

```bash
python scripts/collect_vllm_metrics.py
```

目前 `scripts/collect_vllm_metrics.py` 的 `METRICS_URL` 是程式內常數。
執行 H200 實驗前，請先將它改為實際 vLLM server 的 `/metrics` URL。
腳本每秒取樣一次，並在目前目錄產生 `vllm_metrics_YYYYMMDD_HHMMSS.csv`。

對 H200 實驗，除了 `kv_cache_usage_perc` 外，應優先記錄：

- `vllm:prefix_cache_queries` 與 `vllm:prefix_cache_hits`
- `vllm:external_prefix_cache_queries` 與
  `vllm:external_prefix_cache_hits`（若 middleware/connector 有暴露）
- middleware log、SSD read/write bytes 與 external KV load event

## 判讀原則與限制

- `A2` 的 local prefix hit 應以 prefix-cache counters 為主要證據，不能只看
  `kv_cache_usage_perc`。完成的 request 可能釋放 active blocks，但仍保留可重用
  的 cached block hash。
- A3 是否成為 external/SSD hit，取決於 GPU blocks 是否被回收、middleware 是否
  已將 A 寫入 external cache，以及 token、model、chat template 與 cache namespace
  是否完全相同。
- OpenAI 的 `cached_tokens` 可用來驗證 prompt cache accounting 與腳本流程，但其
  託管架構無法模擬 H200 的 GPU block eviction、aiDAPTIVCache SSD I/O 或 external
  prefix cache 行為。
- H200 的最終結論應以 vLLM `/metrics`、middleware logs、external prefix cache
  counters 與 SSD I/O 為準。

## 驗證

```bash
.venv/bin/python -m py_compile \
  scripts/collect_vllm_metrics.py \
  scripts/send_prefix_prompt.py \
  prompts/generate_long_prompts.py
```
