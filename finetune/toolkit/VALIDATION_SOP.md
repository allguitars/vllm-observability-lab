# Toolkit 與 MIG 小型驗證 SOP

目的：確認 aiDAPTIV Toolkit 啟動前，H200 host、Docker container 與 Toolkit 使用的是
預期的 MIG slice。這份 SOP 只確認設備映射與保留證據；不以結果宣告 aiDAPTIVLink
2.0 已支援或已驗證 MIG fine-tuning。

## 前置條件

- 已在 H200 host 取得本次要使用的 MIG UUID：`nvidia-smi -L` 的 `MIG-...` 值。
- `finetune/container/compose.env` 的 `MIG_DEVICE_ID` 已填入該 UUID。
- Toolkit 可在容器的 Python 環境載入 `torch`，且容器能讀取模型與 aiDAPTIVCache。
- 本次 run 有一個可寫的輸出位置，例如 `/workspace/finetune/runs/<run-id>/`。

> 目前 `finetune/container/docker-compose.yml` 將 Toolkit 掛成唯讀。不要在已提交的
> `toolkit_v2.3.1_python312/project.ini` 直接填寫真實路徑或密碼。正式執行前，應先
> 準備未追蹤的可寫 Toolkit 工作副本，並讓容器使用該副本。

## 1. 建立本次證據目錄

在 H200 host 的 repo 根目錄執行。`RUN_ID` 可依日期、MIG 或實驗目的命名。

```bash
RUN_ID="toolkit-mig-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="finetune/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"
```

## 2. 啟動容器前：記錄 host 的 MIG 狀態

```bash
nvidia-smi -L | tee "$RUN_DIR/host-nvidia-smi-L.txt"
nvidia-smi | tee "$RUN_DIR/host-nvidia-smi.txt"
```

確認檔案中包含預定的 `MIG-...` UUID。也保存本次使用的 Compose 設定，但不要將含有
密碼或其他敏感值的 `compose.env` 提交到 Git。

```bash
cd finetune/container
docker compose --env-file compose.env config > "../runs/${RUN_ID}/compose.rendered.yml"
docker compose --env-file compose.env up -d
```

## 3. Toolkit 前：確認容器實際可見的設備

以下命令在容器內執行；將輸出寫回 host 的 run 目錄。

```bash
docker compose --env-file compose.env exec phison-finetune bash -lc '
  nvidia-smi -L
  nvidia-smi
  python3 - <<"PY"
import torch
print("count =", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(index)
    print(index, prop.name, prop.total_memory)
PY
' | tee "../runs/${RUN_ID}/container-gpu-before.txt"
```

判讀：若本次只分配一個 MIG slice，容器通常只應顯示一個 CUDA device，Toolkit 的
`specify_gpu_index=0` 指的就是這個容器內 device。`0` 不是 host 上的 MIG UUID；需以
第 2 步與此步的 `nvidia-smi -L` 輸出交叉對照。

## 4. 設定並執行 Toolkit

在 **可寫的 Toolkit 工作副本** 中，先設定 `project.ini`：

```ini
[ENV_setting]
specify_gpu_index=0
num_gpus=1
model_name_or_path=<container 內模型路徑>
nvme_path=<container 內 aiDAPTIVCache 掛載點>

[Performance_test]
start_bs=1
end_bs=<保守上限>
seq_len=<本次測試序列長度>
training_hour=<短時間 smoke test>
triton=True
```

先確認 Toolkit 可載入，再做小範圍效能測試：

```bash
cd <可寫 Toolkit 工作副本>/Script/Model_Test
python3 aidaptest_run.py --version
python3 aidaptest_run.py --t 1
```

確認無誤後才使用 `--t 2` 找最大 batch，或 `--t 3` 先找最大 batch 再做效能測試。

## 5. 執行期間與結束後：保存證據

Toolkit 執行期間或剛結束時，在容器內再次執行：

```bash
nvidia-smi
```

保存到同一個 run 目錄，並一併保留 Toolkit `Log/` 下的 training log、GPU/DRAM 圖表
與 `performance_result.xlsx`。確認其 GPU 記憶體上限與第 3 步看到的 MIG slice 容量一致。

## 完成條件

- host 與 container 的 `nvidia-smi -L` 證據可對應到本次指定的 MIG UUID。
- 容器內 `torch.cuda.device_count()` 與 `project.ini` 的 `num_gpus` 一致。
- Toolkit log、圖表與 Excel 都保存到本次 `RUN_DIR`。

若任一項不符，停止在環境確認階段；不要將該次結果解讀成 middleware 或 fine-tuning
效益。
