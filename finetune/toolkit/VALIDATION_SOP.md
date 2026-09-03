# Toolkit 與 MIG 小型驗證 SOP

目的：確認 aiDAPTIV Toolkit 啟動前，H200 host、Docker container 與 Toolkit 使用的是
預期的 MIG slice。這份 SOP 只確認設備映射與保留證據；不以結果宣告 aiDAPTIVLink
2.0 已支援或已驗證 MIG fine-tuning。

## 目前驗證範圍

- 依群聯口頭資訊，支援 MIG 的 middleware 預計於第四季釋出；現行 2.0 容器尚不可視為
  已支援 MIG fine-tuning。
- 可依投影片的 `privileged` 啟動方式進行 Toolkit 與 aiDAPTIVCache 的功能 smoke test。
- 這類結果只能證明訓練流程是否可執行；不可據此宣告 MIG 隔離、MIG 支援或 middleware
  效益已驗證。

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

### 3.1 若容器仍看見多個 MIG：停止並收集 NVIDIA runtime 證據

若 rendered Compose 指定一個 `MIG_DEVICE_ID`，但 container 的 `nvidia-smi -L` 仍列出
多個 MIG slice，或 `torch.cuda.device_count()` 不為 `1`，不要執行 Toolkit。這代表目前
container 的裝置隔離尚未驗證；`project.ini` 的 `specify_gpu_index=0` 不能補救這個問題。

在 H200 host 執行以下命令。`CONTAINER_ID` 可使用完整 container ID 或可唯一識別的前綴；
這些是非互動式查詢，不需要 `-it`。

```bash
CONTAINER_ID=<container-id>

docker inspect "$CONTAINER_ID" \
  --format '{{json .HostConfig.DeviceRequests}}' \
  | tee "../runs/${RUN_ID}/container-device-requests.txt"

docker inspect "$CONTAINER_ID" \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^NVIDIA_' \
  | tee "../runs/${RUN_ID}/container-config-nvidia-env.txt"

docker exec "$CONTAINER_ID" env \
  | grep '^NVIDIA_' \
  | tee "../runs/${RUN_ID}/container-runtime-nvidia-env.txt"
```

應同時保留第 2 步的 `compose.rendered.yml`。若 `NVIDIA_VISIBLE_DEVICES` 存在，應記錄其值
是否等於本次的 `MIG_DEVICE_ID`。NVIDIA runtime 支援以 MIG UUID 作為
`NVIDIA_VISIBLE_DEVICES` 值。目前的 `finetune/container/docker-compose.yml` 應包含：

```yaml
environment:
  NVIDIA_VISIBLE_DEVICES: ${MIG_DEVICE_ID}
  NVIDIA_DRIVER_CAPABILITIES: compute,utility
```

套用此 Compose 後必須重新建立 container，並重跑第 2–3 步。若 container 仍看見多個 slice，或
`privileged: true`、vendor image 與 NVIDIA runtime 的組合使隔離行為無法符合預期，保留上述
證據並向群聯確認 aiDAPTIVLink 2.0 的 MIG fine-tuning 支援範圍；不要將未隔離環境的結果解讀為
middleware 效益。

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

## 5. 執行期間：收集 aiDAPTIVCache SSD I/O 證據

在 H200 host 的另一個 terminal 執行。此環境中 `/mnt/nvme0` 的 `ai-ai` LVM
由 `nvme1n1` 與 `nvme2n1` 組成；只監控這兩顆 aiDAPTIVCache SSD，不納入作為
系統與一般資料碟的 `nvme0n1`。

```bash
findmnt -no SOURCE /mnt/nvme0
lsblk -o NAME,TYPE,SIZE,MOUNTPOINTS
```

接著在執行 Toolkit 前啟動監控；保持它持續執行至 Toolkit 結束，再停止背景程序。

```bash
# 模型、t1 | t2 | t3
RUN_ID="toolkit-gemma-3-12b-it-t1-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="finetune/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

iostat -dxm 1 nvme1n1 nvme2n1 > "$RUN_DIR/host-ssd-iostat.txt" &
IOSTAT_PID=$!

# 在另一個 terminal 於容器中執行 Toolkit --t 1 / --t 2 / --t 3
# python3 aidaptest_run.py --t 1

kill "$IOSTAT_PID"
```

判讀 `host-ssd-iostat.txt` 時，將目標 SSD 在 Toolkit 前的閒置值與執行期間的
`r/s`、`w/s`、`rMB/s`、`wMB/s`、`await`、`%util` 對照。若 I/O 在 Toolkit 執行時間
內明顯高於閒置值，這是 aiDAPTIVCache SSD 被使用的佐證；但同一顆 SSD 的其他工作負載也
可能造成 I/O，因此這項證據本身不能單獨證明是 middleware offload 或量化其效益。

## 6. 執行期間與結束後：保存證據

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
