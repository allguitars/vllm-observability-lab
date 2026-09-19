# aiDAPTIVCache SSD I/O 驗證報告

## 本次測試

- Run ID：`toolkit-gemma-3-12b-it-t1-20260903-152618`
- 工作：以 `gemma-3-12b-it` 執行 aiDAPTIV Toolkit `--t 1`
- 目的：確認訓練期間是否實際使用掛載於 `/mnt/nvme0` 的 aiDAPTIVCache SSD。

## 儲存裝置範圍

`/mnt/nvme0` 掛載的 `ai-ai` LVM 由下列兩顆實體 SSD 組成：

- `nvme1n1`
- `nvme2n1`

系統與一般資料碟 `nvme0n1` 不納入本次觀察。測試期間主機沒有其他使用者或工作負載。

## 驗證程序

1. 在 H200 host 建立本次 run 目錄。
2. 在執行 Toolkit 前，於另一個 terminal 啟動下列監控：

   ```bash
   iostat -dxm 1 nvme1n1 nvme2n1 > "$RUN_DIR/host-ssd-iostat.txt" &
   IOSTAT_PID=$!
   ```

3. 執行 Toolkit `--t 1`。
4. Toolkit 結束後停止監控：

   ```bash
   kill "$IOSTAT_PID"
   ```

5. 以 [host-ssd-iostat.txt](host-ssd-iostat.txt) 檢視兩顆 Cache SSD 的讀寫速率、等待時間與使用率。

## 結果解讀

`iostat` 第一組數值是自開機以來的累積平均，未納入下列統計。其後共收集約 447 個一秒區間：

| 指標 | nvme1n1 | nvme2n1 |
| --- | ---: | ---: |
| I/O 大於等於 1 MB/s 的區間數 | 268 | 268 |
| 累積讀取量（監控區間加總） | 約 654 GB | 約 654 GB |
| 累積寫入量（監控區間加總） | 約 476 GB | 約 476 GB |
| 讀取峰值 | 約 6.68 GB/s | 約 6.68 GB/s |
| 寫入峰值 | 約 3.58 GB/s | 約 3.58 GB/s |

大量 I/O 約在開始監控後第 135 秒出現，持續至約第 420 秒。兩顆 SSD 的讀寫量與波形幾乎一致，符合它們共同承載 `ai-ai` LVM 的預期。

## 結論與限制

在「主機無其他工作負載」且兩顆 SSD 專供 `/mnt/nvme0` aiDAPTIVCache 的前提下，本次結果高度支持 Toolkit `--t 1` 訓練流程實際使用了 aiDAPTIVCache 的外部 SSD。

本次驗證不比較無 middleware 的基線，因此不能據此量化 middleware 的效能、容量或成本效益。後續應以相同模型、資料集、batch size、序列長度與 GPU 資源，分別執行有／無 middleware 的測試，再比較訓練吞吐量、最大 batch、DRAM/VRAM 與 SSD I/O。
