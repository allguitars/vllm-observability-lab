# aiDAPTIVLink 2.0 訓練踩坑清單

- Host 路徑無法直接供訓練程式使用；所有 YAML 路徑都改成容器內可見的路徑，並以 bind mount 對應到 Host。
- 修改 Compose 後舊容器仍看不到新增掛載；重建容器後才讓設定、腳本與輸出目錄正確掛入。
- `output_dir` 不存在會讓 `phisonai2` 直接失敗；訓練前先建立該目錄，並放在 Host bind mount 中避免刪除容器時遺失模型。
- `RUN_DIR` 與 `env_config.yaml` 的 `output_dir`、`log_name` 不會自動連動；每次執行前將三者設成同一個新 run，避免覆蓋或混入舊紀錄。
- 容器缺少 `/usr/bin/time` 導致 wrapper 立即失敗；改用 `date +%s` 記錄開始、結束與總耗時。
- 腳本把輸出導向檔案後看似完全沒動靜；改為查看 `train-wrapper.log`、時間檔與訓練 log 判斷進度。
- 群聯子程序失敗時仍曾回傳 exit code 0；成功判定改為同時確認完成訊息、checkpoint 與退出碼。
- Exp config 的五組設定放在 YAML 頂層造成 validation errors；將 `model_saver`、`lr_scheduler`、`optimizer`、`early_stop`、`lora` 全部移到 `run_settings` 下。
- Dataset config 的註解含有 `data_path`，被群聯 parser 誤算成第二個資料集；移除註解中的該字串，單一資料集不另外加入 `---`。
- JSON 欄位與 `question_key`、`answer_key`、`label_key` 不一致會破壞資料對應；統一實際資料與設定檔的欄位名稱。
- 訓練容器可看到多個 CUDA device，無法保證用到指定 MIG；在執行環境設定 `CUDA_VISIBLE_DEVICES=0`，並用 PyTorch 重新確認只剩目標裝置。
- 同一個 MIG 已被推論模型占用會影響訓練資源；訓練前先停止該服務並確認 VRAM 已釋放。
- H200 上的實際 `env_config.yaml` 容易在同步時發生衝突；將它與 scripts `.env` 加入 Git ignore，只追蹤可複製的 example。
- 訓練 checkpoint 缺少 `preprocessor_config.json`，導致 Gemma 3 無法由 vLLM 啟動；從完全相同的基礎模型補回該前處理 metadata 後成功載入。
- Dify 測試沒有使用訓練時的 system prompt，導致輸入格式不一致；改用相同 system prompt、每題開新對話並固定推論條件後再驗證。
