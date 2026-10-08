# twbt — 台股回測框架

自己寫的台股回測框架。目前完成**資料層**：從 FinMind 下載、整理成乾淨資料，並提供不會偷看未來的查詢介面。回測引擎和策略是下一步。

## 快速開始

```bash
pip install -e ".[dev]"
export FINMIND_TOKEN=...            # finmindtrade.com 註冊後取得，免費版每小時 600 次

twbt update --stocks 2330,2317      # 先用兩檔試跑，約 13 次呼叫
twbt build
twbt status
```

完整下載：

```bash
twbt update      # 第一次約 9,000 次呼叫，免費版約 16 小時；可隨時 Ctrl-C，再執行一次就會接著下載
twbt build
```

之後每月更新一次，只抓新資料。

**不用自己的電腦：GitHub Actions**

`.github/workflows/data.yml` 會在 GitHub 的機器上每 6 小時跑一次下載，每次跑約 5 小時後自動停下，進度存在 Actions cache，下次接著跑。第一次完整下載約需 3–4 次執行，之後大約每 25 天更新一次。整理好的 clean 資料會上傳成 `clean-data` artifact。

設定：repo 設為 private（FinMind 授權不允許再散布資料），在 Settings → Secrets and variables → Actions 新增 `FINMIND_TOKEN`。

```python
from twbt.data.query import Data

data = Data("data")
data.universe_on("2024-04-11")          # 當天可交易的股票（含之後才下市的）
data.revenue_asof("2024-04-11")          # 當天已經公布的月營收
data.prices("2024-01-01", "2024-12-31", stocks=["2330"])
```

## 設計

```
FinMind API ──► raw/（原樣保存，只追加）──► clean/（可隨時重建）──► Data（point-in-time 查詢）
```

**下載（`update`）**
- 用戶端自行限流（滑動視窗，預設每小時 550 次，比上限留一點餘裕）。伺服器仍回 402 時等一個完整視窗再重試
- 網路錯誤和 5xx 用指數退避重試；連續失敗 5 次就停下，避免 token 錯誤時白白浪費額度
- 每抓完一檔就寫入進度（`raw/_manifest.json`），所以中斷後可以續傳
- 所有寫入都是先寫暫存檔再原子替換，中斷不會留下壞掉的檔案
- 增量更新：價格從上次最後一天的隔天開始抓；月營收往回重抓 120 天，以收到公司的營收修正
- 已下市股票抓過一次就不再抓

**股票池**
- 上市、上櫃普通股（四位數、非 0 開頭），排除 ETF、特別股、權證、存託憑證、興櫃
- 包含回測起始日後下市的公司，避免 survivorship bias
- 興櫃轉上市櫃的股票在 FinMind 有兩列，取最新那列

**避免 look-ahead**
- 月營收的 `available_date` = 申報期限後的下一個交易日。期限是次月 10 日，遇假日順延（例如 10/10 國慶、週末）。公司提早公布也不提前使用，讓整段歷史用同一條規則
- FinMind 自 2026-04-21 起有 `create_time`。晚於期限才出現的資料，改用 `create_time` 後的下一個交易日
- 策略只能透過 `revenue_asof(date)` 取得營收，結構上碰不到未公布的資料

**價格與公司行動**
- `prices` 是**未還原**的原始價格，回測成交一律用原始價
- FinMind 對無公告成交價的日子填 0，這裡轉成 NaN 並標記 `has_quote=False`
- 公司行動（除權息、減資、分割、變更面額）統一成 `factor = 除權息前收盤 / 參考價`，引擎持有跨過事件時股數乘上 factor
- `adjusted_close()` 由上述事件自己算還原價，只給訊號用

## 已知限制

- **營收修正沒有版本紀錄**：公司修正營收後只保留最新值，無法重現當時看到的錯誤數字
- **現金股利視為除息日再投入**：沒有模擬除息日到發放日之間的現金空窗
- **已下市股票的市場別未知**：FinMind 的股票清單只有現存股票
- **代號重複使用**：已下市代號被新公司使用時，視為現存公司，舊公司的資料會混在同一個代號下
- 價格、月營收、除權息三個資料集的欄位已用 2330 的實際 API 回應確認（2026-10-08）；其餘依 FinMind 文件

## 測試

```bash
pytest -q
ruff check .
```

測試用假的 FinMind 伺服器，不需要網路。涵蓋限流、伺服器限流重試、續傳、增量起始日、原子寫入、股票池篩選、營收可用日、零價處理、公司行動與還原價、point-in-time 查詢。

資料來源：[FinMind](https://finmind.github.io/)
