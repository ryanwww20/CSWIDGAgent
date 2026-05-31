/colab-demo-generator

請把 @kvcache_interactive_skill.ipynb 改成一份更嚴謹、適合課堂使用的互動式 KV-cache demo notebook。

你必須建立一個新的 Colab notebook，而不是只描述修改內容。

新的 notebook 檔名請命名為：

kvcahce_interactive_skill_refined.ipynb

注意：請照這個檔名拼法輸出。

## 目標

根據以下評估結果改善原本的 notebook：

Overall: Acceptable, usable after minor fixes

主要問題：

1. GQA group-count logic 在 non-divisible head/group settings 時可能錯誤
2. Pruning simulation 對 recent tokens 的處理不一致，而且 edge cases 比較脆弱
3. 部分視覺化與單位選擇容易讓學生混淆
4. 教學引導可以更強

## 必做修改

### 1. Correctness: GQA / variants section

在 draw_variants 的邏輯中：

- 將 floor grouping 改成 ceiling grouping：
  - 使用 ceil(n_heads / group_size)
- 確保以下項目在 non-divisible cases 中都一致：
  - head 到 group 的 mapping
  - group boxes
  - memory formulas
- 請特別確認這個例子可以正確運作：
  - n_heads = 10
  - group_size = 8
- 在 markdown 中加入一段簡短說明：
  - 當 head 數量無法整除 group_size 時，剩下的 remainder heads 會如何被分配

### 2. Correctness and robustness: pruning section

在 simulate_pruning 中：

- recent tokens 的處理要前後一致
  - 如果 recent tokens 被排除在 ranking 之外，那 plotting 和 labels 中使用的 trailing token range 也必須是同一段
- 避免 divide-by-zero 或 NaN
- 要能穩定處理極端 slider 組合，例如：
  - 很小的 seq_len
  - recent_tokens 接近 seq_len
  - keep_ratio 接近邊界值
- 如有需要，可以動態調整 widget bounds，或在 function logic 中加入 safe clamps
- 程式應該 gracefully handle edge cases，不要直接壞掉

### 3. Visualization clarity: memory section

在 KV memory visualization 中：

- 避免同一張 bar chart 混用不同單位
- 每張 chart 請使用一致單位，或清楚分成不同 chart
- 明確說明使用的是 binary conversion 還是 decimal conversion
- chart labels 要 beginner-friendly，讓學生一看就懂

### 4. Pedagogy improvements

在不讓 notebook 過度膨脹的前提下：

- 在重要互動章節加入 1 到 2 個簡短的「predict first, then test」checkpoint prompts
- 加入一個 misconception callout，例如：
  - KV cache reduces repeated K/V compute, but does not remove all decode cost
- 至少把一個 reflection prompt 改成具體 mini exercise
  - 要包含學生應該觀察到的 expected outcome

## Constraints

- dependencies 要輕量，適合 Colab
- 不要使用 local file paths
- 保留原本互動式 demo 的精神與章節順序
- 程式碼要可讀
- 優先使用小 helper functions，不要 over-engineer
- markdown 要清楚，適合 beginner-to-intermediate students

## Validation requirements

完成修改後，請執行以下檢查：

1. Run a full top-to-bottom execution sanity check
2. Run targeted edge-case checks：
   - variants:
     - n_heads = 10
     - group_size = 8
   - pruning:
     - low keep_ratio
     - high keep_ratio
     - recent_tokens near seq_len

## Final report

請在最後回報：

1. What changed by section
2. Any assumptions or remaining limitations
3. Whether execution succeeded

## Output

請輸出一個完整的新 notebook：

kvcahce_interactive_skill_refined.ipynb