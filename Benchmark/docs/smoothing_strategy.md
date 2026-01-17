# Smoothing Strategy: Gaussian Filter & Unified Pipeline

## 1. 我們現在的平滑是怎麼做的？

### 技術實作 (`scipy.ndimage.gaussian_filter1d`)
目前程式碼中使用的是 **一維高斯濾波 (1D Gaussian Filter)**。
```python
kps_smooth = gaussian_filter1d(kps, sigma=1.0, axis=0) # axis=0 代表對時間軸平滑
```

### 數學意義
這是一個 **加權移動平均 (Weighted Moving Average)**。
對於每一個時間點 $t$ 的座標 $x_t$，我們不只看它自己，還看它前後鄰居 ($t-1, t+1, \dots$)，並根據距離給予權重。
權重分佈符合 **鐘形曲線 (Bell Curve/Normal Distribution)**。

*   **Sigma ($\sigma$)**: 控制「看多寬」。
    *   $\sigma=1.0$: 主要參考前後約 $\pm 3$ 幀 (即 $t-3$ 到 $t+3$) 的資訊。
    *   $\sigma$ 越大，曲線越平滑，但反應越遲鈍 (Lag)。

### 實際意義 (Why do we need it?)
*   **消除抖動 (De-noising)**: 模型輸出的關鍵點就像手抖的人拿筆畫線，會有不自然的鋸齒 (Jitter)。高斯濾波就像用熨斗把它燙平，還原出真實的運動軌跡。
*   **微分的前提 (Conditioning for Differentiation)**: 
    *   物理世界是連續且慣性的，老鼠不可能在 0.03 秒內把頭瞬移 5 pixel 再移回來。
    *   如果沒有平滑直接微分，那些「瞬移 (Outliers/Noise)」會被算成巨大的速度與加速度，完全掩蓋真實動作。

---

## 2. 修改建議：統一先平滑，還是分開做？

### 選項 A：維持現狀，只修補漏洞 (Func-level Smoothing)
*   **作法**: 在 `compute_orientation` 裡面補上一行 `gaussian_filter1d`。
*   **優點**: 風險最低，不影響已經正確的 Velocity/Acceleration。
*   **缺點**: 容易發生「這裡有做、那裡沒做」的遺漏 (就像這次)。

### 選項 B：統一在最源頭平滑 (Global Smoothing) ✅ **(推薦)**
*   **作法**: 在 `run_benchmark.py` 一讀進 Keypoints 後，立刻對 `kps` 做一次統一的 `gaussian_filter1d`，產生 `kps_smooth_global`。然後所有物理計算 (Kinematics, Orientation, Morphology) 全都使用這個乾淨的數據。
*   **優點**:
    1.  **一致性 (Consistency)**: 保證所有指標 (速度、角度、外型) 都是基於同一個物理實體計算的。不會發生「速度很平滑，但角度很抖」的矛盾。
    2.  **效率**: 不用在每個函式裡重複算好幾次 Filter。
    3.  **防呆**: 未來增加新指標時，不用擔心忘記加平滑。

### 潛在風險與解決
*   **Compactness 需要平滑嗎？**: 
    *   雖然它不需要平滑也能看，但給它平滑後的數據**更正確**。因為 `Compactness` 是身體的物理屬性，物理屬性本來就不該包含高頻抖動。使用平滑後的數據只會讓結果更純淨。
*   **PCA 需要平滑嗎？**:
    *   **注意**: 之前的 PCA 分析是為了看「捕捉到的所有資訊量」。如果我們過度平滑，可能會把一些真實的微小動作 (Micro-movements) 濾掉。
    *   **決策**: 
        *   **Trace Plots (物理指標)**: 絕對應該用平滑數據。
        *   **PCA (維度分析)**: 建議保留**原始數據**或使用**輕微平滑**。但為了 Benchmark 的一致性，通常物理指標用平滑，Embedding 分析用原始。

## 3. 執行計畫 (Refined Plan)

1.  **修改 `src/physics.py`**:
    *   移除 `compute_keypoint_change_score` 和 `compute_kinematics` 內部的 `gaussian_filter1d`。
    *   改為假設傳進來的 `kps` 已經是乾淨的。
    *   或者：保留 `sigma` 參數，但預設為 `None` (不重複做)，若外部沒做才做。

2.  **修改 `run_benchmark.py`**:
    *   在 **Phase 2** 開頭：
        ```python
        # Global Smoothing
        print(f"Applying Global Gaussian Smoothing (sigma={sigma})...")
        kps_smooth = gaussian_filter1d(kps, sigma=sigma, axis=0)
        ```
    *   將所有 `physics.*` 的呼叫參數改為傳入 `kps_smooth`。

這樣就能一次解決所有鋸齒問題，並且讓物理意義更統一。
