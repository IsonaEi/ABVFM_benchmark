# ABVFM Benchmark Metrics: Mathematical Derivation & Signal Characteristics

這份文件詳細列出了所有 `Event-Triggered Trace`圖表中的 Metrics 是如何從原始資料計算出來的，並解釋「鋸齒 (Jaggedness)」與「平滑 (Smoothness)」的數學成因。

---

## 原始輸入資料 (Source Data)
*   **Keypoints ($X_{raw}$)**: 形狀 $(T, 27, 2)$。即 $T$ 幀，$27$ 個身體部位的 $(x, y)$ 座標。
    *   **特性**: 含有高頻雜訊 (Jitter)，來自 Pose Estimation 模型的抖動 (1-2 pixels)。
*   **Optical Flow ($F_{raw}$)**: 形狀 $(T, H, W)$。每幀每個像素的移動量。

---

## 各指標計算流程 (Processing Pipeline)

### 1. Velocity (平滑)
*   **定義**: 身體部位的平均移動速度。
*   **計算過程**:
    1.  **Smoothing**: 對 $X_{raw}$ 進行 Gaussian Smoothing ($\sigma=1.0$) $\rightarrow X_{smooth}$。
        *   $$X_{smooth}(t) = X_{raw} * G_\sigma(t)$$
        *   **作用**: 這裡的平滑**抑制了高頻抖動**。
    2.  **Difference**: 計算相鄰幀的位移向量。
        *   $$V(t) = X_{smooth}(t) - X_{smooth}(t-1)$$
    3.  **Norm & Mean**: 取向量長度，再對 27 個點取平均。
        *   $$Speed(t) = \frac{1}{27} \sum_{k=1}^{27} \|V_k(t)\|_2$$
*   **為何平滑？**: 因為第一步就做了 Gaussian Smoothing。

### 2. Acceleration (平滑)
*   **定義**: 速度的變化率。
*   **計算過程**:
    1.  **Diff of Velocity**: 對上述的 $V(t)$ (向量速度) 再做一次微分。
        *   $$A(t) = V(t) - V(t-1)$$
    2.  **Norm & Mean**:
        *   $$Acc(t) = \frac{1}{27} \sum \|A_k(t)\|_2$$
*   **為何平滑？**: 基礎資料 $X_{smooth}$ 已經乾淨，且微分雖會放大雜訊，但因為源頭已處理過，尚在可控範圍。

### 3. Compactness (非常平滑)
*   **定義**: 身體像球一樣緊縮的程度 (縮成一團 vs 伸展開)。
*   **計算過程**:
    1.  **Centroid**: 計算每幀的重心 $C(t)$。
    2.  **Distance Sum**: 加總所有點到重心的距離。
        *   $$Compactness(t) = \sum_{k=1}^{27} \|X_{raw, k}(t) - C(t)\|_2$$
*   **為何非常平滑？**: 
    1.  它是 **0 階指標 (0-th Order)**：完全不涉及微分 ($t - (t-1)$)。微分是雜訊放大器，不微分就不會放大。
    2.  它是 **加總指標 (Aggregation)**：27 個點的隨機抖動 (有的偏左、有的偏右) 在加總時會互相**抵銷 (Averaging Effect)**。

### 4. AbsAngAcc (角加速度 - 鋸齒狀 ⚡️)
*   **定義**: 脊椎或頭部轉動的加速度量值 (旋轉多快地在改變)。
*   **計算過程 (問題所在)**:
    1.  **Vector**: 取出兩個關鍵點 (e.g., Neck, Head) 構成向量 $V_{head}(t)$。
    2.  **Angle**: 計算角度 $$\theta(t) = \arctan2(y, x)$$。
        *   **特性**: 這裡用的是 $X_{raw}$ (無平滑)。
    3.  **Unwrap**: 處理 $360^\circ$ 跳變。
    4.  **1st Diff (AngVel)**: $$\omega(t) = \theta(t) - \theta(t-1)$$
        *   **雜訊放大**: 原始座標跳動 1 pixel，角度可能跳動 $5^\circ$。
    5.  **2nd Diff (AngAcc)**: $$\alpha(t) = \omega(t) - \omega(t-1)$$
        *   **雜訊再放大**: 二次微分會極劇烈地放大高頻雜訊。
    6.  **Absolute**: 取絕對值 $$|\alpha(t)|$$。
*   **為何鋸齒？**: 
    1.  **漏了 Smoothing**: 原始座標 $X_{raw}$ 沒有先過 Gaussian Filter。
    2.  **二次微分**: $Noise_{acc} \propto f^2 \times Noise_{pos}$。
    3.  **取絕對值**: 把原本在 0 附近震盪的高頻雜訊全部翻成正值，變成看起來像「草叢」一樣的波形。

### 5. Optical Flow Magnitude (中等/平滑)
*   **定義**: 畫面整體的像素流動量。
*   **計算過程**:
    1.  **Load**: 讀取預計算流程。
    2.  **Masking**: 只看老鼠區域。
    3.  **Smoothing**: 腳本中有 `gaussian_filter1d(flow, sigma=1.0)`。
*   **結果**: 曲線通常是平滑的。

---

## 總結：鋸齒成因表

| Metrics | 數學階數 (Order) | 有無 Smoothing | 特性 |
| :--- | :---: | :---: | :--- |
| **Compactness** | 0 (位置) | 自動 (平均效應) | 🟢 非常平滑 |
| **Velocity** | 1 (速度) | ✅ 有 | 🟢 平滑 |
| **Acceleration** | 2 (加速度) | ✅ 有 | 🟢 平滑 |
| **AbsAngVel** | 1 (角速度) | ❌ **無** | ⚠️ 小鋸齒 |
| **AbsAngAcc** | 2 (角加速度) | ❌ **無** | ⚡️ **嚴重鋸齒** |

### 數學直觀
想像原始資料是一條有點毛邊的線：
- **Compactness** 就像是「量這條線離中心的平均距離」，毛邊被平均掉了。
- **Velocity** 是「看這條線的斜率」，但在看之前我們先用熨斗 (Gaussian) 燙平了，所以斜率很順。
- **AbsAngAcc** 是「直接拿放大鏡看毛邊的斜率變化」，而且還看了兩次 (斜率的斜率)，所以看到的都是毛邊的劇烈跳動。
