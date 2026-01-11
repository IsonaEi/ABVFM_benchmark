# CASTLE 行為分析模型驗證計畫 (Benchmark Strategy)

## 核心目標
證明 **CASTLE** 作為基於視覺特徵（DINO-based / Human-Guided）的模型，在以下方面優於傳統基於骨架（Keypoint-based）的方法（如 B-SOID, KPMS）：
1.  **時間精確度 (Temporal Precision)**
2.  **細微動作感知能力 (Micro-movement Perception)**
3.  **生物學意義 (Biological Relevance)**

---

## Figure 1: 物理層驗證 (Physical Validation)

**戰略目標**：證明 CASTLE 切分出的「狀態轉換點 (Transitions)」具有多重物理意義，且在時間對齊上具有統計顯著性。

### Panel A: 多維度物理變化分數對齊 (Multi-channel Physical Change Scores)

**圖表內容**：繪製時間軸圖 (Time-series traces)。以 CASTLE 偵測到的 Transition 為中心 ($t=0$)，疊加顯示以下物理指標的平均波形：

*   **Locomotion Score**: 全局移動速度 (Velocity) 與 加加速度 (Jerk)。(驗證跑/停切換)
*   **Orientation Score**: 頭部/身體的角度變化率 (Angular velocity)。(驗證轉向)
*   **Morphological Score**: 二維投影變形率 (2D Projection Foreshortening) 或 長寬比 (Aspect Ratio) 變化。(驗證 Rearing, Hunching, Stretching)
*   **Optical Flow Score (Enhanced)**:
    *   **Methodology**: 使用 **GMFlow (Sintel Pretrained)** 進行全解析度 (720x720) 光流估計。
    *   **Robustness**: 實作 **動態背景補償 (Dynamic Background Compensation)** 以消除全域閃爍 (Global Flickering)，並採用 **95th Percentile** 聚合策略，確保捕捉到局部微細動作（如耳動、鼻動）而不被靜止的身體平均掉。
    *   **Residual Motion**: 扣除骨架移動後的殘差，精確鎖定 "Killer Case" (High Vision Flow, Low Skeleton Velocity)。

**預期結果**：
CASTLE 的轉換點應精準落在這些物理指標的峰值 (Peak) 上。

> **核心指標：State Stability Index (SSI)**
>
> 比較轉換點前後的時間窗（Time Window, e.g., +/- 0.5s）內的特徵一致性。
>
> $$ SSI = \frac{\text{Inter-state Distance}}{\text{Intra-state Variance}} $$
>
> **預期結果**：CASTLE 的轉換點具有更高的 SSI，證明其切分出的是具有穩定性 (Stability) 的行為狀態，而非模型雜訊 (Flickering)。

### Panel B: 統計顯著性檢定 (Statistical Significance)

**圖表內容**：Box plot 或 Violin plot。

**比較對象**：
1.  **CASTLE** (Real Transitions)
2.  **Shuffled Control** (隨機時間點)
3.  **Keypoint-based Methods** (如 B-SOID)

**指標**：在轉換點附近的「物理變化峰值強度 (Peak Amplitude)」。

**統計方法**：使用 **Mann-Whitney U test (Two-sided)** 證明 CASTLE 的峰值顯著高於 Control Group，且與 B-SOID 相當或更優（證明沒有因為是弱監督而犧牲時間精確度）。
**輸出**：生成 `significance_test.csv` 表格，列出 U-Stat 與 p-value。

---

## Figure 2: 資料中心與視覺驗證 (Data-centric Validation)

### Panel A: 殘差運動分析 (Residual Motion Analysis)

**戰略目標**：證明 CASTLE 能捕捉到「骨架看不到」的視覺資訊。

**概念**：特別挑選 **「骨架速度低」但「光流強度高」** 的片段，並分析這些片段是否伴隨著行為轉換。

**圖表**：`trace_feature_ResidualMotion.png` 顯示了在 Transition 附近的殘差能量變化。

### Panel B: 特徵完備性與維度分析 (Feature Completeness & Dimensionality)

**戰略目標**：證明 DINO 特徵 (CASTLE) 包含了 Keypoint 缺失的細微資訊 (Texture, Body Surface Dynamics)。

**1. 特徵重建分析 (Reconstruction Analysis)**
*   **方法**：使用 **Ridge Regression (L2 Regularization)** 訓練映射：$f: DINO \to KP$ 和 $g: KP \to DINO$。
*   **對照組 (Random Baseline)**：使用與輸入特徵同維度的隨機高斯雜訊 (Random Gaussian Noise) 作為輸入進行重建，作為 Chance Level。
*   **指標**：$R^2$ Score (Coefficient of Determination)。
*   **預期結果**：
    *   **Chance Level**：隨機特徵的 $R^2$ 應接近 0，證明 DINO 與 KP 的關聯並非來自高維度過擬合。
    *   $R^2_{DINO \to KP}$ 很高 ($\approx 1.0$)：證明 DINO 特徵包含了所有骨架資訊。
    *   $R^2_{KP \to DINO}$ 顯著較低 (< 1.0)：證明 **骨架資訊不足以重建視覺特徵**，即 DINO 捕捉到了額外的生物學資訊 (Information Gain)。

**2. 內在維度分析 (Intrinsic Dimensionality)**
*   **方法**：PCA Cumulative Variance Analysis。
*   **指標**：解釋 90% 變異量所需的主成分數量 (Number of PCs for 90% Variance)。
*   **預期結果**：DINO 空間的有效維度應高於 Keypoint 空間，反映了行為的複雜度被更完整地保留。

### Panel C: 行為定義一致性 (Behavioral Mapping Consistency)

**圖表內容**：成對混淆矩陣 (Pairwise Confusion Matrices) - `confusion_CASTLE_vs_BSOID.png` 等。

**分析邏輯**：
*   **行歸一化 (Row-Normalized)**：計算 $P(\text{Method B Label} | \text{Method A Label})$。
*   **用途**：
    1.  **類別對應 (Class Mapping)**：自動發現不同方法間的語意對應關係 (e.g., CASTLE Class 3 $\approx$ B-SOID Group 5)。
    2.  **混淆分析 (Ambiguity check)**：若 CASTLE 的一個狀態對應到 B-SOID 的多個雜亂狀態，且該狀態具有高 SSI，則證明 CASTLE 成功整合了被其他方法過度切碎的行為。

---

## Figure 3: 行為動力學與組成 (Behavioral Dynamics & Ethogram)

**戰略目標**：直觀展示與量化行為的時序結構與分佈差異。

### Panel A: 行為譜 (Ethogram Visualization)

**圖表內容**：色彩編碼的時間軸 Barcode (`benchmark_fig3_style.png` Top Panel)。
**用途**：
*   **定性分析**：肉眼檢視行為的破碎程度 (Fragmentation)。Keypoint 方法通常會產生過度切換 (Flickering)，表現為密集的條紋；而 CASTLE 應展現出更連貫的區塊 (Block-like structure)。

### Panel B: 狀態持續時間分佈 (State Duration Distribution)

**圖表內容**：Histogram + Median Marker (`benchmark_fig3_style.png` Right Panel)。
**指標**：Median Duration (ms)。
**預期結果**：預期 CASTLE 的行為持續時間顯著長於 Keypoint-based 方法，證明其捕捉到了更宏觀的動作單元 (Action Units) 而非僅是姿態 (Pose) 的瞬間變化。

### Panel C: 統計彙總 (Statistical Summary)

**輸出**：`statistics.csv`
*   **Number of Transitions**: 轉換次數 (反向指標，越低通常代表越穩定)。
*   **Mean/Median Duration**: 平均/中位數持續時間。
*   **Mean/Median SSI**: 狀態穩定性指標。

---

## Future Work (Functional Validation)

> 此部分為後續生物學驗證計畫，目前程式碼尚未包含。
> *   Line Classification (WT vs Disease)
> *   Standard Benchmark (CalMS21)
