
# Keypoint-MoSeq Motif Merging Strategy
**Date:** 2026-01-04
**Model:** 20260104-053710-3
**Author:** 玉茗 (Camellia)

## Philosophy
The goal of this merging strategy is to address the issue of "motif fragmentation" (over-segmentation) while preserving the stability of the ethogram. We adhere to the core philosophy of Keypoint-MoSeq: **similar behaviors have similar representations in the latent space.**

## Criteria
1.  **Stability Threshold:** A motif is considered "Stable" if it has at least one continuous instance lasting **10 frames or more**. All other motifs are classified as "Short/Fragmented" (likely noise or transitional glitches).
    *   **Total Motifs:** 93
    *   **Stable Motifs:** 38
    *   **Short Motifs:** 55

2.  **Merging Logic (Nearest Neighbor in Latent Space):**
    *   For each motif, we calculated its **Centroid** in the model's high-dimensional latent space (`model['states']['x']`).
    *   Each "Short Motif" was assigned to the "Stable Motif" with the closest centroid (based on Euclidean distance).
    *   This ensures that fragmented video frames are reassigned to the robust behavior they most statistically resemble.

## Merging Map
The following merges were applied:

| Main (Stable) | Merged (Short) |
| :--- | :--- |
| **0** | 68 |
| **1** | 58 |
| **2** | 7, 75 |
| **3** | 85 |
| **4** | 19, 37 |
| **5** | 45, 59 |
| **6** | 30, 80, 87, 86 |
| **10** | 24, 69, 74, 78 |
| **12** | 9, 16 |
| **13** | 27, 44, 64 |
| **14** | 23 |
| **18** | 33 |
| **31** | 73 |
| **32** | 83 |
| **34** | 40 |
| **36** | 89 |
| **42** | 20, 28 |
| **46** | 92 |
| **49** | 17, 22 |
| **50** | 8, 11, 39 |
| **52** | 41 |
| **54** | 25, 35, 53, 71 |
| **55** | 79 |
| **56** | 48 |
| **57** | 91 |
| **60** | 62 |
| **61** | 63, 84 |
| **70** | 15 |
| **72** | 38 |
| **76** | 90 |
| **77** | 51, 66, 82, 88 |
| **81** | 47 |

This reduces the total number of unique behavioral motifs from **93** to **38**, significantly cleaning up the analysis.
