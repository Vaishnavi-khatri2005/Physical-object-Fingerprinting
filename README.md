# ShoePrint — Instance-Level Physical Object Identification

**ShoePrint** is a computer vision research pipeline and interactive application built to identify the **exact physical shoe instance** from a photograph. Unlike categorical classification (which simply identifies a brand or model), instance-level fingerprinting identifies the specific individual physical shoe by matching unique surface characteristics: micro-creases, wear and tear abrasion patterns, outsole tread erosion, and stitching nuances.

---

## 1. Problem Statement

In retail, forensic science, and inventory tracking, identifying an exact physical item rather than its generic product category is a challenging open-set vision problem. Two pairs of identical Nike sneakers from the same batch are visually identical when brand new. However, once worn:
* Unique micro-creasing forms along stress points of the leather/mesh.
* Asymmetrical outsole abrasion occurs based on the wearer's gait.
* Surface scuffs, stitch fraying, and stone punctures create a distinctive physical "fingerprint".

**Objective**: Given a single query photograph of a shoe taken with a mobile phone, identify which registered physical shoe it belongs to, or reject it if it is unknown or unconfident.

---

## 2. Dataset

The dataset was captured locally under realistic smartphone conditions:
* **7 Unique Physical Shoes**: `shoe1`, `shoe2`, `shoe3`, `shoe4`, `shoe5`, `shoe6`, `shoe7`.
* **154 Total Images**: $16$ to $31$ photographs per physical shoe.
* **Image Decodability**: $100\%$ valid JPEGs ($0$ corrupted files).
* **Resolutions**: $576 \times 576$ to $1280 \times 1280$ pixels.
* **Evaluation Split**: Deterministic 70/30 disjoint partition using `seed=42`:
  * **108 Reference (Gallery) Images**
  * **46 Held-Out Query (Test) Images**
  * *Strict Disjoint Rule*: No test image was ever used as a reference image, and zero self-matching was permitted.

| Shoe ID | Reference Images | Query Images | Total Photos |
|:---|:---:|:---:|:---:|
| `shoe1` | 15 | 6 | 21 |
| `shoe2` | 11 | 5 | 16 |
| `shoe3` | 22 | 9 | 31 |
| `shoe4` | 19 | 8 | 27 |
| `shoe5` | 15 | 7 | 22 |
| `shoe6` | 14 | 6 | 20 |
| `shoe7` | 12 | 5 | 17 |
| **TOTAL** | **108** | **46** | **154** |

---

## 3. Method Architecture

The pipeline uses classical local invariant feature matching and robust geometric verification:

1. **Feature Detection & Description (SIFT / RootSIFT)**:
   * Keypoints are detected across scale-space using Difference-of-Gaussians (DoG).
   * 128-dimensional local gradient orientation histograms are extracted (up to 2,500 features per image).
2. **RootSIFT Transformation** (Arandjelovic & Zisserman, CVPR 2012):
   * Each descriptor is $L_1$-normalized: $\mathbf{x} = \mathbf{d} / \|\mathbf{d}\|_1$.
   * Element-wise square root is applied: $\mathbf{x} = \sqrt{\mathbf{x}}$.
   * Euclidean distance on these vectors corresponds to the **Hellinger distance**, suppressing bursty repetitive texture bias.
3. **Descriptor Matching (BFMatcher + Lowe's Ratio Test)**:
   * Fast Brute-Force $k$-NN search ($k=2$).
   * Lowe's ratio test threshold = $0.75$ ($d_1 < 0.75 \times d_2$).
4. **Geometric Consistency Verification (RANSAC Homography)**:
   * Planar homography estimation ($H$) via RANSAC with a $5.0$-pixel reprojection error threshold.
   * Only point pairs that fit a shared spatial plane are retained as **inliers**.
5. **Instance Scoring**:
   * For each physical shoe candidate, its score is the **maximum RANSAC inlier count** achieved by any reference photograph of that shoe.

---

## 4. SIFT Baseline Results

Under the strict 46-query evaluation protocol, the classical unmasked SIFT baseline achieved:
* **Top-1 Accuracy**: **65.2%** (30 / 46 correct)
* **Top-3 Accuracy**: **78.3%** (36 / 46 in top-3)
* **Average Query Latency**: ~2.23 seconds/query

### Failure Analysis of the Baseline:
* Misclassifications averaged **69.9 inliers**, while correct predictions averaged **144.8 inliers**.
* `shoe3` had a **77.8% failure rate** (7 / 9 failed), primarily misclassified as `shoe5` due to background floor tiles and generic midsole ridges matching across photos.

---

## 5. Foreground-Isolation Experiment (Why It Was Rejected)

To eliminate floor and rug background keypoints, a lightweight OpenCV segmentation pipeline (CIELAB corner-guided $K$-Means clustering + morphological refinement) was tested before SIFT:

* **Top-1 Accuracy**: Dropped from **65.2% $\rightarrow$ 58.7%** (-6.5%).
* **Top-3 Accuracy**: Dropped from **78.3% $\rightarrow$ 73.9%** (-4.4%).
* **Why it was rejected**:
  * While it successfully helped high-contrast shoes (`shoe3` accuracy doubled from $22.2\% \rightarrow 44.4\%$, and `shoe4` reached $100\%$), it severely degraded darker shoes (`shoe5` dropped from $71.4\% \rightarrow 14.3\%$, and `shoe7` dropped from $80.0\% \rightarrow 20.0\%$).
  * The dark rubber and leather of `shoe5` and `shoe7` had low chromatic contrast with shadowy floor margins, causing the mask to accidentally clip the toe box and sole contours, discarding vital inliers.
  * **Conclusion**: Classical color-based foreground segmentation is too fragile across arbitrary shoe colors and was rejected from the final pipeline.

---

## 6. RootSIFT Improvement

Evaluating RootSIFT (with full unmasked images) delivered a decisive improvement across the board:

| Metric | Baseline SIFT | RootSIFT | Improvement |
|:---|:---:|:---:|:---:|
| **Top-1 Accuracy** | **65.2%** (30/46) | **78.3%** (36/46) | <span style="color:green">**+13.1%**</span> |
| **Top-3 Accuracy** | **78.3%** (36/46) | **93.5%** (43/46) | <span style="color:green">**+15.2%**</span> |
| **Query Latency** | ~2.23s | **1.12s** | **2x faster** |

### Per-Shoe Accuracy with RootSIFT:
* **`shoe1`**: 6 / 6 (**100.0%**)
* **`shoe2`**: 2 / 5 (**40.0%**)
* **`shoe3`**: 5 / 9 (**55.6%**) *(jumped from 22.2%)*
* **`shoe4`**: 8 / 8 (**100.0%**) *(jumped from 75.0%)*
* **`shoe5`**: 5 / 7 (**71.4%**)
* **`shoe6`**: 6 / 6 (**100.0%**) *(jumped from 83.3%)*
* **`shoe7`**: 4 / 5 (**80.0%**)

RootSIFT resolved 8 previous baseline failures with only 2 minor borderline regressions, achieving the highest identification rate without any segmentation artifacts.

---

## 7. Biometric Verification Results (FAR, FRR, EER)

Evaluating the RootSIFT pipeline as a 1:1 / open-set verification system:

* **Genuine Score Distribution** (same shoe): Mean = **108.2 inliers**, Median = **52.5**, Max = **627**
* **Impostor Score Distribution** (different shoes): Mean = **19.9 inliers**, Median = **8.0**, Max = **105**
* **Equal Error Rate (EER)**: **30.4%**
* **EER Decision Threshold**: **22 inliers**
  * **FAR (False Acceptance Rate)** at $\theta = 22$: **30.4%**
  * **FRR (False Rejection Rate)** at $\theta = 22$: **30.4%**

*(Note: Threshold $22$ is an experimental diagnostic metric for this split; not an optimal production threshold).*

---

## 8. Unknown-Object Testing Results

* Automatic open-set rejection is integrated via `photos_unknown/`.
* On the current repository, `photos_unknown/` is not provided, so no synthetic unknown data was fabricated.
* Any query scoring below the EER threshold of **22 inliers** is automatically rejected by the application as **UNKNOWN / NO CONFIDENT MATCH**.

---

## 9. How to Run the Application

### Installation
Ensure Python 3.9+ is installed:
```powershell
cd C:\Users\A\OneDrive\Desktop\shoeprint
pip install -r requirements.txt
```

### Launch the Streamlit Web App (Local PC & Smartphone)

To launch the web app and access it from your smartphone on the same Wi-Fi:
```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

* **On your PC**: Open `http://localhost:8501`
* **On your Smartphone**:
  1. Find your computer's local Wi-Fi IP address (e.g. run `ipconfig` in PowerShell; look for IPv4 Address under Wi-Fi, such as `192.168.1.9`).
  2. Open mobile browser (Chrome/Safari) and go to: `http://<your-laptop-ip>:8501` (e.g. `http://192.168.1.9:8501`).
  3. Tap **📷 Take a photo**, grant camera permission, snap the shoe, and view the instant identification result!
  4. (Fallback): Use **📁 Or upload an image** to pick an existing photo from your phone's camera roll.


### Command-Line Tools
* **Single Image Matching**:
  ```powershell
  python match.py --query "path/to/test_image.jpg" --dataset "photos"
  ```
* **Dataset Integrity Inspection**:
  ```powershell
  python inspect_dataset.py --dataset "photos"
  ```
* **Run RootSIFT Benchmark**:
  ```powershell
  python evaluate_rootsift.py
  ```
* **Run Biometric EER Evaluation**:
  ```powershell
  python evaluate_biometrics.py
  ```

---

## 10. Limitations

* **Controlled Dataset Scale**: Evaluated on 7 physical shoes (154 photos). Accuracy will naturally degrade as gallery size scales to hundreds of shoes without indexing.
* **3D Perspective & Non-Planar Surfaces**: RANSAC homography assumes locally planar surfaces. High-angle perspective changes between upper leather and outsole tread cannot be modeled by a single 2D plane.
* **Low-Texture Regions**: Smooth, uncreased leather or clean rubber without surface abrasion produces fewer keypoints, increasing FRR.
* **Lighting Variations**: Severe shadows or flash over-exposure distort local gradient histograms.

---

## 11. Future Improvements

1. **Learned Deep Local Features**: Replace handcrafted SIFT with modern robust matchers like **SuperPoint + LightGlue** for viewpoint invariance.
2. **Deep Metric Embeddings**: Train a deep Siamese / Triplet ResNet/ConvNeXt on shoe crops to provide global coarse filtering before local geometric verification.
3. **Robust Object Segmentation**: Use a trained segmentation model (such as MobileSAM or YOLOv8-seg) that reliably segments dark shoes without border clipping.
4. **Spatial Dispersion Scoring**: Weight inliers that span multiple distinct shoe zones (heel, toe, lateral side) higher than single-cluster matches.
