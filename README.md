# EMG Gesture Classification - Sliding Window vs Event-Based (Ninapro DB3)

This repository contains the complete pipeline for classification of hand gestures based on EMG signals using two different approaches:
- **Sliding Window** segmentation
- **Event-Based** segmentation

Both approaches are evaluated on the **Ninapro Database 3 (DB3)**.

---

## 📦 Environment Setup

### 1. Install dependencies

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

*Note:* We recommend Python version 3.11 or later.

### 2. Project Structure

```
/your-project-folder
    ├── README.md
    ├── requirements.txt
    ├── main_slide_v2.py
    ├── main_event_based_v2.py
    ├── analise_comparativa_global_v2.py
    ├── /data
    │     └── (folder where Ninapro DB3 files will be placed)
    ├── /resultados_analise
    │     └── (generated automatically after running scripts)
```

---

## 📂 Data Organization

- The `/data` folder must contain the files from the **Ninapro Database 3 (DB3)**.
- The expected organization is:
    - Subject folders named as `s1_0`, `s2_0`, `s3_0`, ..., etc.
    - Inside each subject folder, the `.mat` files should be organized by exercise:
        - Example: `DB3_s1/S1_E1_A1.mat`, `DB3_s1/S1_E2_A1.mat`, etc.

### How to Obtain the Data:
- Access the official [Ninapro database page](http://ninapro.hevs.ch/DB3)
- Request access and download the DB3 files.
- Extract them maintaining the subject folder organization.

*Important:*  
Only `.mat` files containing the EMG signals and movement labels (`emg` and `restimulus`) are needed.

---

## 🛠️ How to Run the Scripts

1. **Sliding Window Analysis**

```bash
python main_slide_v2.py
```

This will perform feature extraction, forward feature selection, model training and validation using sliding window segmentation.

2. **Event-Based Analysis**

```bash
python main_event_based_v2.py
```

This will perform the same pipeline, but based on segmenting complete movement events instead of fixed windows.

3. **Global Comparative Analysis**

```bash
python analise_comparativa_global_v2.py
```

This script loads all previous results and generates:
- Summary tables
- Statistical tests
- Scientific plots ready for papers (SVG/PDF)

---

## 🗂️ Outputs

All generated outputs (results, models, plots) will be saved inside:

```
/resultados_analise/
```

The outputs include:
- FFS history CSVs
- Fold results CSVs
- Statistical test results (Excel)
- Final models (`.pkl`)
- Plots (`.svg` and `.pdf`) ready for publication

---

# EMG Gesture Classification - Sliding Window vs Event-Based (Ninapro DB3)

This repository contains the complete pipeline for classification of hand gestures based on EMG signals using two different approaches:
- **Sliding Window** segmentation
- **Event-Based** segmentation

Both approaches are evaluated on the **Ninapro Database 3 (DB3)**.

---

## 📦 Environment Setup

### 1. Install dependencies

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

*Note:* We recommend Python version 3.11 or later.

### 2. Project Structure

```
/your-project-folder
    ├── README.md
    ├── requirements.txt
    ├── main_slide_v2.py
    ├── main_event_based_v2.py
    ├── analise_comparativa_global_v2.py
    ├── /data
    │     └── (folder where Ninapro DB3 files will be placed)
    ├── /resultados_analise
    │     └── (generated automatically after running scripts)
```

---

## 📂 Data Organization

- The `/data` folder must contain the files from the **Ninapro Database 3 (DB3)**.
- The expected organization is:
    - Subject folders named as `s1_0`, `s2_0`, `s3_0`, ..., etc.
    - Inside each subject folder, the `.mat` files should be organized by exercise:
        - Example: `DB3_s1/S1_E1_A1.mat`, `DB3_s1/S1_E2_A1.mat`, etc.

### How to Obtain the Data:
- Access the official [Ninapro database page](http://ninapro.hevs.ch/DB3)
- Request access and download the DB3 files.
- Extract them maintaining the subject folder organization.

*Important:*  
Only `.mat` files containing the EMG signals and movement labels (`emg` and `restimulus`) are needed.

---

## 🛠️ How to Run the Scripts

1. **Sliding Window Analysis**

```bash
python main_slide_v2.py
```

This will perform feature extraction, forward feature selection, model training and validation using sliding window segmentation.

2. **Event-Based Analysis**

```bash
python main_event_based_v2.py
```

This will perform the same pipeline, but based on segmenting complete movement events instead of fixed windows.

3. **Global Comparative Analysis**

```bash
python analise_comparativa_global_v2.py
```

This script loads all previous results and generates:
- Summary tables
- Statistical tests
- Scientific plots ready for papers (SVG/PDF)

---

## 🗂️ Outputs

All generated outputs (results, models, plots) will be saved inside:

```
/resultados_analise/
```

The outputs include:
- FFS history CSVs
- Fold results CSVs
- Statistical test results (Excel)
- Final models (`.pkl`)
- Plots (`.svg` and `.pdf`) ready for publication

---

## 🖋️ Methodology Details

### Database
The Ninapro DB3 database was used, which contains EMG recordings from 12 electrodes placed on the forearm, captured during the execution of different hand and wrist movements.

### Preprocessing
- Signals were loaded from `.mat` files containing `emg` and `restimulus` variables.
- The sampling rate is 2 kHz.
- Only the EMG signals were used; no additional preprocessing or filtering was applied before feature extraction.

### Feature Extraction
- Features were computed over either:
  - Fixed-length sliding windows (Sliding Window method)
  - Entire movement segments based on restimulus labels (Event-Based method)

- Extracted features include:
  - Mean Absolute Value (MAV)
  - Root Mean Square (RMS)
  - Waveform Length (WL)
  - Zero Crossing (ZC)
  - Slope Sign Changes (SSC)
  - Integrated EMG (IEMG)
  - Willison Amplitude (WAMP)

Features were extracted independently from all 12 EMG channels and concatenated into a single feature vector.

### Forward Feature Selection (FFS)
- A forward selection strategy was applied to select the best subset of features.
- The evaluation metric was the classification accuracy over a simple train/test split.
- Support Vector Machine (SVM) with a linear kernel was used during feature selection.

### Model Training and Validation
- After feature selection, a final model was trained and evaluated using 10-fold stratified cross-validation.
- Classifiers evaluated include:
  - Support Vector Machine (SVM, linear kernel)

### Statistical Analysis
- A Wilcoxon signed-rank test was applied to compare the performance between Sliding and Event-Based approaches for each exercise.
- Results were considered statistically significant when p < 0.05.

### Output Files
- Feature selection history for each method and exercise (`ffs_history_*.csv`)
- Fold-level cross-validation results (`folds_result_*.csv`)
- Statistical analysis results (`resultado_estatistica.xlsx`)
- Final trained models (`modelo_final_*.pkl`)
- Graphs in `.svg` and `.pdf` formats suitable for scientific publication

---

For any questions or suggestions, feel free to open an issue!

