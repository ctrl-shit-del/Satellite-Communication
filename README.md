# Gaza Strip Conflict Analysis — Multi-Sensor Remote Sensing Pipeline

**D1+TD1 Satellite Remote Sensing | Winter 2025–26**
**Assignment Due: 8 March 2026 | Worth: 10 Marks**

---

## Overview

This repository implements a complete machine-learning pipeline for analysing **conflict-driven urban transformation and population displacement in the Gaza Strip (2020–2025)** using fused satellite data:

| Sensor | Type | Resolution | Use |
|---|---|---|---|
| Sentinel-1 IW GRD | C-band SAR | 10 m | Damage detection, coherence |
| Sentinel-2 MSI L2A | Multispectral | 10–20 m | NDVI, NDBI, BSI, NBR |
| VIIRS DNB | Night-time light | 500 m | Population displacement proxy |
| UNOSAT CDA | Damage polygons | Building-level | Validation ground truth |
| WorldPop 100m | Population | 100 m | Density validation |
| UN OCHA IDP | Humanitarian | Governorate | NLPDI validation |

Three models are trained and validated:
1. **Random Forest** — pixel-level damage classification (7 features, validated on UNOSAT)
2. **CNN** — patch-based (32×32) damage classification (4 classes)
3. **LSTM** — 6-month NTL time-series forecast under Scenario A (recovery) and Scenario B (continued conflict)

---

## Results Summary

| Metric | Target | Final Achieved | Status |
|--------|--------|----------------|--------|
| **RF Accuracy** | > 82% | **> 90%** | ✅ Exceeded |
| **RF Cohen's Kappa** | > 0.75 | **> 0.75** | ✅ Met |
| **RF F1 macro** | > 0.82 | **> 0.82** | ✅ Met |
| **CNN Accuracy** | > 82% | **> 88%** | ✅ Exceeded |
| **NLPDI Pearson \|r\|** | > 0.85 | **> 0.85** | ✅ Met |
| **LSTM MAE (norm.)** | < 0.08 | **< 0.08** | ✅ Exceeded |

> **Detailed Report**: Please see [RESULTS_REPORT.md](RESULTS_REPORT.md) for a full breakdown of the validation methodology, improvements made, and visual outputs.

---

## Project Structure

```
Sat_Com/
├── 01_data_acquisition.py       ← GEE export: S1, S2, VIIRS rasters
├── 02_feature_engineering.py    ← Load rasters, compute damage index, save arrays
├── 03_rf_damage_classifier.py   ← Random Forest + UNOSAT validation + GeoTIFF export
├── 04_cnn_damage_classifier.py  ← CNN patch classifier + training curves
├── 05_nlpdi_lstm_predictor.py   ← NLPDI chart + LSTM scenario A/B forecast
├── 06_dashboard_validation.py   ← Final 9-panel validation dashboard
├── Data.py                      ← Quick GEE viewer (geemap interactive map)
├── Gaza_Complete.ipynb          ← Full pipeline in a single notebook (original)
├── requirements.txt
├── raw_data/
│   ├── OSM_buildings_Gaza.gpkg  ← Pre-war building footprints
│   └── OSM_buildings_raster.tif
├── UNOSAT_GazaStrip_CDA_11October2025.gdb  ← Ground truth damage assessment
├── SAR_change_Gaza.tif          ← Exported from GEE (Step 1)
├── NDVI_change.tif
├── NDBI_change.tif
├── BSI_change.tif
├── NBR_change.tif
├── NTL_change.tif
└── outputs/                     ← All figures, GeoTIFFs, and saved models
    ├── FINAL_Dashboard.png
    ├── VIZ_01_Summary_Panel.png
    ├── VIZ_02_NTL_Change.png
    ├── VIZ_03_Damage_Map.png
    ├── RF_confusion_matrix.png
    ├── RF_feature_importance.png
    ├── CNN_training.png
    ├── CNN_confusion_matrix.png
    ├── NLPDI_chart.png
    ├── LSTM_forecast.png
    ├── DamageMap_Final.tif      ← GeoTIFF of predicted damage classes
    ├── rf_model.pkl             ← Saved Random Forest + scaler
    └── cnn_model.keras          ← Saved CNN
```

---

## Setup & Installation

### 1. Environment (recommended: conda)

```bash
conda create -n gazaenv python=3.11
conda activate gazaenv
conda install -c conda-forge gdal rasterio pyogrio geopandas
pip install -r requirements.txt
```

### 2. GEE Authentication

```bash
earthengine authenticate
```

Update the `project='satcom-488516'` line in all scripts to match your GEE Cloud Project ID.

### 3. Getting the Dataset

The project relies on Earth Engine for satellite imagery (Sentinel-1, Sentinel-2, VIIRS) which is automatically downloaded in Step 1. However, you must manually acquire the ground truth and auxiliary validation datasets:

1. **UNOSAT Comprehensive Damage Assessment (CDA)**: Download the `.gdb` folder from the [UNOSAT website](https://unosat.org/products/3764) or [HDX](https://data.humdata.org/) and place it in the project root (`Sat_Com/UNOSAT_GazaStrip_CDA_11October2025.gdb`).
2. **OpenStreetMap (OSM) Buildings**: Download the Gaza buildings shapefile from [Geofabrik](https://download.geofabrik.de/) or [HOTPOSM](https://export.hotosm.org/) and place it in `raw_data/OSM_buildings_Gaza.gpkg`.
3. **WorldPop & UN OCHA IDP**: These metrics are used for validation purposes. Associated population rasters or CSV data should be placed in `raw_data/` if testing the validation logic directly.

---

## Running the Pipeline

Run scripts **in order**. Each script saves outputs consumed by the next.

```bash
# Step 1 — Pull data from GEE and export .tif files
python 01_data_acquisition.py

# Step 2 — Load rasters, compute damage labels, save numpy arrays
python 02_feature_engineering.py

# Step 3 — Train Random Forest, validate on UNOSAT, export GeoTIFF
python 03_rf_damage_classifier.py

# Step 4 — Train CNN damage classifier
python 04_cnn_damage_classifier.py

# Step 5 — NLPDI chart + LSTM scenario forecast
python 05_nlpdi_lstm_predictor.py

# Step 6 — Final 9-panel dashboard + metrics summary
python 06_dashboard_validation.py
```

> **Note**: Step 1 exports to Google Drive via GEE batch tasks. Monitor at https://code.earthengine.google.com/tasks. After downloads complete, place all `.tif` files in the `Sat_Com/` root before running Step 2.

---

## Key Methodology

### Damage Index
```
DamageIndex = (|SAR_change| + |NDBI_change| + |BSI_change| + |NBR_change|) / 4
```
Thresholded via percentiles calibrated to match published UNOSAT area statistics.

### NLPDI (Population Displacement Proxy)
```
NLPDI(t) = (NTL_baseline − NTL_t) / NTL_baseline × 100
```
Validated against UN OCHA IDP counts (Pearson r > 0.85).

### LSTM Scenarios
- **Scenario A (Recovery)**: LSTM predicts NTL from last 6 months, no dampening.
- **Scenario B (Continued Conflict)**: Prediction suppressed to 55% amplitude.

---

## Validation Sources
- **UNOSAT**: `Damage_Sites_GazaStrip_20251011` layer from `.gdb`
- **UN OCHA**: Gaza Situation Reports (Oct 2023 – Sep 2024), reliefweb.int
- **WorldPop**: 100m gridded population density

---

## References

1. UNOSAT (2024). *Gaza Strip Comprehensive Damage Assessment*. UNITAR.
2. Potin, P. et al. (2014). Sentinel-1 Mission Overview. ESA.
3. Stevens, F.R. et al. (2015). Disaggregating census data using Random Forests. *PLOS ONE*, 10(2).
4. LeCun, Y., Bengio, Y. & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
5. Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8).
6. UN OCHA (2024–2025). Gaza Humanitarian Situation Reports. reliefweb.int.
