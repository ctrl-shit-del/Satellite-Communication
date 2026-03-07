"""
=============================================================================
 03_RF_DAMAGE_CLASSIFIER.PY — GAZA STRIP CONFLICT ANALYSIS
 D1+TD1 Satellite Remote Sensing | Winter 2025-26
=============================================================================
 PURPOSE : Train a Random Forest classifier ON UNOSAT GROUND-TRUTH labels
           (70% train / 30% test stratified split) and validate on the
           withheld test set. Outputs full-scene GeoTIFF and metrics.

 KEY FIX vs earlier version:
   - OLD: RF trained on synthetic rule-based pixel labels → validated on
          UNOSAT (different domain) → Accuracy=29.7%, Kappa=0.030
   - NEW: RF trained directly on UNOSAT-labeled pixel coordinates →
          validated on a 30% withheld UNOSAT split → accurate, fair metrics.

 INPUTS  (from 02_feature_engineering.py outputs):
   outputs/bands_stack.npy    — 7-band feature cube  (H × W × 7)
   UNOSAT_GazaStrip_CDA_11October2025.gdb  — ground truth

 OUTPUTS:
   RF_confusion_matrix.png, RF_feature_importance.png,
   VIZ_03_Damage_Map.png, DamageMap_Final.tif, rf_model.pkl
=============================================================================
"""

import os, pickle
import numpy as np
import rasterio
from rasterio.transform import rowcol
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, cohen_kappa_score, f1_score)
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(BASE, 'outputs')
GDB   = os.path.join(BASE, 'UNOSAT_GazaStrip_CDA_11October2025.gdb')
LAYER = 'Damage_Sites_GazaStrip_20251011'
os.makedirs(OUT, exist_ok=True)

DAMAGE_NAMES  = ['No Damage', 'Minor Damage', 'Moderate Damage', 'Severe/Destroyed']
DAMAGE_COLORS = ['#27ae60', '#f1c40f', '#e67e22', '#c0392b']
FEAT_NAMES    = ['SAR_chg', 'NDBI_chg', 'NDVI_chg', 'BSI_chg',
                 'NBR_chg', 'NTL_chg', 'Damage_Index']
PLT_DPI = 150

print("=" * 65)
print("  RANDOM FOREST DAMAGE CLASSIFIER — GAZA (UNOSAT-trained)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD FEATURE ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading feature arrays...")
bands_stack   = np.load(os.path.join(OUT, 'bands_stack.npy'))
damage_labels = np.load(os.path.join(OUT, 'damage_labels.npy'))   # for map display only
h, w          = damage_labels.shape

with rasterio.open(os.path.join(BASE, 'SAR_change_Gaza.tif')) as src:
    transform = src.transform
    crs       = src.crs
    profile   = src.profile
print(f"  Stack: {bands_stack.shape}  CRS: {crs}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD UNOSAT POINTS — extract pixel features at each building centroid
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Extracting features at UNOSAT building centroids...")

# UNOSAT class: 1=Destroyed → label 3, 2=Severe → 2, 3=Moderate → 1, 4=None → 0
CLASS_MAP = {1: 3, 2: 2, 3: 1, 4: 0}
gdf = gpd.read_file(GDB, layer=LAYER, engine='pyogrio').to_crs(crs)
gdf = gdf[gdf['Main_Damage_Site_Class_2'].notna()].copy()
gdf['label'] = gdf['Main_Damage_Site_Class_2'].astype(int).map(CLASS_MAP)
gdf = gdf[gdf['label'].notna()].copy()

# Extract features at each UNOSAT point + 3×3 neighbourhood (for sample volume)
X_pts, y_pts = [], []
offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
for _, pt in gdf.iterrows():
    try:
        r0, c0 = rowcol(transform, pt.geometry.x, pt.geometry.y)
        r0, c0 = int(r0), int(c0)
        lbl    = int(pt['label'])
        for dr, dc in offsets:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < h and 0 <= c < w:
                feat = np.nan_to_num(bands_stack[r, c, :], nan=0.0)
                X_pts.append(feat)
                y_pts.append(lbl)
    except Exception:
        continue

X_pts = np.array(X_pts, dtype=np.float32)
y_pts = np.array(y_pts, dtype=np.int32)
print(f"  UNOSAT samples (with neighbourhood): {len(y_pts):,}")
print(f"  Class distribution: {Counter(y_pts.tolist())}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN RANDOM FOREST ON UNOSAT LABELS (70% train / 30% test)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Training Random Forest...")

scaler = StandardScaler()
X_s    = scaler.fit_transform(X_pts)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_s, y_pts, test_size=0.30, random_state=42, stratify=y_pts)
print(f"  Train: {len(y_tr):,}  |  Test: {len(y_te):,}")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_tr, y_tr)

# --- Metrics on held-out test set ---
y_pred = rf.predict(X_te)
rf_acc   = accuracy_score(y_te, y_pred)
rf_kappa = cohen_kappa_score(y_te, y_pred)
rf_f1    = f1_score(y_te, y_pred, average='macro')

# 5-fold cross-validated accuracy for robustness
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, max_features='sqrt',
                           class_weight='balanced', n_jobs=-1, random_state=42),
    X_s, y_pts, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy', n_jobs=-1)

print(f"\n  ── RF RESULTS (UNOSAT 30% test) ──")
print(f"  Accuracy : {rf_acc*100:.2f}%  (target >82%)")
print(f"  Kappa    : {rf_kappa:.4f}  (target >0.75)")
print(f"  F1 macro : {rf_f1:.4f}  (target >0.82)")
print(f"  5-fold CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(classification_report(y_te, y_pred, target_names=DAMAGE_NAMES))

# Save model
with open(os.path.join(OUT, 'rf_model.pkl'), 'wb') as f:
    pickle.dump({'model': rf, 'scaler': scaler,
                 'rf_acc': rf_acc, 'rf_kappa': rf_kappa, 'rf_f1': rf_f1,
                 'fi': rf.feature_importances_}, f)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FULL-SCENE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Full-scene prediction...")

flat   = np.nan_to_num(bands_stack.reshape(-1, bands_stack.shape[2]), nan=0.0)
flat_s = scaler.transform(flat)
BATCH, preds = 500_000, np.zeros(flat_s.shape[0], dtype=np.uint8)
for i in range(0, flat_s.shape[0], BATCH):
    preds[i:i+BATCH] = rf.predict(flat_s[i:i+BATCH])
    print(f"  {min(i+BATCH, flat_s.shape[0])/flat_s.shape[0]*100:.0f}%...", end='\r')
ml_map = preds.reshape(h, w)

out_prof = profile.copy(); out_prof.update(dtype='uint8', count=1, nodata=255)
tif_path = os.path.join(OUT, 'DamageMap_Final.tif')
with rasterio.open(tif_path, 'w', **out_prof) as dst:
    dst.write(ml_map.astype('uint8').reshape(1, h, w))
print(f"\n  ✅ GeoTIFF saved: {tif_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving visualisations...")
fi     = rf.feature_importances_
cmap_c = mcolors.ListedColormap(DAMAGE_COLORS)
norm_c = mcolors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], cmap_c.N)

# Confusion Matrix
cm = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=DAMAGE_NAMES, yticklabels=DAMAGE_NAMES, linewidths=0.5)
ax.set_title(f'RF Confusion Matrix (UNOSAT 30% Test)\n'
             f'Acc={rf_acc*100:.1f}%  Kappa={rf_kappa:.3f}  F1={rf_f1:.3f}',
             fontsize=11, fontweight='bold')
ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'RF_confusion_matrix.png'), dpi=PLT_DPI, bbox_inches='tight')
plt.close()

# Feature Importance
colors = ['#e74c3c' if v == max(fi) else '#3498db' for v in fi]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(FEAT_NAMES, fi, color=colors)
ax.axvline(1/len(FEAT_NAMES), color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax.set_title('RF Feature Importance (Red=Most Important)', fontsize=11, fontweight='bold')
ax.set_xlabel('Importance'); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'RF_feature_importance.png'), dpi=PLT_DPI, bbox_inches='tight')
plt.close()

# Side-by-side map
fig, axes = plt.subplots(1, 2, figsize=(20, 11))
for ax, (data, title) in zip(axes, [
    (damage_labels, 'Rule-Based Index Map'),
    (ml_map, f'RF Prediction (UNOSAT-trained)\nAcc={rf_acc*100:.1f}%  Kappa={rf_kappa:.3f}')
]):
    im = ax.imshow(data, cmap=cmap_c, norm=norm_c, interpolation='nearest', aspect='auto')
    ax.set_title(title, fontsize=12, fontweight='bold'); ax.axis('off')
    ax.legend(handles=[mpatches.Patch(color=c, label=n)
                       for c, n in zip(DAMAGE_COLORS, DAMAGE_NAMES)],
              loc='lower left', fontsize=9, framealpha=0.9)
plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02, ticks=[0,1,2,3]).ax.set_yticklabels(DAMAGE_NAMES)
plt.suptitle('Gaza Damage Classification — Index Map vs UNOSAT-Trained RF',
             fontsize=14, fontweight='bold')
plt.savefig(os.path.join(OUT, 'VIZ_03_Damage_Map.png'), dpi=PLT_DPI, bbox_inches='tight')
plt.close()

print(f"\n📊 RF Accuracy: {rf_acc*100:.2f}% | Kappa: {rf_kappa:.4f} | F1: {rf_f1:.4f}")
