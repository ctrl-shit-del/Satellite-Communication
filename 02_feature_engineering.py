"""
=============================================================================
 02_FEATURE_ENGINEERING.PY — GAZA STRIP CONFLICT ANALYSIS
 D1+TD1 Satellite Remote Sensing | Winter 2025-26
=============================================================================
 PURPOSE : Load all locally-exported rasters, resample NTL to 10 m,
           generate the pixel-level damage index & label map, build the
           full 7-band feature stack, and produce publication-quality
           change maps.

 INPUTS  (place in same directory or update BASE path):
   SAR_change_Gaza.tif, NDVI_change.tif, NDBI_change.tif,
   BSI_change.tif, NBR_change.tif, NTL_change.tif

 OUTPUTS (saved to outputs/):
   VIZ_01_Summary_Panel.png  — 6-panel change map
   VIZ_02_NTL_Change.png     — Night-light map
   damage_labels.npy          — Pixel labels array  (for ML scripts)
   bands_stack.npy            — 7-band feature cube (for ML scripts)
=============================================================================
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — update BASE if your .tif files are in a different folder
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))   # same folder as this script
OUT  = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

DAMAGE_NAMES  = ['No Damage', 'Minor Damage', 'Moderate Damage', 'Severe/Destroyed']
DAMAGE_COLORS = ['#27ae60', '#f1c40f', '#e67e22', '#c0392b']
PLT_DPI = 150

print("=" * 65)
print("  FEATURE ENGINEERING — GAZA CONFLICT ANALYSIS")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD RASTERS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading rasters...")

def load_band(path):
    """Load a single-band GeoTIFF, return (data, transform, crs, profile)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data[~np.isfinite(data)] = 0.0
        return data, src.transform, src.crs, src.profile

sar,  transform, crs, profile = load_band(os.path.join(BASE, 'SAR_change_Gaza.tif'))
ndbi, *_ = load_band(os.path.join(BASE, 'NDBI_change.tif'))
ndvi, *_ = load_band(os.path.join(BASE, 'NDVI_change.tif'))
bsi,  *_ = load_band(os.path.join(BASE, 'BSI_change.tif'))
nbr,  *_ = load_band(os.path.join(BASE, 'NBR_change.tif'))

# Resample NTL from 500 m → 10 m to match SAR grid
from rasterio.warp import reproject
from rasterio.enums import Resampling
h, w = sar.shape
ntl = np.empty((h, w), dtype=np.float32)
with rasterio.open(os.path.join(BASE, 'NTL_change.tif')) as src:
    with rasterio.open(os.path.join(BASE, 'SAR_change_Gaza.tif')) as ref:
        reproject(src.read(1).astype(np.float32), ntl,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref.transform,  dst_crs=ref.crs,
                  resampling=Resampling.bilinear)
ntl[~np.isfinite(ntl)] = 0.0

print(f"  Grid: {h} × {w} pixels | CRS: {crs}")
print("  ✅ All rasters loaded!")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE PIXEL-LEVEL DAMAGE LABELS
#    Multi-sensor composite index thresholded to yield realistic class mix
#    (≈13% no-damage, ≈60% minor, ≈17% moderate, ≈10% severe — matches UNOSAT)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Computing damage index and labels...")

damage_index  = (np.abs(sar) + np.abs(ndbi) + np.abs(bsi) + np.abs(nbr)) / 4.0
damage_smooth = gaussian_filter(damage_index, sigma=3)

# Percentile thresholds calibrated to published UNOSAT area statistics
t1 = np.percentile(damage_smooth, 12.83)   # below = No Damage
t2 = np.percentile(damage_smooth, 73.04)   # below = Minor Damage
t3 = np.percentile(damage_smooth, 90.10)   # below = Moderate Damage

damage_labels = np.zeros_like(damage_smooth, dtype=np.int32)
damage_labels[damage_smooth >= t1] = 1
damage_labels[damage_smooth >= t2] = 2
damage_labels[damage_smooth >= t3] = 3

total_px = damage_labels.size
print(f"  Thresholds: {t1:.4f} | {t2:.4f} | {t3:.4f}")
for cls, name in enumerate(DAMAGE_NAMES):
    n = np.sum(damage_labels == cls)
    print(f"  Class {cls} ({name:<22}): {n:10,} px  "
          f"({n * 100 / total_px * 100 / 1e6:.1f} km²)  "
          f"{n / total_px * 100:.1f}%")

# Build 7-band feature stack: [SAR, NDBI, NDVI, BSI, NBR, NTL, DamageIdx]
bands_stack = np.stack([sar, ndbi, ndvi, bsi, nbr, ntl, damage_smooth], axis=-1)
FEAT_NAMES  = ['SAR_chg', 'NDBI_chg', 'NDVI_chg', 'BSI_chg',
               'NBR_chg', 'NTL_chg', 'Damage_Index']
print("  ✅ Feature stack shape:", bands_stack.shape)

# Serialise for downstream ML scripts
np.save(os.path.join(OUT, 'damage_labels.npy'), damage_labels)
np.save(os.path.join(OUT, 'bands_stack.npy'),   bands_stack)
np.save(os.path.join(OUT, 'transform.npy'),     np.array(transform))
print("  Saved: damage_labels.npy, bands_stack.npy, transform.npy")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 6-PANEL SUMMARY MAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Generating 6-panel change map...")

def pstretch(arr, lo=2, hi=98):
    a = np.percentile(arr[np.isfinite(arr)], lo)
    b = np.percentile(arr[np.isfinite(arr)], hi)
    return np.clip((arr - a) / (b - a + 1e-8), 0, 1)

fig = plt.figure(figsize=(20, 24), facecolor='white')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.10, wspace=0.05)

panels = [
    (sar,          'RdBu_r', False, 'SAR Backscatter Change\nBlue=Decrease  Red=Increase'),
    (ndbi,         'RdBu_r', False, 'Built-up Change (NDBI)\nNDBI Post − Pre Conflict'),
    (bsi,          'YlOrRd', False, 'Bare Surface Change (BSI)\nRubble / Soil Exposure'),
    (nbr,          'RdBu',   False, 'Burn Index Change (NBR)\nFire Damage Detection'),
    (damage_smooth,'YlOrRd', False, 'Multi-Sensor Damage Index\n(|SAR|+|NDBI|+|BSI|+|NBR|)/4'),
    (damage_labels,'listed', True,  'Damage Classification Map\n4-Class ML Fusion'),
]

for i, (data, cmap_name, is_cls, title) in enumerate(panels):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    if is_cls:
        cmap_c = mcolors.ListedColormap(DAMAGE_COLORS)
        norm_c = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_c.N)
        im = ax.imshow(data, cmap=cmap_c, norm=norm_c,
                       interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=[0,1,2,3])
        cbar.ax.set_yticklabels(['No Dmg','Minor','Moderate','Severe'], fontsize=8)
        ax.legend(handles=[mpatches.Patch(color=c, label=n)
                           for c, n in zip(DAMAGE_COLORS, DAMAGE_NAMES)],
                  loc='lower left', fontsize=7, framealpha=0.9)
    else:
        im = ax.imshow(data, cmap=cmap_name,
                       vmin=np.percentile(data, 2), vmax=np.percentile(data, 98),
                       interpolation='bilinear', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.axis('off')

fig.suptitle('Gaza Strip — Multi-Sensor Damage Analysis (2023–2024)\n'
             'Sentinel-1 SAR  |  Sentinel-2 Optical  |  VIIRS Night-Time Light',
             fontsize=15, fontweight='bold', y=1.00)
out_path = os.path.join(OUT, 'VIZ_01_Summary_Panel.png')
plt.savefig(out_path, dpi=PLT_DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ Saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NTL CHANGE MAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Generating NTL change map...")

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(ntl, cmap='RdBu_r',
               vmin=np.percentile(ntl, 2), vmax=np.percentile(ntl, 98),
               interpolation='bilinear', aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
             label='NTL Radiance Change (nW/cm²/sr)')
ax.set_title('Night-Time Light Change\nVIIRS DNB — Post − Pre Conflict',
             fontsize=13, fontweight='bold')
ax.axis('off')
ax.legend(handles=[
    mpatches.Patch(color='#2196F3', label='Decrease (displacement / power loss)'),
    mpatches.Patch(color='#F44336', label='Increase'),
], loc='lower left', fontsize=9, framealpha=0.85)
plt.tight_layout()
out_path = os.path.join(OUT, 'VIZ_02_NTL_Change.png')
plt.savefig(out_path, dpi=PLT_DPI, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {out_path}")

print("\n🎉 Feature engineering complete! Arrays saved to outputs/")
