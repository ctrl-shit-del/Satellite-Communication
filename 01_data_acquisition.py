"""
=============================================================================
 01_DATA_ACQUISITION.PY — GAZA STRIP CONFLICT ANALYSIS
 D1+TD1 Satellite Remote Sensing | Winter 2025-26
=============================================================================
 PURPOSE : Pull Sentinel-1 SAR, Sentinel-2 Optical, and VIIRS Night-Time
           Light data from Google Earth Engine for the Gaza Strip (2020–2025)
           and export all change rasters to Google Drive / local disk.

 DATASETS:
   • Sentinel-1 IW GRD  — C-band SAR, VV+VH, 10 m, 6-day repeat
   • Sentinel-2 MSI L2A — Multispectral optical, 10–20 m, 5-day repeat
   • VIIRS DNB Monthly  — Night-time light, 500 m, monthly composites

 OUTPUTS (exported to Google Drive → Sat_Com/):
   SAR_change_Gaza.tif   — VV backscatter change (dB)
   NDVI_change.tif       — Vegetation loss / recovery
   NDBI_change.tif       — Built-up area change
   BSI_change.tif        — Bare soil / rubble exposure
   NBR_change.tif        — Burn / fire damage
   NTL_change.tif        — Night-light radiance change (VIIRS DNB)
=============================================================================
"""

import ee

# ─────────────────────────────────────────────────────────────────────────────
# 0. INITIALIZE GEE
# ─────────────────────────────────────────────────────────────────────────────
try:
    ee.Initialize(project='satcom-488516')   # ← replace with your GEE project ID
except Exception:
    ee.Authenticate()
    ee.Initialize(project='satcom-488516')

print("✅ GEE initialised")

# ─────────────────────────────────────────────────────────────────────────────
# 1. AREA OF INTEREST — Gaza Strip
#    Bounding box: 34.22–34.56°E, 31.21–31.61°N  (≈ 365 km²)
# ─────────────────────────────────────────────────────────────────────────────
aoi = ee.Geometry.Rectangle([34.22, 31.21, 34.56, 31.61])

# ─────────────────────────────────────────────────────────────────────────────
# 2. SENTINEL-1 SAR CHANGE DETECTION
#    Pre-conflict  : Jan 2023 – Sep 2023
#    Post-conflict : Oct 2023 – Jan 2025
# ─────────────────────────────────────────────────────────────────────────────
s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(aoi)
            .filterDate('2023-01-01', '2025-01-01')
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .select(['VV', 'VH']))

pre_s1  = s1_col.filterDate('2023-01-01', '2023-10-01').mean()
post_s1 = s1_col.filterDate('2023-10-01', '2025-01-01').mean()

# SAR backscatter change in dB (post − pre)
sar_change = post_s1.subtract(pre_s1).select(['VV'], ['VV_change'])
print("  SAR change image computed")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SENTINEL-2 SPECTRAL INDICES
#    Indices computed per epoch, then differenced (post − pre)
# ─────────────────────────────────────────────────────────────────────────────
def mask_s2_clouds(img):
    """Cloud mask using Sentinel-2 Scene Classification Layer (SCL)."""
    scl = img.select('SCL')
    mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
               .And(scl.neq(10)).And(scl.neq(11)))
    return img.updateMask(mask).divide(10000).copyProperties(
        img, ['system:time_start'])

def add_indices(img):
    """Compute NDVI, NDBI, BSI, NBR for a Sentinel-2 image."""
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi = img.normalizedDifference(['B11', 'B8']).rename('NDBI')
    bsi  = img.normalizedDifference(['B11', 'B2']).rename('BSI')
    nbr  = img.normalizedDifference(['B8', 'B11']).rename('NBR')
    return img.addBands([ndvi, ndbi, bsi, nbr])

s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
            .map(mask_s2_clouds)
            .map(add_indices))

pre_s2  = s2_col.filterDate('2023-01-01', '2023-10-01').mean()
post_s2 = s2_col.filterDate('2023-10-01', '2025-01-01').mean()

index_change = post_s2.subtract(pre_s2).select(['NDVI', 'NDBI', 'BSI', 'NBR'])
print("  Spectral indices computed")

# ─────────────────────────────────────────────────────────────────────────────
# 4. VIIRS NIGHT-TIME LIGHT (NLPDI)
#    Baseline  : Jan 2022 – Sep 2023 (pre-conflict)
#    Conflict  : Oct 2023 – present
# ─────────────────────────────────────────────────────────────────────────────
viirs = (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
           .filterBounds(aoi)
           .select('avg_rad'))

ntl_baseline = viirs.filterDate('2022-01-01', '2023-10-01').mean()
ntl_conflict = viirs.filterDate('2023-10-01', '2025-01-01').mean()
ntl_change   = ntl_conflict.subtract(ntl_baseline).rename('NTL_change')

# Night-Light Population Displacement Index (NLPDI)
# NLPDI(t) = (NTL_baseline − NTL_t) / NTL_baseline × 100
nlpdi = (ntl_baseline.subtract(ntl_conflict)
                     .divide(ntl_baseline)
                     .multiply(100)
                     .rename('NLPDI'))
print("  VIIRS NTL change computed")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPORT ALL RASTERS TO GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_FOLDER = 'Sat_Com'
EXPORT_SCALE = 10   # 10 m for SAR / optical; overridden to 500 m for NTL

def export_to_drive(image, description, scale=10):
    task = ee.batch.Export.image.toDrive(
        image=image.toFloat(),
        description=description,
        folder=DRIVE_FOLDER,
        region=aoi,
        scale=scale,
        maxPixels=1e10,
        crs='EPSG:4326'
    )
    task.start()
    print(f"  Export started → {DRIVE_FOLDER}/{description}.tif")

print("\n[Exports] Starting GEE export tasks...")
export_to_drive(sar_change,                         'SAR_change_Gaza',  scale=10)
export_to_drive(index_change.select('NDVI'),        'NDVI_change',      scale=10)
export_to_drive(index_change.select('NDBI'),        'NDBI_change',      scale=10)
export_to_drive(index_change.select('BSI'),         'BSI_change',       scale=10)
export_to_drive(index_change.select('NBR'),         'NBR_change',       scale=10)
export_to_drive(ntl_change,                         'NTL_change',       scale=500)

print("\n✅ All 6 export tasks submitted to GEE.")
print("   Monitor progress at: https://code.earthengine.google.com/tasks")
print("   Files will appear in Google Drive → Sat_Com/ when complete.")
