"""
=============================================================================
 DATA.PY — GAZA STRIP — GEE Data Acquisition Helper
 D1+TD1 Satellite Remote Sensing | Winter 2025-26
=============================================================================
 Quick standalone script to load and visualise all three data sources
 (Sentinel-2, Sentinel-1, VIIRS) for the Gaza Strip AOI in a Jupyter-style
 interactive map. For full export pipeline, use 01_data_acquisition.py.
=============================================================================
"""

import ee
import geemap

# ─────────────────────────────────────────────────────────────────────────────
# 0. INITIALIZE GEE
# ─────────────────────────────────────────────────────────────────────────────
try:
    ee.Initialize(project='satcom-488516')   # ← replace with your GEE project ID
except Exception:
    ee.Authenticate()
    ee.Initialize(project='satcom-488516')

# ─────────────────────────────────────────────────────────────────────────────
# 1. AREA OF INTEREST — Gaza Strip
#    Bounding box: 34.22–34.56°E, 31.21–31.61°N  (≈ 365 km²)
# ─────────────────────────────────────────────────────────────────────────────
roi = ee.Geometry.Rectangle([34.22, 31.21, 34.56, 31.61])

# ─────────────────────────────────────────────────────────────────────────────
# 2. SENTINEL-2 — OPTICAL (cloud-masked median composite)
# ─────────────────────────────────────────────────────────────────────────────
def mask_s2_clouds(image):
    scl  = image.select('SCL')
    mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
               .And(scl.neq(10)).And(scl.neq(11)))
    return image.updateMask(mask).divide(10000).copyProperties(
        image, ['system:time_start'])

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate('2023-01-01', '2023-10-01')   # pre-conflict baseline
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
        .map(mask_s2_clouds)
        .median()
        .clip(roi))

# ─────────────────────────────────────────────────────────────────────────────
# 3. SENTINEL-1 — SAR / MICROWAVE (IW mode, VV+VH)
# ─────────────────────────────────────────────────────────────────────────────
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(roi)
        .filterDate('2023-01-01', '2023-10-01')
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .select(['VV', 'VH'])
        .median()
        .clip(roi))

# ─────────────────────────────────────────────────────────────────────────────
# 4. VIIRS DNB — NIGHT-TIME LIGHT (monthly mean)
# ─────────────────────────────────────────────────────────────────────────────
ntl = (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
          .filterBounds(roi)
          .filterDate('2022-01-01', '2023-10-01')
          .select('avg_rad')
          .mean()
          .clip(roi)
          .rename('NTL_baseline'))

# ─────────────────────────────────────────────────────────────────────────────
# 5. QUICK CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("S2 bands :", s2.bandNames().getInfo())
print("S1 bands :", s1.bandNames().getInfo())
print("NTL bands:", ntl.bandNames().getInfo())

# ─────────────────────────────────────────────────────────────────────────────
# 6. INTERACTIVE MAP (run in Jupyter/VS Code notebook)
# ─────────────────────────────────────────────────────────────────────────────
Map = geemap.Map(center=[31.41, 34.39], zoom=11)
Map.add_basemap('SATELLITE')

Map.addLayer(s2, {'bands': ['B4','B3','B2'], 'min': 0, 'max': 0.3},
             'S2 Natural Color (pre-conflict)')
Map.addLayer(s2, {'bands': ['B8','B4','B3'], 'min': 0, 'max': 0.4},
             'S2 False Color NIR', shown=False)
Map.addLayer(s1.select('VV'), {'min': -20, 'max': 0,
             'palette': ['black','white']}, 'S1 VV Backscatter')
Map.addLayer(s1.select('VH'), {'min': -25, 'max': -5,
             'palette': ['black','white']}, 'S1 VH Backscatter', shown=False)
Map.addLayer(ntl, {'min': 0, 'max': 15,
             'palette': ['black','yellow','white']}, 'NTL Baseline', shown=False)

Map   # Display in Jupyter

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORT TO GOOGLE DRIVE  (uncomment to run)
# ─────────────────────────────────────────────────────────────────────────────
def export(image, name, scale=10):
    ee.batch.Export.image.toDrive(
        image=image.toFloat(), description=name,
        folder='Sat_Com', region=roi,
        scale=scale, maxPixels=1e10, crs='EPSG:4326'
    ).start()
    print(f"Export started → {name}")

# export(s2.select(['B4','B3','B2','B8','B11','B12']).multiply(10000).toInt16(),
#        'S2_Gaza_PreConflict', scale=10)
# export(s1, 'S1_Gaza_PreConflict', scale=10)
# export(ntl, 'NTL_Gaza_Baseline', scale=500)