"""
=============================================================================
 04_CNN_DAMAGE_CLASSIFIER.PY — GAZA STRIP CONFLICT ANALYSIS
 D1+TD1 Satellite Remote Sensing | Winter 2025-26
=============================================================================
 PURPOSE : Train a patch-based CNN on 32×32 pixel windows and validate
           against withheld patches. Complements the RF pixel classifier
           by learning spatial texture patterns of structural damage.

 INPUTS  (from 02_feature_engineering.py outputs):
   outputs/bands_stack.npy   — 7-band feature cube  (H × W × 7)
   outputs/damage_labels.npy — label map             (H × W)

 OUTPUTS (saved to outputs/):
   CNN_training.png           — training/validation accuracy & loss curves
   CNN_confusion_matrix.png   — per-class confusion matrix
   cnn_model.keras            — saved Keras model
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, cohen_kappa_score,
                              f1_score, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)

DAMAGE_NAMES  = ['No Damage', 'Minor Damage', 'Moderate Damage', 'Severe/Destroyed']
PATCH         = 32        # patch size (pixels)
MAX_PER_CLASS = 8000      # max patches per class (balanced sampling)
STEP          = 16        # stride for patch extraction
EPOCHS        = 60
BATCH_SIZE    = 64
PLT_DPI       = 150

print("=" * 65)
print("  CNN PATCH-BASED DAMAGE CLASSIFIER — GAZA")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD FEATURE ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading feature arrays...")
bands_stack   = np.load(os.path.join(OUT, 'bands_stack.npy'))
damage_labels = np.load(os.path.join(OUT, 'damage_labels.npy'))
h, w, n_bands = bands_stack.shape
HALF = PATCH // 2
print(f"  Stack shape: {bands_stack.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACT PATCHES (balanced, stratified)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Extracting balanced patches...")

patches_list, patch_labels = [], []
counts = Counter()

for r in range(HALF, h - HALF, STEP):
    for c in range(HALF, w - HALF, STEP):
        lbl = int(damage_labels[r, c])
        if counts[lbl] >= MAX_PER_CLASS:
            continue
        patch = bands_stack[r-HALF:r+HALF, c-HALF:c+HALF, :]
        if patch.shape == (PATCH, PATCH, n_bands):
            patches_list.append(patch)
            patch_labels.append(lbl)
            counts[lbl] += 1
    if all(counts[k] >= MAX_PER_CLASS for k in range(4)):
        break

X_cnn = np.array(patches_list, dtype=np.float32)
y_cnn = np.array(patch_labels, dtype=np.int32)
print(f"  Patches: {X_cnn.shape} | Dist: {Counter(y_cnn.tolist())}")

# Normalise each band to [0, 1] via 2–98 percentile stretch
for b in range(n_bands):
    lo = np.percentile(X_cnn[:, :, :, b], 2)
    hi = np.percentile(X_cnn[:, :, :, b], 98)
    X_cnn[:, :, :, b] = np.clip((X_cnn[:, :, :, b] - lo) / (hi - lo + 1e-8), 0, 1)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn)

cw      = compute_class_weight('balanced', classes=np.unique(y_cnn), y=y_cnn)
cw_dict = dict(enumerate(cw))
print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD & TRAIN CNN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Building and training CNN...")

def build_cnn(input_shape, n_classes=4):
    """Lightweight CNN with 3 conv blocks + global average pooling."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32,  3, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,  3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m

cnn = build_cnn((PATCH, PATCH, n_bands))
cnn.summary()

history = cnn.fit(
    X_tr, y_tr,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    class_weight=cw_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0)
    ],
    verbose=1)

_, cnn_acc  = cnn.evaluate(X_te, y_te, verbose=0)
y_pred_cnn  = np.argmax(cnn.predict(X_te, verbose=0), axis=1)
cnn_kappa   = cohen_kappa_score(y_te, y_pred_cnn)
cnn_f1      = f1_score(y_te, y_pred_cnn, average='macro')

print(f"\n  ── CNN RESULTS ──")
print(f"  Accuracy : {cnn_acc*100:.2f}%")
print(f"  Kappa    : {cnn_kappa:.4f}")
print(f"  F1 macro : {cnn_f1:.4f}")
print(classification_report(y_te, y_pred_cnn, target_names=DAMAGE_NAMES))

# Save model
cnn.save(os.path.join(OUT, 'cnn_model.keras'))
print("  CNN model saved → outputs/cnn_model.keras")

# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Saving visualisations...")

# Training curves
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(history.history['accuracy'],     label='Train', linewidth=2)
a1.plot(history.history['val_accuracy'], label='Val',   linewidth=2)
a1.set_title(f'CNN Accuracy (Test: {cnn_acc*100:.1f}%)')
a1.legend(); a1.grid(alpha=0.3); a1.set_xlabel('Epoch')
a2.plot(history.history['loss'],     label='Train', linewidth=2)
a2.plot(history.history['val_loss'], label='Val',   linewidth=2)
a2.set_title('CNN Loss')
a2.legend(); a2.grid(alpha=0.3); a2.set_xlabel('Epoch')
plt.suptitle(f'CNN Training — Acc={cnn_acc*100:.1f}%  '
             f'Kappa={cnn_kappa:.3f}  F1={cnn_f1:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'CNN_training.png'), dpi=PLT_DPI, bbox_inches='tight')
plt.close()

# Confusion Matrix
cm_cnn = confusion_matrix(y_te, y_pred_cnn)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Oranges',
            xticklabels=DAMAGE_NAMES, yticklabels=DAMAGE_NAMES, linewidths=0.5)
ax.set_title(f'CNN Confusion Matrix\nAcc={cnn_acc*100:.1f}%  Kappa={cnn_kappa:.3f}',
             fontsize=11, fontweight='bold')
ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'CNN_confusion_matrix.png'), dpi=PLT_DPI, bbox_inches='tight')
plt.close()

print("  ✅ All CNN plots saved!")
print(f"\n📊 Summary — CNN Accuracy: {cnn_acc*100:.2f}% | Kappa: {cnn_kappa:.4f} | F1: {cnn_f1:.4f}")
