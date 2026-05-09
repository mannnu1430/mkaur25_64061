"""
================================================================================
  Deep Learning for Natural Language Processing
  Sentiment Analysis: Bidirectional LSTM vs. Transformer Encoder
  Advanced Machine Learning — Final Exam / Project
================================================================================

  OVERVIEW
  --------
  This script implements, trains, and evaluates two deep learning architectures
  for binary sentiment classification on the IMDB movie review dataset:

    1. Bidirectional LSTM (BiLSTM)
         - Stacked bidirectional recurrent layers with gating mechanisms
         - SpatialDropout for embedding regularization
         - Recurrent dropout within LSTM cells

    2. Transformer Encoder
         - Multi-head self-attention (4 heads)
         - Learned positional embeddings
         - Residual connections and layer normalization
         - Global average pooling for sequence aggregation

  DATASET
  -------
  IMDB Movie Reviews (Keras built-in)
    - 50,000 reviews total: 25,000 train / 25,000 test
    - Binary labels: 1 = positive sentiment, 0 = negative sentiment
    - Perfectly balanced: 50% positive, 50% negative in each split
    - Reviews rated >= 7/10 labeled positive, <= 4/10 labeled negative

  EXPERIMENTAL SETUP
  ------------------
  Both models are trained under identical conditions:
    - Same vocabulary size, sequence length, embedding dimension
    - Same optimizer, learning rate, batch size, and callbacks
    - EarlyStopping (patience=3) on validation loss
    - ReduceLROnPlateau (patience=2, factor=0.5) on validation loss

  OUTPUTS
  -------
  - figures/fig1_training_curves.png  : accuracy/loss curves + confusion matrices
  - figures/fig2_model_comparison.png : test accuracy and ROC-AUC bar chart
  - figures/fig3_sequence_lengths.png : review length distribution histogram
  - figures/fig4_efficiency.png       : parameter count and training time comparison
  - figures/metrics.json              : all numerical results (for HTML report)

  REQUIREMENTS
  ------------
  pip install tensorflow>=2.10 numpy matplotlib seaborn scikit-learn pandas

  USAGE
  -----
  python nlp_deep_learning.py

================================================================================
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
# pad_sequences moved to keras.utils in TF 2.x; the preprocessing path is deprecated
try:
    from tensorflow.keras.utils import pad_sequences          # TF 2.6+
except ImportError:
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # fallback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ==============================================================================
# 1. HYPERPARAMETERS
# ==============================================================================

VOCAB_SIZE  = 10_000   # Keep top 10,000 most frequent words
MAX_LEN     = 256      # Truncate/pad all sequences to this length
EMBED_DIM   = 64       # Dimensionality of token embeddings
LSTM_UNITS  = 64       # LSTM hidden units per direction
NUM_HEADS   = 4        # Number of attention heads in Transformer
FF_DIM      = 128      # Feed-forward inner dimension in Transformer
BATCH_SIZE  = 128      # Samples per gradient update
EPOCHS      = 15       # Maximum training epochs (EarlyStopping fires earlier)

print("=" * 70)
print("  Deep Learning for NLP -- Sentiment Analysis")
print("  Bidirectional LSTM  vs.  Transformer Encoder")
print("=" * 70)
print(f"\n  TensorFlow  : {tf.__version__}")
print(f"  Vocab size  : {VOCAB_SIZE:,}")
print(f"  Max length  : {MAX_LEN} tokens")
print(f"  Embed dim   : {EMBED_DIM}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Max epochs  : {EPOCHS}")
print()


# ==============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ==============================================================================

print("[Step 1/5]  Loading IMDB dataset ...")

(X_train_raw, y_train), (X_test_raw, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print(f"  Training samples : {len(X_train_raw):>6,}")
print(f"  Test samples     : {len(X_test_raw):>6,}")
print(f"  Positive (train) : {y_train.sum():>6,}  ({y_train.mean()*100:.1f}%)")
print(f"  Negative (train) : {(1-y_train).sum():>6,}  ({(1-y_train).mean()*100:.1f}%)")

# Sequence length statistics (before padding)
raw_lengths = [len(seq) for seq in X_train_raw]
print(f"\n  Sequence length statistics (training set):")
print(f"    Mean   : {np.mean(raw_lengths):.1f} tokens")
print(f"    Median : {np.median(raw_lengths):.1f} tokens")
print(f"    Std    : {np.std(raw_lengths):.1f} tokens")
print(f"    Min    : {min(raw_lengths)} tokens")
print(f"    Max    : {max(raw_lengths)} tokens")
print(f"    Covered by MAX_LEN={MAX_LEN}: "
      f"{100*np.mean(np.array(raw_lengths) <= MAX_LEN):.1f}% of reviews")

# Pad sequences: post-padding and post-truncation
# Post-padding appends zeros at the end, preserving natural reading order
X_train = pad_sequences(
    X_train_raw, maxlen=MAX_LEN, padding='post', truncating='post'
)
X_test = pad_sequences(
    X_test_raw, maxlen=MAX_LEN, padding='post', truncating='post'
)

print(f"\n  Padded train shape : {X_train.shape}")
print(f"  Padded test shape  : {X_test.shape}")


# ==============================================================================
# 3. MODEL DEFINITIONS
# ==============================================================================

print("\n[Step 2/5]  Building models ...")


# ------------------------------------------------------------------------------
# 3a. Bidirectional LSTM
# ------------------------------------------------------------------------------

def build_bilstm(vocab_size, max_len, embed_dim, lstm_units):
    """
    Stacked Bidirectional LSTM for text classification.

    Architecture:
      Embedding (vocab_size x embed_dim)
        --> SpatialDropout1D(0.2)
        --> BiLSTM(lstm_units, return_sequences=True, recurrent_dropout=0.1)
        --> BiLSTM(lstm_units//2, recurrent_dropout=0.1)
        --> Dense(64, relu)
        --> Dropout(0.3)
        --> Dense(1, sigmoid)

    The first LSTM returns the full sequence so the second layer has access to
    all time steps. The second LSTM returns only the final hidden state, which
    summarises the entire review into a fixed-length vector.
    """
    inputs = keras.Input(shape=(max_len,), name='token_ids')

    # Learned word embeddings; mask_zero=True propagates padding masks
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        name='embedding'
    )(inputs)

    # SpatialDropout zeros entire embedding dimensions rather than individual
    # units, which is better for sequences where adjacent positions are correlated
    x = layers.SpatialDropout1D(0.2, name='spatial_dropout')(x)

    # First BiLSTM: returns full sequence for stacking
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.1),
        name='bilstm_1'
    )(x)

    # Second BiLSTM: returns final hidden state only
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, recurrent_dropout=0.1),
        name='bilstm_2'
    )(x)

    x = layers.Dense(64, activation='relu', name='fc')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs, outputs, name='BiLSTM')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ------------------------------------------------------------------------------
# 3b. Transformer Encoder
# ------------------------------------------------------------------------------

class TransformerBlock(layers.Layer):
    """
    A single Transformer encoder block.

    Implements the standard architecture from Vaswani et al. (2017):
      MultiHeadSelfAttention --> Add & LayerNorm --> FFN --> Add & LayerNorm

    Args:
        embed_dim : Dimensionality of token representations.
        num_heads : Number of parallel attention heads.
        ff_dim    : Inner dimensionality of the feed-forward sublayer.
        dropout   : Dropout rate applied after attention and FFN.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-attention sublayer with residual connection
        attn_out = self.attention(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        # Feed-forward sublayer with residual connection
        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.norm2(x + ffn_out)


class TokenAndPositionEmbedding(layers.Layer):
    """
    Combines learned token embeddings with learned positional embeddings.

    Unlike sinusoidal positional encoding (Vaswani et al., 2017), learned
    positional embeddings are optimised end-to-end with the rest of the model.

    Args:
        max_len    : Maximum sequence length.
        vocab_size : Vocabulary size.
        embed_dim  : Embedding dimensionality.
    """

    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, name='token_emb'
        )
        self.pos_emb = layers.Embedding(
            input_dim=max_len, output_dim=embed_dim, name='pos_emb'
        )

    def call(self, token_ids):
        seq_len   = tf.shape(token_ids)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return self.token_emb(token_ids) + self.pos_emb(positions)


def build_transformer(vocab_size, max_len, embed_dim, num_heads, ff_dim):
    """
    Compact Transformer encoder for text classification.

    Architecture:
      TokenAndPositionEmbedding
        --> TransformerBlock (multi-head self-attention + FFN)
        --> GlobalAveragePooling1D
        --> Dense(64, relu)
        --> Dropout(0.3)
        --> Dense(1, sigmoid)

    GlobalAveragePooling aggregates per-token representations into a single
    fixed-length vector by averaging across the sequence dimension. This is
    more stable than using a single [CLS] token when training from scratch.
    """
    inputs = keras.Input(shape=(max_len,), name='token_ids')

    x = TokenAndPositionEmbedding(
        max_len, vocab_size, embed_dim, name='token_pos_emb'
    )(inputs)

    x = TransformerBlock(
        embed_dim, num_heads, ff_dim, dropout=0.1, name='transformer_block'
    )(x)

    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

    x = layers.Dense(64, activation='relu', name='fc')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs, outputs, name='Transformer')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Build models
bilstm_model      = build_bilstm(VOCAB_SIZE, MAX_LEN, EMBED_DIM, LSTM_UNITS)
transformer_model = build_transformer(VOCAB_SIZE, MAX_LEN, EMBED_DIM, NUM_HEADS, FF_DIM)

print("\n  --- BiLSTM Architecture ---")
bilstm_model.summary()
print("\n  --- Transformer Architecture ---")
transformer_model.summary()


# ==============================================================================
# 4. TRAINING
# ==============================================================================

print("\n[Step 3/5]  Training models ...")

# Shared callbacks applied identically to both models
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
]

# Train BiLSTM
print("\n  Training BiLSTM ...")
t_start = time.time()
history_bilstm = bilstm_model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)
time_bilstm = time.time() - t_start
final_lr_bilstm = float(bilstm_model.optimizer.learning_rate)
print(f"  BiLSTM training complete: {time_bilstm:.1f}s "
      f"| {len(history_bilstm.history['loss'])} epochs "
      f"| final LR: {final_lr_bilstm:.2e}")

# Train Transformer
print("\n  Training Transformer ...")
t_start = time.time()
history_transformer = transformer_model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)
time_transformer = time.time() - t_start
final_lr_transformer = float(transformer_model.optimizer.learning_rate)
print(f"  Transformer training complete: {time_transformer:.1f}s "
      f"| {len(history_transformer.history['loss'])} epochs "
      f"| final LR: {final_lr_transformer:.2e}")


# ==============================================================================
# 5. EVALUATION
# ==============================================================================

print("\n[Step 4/5]  Evaluating on test set ...")


def evaluate_model(model, X_test, y_test, name):
    """
    Evaluate a trained model on the test set.

    Computes: loss, accuracy, ROC-AUC, precision, recall, F1-score,
    confusion matrix, and full classification report.

    Returns a dict with all scalar metrics plus raw arrays.
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n  {name}")
    print(f"  {'=' * 45}")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Pred Neg   Pred Pos")
    print(f"  Actual Neg  :  {tn:>7,}    {fp:>7,}")
    print(f"  Actual Pos  :  {fn:>7,}    {tp:>7,}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=['Negative', 'Positive'],
        digits=4
    ))

    return dict(
        loss=float(test_loss), accuracy=float(test_acc), auc=float(auc),
        precision=float(precision), recall=float(recall), f1=float(f1),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        cm=cm, y_pred=y_pred, y_prob=y_prob
    )


results_bilstm      = evaluate_model(bilstm_model,      X_test, y_test, "BiLSTM")
results_transformer = evaluate_model(transformer_model, X_test, y_test, "Transformer")

# Parameter counts
params_bilstm      = bilstm_model.count_params()
params_transformer = transformer_model.count_params()

# Print side-by-side comparison
comparison = pd.DataFrame({
    'Metric'      : ['Test Accuracy', 'ROC-AUC', 'Precision', 'Recall',
                     'F1-Score', 'Test Loss', 'Parameters', 'Train Time (s)', 'Epochs'],
    'BiLSTM'      : [
        f"{results_bilstm['accuracy']:.4f}",
        f"{results_bilstm['auc']:.4f}",
        f"{results_bilstm['precision']:.4f}",
        f"{results_bilstm['recall']:.4f}",
        f"{results_bilstm['f1']:.4f}",
        f"{results_bilstm['loss']:.4f}",
        f"{params_bilstm:,}",
        f"{time_bilstm:.1f}",
        f"{len(history_bilstm.history['loss'])}",
    ],
    'Transformer' : [
        f"{results_transformer['accuracy']:.4f}",
        f"{results_transformer['auc']:.4f}",
        f"{results_transformer['precision']:.4f}",
        f"{results_transformer['recall']:.4f}",
        f"{results_transformer['f1']:.4f}",
        f"{results_transformer['loss']:.4f}",
        f"{params_transformer:,}",
        f"{time_transformer:.1f}",
        f"{len(history_transformer.history['loss'])}",
    ],
})
print("\n  MODEL COMPARISON SUMMARY")
print("  " + "-" * 52)
print(comparison.to_string(index=False))


# ==============================================================================
# 6. VISUALISATIONS
# ==============================================================================

print("\n[Step 5/5]  Generating figures ...")
os.makedirs('figures', exist_ok=True)

# Grayscale-safe distinct styles for both models
STYLE = {
    'b_train' : dict(color='#1a1a1a', lw=2.2, ls='-',  marker='o', ms=4, markevery=1),
    'b_val'   : dict(color='#1a1a1a', lw=1.8, ls='--', marker='s', ms=3.5, markevery=1),
    't_train' : dict(color='#606060', lw=2.2, ls='-',  marker='^', ms=4, markevery=1),
    't_val'   : dict(color='#606060', lw=1.8, ls='--', marker='D', ms=3.5, markevery=1),
}

# ── Figure 1: Training curves + confusion matrices ────────────────────────────
fig = plt.figure(figsize=(15, 11))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
ax_acc = fig.add_subplot(gs[0, 0])
ax_los = fig.add_subplot(gs[0, 1])
ax_cm1 = fig.add_subplot(gs[1, 0])
ax_cm2 = fig.add_subplot(gs[1, 1])

e_b = range(1, len(history_bilstm.history['accuracy'])      + 1)
e_t = range(1, len(history_transformer.history['accuracy']) + 1)

ax_acc.plot(e_b, history_bilstm.history['accuracy'],
            **STYLE['b_train'], label='BiLSTM — Train')
ax_acc.plot(e_b, history_bilstm.history['val_accuracy'],
            **STYLE['b_val'],   label='BiLSTM — Val')
ax_acc.plot(e_t, history_transformer.history['accuracy'],
            **STYLE['t_train'], label='Transformer — Train')
ax_acc.plot(e_t, history_transformer.history['val_accuracy'],
            **STYLE['t_val'],   label='Transformer — Val')
ax_acc.set_title('Accuracy per Epoch', fontweight='bold', fontsize=12)
ax_acc.set_xlabel('Epoch'); ax_acc.set_ylabel('Accuracy')
ax_acc.legend(fontsize=9); ax_acc.grid(alpha=0.3)

ax_los.plot(e_b, history_bilstm.history['loss'],
            **STYLE['b_train'], label='BiLSTM — Train')
ax_los.plot(e_b, history_bilstm.history['val_loss'],
            **STYLE['b_val'],   label='BiLSTM — Val')
ax_los.plot(e_t, history_transformer.history['loss'],
            **STYLE['t_train'], label='Transformer — Train')
ax_los.plot(e_t, history_transformer.history['val_loss'],
            **STYLE['t_val'],   label='Transformer — Val')
ax_los.set_title('Loss per Epoch', fontweight='bold', fontsize=12)
ax_los.set_xlabel('Epoch'); ax_los.set_ylabel('Binary Cross-Entropy Loss')
ax_los.legend(fontsize=9); ax_los.grid(alpha=0.3)

for ax, res, name in [
    (ax_cm1, results_bilstm,      'BiLSTM'),
    (ax_cm2, results_transformer, 'Transformer'),
]:
    sns.heatmap(
        res['cm'], annot=True, fmt='d', cmap='Greys',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=ax, cbar=False,
        annot_kws={'size': 13, 'weight': 'bold'},
        linewidths=0.5, linecolor='#cccccc',
    )
    acc_v = (res['cm'][0, 0] + res['cm'][1, 1]) / res['cm'].sum()
    ax.set_title(f'{name} — Confusion Matrix\n(Accuracy: {acc_v:.4f})',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')

fig.suptitle(
    'Training History and Test Set Confusion Matrices\n'
    'BiLSTM vs. Transformer — IMDB Sentiment Analysis',
    fontsize=13, fontweight='bold', y=1.01
)
plt.savefig('figures/fig1_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/fig1_training_curves.png")

# ── Figure 2: Performance bar chart ──────────────────────────────────────────
metric_names = ['Test Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']
b_vals = [results_bilstm['accuracy'], results_bilstm['auc'],
          results_bilstm['precision'], results_bilstm['recall'],
          results_bilstm['f1']]
t_vals = [results_transformer['accuracy'], results_transformer['auc'],
          results_transformer['precision'], results_transformer['recall'],
          results_transformer['f1']]

x = np.arange(len(metric_names))
w = 0.34
fig, ax = plt.subplots(figsize=(11, 5.5))
bars1 = ax.bar(x - w/2, b_vals, w, label='BiLSTM',
               color='#333333', edgecolor='white', lw=0.8)
bars2 = ax.bar(x + w/2, t_vals, w, label='Transformer',
               color='#888888', edgecolor='white', lw=0.8)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f'{bar.get_height():.4f}',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_ylim(0.82, 1.04)
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Performance Metrics Comparison — 25,000 Sample Test Set',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('figures/fig2_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/fig2_model_comparison.png")

# ── Figure 3: Sequence length distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(raw_lengths, bins=70, color='#555555', edgecolor='white', alpha=0.85)
ax.axvline(MAX_LEN, color='#111111', ls='--', lw=2.2,
           label=f'Truncation threshold ({MAX_LEN} tokens)')
ax.axvline(int(np.mean(raw_lengths)), color='#888888', ls='-', lw=2.0,
           label=f'Mean ({int(np.mean(raw_lengths))} tokens)')
ax.set_xlabel('Review Length (word tokens)', fontsize=12)
ax.set_ylabel('Number of Reviews', fontsize=12)
ax.set_title('IMDB Training Set — Review Length Distribution',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('figures/fig3_sequence_lengths.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/fig3_sequence_lengths.png")

# ── Figure 4: Efficiency ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, vals, ylabel, title, fmt in [
    (axes[0],
     [params_bilstm / 1e6, params_transformer / 1e6],
     'Parameters (Millions)', 'Trainable Parameters', '{:.2f}M'),
    (axes[1],
     [time_bilstm, time_transformer],
     'Time (seconds)', 'Total Training Time (EarlyStopping)', '{:.0f}s'),
]:
    bars = ax.bar(['BiLSTM', 'Transformer'], vals,
                  color=['#333333', '#888888'],
                  edgecolor='white', lw=1.2, width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                fmt.format(v),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Computational Efficiency Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig4_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/fig4_efficiency.png")

# ── Save metrics JSON ─────────────────────────────────────────────────────────
metrics_out = {
    "bilstm": {
        "accuracy"   : round(results_bilstm['accuracy'], 4),
        "auc"        : round(results_bilstm['auc'],       4),
        "precision"  : round(results_bilstm['precision'], 4),
        "recall"     : round(results_bilstm['recall'],    4),
        "f1"         : round(results_bilstm['f1'],        4),
        "loss"       : round(results_bilstm['loss'],      4),
        "tn"         : results_bilstm['tn'],
        "fp"         : results_bilstm['fp'],
        "fn"         : results_bilstm['fn'],
        "tp"         : results_bilstm['tp'],
        "parameters" : int(params_bilstm),
        "train_time" : round(time_bilstm, 1),
        "epochs"     : len(history_bilstm.history['loss']),
        "train_acc"  : [round(v, 4) for v in history_bilstm.history['accuracy']],
        "val_acc"    : [round(v, 4) for v in history_bilstm.history['val_accuracy']],
        "train_loss" : [round(v, 4) for v in history_bilstm.history['loss']],
        "val_loss"   : [round(v, 4) for v in history_bilstm.history['val_loss']],
        "cm"         : results_bilstm['cm'].tolist(),
    },
    "transformer": {
        "accuracy"   : round(results_transformer['accuracy'], 4),
        "auc"        : round(results_transformer['auc'],       4),
        "precision"  : round(results_transformer['precision'], 4),
        "recall"     : round(results_transformer['recall'],    4),
        "f1"         : round(results_transformer['f1'],        4),
        "loss"       : round(results_transformer['loss'],      4),
        "tn"         : results_transformer['tn'],
        "fp"         : results_transformer['fp'],
        "fn"         : results_transformer['fn'],
        "tp"         : results_transformer['tp'],
        "parameters" : int(params_transformer),
        "train_time" : round(time_transformer, 1),
        "epochs"     : len(history_transformer.history['loss']),
        "train_acc"  : [round(v, 4) for v in history_transformer.history['accuracy']],
        "val_acc"    : [round(v, 4) for v in history_transformer.history['val_accuracy']],
        "train_loss" : [round(v, 4) for v in history_transformer.history['loss']],
        "val_loss"   : [round(v, 4) for v in history_transformer.history['val_loss']],
        "cm"         : results_transformer['cm'].tolist(),
    },
}
with open('figures/metrics.json', 'w') as f:
    json.dump(metrics_out, f, indent=2)
print("  Saved: figures/metrics.json")

print("\n" + "=" * 70)
print("  All steps complete. Outputs saved to ./figures/")
print("=" * 70)
