# Model card: DepthCastNet

## Model overview

`DepthCastNet` is a convolutional-recurrent classifier that predicts the
direction of the next mid-price move from a window of limit-order-book
snapshots. It is a clean-room reimplementation of the architecture
popularised by the DeepLOB research line, packaged as a typed, tested
library.

- **Task:** three-class classification (up / stationary / down).
- **Input:** `(1, window, 40)` order-book window (default window 100).
- **Output:** logits over three movement classes.
- **Parameters:** about 144k at the default configuration.
- **Framework:** PyTorch 2.2+.

## Intended use

- Research and education on market-microstructure deep learning.
- A reproducible baseline for limit-order-book direction forecasting.
- A teaching example of packaging a research model as a tested tool.

### Out of scope

- **Live trading or investment decisions.** DepthCast is a research
  artifact, not financial advice, and carries no performance guarantee.
- Asset classes or book depths whose feature layout differs from FI-2010
  without adapting `num_features` and the data reader.

## Training data

The reference target dataset is **FI-2010** (Ntakaris et al.), a public
benchmark of normalised limit-order-book snapshots with direction labels
at five horizons. DepthCast also ships a deterministic synthetic
generator with the same layout for tests and demonstrations. The
synthetic data is **not** representative of real markets and should not
be used to draw conclusions about model performance.

## Evaluation

`evaluate_classifier` reports accuracy, macro F1, per-class F1, and a
confusion matrix via scikit-learn. On FI-2010 the literature reports
strong performance at the longer horizons; reproducing those numbers
requires the full dataset and a GPU and is outside the scope of the
shipped tests, which validate correctness on synthetic data.

## Factors and limitations

- **Horizon sensitivity.** Performance depends heavily on the prediction
  horizon (`horizon_index`); shorter horizons are noisier.
- **Class imbalance.** Real order-book data is dominated by the
  stationary class; consider class weighting or resampling for
  production-grade training.
- **Distribution shift.** A model trained on one instrument or regime may
  not transfer to another.
- **Determinism.** Results are seeded and reproducible on CPU; exact GPU
  reproducibility depends on cuDNN settings.

## Ethical considerations

Automated trading models can amplify market instability and encode biases
present in historical data. Use DepthCast responsibly, validate
thoroughly on out-of-sample data, and do not deploy it as a sole basis
for financial decisions.

## Citation

If you use DepthCast in academic work, please cite the original DeepLOB
paper for the architecture and this repository for the implementation.
