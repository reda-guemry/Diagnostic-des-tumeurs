# Notebook pipeline

Run the notebooks in order. They work from the project root or from the
`notebooks/` directory. Every code cell is preceded by a small summary that
explains what the step does and how to read its output.

1. **`01_data_overview.ipynb`** — class discovery, file checks (readable images,
   dimensions, channels), class distribution, sample images and a stratified
   70/15/15 train/validation/test split.
   - Outputs: `data/processed/splits.json`, `data/processed/image_metadata.csv`.

2. **`02_preprocessing_augmentation.ipynb`** — image loading, blur / flip /
   Gaussian-noise augmentation, class balancing by oversampling and a `tf.data`
   pipeline with on-the-fly augmentation.
   - Output: `data/processed/train_balanced.json`.
   - Augmentation is applied in memory only; no augmented image is written.

3. **`03_feature_extraction.ipynb`** — builds a **single** model made of a
   frozen InceptionV3 backbone plus a small classifier head, trains it on the
   balanced training set (only the head learns), evaluates it on the test split
   and saves the model.
   - Output: `models/inceptionv3_transfer.keras`.
   - The backbone and the head are never split into separate models and no
     `.npy` feature arrays are produced.

4. **`04_fine_tuning.ipynb`** — loads the saved model, unfreezes the last 20
   InceptionV3 layers (BatchNorm stays frozen), fine-tunes with a small learning
   rate on the original training split with class weights, evaluates on the test
   split and saves the final model.
   - Output: `models/inceptionv3_finetuned.keras`.

## Notes

- Notebook 03 uses InceptionV3 from Keras Applications instead of the old
  InceptionV1/TF-Hub model because `tensorflow-hub` 0.16.1 provides a Keras 2
  layer, which is incompatible with Keras 3.
- Notebook 03 must run before notebook 04: notebook 04 starts from
  `models/inceptionv3_transfer.keras`.
- Training in notebook 03 runs the frozen backbone for every batch, so on CPU
  expect a few minutes per epoch. Early stopping usually ends training before
  the maximum number of epochs.
- Run `uv sync` once before executing the notebooks.
