# ViT-like Semantic Segmentation on ImageNet-S

This repository is the **baseline version** of Project 02. It provides a complete and reproducible ImageNet-S semantic segmentation pipeline, including model implementation, training, evaluation, visualization, and sample inference. Later improved versions can compare their results against this baseline.

## 1. Project Objective

This project is the second sub-project of the Computer Vision final project: **training a ViT-like model for object segmentation using an ImageNet-based dataset**.

The goal is to build a complete semantic segmentation baseline that can:

- train on ImageNet-style images with pixel-level masks,
- predict object regions at the pixel level,
- evaluate segmentation quality with quantitative metrics,
- generate visual results for analysis,
- provide a runnable sample inference script and notebook demo.

Because the original ImageNet classification dataset does not provide dense semantic segmentation masks for ordinary classification images, this project uses **ImageNet-S50**, an ImageNet-derived segmentation dataset with pixel-level object masks.

## 2. Solution Approach

### 2.1 Dataset Choice

The project uses **ImageNet-S50**. ImageNet-S keeps the ImageNet-style object categories and adds segmentation masks, making it suitable for supervised semantic segmentation.

Local dataset path:

```text
data/imagenet-s/ImageNetS50
```

Expected structure:

```text
data/imagenet-s/ImageNetS50/
|-- train-semi/
|-- train-semi-segmentation/
|-- validation/
`-- validation-segmentation/
```

The full dataset is not included in this repository. Only lightweight testing examples are provided under:

```text
data/test_examples/
```

Dataset links and notes are provided in:

```text
data/dataset_info.txt
```

### 2.2 Class Setting

The main experiment uses 10 foreground classes from ImageNet-S50 plus one background class. Therefore, the model predicts 11 labels in total.

Selected classes:

| Label | WordNet ID | Description |
|---:|---|---|
| 0 | background | background |
| 1 | n01443537 | goldfish |
| 2 | n01491361 | tiger shark |
| 3 | n01531178 | goldfinch |
| 4 | n01644373 | lizard |
| 5 | n02104029 | dog |
| 6 | n02119022 | red fox |
| 7 | n02123597 | Siamese cat |
| 8 | n02690373 | airliner |
| 9 | n02342885 | hamster |
| 10 | n02504458 | African elephant |

The main configuration file is:

```text
configs/imagenet_s_animals_10cls_trainval_holdout_ade.yaml
```

### 2.3 Train/Validation Split

ImageNet-S50 contains a limited number of annotated samples per class. To improve baseline training stability, this project uses the following split strategy:

- All original `train-semi` images are used for training.
- Most images from the original `validation` split are also added to the training set.
- One validation image per class is held out for final validation/testing.
- Held-out validation images are excluded from training, so there is no overlap between training and final validation samples.

For the 10-class configuration, the current split is:

```text
training samples: 234
validation samples: 10
```

This strategy is used because the project is a small-data baseline. It increases the amount of training data while still preserving independent held-out samples for evaluation and visualization.

### 2.4 Preprocessing

The preprocessing pipeline is implemented in `src/dataset.py` and `src/transforms.py`.

Main steps:

1. Load RGB images and corresponding segmentation masks.
2. Resize images to `256 x 256`.
3. Resize masks with nearest-neighbor interpolation to preserve discrete labels.
4. Convert class-specific foreground masks into a unified multiclass label map.
5. Normalize images with ImageNet mean and standard deviation.
6. Convert labels to `torch.long` for semantic segmentation loss computation.

Nearest-neighbor interpolation is important for masks because bilinear interpolation would create invalid intermediate class IDs.

### 2.5 Model Architecture

The model is implemented in:

```text
src/own_segformer.py
```

After reviewing ViT-based segmentation methods, this project adopts a hierarchical Transformer encoder and lightweight multi-scale decoder design. The model is implemented inside this repository and adapted to the ImageNet-S 10-class segmentation setting.

Overall model flow:

```text
Input Image
  -> Overlap Patch Embedding
  -> Four-stage Hierarchical Transformer Encoder
  -> Multi-scale MLP Decoder
  -> 1x1 Pixel Classification Head
  -> Segmentation Mask
```

Input images are resized to:

```text
3 x 256 x 256
```

The encoder produces four feature levels:

| Stage | Approx. feature size | Main role |
|---|---|---|
| Stage 1 | 64 x 64 | local edges, colors, and low-level texture |
| Stage 2 | 32 x 32 | larger local parts and mid-level patterns |
| Stage 3 | 16 x 16 | high-level object structure |
| Stage 4 | 8 x 8 | global context and semantic representation |

#### Overlap Patch Embedding

Instead of splitting the image into non-overlapping patches, the model uses convolutional patch embedding with overlap. This preserves local continuity between neighboring patches, which is useful for object boundaries in segmentation.

Code module:

```text
OwnSegformerOverlapPatchEmbeddings
```

#### Efficient Self-Attention

Self-attention helps each token model long-range relationships with other tokens. To reduce computation, the model applies spatial reduction to key and value features before attention computation.

Code module:

```text
OwnSegformerEfficientSelfAttention
```

#### Mix-FFN

The feed-forward network includes depthwise convolution:

```text
Linear -> Depthwise Conv -> GELU -> Linear
```

This adds local spatial modeling to the Transformer block while keeping the parameter cost low.

Code modules:

```text
OwnSegformerMixFFN
OwnSegformerDWConv
```

#### Multi-scale Decoder

The decoder receives four feature maps from the encoder. It:

1. projects each feature map to a shared hidden dimension,
2. upsamples all features to the same spatial size,
3. concatenates them along the channel dimension,
4. fuses them with a 1x1 convolution,
5. predicts pixel-level class logits with a final 1x1 classifier.

Code modules:

```text
OwnSegformerDecodeHead
OwnSegformerLinearProjection
```

The final classifier outputs 11 channels: background plus 10 foreground classes.

### 2.6 ADE20K Pretrained Initialization

The model uses an ADE20K-pretrained SegFormer-B0 checkpoint as initialization:

```text
models/segformer-b0-ade
```

Matching encoder and decoder tensors are loaded. The original ADE20K classifier head has 150 output classes, while this project uses 11 classes. Therefore, the final classifier weight and bias are reinitialized for the ImageNet-S task.

During loading, the log is expected to show that most tensors are loaded and two classifier tensors are skipped:

```text
loaded 206 pretrained tensors; skipped 2 tensors
```

This initialization improves training stability under the small-data setting.

### 2.7 Loss Function

The training loss combines Weighted Cross Entropy and Dice Loss:

```text
Loss = 1.0 * Weighted Cross Entropy + 0.5 * Dice Loss
```

Weighted Cross Entropy is used because background pixels usually dominate semantic segmentation masks. The project assigns a lower weight to background and a higher weight to foreground:

```yaml
background_weight: 0.7
foreground_weight: 1.3
```

Dice Loss measures region overlap and helps the model learn more complete foreground masks.

### 2.8 Baseline Position

This submission is designed as a **baseline version**. It provides a complete training and evaluation pipeline with a reasonable ViT-like segmentation model. Later improvements can compare against this baseline by changing data augmentation, decoder design, input resolution, loss function, or training data scale.

## 3. Instructions

### 3.1 Environment Setup

Create a Python environment:

```bash
conda create -n imagenet_seg python=3.10 -y
conda activate imagenet_seg
```

Install CPU PyTorch:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Install project dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows, if `conda` or `python` is not recognized, use the full path to the Python executable in your own environment and run the same `-m pip` or `-m src...` commands.

### 3.2 Dataset Preparation

If the ImageNet-S archive is still compressed, extract it with:

```bash
tar -xzf <path-to-imagenet-s.tar.gz> -C data
```

After extraction, the dataset should be located at:

```text
data/imagenet-s/ImageNetS50
```

### 3.3 Training

Train the 10-class baseline:

```bash
python -m src.train --config configs/imagenet_s_animals_10cls_trainval_holdout_ade.yaml
```

Training outputs are saved to:

```text
outputs/imagenet_s_animals_10cls_trainval_holdout_ade/
```

### 3.4 Evaluation

Evaluate the trained model:

```bash
python -m src.evaluate --model_dir outputs/imagenet_s_animals_10cls_trainval_holdout_ade/best_model --config configs/imagenet_s_animals_10cls_trainval_holdout_ade.yaml --split test
```

Evaluation metrics are saved under:

```text
outputs/imagenet_s_animals_10cls_trainval_holdout_ade/eval/
```

### 3.5 Sample Inference

The required sample inference entry is:

```text
src/main.py
```

Run inference on the provided testing examples:

```bash
python -m src.main --model_dir outputs/imagenet_s_animals_10cls_trainval_holdout_ade/best_model --input data/test_examples --output results/demo_inference_10cls
```

Each output folder contains:

```text
pred_mask.png
pred_color.png
overlay.png
pred_labels.txt
```

The `overlay.png` file is the most useful qualitative visualization because it overlays the predicted mask on the original image.

### 3.6 Demo Notebook

A notebook demo is provided in:

```text
demos/inference_demo.ipynb
```

It demonstrates the inference workflow and shows how the model output can be visualized.

### 3.7 Export Report Assets

After training and evaluation, export report-friendly files:

```bash
python -m src.export_submission --output_dir outputs/imagenet_s_animals_10cls_trainval_holdout_ade --results_dir results
```

## 4. Results & Analysis

### 4.1 Quantitative Results

The 10-class baseline was trained for 30 epochs on CPU. The best validation result was obtained at epoch 28.

Main test metrics:

| Metric | Value |
|---|---:|
| Pixel Accuracy | 0.9435 |
| Mean Pixel Accuracy | 0.8877 |
| mIoU | 0.7898 |
| Mean Foreground IoU | 0.7769 |
| Foreground IoU | 0.4338 |
| Mean Dice | 0.8728 |
| Mean Foreground Dice | 0.8643 |
| Foreground Dice | 0.6051 |
| Background IoU | 0.9191 |
| Background Dice | 0.9578 |
| Test Loss | 0.1674 |
| Inference FPS | 54.89 |

Training summary:

| Item | Value |
|---|---:|
| Training samples | 234 |
| Validation/Test samples | 10 |
| Number of labels | 11 |
| Total parameters | 3,716,971 |
| Best epoch | 28 |
| Best validation mIoU | 0.7898 |
| Total training time | 580.32 s |
| Average epoch time | 19.34 s |

Detailed metrics are saved in:

```text
outputs/imagenet_s_animals_10cls_trainval_holdout_ade/summary.json
outputs/imagenet_s_animals_10cls_trainval_holdout_ade/eval/test_metrics.json
outputs/imagenet_s_animals_10cls_trainval_holdout_ade/eval/test_metrics.csv
results/tables/
```

### 4.2 Output Files

The main experiment produces:

```text
outputs/imagenet_s_animals_10cls_trainval_holdout_ade/
|-- best_model/
|-- checkpoints/
|-- eval/
|   |-- test_metrics.json
|   `-- test_metrics.csv
|-- figures/
|   |-- curves/
|   |-- dataset_samples/
|   |-- predictions/
|   |-- failure_cases/
|   `-- confusion_matrix.png
|-- logs/
|   |-- train_log.csv
|   `-- train_log.json
`-- summary.json
```

Important files for the report:

```text
summary.json
eval/test_metrics.json
logs/train_log.csv
figures/curves/
figures/predictions/
figures/failure_cases/
figures/confusion_matrix.png
```

Exported report-friendly result folders:

```text
results/figures/imagenet_s_animals_10cls_trainval_holdout_ade/
results/tables/
results/demo_inference_10cls/
```

### 4.3 Metric Definitions

The project reports:

- Pixel Accuracy: percentage of correctly classified pixels.
- mIoU: mean intersection-over-union over classes.
- Foreground mIoU: mIoU over foreground classes only.
- Dice: region overlap between prediction and ground truth.
- Foreground Dice: Dice score over foreground classes only.
- Confusion Matrix: class-level confusion analysis.

IoU is computed as:

```text
intersection / union
```

Dice is computed from the overlap between predicted and ground-truth regions. Both metrics are higher when the predicted object mask matches the ground truth more accurately.

### 4.4 Progressive Visualizations

Training automatically saves progressive visualizations:

- loss curve,
- mIoU curve,
- Dice curve,
- pixel accuracy curve,
- dataset sample grid,
- prediction grid,
- failure case grid,
- confusion matrix.

The main visualization files are:

```text
results/figures/imagenet_s_animals_10cls_trainval_holdout_ade/curves/loss_curve.png
results/figures/imagenet_s_animals_10cls_trainval_holdout_ade/curves/miou_curve.png
results/figures/imagenet_s_animals_10cls_trainval_holdout_ade/curves/dice_curve.png
results/figures/imagenet_s_animals_10cls_trainval_holdout_ade/curves/pixel_acc_curve.png
results/figures/imagenet_s_animals_10cls_trainval_holdout_ade/dataset_samples/sample_grid.png
results/figures/imagenet_s_animals_10cls_trainval_holdout_ade/predictions/pred_grid.png
results/figures/imagenet_s_animals_10cls_trainval_holdout_ade/failure_cases/failure_grid.png
results/figures/imagenet_s_animals_10cls_trainval_holdout_ade/confusion_matrix.png
```

These figures are used to analyze whether the model is learning, whether validation metrics improve, and which classes are more likely to be confused.

### 4.5 Demo Inference Results

The sample inference outputs for the three testing examples are saved in:

```text
results/demo_inference_10cls/
```

Each example folder contains:

```text
pred_mask.png
pred_color.png
overlay.png
pred_labels.txt
```

The `overlay.png` image is used as the expected visual output. It overlays the predicted segmentation mask on the original image and is the most direct qualitative result for the demo.

### 4.6 Analysis

The baseline is trained under a small-data setting. The model can learn coarse object regions, but the following errors may appear:

- rough object boundaries,
- missing small foreground parts,
- confusion between visually similar categories,
- unstable validation metrics because only one held-out image per class is used,
- weaker performance on cluttered backgrounds.

These limitations are expected for a CPU-trained baseline with limited ImageNet-S50 samples.

Overall, the baseline achieves a test mIoU of 0.7898 and a mean foreground Dice score of 0.8643 on the held-out validation/test subset. This indicates that the model can segment the selected ImageNet-S foreground categories reasonably well under the current small-data setup. However, since only one held-out image per class is used, the numerical metrics should be interpreted as baseline reference values rather than as large-scale benchmark results.

## 5. Repository Structure

```text
project_02_tangzhichao/
|-- README.md
|-- requirements.txt
|-- references.md
|-- configs/
|   `-- imagenet_s_animals_10cls_trainval_holdout_ade.yaml
|-- data/
|   |-- dataset_info.txt
|   `-- test_examples/
|-- demos/
|   `-- inference_demo.ipynb
|-- models/
|   `-- segformer-b0-ade/
|-- results/
|   |-- figures/
|   `-- tables/
|-- scripts/
`-- src/
    |-- main.py
    |-- own_segformer.py
    |-- model.py
    |-- dataset.py
    |-- train.py
    |-- evaluate.py
    |-- infer.py
    |-- visualize.py
    |-- metrics.py
    |-- transforms.py
    |-- config.py
    `-- utils.py
```

## 6. Conclusion

This project builds a complete ImageNet-S semantic segmentation baseline using a ViT-like architecture. The model uses a hierarchical Transformer encoder, multi-scale decoder, ADE20K pretrained initialization, weighted cross entropy, and Dice loss. The implementation supports training, evaluation, visualization, sample inference, and notebook demonstration.

The main insight is that ImageNet-S50 segmentation is challenging under limited data. Adding most validation images into the training set improves the amount of supervision, while one held-out image per class is preserved for final validation. ADE20K initialization helps stabilize training, and the combined loss encourages both pixel-level correctness and foreground region overlap.

Future improvements may include:

- training on more ImageNet-S classes or a larger ImageNet-S split,
- using GPU training for more epochs and higher resolution,
- adding stronger data augmentation,
- improving the decoder for finer boundaries,
- comparing this baseline with enhanced models as a follow-up experiment.

## 7. References and Attribution

This project uses PyTorch, Hugging Face model weights, ImageNet, ImageNet-S, and SegFormer-related ideas. Detailed citations and attribution are listed in:

```text
references.md
```
