# CHILI — Disentangled CLIP Explanations

Implementation of [Enhancing Concept Localization in CLIP-based Concept Bottleneck Models](https://arxiv.org/abs/2510.07115).

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download the dataset**

CUB: https://huggingface.co/datasets/RemiKaz/CUB_disantangle_CLIP/tree/main

---

## Usage

### Explain a prediction (`plot_exp_multiple.py`)

Generates a SHAP explanation and attention heatmaps for one or more images. Example:

```bash
python plot_exp_multiple.py \
    --dataset cub \
    --dataset_root /path/to/CUB \
    --model_pth best_model_cub_precompute_clip_linear_precompute_median-loc-random.pt \
    --weights_pth weights/weights_cub_median_random_loc_all_classes.npy \
    --image_paths /path/to/CUB/test/189.Red_bellied_Woodpecker/image.jpg \
    --output_dir ./results \
    --disantangled_version median-loc-random
```

Key arguments:
| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name | `cub` |
| `--dataset_root` | Path to dataset root | — |
| `--model_pth` | Path to trained CBM `.pt` file | — |
| `--weights_pth` | Path to attention weights `.npy` file | — |
| `--image_paths` | One or more image paths to explain | — |
| `--output_dir` | Directory to save outputs | `./one_sample_output` |
| `--model_name` | CLIP backbone | `ViT-B-16` |
| `--pretrained_name` | CLIP pretrained weights | `laion2b_s34b_b88k` |
| `--device` | Device | `cuda:0` |
| `--disantangled_version` | Disentangling variant | `median-loc-random` |

---

### Statistical analysis (`test_stat.py`)

Runs CLIP score histogram analysis to evaluate disentanglement. Examples:

```bash
python test_stat.py \
    --experiments_to_run plot_clip_scores_histogram \
    --classes_to_probe Least_Tern-small_slim_body \
    --clip_concepts small_slim_body \
    --classifier_dataset cub \
    --root_cub /path/to/CUB \
    --disantangled_version median-loc-random
```

To run without disentangling (baseline):
```bash
python test_stat.py \
    --experiments_to_run plot_clip_scores_histogram \
    --classes_to_probe Least_Tern-small_slim_body \
    --clip_concepts small_slim_body \
    --classifier_dataset cub \
    --root_cub /path/to/CUB \
    --disantangled_version None
```

Key arguments:
| Argument | Description | Default |
|---|---|---|
| `--classifier_dataset` | Dataset in `[imagenet, monumai, coco, cub]` | `imagenet` |
| `--classes_to_probe` | Class-concept pair (e.g. `Least_Tern-small_slim_body`) | — |
| `--clip_concepts` | Concept text for CLIP scoring | — |
| `--disantangled_version` | Disentangling variant (`None` to disable) | `median-LTC-full` |
| `--experiments_to_run` | Comma-separated experiments to run | `plot_clip_scores_histogram,` |
| `--output_dir` | Directory to save CSV results | `results` |
