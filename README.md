# SurfaceGuard

SurfaceGuard is an academic steel-surface inspection prototype. It uses a compact multiclass segmentation model to identify surface-defect regions in NEU-SEG images and in uploaded images through a Streamlit interface.

> The displayed mask is always a **model prediction**, not a ground-truth annotation. Area-based severity is a demonstration heuristic and is not a validated industrial accept/reject decision.

## Highlights

- Compact TinyUNet segmentation model with 66,976 trainable parameters.
- Four pixel classes: background plus inclusion, patch, and scratch defects.
- Native-resolution inspection: inference is performed at 200 x 200, then masks, probability maps, overlays, regions, and areas are visualised at the uploaded image's original resolution.
- Inspection workspace with the original image, predicted colour mask, labelled overlay, class legend, bounding boxes, region statistics, and per-class softmax probability maps.
- Downloadable native-resolution PNG, JSON, region-level CSV, and self-contained HTML inspection report with embedded images.
- Evaluation dashboard using saved test results and model-selection settings.

## Model and class mapping

The model is a grayscale TinyUNet operating on 200 x 200 images. It produces four logits per pixel and uses softmax probabilities to form a multiclass mask.

| Pixel ID | Defect category | Notes |
|---:|---|---|
| 0 | Background | No defect |
| 1 | Inclusion | Verified from the source-mask/image correspondence |
| 2 | Patch | Also described as “patches” in NEU-SEG material |
| 3 | Scratch | Also described as “scratches” in NEU-SEG material |

The application retains neutral `Label 1`–`Label 3` wording in selected interface elements to avoid overstating an unvalidated production taxonomy. The mapping above was established by checking representative source image/mask pairs and that the preparation pipeline preserves their pixel IDs.

## Included model files

The repository includes the checkpoints and selection settings required by the application:

| File | Purpose |
|---|---|
| `models/best_perclass.pt` | Default per-class model used by the application |
| `models/best_perclass_selection.json` | Selection/threshold configuration for the default checkpoint |
| `models/best.pt` | Saved baseline option |
| `models/selection.json` | Selection/threshold configuration for the baseline checkpoint |

Raw data, local environments, training logs, and incomplete Phase 1 experiment artifacts are intentionally excluded.

## Evaluation results

Saved results in `comparison_perclass_results.json` report the following metrics for the 840-image held-out test split:

| Method | Precision | Recall | Dice | IoU |
|---|---:|---:|---:|---:|
| Classical | 0.3142 | 0.0994 | 0.1510 | 0.0817 |
| Saved baseline CNN/hybrid | 0.2070 | 0.9025 | 0.3367 | 0.2025 |
| New model CNN | 0.7585 | 0.8277 | 0.7916 | 0.6551 |
| **New model hybrid** | **0.7688** | **0.8180** | **0.7926** | **0.6565** |

Per-class hybrid Dice is 0.6253 (Label 1), 0.8289 (Label 2), and 0.8183 (Label 3). These are stored evaluation outputs, not metrics recomputed at application launch. The first 150 test images had previously been inspected during development; the reported 840-image result includes them and should not be presented as a wholly fresh blind estimate.

## Installation

Python 3.13 on Windows was used for the verified local run. Other supported Python versions compatible with the listed PyTorch build may also work.

```powershell
git clone https://github.com/5206-hub/SurfaceGuard.git
cd SurfaceGuard
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

## Launch the application

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

Open the local URL shown by Streamlit (normally `http://localhost:8501`). The included checkpoints are sufficient for interactive inspection; a local dataset is not required to run the application.

## Dataset and preparation

SurfaceGuard uses the NEU-SEG steel-defect segmentation dataset. Obtain the dataset from its [Northeastern University source](https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/) and comply with its terms before use. The raw data is deliberately not redistributed in this repository.

After downloading the raw `NEU_Seg-main` directory, prepare it locally:

```powershell
.\\.venv\\Scripts\\python.exe tools\\prepare_neu.py --source path\\to\\NEU_Seg-main --output data
.\\.venv\\Scripts\\python.exe tools\\check_data.py --data data
```

The preparation script preserves source multiclass mask IDs and creates 2,904 training, 726 validation, and 840 held-out test image/mask pairs. Keep the test split separate from training and hyperparameter selection.

## Training and evaluation

The checked-in code supports reproducible training with an explicit seed, augmentation, learning-rate scheduling, early stopping, class weighting, checkpoints, and metric tracking. Do not overwrite the supplied checkpoints when experimenting; write new output paths instead.

```powershell
.\\.venv\\Scripts\\python.exe -u train.py --data data --epochs 5 --output models\\my_experiment.pt --log models\\my_experiment.log
.\\.venv\\Scripts\\python.exe -u compare_models.py
```

`evaluate.py` resolves the selection configuration associated with its supplied checkpoint. `comparison_perclass_results.json` and `results.json` are retained because the evaluation tab loads these saved artifacts.

## Tests

Run the reporting tests with:

```powershell
.\\.venv\\Scripts\\python.exe tests\\test_reporting.py
```

The project also contains Phase 1 test utilities. They may produce local experiment-preview files, which are ignored by Git.

## Limitations

- This is a research and academic demonstration, not a validated production inspection system.
- Prediction quality may degrade on images outside the dataset domain or under substantially different lighting, scale, material, or defect conditions.
- Pixel area is not a physical measurement; no calibration to millimetres or defect depth is supplied.
- The repository does not include raw data or unfinished experiment outputs.
