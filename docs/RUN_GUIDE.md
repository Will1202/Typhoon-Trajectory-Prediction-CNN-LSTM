# Run Guide

[Project Overview](../README.md) · [中文说明](../README.zh-CN.md)

This guide describes the original scripts and their assumptions. The documentation refresh preserves the model code, datasets, checkpoints, and example figures. The commands below have been checked against the source; training and inference have not been run for this refresh.

The original `.idea/` metadata and `EAR5/` directory are preserved. The unused nested Git reference has been removed; the project files are available directly at the repository root.

## Environment Setup

The original [requirements.txt](../requirements.txt) records a Linux Conda environment, including packages installed through pip. Its `package=version=build` entries cannot be installed with `pip install -r requirements.txt`. The file also mixes CPU and CUDA PyTorch entries, so treat it as a historical record when choosing an environment.

Create a Python 3.10 environment:

```bash
conda create -n typhoon-cnn-lstm python=3.10
conda activate typhoon-cnn-lstm
```

Install the appropriate PyTorch 2.1 build for your system using the [official instructions for previous releases](https://pytorch.org/get-started/previous-versions/#v210). The original scripts automatically select CUDA when available and otherwise use the CPU.

The main recorded Python packages are:

| Purpose | Packages in the Original Environment |
| --- | --- |
| Training and inference | PyTorch 2.1.0, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2 |
| Plotting | Matplotlib 3.9.2, Cartopy 0.24.1 |
| ERA5 retrieval and reading | cdsapi 0.7.4, xarray 2024.10.0, cfgrib 0.9.14.1, ecCodes 2.38.3 |

A Jupyter interface is needed to run the notebooks. ERA5 download packages are needed when rebuilding the weather inputs; the repository already includes processed inputs for the example.

### Loading the Original Checkpoints

The `.pth` files contain complete `CNNLSTM` objects. Each inference entry point defines this class before calling `torch.load`.

PyTorch 2.6 changed the default to `weights_only=True`, which can reject these older checkpoints. Reproducing the historical environment or adapting the loader is therefore necessary when using a newer PyTorch release. For trusted checkpoints, PyTorch documents `weights_only=False` as a compatibility option. See the [serialization documentation](https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true). No loader changes are included in this documentation refresh.

## Inference and Custom Inputs

From the repository root:

```bash
cd inference
python inference.py
```

The working directory matters because [inference.py](../inference/inference.py) resolves its input paths relative to the current directory. It uses the following files:

| Input | Repository Path |
| --- | --- |
| Track observations | `train/best_track_records_test.csv` |
| Surface pressure grids | `data_preprocess/sp_data_matrix.json` |
| Sea surface temperature grids | `data_preprocess/sst_data_matrix.json` |
| Latitude model | `train/model/lat_cnn_lstm_model.pth` |
| Longitude model | `train/model/lon_cnn_lstm_model.pth` |

The default storm is `Carmen`. The script uses its first eight observations and predicts the next two coordinate increments. It adds these increments to the last observed position to obtain the predicted track.

Running the example writes `predicted_typhoon_path.csv`, `original_typhoon_path.csv`, and `Carmen.png` inside `inference/`, replacing files with those names.

### Use Another Storm

Prepare a CSV containing one storm, with observations in chronological order. The inference function selects by `Storm Name`, but the export function reads the first rows of the whole CSV. Keep at least eight input observations; ten observations allow the plotted comparison to include both future positions.

The required columns are `Storm Name`, `DateTime(UTC)`, `Latitude (°N)`, and `Longitude (°E)`. For each input observation, both weather JSON files must contain a matching key formed by concatenating the storm name and UTC timestamp, such as `Carmen1949011916`. Match the timestamp format used in the supplied files.

Update both `typhoon_name` assignments in `inference.py`, along with the CSV path, plot title, and output filename as needed. The script has no command-line options for these settings. Coordinate observations alone are insufficient because the model also reads the matching weather grids.

### Notebook Notes

Open [inference.ipynb](../inference/inference.ipynb) with `inference/` as the working directory. Its first inference cell sets `sst_json_path` to `../sst_data_matrix.json`; the included file is at `../data_preprocess/sst_data_matrix.json`. Correct this path in your working copy before running that cell. A later cell already uses the latter path.

The separate [visualization.py](../inference/visualization.py) refers to `infer_typhoon_next_track` without defining or importing it. Use `inference.py` for the complete script workflow.

## Training

Start from the repository root. If you ran the preceding inference example, first return there with `cd ..`:

```bash
cd train
python train.py
```

[train.py](../train/train.py) reads `best_track_records_p6.csv` and both weather JSON files from `data_preprocess/`. Its defaults include eight input steps, two prediction steps, batch size 32, up to 1,500 epochs, and a learning rate of 0.001.

Training saves complete model objects to `train/model/lat_cnn_lstm_model.pth` and `train/model/lon_cnn_lstm_model.pth`. These names match the included checkpoints, so preserve a copy before retraining. Review the implementation notes below before interpreting a new experiment.

## Data Preparation

The preprocessing materials cover CMA best-track parsing and ERA5 single-level surface pressure and sea surface temperature retrieval. See [cma.py](../data_preprocess/cma.py), [era5_t.py](../data_preprocess/era5_t.py), and [data_preprocess.ipynb](../data_preprocess/data_preprocess.ipynb).

Several cells and scripts retain absolute paths from the original machine, including `/home/...` and `/data1/...`. Replace these with your own paths and check each intermediate CSV before proceeding. The repository's weather download directory is named `EAR5/`.

Check the timestamp columns and their source time zone before converting dates. `cma.py` writes a column named `DateTime`, while `era5_t.py` expects `DateTime(BJT)` and subtracts eight hours. A column rename alone does not establish whether that conversion is appropriate for the source data.

Use your own CDS account and follow the [current CDS API setup instructions](https://cds.climate.copernicus.eu/how-to-api). The notebooks retain an older CDS endpoint. Review the endpoint, request format, and credentials before requesting new data; keep credentials out of committed files.

The grid extraction code samples 21 × 21 points around each storm position, spanning ±1° in each direction at 0.1° intervals. It selects the nearest ERA5 grid value for each point. This sampling interval describes the extraction code, not the native resolution of ERA5.

## Reproducibility Notes

These observations describe the preserved source and explain the limits of the archived examples:

- **Longitude target selection.** In `train.py`, `train_model` defaults to the latitude target. The longitude training call does not override this default, and `test_model` always selects the latitude target. Review both selections before using this code for a new latitude/longitude experiment.
- **Scaling at inference.** The inference code fits new coordinate scalers to the eight input observations. Training fits scalers to the full training CSV, and the original checkpoints do not store these scalers. The two stages therefore use different scaling parameters.
- **Evaluation split.** Training randomly splits overlapping windows and fits coordinate scalers before this split. Reported losses do not establish performance on independently held-out storms. A new evaluation should define its storm or time split before fitting preprocessing parameters.
- **Feature processing.** Weather grids are flattened and concatenated with coordinate differences. Three `Conv1d` and pooling blocks reduce the eight time steps to one before the LSTM. Describe this implemented architecture when comparing it with other CNN–LSTM methods.

The saved figures are examples from the original project. They do not establish aggregate forecast accuracy or performance on a new dataset.
