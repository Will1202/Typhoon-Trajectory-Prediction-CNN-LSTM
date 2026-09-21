# Run Guide

[Project Overview](../README.md) · [中文说明](../README.zh-CN.md)

This guide covers environment setup, input data, model training, and trajectory visualization. Run the commands from the indicated directories so that relative paths resolve to the included project files.

## Environment Setup

The [requirements.txt](../requirements.txt) file records a Linux Conda environment, including packages installed through pip. Use it as a package and version reference. Its `package=version=build` format is intended for Conda environment records rather than `pip install -r`. Select a CPU or CUDA PyTorch build to match your platform.

Create a Python 3.10 environment:

```bash
conda create -n typhoon-cnn-lstm python=3.10
conda activate typhoon-cnn-lstm
```

Install the appropriate PyTorch 2.1 build for your system using the [official instructions for previous releases](https://pytorch.org/get-started/previous-versions/#v210). The scripts automatically select CUDA when available and otherwise use the CPU.

The main recorded Python packages are:

| Purpose | Recorded Package Versions |
| --- | --- |
| Training and inference | PyTorch 2.1.0, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2 |
| Plotting | Matplotlib 3.9.2, Cartopy 0.24.1 |
| ERA5 retrieval and reading | cdsapi 0.7.4, xarray 2024.10.0, cfgrib 0.9.14.1, ecCodes 2.38.3 |

A Jupyter interface is needed to run the notebooks. ERA5 download packages are needed when rebuilding the weather inputs; the repository already includes processed inputs for the example.

### Loading the Model Checkpoints

The `.pth` files contain complete `CNNLSTM` objects. Each inference entry point defines this class before calling `torch.load`.

Use the recorded PyTorch 2.1 environment or configure model loading for your installed version. For trusted full-model checkpoints, PyTorch documents `weights_only=False` as a loading option. See the [serialization documentation](https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true) for version-specific behavior.

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

Configure the example directly in `inference.py`: update both `typhoon_name` assignments, along with the CSV path, plot title, and output filename as needed. Supply coordinate observations and matching weather grids together.

### Notebook Notes

Open [inference.ipynb](../inference/inference.ipynb) with `inference/` as the working directory. Set `sst_json_path` to `../data_preprocess/sst_data_matrix.json` in the first inference cell to use the included SST data.

Use `inference.py` for the combined prediction and plotting workflow. The plotting code in [visualization.py](../inference/visualization.py) can be used in a session where `infer_typhoon_next_track` is already defined.

## Training

Start from the repository root. If you ran the preceding inference example, first return there with `cd ..`:

```bash
cd train
python train.py
```

[train.py](../train/train.py) reads `best_track_records_p6.csv` and both weather JSON files from `data_preprocess/`. Its defaults include eight input steps, two prediction steps, batch size 32, up to 1,500 epochs, and a learning rate of 0.001.

Training saves complete model objects to `train/model/lat_cnn_lstm_model.pth` and `train/model/lon_cnn_lstm_model.pth`. These names match the included checkpoints, so preserve a copy before retraining. Configure coordinate targets and preprocessing as described in [experiment configuration](#experiment-configuration).

## Data Preparation

The preprocessing materials cover CMA best-track parsing and ERA5 single-level surface pressure and sea surface temperature retrieval. See [cma.py](../data_preprocess/cma.py), [era5_t.py](../data_preprocess/era5_t.py), and [data_preprocess.ipynb](../data_preprocess/data_preprocess.ipynb).

Set the local data paths in the preprocessing scripts and notebook cells to match your working directory. The repository's weather download directory is named `EAR5/`.

Align timestamp columns and time zones between preprocessing stages. `cma.py` writes `DateTime`; `era5_t.py` reads `DateTime(BJT)` and subtracts eight hours. Apply this conversion to Beijing-time inputs and use UTC timestamps for matching track records with ERA5 grids.

Use your own CDS account and configure the endpoint, request format, and credentials according to the [CDS API setup instructions](https://cds.climate.copernicus.eu/how-to-api). Keep credentials outside version control.

The grid extraction code samples 21 × 21 points around each storm position, spanning ±1° in each direction at 0.1° intervals. It selects the nearest ERA5 grid value for each point. This sampling interval describes the extraction code, not the native resolution of ERA5.

## Experiment Configuration

For a new training run, configure coordinate targets, preprocessing, and the evaluation protocol together:

| Setting | Configuration |
| --- | --- |
| Coordinate targets | Target channel 0 represents latitude; channel 1 represents longitude. Set `is_latitude=False` for longitude training and select channel 1 for longitude evaluation in `test_model`. |
| Coordinate scaling | The supplied training script fits scalers on the track CSV; inference fits them on the input window. For experiments that share a training scale, save the fitted training scalers and reuse them at inference. |
| Data partition | The supplied script randomly splits sequence windows. For evaluation on separate storms or time periods, define those partitions before constructing windows and fitting scalers. |
| Sequence features | Concatenate coordinate changes with flattened SP and SST grids. Three convolution and pooling blocks transform the eight input steps into one feature step for the LSTM. |
| Experiment records | Record the input period, data partition, coordinate target, scaler settings, and checkpoint alongside each run. |

The trajectory maps in the README illustrate the Carmen and Kujira examples included in this repository.
