# Open5GS Training and Inference Guide

This walkthrough explains how to train the bundled PLELog pipeline on the Open5GS logs that ship with this repository and how to reuse the trained artefacts for inference on any additional Open5GS log files.

## Prerequisites

1. Create and activate a Python 3.8+ environment.
2. Install the runtime dependencies from the project root:
   ```bash
   pip install -r requirements.txt
   ```
3. Download `glove.6B.300d.txt` from the [Stanford GloVe project](https://nlp.stanford.edu/projects/glove/) and place it at `datasets/glove.6B.300d.txt`. The representation modules look for the embeddings at this exact location.
4. Ensure the Open5GS dataset is present under `datasets/open5gs/logs/`. The expected layout is:
   ```
   datasets/open5gs/logs/
     ├── UE-INITIATED-DEREGISTRATION-PROCEDURE/
     ├── UE-REQUESTED-PDU-SESSION-RELEASE-PROCEDURE/
     └── UE-TRIGGERED-SERVICE-REQUEST-PROCEDURE/
   ```
   Each procedure directory contains one folder per network function, and those folders hold the `.log` files. Every line inside the log files is treated as an independent log message.

## Training the model

Run the training script from the repository root:

```bash
python approaches/PLELog.py --dataset open5gs --parser IBM --mode train
```

What this command does:

- Recursively scans `datasets/open5gs/logs/` for `.log` files and flattens their contents into `datasets/open5gs/open5gs_combined.log`.
- Learns or reuses a Drain parser with configuration `conf/HDFS.ini`. The parser state is saved under `datasets/open5gs/persistences/` for subsequent runs.
- Generates IBM Drain inputs inside `datasets/open5gs/inputs/IBM/` and computes template embeddings using the supplied GloVe vectors.
- Fits the PLELog model. Checkpoints are stored in `outputs/models/PLELog/open5gs_IBM/model/` (both `*_best.pt` and `*_last.pt` files when training completes).

The first run may take several minutes because the parser is trained from scratch. Subsequent runs reuse the persisted Drain state and skip the expensive parsing stage.

## Running inference on new logs

Once training has finished (and the parser/model artefacts exist), score any Open5GS-formatted log file line-by-line:

```bash
python approaches/PLELog.py --dataset open5gs --parser IBM --inference_file path/to/your_logs.log
```

Additional notes:

- Leave the dataset argument in lowercase (`open5gs`) so the loader can locate `datasets/open5gs`.
- The script loads the saved Drain parser and model weights automatically. If either artefact is missing, rerun the training command first.
- Set `--threshold` to change the anomaly probability cut-off (default is `0.5`).
- Predictions are written to `outputs/results/PLELog/open5gs_IBM/inference/` as `<logname>_predictions.csv`. Each row contains the zero-based line number, the internal block identifier, the raw log line, the predicted label, and the anomaly score output by the model.

## Resetting the pipeline

To retrain from scratch, delete the generated artefacts before running the training command again:

- Remove `datasets/open5gs/open5gs_combined.log` and the folders under `datasets/open5gs/persistences/` to force the Drain parser to relearn templates.
- Remove `outputs/models/PLELog/open5gs_IBM/` if you want to discard saved checkpoints.

These steps are optional—retraining with the artefacts in place simply updates them in-place.

## Troubleshooting

- **`Failed to load template embeddings` during inference** – ensure the training command completed successfully so that `datasets/open5gs/persistences/**/templates.vec` exists.
- **`Inference file not found` error** – double-check the path passed to `--inference_file`.
- **`glove.6B.300d.txt` related errors** – confirm that the embeddings file is available at `datasets/glove.6B.300d.txt`.

After the above steps you should be able to iteratively train on the provided Open5GS logs and reuse the resulting model to analyse additional traces without modifying the code further.
