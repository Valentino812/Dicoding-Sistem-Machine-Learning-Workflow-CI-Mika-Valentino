import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_FILE = "harga_emas_dan_swiss_franc_preprocessing.csv"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_TRACKING_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlruns.db")

EXPERIMENT_NAME = "workflow-ci-random-forest-gold"

DATE_COLUMN = "Date"
TARGET_COLUMN = "return_gold"
FEATURE_COLUMNS = [
    "return_gold_lag_1",
    "return_gold_lag_2",
    "return_gold_lag_3",
    "return_gold_lag_4",
    "return_gold_lag_5",
    "return_gold_lag_6",
    "return_gold_lag_7",
    "return_gold_lag_8",
    "return_gold_lag_9",
    "return_gold_lag_10",
    "return_chf_lag_1",
    "return_chf_lag_2",
    "return_chf_lag_3",
    "return_chf_lag_4",
    "return_chf_lag_5",
    "return_chf_lag_6",
    "return_chf_lag_7",
    "return_chf_lag_8",
    "return_chf_lag_9",
    "return_chf_lag_10",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = None

MODEL_ARTIFACT_PATH = "model"
RUN_ID_FILENAME = "mlflow_run_id.txt"
METRICS_FILENAME = "metrics.json"


# ---------------------------------------------------------------------------
# Argumen
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pelatihan ulang RandomForestRegressor untuk return_gold."
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_FILE,
        help="Path dataset hasil preprocessing (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Direktori penyimpanan run id dan metrik (default: %(default)s).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help="Jumlah pohon RandomForest (default: %(default)s).",
    )
    return parser.parse_args(argv)


def resolve_data_path(path: str) -> str:
    """Cari dataset pada beberapa kandidat lokasi agar konsisten."""
    candidates = [
        path,
        os.path.join(BASE_DIR, path),
        os.path.join(os.getcwd(), path),
        os.path.join(BASE_DIR, os.path.basename(path)),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "Dataset tidak ditemukan. Lokasi yang dicoba: "
        + ", ".join(sorted({os.path.abspath(c) for c in candidates if c}))
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)

    required = [DATE_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada dataset: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Dataset mengandung nilai kosong pada kolom yang dibutuhkan.")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame):
    n_train = int(len(df) * (1.0 - TEST_SIZE))

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(mse ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n===== Metrik {label} =====")
    for name in ["MAE", "MSE", "RMSE", "R2"]:
        print(f"{name:>5}: {metrics[name]:.6f}")


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------
def ensure_experiment(name: str) -> None:
    """Buat experiment bila belum ada dengan artifact root absolut."""
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return
    artifact_root = Path(BASE_DIR, "mlruns").resolve().as_uri()
    mlflow.create_experiment(name, artifact_location=artifact_root)


def log_model(model) -> str:
    """Catat model pada artefak MLflow dengan lokasi konsisten 'model'."""
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs = {}
    if "skops_trusted_types" in params:
        # MLflow >= 3.5 memakai serialisasi skops; RandomForest yang dilatih
        # sendiri perlu menandai tipe pohon sebagai terpercaya.
        kwargs["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
    if "name" in params:
        return mlflow.sklearn.log_model(model, name=MODEL_ARTIFACT_PATH, **kwargs)
    return mlflow.sklearn.log_model(model, artifact_path=MODEL_ARTIFACT_PATH, **kwargs)


def write_outputs(output_dir: str, run_id: str, summary: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)

    run_id_path = os.path.join(output_dir, RUN_ID_FILENAME)
    with open(run_id_path, "w", encoding="utf-8") as handle:
        handle.write(run_id + "\n")

    metrics_path = os.path.join(output_dir, METRICS_FILENAME)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Run ID disimpan di : {run_id_path}")
    print(f"Metrik disimpan di : {metrics_path}")


def download_model_artifact(run_id: str, output_dir: str) -> str:
    """Unduh artefak model dari run terbaru ke output_dir/model."""
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
    local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=output_dir)
    print(f"Model artefak ({model_uri}) tersalin ke: {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    args = parse_args(argv)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)

    data_path = resolve_data_path(args.data_path)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Jika dijalankan oleh MLflow Project, MLFLOW_RUN_ID sudah diisi dan
    # experiment sudah dibuat oleh perintah `mlflow run`.
    running_via_project = bool(os.environ.get("MLFLOW_RUN_ID"))
    if not running_via_project:
        ensure_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_dataset(data_path)
    x_train, y_train, x_test, y_test = chronological_split(df)

    print(f"Tracking URI         : {tracking_uri}")
    print(f"Dataset              : {data_path}")
    print(f"Jumlah baris         : {len(df)}")
    print(f"Rentang tanggal      : {df[DATE_COLUMN].min().date()} s.d. {df[DATE_COLUMN].max().date()}")
    print(f"Fitur                : {FEATURE_COLUMNS}")
    print(f"Target               : {TARGET_COLUMN}")
    print(f"Data training/testing: {len(x_train)} / {len(x_test)}")

    params = {
        "model": "RandomForestRegressor",
        "n_estimators": int(args.n_estimators),
        "max_depth": str(MAX_DEPTH) if MAX_DEPTH is not None else "None",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target": TARGET_COLUMN,
        "n_features": len(FEATURE_COLUMNS),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }

    model = RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="rf_retrain") as run:
        run_id = run.info.run_id

        mlflow.log_params(params)

        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        train_metrics = evaluate(y_train, y_pred_train)
        test_metrics = evaluate(y_test, y_pred_test)

        print_metrics("Training", train_metrics)
        print_metrics("Testing", test_metrics)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        log_model(model)

        experiment_name = mlflow.get_experiment(run.info.experiment_id).name
        summary = {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "experiment_name": experiment_name,
            "tracking_uri": tracking_uri,
            "data_path": data_path,
            "target": TARGET_COLUMN,
            "features": FEATURE_COLUMNS,
            "n_estimators": int(args.n_estimators),
            "test_size": TEST_SIZE,
            "n_train": len(x_train),
            "n_test": len(x_test),
            "train": train_metrics,
            "test": test_metrics,
        }
        write_outputs(output_dir, run_id, summary)
        download_model_artifact(run_id, output_dir)

        print(f"\nSelesai. MLflow run id: {run_id}")


if __name__ == "__main__":
    main()
