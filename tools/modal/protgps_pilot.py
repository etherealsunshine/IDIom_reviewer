# tools/modal/protgps_pilot.py
import modal

app = modal.App("idiom-protgps-pilot")
model_volume = modal.Volume.from_name("idiom-protgps-models", create_if_missing=True)
MODEL_ROOT = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "pandas",
        "numpy",
        "torch==2.4.0",
        "pytorch-lightning==1.9.5",
        "torchmetrics<1.0",
        "tqdm",
        "biopython",
        "fair-esm",
        "openpyxl",
        "huggingface_hub",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/rotskoff-group/idiom /opt/idiom"
    )
    .env({
        "PYTHONPATH": "/opt/idiom/src:/opt/idiom/rewards/protgps"
    })
)


@app.function(image=image, gpu="L40S", timeout=10 * 60,volumes={MODEL_ROOT:model_volume})
def path_smoke():
    import os
    import sys
    from pathlib import Path
    import pytorch_lightning as pl

    cloud_io_imports = False
    try:
        from pytorch_lightning.utilities.cloud_io import load as pl_load
        cloud_io_imports = True
    except Exception:
        pass

    return {
        "pythonpath": os.environ.get("PYTHONPATH"),
        "sys_path_head": sys.path[:8],
        "idiom_pkg_exists": Path("/opt/idiom/src/idiom/__init__.py").exists(),
        "protgps_pkg_exists": Path("/opt/idiom/rewards/protgps/protgps/__init__.py").exists(),
        "protgps_inference_exists": Path("/opt/idiom/rewards/protgps/scripts/inference.py").exists(),
        "pytorch_lightning": pl.__version__,
        "cloud_io_imports": cloud_io_imports,
    }


@app.function(image=image, gpu="L40S", timeout=10 * 60,volumes={MODEL_ROOT:model_volume})
def import_smoke():
    import idiom
    import protgps
    from protgps.utils.registry import get_object
    from scripts.inference import COMPARTMENTS, predict_condensates

    return {
        "idiom_file": idiom.__file__,
        "protgps_file": protgps.__file__,
        "registry_func": str(get_object),
        "predict_func": str(predict_condensates),
        "n_compartments": len(COMPARTMENTS),
        "compartments": COMPARTMENTS,
    }

@app.function(
    image=image,
    gpu="L40S",
    timeout=10 * 60,
    volumes={MODEL_ROOT: model_volume},
)
def model_layout_smoke():
    from pathlib import Path

    expected = {
        "args": Path(MODEL_ROOT) / "protgps/protgps/32bf44b16a4e770a674896b81dfb3729.args",
        "ckpt": Path(MODEL_ROOT) / "protgps/protgps/32bf44b16a4e770a674896b81dfb3729epoch=26.ckpt",
        "esm_dir": Path(MODEL_ROOT) / "protgps/esm_models/esm2",
        "esm_model": Path(MODEL_ROOT) / "protgps/esm_models/esm2/checkpoints/esm2_t6_8M_UR50D.pt",
        "esm_contact": Path(MODEL_ROOT) / "protgps/esm_models/esm2/checkpoints/esm2_t6_8M_UR50D-contact-regression.pt",
    }

    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        }
        for name, path in expected.items()
    }

@app.function(
    image=image,
    gpu="L40S",
    timeout=20 * 60,
    volumes={MODEL_ROOT: model_volume},
)
def model_load_smoke():
    import sys
    import torch
    from pathlib import Path

    sys.path.insert(0, "/opt/idiom/rewards/protgps")

    from scripts.inference import load_model

    model_path = Path(MODEL_ROOT) / "protgps/protgps/32bf44b16a4e770a674896b81dfb3729epoch=26.ckpt"
    esm_dir = Path(MODEL_ROOT) / "protgps/esm_models/esm2"

    model = load_model(str(model_path), str(esm_dir))
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "loaded": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_class": model.__class__.__name__,
        "model_path_exists": model_path.exists(),
        "esm_dir_exists": esm_dir.exists(),
    }

@app.function(
    image=image,
    timeout=60 * 60,
    volumes={MODEL_ROOT: model_volume},
)
def download_protgps_assets():
    from pathlib import Path
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="jxliu2/idiom",
        repo_type="model",
        local_dir=MODEL_ROOT,
        allow_patterns=[
            "protgps/protgps/32bf44b16a4e770a674896b81dfb3729.args",
            "protgps/protgps/32bf44b16a4e770a674896b81dfb3729epoch=26.ckpt",
            "protgps/esm_models/esm2/checkpoints/esm2_t6_8M_UR50D.pt",
            "protgps/esm_models/esm2/checkpoints/esm2_t6_8M_UR50D-contact-regression.pt",
        ],
    )

    model_volume.commit()

    return model_layout_smoke.local()

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={MODEL_ROOT: model_volume},
)
def score_sequence_rows(rows: list[dict], batch_size: int = 32) -> list[dict]:
    import sys
    import torch
    from pathlib import Path

    sys.path.insert(0, "/opt/idiom/rewards/protgps")
    from scripts.inference import COMPARTMENTS, load_model, predict_condensates

    model_path = Path(MODEL_ROOT) / "protgps/protgps/32bf44b16a4e770a674896b81dfb3729epoch=26.ckpt"
    esm_dir = Path(MODEL_ROOT) / "protgps/esm_models/esm2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(model_path), str(esm_dir))
    model.eval().to(device)

    valid_rows = []
    sequences = []
    for row in rows:
        seq = str(row["sequence"]).upper()
        if 0 < len(seq) < 1800:
            clean_row = dict(row)
            clean_row["sequence"] = seq
            valid_rows.append(clean_row)
            sequences.append(seq)

    if not sequences:
        return []

    scores = predict_condensates(
        model,
        sequences,
        batch_size=batch_size,
        round=False,
    )
    out = []
    for row, score_vec in zip(valid_rows, scores.tolist()):
        for compartment, score in zip(COMPARTMENTS, score_vec):
            row[f"protgps_{compartment}"] = float(score)
        out.append(row)

    return out

@app.local_entrypoint()
def main(
    input_csv: str = "results/smoke/cheap.csv",
    output_csv: str = "results/smoke/cheap_modal_scores.csv",
    limit: int = 12,
):
    import pandas as pd
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not output_path.is_absolute():
        output_path = project_root / output_path

    df = pd.read_csv(input_path).head(limit)
    rows = df.to_dict(orient="records")

    scored = score_sequence_rows.remote(rows, batch_size=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scored).to_csv(output_path, index=False)

    print(f"Wrote {len(scored)} scored rows to {output_path}")
