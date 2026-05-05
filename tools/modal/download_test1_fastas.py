from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "idiom-test1-data"
DATA_ROOT = "/audit_data"

data_volume = modal.Volume.from_name("idiom-audit-data", create_if_missing=True)
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
)

TEST1_FASTA_PATTERNS = [
    "idr_datasets/generated_sequences/generated_idps/generated_idrs.fasta",
    "idr_datasets/generated_sequences/generated_idps/generated_full.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_idrs.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_full.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_idrs.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_full.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_idrs.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_full.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_idrs.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_full.fasta",
]


def _expected_paths() -> dict[str, Path]:
    root = Path(DATA_ROOT)
    return {
        "base_idps_idrs": root / "idr_datasets/generated_sequences/generated_idps/generated_idrs.fasta",
        "base_idps_full": root / "idr_datasets/generated_sequences/generated_idps/generated_full.fasta",
        "rl_nucleolus_idrs": root / "idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_idrs.fasta",
        "rl_nucleolus_full": root / "idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_full.fasta",
        "rl_chromosome_idrs": root / "idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_idrs.fasta",
        "rl_chromosome_full": root / "idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_full.fasta",
        "rl_pbody_idrs": root / "idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_idrs.fasta",
        "rl_pbody_full": root / "idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_full.fasta",
        "rl_stress_granule_idrs": root / "idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_idrs.fasta",
        "rl_stress_granule_full": root / "idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_full.fasta",
    }


@app.function(
    image=image,
    timeout=10 * 60,
    volumes={DATA_ROOT: data_volume},
)
def layout_smoke() -> dict:
    expected = _expected_paths()
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
    timeout=10 * 60,
    volumes={DATA_ROOT: data_volume},
)
def inspect_dataset_repo() -> dict:
    from huggingface_hub import list_repo_files

    files = list_repo_files("jxliu2/idiom-datasets", repo_type="dataset")
    matches = [
        path
        for path in files
        if "generated_sequences" in path
        and (
            "generated_idps" in path
            or "generated_protgps" in path
        )
        and path.endswith(".fasta")
    ]
    return {
        "n_files": len(files),
        "n_generated_fasta_matches": len(matches),
        "generated_fasta_matches": matches[:300],
    }


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={DATA_ROOT: data_volume},
)
def download_test1_fastas() -> dict:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="jxliu2/idiom-datasets",
        repo_type="dataset",
        local_dir=DATA_ROOT,
        allow_patterns=TEST1_FASTA_PATTERNS,
    )
    data_volume.commit()
    return layout_smoke.local()


@app.local_entrypoint()
def main(action: str = "layout"):
    if action == "layout":
        print(layout_smoke.remote())
    elif action == "inspect":
        print(inspect_dataset_repo.remote())
    elif action == "download":
        print(download_test1_fastas.remote())
    else:
        raise ValueError("action must be one of: layout, inspect, download")
