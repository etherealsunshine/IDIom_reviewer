from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "idiom-deeploc-validation"
DATA_ROOT = "/audit_data"
DEEPLOC_CACHE = "/deeploc_volume"

data_volume = modal.Volume.from_name("idiom-audit-data", create_if_missing=True)
deeploc_volume = modal.Volume.from_name("idiom-deeploc-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy<2",
        "pandas<2.0.0",
        "scipy",
        "biopython",
        "scikit-learn",
        "matplotlib",
        "h5py",
        "sentencepiece",
        "transformers==4.48.3",
        "fair-esm==0.4.0",
        "torch==2.4.0",
        "pytorch-lightning==1.9.5",
        "torchmetrics<1.0",
        "onnxruntime",
    )
    .env(
        {
            "TORCH_HOME": f"{DEEPLOC_CACHE}/torch",
            "HF_HOME": f"{DEEPLOC_CACHE}/huggingface",
            "TRANSFORMERS_CACHE": f"{DEEPLOC_CACHE}/huggingface",
        }
    )
)

app = modal.App(APP_NAME)

SOURCE_FASTAS = {
    "base_idp": "idr_datasets/generated_sequences/generated_idps/generated_idrs.fasta",
    "rl_nucleolus": "idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_idrs.fasta",
    "rl_chromosome": "idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_idrs.fasta",
    "rl_p-body": "idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_idrs.fasta",
    "rl_stress_granule": "idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_idrs.fasta",
    "disprot_idr": "idr_datasets/reference_sequences/DisProt/disprot_idrs.fasta",
}

PACKAGE_PATTERNS = (
    "packages/deeploc2*.tar.gz",
    "packages/deeploc*.tar.gz",
    "packages/deeploc2*.zip",
    "packages/deeploc*.zip",
    "deeploc2*.tar.gz",
    "deeploc*.tar.gz",
    "deeploc2*.zip",
    "deeploc*.zip",
)


def _run(cmd: list[str], cwd: Path | None = None) -> dict:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _find_deeploc_package() -> Path | None:
    root = Path(DEEPLOC_CACHE)
    candidates: list[Path] = []
    for pattern in PACKAGE_PATTERNS:
        candidates.extend(root.glob(pattern))
    for path in sorted(candidates):
        if path.is_file():
            return path
    for path in sorted(root.glob("packages/*")) + sorted(root.glob("deeploc*")):
        if path.is_dir() and ((path / "setup.py").exists() or (path / "pyproject.toml").exists()):
            return path
    return None


def _ensure_deeploc_cli() -> dict:
    existing = shutil.which("deeploc2")
    if existing:
        return {"installed": True, "deeploc2": existing, "install": None}

    package = _find_deeploc_package()
    if package is None:
        seen = [str(p.relative_to(DEEPLOC_CACHE)) for p in Path(DEEPLOC_CACHE).glob("**/*") if p.is_file()][:40]
        raise FileNotFoundError(
            "No installable DeepLoc 2 package found in the idiom-deeploc-cache Modal volume. "
            "Upload the official DTU tarball with: "
            "modal volume put idiom-deeploc-cache /path/to/deeploc2.tar.gz packages/ --force. "
            f"Files seen in cache volume: {seen}"
        )

    install = _run([sys.executable, "-m", "pip", "install", "--no-deps", str(package)])
    if install["returncode"] != 0:
        raise RuntimeError(f"DeepLoc package install failed: {install}")

    installed = shutil.which("deeploc2")
    if installed is None:
        raise RuntimeError(f"Installed {package}, but no deeploc2 executable appeared on PATH: {install}")
    return {"installed": True, "deeploc2": installed, "package": str(package), "install": install}


def _clean_sequence(seq: str) -> str:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(aa for aa in seq.upper().replace("*", "") if aa in allowed)


def _write_limited_fasta(input_path: Path, output_path: Path, source: str, limit: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    seq_id = None
    chunks: list[str] = []

    with output_path.open("w") as out:
        def flush():
            nonlocal n
            if seq_id is None:
                return
            seq = _clean_sequence("".join(chunks))
            if not seq:
                return
            n += 1
            out.write(f">{source}|{seq_id}\n")
            for start in range(0, len(seq), 80):
                out.write(seq[start : start + 80] + "\n")

        with input_path.open() as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    flush()
                    if limit and n >= limit:
                        return n
                    seq_id = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
        flush()
    return n


@app.function(
    image=image,
    gpu="L40S",
    timeout=12 * 60 * 60,
    volumes={DATA_ROOT: data_volume, DEEPLOC_CACHE: deeploc_volume},
)
def deeploc_install_smoke() -> dict:
    install = _ensure_deeploc_cli()
    help_result = _run(["deeploc2", "-h"])
    deeploc_volume.commit()
    return {"install": install, "help": help_result}


@app.function(
    image=image,
    gpu="L40S",
    timeout=12 * 60 * 60,
    volumes={DATA_ROOT: data_volume, DEEPLOC_CACHE: deeploc_volume},
)
def run_deeploc(
    run_name: str = "deeploc_pilot_2k",
    per_source_limit: int = 2000,
    disprot_limit: int = 0,
    model: str = "Fast",
    device: str = "cuda",
) -> dict:
    install = _ensure_deeploc_cli()
    root = Path(DATA_ROOT)
    input_dir = root / "deeploc_inputs" / run_name
    output_dir = root / "deeploc_outputs" / run_name
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    outputs = {}
    for source, rel_path in SOURCE_FASTAS.items():
        limit = disprot_limit if source == "disprot_idr" else per_source_limit
        fasta_path = input_dir / f"{source}.fasta"
        counts[source] = _write_limited_fasta(root / rel_path, fasta_path, source, limit)
        source_out = output_dir / source
        if source_out.exists():
            shutil.rmtree(source_out)
        source_out.mkdir(parents=True, exist_ok=True)
        cmd = ["deeploc2", "-f", str(fasta_path), "-o", str(source_out), "-m", model, "-d", device]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        outputs[source] = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "output_dir": str(source_out.relative_to(root)),
            "files": [str(p.relative_to(source_out)) for p in source_out.rglob("*") if p.is_file()],
        }
        if proc.returncode != 0:
            data_volume.commit()
            deeploc_volume.commit()
            raise RuntimeError(f"DeepLoc failed for {source}: {outputs[source]}")

    data_volume.commit()
    deeploc_volume.commit()
    return {
        "run_name": run_name,
        "model": model,
        "device": device,
        "install": install,
        "counts": counts,
        "outputs": outputs,
        "remote_output_dir": str(output_dir.relative_to(root)),
    }


@app.local_entrypoint()
def main(
    run_name: str = "deeploc_pilot_2k",
    per_source_limit: int = 2000,
    disprot_limit: int = 0,
    model: str = "Fast",
    device: str = "cuda",
    install_smoke: bool = False,
):
    if install_smoke:
        print(deeploc_install_smoke.remote())
        return
    print(
        run_deeploc.remote(
            run_name=run_name,
            per_source_limit=per_source_limit,
            disprot_limit=disprot_limit,
            model=model,
            device=device,
        )
    )
