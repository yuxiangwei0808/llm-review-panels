"""Repository-relative locations. Nothing in the analysis code uses absolute paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MANIFEST = DATA / "manifests" / "study_manifest.json"
DOSSIERS = DATA / "dossiers"
REFERENCE = PROCESSED / "reference_results"
RESULTS = ROOT / "results"


def results_dir(name: str) -> Path:
    """Create and return ``results/<name>/`` (``results/<experiment>/<name>/`` off the paper)."""
    from .experiment import current
    experiment = current()
    path = (RESULTS if experiment.is_paper else RESULTS / experiment.name) / name
    path.mkdir(parents=True, exist_ok=True)
    return path
