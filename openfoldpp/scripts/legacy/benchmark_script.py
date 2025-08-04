Benchmark script
#!/usr/bin/env python3
"""
Benchmark CASP/other target sets against a local (or remote) fold‑prediction API.

Key features added compared with the minimal script:
  • CLI configuration for dataset paths, API URL, thread workers, and output CSV.
  • ThreadPoolExecutor for parallel folds (default: 4 workers).
  • Hardened HTTP layer (requests Session with retry/timeout/back‑off).
  • Per‑target wall‑clock timing, RMSD (CA) and optional TM‑score (if `tmtools` is installed).
  • Rich logging plus tqdm progress bar so long runs show activity.
  • Preservation of all prediction PDBs; failures moved to a dedicated directory for post‑mortem.
  • Clean CSV schema ready for downstream analytics.

Install deps (Ubuntu example):
    pip install biopython requests pandas tqdm
    # optional for TM‑score
    pip install tmtools

Usage example:
    python benchmark_fold_api.py \
        --fasta-dir ./casp14/fasta \
        --pdb-dir   ./casp14/pdb   \
        --api-url   http://localhost:8000/fold \
        --workers   4 \
        --output    casp14_benchmark.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from Bio.PDB import PDBParser, Superimposer
import pandas as pd

# Progress bar; if tqdm unavailable, fall back to identity iterator
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(iterable, *_, **__):  # type: ignore
        return iterable

# Optional TM‑score via tmtools (https://github.com/realbigws/tmtools)
try:
    from tmtools import tmscore  # type: ignore
except ImportError:  # pragma: no cover
    tmscore = None  # type: ignore

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def make_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Return a Requests session with sensible retry/back‑off."""
    session = requests.Session()
    retry_cfg = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_cfg, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fold_sequence(session: requests.Session, api_url: str, seq: str, timeout: int = 60) -> Optional[str]:
    """Call the fold API and return PDB string or None on failure."""
    try:
        resp = session.post(api_url, json={"sequence": seq}, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("pdb")
    except requests.RequestException as exc:
        logging.warning("HTTP error: %s", exc)
    return None


def compute_rmsd(pred_path: Path, ref_path: Path) -> Optional[float]:
    parser = PDBParser(QUIET=True)
    pred = parser.get_structure("pred", str(pred_path))
    ref = parser.get_structure("ref", str(ref_path))
    pred_atoms = [a for a in pred.get_atoms() if a.get_id() == "CA"]
    ref_atoms = [a for a in ref.get_atoms() if a.get_id() == "CA"]
    if len(pred_atoms) != len(ref_atoms):
        return None
    sup = Superimposer()
    sup.set_atoms(ref_atoms, pred_atoms)
    return float(sup.rms)


def compute_tm_score(pred_path: Path, ref_path: Path) -> Optional[float]:
    if tmscore is None:
        return None
    try:
        score = tmscore(ref=str(ref_path), model=str(pred_path)).tm_score
        return float(score)
    except Exception as exc:  # noqa
        logging.debug("TM‑score error on %s vs %s: %s", pred_path, ref_path, exc)
        return None


def read_fasta(seq_file: Path) -> str:
    with open(seq_file) as fh:
        lines = [l.strip() for l in fh if not l.startswith(">")]
    return "".join(lines)


def process_target(seq_file: Path, args: argparse.Namespace, session: requests.Session) -> Dict[str, object]:
    seq_id = seq_file.stem
    seq = read_fasta(seq_file)
    t0 = time.perf_counter()

    pdb_text = fold_sequence(session, args.api_url, seq)
    if pdb_text is None:
        return {"seq_id": seq_id, "len": len(seq), "status": "folding_failed"}

    pred_path = args.tmp_dir / f"{seq_id}_pred.pdb"
    with open(pred_path, "w") as fh:
        fh.write(pdb_text)

    ref_path = args.pdb_dir / f"{seq_id}.pdb"
    if not ref_path.exists():
        return {"seq_id": seq_id, "len": len(seq), "status": "reference_missing"}

    try:
        rmsd = compute_rmsd(pred_path, ref_path)
        tm = compute_tm_score(pred_path, ref_path)
        status = "success" if rmsd is not None else "rmsd_mismatch"
    except Exception as exc:  # noqa
        logging.exception("RMSD error on %s", seq_id)
        (args.failed_dir / pred_path.name).write_text(pdb_text)
        return {"seq_id": seq_id, "len": len(seq), "status": f"rmsd_error: {exc}"}

    elapsed = time.perf_counter() - t0
    return {
        "seq_id": seq_id,
        "len": len(seq),
        "status": status,
        "rmsd_ca": rmsd,
        "tm_score": tm,
        "time_s": round(elapsed, 2),
    }

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fold API against reference PDBs.")
    parser.add_argument("--fasta-dir", type=Path, required=True, help="Directory with FASTA files.")
    parser.add_argument("--pdb-dir", type=Path, required=True, help="Directory with reference PDB files.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/fold", help="Fold API endpoint.")
    parser.add_argument("--workers", type=int, default=4, help="Thread workers for concurrent folds.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"), help="CSV output path.")
    parser.add_argument("--tmp-dir", type=Path, default=Path("./tmp"), help="Dir to store prediction PDBs.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging verbosity.")
    args = parser.parse_args()

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.failed_dir = args.tmp_dir / "failed"
    args.failed_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    fasta_files = sorted(args.fasta_dir.glob("*.fasta"))
    if not fasta_files:
        raise SystemExit(f"No FASTA files found in {args.fasta_dir}")

    session = make_session()

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for res in tqdm(pool.map(lambda f: process_target(f, args, session), fasta_files), total=len(fasta_files)):
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logging.info("Wrote results to %s", args.output)
    print(df)


if __name__ == "__main__":
    main()
