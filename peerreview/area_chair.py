"""The LLM area-chair experiment (Section 4): a model reads a paper and a review record and
decides accept or reject.

Arms: ``H0`` human reviews; ``H2`` human reviews plus the author-reviewer discussion; ``P0`` LLM
reviews; ``U1`` LLM reviews revised after the human rebuttal; ``S1`` the same after a
model-written rebuttal (``-H`` model-chosen reviewer scores, ``-I`` rule-calculated scores).

The review records (evidence packets) of the 60 study papers are frozen under
``data/llm_as_ac_60`` and ``data/llm_as_ac_simulated_author_60`` and indexed by
``data/processed/area_chairs/packet_index.csv``, so any model can be run as an area chair on
exactly the inputs the paper used. Building packets for other papers is not covered here.
"""
from __future__ import annotations

import datetime
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from . import prompts
from .config import DATA, RAW_OUTPUTS, ROOT, TEMPERATURE, models, venues
from .papers import paper_text, study_papers
from .runner import RunWriter

PACKET_INDEX = DATA / "processed" / "area_chairs" / "packet_index.csv"
MAX_OUTPUT_TOKENS = 2048
DECISION = re.compile(r"<decision>\s*(accept|reject)\s*</decision>", re.IGNORECASE)
META_REVIEW = re.compile(r"<meta_review>(.*?)</meta_review>", re.DOTALL | re.IGNORECASE)
STAGE = "llm_as_ac"


def render_prompt(venue_key: str, paper: str, packet: str) -> str:
    venue = venues()[venue_key]
    return prompts.render("ac_binary", venue_name=venue.name,
                          rating_scale_block=venue.rating_scale_block,
                          accept_threshold=str(venue.accept_threshold), paper_text=paper,
                          evidence_packet=packet)


def parse_decision(raw: str) -> tuple[str | None, str]:
    decision = DECISION.findall(raw or "")
    review = META_REVIEW.search(raw or "")
    return (decision[-1].lower() if decision else None), (review.group(1).strip() if review else "")


def packet_index() -> pd.DataFrame:
    """paper_id, venue_key, arm, run_idx, packet (path relative to the repository root)."""
    return pd.read_csv(PACKET_INDEX)


def run(client, model_keys: list[str], arms: list[str], tag: str, runs: int = 3,
        papers: list[str] | None = None, workers: int = 8, root: Path = RAW_OUTPUTS) -> None:
    """Run models as area chairs on the frozen review records; one record file per venue."""
    index = packet_index()
    index = index[index.arm.isin(arms) & (index.run_idx < runs)]
    if papers:
        index = index[index.paper_id.isin(papers)]
    study = study_papers().set_index("paper_id")
    for key in model_keys:
        model = models()[key]
        writers = {v: RunWriter(root / key / v / f"{STAGE}_{tag}", STAGE) for v in index.venue_key.unique()}

        def decide(row, model=model, writers=writers):
            writer = writers[row.venue_key]
            if (row.paper_id, row.run_idx, row.arm) in writer.done:
                return
            prompt = render_prompt(row.venue_key, paper_text(study.loc[row.paper_id]),
                                   (ROOT / row.packet).read_text())
            artifact, prompt_sha = writer.artifact("input_artifacts", prompt, ".txt")
            text, usage = client.complete(model.api_model, prompt, run_idx=row.run_idx,
                                          temperature=TEMPERATURE if model.accepts_temperature
                                          else None, max_tokens=MAX_OUTPUT_TOKENS,
                                          **({"extra": model.request} if model.request else {}))
            decision, meta_review = parse_decision(text)
            writer.append(dict(
                paper_id=row.paper_id, venue_key=row.venue_key, arm=row.arm, run_idx=int(row.run_idx),
                call_type=row.arm, ac_model_id=model.key,
                model_snapshot=usage.get("execution_model", model.api_model), decision=decision,
                parse_ok=decision is not None, meta_review=meta_review, raw_response=text,
                rendered_prompt_artifact=artifact, rendered_prompt_sha256=prompt_sha,
                evidence_packet=row.packet, usage=usage,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat()))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(decide, index.itertuples()))
        print(f"{key}: {len(index)} decisions, ${client.spent:.2f} spent so far", flush=True)


def decision_table(model_keys: list[str], tag: str, root: Path = RAW_OUTPUTS) -> pd.DataFrame:
    """Decisions in the layout the Section 4 analysis reads (``ac_records_main.parquet``)."""
    study = study_papers().set_index("paper_id")
    rows = []
    for key in model_keys:
        for path in sorted((root / key).glob(f"*/{STAGE}_{tag}/{STAGE}.jsonl")):
            for line in path.open():
                r = json.loads(line)
                if r["parse_ok"]:
                    paper = study.loc[r["paper_id"]]
                    rows.append(dict(paper_id=r["paper_id"], venue_key=r["venue_key"],
                                     outcome=paper.outcome, truth=int(paper.outcome == "accept"),
                                     ac_model_id=key, arm=r["arm"], run_idx=r["run_idx"],
                                     prediction=int(r["decision"] == "accept")))
    return pd.DataFrame(rows)
