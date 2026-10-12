"""Consolidate curated attribute labels with confident tag-probe predictions.

For each scoped field a probe prediction replaces the annotator's value when its
confidence clears the field gate (the same gates as the large-public labels); the
evidence-route scope only touches accompaniment and performance, and overriding an
existing performance value there requires a stricter gate. A field that does not clear
its gate is restored to its pre-consolidation original. Each modified label.json gets
`field_sources[field] = "probe_consolidation"` (or `"probe_fill"` when the original value
was empty) and a `consolidation` block holding the previous value and confidence; the
stored `prompt` string keeps the pre-consolidation annotation. Originals are backed up
under the dataset's `archives/` before modification.

    python -m taste.annotation.consolidation --scope audio_route --execute
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

from taste import paths

CORPUS = paths.data("curated")
FEATURES = CORPUS / "features"
MODELS = paths.ckpt("probes", "tag_probes")
FEATURE_ARCHIVES = ("clamp3_features.npz", "aria_features.npz",
                    "symbolic_features.npz")
GATES = {
    "emotion": 0.7319,
    "genre": 0.3968,
    "accompaniment": 0.8097099255483835,
    "performance": 0.9082847094342009,
}
BACKUP = CORPUS / "archives" / "consolidation_backup"
# Gates fill empty slots and update provenance; overriding an existing value on
# evidence-route rows uses a top-confidence band for performance (0.9982 min-axis
# confidence, about 7.5% of those pieces). Accompaniment keeps one rule.
SCOPES = {
    "audio_route": {
        "fields": ("emotion", "genre", "accompaniment", "performance"),
        "override_gates": {},
    },
    "evidence_route": {
        "fields": ("accompaniment", "performance"),
        "override_gates": {"performance": 0.9982},
    },
}
# Pre-consolidation values for every scoped piece-field; this is the revert reference.
ORIGINALS = CORPUS / "archives" / "pre_consolidation_originals.jsonl"
PERFORMANCE_DECODE = {"flexible": 0.45, "detached": 0.75}


def scoped_pieces(scope: str) -> dict[str, Path]:
    pieces = {}
    for label_path in sorted(CORPUS.glob("pieces/*/label.json")):
        record = json.loads(label_path.read_text())
        is_audio_route = str(record.get("revision", "")).startswith("gemini35_flash")
        if is_audio_route == (scope == "audio_route"):
            pieces[record.get("work_id") or label_path.parent.name] = label_path
    return pieces


def originals() -> dict[str, dict]:
    """Pre-consolidation accepted_fields / field_sources, keyed by work_id."""
    return {row["work_id"]: row for row in
            (json.loads(line) for line in ORIGINALS.read_text().splitlines())}


def aligned_features(work_ids: list[str]) -> np.ndarray:
    manifest = pq.read_table(
        FEATURES / "manifest.parquet", columns=["piece_id", "work_id"]
    ).to_pydict()
    work_to_piece = dict(zip(
        (str(v) for v in manifest["work_id"]),
        (str(v) for v in manifest["piece_id"]),
    ))
    piece_ids = [work_to_piece[w] for w in work_ids]
    blocks = []
    for archive in FEATURE_ARCHIVES:
        data = np.load(FEATURES / archive, allow_pickle=True)
        by_id = {str(i): k for k, i in enumerate(data["ids"])}
        blocks.append(np.asarray(data["features"])[[by_id[p] for p in piece_ids]])
    return np.concatenate(blocks, axis=1)


def probe_predictions(x: np.ndarray,
                      fields: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for field in [f for f in ("emotion", "accompaniment", "genre") if f in fields]:
        model = joblib.load(MODELS / f"{field}.joblib")
        proba = model.predict_proba(x)
        out[field] = (model.classes_[np.argmax(proba, axis=1)], proba.max(axis=1))
    expr = joblib.load(MODELS / "performance_expression.joblib")
    artic = joblib.load(MODELS / "performance_articulation.joblib")
    p_flex = expr.predict_proba(x)[:, list(expr.classes_).index("flexible")]
    p_det = artic.predict_proba(x)[:, list(artic.classes_).index("detached")]
    joint = np.char.add(np.char.add(
        np.where(p_flex >= PERFORMANCE_DECODE["flexible"], "flexible", "steady"),
        " "),
        np.where(p_det >= PERFORMANCE_DECODE["detached"], "detached", "connected"))
    confidence = np.minimum(np.maximum(p_flex, 1 - p_flex),
                            np.maximum(p_det, 1 - p_det))
    out["performance"] = (joint, confidence)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=sorted(SCOPES), default="audio_route",
                        help="which curated annotation route to consolidate")
    parser.add_argument("--execute", action="store_true",
                        help="apply changes (default is a dry-run count)")
    args = parser.parse_args()

    fields = SCOPES[args.scope]["fields"]
    override_gates = SCOPES[args.scope]["override_gates"]
    pieces = scoped_pieces(args.scope)
    work_ids = sorted(pieces)
    print(f"{args.scope}-scope pieces: {len(work_ids)}")
    x = aligned_features(work_ids)
    predictions = probe_predictions(x, fields)

    original_records = originals()
    plan: dict[str, dict[str, dict]] = {}
    counts = {f: {"relabel": 0, "fill": 0, "provenance_only": 0, "revert": 0,
                  "unchanged": 0} for f in fields}
    for row, work_id in enumerate(work_ids):
        record = json.loads(pieces[work_id].read_text())
        current_fields = record.get("accepted_fields") or {}
        current_sources = record.get("field_sources") or {}
        original_fields = original_records[work_id]["original_fields"]
        original_sources = original_records[work_id]["original_field_sources"]
        changes = {}
        for field in fields:
            predicted, confidence = predictions[field]
            new = str(predicted[row])
            original = str(original_fields.get(field, "")).strip()
            current = str(current_fields.get(field, "")).strip()
            if not original:
                required = GATES[field]
                action = "fill"
            elif new == current:
                required = GATES[field]
                action = "provenance_only"
            else:
                required = override_gates.get(field, GATES[field])
                action = "relabel"
            if float(confidence[row]) >= required and new != original:
                target = new
                source = ("probe_fill" if action == "fill"
                          else "probe_consolidation")
            else:
                action = "revert" if current != original else "unchanged"
                target, source = original, original_sources.get(field)
            if target == current and current_sources.get(field) == source:
                action = "unchanged"
            counts[field][action] += 1
            if action != "unchanged":
                changes[field] = {
                    "action": action, "previous": current, "original": original,
                    "new": target, "source": source,
                    "confidence": float(confidence[row]),
                }
        if changes:
            plan[work_id] = changes

    print("per-field actions:", json.dumps(counts, indent=1),
          "| pieces touched:", len(plan))
    if not args.execute:
        print("dry run only; rerun with --execute to apply")
        return

    BACKUP.mkdir(parents=True, exist_ok=True)
    for work_id, changes in plan.items():
        label_path = pieces[work_id]
        destination = BACKUP / work_id
        destination.mkdir()
        shutil.copy2(label_path, destination / "label.json")
        record = json.loads(label_path.read_text())
        sources = record.setdefault("field_sources", {})
        for field, change in changes.items():
            record["accepted_fields"][field] = change["new"]
            if change["source"] is None:
                sources.pop(field, None)
            else:
                sources[field] = change["source"]
        record["consolidation"] = {
            "models": "tag_probes",
            "gates": {f: GATES[f] for f in fields},
            "note": "prompt string reflects pre-consolidation fields",
            "changes": changes,
        }
        label_path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    summary = {
        "scope": args.scope,
        "pieces_scanned": len(work_ids),
        "pieces_modified": len(plan),
        "per_field_actions": counts,
        "gates": {f: GATES[f] for f in fields},
        "models": str(MODELS.relative_to(paths.CKPT_ROOT)),
    }
    (BACKUP / f"summary_{args.scope}.json").write_text(json.dumps(summary, indent=2))
    print("applied; backups at", BACKUP)


if __name__ == "__main__":
    main()
