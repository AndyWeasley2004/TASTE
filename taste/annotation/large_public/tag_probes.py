"""Attribute probes and imagery retrieval from CLaMP 3, Aria and MIDI features.

fit trains probes and gates; apply labels the full deduplicated corpus.
relabel-attributes preserves existing imagery, key and tempo; apply replaces
imagery with retrieved captions. Gated-out values retain their raw predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from taste import paths
from taste.representation.midi_features import tempo_class


def fit_probe(x, y):
    """Median-impute, standardize, class-balanced logistic regression."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1500, class_weight="balanced", random_state=0),
    )
    model.fit(x, y)
    return model


def normalized(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def out_of_fold(x, y, groups, n_splits=5):
    """Group-aware out-of-fold probability matrix aligned to x rows."""
    classes = np.unique(y)
    class_index = {label: i for i, label in enumerate(classes)}
    probabilities = np.full((len(y), len(classes)), np.nan)
    for train_fold, held in GroupKFold(n_splits=n_splits).split(x, y, groups):
        model = fit_probe(x[train_fold], y[train_fold])
        block = model.predict_proba(x[held])
        for j, label in enumerate(model.classes_):
            probabilities[held, class_index[label]] = block[:, j]
    return classes, probabilities


def aligned_text(path: Path, manifest_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Caption embeddings aligned to manifest order; mask marks availability."""
    data = np.load(path, allow_pickle=True)
    by_id = {str(piece_id): index for index, piece_id in enumerate(data["ids"])}
    dim = data["features"].shape[1]
    x = np.zeros((len(manifest_ids), dim), dtype=np.float32)
    mask = np.zeros(len(manifest_ids), dtype=bool)
    features = np.asarray(data["features"], dtype=np.float32)
    for row, piece_id in enumerate(manifest_ids):
        index = by_id.get(piece_id)
        if index is not None:
            x[row] = features[index]
            mask[row] = True
    return normalized(x), mask


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge W minimizing ||xW - y||^2 + alpha||W||^2."""
    d = x.shape[1]
    gram = x.T @ x + alpha * np.eye(d, dtype=np.float64)
    return np.linalg.solve(gram, x.T @ y).astype(np.float32)

EXPERIMENT = paths.data("large_public_annotation")
ROOT = paths.data("aria_midi", "deduped")
MODELS = paths.ckpt("probes", "tag_probes")
EXPECTED = 371_053
FEATURE_ORDER = ("clamp3_c2", "aria", "symbolic")
# Every probe is pruned; the axes carry a `performance_` file-name prefix.
PRUNED_FIELDS = ("emotion", "genre", "accompaniment", "expression", "articulation")
PROBE_NAMES = {
    "emotion": "emotion",
    "genre": "genre",
    "accompaniment": "accompaniment",
    "expression": "performance_expression",
    "articulation": "performance_articulation",
}
PERFORMANCE_THRESHOLDS = {"flexible": 0.45, "detached": 0.75}
GATE_PRECISION = 0.6
MIN_GATE_SUPPORT = 50
# Gate policy: min(0.6-precision threshold, threshold retaining this share of
# the production corpus). The floor half is resolved after corpus scoring.
COVERAGE_FLOOR = 0.75
GATED_FIELDS = ("emotion", "genre", "accompaniment", "performance")
METHOD_MANIFEST = json.dumps({
    "emotion": "all_features_v3_pruned+floor_gated",
    "genre": "aria_metadata_then_all_features_v3_pruned+floor_gated",
    "accompaniment": "all_features_v3_pruned+floor_gated",
    "performance": "all_features_v3_pruned+thresholds+floor_gated",
    "imagery": "hybrid_caption_space_v2",
}, sort_keys=True)
RIDGE_ALPHA = 1.0
STYLE_TOP_K = 50
HYBRID_CAPTION_W = 0.6
HYBRID_AFFECT_W = 0.3


def parse_named(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name:
            raise ValueError(f"expected NAME=PATH, found {value!r}")
        result[name] = Path(path)
    return result


def align(path: Path, ids: list[str]) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    by_id = {str(piece_id): index for index, piece_id in enumerate(data["ids"])}
    missing = [piece_id for piece_id in ids if piece_id not in by_id]
    if missing:
        raise ValueError(f"{path} missing {len(missing)} IDs")
    return np.asarray(data["features"])[[by_id[piece_id] for piece_id in ids]]


def concat_features(paths: dict[str, Path], ids: list[str]) -> np.ndarray:
    missing = [name for name in FEATURE_ORDER if name not in paths]
    if missing:
        raise ValueError(f"missing feature families: {missing}")
    return np.concatenate([align(paths[name], ids) for name in FEATURE_ORDER], axis=1)


def available_mask(values: list, indices: np.ndarray) -> np.ndarray:
    return indices[np.asarray([
        bool(str(values[index] or "").strip()) for index in indices
    ])]


def confident_prune(x, labels, groups, indices) -> np.ndarray:
    """Return the kept subset of `indices` after confident-learning pruning."""
    y = labels[indices].astype(str)
    classes, oof = out_of_fold(x[indices], y, groups[indices].astype(str))
    positions = np.asarray([list(classes).index(v) for v in y])
    p_label = oof[np.arange(len(y)), positions]
    argmax = classes[np.nanargmax(oof, axis=1)]
    self_confidence = {c: float(np.nanmean(p_label[y == c])) for c in classes}
    thresholds = np.asarray([self_confidence[v] for v in y])
    prune = (argmax != y) & (p_label < thresholds)
    return indices[~prune]


def gate_threshold(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Smallest threshold reaching GATE_PRECISION with maximal coverage.

    Scans the observed confidence values: pruned-probe confidences concentrate
    near the top of their range, where a fixed grid is too coarse.
    """
    for threshold in np.unique(np.round(confidence, 4)):
        keep = confidence >= threshold
        if keep.sum() < MIN_GATE_SUPPORT:
            break
        if correct[keep].mean() >= GATE_PRECISION:
            return float(threshold)
    raise ValueError(f"no threshold reaches precision {GATE_PRECISION}")


def coverage_floor_threshold(confidence: np.ndarray, coverage: float) -> float:
    """Smallest threshold retaining at least `coverage` of `confidence`."""
    return float(np.quantile(confidence, 1.0 - coverage, method="lower"))


def softmax_gate(x, manifest, groups, train, val, field: str) -> float:
    """0.6-precision threshold for a softmax field, via a train-side twin."""
    labels = np.asarray(manifest[field], dtype=object)
    kept = confident_prune(x, labels, groups, available_mask(labels, train))
    probe = fit_probe(x[kept], labels[kept].astype(str))
    val_idx = available_mask(labels, val)
    probabilities = probe.predict_proba(x[val_idx])
    predicted = np.asarray(probe.classes_)[np.argmax(probabilities, axis=1)]
    return gate_threshold(probabilities.max(axis=1),
                          predicted == labels[val_idx].astype(str))


def fit_models(benchmark_paths: dict[str, Path], text_path: Path,
               benchmark_manifest: Path) -> None:
    manifest = pq.read_table(benchmark_manifest).to_pydict()
    ids = [str(v) for v in manifest["piece_id"]]
    x = concat_features(benchmark_paths, ids)
    groups = np.asarray(manifest["split_group"], dtype=object)
    split = np.asarray(manifest["split"], dtype=object)
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "val")
    everything = np.arange(len(ids))
    MODELS.mkdir(parents=True, exist_ok=True)
    meta: dict = {"gate_precision": GATE_PRECISION, "pruning": {}, "gates": {}}

    for field in PRUNED_FIELDS:
        labels = np.asarray(manifest[field], dtype=object)
        full_available = available_mask(labels, everything)
        kept = confident_prune(x, labels, groups, full_available)
        model = fit_probe(x[kept], labels[kept].astype(str))
        joblib.dump(model, MODELS / f"{PROBE_NAMES[field]}.joblib")
        meta["pruning"][field] = {
            "available": int(len(full_available)),
            "kept": int(len(kept)),
        }
        print(field, "pruned", len(full_available) - len(kept), "of",
              len(full_available), flush=True)

    # --- gate calibration on the val split with train-side twins ---
    for field in ("emotion", "genre", "accompaniment"):
        meta["gates"][field] = softmax_gate(x, manifest, groups, train, val, field)

    expression = np.asarray(manifest["expression"], dtype=object)
    articulation = np.asarray(manifest["articulation"], dtype=object)
    performance = np.asarray(manifest["performance"], dtype=object)
    expr_kept = confident_prune(x, expression, groups,
                                available_mask(expression, train))
    expr_probe = fit_probe(x[expr_kept], expression[expr_kept].astype(str))
    artic_kept = confident_prune(x, articulation, groups,
                                 available_mask(articulation, train))
    artic_probe = fit_probe(x[artic_kept], articulation[artic_kept].astype(str))
    val_perf = available_mask(performance, val)
    p_flex = expr_probe.predict_proba(x[val_perf])[
        :, list(expr_probe.classes_).index("flexible")]
    p_det = artic_probe.predict_proba(x[val_perf])[
        :, list(artic_probe.classes_).index("detached")]
    joint = np.char.add(np.char.add(
        np.where(p_flex >= PERFORMANCE_THRESHOLDS["flexible"], "flexible", "steady"),
        " "),
        np.where(p_det >= PERFORMANCE_THRESHOLDS["detached"], "detached", "connected"))
    confidence = np.minimum(np.maximum(p_flex, 1 - p_flex),
                            np.maximum(p_det, 1 - p_det))
    meta["gates"]["performance"] = gate_threshold(
        confidence, joint == performance[val_perf])

    # --- imagery: ridge projection + candidate table ---
    aria_x = normalized(align(benchmark_paths["aria"], ids))
    text_x, text_mask = aligned_text(text_path, ids)
    pair_idx = np.flatnonzero(text_mask)
    weights = fit_ridge(aria_x[pair_idx].astype(np.float64),
                        text_x[pair_idx].astype(np.float64), RIDGE_ALPHA)
    candidates = np.asarray([
        i for i in range(len(ids))
        if bool(str(manifest["imagery"][i] or "").strip()) and text_mask[i]
    ])
    np.savez_compressed(
        MODELS / "imagery_retrieval.npz",
        ridge_weights=weights,
        candidate_ids=np.asarray([ids[i] for i in candidates], dtype=object),
        candidate_captions=np.asarray(
            [str(manifest["imagery"][i]) for i in candidates], dtype=object),
        candidate_aria=aria_x[candidates],
        candidate_text=text_x[candidates],
    )
    meta["imagery"] = {
        "ridge_alpha": RIDGE_ALPHA, "candidates": int(len(candidates)),
        "style_top_k": STYLE_TOP_K,
        "caption_w": HYBRID_CAPTION_W, "affect_w": HYBRID_AFFECT_W,
    }
    meta["gate_policy"] = {
        "precision": GATE_PRECISION, "coverage_floor": COVERAGE_FLOOR,
        "rule": "min(0.6-precision threshold, production 75%-coverage threshold)",
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def score_attributes(x: np.ndarray) -> tuple[dict, dict]:
    """Ungated predictions and confidences for the four attribute fields."""
    predictions, confidences = {}, {}
    for field in ("emotion", "genre", "accompaniment"):
        model = joblib.load(MODELS / f"{field}.joblib")
        probabilities = model.predict_proba(x)
        predictions[field] = np.asarray(model.classes_)[
            np.argmax(probabilities, axis=1)].astype(str)
        confidences[field] = probabilities.max(axis=1)

    expr_model = joblib.load(MODELS / "performance_expression.joblib")
    artic_model = joblib.load(MODELS / "performance_articulation.joblib")
    p_flex = expr_model.predict_proba(x)[
        :, list(expr_model.classes_).index("flexible")]
    p_det = artic_model.predict_proba(x)[
        :, list(artic_model.classes_).index("detached")]
    predictions["performance"] = np.char.add(np.char.add(
        np.where(p_flex >= PERFORMANCE_THRESHOLDS["flexible"], "flexible", "steady"),
        " "),
        np.where(p_det >= PERFORMANCE_THRESHOLDS["detached"], "detached", "connected")
    ).astype(str)
    confidences["performance"] = np.minimum(
        np.maximum(p_flex, 1 - p_flex), np.maximum(p_det, 1 - p_det))
    return predictions, confidences


def resolve_gates(confidences: dict, precision_gates: dict,
                  metadata_rows: int, total: int) -> dict:
    """Stage two of the gate policy: lower a field to the coverage floor.

    Genre counts its Aria-metadata rows as retained, so only the probe-scored
    remainder has to clear the threshold.
    """
    resolved = {}
    for field in GATED_FIELDS:
        confidence = confidences[field]
        precision_threshold = float(precision_gates[field])
        if field == "genre":
            needed = (COVERAGE_FLOOR * total - metadata_rows) / len(confidence)
            floor = (coverage_floor_threshold(confidence, needed)
                     if needed > 0 else float(confidence.min()))
        else:
            floor = coverage_floor_threshold(confidence, COVERAGE_FLOOR)
        applied = min(precision_threshold, floor)
        resolved[field] = {
            "precision_threshold": precision_threshold,
            "coverage_floor_threshold": floor,
            "applied": float(applied),
            "binding": ("0.6_precision" if applied == precision_threshold
                        else "coverage_floor"),
        }
    return resolved


def relabel_attributes(deduped_paths: dict[str, Path], deduped_manifest: Path,
                       labels_path: Path, backup_path: Path) -> None:
    """Rebuild the four attribute fields in place on an existing labels.parquet.

    Imagery, key and tempo columns are carried over untouched: the shipped
    captions are rewrites that the full `apply` path would overwrite.
    """
    meta = json.loads((MODELS / "meta.json").read_text())
    table = pq.read_table(labels_path)
    target = pq.read_table(deduped_manifest).to_pydict()
    ids = [str(v) for v in target["piece_id"]]
    if len(ids) != EXPECTED or len(set(ids)) != EXPECTED:
        raise ValueError(f"expected {EXPECTED:,} unique deduped IDs, found {len(ids):,}")
    if table.column("piece_id").to_pylist() != ids:
        raise ValueError("labels.parquet is not in deduped-manifest order")
    if backup_path.exists():
        raise ValueError(f"{backup_path} already exists")
    pq.write_table(table, backup_path, compression="zstd")

    x = concat_features(deduped_paths, ids)
    predictions, confidences = score_attributes(x)

    mapped_genre = [str(v or "").strip() or None for v in target["mapped_genre"]]
    metadata = np.asarray([value is not None for value in mapped_genre])
    predictions["genre"] = np.where(
        metadata, [v or "" for v in mapped_genre], predictions["genre"])
    confidences["genre"] = np.where(metadata, 1.0, confidences["genre"])

    gates = resolve_gates(confidences, meta["gates"], int(metadata.sum()), len(ids))
    print(json.dumps(gates, indent=2), flush=True)

    previous = {field: table.column(field).to_pylist() for field in GATED_FIELDS}
    columns = {}
    changed = {}
    for field in GATED_FIELDS:
        keep = confidences[field] >= gates[field]["applied"]
        gated = np.where(keep, predictions[field], "")
        columns[field] = gated.tolist()
        columns[f"{field}_raw"] = predictions[field].tolist()
        columns[f"{field}_confidence"] = confidences[field].tolist()
        changed[field] = {
            "coverage": float(keep.mean()),
            "gated_rows": int((~keep).sum()),
            "changed_vs_backup": int(sum(
                1 for old, new in zip(previous[field], columns[field])
                if str(old) != new)),
            "distribution": {
                str(value): int(count) for value, count in
                zip(*np.unique(gated[keep], return_counts=True))},
        }
    columns["genre_source"] = [
        "aria_metadata" if flag else "probe:all_features_v3" for flag in metadata]
    columns["method_manifest"] = [METHOD_MANIFEST] * len(ids)

    for name, values in columns.items():
        array = pa.array(values)
        if name in table.schema.names:
            table = table.set_column(table.schema.get_field_index(name), name, array)
        else:
            table = table.append_column(name, array)

    temporary = labels_path.with_name(labels_path.name + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(labels_path)

    verification = {
        "rows": table.num_rows,
        "gates": gates,
        "fields": changed,
        "genre_metadata_rows": int(metadata.sum()),
        "models": str(MODELS),
        "backup": str(backup_path),
    }
    output_dir = labels_path.parent
    (output_dir / "verification_v3_attributes.json").write_text(
        json.dumps(verification, indent=2))
    (output_dir / "meta.json").write_text(json.dumps(
        {"fit_meta": meta, "verification_summary": verification}, indent=2))
    print(json.dumps(verification, indent=2))


def hybrid_imagery(query_aria: np.ndarray, query_affect: np.ndarray,
                      bundle: dict, candidate_affect: np.ndarray,
                      device: str, batch_size: int):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    candidate_aria = torch.from_numpy(bundle["candidate_aria"]).to(device)
    candidate_text = torch.from_numpy(bundle["candidate_text"]).to(device)
    candidate_affect_t = torch.from_numpy(candidate_affect).to(device)
    captions, sources, style_cos, cap_scores, affect_scores = [], [], [], [], []
    for start in range(0, len(query_aria), batch_size):
        aria_block = torch.from_numpy(query_aria[start:start + batch_size]).to(device)
        affect_block = torch.from_numpy(query_affect[start:start + batch_size]).to(device)
        projected = aria_block @ torch.from_numpy(bundle["ridge_weights"]).to(device)
        projected = projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-8)
        style = aria_block @ candidate_aria.T
        top_cos, top_idx = torch.topk(style, STYLE_TOP_K, dim=1)
        caption_term = torch.einsum(
            "qd,qkd->qk", projected, candidate_text[top_idx])
        affect_term = torch.einsum(
            "qd,qkd->qk", affect_block, candidate_affect_t[top_idx])
        combined = HYBRID_CAPTION_W * caption_term + HYBRID_AFFECT_W * affect_term
        pick = torch.argmax(combined, dim=1, keepdim=True)
        chosen = torch.gather(top_idx, 1, pick)[:, 0]
        for row in range(len(chosen)):
            offset = int(chosen[row])
            captions.append(str(bundle["candidate_captions"][offset]))
            sources.append(str(bundle["candidate_ids"][offset]))
        style_cos.extend(
            torch.gather(top_cos, 1, pick)[:, 0].clamp(-1, 1).cpu().tolist())
        cap_scores.extend(torch.gather(caption_term, 1, pick)[:, 0].cpu().tolist())
        affect_scores.extend(torch.gather(affect_term, 1, pick)[:, 0].cpu().tolist())
    return captions, sources, style_cos, cap_scores, affect_scores


def apply_models(deduped_paths: dict[str, Path], deduped_manifest: Path,
                 tempo_features: Path, output_dir: Path,
                 device: str, batch_size: int) -> None:
    meta = json.loads((MODELS / "meta.json").read_text())
    target = pq.read_table(deduped_manifest).to_pydict()
    ids = [str(v) for v in target["piece_id"]]
    relpaths = [str(v) for v in target["midi_relpath"]]
    if len(ids) != EXPECTED or len(set(ids)) != EXPECTED:
        raise ValueError(f"expected {EXPECTED:,} unique deduped IDs, found {len(ids):,}")
    x = concat_features(deduped_paths, ids)

    predictions, confidence_arrays = score_attributes(x)

    mapped_genre = [str(v or "").strip() or None for v in target["mapped_genre"]]
    metadata = np.asarray([value is not None for value in mapped_genre])
    predictions["genre"] = np.where(
        metadata, [v or "" for v in mapped_genre], predictions["genre"])
    confidence_arrays["genre"] = np.where(metadata, 1.0, confidence_arrays["genre"])
    genre_source = ["aria_metadata" if flag else "probe:all_features_v3"
                    for flag in metadata]

    gates = resolve_gates(confidence_arrays, meta["gates"],
                          int(metadata.sum()), len(ids))
    labels: dict[str, list[str]] = {}
    raw: dict[str, list[str]] = {}
    confidences: dict[str, list[float]] = {}
    for field in GATED_FIELDS:
        keep = confidence_arrays[field] >= gates[field]["applied"]
        labels[field] = np.where(keep, predictions[field], "").tolist()
        raw[field] = predictions[field].tolist()
        confidences[field] = confidence_arrays[field].tolist()

    tempo_data = np.load(tempo_features, allow_pickle=True)
    tempo_names = [str(v) for v in tempo_data["feature_names"]]
    tempo_x = align(tempo_features, ids)
    tempo_bpms = tempo_x[:, tempo_names.index("tempo_bpm")]
    tempo_valid = np.isfinite(tempo_bpms)
    tempos = [tempo_class(float(b)) if ok else "moderate"
              for b, ok in zip(tempo_bpms, tempo_valid)]

    bundle = dict(np.load(MODELS / "imagery_retrieval.npz", allow_pickle=True))
    emotion_model = joblib.load(MODELS / "emotion.joblib")
    query_affect = normalized(emotion_model.predict_proba(x))
    candidate_x_bench = concat_features(
        {name: EXPERIMENT / archive for name, archive in (
            ("clamp3_c2", "clamp3_features.npz"),
            ("aria", "aria_features.npz"),
            ("symbolic", "symbolic_features.npz"))},
        [str(v) for v in bundle["candidate_ids"]])
    candidate_affect = normalized(emotion_model.predict_proba(candidate_x_bench))
    query_aria = normalized(align(deduped_paths["aria"], ids))
    captions, sources, style_cos, cap_scores, affect_scores = hybrid_imagery(
        query_aria, query_affect, bundle, candidate_affect, device, batch_size)

    key_table = ds.dataset(
        sorted((ROOT / "key_estimates").glob("part-*.parquet")),
        format="parquet",
    ).to_table(columns=["midi_relpath", "key"])
    keys = dict(zip(key_table["midi_relpath"].to_pylist(),
                    key_table["key"].to_pylist()))
    missing = [p for p in relpaths if p not in keys]
    if missing:
        raise ValueError(f"missing key estimates for {len(missing)} pieces")

    rows = []
    for index, (piece_id, relpath) in enumerate(zip(ids, relpaths)):
        rows.append({
            "piece_id": piece_id,
            "midi_relpath": relpath,
            "emotion": labels["emotion"][index],
            "emotion_raw": raw["emotion"][index],
            "genre": labels["genre"][index],
            "genre_raw": raw["genre"][index],
            "accompaniment": labels["accompaniment"][index],
            "accompaniment_raw": raw["accompaniment"][index],
            "performance": labels["performance"][index],
            "performance_raw": raw["performance"][index],
            "imagery": captions[index],
            "imagery_retrieval": captions[index],
            "imagery_source_piece_id": sources[index],
            "imagery_style_cosine": style_cos[index],
            "imagery_caption_space_score": cap_scores[index],
            "imagery_affect_score": affect_scores[index],
            "key": keys[relpath],
            "tempo": tempos[index],
            "tempo_bpm": float(tempo_bpms[index]) if tempo_valid[index] else None,
            "emotion_confidence": confidences["emotion"][index],
            "genre_confidence": confidences["genre"][index],
            "accompaniment_confidence": confidences["accompaniment"][index],
            "performance_confidence": confidences["performance"][index],
            "genre_source": genre_source[index],
            "method_manifest": METHOD_MANIFEST,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "labels.parquet"
    temporary = label_path.with_name(label_path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(label_path)

    verification = {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "exact_manifest_order": ids == [str(v) for v in target["piece_id"]],
        "coverage": {
            field: 1.0 - sum(1 for v in labels[field] if not v) / len(rows)
            for field in GATED_FIELDS
        },
        "genre_metadata_rows": genre_source.count("aria_metadata"),
        "finite_confidences": bool(np.isfinite(np.asarray(
            [confidences[f] for f in ("emotion", "genre", "accompaniment",
                                      "performance")])).all()),
        "tempo_estimated_rows": int(tempo_valid.sum()),
        "gates": gates,
    }
    (output_dir / "meta.json").write_text(json.dumps(
        {"fit_meta": meta, "verification_summary": verification}, indent=2))
    (output_dir / "verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--benchmark-features", action="append", required=True)
    fit.add_argument("--text-features", type=Path,
                     default=EXPERIMENT / "clamp3_text_features.npz")
    fit.add_argument("--benchmark", type=Path,
                     default=EXPERIMENT / "benchmark.parquet")
    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--features", action="append", required=True)
    apply_parser.add_argument("--deduped-manifest", type=Path,
                              default=EXPERIMENT / "deduped_manifest.parquet")
    apply_parser.add_argument("--tempo-features", type=Path, required=True)
    apply_parser.add_argument("--output-dir", type=Path,
                              default=ROOT / "large_public_labels")
    apply_parser.add_argument("--retrieval-device", default="cuda")
    apply_parser.add_argument("--retrieval-batch-size", type=int, default=1024)
    relabel = sub.add_parser(
        "relabel-attributes",
        help="rebuild the four attribute fields in place, keeping imagery/key/tempo")
    relabel.add_argument("--features", action="append", required=True)
    relabel.add_argument("--deduped-manifest", type=Path,
                         default=EXPERIMENT / "deduped_manifest.parquet")
    relabel.add_argument("--labels", type=Path,
                         default=ROOT / "large_public_labels" / "labels.parquet")
    relabel.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit_models(parse_named(args.benchmark_features), args.text_features,
                   args.benchmark)
    elif args.command == "relabel-attributes":
        relabel_attributes(parse_named(args.features), args.deduped_manifest,
                           args.labels, args.backup)
    else:
        apply_models(parse_named(args.features), args.deduped_manifest,
                     args.tempo_features, args.output_dir,
                     args.retrieval_device, args.retrieval_batch_size)


if __name__ == "__main__":
    main()
