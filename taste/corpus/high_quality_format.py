"""On-disk contract of the high-quality SFT corpus.

Layout under ``data_dir``: ``meta.json`` plus ``pass0/``..``pass4/``
(``train-*.parquet``) and ``val/`` (``val-*.parquet``), with a mirrored
``index/`` tree. The token row, context length, prompt fields and prefix-drop
probability are those of ``large_public_format``; the rows carry no
``stratum`` column and every pass is one uniformly shuffled block.
"""
from __future__ import annotations

import pyarrow as pa

from taste.corpus.large_public_format import (  # noqa: F401  (re-exported)
    PREFIX_DROP_P,
    PROMPT_FIELDS,
    SEQ_LEN,
)

FORMAT_VERSION = "taste_4k_high_quality_25ms"
N_PASSES = 5

ROW_SCHEMA = pa.schema([
    pa.field("input_ids", pa.list_(pa.int32())),
    pa.field("n_tokens", pa.int32()),
    pa.field("pass_idx", pa.int8()),
    pa.field("prefix_dropped", pa.bool_()),
    *(pa.field(name, pa.string()) for name in PROMPT_FIELDS),
])
