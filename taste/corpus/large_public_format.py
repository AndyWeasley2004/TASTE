"""On-disk contract of the large-public SFT corpus.

Layout under ``data_dir``: ``meta.json`` plus ``pass0/``..``pass5/``
(``train-*.parquet``) and ``val/`` (``val-*.parquet``), with a mirrored
``index/`` tree. Rows are one piece each, capped at ``SEQ_LEN``.

The token row is the pretraining grammar (see ``taste.corpus.pretrain_format``).
The text prompt fields ride along the row as raw strings and are tokenized by
the training-time collator; ``prefix_dropped`` marks the rows whose prompt the
collator discards (classifier-free prompt dropout).
"""
from __future__ import annotations

import pyarrow as pa

FORMAT_VERSION = "taste_4k_large_public_25ms"
SEQ_LEN = 4096
N_PASSES = 6
PREFIX_DROP_P = 0.10

PROMPT_FIELDS = ("emotion", "genre", "accompaniment", "performance", "imagery")

ROW_SCHEMA = pa.schema([
    pa.field("input_ids", pa.list_(pa.int32())),
    pa.field("n_tokens", pa.int32()),
    pa.field("pass_idx", pa.int8()),
    # 1 = emotion only, 3 = emotion + accompaniment + performance, 2 = the rest
    pa.field("stratum", pa.int8()),
    pa.field("prefix_dropped", pa.bool_()),
    *(pa.field(name, pa.string()) for name in PROMPT_FIELDS),
])
