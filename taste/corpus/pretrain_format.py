"""On-disk contract of the pretraining corpus.

Layout under ``data_dir``: ``meta.json`` plus ``epoch0/`` .. ``epoch4/``
(``train-*.parquet``) and ``val/`` (``val-*.parquet``). Rows are one piece or
one section-aligned crop of a piece, capped at ``SEQ_LEN``.
"""
from __future__ import annotations

import pyarrow as pa

FORMAT_VERSION = "taste_4k_pretrain_25ms"
SEQ_LEN = 4096
N_EPOCHS = 5

ROW_SCHEMA = pa.schema([
    pa.field("input_ids", pa.list_(pa.int32())),
    pa.field("n_tokens", pa.int32()),
    pa.field("epoch_idx", pa.int8()),
])
