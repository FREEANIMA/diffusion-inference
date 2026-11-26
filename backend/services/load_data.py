import os
import pandas as pd
from src.core import state

def load_tags() -> list[dict]:
    path = os.path.join(state.config.CKPT_LOCAL_DIR, state.config.TAG_TYPE_FILENAME)
    df = pd.read_parquet(path, columns=["tag", "type", "dup", "count"])
    df = df.loc[df["count"] > 100].sort_values("count", ascending=False)
    df["dup"] = df["dup"].fillna("")
    return df.to_dict("records")      # [{tag, type, dup, count}, ...]