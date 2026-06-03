"""POC: convert a structure.json to the Zarr results store and compare.

Usage:
    python _bench/poc_zarr_store.py [path/to/structure.json]

Reports JSON-vs-Zarr size, write time, verifies an exact round-trip, and
demonstrates the Results accessor ergonomics.
"""
import os
import shutil
import sys
import time

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sarcasm.ioutils import IOUtils
from sarcasm.results_store import Results, write_results

DEFAULT = "test_data/high_speed_single_ACTN2-citrine_CM/30kPa/data/structure.json"


def du(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(path) for f in fs)


def equal(a, b):
    """Deep equality tolerant of NaN, arrays, ragged lists, scalars, sparse."""
    if sparse.issparse(a) or sparse.issparse(b):
        a = a.toarray() if sparse.issparse(a) else np.asarray(a)
        b = b.toarray() if sparse.issparse(b) else np.asarray(b)
        return a.shape == b.shape and np.allclose(a, b, equal_nan=True)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            return np.allclose(a.astype(float), b.astype(float), equal_nan=True)
        return np.array_equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (np.isnan(a) and np.isnan(b))
    return a == b


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print(f"source: {src}")
    json_bytes = du(src)
    print(f"  structure.json: {json_bytes / 1e6:8.1f} MB")

    t = time.time()
    data = IOUtils.json_deserialize(src)
    print(f"  json load:      {time.time() - t:8.2f} s   ({len(data)} keys)")

    store = os.path.join(os.path.dirname(src), "data.zarr")
    if os.path.exists(store):
        shutil.rmtree(store)
    t = time.time()
    write_results(data, store)
    write_s = time.time() - t
    zarr_bytes = du(store)
    print(f"  -> zarr write:  {write_s:8.2f} s")
    print(f"  analysis.zarr:  {zarr_bytes / 1e6:8.1f} MB   "
          f"({json_bytes / max(zarr_bytes, 1):.1f}x smaller)")

    # ---- exact round-trip on every key ---- #
    t = time.time()
    r = Results(store)
    print(f"  Results open:   {time.time() - t:8.4f} s (lazy)")
    bad = [k for k in data if not equal(data[k], r[k])]
    print(f"  round-trip:     {'OK — all keys match' if not bad else f'MISMATCH: {bad[:8]}'}"
          f"  ({len(data)} keys)")

    # ---- lazy single-array load ---- #
    arr_keys = [k for k in data if isinstance(data[k], np.ndarray) and data[k].size > 1000]
    if arr_keys:
        k = max(arr_keys, key=lambda k: data[k].size)
        r2 = Results(store)
        t = time.time()
        _ = r2[k]
        print(f"  lazy load 1 array {k!r} {data[k].shape}: {time.time() - t:.4f} s "
              f"(vs {json_bytes/1e6:.0f} MB full json reparse)")

    # ---- duplication audit (the motionfield_tracker_* aliases) ---- #
    dup = sum(data[k].nbytes for k in data
              if k.startswith("motionfield_tracker_") and isinstance(data[k], np.ndarray))
    if dup:
        print(f"  alias dup waste: {dup / 1e6:.1f} MB of motionfield_tracker_* duplicates raw")

    # ---- ergonomics demo (native group tree) ---- #
    print("\n  ergonomics (native group tree, attribute access):")
    print(f"    dir(r)                 = {dir(r)}")
    for grp in ("tracks", "motion", "structure", "params"):
        try:
            print(f"    r.{grp:<10} -> {getattr(r, grp)!r}")
        except AttributeError:
            pass
    try:
        print(f"    r.structure.sarcomere  -> {r.structure.sarcomere!r}")
    except AttributeError:
        pass
    # a concrete leaf, whichever exists; show attr == legacy flat key
    for attr, key in [("tracks.slen", "tracks_slen"),
                      ("structure.sarcomere.length_vectors", "sarcomere_length_vectors"),
                      ("structure.vectors.pos_vectors", "pos_vectors")]:
        if key in r:
            obj = r
            for part in attr.split("."):
                obj = getattr(obj, part)
            shape = getattr(obj, "shape", f"list[{len(obj)}]")
            print(f"    r.{attr:<34} == r[{key!r}]   {shape}")
            break


if __name__ == "__main__":
    main()
