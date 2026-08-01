"""Head-to-head on dense track data: current structure.json path vs Zarr store.

Synthesizes the exact arrays track_sarcomere_vectors() emits at 10^4-track
scale (most are NaN outside each track's alive window, as real tracks are),
then serializes the *same* dict both ways.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sarcasm.ioutils import IOUtils
from sarcasm.results_store import Results, write_results

rng = np.random.default_rng(0)
N, T = 10_000, 200
print(f"synthesizing {N} tracks x {T} frames "
      f"(~{N * T / 1e6:.1f}M cells per (N,T) array)\n")


def alive_mask():
    """Each track alive over one contiguous window -> realistic NaN sparsity."""
    starts = rng.integers(0, T - 10, N)
    lens = rng.integers(10, T, N)
    m = np.zeros((N, T), bool)
    for i in range(N):
        e = min(T, starts[i] + lens[i])
        m[i, starts[i]:e] = True
    return m


live = alive_mask()


def f32(scale=1.0, base=0.0):
    a = (rng.standard_normal((N, T)).astype(np.float32) * scale + base)
    a[~live] = np.nan
    return a


pos_um = np.stack([f32(20, 50), f32(20, 50)], -1)
slen = f32(0.2, 1.8)
data = {
    "n_tracks": N,
    "track_ids": np.arange(N, dtype=np.int32),
    "track_start_frame": live.argmax(1).astype(np.int32),
    "track_lengths": live.sum(1).astype(np.int32),
    "tracks_positions_um": pos_um,
    "tracks_positions_px": pos_um / 0.65,
    "tracks_slen": slen,
    "tracks_orientations": f32(1.0),
    "tracks_observed": live & (rng.random((N, T)) > 0.1),
    "tracks_detection_id": np.where(live, rng.integers(0, 5000, (N, T)), -1).astype(np.int32),
    "tracks_midline_id": np.where(live, rng.integers(0, 800, (N, T)), -1).astype(np.int32),
    "displacement_magnitude": f32(2.0),
    "displacement_along_sarcomere": f32(2.0),
    "displacement_perpendicular": f32(0.5),
    "velocity_magnitude": f32(2.0),
    "flow_at_vectors": np.stack([f32(2), f32(2)], -1),
    "params.track_sarcomere_vectors.frames": list(range(T)),
    "params.track_sarcomere_vectors.max_disp_along_px": 15.0,
}
# the duplicate aliases the current code writes (structure.py:1663-1664)
for k in ("displacement_magnitude", "displacement_along_sarcomere",
          "displacement_perpendicular", "velocity_magnitude", "flow_at_vectors"):
    data[f"motionfield_tracker_{k}"] = data[k]

raw = sum(v.nbytes for v in data.values() if isinstance(v, np.ndarray))
print(f"raw in-memory array bytes: {raw / 1e6:.0f} MB   (NaN fraction {1 - live.mean():.0%})\n")

tmp = tempfile.mkdtemp()

# ---- current path: structure.json via IOUtils ---- #
jpath = os.path.join(tmp, "structure.json")
t = time.time()
IOUtils.json_serialize(data, jpath)
js = time.time() - t
jbytes = os.path.getsize(jpath)
with open(jpath) as fh:
    nlines = sum(1 for _ in fh)
t = time.time()
_ = IOUtils.json_deserialize(jpath)
jload = time.time() - t
print(f"structure.json (current):  {jbytes / 1e6:8.0f} MB   "
      f"write {js:5.1f}s  read {jload:5.1f}s  ({nlines / 1e6:.0f}M lines)")

# ---- new path: Zarr ---- #
zpath = os.path.join(tmp, "data.zarr")
t = time.time()
write_results(data, zpath)
zs = time.time() - t
zbytes = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fs in os.walk(zpath) for f in fs)
t = time.time()
r = Results(zpath)
zopen = time.time() - t
t = time.time()
_ = r["tracks_slen"]
zone = time.time() - t
print(f"data.zarr (proposed):      {zbytes / 1e6:8.0f} MB   "
      f"write {zs:5.1f}s  open {zopen * 1e3:4.1f}ms  full-array {zone * 1e3:4.1f}ms")
print(f"\n  -> {jbytes / zbytes:.0f}x smaller on disk, "
      f"{js / zs:.0f}x faster to write, "
      f"open is lazy ({jload / max(zopen, 1e-9):.0f}x faster than full reparse)")

# ---- lazy single-track access: r.tracks.slen[i] reads one chunk ---- #
print("\n  lazy native-group access:")
handle = r.tracks.slen                       # zarr array, nothing loaded
print(f"    r.tracks.slen           -> {handle!r}")
t = time.time()
row = r.tracks.slen[5]                        # one track out of 10,000
dt_one = time.time() - t
t = time.time()
full = r.tracks.slen[:]                        # whole block
dt_full = time.time() - t
print(f"    r.tracks.slen[5]         {row.shape}  {dt_one * 1e3:5.2f} ms  "
      f"(one track)")
print(f"    r.tracks.slen[:]         {full.shape}  {dt_full * 1e3:5.2f} ms  "
      f"(full block) -> single-track read is {dt_full / max(dt_one, 1e-9):.0f}x cheaper")
print(f"    r.tracks.positions_um[100:110].shape = {r.tracks.positions_um[100:110].shape}")
print(f"    r.params.track_sarcomere_vectors.max_disp_along_px = "
      f"{r.params.track_sarcomere_vectors.max_disp_along_px}")

# round-trip + dedup note
bad = [k for k in data if not (
    np.array_equal(np.asarray(data[k]), np.asarray(r[k]), equal_nan=True)
    if isinstance(data[k], np.ndarray) else data[k] == r[k])]
print(f"  round-trip: {'OK' if not bad else bad}")
dup = sum(data[k].nbytes for k in data if k.startswith('motionfield_tracker_'))
print(f"  (of which {dup / 1e6:.0f} MB raw is the motionfield_tracker_* duplication "
      f"the new layout would drop)")
shutil.rmtree(tmp)
