# -*- coding: utf-8 -*-
# Copyright (c) 2025 University Medical Center Göttingen, Germany.
# All rights reserved.
#
# Patent Pending: DE 10 2024 112 939.5
# SPDX-License-Identifier: LicenseRef-Proprietary-See-LICENSE
#
# This software is licensed under a custom license. See the LICENSE file
# in the root directory for full details.
#
# **Commercial use is prohibited without a separate license.**
# Contact MBM ScienceBridge GmbH (https://sciencebridge.de/en/) for licensing.

"""Headless command-line interface: ``sarcasm <command> ...``.

Every command writes one JSON document to stdout and nothing else; progress and
log messages go to stderr, and the warnings logged during the run are repeated in
the result's ``"warnings"`` list (they explain null values). An error is a JSON
object on the last line of stderr (``{"error": ..., "message": ..., "hint": ...}``) with a non-zero exit code:
2 for bad usage, 3 for bad input (missing file, missing calibration, frames out
of range), 1 for anything else. ``sarcasm`` without a command opens the GUI.
"""

import argparse
import difflib
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

EXIT_ERROR, EXIT_USAGE, EXIT_INPUT = 1, 2, 3


class _WarningCollector(logging.Handler):
    """Keeps the package's warnings so they can be returned inside the JSON result."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg not in self.messages:
            self.messages.append(msg)


class CLIInputError(Exception):
    """Invalid input the user can fix; carries a hint for the fix."""

    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.hint = hint


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_frames(text: str) -> Union[str, List[int]]:
    """``'all'``, one frame, comma-separated frames, or a half-open range ``'a:b'``."""
    s = str(text).strip()
    if ':' in s:
        a, _, b = s.partition(':')
        if not (a.strip().isdigit() and b.strip().isdigit()) or int(b) <= int(a):
            raise argparse.ArgumentTypeError(
                f"{text!r} is not a frame range. Use 'a:b' with a < b (b exclusive), e.g. 0:100.")
        return list(range(int(a), int(b)))
    from sarcasm._internal.frames import parse_frames_text
    try:
        frames = parse_frames_text(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return 'all' if frames is None else frames


def _jsonable(x: Any) -> Any:
    """Numpy / sparse / NaN-safe conversion to plain JSON values (NaN -> null)."""
    import numpy as np
    from scipy import sparse
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if sparse.issparse(x):
        return {'sparse_shape': list(x.shape), 'nnz': int(x.nnz)}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (str, int)) or x is None:
        return x
    return str(x)


def _emit(doc: Dict[str, Any]) -> None:
    json.dump(_jsonable(doc), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write('\n')
    sys.stdout.flush()


def _fail(kind: str, message: str, hint: Optional[str], code: int) -> int:
    err = {'error': kind, 'message': message}
    if hint:
        err['hint'] = hint
    json.dump(err, sys.stderr, ensure_ascii=False)
    sys.stderr.write('\n')
    return code


# --------------------------------------------------------------------------- #
# Shared object construction
# --------------------------------------------------------------------------- #
def _open(args, *, restart: bool = False):
    """SarcAsM object for ``args.file`` with the calibration flags applied."""
    from sarcasm import SarcAsM
    return SarcAsM(args.file, restart=restart, pixelsize=args.pixelsize, frametime=args.frametime,
                   channel=args.channel, axes=args.axes, device=args.device,
                   log_level=args.log_level)


def _metadata(sarc) -> Dict[str, Any]:
    md = sarc.metadata
    return {'file': sarc.file_path, 'store': str(sarc.store_path),
            'axes': md.axes, 'shape': md.shape, 'n_frames': md.n_stack,
            'pixelsize_um': md.pixelsize, 'frametime_s': md.frametime,
            'channel': md.channel}


def _version() -> str:
    from sarcasm import __version__
    return __version__


def _frames_used(sarc, frames) -> List[int]:
    canon = sarc._canonical_frames(frames)
    return list(range(sarc.metadata.n_stack)) if canon == 'all' else list(canon)


def _per_frame(value: Any, frames: List[int], n_stack: int) -> Any:
    """Restrict a frame-filed array (length n_stack) to the analyzed frames."""
    import numpy as np
    if isinstance(value, (list, np.ndarray)) and len(value) == n_stack and n_stack > 1:
        return [value[f] for f in frames]
    return value


def _require_frametime(sarc) -> None:
    if sarc.metadata.frametime is None:
        raise CLIInputError('Frame time is unknown: the file metadata holds none.',
                            hint='Pass --frametime <seconds per frame>, e.g. --frametime 0.0164 for 61 fps.')
    if sarc.metadata.n_stack < 2:
        raise CLIInputError('Motion analysis needs a movie, but the file holds a single frame.',
                            hint="Use 'sarcasm structure' for single images.")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_info(args) -> Dict[str, Any]:
    sarc = _open(args)
    return {'command': 'info', 'sarcasm_version': _version(), **_metadata(sarc),
            'masks': sorted(sarc.store.mask_names()) if sarc.store.exists else [],
            'n_result_keys': len(sarc.data.keys())}


def cmd_structure(args) -> Dict[str, Any]:
    from sarcasm import Export
    sarc = _open(args, restart=args.restart)
    frames = _frames_used(sarc, args.frames)
    sel = 'all' if args.frames == 'all' else frames
    sarc.detect_sarcomeres(frames=sel, model_path=args.model)
    sarc.full_analysis_structure(frames=sel)
    n = sarc.metadata.n_stack
    features = {k: _per_frame(sarc.data.get(k), frames, n) for k in Export.structure_keys_default}
    return {'command': 'structure', 'sarcasm_version': _version(), **_metadata(sarc),
            'frames': frames, 'features': features,
            'note': 'Each feature is a list aligned with "frames" (or a scalar). '
                    "Units and definitions: 'sarcasm features --registry structure'. "
                    "Per-object distributions: 'sarcasm results FILE <key>'."}


def cmd_motion(args) -> Dict[str, Any]:
    from sarcasm import Export
    sarc = _open(args, restart=args.restart)
    _require_frametime(sarc)
    frames = _frames_used(sarc, args.frames)
    sel = 'all' if args.frames == 'all' else frames
    sarc.detect_sarcomeres(frames=sel, model_path=args.model)
    if args.fast_movie:
        sarc.detect_z_bands_fast_movie()
    sarc.analyze_sarcomere_vectors(frames=sel)
    sarc.track_sarcomere_vectors(frames=sel, motion_predictor=args.motion_predictor)
    if args.by in ('myofibril', 'domain'):
        ref = frames[0]
        sarc.analyze_myofibrils(frames=[ref])
        if args.by == 'domain':
            sarc.analyze_sarcomere_domains(frames=[ref])
        sarc.analyze_track_motion(by=args.by, reference_frame=ref)
    else:
        sarc.analyze_track_motion(by=args.by)
    meta_keys = set(sarc.metadata.to_dict())
    groups = [{k: v for k, v in rec.items() if k not in meta_keys and k != 'tif_name'}
              for rec in Export.get_motion_dict_per_group(sarc, kind=args.by)]
    return {'command': 'motion', 'sarcasm_version': _version(), **_metadata(sarc),
            'frame_range': [frames[0], frames[-1] + 1], 'by': args.by,
            'n_tracks': len(sarc.data.get('motion.tracks.slen', [])),
            'groups': groups,
            'note': 'One record per group; per-cycle features are averaged over cycles. '
                    "Units and definitions: 'sarcasm features --registry motion'. "
                    f"Full arrays: 'sarcasm results FILE --find motion.{args.by}'."}


def cmd_results(args) -> Dict[str, Any]:
    from sarcasm import describe_key
    sarc = _open(args)
    keys = sarc.data.keys()
    if not args.keys:
        sel = [k for k in keys if args.find in k] if args.find else keys
        return {'command': 'results', 'file': sarc.file_path, 'n_keys': len(sel),
                'keys': {k: (describe_key(k) or {}).get('name', '') for k in sel}}
    missing = [k for k in args.keys if k not in sarc.data]
    if missing:
        close = sorted({c for m in missing for c in difflib.get_close_matches(m, keys, n=5, cutoff=0.6)})
        raise CLIInputError(f'Keys not in this store: {missing}.',
                            hint=f'Similar keys: {close}' if close else
                            "List keys with 'sarcasm results FILE' (optionally --find <text>).")
    return {'command': 'results', 'file': sarc.file_path,
            'values': {k: sarc.data[k] for k in args.keys}}


def cmd_features(args) -> Dict[str, Any]:
    from sarcasm.features import structure_feature_dict, motion_feature_dict, MOTION_KINDS

    def rows(registry, prefix=''):
        return {prefix + k: {'name': v.get('name'), 'description': v.get('description'),
                             'function': v.get('function')}
                for k, v in registry.items() if not args.find or args.find in k}

    out: Dict[str, Any] = {'command': 'features'}
    if args.registry in (None, 'structure'):
        out['structure'] = rows(structure_feature_dict)
    if args.registry in (None, 'motion'):
        out['motion'] = rows(motion_feature_dict)
        out['motion_key_pattern'] = 'motion.<kind>.<feature>, kind in ' + ', '.join(MOTION_KINDS)
    return out


def cmd_gui(args) -> Dict[str, Any]:
    from sarcasm_app.__main__ import main as gui_main
    gui_main()
    return {}


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #
def _add_file_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('file', help='TIFF (or OME-Zarr) image or movie.')
    g = p.add_argument_group('calibration (override file metadata)')
    g.add_argument('--pixelsize', type=float, help='Pixel size in µm. Required if the file has none.')
    g.add_argument('--frametime', type=float, help='Time between frames in s. Required for motion if the '
                                                   'file has none.')
    g.add_argument('--channel', type=int, help='Channel index with the sarcomere signal.')
    g.add_argument('--axes', help="Dimension order, e.g. 'TYX' or 'TCYX'. Default: auto-detect.")
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--log-level', default='WARNING',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='stderr verbosity. Default: WARNING.')


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--frames', type=parse_frames, default='all',
                   help="'all' (default), one frame '3', a list '0,4,9' or a range '0:100' (end exclusive).")
    p.add_argument('--model', default=None,
                   help="U-Net: 'generalist', 'legacy' or a weights path. Default: chosen by pixel size.")
    p.add_argument('--restart', action='store_true',
                   help='Delete previous results for this file before running.')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='sarcasm',
        description='SarcAsM: sarcomere structure and motion analysis. Each command prints one JSON '
                    'document to stdout; logs go to stderr. Without a command the GUI opens.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {_version()}')
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')

    p = sub.add_parser('info', help='Image metadata, calibration and stored results.')
    _add_file_args(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser('structure', help='Detect sarcomeres and analyze structure; print per-frame features.')
    _add_file_args(p)
    _add_run_args(p)
    p.set_defaults(func=cmd_structure)

    p = sub.add_parser('motion', help='Track sarcomeres in a movie and analyze contractions per group.')
    _add_file_args(p)
    _add_run_args(p)
    p.add_argument('--by', default='pool', choices=['pool', 'mband', 'myofibril', 'domain', 'loi'],
                   help='Track grouping. Default: pool (all tracks averaged into one signal).')
    p.add_argument('--motion-predictor', default='none', choices=['none', 'flow'],
                   help="'flow' (optical flow) only for coarse frame rates, where one contraction moves a "
                        "sarcomere a good part of its length between frames. Default: none.")
    p.add_argument('--fast-movie', action='store_true',
                   help='Refine Z-bands with the 3D U-Net (high-speed movies only).')
    p.set_defaults(func=cmd_motion)

    p = sub.add_parser('results', help='List or read stored result keys.')
    _add_file_args(p)
    p.add_argument('keys', nargs='*', help="Keys to read, e.g. 'structure.sarcomere.slen_mean'. "
                                           'Without keys: list all keys.')
    p.add_argument('--find', help='Only list keys containing this text.')
    p.set_defaults(func=cmd_results)

    p = sub.add_parser('features', help='Definitions of all result features (no file needed).')
    p.add_argument('--registry', choices=['structure', 'motion'])
    p.add_argument('--find', help='Only features whose key contains this text.')
    p.set_defaults(func=cmd_features)

    p = sub.add_parser('gui', help='Open the napari GUI.')
    p.set_defaults(func=cmd_gui)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return cmd_gui(None) or 0
    args = build_parser().parse_args(argv)
    if args.command is None:
        return cmd_gui(None) or 0
    if getattr(args, 'file', None) is not None and not os.path.exists(args.file):
        return _fail('FileNotFoundError', f'No such file: {args.file}', None, EXIT_INPUT)

    from sarcasm.exceptions import MetaDataError
    # not a StreamHandler, so SarcAsMBase._setup_logging leaves it attached
    collector = _WarningCollector()
    logging.getLogger('sarcasm').addHandler(collector)
    try:
        doc = args.func(args)
    except CLIInputError as e:
        return _fail('InputError', str(e), e.hint, EXIT_INPUT)
    except MetaDataError as e:
        hint = ("Rerun 'structure' or 'motion' with --restart to re-analyze with the new calibration "
                "(deletes stored results), or drop the conflicting flags." if 'restart=True' in str(e) else
                'Pass the calibration explicitly: --pixelsize <µm>, --frametime <s>, --axes, --channel.')
        return _fail('MetaDataError', str(e), hint, EXIT_INPUT)
    except (FileNotFoundError, ValueError) as e:
        return _fail(type(e).__name__, str(e), None, EXIT_INPUT)
    except Exception as e:  # noqa: BLE001 — report every failure as JSON
        return _fail(type(e).__name__, str(e), 'Rerun with --log-level DEBUG for details.', EXIT_ERROR)
    if doc:
        doc['warnings'] = collector.messages
        _emit(doc)
    return 0


if __name__ == '__main__':
    sys.exit(main())
