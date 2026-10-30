============
Command line
============

The ``sarcasm`` command runs the analysis without Python code or the GUI. It is
meant for shell scripts, cluster jobs and AI coding agents: every command prints
one JSON document to stdout, while log messages and progress bars go to stderr.
Without a command, ``sarcasm`` opens the GUI.

.. code-block:: console

    $ sarcasm info movie.tif                   # check axes and calibration first
    $ sarcasm structure movie.tif --frames 0   # structure features of frame 0
    $ sarcasm motion movie.tif --by myofibril  # tracking + contraction analysis
    $ sarcasm results movie.tif --find slen    # list stored result keys
    $ sarcasm results movie.tif structure.sarcomere.slen_mean
    $ sarcasm features --registry motion       # definitions and units

Commands
========

``info FILE``
    Axes, shape, frame count, pixel size (µm), frame time (s), stored masks.

``structure FILE``
    Sarcomere detection followed by :meth:`~sarcasm.SarcAsM.full_analysis_structure`.
    Prints the summary features of :attr:`~sarcasm.Export.structure_keys_default`, each
    a list aligned with ``"frames"``.

``motion FILE``
    Detection, sarcomere vectors, :meth:`~sarcasm.SarcAsM.track_sarcomere_vectors` and
    :meth:`~sarcasm.SarcAsM.analyze_track_motion`. Prints one record per group (``--by``;
    default ``pool``), with per-cycle features averaged over cycles.

``results FILE [KEY ...]``
    Without keys: all stored keys with their display names (``--find`` filters).
    With keys: their full values (time series, per-object distributions).

``features``
    Name, definition and producing function of every structure and motion feature.

Options
=======

``--pixelsize`` (µm), ``--frametime`` (s), ``--axes`` (e.g. ``TYX``), ``--channel``
    Calibration. Explicit values override the file metadata. A missing or implausible
    pixel size is an error; ``motion`` also needs a frame time.

``--frames``
    ``all`` (default), one frame (``3``), a list (``0,4,9``) or a range (``0:100``, end
    exclusive).

``--restart``
    Delete the stored results for this file first. Required when you change the
    calibration of a file that was already analysed.

``--model``, ``--device``, ``--log-level``
    U-Net checkpoint (``generalist``, ``legacy`` or a path), compute device, stderr
    verbosity (default ``WARNING``).

``motion`` only: ``--by``, ``--motion-predictor flow`` (coarse frame rates only),
``--fast-movie`` (3D U-Net Z-band refinement for high-speed movies).

Output and errors
=================

Results go to the sibling ``<name>.ome.zarr`` store, exactly as with the Python API,
so the GUI and ``SarcAsM(file)`` see what the CLI computed and the other way round.

Warnings raised during a run are repeated in the result's ``"warnings"`` list; they
explain null values (for example, no beating rate when fewer than two contraction
cycles were detected).

On failure nothing is written to stdout, the last line of stderr is a JSON object
``{"error": ..., "message": ..., "hint": ...}``, and the exit code is

- ``2``: invalid command-line usage,
- ``3``: input to fix (missing file, missing or conflicting calibration, implausible
  axes, unknown result key),
- ``1``: any other error.

For agents
==========

A concise description for language-model agents is in
`llms.txt <https://github.com/danihae/SarcAsM/blob/main/llms.txt>`_ at the repository root.
