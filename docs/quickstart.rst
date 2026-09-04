==========
Quickstart
==========

Sarcomere structure analysis
============================

Test data for getting started can be found `here <https://zenodo.org/records/15389034/files/test_data.zip?download=1>`_.

More detailed instructions see :doc:`notebooks/tutorial_structure_analysis`.

.. code-block:: python

    from sarcasm import *

    # initialize SarcAsM object for tif-file
    file_path = '/path/to/file.tif'
    sarc_obj = SarcAsM(file_path)

    # detect sarcomere Z-bands, M-bands, orientation, sarcomere mask and cell mask by deep learning
    sarc_obj.detect_sarcomeres()

    # analyze cell mask
    sarc_obj.analyze_cell_mask()

    # analyze Z-band morphology
    sarc_obj.analyze_z_bands()

    # analyze sarcomere vectors
    sarc_obj.analyze_sarcomere_vectors()

    # analyze myofibrils
    sarc_obj.analyze_myofibrils()

    # analyze sarcomere domains
    sarc_obj.analyze_sarcomere_domains()

Sarcomere motion analysis
=========================

Motion analysis tracks every sarcomere vector through the movie, then analyzes the
contractions of *groups* of tracks. More detailed instructions see
:doc:`notebooks/tutorial_sarcomere_tracking`.

.. code-block:: python

    from sarcasm import SarcAsM

    # initialize SarcAsM object for tif-file
    file_path = '/path/to/movie.tif'
    sarc_obj = SarcAsM(file_path)

    # detect sarcomere Z-bands, M-bands, orientation and masks in every frame to
    # be tracked. Motion analysis needs M-bands and orientation per frame, so a
    # single-frame detection is not enough here (pass frames=range(a, b) to
    # restrict the analysis to part of the movie).
    sarc_obj.detect_sarcomeres(frames='all')

    # optional, for high-speed movies: replace the per-frame Z-band mask with the
    # temporally consistent 3D U-Net output
    sarc_obj.detect_z_bands_fast_movie()

    # analyze sarcomere vectors in all frames
    sarc_obj.analyze_sarcomere_vectors(frames='all')

    # track every sarcomere vector through the movie
    sarc_obj.track_sarcomere_vectors()

    # analyze contractions of all tracks pooled into one averaged signal
    sarc_obj.analyze_track_motion(by='pool')

    # ... or group the tracks per myofibril and analyze each group separately
    sarc_obj.analyze_myofibrils(frames=[0])
    sarc_obj.analyze_track_motion(by='myofibril', reference_frame=0)

Tracks can also be grouped by M-band (``by='mband'``), by myofibril domain
(``by='domain'``), along automatically detected lines of interest (``by='loi'``),
or by your own labels (``by='custom'``). Each grouping writes its features under
``<kind>_<feature>`` keys — see :ref:`motion_features`.

Controlling Log Output
======================

SarcAsM provides detailed logging to help track analysis progress and troubleshoot issues.
You can control the verbosity when initializing objects:

.. code-block:: python

    from sarcasm import SarcAsM

    # Default: INFO level - shows analysis progress
    sarc_obj = SarcAsM('file.tif')

    # Verbose: DEBUG level - see all diagnostic details
    sarc_obj = SarcAsM('file.tif', log_level='DEBUG')

    # Quiet: WARNING level - only show warnings and errors
    sarc_obj = SarcAsM('file.tif', log_level='WARNING')
