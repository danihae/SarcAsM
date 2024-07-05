==========
Quickstart
==========

Sarcomere structure analysis
============================

More detailed instructions see :doc:`../notebooks/tutorial_structure_analysis`.

.. code-block:: python

    from sarcasm import *

    # initialize SarcAsM object for tif-file
    filename = '/path/to/file.tif'
    sarc_obj = SarcAsM(filename)

    # predict sarcomere Z-bands and cell mask by deep learning
    sarc_obj.structure.predict_z_bands()
    sarc_obj.structure.predict_cell_mask()

    # analyze Z-band morphology
    sarc_obj.structure.analyze_z_bands()

    # analyze sarcomere vectors
    sarc_obj.structure.analyze_sarcomere_vectors()

    # analyze myofibrils
    sarc_obj.structure.analyze_myofibrils()

    # analyze sarcomere domains
    sarc_obj.structure.analyze_sarcomere_domains()

Sarcomere motion analysis
=========================

More detailed instruction see :doc:`../notebooks/tutorial_motion_analysis`.

.. code-block:: python

    from sarcasm import *

    # initialize SarcAsM object for tif-file
    filename = '/path/to/file.tif'
    sarc_obj = SarcAsM(filename)

    # automatically detect lines of interest (LOIs) for sarcomere tracking
    sarc_obj.structure.detect_lois(n_lois=4)

    # get list of LOIs and select single LOI
    list_lois = sarc_obj.structure.get_list_lois()
    file, loi = list_lois[0]

    # initialize Motion object for LOI
    mot_obj = Motion(file, loi)

    # track individual Z-bands
    mot_obj.detect_peaks()
    mot_obj.track_z_bands()

    # predict contraction intervals using neural network ContractionNet and analyze contractions
    mot_obj.detect_analyze_contractions()

    # calculate sarcomere length change and velocity of individual sarcomeres and average
    mot_obj.get_trajectories()

    # analyze individual and average sarcomere trajectories
    mot_obj.analyze_trajectories()