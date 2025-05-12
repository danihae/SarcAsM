==========
Quickstart
==========

Sarcomere structure analysis
============================

Test data for getting started can be found `here <https://zenodo.org/records/15389034/files/test_data.zip?download=1>`_.

More detailed instructions see :doc:`../notebooks/tutorial_structure_analysis`.

.. code-block:: python

    from sarcasm import *

    # initialize Structure object for tif-file
    filename = '/path/to/file.tif'
    sarc_obj = Structure(filename)

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

More detailed instruction see :doc:`../notebooks/tutorial_motion_analysis`.

.. code-block:: python

    from sarcasm import *

    # initialize Structure object for tif-file
    filename = '/path/to/file.tif'
    sarc_obj = Structure(filename)

    # automatically detect lines of interest (LOIs) for sarcomere tracking
    sarc_obj.detect_lois(n_lois=4)

    # get list of LOIs and select single LOI
    list_lois = sarc_obj.get_list_lois()
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