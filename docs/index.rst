.. SarcAsM documentation master file, created by
   sphinx-quickstart on Sat Nov 12 10:52:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SarcAsM is a Python package for analysis of sarcomere structure and motion in microscopy images and movies of cardiomyocytes.

.. image:: images/graphical_abstract.png
**Summary of SarcAsM:** (a) Detailed structural analysis of sarcomeres Z-bands, sarcomeres, myofibrils, and sarcomere domains. (b) Tracking and analysis of individual and average sarcomere motion. (c) Information about SarcAsM.

Using SarcAsM
-------------

There are two ways to use SarcAsM:

1. Python package with high-level API for programming-affine users

2. Easy-to-use standalone application with graphical user interface (currently in beta phase)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart
   gui

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   ./notebooks/tutorial_structure_analysis
   ./notebooks/tutorial_motion_analysis
   ./notebooks/tutorial_batch_analysis
   ./notebooks/tutorial_data_export
   ./notebooks/tutorial_data_visualization
   ./notebooks/tutorial_training_unet
   ./notebooks/tutorial_training_3dunet
   ./notebooks/tutorial_training_contraction_net

.. toctree::
   :maxdepth: 2
   :caption: Features:

   structure_features
   motion_features

.. toctree::
   :maxdepth: 2
   :caption: API reference:


Citing SarcAsM
---------------
Authors: Daniel Härtter, Lara Hauke, Til Driehorst, Yuxi Long, Guobin Bao, Andreas Primeßnig, Branimir Berecic, Lukas Cyganek, Malte Tiburcy, Christoph F. Schmidt, Wolfram-Hubertus Zimmermann

Title: " SarcAsM (Sarcomere Analysis Multi-tool): a comprehensive software tool for structural and functional analysis of sarcomeres in cardiomyocytes"

.. todo::
   Update this upon actual publication. Probably insert bibtex as well.

Contact
--------------
For questions, requests and issues, please contact us at daniel.haertter@med.uni-goettingen.de or `open an issue on GitHub <https://github.com/danihae/sarcasm/issues>`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`