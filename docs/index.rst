.. SarcAsM documentation master file, created by
   sphinx-quickstart on Sat Nov 12 10:52:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SarcAsM is a Python package for analysis of sarcomere structure and motion in microscopy images and movies of cardiomyocytes.

.. image:: images/graphical_abstract.png
**Summary of SarcAsM:** (a) Detailed structural analysis of sarcomeres Z-bands, sarcomeres, myofibrils, and sarcomere domains. (b) Tracking and analysis of individual and average sarcomere motion. (c) Information about SarcAsM.

Using SarcAsM
--------------

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
   ./notebooks/tutorial_training_3Dunet
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
Authors: *Daniel Härtter, Lara Hauke, Til Driehorst, Yuxi Long, Guobin Bao, Andreas Primeßnig, Branimir Berecic, Lukas Cyganek, Malte Tiburcy, Wolfram-Hubertus Zimmermann, Christoph F. Schmidt*

Title: SarcAsM (Sarcomere Analysis Multi-tool): A comprehensive software tool for structural and functional analysis of sarcomeres in cardiomyocytes

.. todo::
   Update this upon actual publication. Probably insert bibtex as well.

License
--------------
This software is patent pending (Patent Application No. DE 10 2024 112 939.5, Priority Date: 8.5.2024).

Academic and Non-Commercial Use
==================
This software is free for academic and non-commercial use. Users are granted a non-exclusive, non-transferable license to use and modify the software for research, educational, and other non-commercial purposes.

Commercial Use Restrictions
==================
Commercial use of this software is strictly prohibited without obtaining a separate license agreement. This includes but is not limited to:

- Using the software in a commercial product or service
- Using the software to provide services to third parties
- Reselling or redistributing the software

For commercial licensing inquiries, please contact:

**MBM ScienceBridge GmbH**,
Hans-Adolf-Krebs-Weg 1,
37077 Göttingen,
https://sciencebridge.de/en/

All rights not expressly granted are reserved. Unauthorized use may result in legal action.

Contact
--------------
For questions, requests and issues, please contact us at daniel.haertter@med.uni-goettingen.de or `open an issue on GitHub <https://github.com/danihae/sarcasm/issues>`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`