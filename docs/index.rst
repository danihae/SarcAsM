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
   gui

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   ./notebooks/tutorial_structure_analysis
   ./notebooks/tutorial_motion_analysis
   ./notebooks/tutorial_batch_analysis

.. toctree::
   :maxdepth: 2
   :caption: Python package:

   API reference <_autosummary/sarcasm>



Citing SarcAsM
---------------
Authors: *Daniel Härtter, Lara Hauke, Til Driehorst, Yuxi Long, Guobin Bao, Andreas Primeßnig, Branimir Berecic, Lukas Cyganek, Malte Tiburcy, Wolfram-Hubertus Zimmermann, Christoph F. Schmidt*

Title: SarcAsM (Sarcomere Analysis Multi-tool): A comprehensive software tool for structural and functional analysis of sarcomeres in cardiomyocytes

.. todo::
   Update this upon actual publication. Probably insert bibtex as well.

License
--------------

Contact
--------------
For questions, requests and issues, please contact us at :mailto:'daniel.haertter@med.uni-goettingen.de' or `open an issue on GitHub <https://github.com/danihae/sarcasm/issues>`_.

.. To-Do:
   Fix formatting of notebooks: Add more hashtags to the markdown headers so that they appear well in the toctree.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`