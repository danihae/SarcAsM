.. SarcAsM documentation master file, created by
   sphinx-quickstart on Sat Nov 12 10:52:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/graphical_abstract.png
**Graphical summary of SarcAsM.**


SarcAsM is a comprehensive Python package for automated multiscale analysis of sarcomere organization and contractility in cardiomyocytes. Using a combination of custom deep learning and graph-based methods, SarcAsM simultaneously detects Z-bands, M-bands, sarcomere orientation, and cell/sarcomere masks to enable robust analysis across four hierarchical levels: Z-band morphology, individual sarcomere structure and dynamics, myofibril architecture, and whole-cell sarcomere domain patterns. The AI-based approach provides superior detection compared to traditional filter-based methods, especially in challenging imaging conditions with low signal-to-noise ratios.

The package offers precise tracking of individual sarcomere motion with sub-20 nm spatial accuracy, allowing researchers to identify subtle contractile phenotypes in both structural and functional assays. SarcAsM supports a wide range of applications including developmental studies, disease modeling, drug screening, and mechanobiological investigations. The multi-parameter analysis capabilities make it possible to characterize and distinguish even subtle pathological or drug-induced phenotypes in cardiac cells.

SarcAsM is designed to be accessible and versatile, featuring a high-level API for seamless integration into your analysis workflows and customization for specific research needs. The package includes a pre-trained generalist deep learning model that works effectively across diverse cardiac imaging datasets without requiring extensive retraining, making advanced sarcomere analysis immediately available to researchers regardless of their computational expertise.

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
   ./notebooks/tutorial_training_data_generation
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