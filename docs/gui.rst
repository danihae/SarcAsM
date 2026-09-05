================================
GUI Usage
================================

SarcAsM's GUI application uses `Napari image browser <https://napari.org/stable/index.html>`_ as image viewer. Check out `this page <https://napari.org/stable/tutorials/fundamentals/viewer.html>`_ for a primer on Napari's GUI.

Test data for getting started can be found `here <https://zenodo.org/records/15389034/files/test_data.zip?download=1>`_.

Starting the App
================

There are two main ways to start the SarcAsM GUI:

**1. Standalone Applications (Recommended for ease of use):**

*   Pre-built applications for **Windows (.exe)** and **macOS (.app)** are available for download directly from the `GitHub Releases page <https://github.com/danihae/SarcAsM/releases>`_.
*   This method does not require a separate Python installation.
*   **Note:**

    *   These standalone applications are **early versions** and may take a **significant amount of time to start up** initially.
    *   They are built using **Python 3.12**.
    *   The **Windows version currently only utilizes the CPU** and does not support CUDA GPU acceleration. For high-performance needs, consider using the Python API.

**2. From your Python Environment (Recommended for developers or API users):**

*   If you have installed SarcAsM into a Python environment (e.g., via pip or conda):
*   First, remember to activate your installation environment::

      conda activate sarcasm-env # Or your specific environment name

*   Then, to start the app, run this command::

      python -m sarcasm_app

Using the App
=============

.. image:: images/SarcAsM_app.gif
   :alt: Workflow of the SarcAsM application with GUI

The SarcAsM GUI integrates controls (left panel) with a Napari image viewer (right panel). Follow the steps in the control panel sections for analysis.

**Initial Steps:**

1.  **Load File:** Use **Search** in **File Selection** to load a grayscale TIFF file.
2.  **Set Metadata:** In **Metadata**, check the **Pixel size [µm]**. If the field is red or incorrect, enter the correct value and press **Store Metadata**.

**Analysis Sections:**

Proceed through the collapsible sections in the control panel:

*   **Parameter import/export:** Load/save analysis settings or reset to defaults.
*   **Structure:** Analyze sarcomere structure (Z-bands, sarcomere vectors, myofibrils, domains).
*   **Motion:** Track every sarcomere through the movie, group the tracks (whole cell,
    M-band, myofibril, domain or line of interest — drawn or auto-detected) and analyze
    the contraction of each group. The **Display** group controls how the tracks appear
    in the viewer: **Trajectories** (napari tracks with fading trails), **Sarcomeres**
    (every tracked sarcomere at its current position, coloured per frame by ΔSL, SL,
    velocity, group or coverage) and **Groups** (the fibre paths of a myofibril / LOI
    grouping, labelled by id). Click a sarcomere or a fibre path to select its group:
    its members are ringed in the viewer, the per-fibre detail points at it, and the
    **time-series panel** below the viewer overlays the group's SL / ΔSL / velocity
    traces with the group mean and the clicked sarcomere highlighted. **Overlay figure**
    and **Cycle raster** open the corresponding matplotlib figures.
*   **Batch Processing:** Process multiple files automatically.

**Message Area:**

The message area at the bottom of the window displays:

*   Analysis progress and status updates
*   Warnings (shown in yellow) for non-critical issues
*   Errors (shown in red) when operations fail

This provides real-time feedback during analysis without needing to check the console.

**Using Parameters:**

*   **Tooltips:** Hover over parameter names/fields for descriptions.
*   If needed, optimize settings within each section before running the corresponding analysis step (e.g., clicking **Detect sarcomeres**).
