.. TMAS documentation
.. ==================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

..
..    :maxdepth: 2
..    :caption: Contents:

..    changelog.md
..    contributing.md
..    conduct.md

TMAS's documentation
====================

.. _usage:
.. _installation-github:
.. _installation-python:
.. _tutorial:

Overview
--------

- The package uses deep learning to detect M. tuberculosis growth in 96-well microtiter plates and determines **Minimum Inhibitory Concentrations (MICs)**.
.. - The model weights (``best_model_yolo.pt``) is downloaded automatically and separately to avoid including large files directly in the package. It is downloaded only once.

:ref:`Usage <usage>`
--------------------
``TMAS`` can be used to detect growth in a 96-well plate and calculate the MIC result of each drug based on the assigned plate design (*UKMYC5* or *UKMYC6*) and plot the results

:ref:`Installation - Python Package<installation-python>`
-------------------------------------------------------------------------

1. Install ``TMAS`` PyPi package:

.. code-block:: console

   $ pip install tmas==1.0.1

2. Run ``TMAS``:

.. code-block:: console

   $ run_tmas -visualize [folder_path] [output_format]

- (Optional) -visualize/--visualize: to illustrate the output image 
- folder_path: The path to the folder of the raw images
- output_format: output MIC of each drug in ``csv`` or ``json`` file (default format is ``csv``)

:ref:`Installation - GitHub <installation-github>`
--------------------------------------------------

1. Clone the repository and navigate to the project directory.

.. code-block:: console

   $ git clone https://github.com/Oucru-Innovations/tmas/
   $ cd tmas

2. Install the ``TMAS`` package using:

.. code-block:: console
   
   $ pip install -e .

3. Run ``TMAS``:

.. code-block:: console
   
   $ run_tmas -visualize [folder_path] [output_format]

- (Optional) -visualize/--visualize: to illustrate the output image 
- folder_path: The path to the folder of the raw images
- output_format: output MIC of each drug in ``csv`` or ``json`` file (default format is ``csv``)

If encounting any error in Installing the packages, please refer to the Debugging section.



:ref:`Tutorial<tutorial>` (to be updated when the examples are uploaded)
------------------------------------------------------------------------

1. Explore the examples folder

.. code-block:: console

   $ cd data
   $ ls
   1/ 2/ 3/ 4/ 5/

In each examples folder, there is the raw image with the exact same name with the folder

.. code-block:: console

   $ ls 1/
   01-DR0013-DR0013-1-14-UKMYC6-raw.png

2. To process and analyse a single image using the default settings is simply

Choose your desired MIC output file:

- json: with only 1 image

.. code-block:: console

   $ run_tmas data/1/01-DR0013-DR0013-1-14-UKMYC6-raw.png json

- json: with a whole folder

.. code-block:: console

   $ run_tmas data/1 json

or

- csv: with only 1 image

.. code-block:: console

   $ tmas_run 01-DR0013-DR0013-1-14-UKMYC6-raw.png csv

- csv: with a whole folder

.. code-block:: console

   $ run_tmas data/1 csv

3. Growth detection output:

.. image:: ../../data/1/output/sampleoutput.png
   :alt: Output Example
   :width: 600px


4. Output files:
After ``TMAS`` has done running, the growth detection and MIC results will be displayed in your terminal.

Not only that, the growth detection image and the MIC results file with the chosen format will be saved in the same folder with the input image.

.. code-block:: console

   $ ls -a 1/
   output/ 01-DR0013-DR0013-1-14-UKMYC6-raw.png
   $ ls -a 1/output/
   01-DR0013-DR0013-1-14-UKMYC6-growth-matrix.png
   01-DR0013-DR0013-1-14-UKMYC6-mics.csv
   01-DR0013-DR0013-1-14-UKMYC6-mics.json
   01-DR0013-DR0013-1-14-UKMYC6-filtered.png

- ``01-DR0013-DR0013-1-14-UKMYC6-raw.png`` is the original image.
- ``01-DR0013-DR0013-1-14-UKMYC6-filered.png`` is the filtered image after preprocessing.
- ``01-DR0013-DR0013-1-14-UKMYC6-growth-matrix.png`` is the image with the growth detection plotted.
- ``01-DR0013-DR0013-1-14-UKMYC6-mics.csv`` contains the information, including filename, drug name, growth detection results, MIC result.
- ``01-DR0013-DR0013-1-14-UKMYC6-mics.json`` contains the same information as the ``csv`` file but in a different format per requested.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   changelog.md
   contributing.md
   conduct.md