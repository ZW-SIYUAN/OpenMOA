.. OpenMOA documentation master file, created by
   sphinx-quickstart on Fri Feb 23 08:41:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


OpenMOA
=======

.. image:: /images/OpenMOA.jpeg
   :alt: OpenMOA

.. image:: https://img.shields.io/pypi/v/openmoa
   :target: https://pypi.org/project/openmoa/
   :alt: Link to PyPI
   
.. image:: https://img.shields.io/discord/1235780483845984367?label=Discord
   :target: https://discord.gg/spd2gQJGAb
   :alt: Link to Discord

.. image:: https://img.shields.io/github/stars/ZW-SIYUAN/OpenMOA?style=flat
   :target: https://github.com/ZW-SIYUAN/OpenMOA
   :alt: Link to GitHub

A Python library for **Utilitarian Online Learning (UOL)** in dynamic feature
spaces. OpenMOA provides a unified API integrating MOA (**Stream Learners**),
CapyMOA (**Stream Learning Backend**), and PyTorch (**Deep Models**) for
reproducible, real-time learning under evolving feature spaces.

To setup OpenMOA, simply install it via pip. If you have any issues with the
installation (like not having Java installed) or if you want GPU support, please
refer to the :ref:`installation`. Once installed take a look at the
:ref:`tutorials` to get started.


.. code-block:: bash

   # OpenMOA requires Java. This checks if you have it installed
   java -version

   # OpenMOA requires PyTorch. This installs the CPU version
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # Install OpenMOA and its dependencies
   pip install openmoa

   # Check that the install worked
   python -c "import openmoa; print(openmoa.__version__)"

.. warning::

   OpenMOA is still in the early stages of development. The API is subject to
   change until version 1.0.0. If you encounter any issues, please report them
   on the `GitHub Issues <https://github.com/ZW-SIYUAN/OpenMOA/issues>`_
   or talk to us on `Discord <https://discord.gg/spd2gQJGAb>`_.

📖 Cite Us
--------------

If you use OpenMOA in your research, please cite us using the following Bibtex entry::

   @misc{
      ZhiliWang2025OpenMOA,
      title={{OpenMOA}: A Python Library for Utilitarian Online Learning},
      author={Zhili Wang and Heitor M. Gomes and Yi He},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/}
   }

.. _installation:

🚀 Installation
---------------

Installation instructions for OpenMOA:

.. toctree::
   :maxdepth: 2

   installation
   docker

🎓 Tutorials
------------
Tutorials to help you get started with OpenMOA.

.. toctree::
   :maxdepth: 2

   tutorials

📚 Reference Manual
-------------------
Reference documentation describing the interfaces fo specific classes, functions,
and modules.

.. toctree::
   :maxdepth: 2

   api/index

ℹ️ About us
-----------

.. toctree::
   about

🏗️ Contributing
---------------
This part of the documentation is for developers and contributors.

.. toctree::
   :maxdepth: 2

   contributing/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
