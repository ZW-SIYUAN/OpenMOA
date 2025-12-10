API Reference
=============

Welcome to the openmoa API reference. This documentation is automatically
generated from the source code and provides detailed information on the classes
and functions available in openmoa. 

If you are looking to just use OpenMOA, you should start with the
:ref:`tutorials<tutorials>`.

Types
-----

These module provide interfaces for learners, and other basic types used by
openmoa.

..  autosummary::
    :toctree: modules
    :caption: Types
    :recursive:

    openmoa.base
    openmoa.type_alias
    openmoa.instance

Data Streams
------------

These modules provide classes for loading, and simulating data streams. It also
includes utilities for simulating concept drifts.

..  autosummary::
    :toctree: modules
    :caption: Data Streams
    :recursive:

    openmoa.datasets
    openmoa.stream

Problem Settings
----------------

These modules provide classes for defining machine learning problem settings.
    
..  autosummary::
    :toctree: modules
    :caption: Problem Settings
    :recursive:

    openmoa.classifier
    openmoa.regressor
    openmoa.anomaly
    openmoa.ssl
    openmoa.ocl
    openmoa.drift
    openmoa.clusterers
    openmoa.automl

Evaluation
----------

These modules provide classes for evaluating learners.

..  autosummary::
    :toctree: modules
    :caption: Evaluation
    :recursive:

    openmoa.evaluation
    openmoa.prediction_interval

Miscellaneous
-------------

These modules provide miscellaneous utilities.

..  autosummary::
    :toctree: modules
    :caption: Miscellaneous
    :recursive:

    openmoa.ann
    openmoa.splitcriteria
    openmoa.misc
    openmoa.env

Functions
---------

..  automodule:: openmoa
    :members: