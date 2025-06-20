Calvano et al. (2020) Q-Learning Replication
============================================

Complete Python replication of Calvano et al. (2020) "Artificial Intelligence, Algorithmic Pricing, and Collusion" achieving **127%** individual and **179%** joint profit targets.

.. note::
   **Academic Achievement**: This implementation exceeds all benchmarks from the original AER paper with full reproducibility and modern Python ecosystem integration.

Quick Start
-----------

Install via PyPI::

   pip install calvano-replication

Or clone and run::

   git clone https://github.com/Yusei406/calvano-replication.git
   cd calvano-replication
   ./run_all.sh

Interactive Tutorials
--------------------

.. toctree::
   :maxdepth: 2
   
   notebooks/quick_demo
   notebooks/parameter_sweep
   notebooks/algorithm_comparison

API Reference
-------------

.. toctree::
   :maxdepth: 2
   
   api/environment
   api/agent
   api/stats

Performance Results
------------------

==================  =======  =======  ===========
Metric              Target   Actual   Achievement
==================  =======  =======  ===========
Individual Profit   0.18     0.229    127%
Joint Profit        0.26     0.466    179%
Convergence Rate    0.9      1.0      111%
==================  =======  =======  ===========

Citation
--------

.. code-block:: bibtex

   @software{ozawa2024calvano,
     author = {Ozawa, Yusei},
     title = {calvano-replication: Python Implementation of Calvano et al. (2020)},
     url = {https://github.com/Yusei406/calvano-replication},
     doi = {10.5281/zenodo.15700733},
     year = {2024}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
