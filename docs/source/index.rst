.. jax-sn documentation master file, created by
   sphinx-quickstart on Thu Jan  8 13:19:34 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to jax-sn's documentation!
==================================
This package uses a multigroup, discontinuous Galerkin (DG) discrete ordinates (SN) method to solve a linear Boltzmann transport equation in JAX (ax), enabling
differentiable neutron transport simulations. 

It is written in pure functional JAX primitives and uses linear operators to represent the discretised transport operator.

Getting started
-------------------

First install a JAX distribution, see `here <https://docs.jax.dev/en/latest/installation.html>`_. GPU support is strongly preferred.

Then, 

.. code-block:: bash
   
   pip install .

to install. For optional dependencies, :code:`jax-sbgeom` is required for location in simplex meshes, :code:`sympy` and :code:`pytest` are used in the testing suite. Install with 

.. code-block:: bash

   pip install .[geometry,warp,docs]

See the example problems for some instructive examples.


Publications
--------------

This model has been described and benchmarked in the following publications. If this work has been useful to you, please cite [3]:

.. [1] Bogaarts, T. J., & Warmer, F. (2025). A novel discontinuous-Galerkin deterministic neutronics model for fusion applications: development and benchmarking. *Nuclear Fusion*, 65(7), 076015.
   DOI `10.1088/1741-4326/adddc9 <https://doi.org/10.1088/1741-4326/adddc9>`_

.. [2] Bogaarts, T. J., & Warmer, F. (2026). A novel discontinuous-Galerkin deterministic neutronics model for fusion applications: workflow for stellarator reactor design studies. *Nuclear Fusion*, 66(4), 046020.
   DOI `10.1088/1741-4326/ae4e48 <https://doi.org/10.1088/1741-4326/ae4e48>`_

.. [3] Bogaarts, T. J. (2026). Differentiable neutronics for design optimization with application to stellarator blanket optimization. Submitted.
.. [4] Bogaarts, T. J. (2026). Differentiable deterministic neutronics for fusion design. PhD Thesis, Eindhoven University of Technology.


.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   theory.rst   
   user_guide.rst
   examples.rst
   api_reference.rst
   appendices.rst



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
