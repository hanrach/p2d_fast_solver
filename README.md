# P2D Solver
*by Rachel Han, Brian Wetton and Colin Macdonald*

This repository contains the solver for pseudo-2D model of Li-ion battery based on Finite Difference Method with automatic differentiation via [JAX](https://github.com/google/jax).

The model is taken from the P2D model outlined in the [paper](http://web.mit.edu/braatzgroup/Torchio_JElectSoc_2016.pdf) by Torchio et al.

## Getting started

* `pip install numpy scipy` (or get these from your OS)
* `pip install jax[cpu]`
* `pip install scikit-umfpack`

Then run `examples/main_reorder.py`.
