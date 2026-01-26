## Thermomechanical phasefield solving schemes
This repository provides three solving schemes for the thermomechanically coupled phase-field crack simulations, including the L-BFGS monolithic scheme, the fully staggered scheme, and the partially staggered scheme.

### Purpose
This repository provides the source code and the input files for the numerical examples used in the paper titled “Fully staggered, partially staggered, and monolithic schemes for thermomechanically coupled phase-field crack simulations: solver development and performance evaluation”. The tightly coupled governing equations are listed below
<p align="center">
<img src="./doc/eq1.png" width="350">
</p>
which are used to solve for the displacement field, the temperature field, and the phase-field. Three different solving schemes are provided:
1. L-BFGS monolithic scheme

2. Fully staggered scheme
3. partially staggered scheme
