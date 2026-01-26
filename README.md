## Thermomechanical phase-field solving schemes
This repository provides three solving schemes for the thermomechanically coupled phase-field crack simulations, including the L-BFGS monolithic scheme, the fully staggered scheme, and the partially staggered scheme.

### Purpose
This repository provides the source code and the input files for the numerical examples used in the paper titled “Fully staggered, partially staggered, and monolithic schemes for thermomechanically coupled phase-field crack simulations: solver development and performance evaluation”. The tightly coupled governing equations are listed below
<p align="center">
<img src="./doc/eq1.png" width="300">
</p>
which are used to solve for the displacement field, the temperature field, and the phase-field. Particularly, the material critical energy release rate is a temperature-dependent property:
<p align="center">
<img src="./doc/eq3.png" width="300">
</p>
and the material thermal conductivity is degraded by the phase-field:
<p align="center">
<img src="./doc/eq2.png" width="300">
</p>


Three different solving schemes are provided:

1. L-BFGS monolithic scheme, which updates the three fields simultaneously:
<p align="center">
<img src="./doc/eq4.png" width="300">
</p>

2. Fully staggered scheme, which partitions the coupled problem into three sub-problems:
<p align="center">
<img src="./doc/eq6.png" width="450">
</p>

3. partially staggered scheme, which partitions the coupled problem into a displacement-temperature sub-problem and a phase-field sub-problem:
<p align="center">
<img src="./doc/eq5.png" width="450">
</p>

### Content
The repository contains the following content:
1. the source code of the L-BFGS monolithic scheme, the fully staggered scheme, and the partially staggered scheme for the thermomechanical phase-field crack modeling as well as the $L_2$-projection based adaptive mesh refinement technique.
2. the input files for several numerical examples included in the aforementioned manuscript.

### How to compile
The three solving schemes are implemented in [deal.II](https://www.dealii.org/) (originally with version 9.4.0 and also tested for the developed branch as Sept. 10th, 2025), which is an open source finite element library. In order to use the code (**main.cc**) provided here, deal.II should be configured with MPI and at least with the interfaces to BLAS, LAPACK, Threading Building Blocks (TBB), and UMFPACK. For optional interfaces to other software packages, see https://www.dealii.org/developer/readme.html.

Once the deal.II library is compiled, for instance, to "~/dealii-dev/bin/", follow the steps listed below:
1. cd SourceCode
2. cmake -DDEAL_II_DIR=~/dealii-dev/bin/  .
3. make debug or make release
4. make

### How to run
1. Go into one of the examples folders.
3. For 2D test cases, run via ./../SourceCode/main 2
4. For 3D test cases, run via ./../SourceCode/main 3

### How to expand this code
If you want to use the current code to solve new 2D or 3D phase-field crack problems, you need to do the following:
1. Add a new mesh under the function `void make_grid()`.
2. Add the boundary conditions for your new mesh in the function `void make_constraints(const unsigned int it_nr)`.
3. Modify the text file `timeDataFile` for the load/time step sizes and the text file `materialDataFile` for the material properties.
4. Modify the input file `parameters.prm` accordingly.

If you want to use a new phase-field degradation function (the current code uses the standard quadratic degradation function), you can modify the following functions `double degradation_function(const double d)`, `double degradation_function_derivative(const double d)`, and `double degradation_function_2nd_order_derivative(const double d)`.

If you want to modify the phase-field model completely but still use the L-BFGS monolithic solver, you need to modify the calculations of the initial BFGS matrix
```
void assemble_system_B0_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data) const;
```
and the residuals
```
void assemble_system_rhs_BFGS_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM_RHS_BFGS &                           scratch,
      PerTaskData_ASM_RHS_BFGS &                           data) const;
```

### Representative results
1. The combined effects of displacement-controlled shear load and thermal load:
<p align="center">
<img src="./doc/figure1.png" width="450">
</p>

2. Quenching test:
<p align="center">
<img src="./doc/figure2.png" width="450">
</p>

3. $L_2$-projection based adaptive mesh refinement
<p align="center">
<img src="./doc/figure3.png" width="450">
</p>

4. Comparison of the computational costs associated with the three solving schemes:
<p align="center">
<img src="./doc/tab1.png" width="800">
</p>

### How to cite this work
TBD...
