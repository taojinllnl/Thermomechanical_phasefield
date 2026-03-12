/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Tao Jin
 *         University of Ottawa, Ottawa, Ontario, Canada
 *         Oct. 2025
 *
 * How to cite:
 *         TBD
 */

/* A PURELY staggered scheme to solve the phase-field thermomechanically coupled
 * crack problem. The entire problem is decoupled into three subproblems, the damage
 * (phase-field) problem (d), the thermal problem (T), and the mechanical problem (u):
 * 1. The phase-field formulation itself is based on "A phase field model for rate-independent
 *    crack propagation - Robust algorithmic implementation based on operator splits"
 *    by Christian Miehe , Martina Hofacker, Fabian Welschinger.
 * 2. The thermal conductivity tensor is isotropic and degraded by the phase-field.
 * 3. The thermal equation is transient and considers the temperature
 *    changing with time (T_dot). The backward Euler time integrator is used.
 * 4. The mechanical problem is quasi-static and does not consider the inertial effort
 *    (no acceleration term).
 * 5. This code implements a PURELY staggered approach. The displacement, the temperature,
 *    and the phase-field are updated separately. The phase-field
 *    irreversibility is enforced through the history field Phi_0^+.
 * 6. Using TBB for stiffness assembly and Gauss point calculation.
 * 7. Using adaptive mesh refinement.
 * 8. The displacement is solved using Newton method, and the temperature problem
 *    and the phase-field problem are both linear.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>


#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparse_ilu.h>


#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>

#include "SpectrumDecomposition.h"
#include "Utilities.h"


namespace PhaseField_T_and_u_and_d
{
  using namespace dealii;

  // body force
  template <int dim>
  void right_hand_side(const std::vector<Point<dim>> &points,
		       std::vector<Tensor<1, dim>> &  values,
		       const double fx,
		       const double fy,
		       const double fz)
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
	if (dim == 2)
	  {
	    values[point_n][0] = fx;
	    values[point_n][1] = fy;
	  }
	else
	  {
	    values[point_n][0] = fx;
	    values[point_n][1] = fy;
	    values[point_n][2] = fz;
	  }
      }
  }

  // heat supply
  template <int dim>
  void heat_supply(const std::vector<Point<dim>> &points,
		   std::vector<double> &  values,
		   const double heat_supply)
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
	values[point_n] = heat_supply;
      }
  }

  double degradation_function(const double d)
  {
    return (1.0 - d) * (1.0 - d);
  }

  double degradation_function_derivative(const double d)
  {
    return 2.0 * (d - 1.0);
  }

  double degradation_function_2nd_order_derivative(const double d)
  {
    (void) d;
    return 2.0;
  }

  namespace Parameters
  {
    struct Scenario
    {
      unsigned int m_scenario;
      std::string m_logfile_name;
      bool m_output_iteration_history;
      bool m_coupling_on_heat_eq;
      bool m_degrade_conductivity;
      std::string m_type_nonlinear_solver;
      std::string m_type_linear_solver;
      std::string m_refinement_strategy;
      unsigned int m_global_refine_times;
      unsigned int m_local_prerefine_times;
      unsigned int m_max_adaptive_refine_times;
      int m_max_allowed_refinement_level;
      double m_phasefield_refine_threshold;
      double m_allowed_max_h_l_ratio;
      unsigned int m_total_material_regions;
      std::string m_material_file_name;
      int m_reaction_force_face_id;

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };

    void Scenario::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        prm.declare_entry("Scenario number",
                          "1",
                          Patterns::Integer(0),
                          "Geometry, loading and boundary conditions scenario");

        prm.declare_entry("Log file name",
			  "Output.log",
                          Patterns::FileName(Patterns::FileName::input),
			  "Name of the file for log");

        prm.declare_entry("Output iteration history",
			  "yes",
                          Patterns::Selection("yes|no"),
			  "Shall we write iteration history to the log file?");

        prm.declare_entry("Coupling on heat equation",
			  "no",
                          Patterns::Selection("yes|no"),
			  "Does the heat equation contain the coupling term?");

        prm.declare_entry("Degrade thermal conductivity",
			  "yes",
                          Patterns::Selection("yes|no"),
			  "Degrade thermal conductivity or not?");

        prm.declare_entry("Nonlinear solver type",
                          "Newton",
                          Patterns::Selection("Newton"),
                          "Type of solver used to solve the mechanical (u) nonlinear system");

        prm.declare_entry("Linear solver type",
                          "Direct",
                          Patterns::Selection("Direct|CG"),
                          "Type of solver used to solve the linear system");

        prm.declare_entry("Mesh refinement strategy",
                          "pre-refine",
                          Patterns::Selection("pre-refine|adaptive-refine"),
                          "Mesh refinement strategy: pre-refine or adaptive-refine");

        prm.declare_entry("Global refinement times",
                          "0",
                          Patterns::Integer(0),
                          "Global refinement times (across the entire domain)");

        prm.declare_entry("Local prerefinement times",
                          "0",
                          Patterns::Integer(0),
                          "Local pre-refinement times (assume crack path is known a priori), "
                          "only refine along the crack path.");

        prm.declare_entry("Max adaptive refinement times",
                          "100",
                          Patterns::Integer(0),
                          "Maximum number of adaptive refinement times allowed in each step");

        prm.declare_entry("Max allowed refinement level",
                          "100",
                          Patterns::Integer(0),
                          "Maximum allowed cell refinement level");

        prm.declare_entry("Phasefield refine threshold",
			  "0.8",
			  Patterns::Double(),
			  "Phasefield-based refinement threshold value");

        prm.declare_entry("Allowed max hl ratio",
			  "0.25",
			  Patterns::Double(),
			  "Allowed maximum ratio between mesh size h and length scale l");

        prm.declare_entry("Material regions",
                          "1",
                          Patterns::Integer(0),
                          "Number of material regions");

        prm.declare_entry("Material data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Material data file");

        prm.declare_entry("Reaction force face ID",
                          "1",
                          Patterns::Integer(),
                          "Face id where reaction forces should be calculated "
                          "(negative integer means not to calculate reaction force)");
      }
      prm.leave_subsection();
    }

    void Scenario::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        m_scenario = prm.get_integer("Scenario number");
        m_logfile_name = prm.get("Log file name");
        m_output_iteration_history = prm.get_bool("Output iteration history");
        m_coupling_on_heat_eq = prm.get_bool("Coupling on heat equation");
        m_degrade_conductivity = prm.get_bool("Degrade thermal conductivity");
        m_type_nonlinear_solver = prm.get("Nonlinear solver type");
        m_type_linear_solver = prm.get("Linear solver type");
        m_refinement_strategy = prm.get("Mesh refinement strategy");
        m_global_refine_times = prm.get_integer("Global refinement times");
        m_local_prerefine_times = prm.get_integer("Local prerefinement times");
        m_max_adaptive_refine_times = prm.get_integer("Max adaptive refinement times");
        m_max_allowed_refinement_level = prm.get_integer("Max allowed refinement level");
        m_phasefield_refine_threshold = prm.get_double("Phasefield refine threshold");
        m_allowed_max_h_l_ratio = prm.get_double("Allowed max hl ratio");
        m_total_material_regions = prm.get_integer("Material regions");
        m_material_file_name = prm.get("Material data file");
        m_reaction_force_face_id = prm.get_integer("Reaction force face ID");
      }
      prm.leave_subsection();
    }

    struct FESystem
    {
      unsigned int m_poly_degree;
      unsigned int m_quad_order;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };


    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "1",
                          Patterns::Integer(0),
                          "Phase field polynomial order");

        prm.declare_entry("Quadrature order",
                          "2",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        m_poly_degree = prm.get_integer("Polynomial degree");
        m_quad_order  = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    // body force (N/m^3)
    struct BodyForce
    {
      double m_x_component;
      double m_y_component;
      double m_z_component;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void BodyForce::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        prm.declare_entry("Body force x component",
			  "0.0",
			  Patterns::Double(),
			  "Body force x-component (N/m^3)");

        prm.declare_entry("Body force y component",
			  "0.0",
			  Patterns::Double(),
			  "Body force y-component (N/m^3)");

        prm.declare_entry("Body force z component",
			  "0.0",
			  Patterns::Double(),
			  "Body force z-component (N/m^3)");
      }
      prm.leave_subsection();
    }

    void BodyForce::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        m_x_component = prm.get_double("Body force x component");
        m_y_component = prm.get_double("Body force y component");
        m_z_component = prm.get_double("Body force z component");
      }
      prm.leave_subsection();
    }


    // heat supply (Watt/m^3)
    struct HeatSupply
    {
      double m_heat_supply;
      double m_ref_temperature;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void HeatSupply::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Heat supply");
      {
        prm.declare_entry("Heat supply",
			  "0.0",
			  Patterns::Double(),
			  "Heat supply (Watt/m^3)");

        prm.declare_entry("Reference temperature",
			  "300.0",
			  Patterns::Double(),
			  "Reference temperature (K)");
      }
      prm.leave_subsection();
    }

    void HeatSupply::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Heat supply");
      {
        m_heat_supply = prm.get_double("Heat supply");
        m_ref_temperature = prm.get_double("Reference temperature");
      }
      prm.leave_subsection();
    }

    struct NonlinearSolver
    {
      unsigned int m_max_staggered_iteration;
      unsigned int m_max_iterations_u;

      double       m_tol_u_residual;
      double       m_tol_t_residual;
      double       m_tol_d_residual;

      double       m_tol_u_incr;
      double       m_tol_t_incr;
      double       m_tol_d_incr;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max staggered iteration",
                          "1000",
                          Patterns::Integer(0),
                          "Maximum allowed number of staggered iterations");

        prm.declare_entry("Max iterations U",
                          "20",
                          Patterns::Integer(0),
                          "Number of Newton iterations allowed for the U subproblem");

        prm.declare_entry("Tolerance u residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Mechanical residual tolerance");

        prm.declare_entry("Tolerance T residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Thermal residual tolerance");

        prm.declare_entry("Tolerance phasefield residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield residual tolerance");

        prm.declare_entry("Tolerance u increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Mechanical increment tolerance");

        prm.declare_entry("Tolerance T increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Thermal increment tolerance");

        prm.declare_entry("Tolerance phasefield increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield increment tolerance");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
	m_max_staggered_iteration = prm.get_integer("Max staggered iteration");
        m_max_iterations_u = prm.get_integer("Max iterations U");

        m_tol_u_residual = prm.get_double("Tolerance u residual");
        m_tol_t_residual = prm.get_double("Tolerance T residual");
        m_tol_d_residual  = prm.get_double("Tolerance phasefield residual");

        m_tol_u_incr = prm.get_double("Tolerance u increment");
        m_tol_t_incr = prm.get_double("Tolerance T increment");
        m_tol_d_incr  = prm.get_double("Tolerance phasefield increment");
      }
      prm.leave_subsection();
    }

    struct TimeInfo
    {
      double m_end_time;
      std::string m_time_file_name;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void TimeInfo::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Time data file");
      }
      prm.leave_subsection();
    }

    void TimeInfo::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        m_end_time = prm.get_double("End time");
        m_time_file_name = prm.get("Time data file");
      }
      prm.leave_subsection();
    }

    struct AllParameters : public Scenario,
	                   public FESystem,
	                   public BodyForce,
			   public HeatSupply,
			   public NonlinearSolver,
			   public TimeInfo
    {
      AllParameters(const std::string &input_file);

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Scenario::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      BodyForce::declare_parameters(prm);
      HeatSupply::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      TimeInfo::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Scenario::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      BodyForce::parse_parameters(prm);
      HeatSupply::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      TimeInfo::parse_parameters(prm);
    }
  } // namespace Parameters

  class Time
  {
  public:
    Time(const double time_end)
      : m_timestep(0)
      , m_time_current(0.0)
      , m_time_end(time_end)
      , m_delta_t(0.0)
      , m_magnitude(1.0)
    {}

    virtual ~Time() = default;

    double current() const
    {
      return m_time_current;
    }
    double end() const
    {
      return m_time_end;
    }
    double get_delta_t() const
    {
      return m_delta_t;
    }
    double get_magnitude() const
    {
      return m_magnitude;
    }
    unsigned int get_timestep() const
    {
      return m_timestep;
    }
    void increment(std::vector<std::array<double, 4>> time_table)
    {
      double t_1, t_delta, t_magnitude;
      for (auto & time_group : time_table)
        {
	  t_1 = time_group[1];
	  t_delta = time_group[2];
	  t_magnitude = time_group[3];

	  if (m_time_current < t_1 - 1.0e-6*t_delta)
	    {
	      m_delta_t = t_delta;
	      m_magnitude = t_magnitude;
	      break;
	    }
        }

      m_time_current += m_delta_t;
      ++m_timestep;
    }

  private:
    unsigned int m_timestep;
    double       m_time_current;
    const double m_time_end;
    double m_delta_t;
    double m_magnitude;
  };

  template <int dim>
  class LinearIsotropicElasticityAdditiveSplit
  {
  public:
    LinearIsotropicElasticityAdditiveSplit(const double lame_lambda,
			                   const double lame_mu,
				           const double residual_k,
					   const double length_scale,
					   const double viscosity,
					   const double gc_0,
					   const double heat_capacity,
					   const double thermal_conductivity_0,
					   const double thermal_expansion_coeff,
					   const double reference_temperature,
					   const double max_temperature,
					   const double b_1,
					   const double b_2)
      : m_lame_lambda(lame_lambda)
      , m_lame_mu(lame_mu)
      , m_residual_k(residual_k)
      , m_length_scale(length_scale)
      , m_eta(viscosity)
      , m_gc_0(gc_0)
      , m_heat_capacity(heat_capacity)
      , m_kappa_0(thermal_conductivity_0)
      , m_alpha(thermal_expansion_coeff)
      , m_ref_t(reference_temperature)
      , m_max_t(max_temperature)
      , m_b_1(b_1)
      , m_b_2(b_2)
      , m_phase_field_value(0.0)
      , m_grad_phasefield(Tensor<1, dim>())
      , m_strain(SymmetricTensor<2, dim>())
      , m_stress(SymmetricTensor<2, dim>())
      , m_stress_positive(SymmetricTensor<2, dim>())
      , m_mechanical_C(SymmetricTensor<4, dim>())
      , m_strain_energy_positive(0.0)
      , m_strain_energy_negative(0.0)
      , m_strain_energy_total(0.0)
      , m_crack_energy_dissipation(0.0)
      , m_gc_t(0.0)
      , m_kappa_d(0.0)
      , m_temperature(0.0)
      , m_grad_temperature(Tensor<1, dim>())
      , m_heat_flux(Tensor<1, dim>())
    {
      Assert(  ( lame_lambda / (2*(lame_lambda + lame_mu)) <= 0.5)
	     & ( lame_lambda / (2*(lame_lambda + lame_mu)) >=-1.0),
	     ExcInternalError() );
    }

    const SymmetricTensor<4, dim> & get_mechanical_C() const
    {
      return m_mechanical_C;
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress() const
    {
      return m_stress;
    }

    const SymmetricTensor<2, dim> & get_strain() const
    {
      return m_strain;
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress_positive() const
    {
      return m_stress_positive;
    }

    double get_positive_strain_energy() const
    {
      return m_strain_energy_positive;
    }

    double get_negative_strain_energy() const
    {
      return m_strain_energy_negative;
    }

    double get_total_strain_energy() const
    {
      return m_strain_energy_total;
    }

    double get_crack_energy_dissipation() const
    {
      return m_crack_energy_dissipation;
    }

    double get_phase_field_value() const
    {
      return m_phase_field_value;
    }

    double get_thermal_expansion_coeff() const
    {
      return m_alpha;
    }

    double get_lame_lambda() const
    {
      return m_lame_lambda;
    }

    double get_lame_mu() const
    {
      return m_lame_mu;
    }

    const Tensor<1, dim> get_phase_field_gradient() const
    {
      return m_grad_phasefield;
    }

    double get_temperature_value() const
    {
      return m_temperature;
    }

    double get_ref_temperature() const
    {
      return m_ref_t;
    }

    const Tensor<1, dim> get_temperature_gradient() const
    {
      return m_grad_temperature;
    }

    const Tensor<1, dim> get_heat_flux() const
    {
      return m_heat_flux;
    }

    // temperature-dependent critical energy release rate
    double get_critical_energy_release_rate_temperature() const
    {
      return m_gc_t;
    }

    double get_thermal_conductivity_degraded() const
    {
      return m_kappa_d;
    }

    void update_material_data(const SymmetricTensor<2, dim> & strain,
			      const double phase_field_value,
			      const Tensor<1, dim> & grad_phasefield,
			      const double phase_field_value_previous_step,
			      const double delta_time,
			      const double temperature,
			      const Tensor<1, dim> & grad_temperature,
			      const bool degrade_conductivity_or_not)
    {
      // Total strain grad^{(s)}u
      m_strain = strain;
      m_phase_field_value = phase_field_value;
      m_grad_phasefield = grad_phasefield;
      m_temperature = temperature;
      m_grad_temperature = grad_temperature;

      // Thermal strain
      SymmetricTensor<2, dim> strain_t;
      strain_t = m_alpha * (temperature - m_ref_t)
	                 * Physics::Elasticity::StandardTensors<dim>::I;

      // Effective strain
      SymmetricTensor<2, dim> strain_e;
      strain_e = m_strain - strain_t;

      // temperature-dependent gc
      double term_1 = (temperature - m_ref_t) / m_max_t;
      double coeff = 1.0 - m_b_1 * term_1
	                 + m_b_2 * term_1 * term_1;
      m_gc_t = coeff * m_gc_0;

      Vector<double>              eigenvalues(dim);
      std::vector<Tensor<1, dim>> eigenvectors(dim);
      usr_spectrum_decomposition::spectrum_decomposition<dim>(strain_e,
    							      eigenvalues,
    							      eigenvectors);

      SymmetricTensor<2, dim> strain_positive, strain_negative;
      strain_positive = usr_spectrum_decomposition::positive_tensor(eigenvalues, eigenvectors);
      strain_negative = usr_spectrum_decomposition::negative_tensor(eigenvalues, eigenvectors);

      SymmetricTensor<4, dim> projector_positive, projector_negative;
      usr_spectrum_decomposition::positive_negative_projectors(eigenvalues,
    							       eigenvectors,
							       projector_positive,
							       projector_negative);

      SymmetricTensor<2, dim> stress_positive, stress_negative;
      const double degradation = degradation_function(m_phase_field_value);
      const double I_1 = trace(strain_e);
      stress_positive = m_lame_lambda * usr_spectrum_decomposition::positive_ramp_function(I_1)
                                      * Physics::Elasticity::StandardTensors<dim>::I
                      + 2 * m_lame_mu * strain_positive;
      stress_negative = m_lame_lambda * usr_spectrum_decomposition::negative_ramp_function(I_1)
                                      * Physics::Elasticity::StandardTensors<dim>::I
      		      + 2 * m_lame_mu * strain_negative;

      m_stress = degradation * stress_positive + stress_negative;
      m_stress_positive = stress_positive;

      SymmetricTensor<4, dim> C_positive, C_negative;
      C_positive = m_lame_lambda * usr_spectrum_decomposition::heaviside_function(I_1)
                                 * Physics::Elasticity::StandardTensors<dim>::IxI
		 + 2 * m_lame_mu * projector_positive;
      C_negative = m_lame_lambda * usr_spectrum_decomposition::heaviside_function(-I_1)
                                 * Physics::Elasticity::StandardTensors<dim>::IxI
      		 + 2 * m_lame_mu * projector_negative;
      m_mechanical_C = degradation * C_positive + C_negative;

      m_strain_energy_positive = 0.5 * m_lame_lambda * usr_spectrum_decomposition::positive_ramp_function(I_1)
                                                     * usr_spectrum_decomposition::positive_ramp_function(I_1)
                               + m_lame_mu * strain_positive * strain_positive;

      m_strain_energy_negative = 0.5 * m_lame_lambda * usr_spectrum_decomposition::negative_ramp_function(I_1)
                                                     * usr_spectrum_decomposition::negative_ramp_function(I_1)
                               + m_lame_mu * strain_negative * strain_negative;

      m_strain_energy_total = degradation * m_strain_energy_positive + m_strain_energy_negative;

      // The critical energy release rate m_gc should be temperature-dependent.
      m_crack_energy_dissipation = m_gc_t * (  0.5 / m_length_scale * m_phase_field_value * m_phase_field_value
	                                   + 0.5 * m_length_scale * m_grad_phasefield * m_grad_phasefield)
	                                   // the term due to viscosity regularization
	                                   + (m_phase_field_value - phase_field_value_previous_step)
					   * (m_phase_field_value - phase_field_value_previous_step)
				           * 0.5 * m_eta / delta_time;

      // degraded thermal conductivity or not
      if (degrade_conductivity_or_not)
        m_kappa_d = (degradation + m_residual_k) * m_kappa_0;
      else
	m_kappa_d = 1.0 * m_kappa_0;

      // heat flux
      m_heat_flux = - m_kappa_d * grad_temperature;

      //(void)delta_time;
      //(void)phase_field_value_previous_step;
    }

  private:
    const double m_lame_lambda;
    const double m_lame_mu;
    const double m_residual_k;
    const double m_length_scale;
    const double m_eta;
    const double m_gc_0;
    const double m_heat_capacity;
    const double m_kappa_0;
    const double m_alpha;
    const double m_ref_t;
    const double m_max_t;
    const double m_b_1;
    const double m_b_2;
    double m_phase_field_value;
    Tensor<1, dim> m_grad_phasefield;
    SymmetricTensor<2, dim> m_strain;
    SymmetricTensor<2, dim> m_stress;
    SymmetricTensor<2, dim> m_stress_positive;
    SymmetricTensor<4, dim> m_mechanical_C;
    double m_strain_energy_positive;
    double m_strain_energy_negative;
    double m_strain_energy_total;
    double m_crack_energy_dissipation;
    // temperature-dependent critical energy release rate
    double m_gc_t;
    // degraded thermal conductivity
    double m_kappa_d;
    double m_temperature;
    Tensor<1, dim> m_grad_temperature;
    Tensor<1, dim> m_heat_flux;
  };


  template <int dim>
  class PointHistory
  {
  public:
    PointHistory()
      : m_length_scale(0.0)
      , m_viscosity(0.0)
      , m_history_max_positive_strain_energy(0.0)
      , m_heat_capacity(0.0)
      , m_coupling_on_heat_eq(false)
    {}

    virtual ~PointHistory() = default;

    void setup_lqp(const double lame_lambda,
		   const double lame_mu,
		   const double length_scale,
		   const double gc_0,
		   const double viscosity,
		   const double residual_k,
		   const double heat_capacity,
		   const double thermal_conductivity_0,
		   const double thermal_expansion_coeff,
		   const double reference_temperature,
		   const double max_temperature,
		   const double b_1,
		   const double b_2,
		   const bool   coupling_on_heat_eq)
    {
      m_material =
              std::make_shared<LinearIsotropicElasticityAdditiveSplit<dim>>(lame_lambda,
        	                                                            lame_mu,
								            residual_k,
									    length_scale,
									    viscosity,
									    gc_0,
									    heat_capacity,
									    thermal_conductivity_0,
									    thermal_expansion_coeff,
									    reference_temperature,
									    max_temperature,
									    b_1,
									    b_2);
      m_history_max_positive_strain_energy = 0.0;
      m_length_scale = length_scale;
      m_viscosity = viscosity;
      m_heat_capacity = heat_capacity;
      m_coupling_on_heat_eq = coupling_on_heat_eq;

      update_field_values(SymmetricTensor<2, dim>(), 0.0, Tensor<1, dim>(),
			  0.0, 1.0, reference_temperature, Tensor<1, dim>(), true);
    }

    void update_field_values(const SymmetricTensor<2, dim> & strain,
		             const double phase_field_value,
			     const Tensor<1, dim> & grad_phasefield,
			     const double phase_field_value_previous_step,
			     const double delta_time,
			     const double temperature,
			     const Tensor<1, dim> & grad_temperature,
			     const bool degrade_conductivity_or_not)
    {
      m_material->update_material_data(strain, phase_field_value, grad_phasefield,
				       phase_field_value_previous_step, delta_time,
				       temperature, grad_temperature,
				       degrade_conductivity_or_not);
    }

    void update_history_variable()
    {
      double current_positive_strain_energy = m_material->get_positive_strain_energy();
      m_history_max_positive_strain_energy = std::fmax(m_history_max_positive_strain_energy,
					               current_positive_strain_energy);
    }

    // This is the function used to assign the history variable after remeshing
    void assign_history_variable(double history_variable_value)
    {
      m_history_max_positive_strain_energy = history_variable_value;
    }

    double get_current_positive_strain_energy() const
    {
      return m_material->get_positive_strain_energy();
    }

    const SymmetricTensor<4, dim> & get_mechanical_C() const
    {
      return m_material->get_mechanical_C();
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress() const
    {
      return m_material->get_cauchy_stress();
    }

    const SymmetricTensor<2, dim> & get_strain() const
    {
      return m_material->get_strain();
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress_positive() const
    {
      return m_material->get_cauchy_stress_positive();
    }

    double get_total_strain_energy() const
    {
      return m_material->get_total_strain_energy();
    }

    double get_crack_energy_dissipation() const
    {
      return m_material->get_crack_energy_dissipation();
    }

    double get_phase_field_value() const
    {
      return m_material->get_phase_field_value();
    }

    double get_temperature_value() const
    {
      return m_material->get_temperature_value();
    }

    double get_ref_temperature() const
    {
      return m_material->get_ref_temperature();
    }

    const Tensor<1, dim> get_phase_field_gradient() const
    {
      return m_material->get_phase_field_gradient();
    }

    const Tensor<1, dim> get_temperature_gradient() const
    {
      return m_material->get_temperature_gradient();
    }

    const Tensor<1, dim> get_heat_flux() const
    {
      return m_material->get_heat_flux();
    }

    // return the temperature-dependent gc
    double get_critical_energy_release_rate() const
    {
      return m_material->get_critical_energy_release_rate_temperature();
    }

    double get_thermal_conductivity() const
    {
      return m_material->get_thermal_conductivity_degraded();
    }

    double get_history_max_positive_strain_energy() const
    {
      return m_history_max_positive_strain_energy;
    }

    double get_length_scale() const
    {
      return m_length_scale;
    }

    double get_viscosity() const
    {
      return m_viscosity;
    }

    double get_heat_capacity() const
    {
      return m_heat_capacity;
    }

    bool get_heat_coupling_flag() const
    {
      return m_coupling_on_heat_eq;
    }

    double get_thermal_expansion_coeff() const
    {
      return m_material->get_thermal_expansion_coeff();
    }

    double get_lame_lambda() const
    {
      return m_material->get_lame_lambda();
    }

    double get_lame_mu() const
    {
      return m_material->get_lame_mu();
    }
  private:
    std::shared_ptr<LinearIsotropicElasticityAdditiveSplit<dim>> m_material;
    double m_length_scale;
    double m_viscosity;
    double m_history_max_positive_strain_energy;
    double m_heat_capacity;
    bool   m_coupling_on_heat_eq;
  };

  template <int dim>
  class SplitSolveTandUandD
  {
  public:
    SplitSolveTandUandD(const std::string &input_file);

    virtual ~SplitSolveTandUandD() = default;
    void run();

  private:
    using IteratorTuple3 =
      std::tuple<typename DoFHandler<dim>::active_cell_iterator,
                 typename DoFHandler<dim>::active_cell_iterator,
		 typename DoFHandler<dim>::active_cell_iterator>;

    using IteratorPair3 = SynchronousIterators<IteratorTuple3>;

    using IteratorTuple2 =
      std::tuple<typename DoFHandler<dim>::active_cell_iterator,
		 typename DoFHandler<dim>::active_cell_iterator>;

    using IteratorPair2 = SynchronousIterators<IteratorTuple2>;

    struct PerTaskData_ASM_U;
    struct ScratchData_ASM_U;

    struct PerTaskData_RHS_U;
    struct ScratchData_RHS_U;

    struct PerTaskData_ASM_T;
    struct ScratchData_ASM_T;

    struct PerTaskData_RHS_T;
    struct ScratchData_RHS_T;

    struct PerTaskData_ASM_D;
    struct ScratchData_ASM_D;

    struct PerTaskData_RHS_D;
    struct ScratchData_RHS_D;

    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    Parameters::AllParameters m_parameters;
    Triangulation<dim> m_triangulation;

    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim>> m_quadrature_point_history;

    Time m_time;
    std::ofstream m_logfile;
    mutable TimerOutput m_timer;

    DoFHandler<dim> m_dof_handler_u;
    DoFHandler<dim> m_dof_handler_t;
    DoFHandler<dim> m_dof_handler_d;

    FESystem<dim> m_fe_u;
    FESystem<dim> m_fe_t;
    FESystem<dim> m_fe_d;

    const QGauss<dim>     m_qf_cell;
    const QGauss<dim - 1> m_qf_face;
    const unsigned int    m_n_q_points;

    double m_vol_reference;

    AffineConstraints<double> m_constraints_u;
    SparsityPattern      m_sparsity_pattern_u;
    SparseMatrix<double> m_tangent_matrix_u;
    Vector<double>       m_system_rhs_u;
    Vector<double>       m_solution_u;
    Vector<double>       m_solution_previous_u;

    AffineConstraints<double> m_constraints_t;
    SparsityPattern m_sparsity_pattern_t;
    SparseMatrix<double> m_tangent_matrix_t;
    Vector<double> m_system_rhs_t;
    Vector<double> m_solution_t;
    Vector<double> m_solution_previous_t;

    AffineConstraints<double> m_constraints_d;
    SparsityPattern m_sparsity_pattern_d;
    SparseMatrix<double> m_tangent_matrix_d;
    Vector<double> m_system_rhs_d;
    Vector<double> m_solution_d;
    Vector<double> m_solution_previous_d;

    std::map<unsigned int, std::vector<double>> m_material_data;

    std::vector<std::pair<double, std::vector<double>>> m_history_reaction_force;
    std::vector<std::pair<double, std::array<double, 3>>> m_history_energy;

    void make_grid();
    void make_grid_case_1();
    void make_grid_case_2();
    void make_grid_case_3();
    void make_grid_case_4();
    void make_grid_case_5();
    void make_grid_case_6();

    void setup_system();
    void setup_system_u();
    void setup_system_t();
    void setup_system_d();

    void setup_temperature_initial_conditions();

    unsigned int displacement_step(unsigned int itr_stagger);
    void make_constraints_u(const unsigned int it_nr,
			    const unsigned int itr_stagger);
    void assemble_system_u();
    void assemble_system_one_cell_u(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM_U                                    &scratch,
      PerTaskData_ASM_U                                    &data) const;
    void assemble_rhs_u();
    void assemble_rhs_one_cell_u(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_RHS_U &                           scratch,
      PerTaskData_RHS_U &                           data) const;
    unsigned int solve_nonlinear_newton_u(Vector<double> & solution_delta_u,
					  unsigned int itr_stagger);
    void solve_linear_system_u(Vector<double> & newton_update_u);

    void temperature_step();
    void make_constraints_t();
    void assemble_system_t();
    void assemble_system_one_cell_t(const IteratorPair2 & synchronous_iterators,
                                    ScratchData_ASM_T & scratch,
                                    PerTaskData_ASM_T & data) const;
    void assemble_rhs_t();
    void assemble_rhs_one_cell_t(const IteratorPair2 & synchronous_iterators,
                                 ScratchData_RHS_T & scratch,
                                 PerTaskData_RHS_T & data) const;
    void solve_linear_system_t();

    void phasefield_step();
    void make_constraints_d();
    void assemble_system_d();
    void assemble_system_one_cell_d(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                    ScratchData_ASM_D & scratch,
                                    PerTaskData_ASM_D & data) const;
    void assemble_rhs_d();
    void assemble_rhs_one_cell_d(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 ScratchData_RHS_D & scratch,
                                 PerTaskData_RHS_D & data) const;
    void solve_linear_system_d();

    void update_history_field_step();

    void output_results() const;

    void setup_qph();

    void update_qph_incremental(const Vector<double> &solution_delta_u);

    void update_qph_incremental_one_cell(
      const IteratorPair3 & synchronous_iterators,
      ScratchData_UQPH   & scratch,
      PerTaskData_UQPH   & data);

    void copy_local_to_global_UQPH(const PerTaskData_UQPH & /*data*/)
    {}

    bool local_refine_and_solution_transfer();

    Vector<double>
    get_total_solution_u(const Vector<double> &solution_delta_u) const;

    // Should not make this function const
    void read_material_data(const std::string &data_file,
			    const unsigned int total_material_regions);

    void read_time_data(const std::string &data_file,
    		        std::vector<std::array<double, 4>> & time_table);

    void print_conv_header();

    void print_parameter_information();

    void calculate_reaction_force(unsigned int face_ID);

    void write_history_data();

    double calculate_energy_functional() const;

    std::pair<double, double> calculate_total_strain_energy_and_crack_energy_dissipation() const;
  }; // class SplitSolveTandUandD

  template <int dim>
  void SplitSolveTandUandD<dim>::read_material_data(const std::string &data_file,
				                 const unsigned int total_material_regions)
  {
    std::ifstream myfile (data_file);

    double lame_lambda, lame_mu, length_scale, gc_0, viscosity, residual_k;
    double heat_capacity, thermal_conductivity_0, thermal_expansion_coeff;
    double reference_temperature, max_temperature, b_1, b_2;
    int material_region;
    double poisson_ratio;
    if (myfile.is_open())
      {
        m_logfile << "Reading material data file ..." << std::endl;

        while ( myfile >> material_region
                       >> lame_lambda
		       >> lame_mu
		       >> length_scale
		       >> gc_0
		       >> viscosity
		       >> residual_k
		       >> heat_capacity
		       >> thermal_conductivity_0
		       >> thermal_expansion_coeff
		       >> reference_temperature
		       >> max_temperature
		       >> b_1
		       >> b_2)
          {
            m_material_data[material_region] = {lame_lambda,
        	                                lame_mu,
						length_scale,
						gc_0,
						viscosity,
                                                residual_k,
                                                heat_capacity,
                                                thermal_conductivity_0,
                                                thermal_expansion_coeff,
                                                reference_temperature,
                                                max_temperature,
                                                b_1,
                                                b_2};
            poisson_ratio = lame_lambda / (2*(lame_lambda + lame_mu));
            Assert( (poisson_ratio <= 0.5)&(poisson_ratio >=-1.0) , ExcInternalError());

            if (reference_temperature != m_parameters.m_ref_temperature)
              Assert(false, ExcMessage("Reference temperature inconsistent "
        	  "in the parameters.prm file and materialDataFile"));

            m_logfile << "\tRegion " << material_region << " : " << std::endl;
            m_logfile << "\t\tLame lambda = " << lame_lambda << std::endl;
            m_logfile << "\t\tLame mu = "  << lame_mu << std::endl;
            m_logfile << "\t\tPoisson ratio = "  << poisson_ratio << std::endl;
            m_logfile << "\t\tPhase field length scale (l) = " << length_scale << std::endl;
            m_logfile << "\t\tCritical energy release rate (gc_0) = "  << gc_0 << std::endl;
            m_logfile << "\t\tViscosity for regularization (eta) = "  << viscosity << std::endl;
            m_logfile << "\t\tResidual_k (k) = "  << residual_k << std::endl;
            m_logfile << "\t\tHeat capacity (c, density * specific capacity) = "  << heat_capacity << std::endl;
            m_logfile << "\t\tThermal conductivity (kappa_0) = "  << thermal_conductivity_0 << std::endl;
            m_logfile << "\t\tThermal expansion coeff (alpha) = "  << thermal_expansion_coeff << std::endl;
            m_logfile << "\t\tReference temperature (T_0) = "  << reference_temperature << std::endl;
            m_logfile << "\t\tMax temperature (T_max) = "  << max_temperature << std::endl;
            m_logfile << "\t\tb1 (temperature dependent coeff) = "  << b_1 << std::endl;
            m_logfile << "\t\tb2 (temperature dependent coeff) = "  << b_2 << std::endl;
          }

        if (m_material_data.size() != total_material_regions)
          {
            m_logfile << "Material data file has " << m_material_data.size() << " rows. However, "
        	      << "the mesh has " << total_material_regions << " material regions."
		      << std::endl;
            Assert(m_material_data.size() == total_material_regions,
                       ExcDimensionMismatch(m_material_data.size(), total_material_regions));
          }
        myfile.close();
      }
    else
      {
	m_logfile << "Material data file : " << data_file << " not exist!" << std::endl;
	Assert(false, ExcMessage("Failed to read material data file"));
      }
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::read_time_data(const std::string &data_file,
				             std::vector<std::array<double, 4>> & time_table)
  {
    std::ifstream myfile (data_file);

    double t_0, t_1, delta_t, t_magnitude;

    if (myfile.is_open())
      {
	m_logfile << "Reading time data file ..." << std::endl;

	while ( myfile >> t_0
		       >> t_1
		       >> delta_t
		       >> t_magnitude)
	  {
	    Assert( t_0 < t_1,
		    ExcMessage("For each time pair, "
			       "the start time should be smaller than the end time"));
	    time_table.push_back({{t_0, t_1, delta_t, t_magnitude}});
	  }

	Assert(std::fabs(t_1 - m_parameters.m_end_time) < 1.0e-9,
	       ExcMessage("End time in time table is inconsistent with input data in parameters.prm"));

	Assert(time_table.size() > 0,
	       ExcMessage("Time data file is empty."));
	myfile.close();
      }
    else
      {
        m_logfile << "Time data file : " << data_file << " not exist!" << std::endl;
        Assert(false, ExcMessage("Failed to read time data file"));
      }

    for (auto & time_group : time_table)
      {
	m_logfile << "\t\t"
	          << time_group[0] << ",\t"
	          << time_group[1] << ",\t"
		  << time_group[2] << ",\t"
		  << time_group[3] << std::endl;
      }
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::setup_qph()
  {
    m_logfile << "\t\tSetting up quadrature point data ("
	      << m_n_q_points
	      << " points per cell)" << std::endl;

    m_quadrature_point_history.clear();
    for (auto const & cell : m_triangulation.active_cell_iterators())
      {
	m_quadrature_point_history.initialize(cell, m_n_q_points);
      }

    unsigned int material_id;
    double lame_lambda = 0.0;
    double lame_mu = 0.0;
    double length_scale = 0.0;
    double gc_0 = 0.0;
    double viscosity = 0.0;
    double residual_k = 0.0;
    double heat_capacity = 0.0;
    double thermal_conductivity_0 = 0.0;
    double thermal_expansion_coeff = 0.0;
    double reference_temperature = 0.0;
    double max_temperature = 0.0;
    double b_1 = 0.0;
    double b_2 = 0.0;

    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
        material_id = cell->material_id();
        if (m_material_data.find(material_id) != m_material_data.end())
          {
            lame_lambda                = m_material_data[material_id][0];
            lame_mu                    = m_material_data[material_id][1];
            length_scale               = m_material_data[material_id][2];
            gc_0                       = m_material_data[material_id][3];
            viscosity                  = m_material_data[material_id][4];
            residual_k                 = m_material_data[material_id][5];
            heat_capacity              = m_material_data[material_id][6];
            thermal_conductivity_0     = m_material_data[material_id][7];
            thermal_expansion_coeff    = m_material_data[material_id][8];
            reference_temperature      = m_material_data[material_id][9];
            max_temperature            = m_material_data[material_id][10];
            b_1                        = m_material_data[material_id][11];
            b_2                        = m_material_data[material_id][12];
	  }
        else
          {
            m_logfile << "Could not find material data for material id: " << material_id << std::endl;
            AssertThrow(false, ExcMessage("Could not find material data for material id."));
          }

        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(lame_lambda, lame_mu, length_scale,
				   gc_0, viscosity, residual_k,
				   heat_capacity, thermal_conductivity_0,
				   thermal_expansion_coeff, reference_temperature,
				   max_temperature, b_1, b_2,
				   m_parameters.m_coupling_on_heat_eq);
      }
  }

  template <int dim>
  Vector<double> SplitSolveTandUandD<dim>::get_total_solution_u(
    const Vector<double> &solution_delta_u) const
  {
    Vector<double> solution_total_u(m_solution_u);
    solution_total_u += solution_delta_u;
    return solution_total_u;
  }

  template <int dim>
  void
  SplitSolveTandUandD<dim>::update_qph_incremental(const Vector<double> &solution_delta_u)
  {
    m_timer.enter_subsection("Update QPH data");

    const Vector<double> solution_total_u(get_total_solution_u(solution_delta_u));

    const UpdateFlags uf_UQPH(update_values | update_gradients);
    PerTaskData_UQPH  per_task_data_UQPH;
    ScratchData_UQPH  scratch_data_UQPH(m_fe_u,
					m_fe_t,
					m_fe_d,
					m_qf_cell,
					uf_UQPH,
					solution_total_u,
					m_solution_t,
					m_solution_d,
					m_solution_previous_d,
					m_time.get_delta_t(),
					m_parameters.m_degrade_conductivity);

    auto worker = [this](const IteratorPair3 & synchronous_iterators,
	                 ScratchData_UQPH & scratch,
	                 PerTaskData_UQPH & data)
      {
        this->update_qph_incremental_one_cell(synchronous_iterators, scratch, data);
      };

    auto copier = [this](const PerTaskData_UQPH &data)
      {
        this->copy_local_to_global_UQPH(data);
      };

    WorkStream::run(
	IteratorPair3(IteratorTuple3(m_dof_handler_u.begin_active(),
				     m_dof_handler_t.begin_active(),
				     m_dof_handler_d.begin_active())),
	IteratorPair3(IteratorTuple3(m_dof_handler_u.end(),
				     m_dof_handler_t.end(),
				     m_dof_handler_d.end())),
	worker,
	copier,
	scratch_data_UQPH,
	per_task_data_UQPH);

    m_timer.leave_subsection();
  }

  template <int dim>
  struct SplitSolveTandUandD<dim>::PerTaskData_UQPH
  {
    void reset()
    {}
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::ScratchData_UQPH
  {
    const Vector<double> & m_solution_u_UQPH;
    const Vector<double> & m_solution_t_UQPH;
    const Vector<double> & m_solution_d_UQPH;
    const Vector<double> & m_solution_previous_d_UQPH;

    std::vector<SymmetricTensor<2, dim>> m_solution_symm_grads_u_cell;

    std::vector<double>         m_solution_values_phasefield_cell;
    std::vector<Tensor<1, dim>> m_solution_grad_phasefield_cell;
    std::vector<double>         m_phasefield_previous_step_cell;

    std::vector<double>         m_solution_values_temperature_cell;
    std::vector<Tensor<1, dim>> m_solution_grad_temperature_cell;

    FEValues<dim> m_fe_values_u;
    FEValues<dim> m_fe_values_t;
    FEValues<dim> m_fe_values_d;

    const double m_delta_time;

    const bool m_degrade_conductivity_or_not;

    ScratchData_UQPH(const FiniteElement<dim> & fe_cell_u,
		     const FiniteElement<dim> & fe_cell_t,
		     const FiniteElement<dim> & fe_cell_d,
                     const QGauss<dim> &        qf_cell,
                     const UpdateFlags          uf_cell,
                     const Vector<double>      &solution_total_u,
		     const Vector<double>      &solution_total_t,
		     const Vector<double>      &solution_total_d,
		     const Vector<double>      &solution_previous_d,
		     const double delta_time,
		     const bool degrade_conductivity_or_not)
      : m_solution_u_UQPH(solution_total_u)
      , m_solution_t_UQPH(solution_total_t)
      , m_solution_d_UQPH(solution_total_d)
      , m_solution_previous_d_UQPH(solution_previous_d)
      , m_solution_symm_grads_u_cell(qf_cell.size())
      , m_solution_values_phasefield_cell(qf_cell.size())
      , m_solution_grad_phasefield_cell(qf_cell.size())
      , m_phasefield_previous_step_cell(qf_cell.size())
      , m_solution_values_temperature_cell(qf_cell.size())
      , m_solution_grad_temperature_cell(qf_cell.size())
      , m_fe_values_u(fe_cell_u, qf_cell, uf_cell)
      , m_fe_values_t(fe_cell_t, qf_cell, uf_cell)
      , m_fe_values_d(fe_cell_d, qf_cell, uf_cell)
      , m_delta_time(delta_time)
      , m_degrade_conductivity_or_not(degrade_conductivity_or_not)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      : m_solution_u_UQPH(rhs.m_solution_u_UQPH)
      , m_solution_t_UQPH(rhs.m_solution_t_UQPH)
      , m_solution_d_UQPH(rhs.m_solution_d_UQPH)
      , m_solution_previous_d_UQPH(rhs.m_solution_previous_d_UQPH)
      , m_solution_symm_grads_u_cell(rhs.m_solution_symm_grads_u_cell)
      , m_solution_values_phasefield_cell(rhs.m_solution_values_phasefield_cell)
      , m_solution_grad_phasefield_cell(rhs.m_solution_grad_phasefield_cell)
      , m_phasefield_previous_step_cell(rhs.m_phasefield_previous_step_cell)
      , m_solution_values_temperature_cell(rhs.m_solution_values_temperature_cell)
      , m_solution_grad_temperature_cell(rhs.m_solution_grad_temperature_cell)
      , m_fe_values_u(rhs.m_fe_values_u.get_fe(),
                      rhs.m_fe_values_u.get_quadrature(),
                      rhs.m_fe_values_u.get_update_flags())
      , m_fe_values_t(rhs.m_fe_values_t.get_fe(),
                      rhs.m_fe_values_t.get_quadrature(),
                      rhs.m_fe_values_t.get_update_flags())
      , m_fe_values_d(rhs.m_fe_values_d.get_fe(),
                      rhs.m_fe_values_d.get_quadrature(),
                      rhs.m_fe_values_d.get_update_flags())
      , m_delta_time(rhs.m_delta_time)
      , m_degrade_conductivity_or_not(rhs.m_degrade_conductivity_or_not)
    {}

    void reset()
    {
      const unsigned int n_q_points = m_solution_symm_grads_u_cell.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          m_solution_symm_grads_u_cell[q]  = 0.0;
          m_solution_values_phasefield_cell[q] = 0.0;
          m_solution_grad_phasefield_cell[q] = 0.0;
          m_phasefield_previous_step_cell[q] = 0.0;
          m_solution_values_temperature_cell[q] = 0.0;
          m_solution_grad_temperature_cell[q] = 0.0;
        }
    }
  };

  template <int dim>
  void SplitSolveTandUandD<dim>::update_qph_incremental_one_cell(
    const IteratorPair3 & synchronous_iterators,
    ScratchData_UQPH & scratch,
    PerTaskData_UQPH & /*data*/)
  {
    scratch.reset();

    scratch.m_fe_values_u.reinit(std::get<0>(*synchronous_iterators));
    scratch.m_fe_values_t.reinit(std::get<1>(*synchronous_iterators));
    scratch.m_fe_values_d.reinit(std::get<2>(*synchronous_iterators));

    const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(std::get<0>(*synchronous_iterators));
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const FEValuesExtractors::Vector displacements (0);

    scratch.m_fe_values_u[displacements].get_function_symmetric_gradients(
      scratch.m_solution_u_UQPH, scratch.m_solution_symm_grads_u_cell);

    scratch.m_fe_values_t.get_function_values(
      scratch.m_solution_t_UQPH, scratch.m_solution_values_temperature_cell);
    scratch.m_fe_values_t.get_function_gradients(
      scratch.m_solution_t_UQPH, scratch.m_solution_grad_temperature_cell);

    scratch.m_fe_values_d.get_function_values(
      scratch.m_solution_d_UQPH, scratch.m_solution_values_phasefield_cell);
    scratch.m_fe_values_d.get_function_gradients(
      scratch.m_solution_d_UQPH, scratch.m_solution_grad_phasefield_cell);
    scratch.m_fe_values_d.get_function_values(
      scratch.m_solution_previous_d_UQPH, scratch.m_phasefield_previous_step_cell);

    for (const unsigned int q_point :
         scratch.m_fe_values_u.quadrature_point_indices())
      lqph[q_point]->update_field_values(scratch.m_solution_symm_grads_u_cell[q_point],
                                         scratch.m_solution_values_phasefield_cell[q_point],
					 scratch.m_solution_grad_phasefield_cell[q_point],
					 scratch.m_phasefield_previous_step_cell[q_point],
					 scratch.m_delta_time,
                                         scratch.m_solution_values_temperature_cell[q_point],
					 scratch.m_solution_grad_temperature_cell[q_point],
					 scratch.m_degrade_conductivity_or_not);
  }

  template <int dim>
  struct SplitSolveTandUandD<dim>::PerTaskData_ASM_U
  {
    FullMatrix<double>                   m_cell_matrix;
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM_U(const unsigned int dofs_per_cell)
      : m_cell_matrix(dofs_per_cell, dofs_per_cell)
      , m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_matrix = 0.0;
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::PerTaskData_ASM_T
  {
    FullMatrix<double>                   m_cell_matrix;
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM_T(const unsigned int dofs_per_cell)
      : m_cell_matrix(dofs_per_cell, dofs_per_cell)
      , m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_matrix = 0.0;
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::PerTaskData_ASM_D
  {
    FullMatrix<double>                   m_cell_matrix;
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM_D(const unsigned int dofs_per_cell)
      : m_cell_matrix(dofs_per_cell, dofs_per_cell)
      , m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_matrix = 0.0;
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::PerTaskData_RHS_T
  {
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_RHS_T(const unsigned int dofs_per_cell)
      : m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::PerTaskData_RHS_D
  {
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_RHS_D(const unsigned int dofs_per_cell)
      : m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::PerTaskData_RHS_U
  {
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_RHS_U(const unsigned int dofs_per_cell)
      : m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_rhs    = 0.0;
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::ScratchData_ASM_U
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<Tensor<1, dim>>>          m_Nx_disp;       // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx_disp;  // gradient of shape function values for displacement
    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx_disp;  // symmetric gradient of shape function values for displacement

    ScratchData_ASM_U(const FiniteElement<dim> & fe_cell,
                       const QGauss<dim> &        qf_cell,
                       const UpdateFlags          uf_cell,
		       const QGauss<dim - 1> &    qf_face,
		       const UpdateFlags          uf_face)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx_disp(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_disp(qf_cell.size(),
                       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_disp(qf_cell.size(),
                            std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
    {}

    ScratchData_ASM_U(const ScratchData_ASM_U &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
	                 rhs.m_fe_face_values.get_quadrature(),
	                 rhs.m_fe_face_values.get_update_flags())
      , m_Nx_disp(rhs.m_Nx_disp)
      , m_grad_Nx_disp(rhs.m_grad_Nx_disp)
      , m_symm_grad_Nx_disp(rhs.m_symm_grad_Nx_disp)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx_disp.size();
      const unsigned int n_dofs_per_cell = m_Nx_disp[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_disp[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_symm_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_disp[q_point][k]                 = 0.0;
              m_grad_Nx_disp[q_point][k]            = 0.0;
              m_symm_grad_Nx_disp[q_point][k]       = 0.0;
            }
        }
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::ScratchData_ASM_T
  {
    FEValues<dim>     m_fe_values_t;
    FEValues<dim>     m_fe_values_u;
    FEFaceValues<dim> m_fe_face_values_t;

    std::vector<std::vector<double>>                  m_Nx_temperature;      // shape function values for temperature
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx_temperature; // gradient of shape function values for temperature

    const Vector<double> &          m_solution_u_previous_step;
    const Vector<double> &          m_solution_t_previous_step;
    std::vector<SymmetricTensor<2, dim>> m_strain_previous_step_cell;
    std::vector<double>                  m_temperature_previous_step_cell;

    ScratchData_ASM_T(const FiniteElement<dim> & fe_cell_t,
		      const FiniteElement<dim> & fe_cell_u,
                      const QGauss<dim> &        qf_cell,
                      const UpdateFlags          uf_cell,
		      const QGauss<dim - 1> &    qf_face,
		      const UpdateFlags          uf_face,
		      const Vector<double>& solution_old_u,
		      const Vector<double>& solution_old_t)
      : m_fe_values_t(fe_cell_t, qf_cell, uf_cell)
      , m_fe_values_u(fe_cell_u, qf_cell, uf_cell)
      , m_fe_face_values_t(fe_cell_t, qf_face, uf_face)
      , m_Nx_temperature(qf_cell.size(),
	                 std::vector<double>(fe_cell_t.n_dofs_per_cell()))
      , m_grad_Nx_temperature(qf_cell.size(),
		              std::vector<Tensor<1, dim>>(fe_cell_t.n_dofs_per_cell()))
      , m_solution_u_previous_step(solution_old_u)
      , m_solution_t_previous_step(solution_old_t)
      , m_strain_previous_step_cell(qf_cell.size())
      , m_temperature_previous_step_cell(qf_cell.size())
    {}

    ScratchData_ASM_T(const ScratchData_ASM_T &rhs)
      : m_fe_values_t(rhs.m_fe_values_t.get_fe(),
                      rhs.m_fe_values_t.get_quadrature(),
                      rhs.m_fe_values_t.get_update_flags())
      , m_fe_values_u(rhs.m_fe_values_u.get_fe(),
                      rhs.m_fe_values_u.get_quadrature(),
                      rhs.m_fe_values_u.get_update_flags())
      , m_fe_face_values_t(rhs.m_fe_face_values_t.get_fe(),
	                   rhs.m_fe_face_values_t.get_quadrature(),
	                   rhs.m_fe_face_values_t.get_update_flags())
      , m_Nx_temperature(rhs.m_Nx_temperature)
      , m_grad_Nx_temperature(rhs.m_grad_Nx_temperature)
      , m_solution_u_previous_step(rhs.m_solution_u_previous_step)
      , m_solution_t_previous_step(rhs.m_solution_t_previous_step)
      , m_strain_previous_step_cell(rhs.m_strain_previous_step_cell)
      , m_temperature_previous_step_cell(rhs.m_temperature_previous_step_cell)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx_temperature.size();
      const unsigned int n_dofs_per_cell = m_Nx_temperature[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_temperature[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_temperature[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_strain_previous_step_cell[q_point] = 0.0;
          m_temperature_previous_step_cell[q_point] = 0.0;

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_temperature[q_point][k]          = 0.0;
              m_grad_Nx_temperature[q_point][k]     = 0.0;
            }
        }
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::ScratchData_ASM_D
  {
    FEValues<dim>     m_fe_values;

    std::vector<std::vector<double>>                  m_Nx;  // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx;
    const Vector<double>&       m_solution_previous_phasefield;
    std::vector<double>         m_old_phasefield_cell;

    ScratchData_ASM_D(const FiniteElement<dim> &fe_cell,
                      const QGauss<dim> &       qf_cell,
                      const UpdateFlags         uf_cell,
		      const Vector<double>&     solution_old_phasefield)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_Nx(qf_cell.size(),
	     std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_solution_previous_phasefield(solution_old_phasefield)
      , m_old_phasefield_cell(qf_cell.size())
    {}

    ScratchData_ASM_D(const ScratchData_ASM_D &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_Nx(rhs.m_Nx)
      , m_grad_Nx(rhs.m_grad_Nx)
      , m_solution_previous_phasefield(rhs.m_solution_previous_phasefield)
      , m_old_phasefield_cell(rhs.m_old_phasefield_cell)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx.size();
      const unsigned int n_dofs_per_cell = m_Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

          Assert(m_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_old_phasefield_cell[q_point] = 0.0;
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx[q_point][k]           = 0.0;
              m_grad_Nx[q_point][k]      = 0.0;
            }
        }
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::ScratchData_RHS_U
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<Tensor<1, dim>>>          m_Nx_disp;       // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx_disp;  // gradient of shape function values for displacement
    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx_disp;  // symmetric gradient of shape function values for displacement

    ScratchData_RHS_U(const FiniteElement<dim> & fe_cell,
                      const QGauss<dim> &        qf_cell,
                      const UpdateFlags          uf_cell,
		      const QGauss<dim - 1> &    qf_face,
		      const UpdateFlags          uf_face)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx_disp(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_disp(qf_cell.size(),
                       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_disp(qf_cell.size(),
                            std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
    {}

    ScratchData_RHS_U(const ScratchData_RHS_U &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
	                 rhs.m_fe_face_values.get_quadrature(),
	                 rhs.m_fe_face_values.get_update_flags())
      , m_Nx_disp(rhs.m_Nx_disp)
      , m_grad_Nx_disp(rhs.m_grad_Nx_disp)
      , m_symm_grad_Nx_disp(rhs.m_symm_grad_Nx_disp)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx_disp.size();
      const unsigned int n_dofs_per_cell = m_Nx_disp[0].size();

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_disp[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_symm_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_disp[q_point][k]                 = 0.0;
              m_grad_Nx_disp[q_point][k]            = 0.0;
              m_symm_grad_Nx_disp[q_point][k]       = 0.0;
            }
        }
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::ScratchData_RHS_D
  {
    FEValues<dim>     m_fe_values;

    std::vector<std::vector<double>>                  m_Nx;  // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx;
    const Vector<double>&       m_solution_previous_phasefield;
    std::vector<double>         m_old_phasefield_cell;

    ScratchData_RHS_D(const FiniteElement<dim> &fe_cell,
                      const QGauss<dim> &       qf_cell,
                      const UpdateFlags         uf_cell,
		      const Vector<double>&     solution_old_phasefield)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_Nx(qf_cell.size(),
	     std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_solution_previous_phasefield(solution_old_phasefield)
      , m_old_phasefield_cell(qf_cell.size())
    {}

    ScratchData_RHS_D(const ScratchData_RHS_D &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_Nx(rhs.m_Nx)
      , m_grad_Nx(rhs.m_grad_Nx)
      , m_solution_previous_phasefield(rhs.m_solution_previous_phasefield)
      , m_old_phasefield_cell(rhs.m_old_phasefield_cell)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx.size();
      const unsigned int n_dofs_per_cell = m_Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

          Assert(m_grad_Nx[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_old_phasefield_cell[q_point] = 0.0;
          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx[q_point][k]           = 0.0;
              m_grad_Nx[q_point][k]      = 0.0;
            }
        }
    }
  };

  template <int dim>
  struct SplitSolveTandUandD<dim>::ScratchData_RHS_T
  {
    FEValues<dim>     m_fe_values_t;
    FEValues<dim>     m_fe_values_u;
    FEFaceValues<dim> m_fe_face_values_t;

    std::vector<std::vector<double>>                  m_Nx_temperature;      // shape function values for temperature
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx_temperature; // gradient of shape function values for temperature

    const Vector<double> &          m_solution_u_previous_step;
    const Vector<double> &          m_solution_t_previous_step;
    std::vector<SymmetricTensor<2, dim>> m_strain_previous_step_cell;
    std::vector<double>                  m_temperature_previous_step_cell;

    ScratchData_RHS_T(const FiniteElement<dim> & fe_cell_t,
		      const FiniteElement<dim> & fe_cell_u,
                      const QGauss<dim> &        qf_cell,
                      const UpdateFlags          uf_cell,
		      const QGauss<dim - 1> &    qf_face,
		      const UpdateFlags          uf_face,
		      const Vector<double>& solution_old_u,
		      const Vector<double>& solution_old_t)
      : m_fe_values_t(fe_cell_t, qf_cell, uf_cell)
      , m_fe_values_u(fe_cell_u, qf_cell, uf_cell)
      , m_fe_face_values_t(fe_cell_t, qf_face, uf_face)
      , m_Nx_temperature(qf_cell.size(),
	                 std::vector<double>(fe_cell_t.n_dofs_per_cell()))
      , m_grad_Nx_temperature(qf_cell.size(),
		              std::vector<Tensor<1, dim>>(fe_cell_t.n_dofs_per_cell()))
      , m_solution_u_previous_step(solution_old_u)
      , m_solution_t_previous_step(solution_old_t)
      , m_strain_previous_step_cell(qf_cell.size())
      , m_temperature_previous_step_cell(qf_cell.size())
    {}

    ScratchData_RHS_T(const ScratchData_RHS_T &rhs)
      : m_fe_values_t(rhs.m_fe_values_t.get_fe(),
                      rhs.m_fe_values_t.get_quadrature(),
                      rhs.m_fe_values_t.get_update_flags())
      , m_fe_values_u(rhs.m_fe_values_u.get_fe(),
                      rhs.m_fe_values_u.get_quadrature(),
                      rhs.m_fe_values_u.get_update_flags())
      , m_fe_face_values_t(rhs.m_fe_face_values_t.get_fe(),
	                   rhs.m_fe_face_values_t.get_quadrature(),
	                   rhs.m_fe_face_values_t.get_update_flags())
      , m_Nx_temperature(rhs.m_Nx_temperature)
      , m_grad_Nx_temperature(rhs.m_grad_Nx_temperature)
      , m_solution_u_previous_step(rhs.m_solution_u_previous_step)
      , m_solution_t_previous_step(rhs.m_solution_t_previous_step)
      , m_strain_previous_step_cell(rhs.m_strain_previous_step_cell)
      , m_temperature_previous_step_cell(rhs.m_temperature_previous_step_cell)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx_temperature.size();
      const unsigned int n_dofs_per_cell = m_Nx_temperature[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_temperature[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_temperature[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_strain_previous_step_cell[q_point] = 0.0;
          m_temperature_previous_step_cell[q_point] = 0.0;

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_temperature[q_point][k]          = 0.0;
              m_grad_Nx_temperature[q_point][k]     = 0.0;
            }
        }
    }
  };

  // constructor has no return type
  template <int dim>
  SplitSolveTandUandD<dim>::SplitSolveTandUandD(const std::string &input_file)
    : m_parameters(input_file)
    , m_triangulation(Triangulation<dim>::maximum_smoothing)
    , m_time(m_parameters.m_end_time)
    , m_logfile(m_parameters.m_logfile_name)
    , m_timer(m_logfile, TimerOutput::summary, TimerOutput::wall_times)
    , m_dof_handler_u(m_triangulation)
    , m_dof_handler_t(m_triangulation)
    , m_dof_handler_d(m_triangulation)
    , m_fe_u(FE_Q<dim>(m_parameters.m_poly_degree),
	      dim)   // displacement
    , m_fe_t(FE_Q<dim>(m_parameters.m_poly_degree),
             1)    // temperature
    , m_fe_d(FE_Q<dim>(m_parameters.m_poly_degree),
             1)    // phase-field
    , m_qf_cell(m_parameters.m_quad_order)
    , m_qf_face(m_parameters.m_quad_order)
    , m_n_q_points(m_qf_cell.size())
    , m_vol_reference(0.0)
  {}

  template <int dim>
  void SplitSolveTandUandD<dim>::make_grid()
  {
    if (m_parameters.m_scenario == 1)
      make_grid_case_1();
    else if (m_parameters.m_scenario == 2)
      make_grid_case_2();
    else if (m_parameters.m_scenario == 3)
      make_grid_case_3();
    else if (m_parameters.m_scenario == 4)
      make_grid_case_4();
    else if (m_parameters.m_scenario == 5)
      make_grid_case_5();
    else if (m_parameters.m_scenario == 6)
      make_grid_case_6();
    else
      Assert(false, ExcMessage("The scenario has not been implemented!"));

    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\tNumber of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t\t\tNumber of used vertices: "
              << m_triangulation.n_used_vertices()
	      << std::endl;

    std::ofstream out("original_mesh.vtu");
    GridOut       grid_out;
    grid_out.write_vtu(m_triangulation, out);

    m_vol_reference = GridTools::volume(m_triangulation);
    m_logfile << "\t\tGrid:\n\t\t\tReference volume: " << m_vol_reference << std::endl;
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::make_grid_case_1()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_tension_unstructured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] + 0.5 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.5 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else
	        face->set_boundary_id(2);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   std::fabs(cell->center()[1]) < 0.01
		    && cell->center()[0] > 0.495)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   std::fabs(cell->center()[1] - 0.0) < 0.05
		    && std::fabs(cell->center()[0] - 0.5) < 0.05)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }


  template <int dim>
  void SplitSolveTandUandD<dim>::make_grid_case_2()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_shear_unstructured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] + 0.5 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.5 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (   (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9)
		       || (std::fabs(face->center()[0] - 1.0 ) < 1.0e-9))
	        face->set_boundary_id(2);
	      else
	        face->set_boundary_id(3);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[0] > 0.45)
		     && (cell->center()[1] < 0.05) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && cell->center()[1] < 0.0 && cell->center()[1] > -0.025)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::make_grid_case_3()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_tension_structured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 1.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else
	        face->set_boundary_id(2);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (std::fabs(cell->center()[1] - 0.5) < 0.025)
		     && (cell->center()[0] > 0.475) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && std::fabs(cell->center()[1] - 0.5) < 0.025 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::make_grid_case_4()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_shear_structured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 1.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (   (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9)
		       || (std::fabs(face->center()[0] - 1.0 ) < 1.0e-9))
	        face->set_boundary_id(2);
	      else
	        face->set_boundary_id(3);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[0] > 0.475)
		     && (cell->center()[1] < 0.525) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && cell->center()[1] < 0.5 && cell->center()[1] > 0.475 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }


  template <int dim>
  void SplitSolveTandUandD<dim>::make_grid_case_5()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tQuenching test (quarter size)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    double const length = 25.0; //mm
    double const width  = 5.0;  //mm

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 100;
    repetitions[1] = 20;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0 ),
					      Point<dim>( length,   width ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[0] - 25.0 ) < 1.0e-9)
	        face->set_boundary_id(2);
	      else if (std::fabs(face->center()[1] - 5.0 ) < 1.0e-9)
	        face->set_boundary_id(3);
	      else
	        face->set_boundary_id(4);
	    }
	}

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	//m_triangulation.refine_global(m_parameters.m_global_refine_times);
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  3.0)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  3.0)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			cell->set_refine_flag();
			initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  0.13)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  0.13)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::make_grid_case_6()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tQuenching test (half size)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    double const length = 25.0; //mm
    double const width  = 10.0;  //mm

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 125;
    repetitions[1] = 50;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0 ),
					      Point<dim>( length,   width ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[0] - length ) < 1.0e-9)
	        face->set_boundary_id(2);
	      else if (std::fabs(face->center()[1] - width ) < 1.0e-9)
	        face->set_boundary_id(3);
	      else
	        face->set_boundary_id(4);
	    }
	}

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	m_triangulation.refine_global(m_parameters.m_global_refine_times);
	/*
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  3.0)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  3.0)
		    || (cell->center()[1] >  width - 3.0)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			cell->set_refine_flag();
			initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
	  */
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  0.13)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  0.13)
		    || (cell->center()[1] >  width - 0.13)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::setup_system()
  {
    m_timer.enter_subsection("Setup system");

    // mechanical problem
    setup_system_u();

    // thermal problem
    setup_system_t();

    // phasefield problem
    setup_system_d();

    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\t Number of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t\t\t Number of used vertices: "
              << m_triangulation.n_used_vertices()
              << "\n\t\t\t Number of active edges: "
              << m_triangulation.n_active_lines()
              << "\n\t\t\t Number of active faces: "
              << m_triangulation.n_active_faces()
              << "\n\t\t\t Number of degrees of freedom (total): "
	      << m_dof_handler_u.n_dofs()
	       + m_dof_handler_t.n_dofs()
	       + m_dof_handler_d.n_dofs()
	      << "\n\t\t\t Number of degrees of freedom (disp): "
	      << m_dof_handler_u.n_dofs()
	      << "\n\t\t\t Number of degrees of freedom (temperature): "
	      << m_dof_handler_t.n_dofs()
	      << "\n\t\t\t Number of degrees of freedom (phasefield): "
	      << m_dof_handler_d.n_dofs()
              << std::endl;

    setup_qph();

    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::setup_system_d()
  {
    m_dof_handler_d.distribute_dofs(m_fe_d);
    m_solution_d.reinit(m_dof_handler_d.n_dofs());
    m_system_rhs_d.reinit(m_dof_handler_d.n_dofs());

    m_constraints_d.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler_d, m_constraints_d);
    m_constraints_d.close();

    DynamicSparsityPattern dsp(m_dof_handler_d.n_dofs(), m_dof_handler_d.n_dofs());
    DoFTools::make_sparsity_pattern(m_dof_handler_d,
                                    dsp,
                                    m_constraints_d,
                                    /*keep_constrained_dofs = */ false);
    m_sparsity_pattern_d.copy_from(dsp);

    m_tangent_matrix_d.reinit(m_sparsity_pattern_d);
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::setup_system_t()
  {
    m_dof_handler_t.distribute_dofs(m_fe_t);
    m_solution_t.reinit(m_dof_handler_t.n_dofs());
    m_system_rhs_t.reinit(m_dof_handler_t.n_dofs());

    m_constraints_t.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler_t, m_constraints_t);
    m_constraints_t.close();

    DynamicSparsityPattern dsp(m_dof_handler_t.n_dofs(), m_dof_handler_t.n_dofs());
    DoFTools::make_sparsity_pattern(m_dof_handler_t,
                                    dsp,
                                    m_constraints_t,
                                    /*keep_constrained_dofs = */ false);
    m_sparsity_pattern_t.copy_from(dsp);

    m_tangent_matrix_t.reinit(m_sparsity_pattern_t);
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::setup_system_u()
  {
    m_dof_handler_u.distribute_dofs(m_fe_u);
    m_solution_u.reinit(m_dof_handler_u.n_dofs());
    m_system_rhs_u.reinit(m_dof_handler_u.n_dofs());

    m_constraints_u.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler_u, m_constraints_u);
    m_constraints_u.close();

    DynamicSparsityPattern dsp(m_dof_handler_u.n_dofs(), m_dof_handler_u.n_dofs());
    DoFTools::make_sparsity_pattern(m_dof_handler_u,
				    dsp,
				    m_constraints_u,
				    /*keep_constrained_dofs = */ false);
    m_sparsity_pattern_u.copy_from(dsp);

    m_tangent_matrix_u.reinit(m_sparsity_pattern_u);
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::setup_temperature_initial_conditions()
  {
    if (   m_parameters.m_scenario == 3
    	|| m_parameters.m_scenario == 4)
      {
	for(unsigned int i = 0; i < m_dof_handler_t.n_dofs(); ++i)
	  {
	    m_solution_t(i) = m_parameters.m_ref_temperature;
	  }
      }
    else if (m_parameters.m_scenario == 5)
      {
	for(unsigned int i = 0; i < m_dof_handler_t.n_dofs(); ++i)
	  {
	    m_solution_t(i) = m_parameters.m_ref_temperature;
	  }

	const double cool_down_temperature = 293.15; // Kelvin

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	support_points_T = DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					                         m_dof_handler_t);

	for (auto const & item : support_points_T)
	  {
	    if (   (std::fabs(item.second[0] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[1] -  0.0) < 1.0e-9))
	      {
		m_solution_t(item.first) = cool_down_temperature;
	      }
	  }
      }
    else if (m_parameters.m_scenario == 6)
      {
	for(unsigned int i = 0; i < m_dof_handler_t.n_dofs(); ++i)
	  {
	    m_solution_t(i) = m_parameters.m_ref_temperature;
	  }

	const double cool_down_temperature = 293.15; // Kelvin

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	support_points_T = DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
								 m_dof_handler_t);

	for (auto const & item : support_points_T)
	  {
	    if (   (std::fabs(item.second[0] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[1] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[1] -  10.0) < 1.0e-9))
	      {
		m_solution_t(item.first) = cool_down_temperature;
	      }
	  }
      }
    else
      {
	Assert(false, ExcMessage("The scenario has not been implemented!"));
      }
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::make_constraints_u(const unsigned int it_nr,
						    const unsigned int itr_stagger)
  {

    // The staggered iteration starts from 1
    const bool apply_dirichlet_bc = ( (it_nr == 0) && (itr_stagger == 1) );

    if ( (it_nr > 1) || (itr_stagger > 1) )
      {
	return;
      }

    if (apply_dirichlet_bc)
      {
	m_constraints_u.clear();
	DoFTools::make_hanging_node_constraints(m_dof_handler_u,
						m_constraints_u);

	const FEValuesExtractors::Scalar x_displacement(0);
	const FEValuesExtractors::Scalar y_displacement(1);
	const FEValuesExtractors::Scalar z_displacement(dim-1);

	if (   m_parameters.m_scenario == 1
	    || m_parameters.m_scenario == 3)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_u,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_u,
						     m_fe_u.component_mask(y_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe_u.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] - 0.0) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler_u);
		  }
	      }
	    m_constraints_u.add_line(node_xy[0]);
	    m_constraints_u.set_inhomogeneity(node_xy[0], 0.0);

	    m_constraints_u.add_line(node_xy[1]);
	    m_constraints_u.set_inhomogeneity(node_xy[1], 0.0);

	    const int boundary_id_top_surface = 1;
	    /*
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));
	    */
            const double time_inc = m_time.get_delta_t();
            double disp_magnitude = m_time.get_magnitude();
	    VectorTools::interpolate_boundary_values(m_dof_handler_u,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, dim),
						     m_constraints_u,
						     m_fe_u.component_mask(y_displacement));
	  }
	else if (   m_parameters.m_scenario == 2
	         || m_parameters.m_scenario == 4)
	  {
	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler_u,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_u);

	    const int boundary_id_top_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler_u,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_u,
						     m_fe_u.component_mask(y_displacement));

	    const double time_inc = m_time.get_delta_t();
	    double disp_magnitude = m_time.get_magnitude();
	    VectorTools::interpolate_boundary_values(m_dof_handler_u,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, dim),
						     m_constraints_u,
						     m_fe_u.component_mask(x_displacement));

	    const int boundary_id_side_surfaces = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler_u,
						     boundary_id_side_surfaces,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_u,
						     m_fe_u.component_mask(y_displacement));
	  }
	else if (m_parameters.m_scenario == 5)
	  {
	    const int boundary_id_mid_surface_x = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler_u,
						     boundary_id_mid_surface_x,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_u,
						     m_fe_u.component_mask(x_displacement));

	    const int boundary_id_mid_surface_y = 3;
	    VectorTools::interpolate_boundary_values(m_dof_handler_u,
						     boundary_id_mid_surface_y,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_u,
						     m_fe_u.component_mask(y_displacement));
	  }
	else if (m_parameters.m_scenario == 6)
	  {
	    const int boundary_id_mid_surface_x = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler_u,
						     boundary_id_mid_surface_x,
						     Functions::ZeroFunction<dim>(dim),
						     m_constraints_u,
						     m_fe_u.component_mask(x_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe_u.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 25.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  5.0) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler_u);
		  }
	      }
	    m_constraints_u.add_line(node_xy[1]);
	    m_constraints_u.set_inhomogeneity(node_xy[1], 0.0);
	  }
	else
	  Assert(false, ExcMessage("The scenario has not been implemented!"));
      }
    else  // inhomogeneous constraints
      {
        if (m_constraints_u.has_inhomogeneities())
          {
            AffineConstraints<double> homogeneous_constraints(m_constraints_u);
            for (unsigned int dof = 0; dof != m_dof_handler_u.n_dofs(); ++dof)
              if (homogeneous_constraints.is_inhomogeneously_constrained(dof))
                homogeneous_constraints.set_inhomogeneity(dof, 0.0);
            m_constraints_u.clear();
            m_constraints_u.copy_from(homogeneous_constraints);
          }
      }
    m_constraints_u.close();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_system_d()
  {
    m_timer.enter_subsection("Assemble phase-field system");

    m_tangent_matrix_d = 0.0;
    m_system_rhs_d     = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_JxW_values);

    PerTaskData_ASM_D per_task_data(m_fe_d.n_dofs_per_cell());
    ScratchData_ASM_D scratch_data(m_fe_d,
				   m_qf_cell,
			           uf_cell,
			           m_solution_previous_d);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_ASM_D & scratch,
	     PerTaskData_ASM_D & data)
      {
        this->assemble_system_one_cell_d(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_ASM_D &data)
      {
        this->m_constraints_d.distribute_local_to_global(data.m_cell_matrix,
                                                         data.m_cell_rhs,
                                                         data.m_local_dof_indices,
                                                         m_tangent_matrix_d,
                                                         m_system_rhs_d);
      };

    WorkStream::run(
      m_dof_handler_d.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_system_u()
  {
    m_timer.enter_subsection("Assemble U system");

    m_tangent_matrix_u = 0.0;
    m_system_rhs_u     = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_ASM_U per_task_data(m_fe_u.n_dofs_per_cell());
    ScratchData_ASM_U scratch_data(m_fe_u, m_qf_cell, uf_cell, m_qf_face, uf_face);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_ASM_U & scratch,
	     PerTaskData_ASM_U & data)
      {
        this->assemble_system_one_cell_u(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_ASM_U &data)
      {
        this->m_constraints_u.distribute_local_to_global(data.m_cell_matrix,
							 data.m_cell_rhs,
                                                         data.m_local_dof_indices,
						         m_tangent_matrix_u,
						         m_system_rhs_u);
      };

    WorkStream::run(
      m_dof_handler_u.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_system_t()
  {
    m_timer.enter_subsection("Assemble T system");

    m_tangent_matrix_t = 0.0;
    m_system_rhs_t     = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_ASM_T per_task_data(m_fe_t.n_dofs_per_cell());
    ScratchData_ASM_T scratch_data(m_fe_t, m_fe_u, m_qf_cell, uf_cell, m_qf_face, uf_face,
				   m_solution_previous_u, m_solution_previous_t);

    auto worker =
      [this](const IteratorPair2 & synchronous_iterators,
	     ScratchData_ASM_T  & scratch,
	     PerTaskData_ASM_T  & data)
      {
        this->assemble_system_one_cell_t(synchronous_iterators, scratch, data);
      };

    auto copier = [this](const PerTaskData_ASM_T &data)
      {
        this->m_constraints_t.distribute_local_to_global(data.m_cell_matrix,
							 data.m_cell_rhs,
                                                         data.m_local_dof_indices,
						         m_tangent_matrix_t,
						         m_system_rhs_t);
      };

    WorkStream::run(
      IteratorPair2(IteratorTuple2(m_dof_handler_t.begin_active(),
			           m_dof_handler_u.begin_active())),
      IteratorPair2(IteratorTuple2(m_dof_handler_t.end(),
			           m_dof_handler_u.end())),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_rhs_u()
  {
    m_timer.enter_subsection("Assemble u RHS");

    m_system_rhs_u = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_RHS_U per_task_data(m_fe_u.n_dofs_per_cell());
    ScratchData_RHS_U scratch_data(m_fe_u, m_qf_cell, uf_cell, m_qf_face, uf_face);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_RHS_U & scratch,
	     PerTaskData_RHS_U & data)
      {
        this->assemble_rhs_one_cell_u(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_RHS_U &data)
      {
        this->m_constraints_u.distribute_local_to_global(data.m_cell_rhs,
                                                         data.m_local_dof_indices,
							 m_system_rhs_u);
      };

    WorkStream::run(
      m_dof_handler_u.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_rhs_one_cell_u(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_RHS_U & scratch,
      PerTaskData_RHS_U & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const FEValuesExtractors::Vector displacements (0);

    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);

    right_hand_side(scratch.m_fe_values.get_quadrature_points(),
		    rhs_values,
		    m_parameters.m_x_component*1.0,
		    m_parameters.m_y_component*1.0,
		    m_parameters.m_z_component*1.0);

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
	    scratch.m_Nx_disp[q_point][k] =
	      scratch.m_fe_values[displacements].value(k, q_point);
	    scratch.m_grad_Nx_disp[q_point][k] =
	      scratch.m_fe_values[displacements].gradient(k, q_point);
	    scratch.m_symm_grad_Nx_disp[q_point][k] =
	      symmetrize(scratch.m_grad_Nx_disp[q_point][k]);
          }
      }

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
        const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();

        const std::vector<Tensor<1,dim>> & N_disp = scratch.m_Nx_disp[q_point];
        const std::vector<SymmetricTensor<2, dim>> & symm_grad_N_disp =
          scratch.m_symm_grad_Nx_disp[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            // Here, the right-hand term represents the residual
	    data.m_cell_rhs(i) += symm_grad_N_disp[i] * cauchy_stress * JxW;

	    // contributions from the body force to right-hand side
	    data.m_cell_rhs(i) -= N_disp[i] * rhs_values[q_point] * JxW;
          }  // i
      }  // q_point

    // if there is surface pressure, this surface pressure always applied to the
    // reference configuration
    const unsigned int face_pressure_id = 100;
    const double p0 = 0.0;

    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == face_pressure_id)
        {
          scratch.m_fe_face_values.reinit(cell, face);

          for (const unsigned int f_q_point : scratch.m_fe_face_values.quadrature_point_indices())
            {
              const Tensor<1, dim> &N = scratch.m_fe_face_values.normal_vector(f_q_point);

              const double         pressure  = p0 * time_ramp;
              const Tensor<1, dim> traction  = pressure * N;

              for (const unsigned int i : scratch.m_fe_values.dof_indices())
                {
		  const Tensor<1, dim> Nx =
		      scratch.m_fe_face_values[displacements].value(i, f_q_point);
    		  const double JxW = scratch.m_fe_face_values.JxW(f_q_point);
    		  data.m_cell_rhs(i) -= Nx * traction * JxW;
                }
            }
        }
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_rhs_d()
  {
    m_timer.enter_subsection("Assemble phase-field RHS");

    m_system_rhs_d = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_JxW_values);

    PerTaskData_RHS_D per_task_data(m_fe_d.n_dofs_per_cell());
    ScratchData_RHS_D scratch_data(m_fe_d,
			           m_qf_cell,
				   uf_cell,
			           m_solution_previous_d);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_RHS_D & scratch,
	     PerTaskData_RHS_D & data)
      {
        this->assemble_rhs_one_cell_d(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_RHS_D &data)
      {
        this->m_constraints_d.distribute_local_to_global(data.m_cell_rhs,
                                                         data.m_local_dof_indices,
                                                         m_system_rhs_d);
      };

    WorkStream::run(
      m_dof_handler_d.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_system_one_cell_u(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM_U & scratch,
      PerTaskData_ASM_U & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const FEValuesExtractors::Vector displacements (0);

    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);

    right_hand_side(scratch.m_fe_values.get_quadrature_points(),
		    rhs_values,
		    m_parameters.m_x_component*1.0,
		    m_parameters.m_y_component*1.0,
		    m_parameters.m_z_component*1.0);

    std::vector<double> heat_supply_values(m_n_q_points);

    heat_supply(scratch.m_fe_values.get_quadrature_points(),
		heat_supply_values,
		m_parameters.m_heat_supply*1.0);

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
            scratch.m_Nx_disp[q_point][k] =
                  scratch.m_fe_values[displacements].value(k, q_point);
            scratch.m_grad_Nx_disp[q_point][k] =
                  scratch.m_fe_values[displacements].gradient(k, q_point);
            scratch.m_symm_grad_Nx_disp[q_point][k] =
                  symmetrize(scratch.m_grad_Nx_disp[q_point][k]);
          }
      }

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
        const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();

        const SymmetricTensor<4, dim> & mechanical_C  = lqph[q_point]->get_mechanical_C();

        const std::vector<Tensor<1,dim>> & N_disp = scratch.m_Nx_disp[q_point];
        const std::vector<SymmetricTensor<2, dim>> & symm_grad_N_disp =
          scratch.m_symm_grad_Nx_disp[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        SymmetricTensor<2, dim> symm_grad_Nx_i_x_C;

        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            // Notice that in the Newton-Raphson iteration, we are solving
            // Ax = -r. Therefore, the right-hand side term should be the negative
            // of the residual
            data.m_cell_rhs(i) -= symm_grad_N_disp[i] * cauchy_stress * JxW;

            // contributions from the body force to right-hand side
            data.m_cell_rhs(i) += N_disp[i] * rhs_values[q_point] * JxW;

            symm_grad_Nx_i_x_C = symm_grad_N_disp[i] * mechanical_C;

            for (const unsigned int j : scratch.m_fe_values.dof_indices_ending_at(i))
              data.m_cell_matrix(i, j) += symm_grad_Nx_i_x_C * symm_grad_N_disp[j] * JxW;
          }  // i
      }  // q_point

    // if there is surface pressure, this surface pressure always applied to the
    // reference configuration
    const unsigned int face_pressure_id = 100;
    const double p0 = 0.0;

    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == face_pressure_id)
        {
          scratch.m_fe_face_values.reinit(cell, face);

          for (const unsigned int f_q_point : scratch.m_fe_face_values.quadrature_point_indices())
            {
              const Tensor<1, dim> &N = scratch.m_fe_face_values.normal_vector(f_q_point);

              const double         pressure  = p0 * time_ramp;
              const Tensor<1, dim> traction  = pressure * N;

              for (const unsigned int i : scratch.m_fe_values.dof_indices())
                {
		  const Tensor<1, dim> Nx =
		      scratch.m_fe_face_values[displacements].value(i, f_q_point);
    		  const double JxW = scratch.m_fe_face_values.JxW(f_q_point);
    		  data.m_cell_rhs(i) += Nx * traction * JxW;
                }
            }
        }

    for (const unsigned int i : scratch.m_fe_values.dof_indices())
      for (const unsigned int j : scratch.m_fe_values.dof_indices_starting_at(i+1))
	data.m_cell_matrix(i, j) = data.m_cell_matrix(j, i);
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_system_one_cell_t(
      const IteratorPair2 & synchronous_iterators,
      ScratchData_ASM_T  & scratch,
      PerTaskData_ASM_T  & data) const
  {
    data.reset();
    scratch.reset();
    std::get<0>(*synchronous_iterators)->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values_t.reinit(std::get<0>(*synchronous_iterators));
    scratch.m_fe_values_u.reinit(std::get<1>(*synchronous_iterators));

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(std::get<0>(*synchronous_iterators));
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const FEValuesExtractors::Vector displacements (0);

    scratch.m_fe_values_u[displacements].get_function_symmetric_gradients(
      scratch.m_solution_u_previous_step, scratch.m_strain_previous_step_cell);

    scratch.m_fe_values_t.get_function_values(
      scratch.m_solution_t_previous_step, scratch.m_temperature_previous_step_cell);

    const double delta_time = m_time.get_delta_t();

    const double time_ramp = (m_time.current() / m_time.end());

    std::vector<double> heat_supply_values(m_n_q_points);

    heat_supply(scratch.m_fe_values_t.get_quadrature_points(),
		heat_supply_values,
		m_parameters.m_heat_supply*1.0);

    for (const unsigned int q_point : scratch.m_fe_values_t.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values_t.dof_indices())
          {
	    scratch.m_Nx_temperature[q_point][k] =
	      scratch.m_fe_values_t.shape_value(k, q_point);
	    scratch.m_grad_Nx_temperature[q_point][k] =
	      scratch.m_fe_values_t.shape_grad(k, q_point);
          }
      }

    for (const unsigned int q_point : scratch.m_fe_values_t.quadrature_point_indices())
      {
        // degraded thermal conductivity
	const double kappa_d                 = lqph[q_point]->get_thermal_conductivity();
	const double heat_capacity           = lqph[q_point]->get_heat_capacity();
        const double ref_t                   = lqph[q_point]->get_ref_temperature();
        const double thermal_expansion       = lqph[q_point]->get_thermal_expansion_coeff();
        const double lame_lambda             = lqph[q_point]->get_lame_lambda();
        const double lame_mu                 = lqph[q_point]->get_lame_mu();
        const bool   coupling_on_heat_eq     = lqph[q_point]->get_heat_coupling_flag();
        const double phasefield_value        = lqph[q_point]->get_phase_field_value();

        double coupling_tensor_coeff = thermal_expansion
             * (trace(Physics::Elasticity::StandardTensors<dim>::I)*lame_lambda + 2.0*lame_mu)
	     * degradation_function(phasefield_value);

        if (!coupling_on_heat_eq)
          coupling_tensor_coeff = 0.0;

        const SymmetricTensor<2, dim> ut_coupling_tensor
             = coupling_tensor_coeff * Physics::Elasticity::StandardTensors<dim>::I;

	// current total strain
	const SymmetricTensor<2, dim> & current_strain = lqph[q_point]->get_strain();
        // previous total strain
	const SymmetricTensor<2, dim> & old_strain = scratch.m_strain_previous_step_cell[q_point];

        const std::vector<double>         &      N_temperature = scratch.m_Nx_temperature[q_point];
        const std::vector<Tensor<1, dim>> & grad_N_temperature = scratch.m_grad_Nx_temperature[q_point];
        const double old_temperature = scratch.m_temperature_previous_step_cell[q_point];

        const double JxW = scratch.m_fe_values_t.JxW(q_point);

        for (const unsigned int i : scratch.m_fe_values_t.dof_indices())
          {
	    data.m_cell_rhs(i) += heat_capacity * N_temperature[i]
						* old_temperature / ref_t * JxW;
	    data.m_cell_rhs(i) += N_temperature[i] * heat_supply_values[q_point]
						   * delta_time /ref_t * JxW;

	    // the mechanical-thermal coupling term
	    data.m_cell_rhs(i) -= N_temperature[i]
				* ut_coupling_tensor
				* (current_strain - old_strain)
				* JxW;

            for (const unsigned int j : scratch.m_fe_values_t.dof_indices_ending_at(i))
	      data.m_cell_matrix(i, j) += (  heat_capacity
					   * N_temperature[i] * N_temperature[j]
					   + kappa_d
					   * grad_N_temperature[i] * grad_N_temperature[j] * delta_time
					  ) / ref_t * JxW;
          }  // i
      }  // q_point

    // surface heat flux (Neumann BC)
    const unsigned int face_flux_id = 100;
    const double h0 = 0.0;

    for (const auto &face : std::get<0>(*synchronous_iterators)->face_iterators())
      if (face->at_boundary() && face->boundary_id() == face_flux_id)
        {
          scratch.m_fe_face_values_t.reinit(std::get<0>(*synchronous_iterators), face);

          for (const unsigned int f_q_point : scratch.m_fe_face_values_t.quadrature_point_indices())
            {
              const double flux  = h0 * time_ramp;
              for (const unsigned int i : scratch.m_fe_values_t.dof_indices())
                {
		  const double Ni = scratch.m_fe_face_values_t.shape_value(i, f_q_point);
		  const double JxW = scratch.m_fe_face_values_t.JxW(f_q_point);
		  data.m_cell_rhs(i) += Ni * flux * JxW;
                }
            }
        }

    for (const unsigned int i : scratch.m_fe_values_t.dof_indices())
      for (const unsigned int j : scratch.m_fe_values_t.dof_indices_starting_at(i+1))
        data.m_cell_matrix(i, j) = data.m_cell_matrix(j, i);
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_system_one_cell_d(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_ASM_D & scratch,
    PerTaskData_ASM_D & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values.get_function_values(
      scratch.m_solution_previous_phasefield, scratch.m_old_phasefield_cell);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double delta_time = m_time.get_delta_t();

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
	    scratch.m_Nx[q_point][k] =
	      scratch.m_fe_values.shape_value(k, q_point);
	    scratch.m_grad_Nx[q_point][k] =
	      scratch.m_fe_values.shape_grad(k, q_point);
          }
      }

    for (unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
	const double length_scale          = lqph[q_point]->get_length_scale();
	const double gc                    = lqph[q_point]->get_critical_energy_release_rate();
	const double eta                   = lqph[q_point]->get_viscosity();
	const double history_strain_energy = lqph[q_point]->get_history_max_positive_strain_energy();
	const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();

	double history_value = history_strain_energy;
	if (current_positive_strain_energy > history_strain_energy)
	  history_value = current_positive_strain_energy;

        const std::vector<double>         &      N = scratch.m_Nx[q_point];
        const std::vector<Tensor<1, dim>> & grad_N = scratch.m_grad_Nx[q_point];
        const double                old_phasefield = scratch.m_old_phasefield_cell[q_point];
        const double                           JxW = scratch.m_fe_values.JxW(q_point);

        for (unsigned int i : scratch.m_fe_values.dof_indices())
          {
	    for (unsigned int j : scratch.m_fe_values.dof_indices_ending_at(i))
	      {
		data.m_cell_matrix(i, j) += (  (gc/length_scale + eta/delta_time + 2.0*history_value)
		                             * N[i] * N[j]
				             + gc * length_scale * grad_N[i] * grad_N[j] )
				           * JxW;
	      } // j
	    // Remember, this is a linear problem,rhs is only the right-hand side
	    // not the residual
	    data.m_cell_rhs(i) += (  eta/delta_time * N[i] * old_phasefield
		                   + 2.0 * N[i] * history_value ) * JxW;
          } // i
      } // q_point

    for (const unsigned int i : scratch.m_fe_values.dof_indices())
      for (const unsigned int j : scratch.m_fe_values.dof_indices_starting_at(i + 1))
        data.m_cell_matrix(i, j) = data.m_cell_matrix(j, i);
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_rhs_one_cell_d(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_RHS_D & scratch,
    PerTaskData_RHS_D & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values.get_function_values(
      scratch.m_solution_previous_phasefield, scratch.m_old_phasefield_cell);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double delta_time = m_time.get_delta_t();

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
	    scratch.m_Nx[q_point][k] =
	      scratch.m_fe_values.shape_value(k, q_point);
	    scratch.m_grad_Nx[q_point][k] =
	      scratch.m_fe_values.shape_grad(k, q_point);
          }
      }

    for (unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
	const double length_scale          = lqph[q_point]->get_length_scale();
	const double gc                    = lqph[q_point]->get_critical_energy_release_rate();
	const double eta                   = lqph[q_point]->get_viscosity();
	const double history_strain_energy = lqph[q_point]->get_history_max_positive_strain_energy();
	const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();

	double history_value = history_strain_energy;
	if (current_positive_strain_energy > history_strain_energy)
	  history_value = current_positive_strain_energy;

	const double phasefield_value        = lqph[q_point]->get_phase_field_value();
	const Tensor<1, dim> phasefield_grad = lqph[q_point]->get_phase_field_gradient();

        const std::vector<double>         &      N = scratch.m_Nx[q_point];
        const std::vector<Tensor<1, dim>> & grad_N = scratch.m_grad_Nx[q_point];
        const double                old_phasefield = scratch.m_old_phasefield_cell[q_point];
        const double                           JxW = scratch.m_fe_values.JxW(q_point);

        for (unsigned int i : scratch.m_fe_values.dof_indices())
          {
	    data.m_cell_rhs(i) += (    gc * length_scale * grad_N[i] * phasefield_grad
				    +  (   gc / length_scale * phasefield_value
					 + eta / delta_time  * (phasefield_value - old_phasefield)
					 + degradation_function_derivative(phasefield_value) * history_value )
				      * N[i]
				  ) * JxW;
          } // i
      } // q_point
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_rhs_t()
  {
    m_timer.enter_subsection("Assemble T RHS");

    m_system_rhs_t     = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_RHS_T per_task_data(m_fe_t.n_dofs_per_cell());
    ScratchData_RHS_T scratch_data(m_fe_t, m_fe_u, m_qf_cell, uf_cell, m_qf_face, uf_face,
				   m_solution_previous_u, m_solution_previous_t);

    auto worker =
      [this](const IteratorPair2 & synchronous_iterators,
	     ScratchData_RHS_T  & scratch,
	     PerTaskData_RHS_T  & data)
      {
        this->assemble_rhs_one_cell_t(synchronous_iterators, scratch, data);
      };

    auto copier = [this](const PerTaskData_RHS_T &data)
      {
        this->m_constraints_t.distribute_local_to_global(data.m_cell_rhs,
                                                         data.m_local_dof_indices,
						         m_system_rhs_t);
      };

    WorkStream::run(
      IteratorPair2(IteratorTuple2(m_dof_handler_t.begin_active(),
			           m_dof_handler_u.begin_active())),
      IteratorPair2(IteratorTuple2(m_dof_handler_t.end(),
			           m_dof_handler_u.end())),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }


  template <int dim>
  void SplitSolveTandUandD<dim>::assemble_rhs_one_cell_t(
      const IteratorPair2 & synchronous_iterators,
      ScratchData_RHS_T  & scratch,
      PerTaskData_RHS_T  & data) const
  {
    data.reset();
    scratch.reset();
    std::get<0>(*synchronous_iterators)->get_dof_indices(data.m_local_dof_indices);

    scratch.m_fe_values_t.reinit(std::get<0>(*synchronous_iterators));
    scratch.m_fe_values_u.reinit(std::get<1>(*synchronous_iterators));

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(std::get<0>(*synchronous_iterators));
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const FEValuesExtractors::Vector displacements (0);

    scratch.m_fe_values_u[displacements].get_function_symmetric_gradients(
      scratch.m_solution_u_previous_step, scratch.m_strain_previous_step_cell);

    scratch.m_fe_values_t.get_function_values(
      scratch.m_solution_t_previous_step, scratch.m_temperature_previous_step_cell);

    const double delta_time = m_time.get_delta_t();

    const double time_ramp = (m_time.current() / m_time.end());

    std::vector<double> heat_supply_values(m_n_q_points);

    heat_supply(scratch.m_fe_values_t.get_quadrature_points(),
		heat_supply_values,
		m_parameters.m_heat_supply*1.0);

    for (const unsigned int q_point : scratch.m_fe_values_t.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values_t.dof_indices())
          {
	    scratch.m_Nx_temperature[q_point][k] =
	      scratch.m_fe_values_t.shape_value(k, q_point);
	    scratch.m_grad_Nx_temperature[q_point][k] =
	      scratch.m_fe_values_t.shape_grad(k, q_point);
          }
      }

    for (const unsigned int q_point : scratch.m_fe_values_t.quadrature_point_indices())
      {
        // degraded thermal conductivity
	const double heat_capacity           = lqph[q_point]->get_heat_capacity();
        const double ref_t                   = lqph[q_point]->get_ref_temperature();
        const double thermal_expansion       = lqph[q_point]->get_thermal_expansion_coeff();
        const double lame_lambda             = lqph[q_point]->get_lame_lambda();
        const double lame_mu                 = lqph[q_point]->get_lame_mu();
        const bool   coupling_on_heat_eq     = lqph[q_point]->get_heat_coupling_flag();
        const double phasefield_value        = lqph[q_point]->get_phase_field_value();

        double coupling_tensor_coeff = thermal_expansion
             * (trace(Physics::Elasticity::StandardTensors<dim>::I)*lame_lambda + 2.0*lame_mu)
	     * degradation_function(phasefield_value);

        if (!coupling_on_heat_eq)
          coupling_tensor_coeff = 0.0;

        const SymmetricTensor<2, dim> ut_coupling_tensor
             = coupling_tensor_coeff * Physics::Elasticity::StandardTensors<dim>::I;

	// current total strain
	const SymmetricTensor<2, dim> & current_strain = lqph[q_point]->get_strain();
        // previous total strain
	const SymmetricTensor<2, dim> & old_strain = scratch.m_strain_previous_step_cell[q_point];

        const std::vector<double>         &      N_temperature = scratch.m_Nx_temperature[q_point];
        const std::vector<Tensor<1, dim>> & grad_N_temperature = scratch.m_grad_Nx_temperature[q_point];
        const double old_temperature = scratch.m_temperature_previous_step_cell[q_point];

	const double temperature_value = lqph[q_point]->get_temperature_value();

        const Tensor<1, dim> & heat_flux = lqph[q_point]->get_heat_flux();

        const double JxW = scratch.m_fe_values_t.JxW(q_point);

        for (const unsigned int i : scratch.m_fe_values_t.dof_indices())
          {
	    data.m_cell_rhs(i) += heat_capacity * N_temperature[i]
			        * (temperature_value - old_temperature) / ref_t * JxW;
	    data.m_cell_rhs(i) -= grad_N_temperature[i]
				* heat_flux * delta_time / ref_t * JxW;
	    data.m_cell_rhs(i) -= N_temperature[i] * heat_supply_values[q_point]
						   * delta_time /ref_t * JxW;
	    // the mechanical-thermal coupling term
	    data.m_cell_rhs(i) += N_temperature[i]
				* ut_coupling_tensor
				* (current_strain - old_strain)
				* JxW;
          }  // i
      }  // q_point

    // surface heat flux (Neumann BC)
    const unsigned int face_flux_id = 100;
    const double h0 = 0.0;

    for (const auto &face : std::get<0>(*synchronous_iterators)->face_iterators())
      if (face->at_boundary() && face->boundary_id() == face_flux_id)
        {
          scratch.m_fe_face_values_t.reinit(std::get<0>(*synchronous_iterators), face);

          for (const unsigned int f_q_point : scratch.m_fe_face_values_t.quadrature_point_indices())
            {
              const double flux  = h0 * time_ramp;
              for (const unsigned int i : scratch.m_fe_values_t.dof_indices())
                {
		  const double Ni = scratch.m_fe_face_values_t.shape_value(i, f_q_point);
		  const double JxW = scratch.m_fe_face_values_t.JxW(f_q_point);
		  data.m_cell_rhs(i) -= Ni * flux * JxW;
                }
            }
        }
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::update_history_field_step()
  {
    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
        std::vector<std::shared_ptr< PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            lqph[q_point]->update_history_variable();
          }
      }
  }

  template <int dim>
  unsigned int SplitSolveTandUandD<dim>::displacement_step(unsigned int itr_stagger)
  {
    Vector<double> solution_delta_u(m_dof_handler_u.n_dofs());
    solution_delta_u = 0.0;
    unsigned int newton_itrs_required
      = solve_nonlinear_newton_u(solution_delta_u, itr_stagger);
    m_solution_u += solution_delta_u;
    return newton_itrs_required;
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::temperature_step()
  {
    make_constraints_t();
    assemble_system_t();
    solve_linear_system_t();

    Vector<double> solution_delta_u(m_dof_handler_u.n_dofs());
    solution_delta_u = 0.0;
    update_qph_incremental(solution_delta_u);
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::phasefield_step()
  {
    make_constraints_d();
    assemble_system_d();
    solve_linear_system_d();

    Vector<double> solution_delta_u(m_dof_handler_u.n_dofs());
    solution_delta_u = 0.0;
    update_qph_incremental(solution_delta_u);
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::make_constraints_d()
  {
    m_constraints_d.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler_d, m_constraints_d);
    m_constraints_d.close();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::make_constraints_t()
  {
    m_constraints_t.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler_t, m_constraints_t);

    if (   m_parameters.m_scenario == 1
	|| m_parameters.m_scenario == 3)
      {
	// Since the thermal problem is linear, we directly set the temperature at the boundaries.
	// This is different from nonlinear iterations where we set increment at boundaries.
	double bottom_temperature = m_parameters.m_ref_temperature; // temperature BC

	// Temperature B.C. bottom surface
	const int boundary_id_bottom_surface = 0;

	// temperature B.C. at the bottom surface
	VectorTools::interpolate_boundary_values(m_dof_handler_t,
						 boundary_id_bottom_surface,
						 Functions::ConstantFunction<dim>(
						     bottom_temperature),
						 m_constraints_t);

	const int boundary_id_top_surface = 1;
	const double current_time = m_time.current();

	double top_temperature = m_parameters.m_ref_temperature; // temperature BC

	// temperature B.C. at the top surface
	if (m_time.current() <= 0.25001e-3)
	  top_temperature -= current_time * 1.0e5; //cool down
	else
	  top_temperature -= 0.25e-3 * 1.0e5;
	//delta_temperature =  time_inc * 1.0e5; //warn up
	//delta_temperature = 0.0; //constant

	VectorTools::interpolate_boundary_values(m_dof_handler_t,
						 boundary_id_top_surface,
						 Functions::ConstantFunction<dim>(
						     top_temperature),
						 m_constraints_t);
      }
    else if (   m_parameters.m_scenario == 2
	     || m_parameters.m_scenario == 4)
      {
	// Since the thermal problem is linear, we directly set the temperature at the boundaries.
	// This is different from nonlinear iterations where we set increment at boundaries.
	double bottom_temperature = m_parameters.m_ref_temperature; // temperature BC

	// Dirichlet B,C. bottom surface
	const int boundary_id_bottom_surface = 0;

	// temperature B.C. at the bottom surface
	VectorTools::interpolate_boundary_values(m_dof_handler_t,
						 boundary_id_bottom_surface,
						 Functions::ConstantFunction<dim>(
						     bottom_temperature),
						 m_constraints_t);

	const int boundary_id_top_surface = 1;

	const double current_time = m_time.current();

	double top_temperature = m_parameters.m_ref_temperature; // temperature BC

	// temperature B.C. at the top surface
	if (m_time.current() <= 10.0001e-3)
//	  top_temperature -= current_time * 2.0e4; //cool down
	  top_temperature += current_time * 2.0e4; //warm up
	else
//	  top_temperature -= 10.0e-3 * 2.0e4;  // cool down
   	  top_temperature += 10.0e-3 * 2.0e4;

	VectorTools::interpolate_boundary_values(m_dof_handler_t,
						 boundary_id_top_surface,
						 Functions::ConstantFunction<dim>(
						     top_temperature),
						 m_constraints_t);
      }
    else if (m_parameters.m_scenario == 5)
      {
	// Since the thermal problem is linear, we directly set the temperature at the boundaries.
	// This is different from nonlinear iterations where we set increment at boundaries.
	const double cool_down_temperature = 293.15; // Kelvin

	const int boundary_id_left_surface = 0;
	VectorTools::interpolate_boundary_values(m_dof_handler_t,
						 boundary_id_left_surface,
						 Functions::ConstantFunction<dim>(
						     cool_down_temperature),
						 m_constraints_t);

	const int boundary_id_bottom_surface = 1;
	VectorTools::interpolate_boundary_values(m_dof_handler_t,
						 boundary_id_bottom_surface,
						 Functions::ConstantFunction<dim>(
						     cool_down_temperature),
						 m_constraints_t);
      }
    else if (m_parameters.m_scenario == 6)
      {
	// Since the thermal problem is linear, we directly set the temperature at the boundaries.
	// This is different from nonlinear iterations where we set increment at boundaries.
	const double cool_down_temperature = 293.15; // Kelvin

	const int boundary_id_left_surface = 0;
	VectorTools::interpolate_boundary_values(m_dof_handler_t,
						 boundary_id_left_surface,
						 Functions::ConstantFunction<dim>(
						     cool_down_temperature),
						 m_constraints_t);

	const int boundary_id_bottom_surface = 1;
	VectorTools::interpolate_boundary_values(m_dof_handler_t,
						 boundary_id_bottom_surface,
						 Functions::ConstantFunction<dim>(
						     cool_down_temperature),
						 m_constraints_t);

	const int boundary_id_top_surface = 3;
	VectorTools::interpolate_boundary_values(m_dof_handler_t,
						 boundary_id_top_surface,
						 Functions::ConstantFunction<dim>(
						     cool_down_temperature),
						 m_constraints_t);
      }
    else
      Assert(false, ExcMessage("The scenario has not been implemented!"));

    m_constraints_t.close();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::
  solve_linear_system_u(Vector<double> & newton_update_u)
  {
    m_timer.enter_subsection("Solve U linear system");

    if (m_parameters.m_type_linear_solver == "Direct")
      {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(m_tangent_matrix_u);
        A_direct.vmult(newton_update_u,
	               m_system_rhs_u);
      }
    else if (m_parameters.m_type_linear_solver == "CG")
      {
	SolverControl            solver_control_u(1e6, 1e-12);
	SolverCG<Vector<double>> cg_u(solver_control_u);

	PreconditionJacobi<SparseMatrix<double>> preconditioner_u;
	preconditioner_u.initialize(m_tangent_matrix_u, 1.0);

	cg_u.solve(m_tangent_matrix_u,
		   newton_update_u,
		   m_system_rhs_u,
		   preconditioner_u);
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected linear solver not implemented "
	        	"for the displacement subproblem!"));
      }

    m_constraints_u.distribute(newton_update_u);

    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::solve_linear_system_t()
  {
    m_timer.enter_subsection("Solve T linear system");

    if (m_parameters.m_type_linear_solver == "Direct")
      {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(m_tangent_matrix_t);
        A_direct.vmult(m_solution_t,
	               m_system_rhs_t);
      }
    else if (m_parameters.m_type_linear_solver == "CG")
      {
	SolverControl            solver_control_t(1e6, 1e-15);
	SolverCG<Vector<double>> cg_t(solver_control_t);

	PreconditionJacobi<SparseMatrix<double>> preconditioner_t;
	preconditioner_t.initialize(m_tangent_matrix_t, 1.0);

	cg_t.solve(m_tangent_matrix_t,
	           m_solution_t,
		   m_system_rhs_t,
	           preconditioner_t);
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected linear solver not implemented for the temperature subproblem!"));
      }

    m_constraints_t.distribute(m_solution_t);

    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::solve_linear_system_d()
  {
    m_timer.enter_subsection("Solve phase-field linear system");

    if (m_parameters.m_type_linear_solver == "Direct")
      {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(m_tangent_matrix_d);
        A_direct.vmult(m_solution_d,
	               m_system_rhs_d);
      }
    else if (m_parameters.m_type_linear_solver == "CG")
      {
	SolverControl            solver_control_phasefield(1e6, 1e-15);
	SolverCG<Vector<double>> cg_phasefield(solver_control_phasefield);

	PreconditionJacobi<SparseMatrix<double>> preconditioner_phasefield;
	preconditioner_phasefield.initialize(m_tangent_matrix_d, 1.0);

	cg_phasefield.solve(m_tangent_matrix_d,
			    m_solution_d,
			    m_system_rhs_d,
			    preconditioner_phasefield);
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected linear solver not implemented for the phase-field subproblem!"));
      }

    m_constraints_d.distribute(m_solution_d);

    m_timer.leave_subsection();
  }

  template <int dim>
  unsigned int SplitSolveTandUandD<dim>::
  solve_nonlinear_newton_u(Vector<double> & solution_delta_u,
		           unsigned int iter_stagger)
  {
    Vector<double> newton_update_u(m_dof_handler_u.n_dofs());
    newton_update_u = 0.0;

    double error_update_u_l2 = 0.0;
    double error_residual_u_l2 = 0.0;

    unsigned int newton_iteration = 0;
    for (; newton_iteration < m_parameters.m_max_iterations_u; ++newton_iteration)
      {
        make_constraints_u(newton_iteration, iter_stagger);
        assemble_system_u();

        Vector<double> error_res_u(m_dof_handler_u.n_dofs());

        for (unsigned int i = 0; i < m_dof_handler_u.n_dofs(); ++i)
          if (!m_constraints_u.is_constrained(i))
            error_res_u(i) = m_system_rhs_u(i);

        error_residual_u_l2 = error_res_u.l2_norm();

        if (    newton_iteration > 0
	     && error_residual_u_l2 <= 1.0e-9
             && error_update_u_l2 <= 1.0e-9
	   )
          {
	    if (m_parameters.m_output_iteration_history)
	      m_logfile << "   " << newton_iteration << "   "
			<< std::setprecision(3)
			<< std::setw(7)
			<< std::scientific
			<< error_residual_u_l2
			<< "  "
			<< error_update_u_l2
			<< std::flush;
            break;
          }

        solve_linear_system_u(newton_update_u);

        Vector<double> error_u(m_dof_handler_u.n_dofs());
        for (unsigned int i = 0; i < m_dof_handler_u.n_dofs(); ++i)
          if (!m_constraints_u.is_constrained(i))
            error_u(i) = newton_update_u(i);

        error_update_u_l2 = error_u.l2_norm();

        solution_delta_u += newton_update_u;

        update_qph_incremental(solution_delta_u);
      }

    AssertThrow(newton_iteration < m_parameters.m_max_iterations_u,
                ExcMessage("No convergence in nonlinear solver for U-subproblem!"));

    return newton_iteration;
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::output_results() const
  {
    m_timer.enter_subsection("Output results");

    DataOut<dim> data_out;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation_u(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name_u(dim, "displacement");

    data_out.add_data_vector(m_dof_handler_u,
	                     m_solution_u,
                             solution_name_u,
                             data_component_interpretation_u);

    data_out.add_data_vector(m_dof_handler_d,
			     m_solution_d,
			     "phasefield");

    data_out.add_data_vector(m_dof_handler_t,
			     m_solution_t,
			     "temperature");

    Vector<double> cell_material_id(m_triangulation.n_active_cells());
    // output material ID for each cell
    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
	cell_material_id(cell->active_cell_index()) = cell->material_id();
      }
    data_out.add_data_vector(cell_material_id, "materialID");

    //L2 projection
    DoFHandler<dim> dof_handler_L2(m_triangulation);
    FE_Q<dim>     fe_L2(m_parameters.m_poly_degree); //FE_Q element is continuous
    dof_handler_L2.distribute_dofs(fe_L2);
    AffineConstraints<double> constraints;
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
    constraints.close();
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
	  data_component_interpretation_L2(1,
				           DataComponentInterpretation::component_is_scalar);

    //stress L2 projection
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
	{
	  Vector<double> stress_field_L2;
	  stress_field_L2.reinit(dof_handler_L2.n_dofs());

	  MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
	  VectorTools::project(mapping,
			       dof_handler_L2,
			       constraints,
			       m_qf_cell,
			       [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
				    const unsigned int q) -> double
			       {
				 return m_quadrature_point_history.get_data(cell)[q]->get_cauchy_stress()[i][j];
			       },
			       stress_field_L2);

	  std::string stress_name = "Cauchy_stress_" + std::to_string(i+1) + std::to_string(j+1)
				  + "_L2";

	  data_out.add_data_vector(dof_handler_L2,
				   stress_field_L2,
				   stress_name,
				   data_component_interpretation_L2);
	}

    // Heat flux L2 projection
    Vector<double> heat_flux_field_L2_x;
    Vector<double> heat_flux_field_L2_y;
    Vector<double> heat_flux_field_L2_z;

    for (unsigned int i = 0; i < dim; ++i)
      {
	Vector<double> heat_flux_field_L2;
	heat_flux_field_L2.reinit(dof_handler_L2.n_dofs());

	MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
	VectorTools::project(mapping,
			     dof_handler_L2,
			     constraints,
			     m_qf_cell,
			     [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
				  const unsigned int q) -> double
			     {
			       return m_quadrature_point_history.get_data(cell)[q]->get_heat_flux()[i];
			     },
			     heat_flux_field_L2);

	if (i == 0)
	  heat_flux_field_L2_x = heat_flux_field_L2;
	else if (i == 1)
	  heat_flux_field_L2_y = heat_flux_field_L2;
	else if (i == 2)
	  heat_flux_field_L2_z = heat_flux_field_L2;
	else
	  AssertThrow(false,
	      	      ExcMessage("Heat flux output is wrong!"));
      }

    DoFHandler<dim> dof_handler_L2_flux(m_triangulation);
    FESystem<dim>   fe_flux_L2(FE_Q<dim>(m_parameters.m_poly_degree), dim);
    dof_handler_L2_flux.distribute_dofs(fe_flux_L2);
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
	  data_component_interpretation_flux_L2(dim,
				                DataComponentInterpretation::component_is_part_of_vector);

    Vector<double> heat_flux_field_L2;
    heat_flux_field_L2.reinit(dof_handler_L2_flux.n_dofs());

    if (dim == 2)
      {
	for (unsigned int i = 0; i < heat_flux_field_L2_x.size(); ++i)
	  {
	    heat_flux_field_L2(0+i*dim) = heat_flux_field_L2_x(i);
	    heat_flux_field_L2(1+i*dim) = heat_flux_field_L2_y(i);
	  }
      }

    if (dim == 3)
      {
	for (unsigned int i = 0; i < heat_flux_field_L2_x.size(); ++i)
	  {
	    heat_flux_field_L2(0+i*dim) = heat_flux_field_L2_x(i);
	    heat_flux_field_L2(1+i*dim) = heat_flux_field_L2_y(i);
	    heat_flux_field_L2(2+i*dim) = heat_flux_field_L2_z(i);
	  }
      }

    std::vector<std::string> solution_name_flux(dim, "Heat_flux_vector");
    data_out.add_data_vector(dof_handler_L2_flux,
    		             heat_flux_field_L2,
			     solution_name_flux,
			     data_component_interpretation_flux_L2);

    data_out.build_patches(m_parameters.m_poly_degree);

    std::ofstream output("Solution-" + std::to_string(dim) + "d-" +
			 Utilities::int_to_string(m_time.get_timestep(),4) + ".vtu");

    data_out.write_vtu(output);
    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::calculate_reaction_force(unsigned int face_ID)
  {
    m_timer.enter_subsection("Calculate reaction force");

    Vector<double>       system_rhs;
    system_rhs.reinit(m_dof_handler_u.n_dofs());

    Vector<double> cell_rhs(m_fe_u.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_dof_indices(m_fe_u.n_dofs_per_cell());

    const double time_ramp = (m_time.current() / m_time.end());

    const FEValuesExtractors::Vector displacements (0);

    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);
    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    FEValues<dim> fe_values(m_fe_u, m_qf_cell, uf_cell);
    FEFaceValues<dim> fe_face_values(m_fe_u, m_qf_face, uf_face);

    // shape function values for displacement field
    std::vector<std::vector<Tensor<1, dim>>>
      Nx(m_qf_cell.size(), std::vector<Tensor<1, dim>>(m_fe_u.n_dofs_per_cell()));
    std::vector<std::vector<Tensor<2, dim>>>
      grad_Nx(m_qf_cell.size(), std::vector<Tensor<2, dim>>(m_fe_u.n_dofs_per_cell()));
    std::vector<std::vector<SymmetricTensor<2, dim>>>
      symm_grad_Nx(m_qf_cell.size(), std::vector<SymmetricTensor<2, dim>>(m_fe_u.n_dofs_per_cell()));

    for (const auto &cell : m_dof_handler_u.active_cell_iterators())
      {
	// if calculate_reaction_force() is defined as const, then
	// we also need to put a const in std::shared_ptr,
	// that is, std::shared_ptr<const PointHistory<dim>>
	const std::vector<std::shared_ptr< PointHistory<dim>>> lqph =
	  m_quadrature_point_history.get_data(cell);
	Assert(lqph.size() == m_n_q_points, ExcInternalError());
        cell_rhs = 0.0;
        fe_values.reinit(cell);
        right_hand_side(fe_values.get_quadrature_points(),
    		        rhs_values,
    		        m_parameters.m_x_component*1.0,
    		        m_parameters.m_y_component*1.0,
    		        m_parameters.m_z_component*1.0);

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            for (const unsigned int k : fe_values.dof_indices())
              {
		Nx[q_point][k] = fe_values[displacements].value(k, q_point);
		grad_Nx[q_point][k] = fe_values[displacements].gradient(k, q_point);
		symm_grad_Nx[q_point][k] = symmetrize(grad_Nx[q_point][k]);
              }
          }

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();

            const std::vector<Tensor<1,dim>> & N = Nx[q_point];
            const std::vector<SymmetricTensor<2, dim>> & symm_grad_N = symm_grad_Nx[q_point];
            const double JxW = fe_values.JxW(q_point);

            for (const unsigned int i : fe_values.dof_indices())
              {
		cell_rhs(i) -= (symm_grad_N[i] * cauchy_stress) * JxW;
		// contributions from the body force to right-hand side
		cell_rhs(i) += N[i] * rhs_values[q_point] * JxW;
              }
          }

        // if there is surface pressure, this surface pressure always applied to the
        // reference configuration
        const unsigned int face_pressure_id = 100;
        const double p0 = 0.0;

        for (const auto &face : cell->face_iterators())
          {
	    if (face->at_boundary() && face->boundary_id() == face_pressure_id)
	      {
		fe_face_values.reinit(cell, face);

		for (const unsigned int f_q_point : fe_face_values.quadrature_point_indices())
		  {
		    const Tensor<1, dim> &N = fe_face_values.normal_vector(f_q_point);

		    const double         pressure  = p0 * time_ramp;
		    const Tensor<1, dim> traction  = pressure * N;

		    for (const unsigned int i : fe_values.dof_indices())
		      {
			const Tensor<1, dim> Nx =
			    fe_face_values[displacements].value(i, f_q_point);
			const double JxW = fe_face_values.JxW(f_q_point);
	    		cell_rhs(i) += Nx * traction * JxW;
		      }
		  }
	      }
          }

        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
      } // for (const auto &cell : m_dof_handler.active_cell_iterators())

    // The difference between the above assembled system_rhs and m_system_rhs
    // is that m_system_rhs is condensed by the m_constraints, which zero out
    // the rhs values associated with the constrained DOFs and modify the rhs
    // values associated with the unconstrained DOFs.

    std::vector< types::global_dof_index > mapping;
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(face_ID);
    DoFTools::map_dof_to_boundary_indices(m_dof_handler_u,
					  boundary_ids,
					  mapping);

    std::vector<double> reaction_force(dim, 0.0);

    for (unsigned int i = 0; i < m_dof_handler_u.n_dofs(); ++i)
      {
	if (mapping[i] != numbers::invalid_dof_index)
	  {
	    reaction_force[i % dim] += system_rhs(i);
	  }
      }

    for (unsigned int i = 0; i < dim; ++i)
      m_logfile << "\t\tReaction force in direction " << i << " on boundary ID " << face_ID
                << " = "
		<< std::fixed << std::setprecision(3) << std::setw(1)
                << std::scientific
		<< reaction_force[i] << std::endl;

    std::pair<double, std::vector<double>> time_force;
    time_force.first = m_time.current();
    time_force.second = reaction_force;
    m_history_reaction_force.push_back(time_force);

    m_timer.leave_subsection();
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::write_history_data()
  {
    m_logfile << "\t\tWrite history data ... \n"<<std::endl;

    std::ofstream myfile_reaction_force ("Reaction_force.hist");
    if (myfile_reaction_force.is_open())
    {
      myfile_reaction_force << 0.0 << "\t";
      if (dim == 2)
	myfile_reaction_force << 0.0 << "\t"
	       << 0.0 << std::endl;
      if (dim == 3)
	myfile_reaction_force << 0.0 << "\t"
	       << 0.0 << "\t"
	       << 0.0 << std::endl;

      for (auto const & time_force : m_history_reaction_force)
	{
	  myfile_reaction_force << time_force.first << "\t";
	  if (dim == 2)
	    myfile_reaction_force << time_force.second[0] << "\t"
	           << time_force.second[1] << std::endl;
	  if (dim == 3)
	    myfile_reaction_force << time_force.second[0] << "\t"
	           << time_force.second[1] << "\t"
		   << time_force.second[2] << std::endl;
	}
      myfile_reaction_force.close();
    }
    else
      m_logfile << "Unable to open file";

    std::ofstream myfile_energy ("Energy.hist");
    if (myfile_energy.is_open())
    {
      myfile_energy << std::fixed << std::setprecision(10) << std::scientific
                    << 0.0 << "\t"
                    << 0.0 << "\t"
	            << 0.0 << "\t"
	            << 0.0 << std::endl;

      for (auto const & time_energy : m_history_energy)
	{
	  myfile_energy << std::fixed << std::setprecision(10) << std::scientific
	                << time_energy.first     << "\t"
                        << time_energy.second[0] << "\t"
	                << time_energy.second[1] << "\t"
		        << time_energy.second[2] << std::endl;
	}
      myfile_energy.close();
    }
    else
      m_logfile << "Unable to open file";
  }

  template <int dim>
  double SplitSolveTandUandD<dim>::calculate_energy_functional() const
  {
    double energy_functional = 0.0;

    FEValues<dim> fe_values(m_fe_u, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler_u.active_cell_iterators())
      {
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            energy_functional += lqph[q_point]->get_total_strain_energy() * JxW;
            energy_functional += lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

    return energy_functional;
  }

  template <int dim>
  std::pair<double, double>
    SplitSolveTandUandD<dim>::calculate_total_strain_energy_and_crack_energy_dissipation() const
  {
    double total_strain_energy = 0.0;
    double crack_energy_dissipation = 0.0;

    FEValues<dim> fe_values(m_fe_u, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler_u.active_cell_iterators())
      {
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            total_strain_energy += lqph[q_point]->get_total_strain_energy() * JxW;
            crack_energy_dissipation += lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

    return std::make_pair(total_strain_energy, crack_energy_dissipation);
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::print_conv_header()
  {
    static const unsigned int l_width = 150;
    m_logfile << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;

    m_logfile << "\tStag-itr  "
	      << "Subp-1    "
              << "Subp-2   "
              << "No.Newton  Res.     Inc.   "
              << "   Subp-3  "
	      << "     Res_t"
	      << "      Res_u"
	      << "     Res_d"
	      << "      Inc_t"
	      << "      Inc_u"
	      << "      Inc_d"
	      << "      Energy"
	      << std::endl;

    m_logfile << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::print_parameter_information()
  {
    m_logfile << "Scenario number = " << m_parameters.m_scenario << std::endl;
    m_logfile << "Log file = " << m_parameters.m_logfile_name << std::endl;
    m_logfile << "Write iteration history to log file? = " << std::boolalpha
	      << m_parameters.m_output_iteration_history << std::endl;
    m_logfile << "Does the heat equation contain the coupling term? = " << std::boolalpha
	      << m_parameters.m_coupling_on_heat_eq << std::endl;
    m_logfile << "Is the thermal conductivity degraded by phasefield? = " << std::boolalpha
    	      << m_parameters.m_degrade_conductivity << std::endl;
    m_logfile << "Nonlinear solver type for the mechanical (u) subproblem = "
	      << m_parameters.m_type_nonlinear_solver << std::endl;
    m_logfile << "Linear solver type = " << m_parameters.m_type_linear_solver << std::endl;
    m_logfile << "Mesh refinement strategy = " << m_parameters.m_refinement_strategy << std::endl;

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	m_logfile << "\tMaximum adaptive refinement times allowed in each step = "
		  << m_parameters.m_max_adaptive_refine_times << std::endl;
	m_logfile << "\tMaximum allowed cell refinement level = "
	    	  << m_parameters.m_max_allowed_refinement_level << std::endl;
	m_logfile << "\tPhasefield-based refinement threshold value = "
		  << m_parameters.m_phasefield_refine_threshold << std::endl;
	//AssertThrow(false,
	//      	    ExcMessage("Adaptive mesh refinement strategy not implemented for ut-d staggered approach!"));
      }

    m_logfile << "Global refinement times = " << m_parameters.m_global_refine_times << std::endl;
    m_logfile << "Local prerefinement times = " <<m_parameters. m_local_prerefine_times << std::endl;
    m_logfile << "Allowed maximum h/l ratio = " << m_parameters.m_allowed_max_h_l_ratio << std::endl;
    m_logfile << "Total number of material types = " << m_parameters.m_total_material_regions << std::endl;
    m_logfile << "Material data file name = " << m_parameters.m_material_file_name << std::endl;
    if (m_parameters.m_reaction_force_face_id >= 0)
      m_logfile << "Calculate reaction forces on Face ID = " << m_parameters.m_reaction_force_face_id << std::endl;
    else
      m_logfile << "No need to calculate reaction forces." << std::endl;

    m_logfile << "Body force = (" << m_parameters.m_x_component << ", "
                                  << m_parameters.m_y_component << ", "
	                          << m_parameters.m_z_component << ") (N/m^3)"
				  << std::endl;
    m_logfile << "Heat supply = " << m_parameters.m_heat_supply << " (Watt/m^3)"
	      << std::endl;
    m_logfile << "Reference temperature = " << m_parameters.m_ref_temperature << " (K)"
	      << std::endl;

    m_logfile << "End time = " << m_parameters.m_end_time << std::endl;
    m_logfile << "Time data file name = " << m_parameters.m_time_file_name << std::endl;
  }


  template <int dim>
  bool SplitSolveTandUandD<dim>::local_refine_and_solution_transfer()
  {
    bool mesh_is_same = true;
    bool cell_refine_flag = true;

    unsigned int material_id;
    double length_scale;
    double cell_length;
    while(cell_refine_flag)
      {
	cell_refine_flag = false;

	std::vector<types::global_dof_index> local_dof_indices(m_fe_d.dofs_per_cell);
	for (const auto &cell : m_dof_handler_d.active_cell_iterators())
	  {
	    cell->get_dof_indices(local_dof_indices);

	    for (unsigned int i = 0; i< m_fe_d.dofs_per_cell; ++i)
	      {
		if (  m_solution_d(local_dof_indices[i])
		    > m_parameters.m_phasefield_refine_threshold )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (dim == 2)
		      cell_length = std::sqrt(cell->measure());
		    else
		      cell_length = std::cbrt(cell->measure());
		    if (  cell_length
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			if (cell->level() < m_parameters.m_max_allowed_refinement_level)
			  {
			    cell->set_refine_flag();
			    break;
			  }
		      }
		  }
	      }
	  }

	for (const auto &cell : m_dof_handler_d.active_cell_iterators())
	  {
	    if (cell->refine_flag_set())
	      {
		cell_refine_flag = true;
		break;
	      }
	  }

	// if any cell is refined, we need to project the solution
	// to the newly refined mesh
	if (cell_refine_flag)
	  {
	    mesh_is_same = false;

	    std::vector<Vector<double> > old_solutions_d(2);
	    old_solutions_d[0] = m_solution_d;
	    old_solutions_d[1] = m_solution_previous_d;

	    // history variable field L2 projection
	    DoFHandler<dim> dof_handler_L2(m_triangulation);
	    FE_DGQ<dim>     fe_L2(m_parameters.m_poly_degree); //Discontinuous Galerkin
	    dof_handler_L2.distribute_dofs(fe_L2);
	    AffineConstraints<double> constraints;
	    constraints.clear();
	    //Since we use discontinuous Lagrange polynomials as shape functions
	    //we don't need to worry about enforcing continuity of the history variable
	    //at hanging nodes.
	    //DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
	    constraints.close();

	    Vector<double> old_history_variable_field_L2;
	    old_history_variable_field_L2.reinit(dof_handler_L2.n_dofs());

	    MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
	    VectorTools::project(mapping,
			     dof_handler_L2,
			     constraints,
			     m_qf_cell,
			     [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
				  const unsigned int q) -> double
			     {
			       return m_quadrature_point_history.get_data(cell)[q]->get_history_max_positive_strain_energy();
			     },
			     old_history_variable_field_L2);

	    m_triangulation.prepare_coarsening_and_refinement();
	    SolutionTransfer<dim, Vector<double>> solution_transfer_u(m_dof_handler_u);
	    solution_transfer_u.prepare_for_coarsening_and_refinement(m_solution_previous_u);
	    SolutionTransfer<dim, Vector<double>> solution_transfer_t(m_dof_handler_t);
	    solution_transfer_t.prepare_for_coarsening_and_refinement(m_solution_previous_t);
	    SolutionTransfer<dim, Vector<double>> solution_transfer_d(m_dof_handler_d);
	    solution_transfer_d.prepare_for_coarsening_and_refinement(old_solutions_d);
	    SolutionTransfer<dim, Vector<double>> solution_transfer_history_variable(dof_handler_L2);
	    solution_transfer_history_variable.prepare_for_coarsening_and_refinement(old_history_variable_field_L2);
	    m_triangulation.execute_coarsening_and_refinement();

	    setup_system();

	    dof_handler_L2.distribute_dofs(fe_L2);
	    constraints.clear();
	    //Since we use discontinuous Lagrange polynomials as shape functions
	    //we don't need to worry about enforcing continuity of the history variable
	    //at hanging nodes.
	    //DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
	    constraints.close();

	    Vector<double> tmp_solution_previous_u;
	    tmp_solution_previous_u.reinit(m_dof_handler_u.n_dofs());
	    Vector<double> tmp_solution_previous_t;
	    tmp_solution_previous_t.reinit(m_dof_handler_t.n_dofs());
	    std::vector<Vector<double>> tmp_solutions_d(2);
	    tmp_solutions_d[0].reinit(m_dof_handler_d.n_dofs());
	    tmp_solutions_d[1].reinit(m_dof_handler_d.n_dofs());

	    Vector<double> new_history_variable_field_L2;
	    new_history_variable_field_L2.reinit(dof_handler_L2.n_dofs());

#  if DEAL_II_VERSION_GTE(9, 7, 0)
	    solution_transfer_u.interpolate(tmp_solution_previous_u);
#  else
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
	    solution_transfer_u.interpolate(m_solution_previous_u, tmp_solution_previous_u);
#  endif

#  if DEAL_II_VERSION_GTE(9, 7, 0)
	    solution_transfer_d.interpolate(tmp_solutions_d);
#  else
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
	    solution_transfer_d.interpolate(old_solutions_d, tmp_solutions_d);
#  endif

#  if DEAL_II_VERSION_GTE(9, 7, 0)
	    solution_transfer_t.interpolate(tmp_solution_previous_t);
#  else
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
	    solution_transfer_t.interpolate(m_solution_previous_t, tmp_solution_previous_t);
#  endif

#  if DEAL_II_VERSION_GTE(9, 7, 0)
            solution_transfer_history_variable.interpolate(new_history_variable_field_L2);
#  else
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
            solution_transfer_history_variable.interpolate(old_history_variable_field_L2, new_history_variable_field_L2);
#  endif

	    m_solution_previous_u = tmp_solution_previous_u;
	    m_solution_previous_t = tmp_solution_previous_t;

	    m_solution_previous_d = tmp_solutions_d[1];
	    m_solution_d = tmp_solutions_d[0];

	    // make sure the projected solutions still satisfy
	    // hanging node constraints
	    m_constraints_u.distribute(m_solution_previous_u);
	    m_constraints_d.distribute(m_solution_previous_d);
	    m_constraints_d.distribute(m_solution_d);
	    m_constraints_t.distribute(m_solution_previous_t);
	    //Since we use discontinuous Lagrange polynomials as shape functions
	    //we don't need to worry about enforcing continuity of the history variable
	    //at hanging nodes.
	    //constraints.distribute(new_history_variable_field_L2);

	    // new_history_variable_field_L2 contains the history variable projected
	    // onto the newly refined mesh
	    FEValues<dim> fe_values(fe_L2,
				    m_qf_cell,
				    update_values | update_gradients |
				    update_quadrature_points | update_JxW_values);

	    for (const auto &cell : dof_handler_L2.active_cell_iterators())
	      {
		fe_values.reinit(cell);

		const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
		      m_quadrature_point_history.get_data(cell);

		std::vector<double> history_variable_values_cell(m_n_q_points);

		fe_values.get_function_values(
		    new_history_variable_field_L2, history_variable_values_cell);

		for (unsigned int q_point : fe_values.quadrature_point_indices())
		  {
		    lqph[q_point]->assign_history_variable(history_variable_values_cell[q_point]);
		  }
	      }
	  } // if (cell_refine_flag)
      } // while(cell_refine_flag)

    // calculate field variables for newly refined cells
    if (!mesh_is_same)
      {
	m_solution_u = m_solution_previous_u;
	m_solution_t = m_solution_previous_t;
	m_solution_d = m_solution_previous_d;

	Vector<double> solution_delta_u(m_dof_handler_u.n_dofs());
	solution_delta_u = 0.0;
	update_qph_incremental(solution_delta_u);

	m_logfile << "\t\tUpdate field variables" << std::endl;

	//Since we want to map the history variable in the previous time step
	//from the coarse mesh to the refined mesh, we should not update them here.
	//update_history_field_step();
      }

    return mesh_is_same;
  }

  template <int dim>
  void SplitSolveTandUandD<dim>::run()
  {
    print_parameter_information();

    read_material_data(m_parameters.m_material_file_name,
    		       m_parameters.m_total_material_regions);

    std::vector<std::array<double, 4>> time_table;

    read_time_data(m_parameters.m_time_file_name, time_table);

    make_grid();

    setup_system();

    // initial conditions for temperature field
    setup_temperature_initial_conditions();

    output_results();

    while(m_time.current() < m_time.end() - m_time.get_delta_t()*1.0e-6)
      {
	m_time.increment(time_table);

	m_logfile << std::endl
		  << "Timestep " << m_time.get_timestep() << " @ " << m_time.current()
		  << 's' << std::endl;

        bool mesh_is_same = false;

	// solutions from the previous time step
	m_solution_previous_t = m_solution_t;
	m_solution_previous_d = m_solution_d;
	m_solution_previous_u = m_solution_u;

	double energy_functional_current = 0.0;

        // local adaptive mesh refinement loop
	unsigned int adp_refine_iteration = 0;
        for (; adp_refine_iteration < m_parameters.m_max_adaptive_refine_times + 1; ++adp_refine_iteration)
          {
            if (m_parameters.m_refinement_strategy == "adaptive-refine")
              m_logfile << "\tAdaptive refinement-"
	                << adp_refine_iteration << ": " << std::endl;

	    Vector<double> solution_d_prev_iter(m_dof_handler_d.n_dofs());
	    Vector<double> solution_d_diff(m_dof_handler_d.n_dofs());
	    Vector<double> solution_u_prev_iter(m_dof_handler_u.n_dofs());
	    Vector<double> solution_u_diff(m_dof_handler_u.n_dofs());
	    Vector<double> solution_t_prev_iter(m_dof_handler_t.n_dofs());
	    Vector<double> solution_t_diff(m_dof_handler_t.n_dofs());

	    if (m_parameters.m_output_iteration_history)
	      print_conv_header();

	    unsigned int linear_solve_needed = 0;

	    unsigned int iter_stagger = 1;

	    for (; iter_stagger <= m_parameters.m_max_staggered_iteration;
		 ++iter_stagger)
	      {
		if (m_parameters.m_output_iteration_history)
		  m_logfile << '\t' << std::setw(4) << iter_stagger
			    << std::flush;

		// staggered approach:
		// first, solve the thermal subproblem (a linear problem)
		if (m_parameters.m_output_iteration_history)
		  m_logfile << "\tsub-T (l)" << std::flush;

		// if the thermal conductivity is not degraded and
		// the thermal coupling term of the heat equation is turned off
		// then the heat conduction problem does not depend on the phase-field
		// nor the displacement. In this case, we only need to solve the
		// heat conduction problem ONCE.
		if (iter_stagger == 1)
		  {
		    temperature_step();
		    ++linear_solve_needed;
		  }
		else
		  {
		    if (   m_parameters.m_degrade_conductivity
			|| m_parameters.m_coupling_on_heat_eq
			)
		      {
			temperature_step();
			++linear_solve_needed;
		      }
		  }

		// second, solve the mechanical subproblem (a nonlinear problem)
		if (m_parameters.m_output_iteration_history)
		  m_logfile << "  sub-U (nl)" << std::flush;
		linear_solve_needed += displacement_step(iter_stagger);

		// last, solve the phase-field subproblem (a linear problem)
		if (m_parameters.m_output_iteration_history)
		  m_logfile << "  sub-PF (l)" << std::flush;
		phasefield_step();
		++linear_solve_needed;

		// calculate the residual of the T-subproblem
		if (iter_stagger == 1)
		  {
		    assemble_rhs_t();
		  }
		else
		  {
		    if (   m_parameters.m_degrade_conductivity
			|| m_parameters.m_coupling_on_heat_eq
			)
		      {
			assemble_rhs_t();
		      }
		  }

		// calculate the residual of the u-subproblem
		assemble_rhs_u();

		// calculate the residual of the phasefield-subproblem
		assemble_rhs_d();

		// calculate the temperature increment
		solution_t_diff = m_solution_t - solution_t_prev_iter;

		// calculate the displacement increment
		solution_u_diff = m_solution_u - solution_u_prev_iter;

		// calculate the phasefield increment
		solution_d_diff = m_solution_d - solution_d_prev_iter;

		for (unsigned int i = 0; i < m_dof_handler_u.n_dofs(); ++i)
		  {
		    if (m_constraints_u.is_constrained(i))
		      {
			solution_u_diff(i) = 0.0;
			m_system_rhs_u(i) = 0.0;
		      }
		  }
		double u_inc_l2 = solution_u_diff.l2_norm();
		double u_residual_l2 = m_system_rhs_u.l2_norm();

		for (unsigned int i = 0; i < m_dof_handler_d.n_dofs(); ++i)
		  {
		    if (m_constraints_d.is_constrained(i))
		      {
			solution_d_diff(i) = 0.0;
			m_system_rhs_d(i) = 0.0;
		      }
		  }
		double d_inc_l2 = solution_d_diff.l2_norm();
		double d_residual_l2 = m_system_rhs_d.l2_norm();

		for (unsigned int i = 0; i < m_dof_handler_t.n_dofs(); ++i)
		  {
		    if (m_constraints_t.is_constrained(i))
		      {
			solution_t_diff(i) = 0.0;
			m_system_rhs_t(i) = 0.0;
		      }
		  }
		double t_inc_l2 = solution_t_diff.l2_norm();
		double t_residual_l2 = m_system_rhs_t.l2_norm();

		energy_functional_current = calculate_energy_functional();

		if (m_parameters.m_output_iteration_history)
		  {
		    m_logfile << "  "
			      << t_residual_l2 << "  "
			      << u_residual_l2 << "  "
			      << d_residual_l2 << "  "
			      << t_inc_l2 << "  "
			      << u_inc_l2 << "  "
			      << d_inc_l2 << "  "
			      << std::fixed << std::setprecision(6) << std::scientific
			      << energy_functional_current
			      << std::endl;
		  }

		if (   (iter_stagger > 1)
		    && (u_residual_l2 < m_parameters.m_tol_u_residual)
		    && (t_residual_l2 < m_parameters.m_tol_t_residual)
		    && (d_residual_l2 < m_parameters.m_tol_d_residual)
		    && (u_inc_l2 < m_parameters.m_tol_u_incr)
		    && (t_inc_l2 < m_parameters.m_tol_t_incr)
		    && (d_inc_l2 < m_parameters.m_tol_d_incr)   )
		  {
		    if (m_parameters.m_output_iteration_history)
		      {
			m_logfile << '\t';
			for (unsigned int i = 0; i < 150; ++i)
			  m_logfile << '_';
			m_logfile << std::endl;
		      }

		    m_logfile << "\tStaggered approach (T-u-d) converges after "
			      << iter_stagger << " iterations."
			      << std::endl;

		    m_logfile << "\tTotally " << linear_solve_needed
			      << " linear solves are required."
			      << std::endl;

		    m_logfile << "\t\tAbsolute residual of mechanical (u) equation: "
			      << u_residual_l2 << std::endl;

		    m_logfile << "\t\tAbsolute residual of thermal (T) equation: "
			      << t_residual_l2 << std::endl;

		    m_logfile << "\t\tAbsolute residual of phasefield (d) equation: "
			      << d_residual_l2 << std::endl;

		    m_logfile << "\t\tAbsolute increment of displacement (u): "
			      << u_inc_l2 << std::endl;

		    m_logfile << "\t\tAbsolute increment of temperature (T): "
			      << t_inc_l2 << std::endl;

		    m_logfile << "\t\tAbsolute increment of phasefield (d): "
			      << d_inc_l2 << std::endl;

		    break;
		  }
		else
		  {
		    solution_t_prev_iter = m_solution_t;
		    solution_u_prev_iter = m_solution_u;
		    solution_d_prev_iter = m_solution_d;
		  }
	      } // 	for (; iter_stagger <= m_parameters.m_max_staggered_iteration; ++iter_stagger)

	    if (iter_stagger > m_parameters.m_max_staggered_iteration)
	      {
		m_logfile << "After " << m_parameters.m_max_staggered_iteration << " iterations, "
			  << "no convergence is achieved in the u-t-d staggered approach."
			  << std::endl;
		AssertThrow(false,
			    ExcMessage("No convergence achieved in u-t-d staggered approach!"));
	      }

	    if (m_parameters.m_refinement_strategy == "adaptive-refine")
	      {

		if (adp_refine_iteration == m_parameters.m_max_adaptive_refine_times)
		  break;

		mesh_is_same = local_refine_and_solution_transfer();

		if (mesh_is_same)
		  break;
	      }
	    else if (m_parameters.m_refinement_strategy == "pre-refine")
	      {
	        break;
	      }
	    else
	      {
		AssertThrow(false,
		            ExcMessage("Selected mesh refinement strategy not implemented!"));
	      }
          } // ++adp_refine_iteration

	m_logfile << "\t\tUpdate history variable" << std::endl;
	update_history_field_step();

	// output vtk files every 10 steps if there are too
	// many time steps
	//if (m_time.get_timestep() % 10 == 0)
        output_results();

	m_logfile << "\t\tEnergy functional (J) = " << std::fixed << std::setprecision(10) << std::scientific
	          << energy_functional_current << std::endl;

	std::pair<double, double> energy_pair = calculate_total_strain_energy_and_crack_energy_dissipation();
	m_logfile << "\t\tTotal strain energy (J) = " << std::fixed << std::setprecision(10) << std::scientific
		  << energy_pair.first << std::endl;
	m_logfile << "\t\tCrack energy dissipation (J) = " << std::fixed << std::setprecision(10) << std::scientific
		  << energy_pair.second << std::endl;

	std::pair<double, std::array<double, 3>> time_energy;
	time_energy.first = m_time.current();
	time_energy.second[0] = energy_pair.first;
	time_energy.second[1] = energy_pair.second;
	time_energy.second[2] = energy_pair.first + energy_pair.second;
	m_history_energy.push_back(time_energy);

	int face_ID = m_parameters.m_reaction_force_face_id;
	if (face_ID >= 0)
	  calculate_reaction_force(face_ID);

        write_history_data();
      } // while(m_time.current() < m_time.end() - m_time.get_delta_t()*1.0e-6)
  }
} // namespace PhaseField_T_and_u_and_d

int main(int argc, char* argv[])
{

  using namespace dealii;

  if (argc != 2)
    AssertThrow(false,
    		ExcMessage("The number of arguments provided to the program has to be 2!"));

  const unsigned int dim = std::stoi(argv[1]);
  if (dim == 2 )
    {
      PhaseField_T_and_u_and_d::SplitSolveTandUandD<2> Phasefield2D("parameters.prm");
      Phasefield2D.run();
    }
  else if (dim == 3)
    {
      PhaseField_T_and_u_and_d::SplitSolveTandUandD<3> Phasefield3D("parameters.prm");
      Phasefield3D.run();
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("Dimension has to be either 2 or 3"));
    }

  return 0;
}
