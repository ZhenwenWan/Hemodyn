# Hybrid Simulation–Data Regression Framework

## Overview
This project implements a universal regression framework for modeling and predicting how complex systems change over time.
It is designed to work across diverse domains, including engineering systems, environmental processes, and biomedical applications.
The method combines mechanistic simulations with observational data to produce accurate, interpretable results.

Mechanistic simulations capture the underlying physics or logic of the system.
These simulations are reduced to lower-order models to minimize computational demands.
Observational data from sensors or experiments are used to calibrate and update the model parameters.
The calibration process is formulated as a regression problem.

Unlike rigid numerical simulations, the model retains flexibility through tunable parameters.
Unlike neural networks, the number of free parameters is kept small, reducing overfitting and data requirements.
Nonlinear relationships can still be represented without excessive computational cost.
The framework is suitable for systems with varying scales, configurations, or operating conditions.

The regression model integrates simulated features with real-world measurements.
Data assimilation ensures that model parameters are updated as new observations arrive.
This allows the model to adapt in real time to changes in the system.
The result is a model that is both predictive and diagnostic.

Potential applications include fault detection, forecasting, and system optimization.
Example use cases range from automotive climate control to large-scale weather analysis and medical condition monitoring.
The framework is implemented in Python and can be extended with domain-specific models.

Its modular design makes it adaptable to both academic research and industrial deployment.

---

## Core Methodology

### 1. Universal Physics Module
- Based on a **reduced-order advection–diffusion–reaction model**.  
- Configurable **dimensions**: 0D (lumped) or 1D (distributed).  
- Adjustable **domain size** and **spatial/temporal resolution**.  
- Physical parameters can be:
  - Fixed by configuration at initialization.
  - Estimated (regressed) from data during training.

This flexibility eliminates the requirement for a fully validated physics model while retaining interpretable physical meaning.

### 2. Data Integration
- Accepts **observational data** from sensors, experiments, or archives.  
- Data are aligned with simulation outputs for parameter estimation and state correction.

### 3. Hybrid Regression
- Treats parameter estimation as a **regression problem**.  
- Allows both **static** and **dynamic** parameters.  
- Can be solved via:
  - Classical regression methods (linear/nonlinear).
  - Physics-informed neural networks (PINNs).
  - Inverse modeling and optimization.

### 4. Data Assimilation
- Incorporates **sequential updating** with new observations.  
- Supports methods like **Kalman filtering**, **variational assimilation**, or **Bayesian updates**.  
- Enables real-time adaptation and fault detection.

### 5. Output & Interpretation
- Produces **predictions** that are both **quantitatively accurate** and **physically interpretable**.  
- Diagnostic outputs help identify anomalies, parameter drifts, or system malfunctions.

---

## Example: HemoDyn Instantiation
In HemoDyn, the universal physics module is configured for **lower extremity blood flow and oxygen transport**:

- **Physics Core**: Reduced-order advection–diffusion–reaction solver for blood flow and oxygen exchange.
- **Data Sources**: Pulse and SpO₂ readings from wearable devices.
- **Regression Target**: Vascular resistance, compliance, and oxygen consumption rates.
- **Application Goal**: Clinical interpretation and monitoring of vascular health.

---

## Adapting the Framework to Other Domains
To use this framework in a different domain:
1. Define your **system variables** and transport processes in the universal physics module.  
2. Configure **dimensionality**, **resolution**, and **initial parameters**.  
3. Identify parameters to **fix** and those to **regress from data**.  
4. Supply **observational datasets** relevant to your system.  
5. Choose an appropriate **regression/assimilation algorithm**.  

Example adaptation domains:
- HVAC temperature and airflow prediction.
- River pollution dispersion modeling.
- Battery thermal and chemical degradation tracking.
- Industrial chemical reactor monitoring.

---

## Example Workflow
python
from hybrid_framework import PhysicsModule, HybridRegressor, load_data

# 1. Configure physics module
physics = PhysicsModule(dim=1, size=1.2, resolution=100,
                        parameters={"diffusivity": 0.01, "velocity": 0.1})

# 2. Load observational data
obs_data = load_data("sensors.csv")

# 3. Initialize regression engine
regressor = HybridRegressor(physics, method="PINN")

# 4. Fit model to data
regressor.train(obs_data)

# 5. Predict and analyze
predictions = regressor.predict(t_span=[0, 60])
regressor.plot_results(predictions)



