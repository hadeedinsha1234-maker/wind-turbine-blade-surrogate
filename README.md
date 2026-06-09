🌬️ Wind Turbine Blade Structural Surrogate

AI-accelerated structural analysis of NACA 0015 composite wind turbine blades using physics-informed machine learning.

🚀 Live Demo
[Launch Interactive Dashboard →](https://wind-turbine-blade-surrogate.streamlit.app)

📊 Key Results

All 6 surrogate models achieve R² above 0.97 on both tip deflection and Von Mises stress prediction. The best model (GPR) achieves R² = 1.0000 on both outputs with an inference time of 0.26 ms — representing a 265,000× speedup over FEM simulation.

🔬 Project Overview

This project builds a surrogate modeling framework that replaces expensive Finite Element Method simulations for composite wind turbine blade structural analysis. The surrogate predicts tip deflection and max Von Mises stress in under 1 millisecond with near-perfect accuracy.

🏗️ Methodology

- 1,000 FEM simulations using FEniCS on a NACA 0015 airfoil cross-section
- E-glass/epoxy composite material with fiber orientation and volume fraction variation
- 6 surrogate models trained and compared
- Sobol' global sensitivity analysis with 8,192 samples
- Live interactive dashboard deployed on Streamlit Cloud

📐 Composite Material

- Material: E-glass/epoxy composite
- Fiber: E-glass (E = 72 GPa)
- Matrix: Epoxy (E = 3.5 GPa)
- Micromechanics: Rule of Mixtures + Classical Laminate Theory
- Stiffness rotation: Transformed Q-matrix for arbitrary fiber angle

🤖 Models

- GPR — Gaussian Process Regression with Matérn kernel and uncertainty quantification
- XGBoost — Gradient boosted trees, fastest scalar model
- MLP — Multilayer Perceptron deep learning baseline
- Random Forest — Ensemble of decision trees
- GNN — Graph Neural Network operating directly on the airfoil mesh
- PI-GNN — Physics-Informed GNN with linear elasticity constraint

🎯 Sobol' Sensitivity Analysis

Tip Deflection: Applied load F dominates (S1=0.464), followed by fiber angle θ (S1=0.336), and fiber volume fraction Vf (S1=0.091)

Max Von Mises Stress: Applied load F overwhelmingly dominates (S1=0.893), fiber angle θ is secondary (S1=0.086), Vf is negligible (S1=0.001)

📁 Repository Structure

- dashboard.py — Streamlit interactive dashboard
- gnn_model.py — GNN model architecture
- naca_mesh.py — NACA 0015 mesh generation
- naca_simulation.py — FEniCS FEM simulation
- generate_airfoil_dataset.py — Dataset generation loop
- train_all_models.py — GPR, XGBoost, MLP, RF training
- train_gnn_tuned.py — GNN and PI-GNN training
- sobol_analysis.py — Sobol' sensitivity analysis
- build_graphs.py — Graph construction for GNN
- airfoil_dataset.csv — 1,000 FEM simulation results
- trained_models.pkl — Trained scalar surrogate models
- gnn_final.pt — Trained GNN weights
- pignn_final.pt — Trained PI-GNN weights

🎯 Target Applications

- Wind turbine blade design optimization
- Digital twin for composite structures
- Real-time structural health monitoring
- AI-augmented FEM solvers

👤 Author

Hadeed Insha — Mechanical Engineer specializing in computational mechanics, Finite Element Analysis, composite materials, and physics-informed machine learning
