"""
Surrogate model integration for the Digital Twin UI.

Provides contact-pressure prediction using a trained neural network surrogate
loaded from the shared data/surrogate/models/latest/ directory.

Sub-modules:
    predictor      — MLflow-aware model loader + inference
    csar_engine    — CSAR vs insertion depth from surrogate predictions
    vtp_processor  — Read / annotate / write VTK PolyData (.vtp) files
"""
