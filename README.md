# Agentic Quantum Foundation Model (AQFM) Demo

This is the interactive demo platform for the B.E. Phase III Project (23BAI70210 / 23BAI70185 / 23BAI70213).

## Setup & Run

1. Open your terminal in this directory.
2. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Architecture Notes
- **Frontend**: Streamlit (Clinical Dashboard style)
- **Quantum Backend**: PennyLane (`default.qubit` simulator) + Scikit-Learn
- **Explainability**: SHAP (`KernelExplainer`)
- **Agent Orchestrator**: Python State Machine Pattern (LangGraph proxy for robust standalone demo)

*Note: As this is a demo, certain quantum optimizations (like full-scale QAOA) are represented via toggleable ablation states to ensure the app remains interactive during live reviews.*
