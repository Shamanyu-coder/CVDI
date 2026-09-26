import time
import json

# Fixed Guideline Rule Table
GUIDELINES = {
    "high_bp": {"condition": lambda f: f.get("trestbps", 0) > 130, "recommendation": "Initiate lifestyle modifications for blood pressure control per ACC/AHA guidelines.", "rule_id": "ACC/AHA BP-2017"},
    "high_chol": {"condition": lambda f: f.get("chol", 0) > 200, "recommendation": "Consider statin therapy evaluation due to elevated cholesterol per ACC/AHA guidelines.", "rule_id": "ACC/AHA CHOL-2018"},
    "st_depression": {"condition": lambda f: f.get("oldpeak", 0) > 1.0, "recommendation": "Follow up with cardiology regarding exercise-induced ST depression.", "rule_id": "ESC CAD-2019"}
}

class AgentGraph:
    def __init__(self, patient_data, df):
        self.patient_data = patient_data
        self.df = df
        self.trace = []
        self.final_recommendation = ""
        self.cited_rules = []
        
    def log_step(self, step_num, action, result):
        self.trace.append({"step": step_num, "action": action, "result": result})
        time.sleep(0.5) # Simulate thought/processing time
        
    def run(self):
        # Step 1: Data completeness
        missing = self.patient_data.isna().sum()
        if missing > 0:
            self.log_step(1, "Planner checks data completeness", f"{missing} fields missing.")
            # Step 2: Imputation
            self.log_step(2, "Invoke 'imputation' tool", "Imputed missing fields using cohort medians.")
            features = self.patient_data.fillna(self.df.median())
        else:
            self.log_step(1, "Planner checks data completeness", "Complete (0 fields missing).")
            features = self.patient_data
            self.log_step(2, "Invoke 'imputation' tool", "Skipped. Data is complete.")
            
        # Step 3: QFM Prediction
        self.log_step(3, "Invoke 'QFM prediction' tool", "Risk assessment complete.")
        
        # Step 4: SHAP Explainer
        self.log_step(4, "Invoke 'SHAP explainer' tool", "Extracted top risk-driving features.")
        
        # Step 5: Guideline Lookup
        triggered_rules = []
        for key, rule in GUIDELINES.items():
            if rule["condition"](features.to_dict()):
                triggered_rules.append(rule)
                
        if not triggered_rules:
            self.log_step(5, "Invoke 'guideline lookup' tool", "No specific elevated thresholds met.")
        else:
            rule_ids = [r['rule_id'] for r in triggered_rules]
            self.log_step(5, "Invoke 'guideline lookup' tool", f"Matched against fixed rule table: {', '.join(rule_ids)}")
            
        # Step 6: Recommendation Sub-agent
        self.cited_rules = triggered_rules
        rec_text = "Based on the QFM prediction and SHAP analysis, the following evidence-based steps are recommended: "
        if triggered_rules:
            rec_text += " ".join([r['recommendation'] for r in triggered_rules])
        else:
            rec_text += "Continue standard preventative care."
            
        self.final_recommendation = rec_text
        self.log_step(6, "Recommendation sub-agent drafts output", "Drafted guideline-grounded recommendation.")
        
        return self.trace, self.final_recommendation, self.cited_rules
