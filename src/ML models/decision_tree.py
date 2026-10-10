"""Decision Tree - rule-based classifier."""
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.ml_utils import train_and_evaluate

INFO = {
    "slug": "decision_tree", "name": "Decision Tree",
    "type": "Supervised - Binary classification", "family": "supervised",
    "summary": "Learns a flowchart of yes/no questions about the patient that ends in a readmission decision.",
    "vital_use": "Produces human-readable rules such as 'prior inpatient visits > 1 and many medications -> high risk'.",
    "how": [
        "A decision tree repeatedly asks the question that best separates readmitted from non-readmitted patients - for example 'number_inpatient <= 0?'. Each answer sends the patient down a branch, and every leaf ends in a predicted class.",
        "To avoid memorising the training patients, the depth is limited (max_depth) and each leaf must contain a minimum number of patients (min_samples_leaf)."],
    "why": [
        "Clinicians can follow the path of a single patient through the tree and check that it is medically sensible, which makes this model easy to explain to non-technical staff.",
        "Trees handle the mixture of numeric measurements (lab procedures, medications) and categorical fields (diagnosis group, admission source) without needing extra feature engineering."],
    "strengths": ["Easy to visualise and explain", "Handles mixed data", "No feature scaling needed"],
    "limits": ["A single tree is unstable and can over-fit", "Usually less accurate than ensembles"],
    "show_params": ["max_depth", "min_samples_leaf", "criterion"],
}


def run(data):
    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE)
    return train_and_evaluate(INFO, model, data, importance="tree")
