"""Gradient Boosting - sequential ensemble that corrects its own errors."""
from sklearn.ensemble import GradientBoostingClassifier

from src.config import RANDOM_STATE
from src.ml_utils import train_and_evaluate

INFO = {
    "slug": "gradient_boost", "name": "Gradient Boosting",
    "type": "Supervised - Binary classification (boosting)", "family": "supervised",
    "summary": "Builds small trees one after another, each one fixing the mistakes of the trees before it.",
    "vital_use": "Often the strongest tabular-data model; finds subtle combinations of utilisation and diagnosis features.",
    "how": [
        "Gradient Boosting starts with a very simple prediction and then adds shallow decision trees one at a time. Each new tree is trained to correct the errors the current ensemble still makes on the training patients.",
        "A small learning rate means each tree only nudges the prediction, so the model improves gradually and generalises well."],
    "why": [
        "Readmission risk in this dataset is driven by many weak signals (age, number of medications, lab tests, diagnosis groups, medication changes). Boosting is very good at adding many weak signals into a sharp overall risk score.",
        "Because it outputs probabilities, the prediction page can show a confidence value for each patient."],
    "strengths": ["High accuracy on tabular data", "Handles interactions", "Gives feature importance"],
    "limits": ["Slower to train (trees are sequential)", "More hyper-parameters to tune"],
    "show_params": ["n_estimators", "learning_rate", "max_depth", "subsample"],
}


def run(data):
    model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=3,
                                       subsample=0.8, random_state=RANDOM_STATE)
    return train_and_evaluate(INFO, model, data, importance="tree")
