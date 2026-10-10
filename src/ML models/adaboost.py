"""AdaBoost - boosting with re-weighted training patients."""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.ml_utils import train_and_evaluate

INFO = {
    "slug": "adaboost", "name": "AdaBoost",
    "type": "Supervised - Binary classification (boosting)", "family": "supervised",
    "summary": "Combines many very simple 'weak' trees, giving more weight to patients earlier trees got wrong.",
    "vital_use": "Focuses on hard-to-classify patients whose readmission outcome is difficult to predict.",
    "how": [
        "AdaBoost trains a sequence of very shallow trees. After each round, patients that were misclassified receive a higher weight, so the next tree concentrates on them.",
        "The final prediction is a weighted vote of all trees - trees that were more accurate get a bigger say."],
    "why": [
        "In readmission data a large group of patients is 'borderline': similar vitals, utilisation and medication profiles but different outcomes. AdaBoost deliberately spends more effort on these difficult cases.",
        "It is a useful contrast to Gradient Boosting: both are boosting methods, but AdaBoost re-weights patients while Gradient Boosting fits residual errors."],
    "strengths": ["Simple, few settings", "Focuses on hard cases", "Less prone to over-fitting than deep trees"],
    "limits": ["Sensitive to noisy labels", "Usually slightly weaker than gradient boosting"],
    "show_params": ["n_estimators", "learning_rate"],
}


def run(data):
    base = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    model = AdaBoostClassifier(estimator=base, n_estimators=100, learning_rate=0.5,
                               random_state=RANDOM_STATE)
    return train_and_evaluate(INFO, model, data, importance="tree")
