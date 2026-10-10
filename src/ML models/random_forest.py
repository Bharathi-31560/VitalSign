"""Random Forest - bagged ensemble of decision trees."""
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE
from src.ml_utils import train_and_evaluate

INFO = {
    "slug": "random_forest", "name": "Random Forest",
    "type": "Supervised - Binary classification (ensemble)", "family": "supervised",
    "summary": "Hundreds of decision trees, each trained on a random sample of patients and features, vote together.",
    "vital_use": "Robust model for noisy hospital records; also ranks which clinical features matter most.",
    "how": [
        "A Random Forest builds many decision trees. Every tree sees a different random sample of the training patients and only a random subset of features at each split, so the trees make different mistakes.",
        "For a new patient, every tree votes and the forest averages the votes into a probability. Averaging many de-correlated trees removes most of the over-fitting of a single tree."],
    "why": [
        "Hospital encounter data is noisy: many patients have unknown payer codes or specialties, and risk depends on combinations such as diagnosis group + number of prior visits + medication changes. A forest captures these interactions automatically.",
        "The feature-importance chart shows which recorded variables (for instance number_inpatient or number_diagnoses) the forest relied on most."],
    "strengths": ["Strong accuracy with little tuning", "Captures non-linear interactions", "Built-in feature importance"],
    "limits": ["Less transparent than one tree", "Larger model file"],
    "show_params": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
}


def run(data):
    model = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=10,
                                   max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)
    return train_and_evaluate(INFO, model, data, importance="tree")
