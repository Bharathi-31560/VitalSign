"""Support Vector Machine - maximum-margin classifier with an RBF kernel."""
from sklearn.svm import SVC

from src.config import RANDOM_STATE
from src.ml_utils import train_and_evaluate

MAX_TRAIN = 8000   # SVM training time grows ~quadratically -> stratified training subsample

INFO = {
    "slug": "svm", "name": "Support Vector Machine",
    "type": "Supervised - Binary classification (kernel method)", "family": "supervised",
    "summary": "Finds the boundary that separates readmitted from non-readmitted patients with the widest margin.",
    "vital_use": "Captures smooth non-linear boundaries in the standardised clinical feature space.",
    "how": [
        "An SVM looks for the decision boundary that keeps the two patient groups as far apart as possible. The RBF kernel lets that boundary bend, so it can separate groups that are not linearly separable.",
        "Only the patients closest to the boundary (the support vectors) define it. Probabilities are obtained with Platt scaling, so the prediction page can show a confidence value."],
    "why": [
        "Readmission risk is not a straight-line function of the clinical numbers; SVMs with an RBF kernel can model such curved relationships.",
        "SVMs require standardised inputs, which the project's preprocessing pipeline provides. Because training cost grows very quickly with data size, the SVM is trained on a stratified subsample of the training set and still evaluated on the full test set."],
    "strengths": ["Handles non-linear boundaries", "Effective in high dimensions"],
    "limits": ["Slow on large datasets", "Harder to interpret"],
    "show_params": ["C", "kernel", "gamma", "probability"],
}


def run(data):
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)
    return train_and_evaluate(INFO, model, data, max_train_samples=MAX_TRAIN)
