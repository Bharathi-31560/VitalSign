"""K-Nearest Neighbours - predicts from the most similar past patients."""
from sklearn.neighbors import KNeighborsClassifier

from src.ml_utils import train_and_evaluate

MAX_TRAIN = 15000   # distance search is memory/time heavy -> stratified training subsample

INFO = {
    "slug": "knn", "name": "K-Nearest Neighbours",
    "type": "Supervised - Binary classification (instance based)", "family": "supervised",
    "summary": "Finds the k most similar past patients and predicts the majority outcome among them.",
    "vital_use": "Case-based reasoning: 'patients with a similar profile were readmitted X % of the time'.",
    "how": [
        "KNN stores the training patients. To classify a new patient it measures the distance to every stored patient (using the standardised numeric features and the one-hot encoded categories) and looks at the k = 25 closest ones.",
        "The prediction is the share of those neighbours who were readmitted. No explicit model is learned - the data is the model."],
    "why": [
        "It mirrors how clinicians often think - 'this patient looks like those previous patients' - so it is easy to justify.",
        "Standardising the vital/utilisation numbers (done in the preprocessing step) is essential here, otherwise a feature with a large range such as lab procedures would dominate the distance. To keep prediction fast, KNN is trained on a stratified sample of the training set."],
    "strengths": ["Very simple idea", "No training phase", "Naturally non-linear"],
    "limits": ["Slow on large data", "Struggles with many one-hot columns (curse of dimensionality)"],
    "show_params": ["n_neighbors", "weights", "metric"],
}


def run(data):
    model = KNeighborsClassifier(n_neighbors=25, weights="uniform", n_jobs=-1)
    return train_and_evaluate(INFO, model, data, max_train_samples=MAX_TRAIN)
