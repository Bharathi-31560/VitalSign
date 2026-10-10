"""Logistic Regression - linear baseline for 30-day-style readmission prediction."""
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE
from src.ml_utils import train_and_evaluate

INFO = {
    "slug": "logistic_regression", "name": "Logistic Regression",
    "type": "Supervised - Binary classification", "family": "supervised",
    "summary": "A linear model that turns a weighted sum of patient features into a readmission probability.",
    "vital_use": "Transparent baseline: each coefficient shows whether a feature (e.g. prior inpatient visits) pushes risk up or down.",
    "how": [
        "Logistic Regression multiplies every input (days in hospital, number of medications, one-hot encoded diagnosis group, and so on) by a learned weight, adds them up and passes the total through an S-shaped sigmoid curve. The output is a number between 0 and 1: the probability that the patient is readmitted.",
        "Training searches for the weights that make the observed outcomes in the training patients most likely. L2 regularisation keeps the weights small so the model does not over-fit the 150+ encoded features."],
    "why": [
        "Hospitals need to understand WHY a patient is flagged. Because the model is linear, its coefficients can be read directly - a positive coefficient on number_inpatient means more prior inpatient stays raise the predicted risk.",
        "It trains in seconds on the standardised data and gives well-calibrated probabilities, so it is the natural baseline every more complex model in this project has to beat."],
    "strengths": ["Fast and stable", "Interpretable coefficients", "Good probability estimates"],
    "limits": ["Cannot capture interactions (e.g. age x medication) unless engineered", "Assumes a linear effect of each feature"],
    "show_params": ["C", "penalty", "max_iter", "solver"],
}


def run(data):
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    return train_and_evaluate(INFO, model, data, importance="coef")
