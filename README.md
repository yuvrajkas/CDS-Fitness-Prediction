# CDS Fitness Readiness Prediction System

## Project Overview

This machine learning mini-project demonstrates how regression and classification algorithms can be applied to synthetic fitness data from CDS aspirants.

The system uses applicant fitness measurements to predict an illustrative fitness performance score and classify applicants according to predefined demonstration fitness targets.

## Machine Learning Algorithms

### 1. Linear Regression
Predicts a numerical fitness performance score based on the applicant's recorded measurements.

### 2. Logistic Regression
Classifies applicants into two categories:
- 1 — Meets the demonstration fitness target
- 0 — Needs improvement

## Dataset

The project uses a synthetic dataset containing 20 fictional applicants.

Features include:
- Age
- Height and weight
- Running time
- Push-up count
- Sit-up count
- Pull-up count
- Flexibility score

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Installation

Install the required libraries using:

```bash
python3 -m pip install -r requirements.txt
```

## How to Run

Keep `main.py` and `cds_fitness_applicants.csv` in the same folder.

Run the program:

```bash
python3 main.py
```

## Generated Outputs

The program generates:
- `regression_results.png`
- `classification_results.png`
- `prediction_results.csv`
- `cds_fitness_with_scores.csv`

## Limitations

The dataset and classification labels are synthetic. The fitness score is based on an illustrative formula and does not represent official CDS fitness standards, medical fitness, or military service eligibility.

The small dataset is intended for academic demonstration and does not establish reliable real-world prediction accuracy.
