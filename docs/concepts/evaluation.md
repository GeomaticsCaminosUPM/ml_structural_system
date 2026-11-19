# Evaluation and feature importance

## Test scores

Each final model is scored once on the test set (`mlss test`):

| Metric | Meaning |
|---|---|
| Macro F1 | Mean F1 over the classes. The headline metric, since accuracy is dominated by the majority class |
| Accuracy | Share of buildings classified correctly |
| Cohen's κ | Agreement beyond what the class shares alone would give |
| True Skill Statistic (TSS) | Recall plus specificity minus one, averaged over the classes |
| Per-class F1 | One score per class |

A majority-class baseline (always predicting the most common class) gives the floor every model should beat. Scores are also broken down per source file and per `group` value.

## Assessment

`mlss assess` (and `mlss report`) compares the models on the same test buildings:

- **Agreement**: how many models agree on each building, Fleiss' κ across models, and buildings flagged as contested. Buildings on which the models disagree concentrate the errors.
- **Bootstrap confidence intervals** of the scores (`assessment.bootstrap`).
- **Generalizability**: the gap between the cross-validated and test scores.
- **Soft ensemble**: an average of the models' probabilities, weighted by their cross-validated scores.

## Feature importance

SHapley Additive exPlanations (SHAP) values for logistic regression and the tree models, and permutation importance (macro F1 lost when a feature is shuffled) for histogram gradient boosting. Each model ranks the features, and the consensus ranking is the mean rank across models (`report/feature_importance.csv`).
