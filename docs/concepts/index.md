# How it works

```mermaid
flowchart TD
    A[Footprints and survey] --> B[Features<br/>missing footprint indices computed]
    B --> C[Labels mapped, rare classes dropped]
    C --> D[Spatial split by blocks<br/>train 65% · validation 15% · test 20%]
    D --> E[Encoder fitted on train]
    E --> F[Cross-validated search on train<br/>hyperparameters × imbalance strategies]
    F --> G[Final fit on train]
    G --> H[Decision rule tuned on validation]
    H --> I[Scored once on test]
    I --> J[Assessment: agreement, intervals, feature importance]
```

Each stage has its own page:

- [Spatial split](splitting.md): why neighbors must stay in the same partition.
- [Model selection and imbalance](training.md): the five models, the search and the imbalance strategies.
- [Decision rule](decision-rule.md): how minority classes get predicted at all.
- [Evaluation and feature importance](evaluation.md): the metrics, the baseline and the assessment.

The method is described in detail in the paper *Estimating building structural systems from footprints: transferability of machine learning models across three Latin American cities*.
