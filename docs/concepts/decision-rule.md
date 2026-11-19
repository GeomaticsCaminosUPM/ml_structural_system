# Decision rule

Most buildings belong to the majority class (masonry, in the paper's cities), so picking the class with the highest probability rarely predicts the others.

Each class probability $p_c$ is divided by a class divisor $d_c$, and the class with the largest ratio wins:

$$
\hat{y} = \arg\max_c \; \frac{p_c}{d_c}
$$

A divisor below 1 makes its class easier to predict, above 1 harder. The divisors are tuned on the **validation** set to maximize macro F1 (or to minimize expected cost, with `calibration_objective: expected_cost`):

- coordinate-wise search, one class at a time
- 120 log-spaced values between 0.05 and 5 per class
- two passes over the classes

The validation set is used for nothing else, so the test scores stay unbiased. The divisors are saved as `trained_models/<Model>/thresholds_<Model>.npy` and applied by [`predict`][ml_structural_system.predict.predict].
