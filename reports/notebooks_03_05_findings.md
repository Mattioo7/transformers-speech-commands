# Findings and conclusions from notebooks 03-05

This report summarizes the saved results from:

- `03_hyperparameter_experiments.ipynb`
- `04_split_unknown_known_experiment.ipynb`
- `05_sampling_experiment.ipynb`

Source CSV files:

- `reports/03_hyperparameter_experiments/hyperparameter_experiment_summary.csv`
- `reports/04_split_unknown_known_experiment/split_unknown_known_summary.csv`
- `reports/05_sampling_experiment/sampling_experiment_summary.csv`

Important caveat: notebook 03 used `train_fraction=0.2` and 10 epochs, while notebooks 04 and 05 used the full training split and 5 epochs. Results should therefore be compared mainly within each notebook, not as one global leaderboard.

## Notebook 03: hyperparameter experiments

Notebook 03 tested LSTM and Transformer models under fixed data settings, changing one hyperparameter at a time. The baseline setup used natural sampling, full validation/test splits, 20% of the training split, dropout `0.2`, learning rate `0.001`, weight decay `0.0001`, batch size `128`, and model capacity `64`.

### Best results by study

| Study | Architecture | Best value | Test accuracy | Validation accuracy |
|---|---:|---:|---:|---:|
| Architecture | LSTM | `lstm` | 87.57% | 88.14% |
| Architecture | Transformer | `transformer` | 87.64% | 87.92% |
| Batch size | LSTM | `32` | 89.90% | 89.29% |
| Batch size | Transformer | `32` | 89.08% | 89.69% |
| Capacity | LSTM | `128` | 89.47% | 90.03% |
| Capacity | Transformer | `128` | 88.69% | 88.41% |
| Dropout | LSTM | `0.1` | 87.94% | 87.30% |
| Dropout | Transformer | `0.0` | 89.27% | 89.18% |
| Learning rate | LSTM | `0.005` | 88.92% | 88.98% |
| Learning rate | Transformer | `0.001` | 87.64% | 87.92% |
| Weight decay | LSTM | `0.0` / `1e-05` | 87.77% | 87.35% |
| Weight decay | Transformer | `0.0001` | 87.64% | 87.92% |

### Findings

- The baseline LSTM and Transformer were nearly tied: 87.57% vs 87.64% test accuracy.
- Smaller batch sizes helped both architectures. Batch size `32` was the best tested value for both LSTM and Transformer, while batch size `256` reduced accuracy clearly.
- Increasing model capacity helped strongly. Capacity `128` was the best tested size for both models; capacity `16` was too small and produced much lower accuracy.
- Learning rate mattered more than weight decay. Very small learning rate `0.0001` underfit badly for both models. LSTM improved most with `0.005`, while Transformer performed best at the baseline `0.001`.
- Dropout behaved differently by architecture. Transformer worked best without dropout and degraded as dropout increased, especially at `0.4` and `0.5`. LSTM was less sensitive, with `0.1` slightly best.
- Weight decay had limited impact except for LSTM at `0.001`, which reduced test accuracy.

### Conclusion

For this setup, the strongest single changes were smaller batch size and larger capacity. A practical next configuration would use capacity `128`, batch size `32`, and architecture-specific regularization: low dropout for LSTM and no dropout for Transformer. The results do not show a decisive architecture winner under the baseline settings.

## Notebook 04: split unknown/known experiment

Notebook 04 tested whether a cascade is better than a single multiclass classifier. The cascade first decides whether a sample is known or `unknown`, then classifies known commands separately.

### Key results

| Architecture | Single multiclass test acc. | Cascade test acc. | Delta | Single known acc. | Cascade known acc. | Single unknown recall | Cascade unknown recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 92.33% | 90.71% | -1.62 pp | 85.08% | 84.42% | 96.70% | 94.49% |
| Transformer | 92.36% | 89.90% | -2.46 pp | 85.20% | 79.78% | 96.67% | 95.99% |

### Findings

- The cascade did not improve total test accuracy for either architecture.
- The single multiclass classifier was better by 1.62 percentage points for LSTM and 2.46 percentage points for Transformer.
- The binary known/unknown gate was strong in isolation, reaching 94.13% for LSTM and 92.85% for Transformer, but the full cascade still lost accuracy after routing samples through two models.
- For LSTM, the cascade reduced both known-command accuracy and unknown recall.
- For Transformer, the cascade preserved unknown recall fairly well, but known-command accuracy dropped sharply from 85.20% to 79.78%.

### Conclusion

The cascade adds complexity without improving accuracy. The likely issue is error compounding: mistakes made by the binary gate cannot be recovered by the second-stage classifier. For this dataset and model family, the single multiclass classifier is the better default.

## Notebook 05: sampling experiment

Notebook 05 compared three training-set sampling strategies while keeping validation and test sets natural:

- `natural`: no physical resampling.
- `unknown_undersampled`: reduce `unknown` to the size of the largest non-unknown class.
- `undersampled_then_oversampled`: reduce the largest class, then oversample smaller classes to the target size.

### Key results

| Strategy | Architecture | Train examples | Unknown train examples | Test accuracy | Validation accuracy |
|---|---|---:|---:|---:|---:|
| `natural` | LSTM | 52,667 | 32,550 | 92.66% | 92.32% |
| `natural` | Transformer | 52,667 | 32,550 | 91.56% | 91.47% |
| `unknown_undersampled` | LSTM | 22,002 | 1,885 | 76.16% | 75.23% |
| `unknown_undersampled` | Transformer | 22,002 | 1,885 | 77.87% | 76.95% |
| `undersampled_then_oversampled` | LSTM | 45,240 | 3,770 | 85.17% | 85.27% |
| `undersampled_then_oversampled` | Transformer | 45,240 | 3,770 | 83.15% | 82.42% |

### Findings

- Natural sampling was best for both architectures.
- Unknown undersampling caused the largest drop: LSTM fell by 16.51 percentage points and Transformer by 13.68 percentage points compared with natural sampling.
- The mixed undersampling/oversampling strategy was better than pure unknown undersampling, but still worse than natural sampling by 7.49 percentage points for LSTM and 8.40 percentage points for Transformer.
- Keeping many `unknown` examples appears useful. The `unknown` class is broad and diverse, so reducing it removes information rather than merely correcting imbalance.
- Oversampling smaller classes did not compensate for removing much of the natural `unknown` distribution.

### Conclusion

Do not physically rebalance the training set in this experiment setup. The natural distribution gives the best generalization on natural validation and test splits. If class imbalance needs further treatment, softer methods such as class weights, focal loss, or threshold tuning should be tested before aggressive resampling.

## Overall conclusions

- A single multiclass model is preferable to the two-stage known/unknown cascade.
- Natural sampling should remain the default because the `unknown` class carries useful variation.
- The most promising tuning directions are larger model capacity and smaller batch size.
- Transformer is not consistently better than LSTM in these experiments. LSTM often matches or beats Transformer, especially under natural sampling and several hyperparameter settings.
- Accuracy improvements should be validated with repeated seeds. Most notebook results appear to use a single seed, so small differences should not be treated as statistically stable.

## Recommended next steps

1. Run a combined best-configuration experiment using full training data, capacity `128`, batch size `32`, and the best architecture-specific dropout/learning-rate choices from notebook 03.
2. Keep natural sampling as the baseline for future experiments.
3. Compare the best LSTM and Transformer configurations with at least 3 random seeds.
4. If `unknown` behavior remains important, evaluate per-class precision/recall and try loss weighting instead of hard resampling.
