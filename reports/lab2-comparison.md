# Lab 2 — Run comparison

Experiment `itcs355-lab2` · 12 trials · modeled total cost 0.0197 THB

`thb_per_point` is cost per percentage point of val_roc_auc above the worst trial. Cheap improvements rank low; expensive improvements rank high, however good the headline number is.

| run_id   | val_roc_auc | cost_thb | n_estimators | max_depth | min_samples_leaf | thb_per_point |
|----------|-------------|----------|--------------|-----------|------------------|---------------|
| 8438ab5b | 0.8426      | 0.0011   | 100          | 4         | 5                | 0.0007        |
| e8be4106 | 0.8424      | 0.0023   | 100          | 4         | 1                | 0.0014        |
| 7814957b | 0.8411      | 0.0021   | 300          | 4         | 5                | 0.0014        |
| a737f61b | 0.8404      | 0.0021   | 300          | 4         | 1                | 0.0015        |
| fc9c7fe7 | 0.8397      | 0.0009   | 100          | 8         | 5                | 0.0007        |
| 1247c714 | 0.8377      | 0.0022   | 300          | 8         | 5                | 0.002         |
| ef586344 | 0.8354      | 0.0021   | 300          | 12        | 5                | 0.0023        |
| 25888242 | 0.8338      | 0.0021   | 300          | 8         | 1                | 0.0029        |
| 2b673461 | 0.8322      | 0.0009   | 100          | 12        | 5                | 0.0016        |
| 34598841 | 0.8312      | 0.0009   | 100          | 8         | 1                | 0.0019        |
| 73d613be | 0.8268      | 0.0009   | 100          | 12        | 1                | 0.0266        |
| 0915f6d6 | 0.8265      | 0.0021   | 300          | 12        | 1                | 0.5469        |

## Compute constraint

The 12-trial comparison study was executed locally because this Azure for Students subscription had a low-priority quota of 0 vCPUs in the tested Azure regions, and the quota increase request returned `ResourceNotAvailableForOffer`. Therefore, the per-trial THB values in this table are modeled costs from the lab pricing table rather than actual discounted managed-compute charges. The managed Azure training job and registered model were completed separately.

## Which model did you register, and why?

I selected the 100-tree, depth-4, leaf-5 configuration (run 8438ab5b). It achieved the highest validation ROC-AUC in the 12-trial study (0.8426) while costing only 0.0011 THB for the trial. There was no different non-highest-scoring model to justify choosing instead. Across seeds 1–5, validation ROC-AUC ranged from 0.8442 to 0.8761, with a mean of 0.8578, showing that random seed affects the result. Using the measured local runtime and the configured Azure Standard_DS3_v2 rate, one retraining has a modeled cost of about 0.0011 THB, so monthly retraining would have an estimated modeled training cost of about 0.0011 THB (excluding other infrastructure costs). This choice could be wrong because the validation split and seed results may not represent future data; changes in the data distribution could make another configuration perform better.
