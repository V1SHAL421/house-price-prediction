from matplotlib import pyplot as plt
import numpy as np


def visualise_results(eval_metric: str, models):
    """Results from experiment are hardcoded"""
    eval_metrics = {
        "log-RMSE": ([0.400, 0.148, 0.140, 0.117], [0.017, 0.036, 0.034, 0.016]),
        "RMSE": ([84012.745, 52459.632, 46286.658, 23702.250], [6079.122, 44458.101, 38709.845, 4872.018]),
        "R²": ([0, 0.855, 0.871, 0.913], [0, 0.075, 0.068, 0.021]
)
    }

    if eval_metric not in eval_metrics:
        raise ValueError(f"Invalid eval_metric: {eval_metric}")
    
    means, stds = eval_metrics[eval_metric]
    x = np.arange(len(models)) # Create an array of integers to represent the models
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, means, yerr=stds, fmt='o', capsize=5, label=eval_metric)
    plt.xticks(x, models)
    plt.xlabel('Models')
    plt.ylabel(eval_metric)
    plt.title(f'{eval_metric} Model Comparison')
    plt.tight_layout()
    plt.show()