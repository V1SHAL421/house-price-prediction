from matplotlib import pyplot as plt
import numpy as np


def visualise_results(eval_metric: str, models):
    """Results from experiment are hardcoded"""
    eval_metrics = {
        "log-RMSE": ([0.397, 0.137, 0.132, 0.114], [0.022, 0.031, 0.025, 0.014]),
        "RMSE": ([82806.654, 49095.737, 44545.119, 22690.323], [6654.365, 38480.162, 31372.118, 2070.870]),
        "R²": ([0.0, 0.873, 0.884, 0.917], [0.0, 0.061, 0.047, 0.016])
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