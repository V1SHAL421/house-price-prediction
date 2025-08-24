from matplotlib import pyplot as plt
import numpy as np


def visualise_results(eval_metric: str, models):
    """Results from experiment are hardcoded"""
    eval_metrics = {
        "log-RMSE": ([0.399, 0.144, 0.136, 0.115], [0.019, 0.032, 0.029, 0.014]),
        "RMSE": ([83892.464, 49801.383, 43384.577, 23298.008], [6502.316, 38313.563, 31335.154, 4334.165]),
        "R²": ([0, 0.863, 0.878, 0.916], [0, 0.065, 0.056, 0.018]
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