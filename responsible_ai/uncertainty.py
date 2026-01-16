import numpy as np

def uncertainty_analysis(probs):
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)

    return {
        "mean_uncertainty": float(entropy.mean()),
        "high_uncertainty_ratio": float((entropy > np.percentile(entropy, 75)).mean())
    }
