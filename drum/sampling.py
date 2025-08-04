import numpy as np
import torch

from .metric import clip_score
    
def coreset_sampling(data, n_sample = 0.1, weight = 1, n_approximate = 10, logit_scale = 100, seed = 42):
    data = np.array(data) if not isinstance(data, np.ndarray) else data
    n_sample = round(len(data) * n_sample) if isinstance(n_sample, float) or (isinstance(n_sample, int) and n_sample < 1) else n_sample
    n_sample = max(min(n_sample, len(data)), 1 if len(data) != 0 else 0)
    weight = 1 if weight is None else weight
    weight = np.transpose(weight) if np.ndim(weight) == 2 else (np.expand_dims(weight, axis = -1) if np.ndim(weight) == 1 else weight)
    
    random = ((np.random.RandomState(seed) if isinstance(seed, int) else seed) if seed is not None else np.random)
    if n_sample == len(data):
        indices = np.arange(n_sample)
    else:
        indices = []
        approx_data = data[random.choice(len(data), min(round(len(data) * n_approximate) if isinstance(n_approximate, float) else n_approximate, len(data)), replace = False)]
        dist = clip_score(data, approx_data, weight = weight, logit_scale = logit_scale, reduce = False)
        dist = np.mean(dist, axis = 1, keepdims = True)
        for i in range(n_sample):
            sample_index = np.argmax(dist)
            indices.append(sample_index)
            sample_dist = clip_score(data, data[[sample_index]], weight = weight, logit_scale = logit_scale, reduce = False)
            dist = np.minimum(dist, sample_dist)
            dist[sample_index] = -np.inf
    return indices
