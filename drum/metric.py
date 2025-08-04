import numpy as np

def clip_score(feature, ref_feature, logit_scale = 100.0, weight = 1, reduce = True):
    ref_feature = np.expand_dims(ref_feature, axis = 0) if np.ndim(ref_feature) == 2 else ref_feature
    batch_size, ref_size = np.shape(ref_feature)[:2]
    feature = feature / np.linalg.norm(feature, axis = -1, keepdims=True)
    ref_feature = ref_feature / np.linalg.norm(ref_feature, axis = -1, keepdims=True)
    sim = logit_scale * np.einsum("bf,btf->bt", feature, ref_feature)
    sim = sim * (np.expand_dims(weight, axis = 0) if np.ndim(weight) == 1 else weight)
    return sim.mean(axis = 1) if reduce else (sim[..., 0] if ref_size == 1 else sim)