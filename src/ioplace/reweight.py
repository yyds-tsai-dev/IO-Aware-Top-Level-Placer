import numpy as np
import torch

def update_net_weights(net_weights, per_net_crossings, alpha=0.5, cap=10.0):
    """w = 1 + alpha * min(crossings, cap), written in-place into net_weights.

    `net_weights` is DREAMPlace's live GPU tensor (placer.data_collections.
    net_weights) -- WeightedAverageWirelength and the preconditioner both hold
    a reference to this same object, so `copy_` (never rebinding the tensor)
    is required for the update to actually affect the optimizer.
    """
    c = torch.as_tensor(np.minimum(np.asarray(per_net_crossings, dtype=np.float32), cap),
                        device=net_weights.device, dtype=net_weights.dtype)
    net_weights.copy_(1.0 + alpha * c)
