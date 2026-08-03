import numpy as np
from ioplace.evaluator_ref import evaluate as evaluate_ref

class GpuEvalContext:
    def __init__(self, nl, rg, device="cuda"):
        self.nl, self.rg, self.device = nl, rg, device
        # 向量化時在此快取:grid tensor、Ph/Pv 前綴和、net degree 分桶索引

    def evaluate(self, node_x, node_y):
        to_np = lambda t: (t.detach().cpu().numpy() if hasattr(t, "detach")
                           else np.asarray(t))
        return evaluate_ref(self.nl, to_np(node_x).astype(np.float64),
                            to_np(node_y).astype(np.float64), self.rg)

def evaluate_gpu(nl, node_x, node_y, rg, max_degree=256, device="cuda"):
    return GpuEvalContext(nl, rg, device).evaluate(node_x, node_y)
