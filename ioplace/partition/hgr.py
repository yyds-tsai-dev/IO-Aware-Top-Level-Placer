import numpy as np

def write_hgr(nl, path):
    lines = []
    n_written = 0
    for net in range(nl.num_nets):
        s, e = nl.flat_net2pin_start[net], nl.flat_net2pin_start[net + 1]
        nodes = np.unique(nl.pin2node[nl.flat_net2pin[s:e]])
        lines.append(" ".join(str(v + 1) for v in nodes))
        n_written += 1
    with open(path, "w") as f:
        f.write(f"{n_written} {nl.num_physical}\n")
        f.write("\n".join(lines) + "\n")
    return n_written
