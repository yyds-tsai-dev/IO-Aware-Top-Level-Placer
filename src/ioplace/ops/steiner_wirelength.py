"""Frozen-topology interior FLUTE branch term from paper Eq. 7.

DREAMPlace supplies the weighted-average term.  This operator supplies only
the smooth lengths of branches attached to strict interior physical pins.
"""
import numpy as np
import torch

from ioplace.route_eval.topology import flute_tree


class FrozenSteinerWirelength:
    """Build FLUTE on detached positions, then evaluate its smooth branches."""

    def __init__(self, nl, num_nodes, net_mask=None, net_weights=None,
                 max_degree=256, coordinate_scale=1000.):
        if int(num_nodes) < 1:
            raise ValueError("positive num_nodes required")
        if not 4 <= int(max_degree) <= 256:
            raise ValueError("max_degree must be in [4,256]")
        if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
            raise ValueError("positive finite coordinate_scale required")
        self.nl = nl
        self.num_nodes = int(num_nodes)
        self.net_mask = net_mask
        self.net_weights = net_weights
        self.max_degree = int(max_degree)
        self.coordinate_scale = float(coordinate_scale)
        self.generation = 0
        self._node_ids = np.empty(0, dtype=np.int64)
        self._net_ids = np.empty(0, dtype=np.int64)
        self._offset_x = np.empty(0, dtype=np.float64)
        self._offset_y = np.empty(0, dtype=np.float64)
        self._anchor_x = np.empty(0, dtype=np.float64)
        self._anchor_y = np.empty(0, dtype=np.float64)
        self._movable = int(getattr(nl, "num_movable", self.num_nodes))

    def _split(self, pos):
        if not torch.is_tensor(pos) or pos.ndim != 1 or pos.numel() != 2 * self.num_nodes:
            raise ValueError("pos must be a flat Torch tensor of length 2*num_nodes")
        return pos[:self.num_nodes], pos[self.num_nodes:]

    @staticmethod
    def _host_bool(values, count, default=True):
        if values is None:
            return np.full(count, default, dtype=bool)
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=bool)
        if values.shape != (count,):
            raise ValueError("net_mask length must equal number of nets")
        return values

    def rebuild(self, pos):
        x, y = self._split(pos)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        starts = np.asarray(self.nl.flat_net2pin_start, dtype=np.int64)
        flat = np.asarray(self.nl.flat_net2pin, dtype=np.int64)
        pin2node = np.asarray(self.nl.pin2node, dtype=np.int64)
        offset_x = np.asarray(self.nl.pin_offset_x, dtype=np.float64)
        offset_y = np.asarray(self.nl.pin_offset_y, dtype=np.float64)
        num_nets = len(starts) - 1
        if getattr(self.nl, "num_nets", num_nets) != num_nets:
            raise ValueError("inconsistent net CSR")
        enabled = self._host_bool(self.net_mask, num_nets)

        nodes, nets, oxs, oys, anchors_x, anchors_y = [], [], [], [], [], []
        meta = {
            "total_nets": num_nets,
            "eligible_nets": 0,
            "wa_only_degree_le_3_nets": 0,
            "masked_nets": 0,
            "over_degree_nets": 0,
            "degenerate_nets": 0,
            "interior_pins": 0,
            "interior_branches": 0,
        }
        for net in range(num_nets):
            if not enabled[net]:
                meta["masked_nets"] += 1
                continue
            pin_ids = flat[starts[net]:starts[net + 1]]
            degree = len(pin_ids)
            if degree <= 3:
                meta["wa_only_degree_le_3_nets"] += 1
                continue
            if degree > self.max_degree:
                meta["over_degree_nets"] += 1
                continue
            node_ids = pin2node[pin_ids]
            if np.any(node_ids < 0) or np.any(node_ids >= self.num_nodes):
                raise ValueError("pin2node index outside pos")
            pins = np.column_stack((x[node_ids] + offset_x[pin_ids],
                                    y[node_ids] + offset_y[pin_ids]))
            if not np.isfinite(pins).all():
                raise ValueError("finite pin positions required")
            low, high = pins.min(axis=0), pins.max(axis=0)
            interior = np.all((pins > low) & (pins < high), axis=1)
            if not np.any(interior):
                meta["degenerate_nets"] += int(np.all(low == high))
                meta["eligible_nets"] += 1
                continue
            tree, _ = flute_tree(pins, coordinate_scale=self.coordinate_scale)
            terminal_rows = tree["terminal_nodes"]
            parents = tree["parents"][terminal_rows]
            positions = tree["positions"]
            if np.any(parents < degree):
                raise RuntimeError("FLUTE terminal-to-Steiner parent invariant violated")
            selected = np.flatnonzero(interior)
            # A raw terminal row has exactly one parent branch for degree >= 3.
            for local in selected:
                pin = pin_ids[local]
                parent = parents[local]
                nodes.append(node_ids[local])
                nets.append(net)
                oxs.append(offset_x[pin])
                oys.append(offset_y[pin])
                anchors_x.append(positions[parent, 0])
                anchors_y.append(positions[parent, 1])
            meta["eligible_nets"] += 1
            meta["interior_pins"] += len(selected)
            meta["interior_branches"] += len(selected)

        self._node_ids = np.asarray(nodes, dtype=np.int64)
        self._net_ids = np.asarray(nets, dtype=np.int64)
        self._offset_x = np.asarray(oxs, dtype=np.float64)
        self._offset_y = np.asarray(oys, dtype=np.float64)
        self._anchor_x = np.asarray(anchors_x, dtype=np.float64)
        self._anchor_y = np.asarray(anchors_y, dtype=np.float64)
        self.generation += 1
        meta["generation"] = self.generation
        return meta

    @staticmethod
    def _phi(delta, gamma):
        return delta * torch.tanh(delta / (2 * gamma))

    def __call__(self, pos, gamma):
        x, y = self._split(pos)
        gamma = torch.as_tensor(gamma, dtype=pos.dtype, device=pos.device)
        if gamma.numel() != 1 or not bool(torch.isfinite(gamma)) or not bool(gamma > 0):
            raise ValueError("positive finite scalar gamma required")
        if not len(self._node_ids):
            return pos[:0].sum()
        node_ids = torch.as_tensor(self._node_ids, dtype=torch.long, device=pos.device)
        net_ids = torch.as_tensor(self._net_ids, dtype=torch.long, device=pos.device)
        ox = torch.as_tensor(self._offset_x, dtype=pos.dtype, device=pos.device)
        oy = torch.as_tensor(self._offset_y, dtype=pos.dtype, device=pos.device)
        ax = torch.as_tensor(self._anchor_x, dtype=pos.dtype, device=pos.device)
        ay = torch.as_tensor(self._anchor_y, dtype=pos.dtype, device=pos.device)
        px, py = x[node_ids] + ox, y[node_ids] + oy
        fixed = node_ids >= self._movable
        px = torch.where(fixed, px.detach(), px)
        py = torch.where(fixed, py.detach(), py)
        branch = self._phi(px - ax, gamma) + self._phi(py - ay, gamma)
        if self.net_weights is not None:
            if torch.is_tensor(self.net_weights):
                weights = self.net_weights.to(device=pos.device, dtype=pos.dtype)
            else:
                weights = torch.as_tensor(self.net_weights, dtype=pos.dtype, device=pos.device)
            if weights.ndim != 1 or len(weights) != len(np.asarray(self.nl.flat_net2pin_start)) - 1:
                raise ValueError("net_weights length must equal number of nets")
            branch = branch * weights[net_ids]
        return branch.sum()
