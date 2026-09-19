"""Compare chosen routes using identical OpenROAD wire-edge resources.

This is a coarse-grid wire occupancy metric, not either router's native
congestion metric. Vias are checked and counted but contribute no edge demand.
Pin access, electrical connectivity, and detailed routability need separate
validation. Different route grids are projected by physical cell membership.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from ioplace.route_eval.joint import ResourceGrid
from ioplace.route_eval.topology import segment_union, union_metrics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_nets(path):
    """Stream complete named nets; reject incomplete or ambiguous syntax."""
    name, opened, records, seen = None, False, [], set()
    with Path(path).open() as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            if line == '(':
                if name is None or opened:
                    raise ValueError(f'invalid route start at line {number}')
                opened = True
            elif line == ')':
                if not opened:
                    raise ValueError(f'invalid route end at line {number}')
                yield name, records
                name, opened, records = None, False, []
            elif not opened:
                if name is not None or line in seen:
                    raise ValueError(f'duplicate net or missing delimiter at line {number}')
                seen.add(line)
                name = line
            else:
                fields = line.split()
                if len(fields) != 6:
                    raise ValueError(f'invalid segment at line {number}')
                x0, y0, l0, x1, y1, l1 = fields
                records.append((int(x0), int(y0), l0, int(x1), int(y1), l1))
    if opened or name is not None:
        raise ValueError('unterminated net')


def resource_arrays(raw):
    grid = ResourceGrid(raw['x_edges_dbu'], raw['y_edges_dbu'])
    names = [str(x) for x in raw['layer_names']]
    directions = [str(x).upper() for x in raw['layer_directions']]
    directions = [{'HORIZONTAL': 'H', 'VERTICAL': 'V'}.get(x, x) for x in directions]
    if len(set(names)) != len(names) or len(directions) != len(names):
        raise ValueError('invalid layer metadata')
    arrays = []
    for label in ('capacity', 'fixed_usage'):
        values = np.asarray(raw[label])
        if (values.shape != (len(names), grid.ny, grid.nx)
                or not np.isfinite(values).all() or (values < 0).any()
                or np.any(values != np.floor(values))):
            raise ValueError(f'invalid {label} array')
        edges = np.zeros((len(names), grid.edge_count), dtype=np.int64)
        for k, direction in enumerate(directions):
            if direction == 'H':
                edges[k, :grid.horizontal_count] = values[k, :, :-1].ravel()
            elif direction == 'V':
                edges[k, grid.horizontal_count:] = values[k, :-1, :].ravel()
            else:
                raise ValueError(f'unknown direction: {direction}')
        arrays.append(edges)
    return grid, names, directions, *arrays


def evaluate(path, baseline, *, region_grid=None, shift=(0, 0), scale=1.):
    grid, layer_names, directions, capacity, fixed = resource_arrays(baseline)
    layer_index = {name: i for i, name in enumerate(layer_names)}
    demand = np.zeros_like(capacity, dtype=np.int64)
    result = dict(routed_net_count=0, nets_with_geometry=0, wire_records=0,
                  via_records=0, nonpreferred_wire_records=0, wirelength_dbu=0.,
                  io_crossings=0 if region_grid is not None else None,
                  normalized_wirelength=0. if region_grid is not None else None)
    names, geometry_names = [], []
    # Access vias to M1 are allowed although the common wire resources start at M2.
    via_layers = set(layer_names)
    if 'metal2' in via_layers:
        via_layers.add('metal1')
    if 'layer_levels' in baseline:
        levels = np.asarray(baseline['layer_levels'])
        if (levels.shape != (len(layer_names),) or not np.isfinite(levels).all()
                or np.any(levels != np.floor(levels)) or len(set(levels.tolist())) != len(levels)):
            raise ValueError('invalid routing-layer levels')
        via_levels = dict(zip(layer_names, map(int, levels)))
        if 'metal2' in via_layers and 'metal1' not in via_levels:
            via_levels['metal1'] = via_levels['metal2'] - 1
    else:
        # Standard numeric layer suffixes encode physical routing order. For
        # opaque names the caller must supply explicit physical layer levels.
        via_levels = {name: int(match.group(1)) for name in via_layers
                      if (match := re.search(r'(\d+)$', name)) is not None}
    for name, records in read_nets(path):
        names.append(name)
        result['routed_net_count'] += 1
        if records:
            geometry_names.append(name)
            result['nets_with_geometry'] += 1
        layers = {}
        for x0, y0, l0, x1, y1, l1 in records:
            if (min(x0, x1) < grid.x_edges[0] or max(x0, x1) > grid.x_edges[-1]
                    or min(y0, y1) < grid.y_edges[0] or max(y0, y1) > grid.y_edges[-1]):
                raise ValueError(f'out-of-grid geometry in {name}: {(x0, y0, x1, y1)}')
            if l0 != l1:
                if x0 != x1 or y0 != y1 or l0 not in via_layers or l1 not in via_layers:
                    raise ValueError(f'invalid via in {name}')
                if l0 not in via_levels or l1 not in via_levels:
                    raise ValueError('explicit routing-layer levels required for opaque via layers')
                if abs(via_levels[l0] - via_levels[l1]) != 1:
                    raise ValueError(f'nonadjacent via in {name}')
                result['via_records'] += 1
                continue
            if l0 not in layer_index:
                raise ValueError(f'wire on unsupported layer {l0} in {name}')
            if x0 != x1 and y0 != y1:
                raise ValueError(f'nonrectilinear wire in {name}')
            if x0 == x1 and y0 == y1:
                continue
            result['wire_records'] += 1
            direction = 'H' if y0 == y1 else 'V'
            result['nonpreferred_wire_records'] += int(direction != directions[layer_index[l0]])
            layers.setdefault(l0, []).append(((x0, y0), (x1, y1)))
        for layer, segments in layers.items():
            geometry = segment_union(np.asarray(segments, dtype=float))
            # Each net consumes an edge once on each layer, even if branches overlap.
            keys = grid.edge_keys(geometry)
            demand[layer_index[layer], keys] += 1
            result['wirelength_dbu'] += float(np.abs(np.diff(geometry, axis=1)).sum())
            if region_grid is not None:
                metrics = union_metrics(region_grid, (geometry - np.asarray(shift)) * scale)
                result['io_crossings'] += metrics['crossings']
                result['normalized_wirelength'] += metrics['wirelength']
    overflow = np.maximum(fixed + demand - capacity, 0)
    result.update(common_wire_edge_overflow=int(overflow.sum()),
                  overflow_edge_count=int(np.count_nonzero(overflow)),
                  max_edge_overflow=int(overflow.max(initial=0)),
                  wire_edge_demand=int(demand.sum()),
                  fixed_edge_demand=int(fixed.sum()),
                  edge_capacity=int(capacity.sum()),
                  effective_edge_capacity=int(np.maximum(capacity - fixed, 0).sum()),
                  baseline_overflow=int(np.maximum(fixed - capacity, 0).sum()),
                  per_layer=[dict(name=name, overflow=int(overflow[k].sum()),
                                  wire_demand=int(demand[k].sum()),
                                  fixed_usage=int(fixed[k].sum()), capacity=int(capacity[k].sum()))
                             for k, name in enumerate(layer_names)],
                  net_names=sorted(names), geometry_net_names=sorted(geometry_names),
                  net_names_sha256=hashlib.sha256(json.dumps(sorted(names), separators=(',', ':')).encode()).hexdigest(),
                  semantics='Physical-cell projection; unique occupancy per net and layer; int64 wire counters; '
                            'OpenROAD fixed obstruction demand; no via demand; nonpreferred wires face zero capacity; '
                            'last native row/column omitted where no connecting edge exists. '
                            'This is not a native router overflow or a connectivity check.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments', type=Path, required=True)
    parser.add_argument('--capacity', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--coord', type=Path)
    parser.add_argument('--regions', type=Path)
    parser.add_argument('--reference-segments', type=Path)
    args = parser.parse_args()
    if bool(args.coord) != bool(args.regions):
        parser.error('--coord and --regions must be supplied together')
    kwargs = {}
    import ioplace.route_eval.joint as joint_module
    import ioplace.route_eval.topology as topology_module
    inputs = [args.segments, args.capacity, Path(__file__).resolve(),
              Path(joint_module.__file__), Path(topology_module.__file__)]
    if args.coord:
        from ioplace.region_grid import RegionGrid
        from ioplace.regions import RegionSet
        coord = json.loads(args.coord.read_text())
        kwargs = dict(region_grid=RegionGrid(RegionSet.from_json(args.regions)),
                      shift=coord['shift_factor'], scale=coord['scale_factor'])
        inputs += [args.coord, args.regions]
    if args.reference_segments:
        inputs.append(args.reference_segments)
    input_hashes = {str(p.resolve()): digest(p) for p in inputs}
    started = time.perf_counter()
    with np.load(args.capacity, allow_pickle=False) as baseline:
        result = evaluate(args.segments, baseline, **kwargs)
    if args.reference_segments:
        reference = {name for name, records in read_nets(args.reference_segments) if records}
        actual = set(result['geometry_net_names'])
        result['reference_cohort'] = dict(missing=sorted(reference - actual),
                                          extra_geometry=sorted(actual - reference),
                                          expected_count=len(reference), actual_count=len(actual))
    result['elapsed_s'] = time.perf_counter() - started
    if any(digest(p) != input_hashes[str(p.resolve())] for p in inputs):
        raise RuntimeError('evaluation input or source changed during execution')
    result['inputs_sha256'] = input_hashes
    result['validation_limit'] = 'Route connectivity and actual physical-pin coverage are separate prerequisites.'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('net_names', 'geometry_net_names')}))


if __name__ == '__main__':
    main()
