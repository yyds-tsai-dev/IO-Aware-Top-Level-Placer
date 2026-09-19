"""Host-side adapter for auditable full OpenROAD detailed routing."""
import hashlib
import json
from pathlib import Path
import subprocess

from ioplace.route_eval.route_crossings import evaluate_route_from_files
from scripts.stage2_fix_def_vias import (
    _DEF_VIA_NAME_RE, _VIAS_HEADER_RE, fix_def_vias, lef_via_names)


def _digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _safe_paths(paths):
    if any(any(char in str(path) for char in "{}\n\r") for path in paths):
        raise ValueError("unsupported Tcl path")


def _logged_errors(path):
    with Path(path).open(errors="replace") as stream:
        return [line.strip() for line in stream if "[ERROR " in line]


def _prepare_def_vias(def_path, lefs, cleaned_path):
    """Remove only verified LEF-vs-DEF VIA duplicates.

    The common path streams the DEF and leaves it untouched.  The existing
    narrow rewrite helper is invoked only when a conflicting name is found,
    avoiding a second full-size DEF copy for ordinary large designs.
    """
    names = lef_via_names(lefs)
    conflicts = set()
    inside = False
    with Path(def_path).open() as stream:
        for line in stream:
            stripped = line.rstrip("\n")
            if not inside:
                inside = _VIAS_HEADER_RE.match(stripped) is not None
                continue
            if stripped == "END VIAS":
                break
            match = _DEF_VIA_NAME_RE.match(stripped)
            if match and match.group(1) in names:
                conflicts.add(match.group(1))
    if not conflicts:
        return Path(def_path), {"applied": False, "dropped_names": [],
                                "original_count": None, "kept_count": None}
    original, cleaned, dropped, kept, count = fix_def_vias(def_path, lefs)
    if set(dropped) != conflicts:
        raise RuntimeError("DEF VIA preflight and rewrite disagree")
    Path(cleaned_path).write_text("".join(cleaned))
    return Path(cleaned_path), {"applied": True, "dropped_names": sorted(dropped),
                                "original_count": count, "kept_count": kept}


def run_detailed_route(export_dir, lefs, out, binary, threads=8):
    """Run full DR and measure IO crossings from decoded detailed wires.

    A zero process return code records tool completion only.  DRC violations,
    unrouted nets, native wirelength reconciliation, and placement identity
    are returned separately for downstream acceptance policy.
    """
    export_dir = Path(export_dir).resolve()
    out = Path(out).resolve()
    binary = Path(binary).resolve()
    lefs = [Path(path).resolve() for path in lefs]
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("positive integer threads required")
    required = [export_dir / name for name in
                ("out.def", "coord.json", "regions.json", "netmap.json")]
    inputs = [*required, *lefs, binary]
    if any(not path.is_file() for path in inputs):
        missing = [str(path) for path in inputs if not path.is_file()]
        raise FileNotFoundError("missing detailed-route input: " + ", ".join(missing))
    _safe_paths([out, *inputs])
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent / "or_scripts" / "run_detailed_route.py"
    clear = source.with_name("clear_signal_routing.tcl")
    source_root = Path(__file__).parents[2]
    dependencies = [
        Path(__file__), source, clear, source.with_name("dump_segments.py"),
        source.with_name("checked_tcl.py"),
        Path(__file__).with_name("route_crossings.py"),
        Path(__file__).with_name("segments.py"),
        source_root / "ioplace/evaluator_ref.py",
        source_root / "ioplace/region_grid.py", source_root / "ioplace/regions.py",
        source_root / "ioplace/netlist.py", source_root / "scripts/stage2_fix_def_vias.py",
    ]
    if any(not path.is_file() for path in dependencies):
        raise FileNotFoundError("missing detailed-route source dependency")
    route_def, via_fix = _prepare_def_vias(required[0], lefs, out / "fixed_vias.def")
    settings = {
        "def": str(route_def), "lefs": list(map(str, lefs)),
        "threads": threads, "clear_script": str(clear.resolve()),
        "cleared": str(out / "cleared_signal_nets.txt"),
        "routing_input": str(out / "routing_input.def"),
        "congestion": str(out / "congestion.rpt"),
        "guide": str(out / "route.guide"), "drc": str(out / "drc.rpt"),
        "routed_def": str(out / "routed.def"),
        "segments": str(out / "segments.npz"),
        "segments_json": str(out / "segments.json"),
        "identity_before": str(out / "identity_before.tsv"),
        "identity_after": str(out / "identity_after.tsv"),
        "metrics": str(out / "route_metrics.json"),
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    command = [str(binary), "-exit", "-python", str(source.resolve()),
               str(out / "settings.json")]
    hashed_inputs = [*inputs, *dependencies]
    if route_def != required[0]:
        hashed_inputs.append(route_def)
    input_hashes = {str(path.resolve()): _digest(path) for path in hashed_inputs}
    with (out / "run.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    logged_errors = _logged_errors(out / "run.log")
    effective_returncode = completed.returncode or (1 if logged_errors else 0)
    receipt = {
        "command": command, "tool_returncode": completed.returncode,
        "returncode": effective_returncode, "logged_errors": logged_errors,
        "inputs": input_hashes,
        "def_via_fix": via_fix,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    if logged_errors and not completed.returncode:
        raise RuntimeError(f"OpenROAD logged an error despite zero exit; see {out / 'run.log'}")
    if completed.returncode:
        raise RuntimeError(f"OpenROAD detailed route failed; see {out / 'run.log'}")

    essential = {
        "routed.def": out / "routed.def", "segments.npz": out / "segments.npz",
        "segments.json": out / "segments.json", "drc.rpt": out / "drc.rpt",
        "identity_before.tsv": out / "identity_before.tsv",
        "identity_after.tsv": out / "identity_after.tsv",
        "route_metrics.json": out / "route_metrics.json",
    }
    missing = [name for name, path in essential.items() if not path.is_file()]
    if missing:
        raise RuntimeError("OpenROAD omitted detailed-route artifacts: " + ", ".join(missing))
    before = _digest(essential["identity_before.tsv"])
    after = _digest(essential["identity_after.tsv"])
    if before != after:
        raise RuntimeError("component/net identity or coordinates changed during routing")

    metrics = json.loads(essential["route_metrics.json"].read_text())
    reconciled = metrics.get("decoded_wirelength_dbu", 0) + metrics.get("rect_reconciliation_dbu", 0)
    native = metrics.get("native_wirelength_dbu", 0)
    metrics["reconciled_wirelength_dbu"] = reconciled
    metrics["native_reconciliation_delta_dbu"] = reconciled - native
    metrics["native_reconciliation_relative_error"] = (
        abs(reconciled - native) / native if native else None)
    metrics["identity_unchanged"] = True
    observation = evaluate_route_from_files(
        essential["segments.npz"], required[2], required[3], required[1])
    metrics.update({
        "actual_io": observation.total_route_cross_raw,
        "actual_wirelength_dbu": observation.total_route_wl,
        "net_io": {net: int(value) for net, value in
                   enumerate(observation.route_cross_raw.tolist()) if value},
    })
    receipt["outputs"] = {
        path.name: _digest(path) for path in out.iterdir()
        if path.is_file() and path.name != "receipt.json"
    }
    receipt.update(metrics)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt
