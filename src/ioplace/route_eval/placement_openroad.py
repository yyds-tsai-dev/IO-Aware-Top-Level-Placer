"""Host adapter for bounded OpenROAD placement repair."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def _digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _identity_digest(path):
    """Order-independent multiset digest for streamed identity records."""
    xor = bytearray(32)
    count = 0
    with Path(path).open("rb") as stream:
        for line in stream:
            value = hashlib.sha256(line).digest()
            for index, byte in enumerate(value):
                xor[index] ^= byte
            count += 1
    return count, bytes(xor).hex()


def _logged_errors(path):
    errors, expected = [], False
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            if "IOPLACE_EXPECTED_PRECHECK_BEGIN" in line:
                expected = True
            elif "IOPLACE_EXPECTED_PRECHECK_END" in line:
                expected = False
            elif "[ERROR " in line and not expected:
                errors.append(line.strip())
    return errors


def legalize_export(export_dir, lefs, out, binary, movable_names, threads=16):
    export_dir, out, binary = map(lambda path: Path(path).resolve(),
                                  (export_dir, out, binary))
    lefs = [Path(path).resolve() for path in lefs]
    names = list(movable_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("nonempty movable_names strings required")
    if len(names) != len(set(names)):
        raise ValueError("duplicate movable name")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("positive integer threads required")
    def_path = export_dir / "out.def"
    source = Path(__file__).parent / "or_scripts/legalize_placement.py"
    checked = source.with_name("checked_tcl.py")
    inputs = [def_path, *lefs, binary, Path(__file__), source, checked]
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError("missing placement-repair input")
    if any(any(char in str(path) for char in "{}\n\r") for path in [out, *inputs]):
        raise ValueError("unsupported Tcl path")
    out.mkdir(parents=True, exist_ok=False)
    names_path = out / "movable_names.json"
    names_path.write_text(json.dumps(names))
    preplacement = out / "preplacement.def"
    shutil.copyfile(def_path, preplacement)
    settings = {
        "def": str(def_path), "lefs": list(map(str, lefs)), "threads": threads,
        "movable_names": str(names_path),
        "preplacement_def": str(preplacement),
        "openroad_legalized_def": str(out / "openroad_legalized.def"),
        "legalized_def": str(out / "legalized.def"),
        "coordinates": str(out / "coordinates.npz"),
        "identity_before": str(out / "identity_before.tsv"),
        "identity_after": str(out / "identity_after.tsv"),
        "report": str(out / "placement_report.json"),
    }
    settings_path = out / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))
    input_hashes = {str(path): _digest(path) for path in [*inputs, names_path, settings_path]}
    command = [str(binary), "-exit", "-python", str(source), str(settings_path)]
    verify_command = [str(binary), "-exit", "-python", str(source),
                      "--verify-only", str(settings_path)]
    with (out / "run.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        verified = (subprocess.run(verify_command, stdout=log, stderr=subprocess.STDOUT)
                    if completed.returncode == 0 else None)
    errors = _logged_errors(out / "run.log")
    verify_returncode = verified.returncode if verified is not None else None
    returncode = completed.returncode or verify_returncode or (1 if errors else 0)
    receipt = {"commands": [command, verify_command],
               "tool_returncodes": [completed.returncode, verify_returncode],
               "returncode": returncode, "logged_errors": errors,
               "inputs": input_hashes}
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    if returncode:
        raise RuntimeError(f"OpenROAD placement repair failed; see {out / 'run.log'}")
    outputs = [out / name for name in ("preplacement.def", "legalized.def",
               "coordinates.npz", "identity_before.tsv", "identity_after.tsv",
               "placement_report.json", "openroad_legalized.def")]
    if any(not path.is_file() for path in outputs):
        raise RuntimeError("OpenROAD omitted placement-repair artifacts")
    before, after = _identity_digest(outputs[3]), _identity_digest(outputs[4])
    if before != after:
        raise RuntimeError("fixed placement, BPin geometry, master identity, or connectivity changed")
    with np.load(outputs[2], allow_pickle=False) as data:
        got_names = data["names"].astype(str).tolist()
        x_dbu, y_dbu = data["x_dbu"].copy(), data["y_dbu"].copy()
        orientations = data["orientations"].astype(str).copy()
    if got_names != names:
        raise RuntimeError("movable name order changed")
    if x_dbu.shape != (len(names),) or y_dbu.shape != x_dbu.shape:
        raise RuntimeError("invalid movable coordinate shape")
    if orientations.shape != x_dbu.shape:
        raise RuntimeError("invalid movable orientation shape")
    report = json.loads(outputs[5].read_text())
    receipt.update(report)
    receipt["immutable_identity_unchanged"] = True
    receipt["outputs"] = {path.name: _digest(path) for path in out.iterdir()
                          if path.is_file() and path != receipt_path}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return {"x_dbu": x_dbu, "y_dbu": y_dbu, "orientations": orientations,
            "legal_def_path": str(outputs[1]), "receipt": receipt,
            "receipt_path": str(receipt_path)}
