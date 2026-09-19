"""Independent OpenDB instance/net/endpoint identity comparison across routing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def compare(before_path,after_path):
    with np.load(before_path,allow_pickle=False) as data:before={key:data[key] for key in data.files}
    with np.load(after_path,allow_pickle=False) as data:after={key:data[key] for key in data.files}
    bm=json.loads(Path(str(before_path)+".json").read_text())
    am=json.loads(Path(str(after_path)+".json").read_text())
    def nodes(data):return {(str(kind),str(name)):i for i,(kind,name) in enumerate(zip(data["node_kind"],data["node_names"]))}
    a,b=nodes(before),nodes(after)
    if len(a)!=len(before["node_names"]) or len(b)!=len(after["node_names"]):raise ValueError("duplicate node identity")
    for data in (before,after):
        if len(set(data["net_names"]))!=len(data["net_names"]):
            raise ValueError("duplicate net identity")
        if len(data["pin_net"])!=len(data["pin_names"]):
            raise ValueError("endpoint arrays have different lengths")
        if np.any(data["pin_net"]<0) or np.any(data["pin_net"]>=len(data["net_names"])):
            raise ValueError("endpoint references an invalid net")
    node_equal=set(a)==set(b)
    net_equal=set(before["net_names"])==set(after["net_names"])
    def endpoints(data):
        return sorted((str(data["net_names"][net]),str(pin)) for net,pin in zip(data["pin_net"],data["pin_names"]))
    endpoint_equal=endpoints(before)==endpoints(after)
    changed=None
    if node_equal:
        keys=sorted(a);left=np.asarray([a[key] for key in keys]);right=np.asarray([b[key] for key in keys])
        moved=np.zeros(len(left),dtype=bool)
        for field in ("node_x","node_y","node_w","node_h","node_orientation"):
            moved |= before[field][left]!=after[field][right]
        changed=int(moved.sum())
    libraries_equal=bm["lef_sha256"]==am["lef_sha256"]
    specials_equal=set(bm["excluded_special_net_names"])==set(am["excluded_special_net_names"])
    result=dict(overall_pass=bool(node_equal and net_equal and endpoint_equal and libraries_equal and specials_equal),
        node_set_equal=node_equal,signal_net_set_equal=net_equal,endpoint_multisets_equal=endpoint_equal,
        lef_hashes_equal=libraries_equal,special_net_sets_equal=specials_equal,
        n_nodes=len(a),n_signal_nets=len(before["net_names"]),n_endpoints=len(before["pin_names"]),
        changed_node_geometry=changed,before_geometry_sha256=sha(before_path),after_geometry_sha256=sha(after_path),
        input_def_sha256=bm["def_sha256"],routed_def_sha256=am["def_sha256"])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",required=True);parser.add_argument("--after",required=True)
    parser.add_argument("--out",required=True);args=parser.parse_args()
    result=compare(args.before,args.after)
    Path(args.out).write_text(json.dumps(result,indent=2)+"\n")
    if not result["overall_pass"]:raise SystemExit("routing changed instance/net/endpoint identity")


if __name__=="__main__":main()
