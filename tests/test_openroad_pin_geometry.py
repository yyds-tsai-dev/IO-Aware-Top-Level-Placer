"""Native OpenDB orientation/union geometry, independently checked analytically."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get("OPENROAD_BIN"),reason="source scripts/openroad_env.sh")
def test_transformed_union_centers_all_orientations_and_long_names(tmp_path):
    lef=tmp_path/"cells.lef"
    lef.write_text('''VERSION 5.8 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;
UNITS DATABASE MICRONS 1000 ; END UNITS
MANUFACTURINGGRID 0.001 ;
LAYER metal1
 TYPE ROUTING ; DIRECTION HORIZONTAL ; PITCH 0.2 ; WIDTH 0.1 ;
END metal1
SITE core
 CLASS CORE ; SIZE 10 BY 20 ; SYMMETRY X Y R90 ;
END core
MACRO cell
 CLASS CORE ; ORIGIN 0 0 ; SIZE 10 BY 20 ; SYMMETRY X Y R90 ; SITE core ;
 PIN A
 DIRECTION INPUT ; USE SIGNAL ;
 PORT LAYER metal1 ; RECT 1 2 3 4 ; RECT 7 8 8 10 ; END
 END A
END cell
END LIBRARY
''')
    orientations=("N","S","E","W","FN","FS","FE","FW")
    long_name="deep/"*60+"cell"
    names=[long_name]+[f"u{i}" for i in range(1,8)]
    components=[f"- {name} cell + PLACED ( {100000*i+100000} 100000 ) {orient} ;"
                for i,(name,orient) in enumerate(zip(names,orientations))]
    pins=[f"- p{i} + NET n{i} + DIRECTION INPUT + USE SIGNAL "
          f"+ LAYER metal1 ( -100 -100 ) ( 100 100 ) + PLACED ( {100000*i+100000} 50000 ) N ;"
          for i in range(8)]
    nets=[f"- n{i} ( {name} A ) ( PIN p{i} ) ;" for i,name in enumerate(names)]
    design=tmp_path/"oriented.def"
    design.write_text('VERSION 5.8 ;\nDIVIDERCHAR "/" ;\nBUSBITCHARS "[]" ;\n'
        'DESIGN oriented ;\nUNITS DISTANCE MICRONS 1000 ;\nDIEAREA ( 0 0 ) ( 1000000 1000000 ) ;\n'
        'COMPONENTS 8 ;\n'+"\n".join(components)+'\nEND COMPONENTS\nPINS 8 ;\n'+"\n".join(pins)
        +'\nEND PINS\nNETS 8 ;\n'+"\n".join(nets)+'\nEND NETS\nEND DESIGN\n')
    output=tmp_path/"pins.npz"
    script=Path(__file__).resolve().parents[1]/"src/ioplace/route_eval/or_scripts/dump_pin_geometry.py"
    run=subprocess.run([os.environ["OPENROAD_BIN"],"-python",str(script),"--lef",str(lef),
                        "--def",str(design),"--out",str(output)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    data=np.load(output,allow_pickle=False)
    metadata=json.loads(Path(str(output)+".json").read_text())
    assert metadata["num_physical"]==16
    assert long_name in data["node_names"]
    w,h,u,v=10000.,20000.,4500.,6000.
    expected=((u,v),(w-u,h-v),(v,w-u),(h-v,u),(w-u,v),(u,h-v),(h-v,w-u),(v,u))
    lookup={tuple(json.loads(str(name))):i for i,name in enumerate(data["pin_names"])}
    for i,(name,offset) in enumerate(zip(names,expected)):
        idx=lookup[("ITerm",name,"A")]
        assert np.array_equal(data["pin_offset"][idx],offset),orientations[i]
        node=data["pin_node"][idx]
        assert (data["node_w"][node],data["node_h"][node]) == ((h,w) if orientations[i] in ("E","W","FE","FW") else (w,h))
        port=lookup[("BTerm",f"p{i}")]
        assert np.array_equal(data["pin_offset"][port],[100.,100.])
    assert data["pin_offset"][lookup[("ITerm",long_name,"A")]][0] != 4750.  # rectangle-center average


@pytest.mark.skipif(not os.environ.get("OPENROAD_BIN"), reason="source scripts/openroad_env.sh")
def test_bterm_multiple_ports_uses_union_bbox_center(tmp_path):
    lef = tmp_path / "bterm.lef"
    lef.write_text('''VERSION 5.8 ;\nUNITS DATABASE MICRONS 1000 ; END UNITS
LAYER metal1 TYPE ROUTING ; DIRECTION HORIZONTAL ; PITCH 0.2 ; WIDTH 0.1 ; END metal1
SITE core CLASS CORE ; SIZE 10 BY 10 ; SYMMETRY X Y R90 ; END core
MACRO cell CLASS CORE ; ORIGIN 0 0 ; SIZE 10 BY 10 ; SITE core ; END cell
END LIBRARY\n''')
    design = tmp_path / "multiport.def"
    design.write_text('''VERSION 5.8 ;\nDIVIDERCHAR "/" ; BUSBITCHARS "[]" ;
DESIGN multiport ; UNITS DISTANCE MICRONS 1000 ; DIEAREA ( 0 0 ) ( 100000 100000 ) ;
COMPONENTS 1 ; - u0 cell + PLACED ( 10000 10000 ) N ; END COMPONENTS
PINS 1 ; - PORT0 + NET n0 + DIRECTION INPUT + USE SIGNAL
+ PORT + LAYER metal1 ( 100 200 ) ( 300 400 ) + FIXED ( 0 0 ) N
+ PORT + LAYER metal1 ( 900 100 ) ( 1500 500 ) + FIXED ( 0 0 ) N
; END PINS
NETS 1 ; - n0 ( PIN PORT0 ) ; END NETS
END DESIGN\n''')
    output = tmp_path / "pins.npz"
    script = Path(__file__).resolve().parents[1] / "src/ioplace/route_eval/or_scripts/dump_pin_geometry.py"
    run = subprocess.run([os.environ["OPENROAD_BIN"], "-python", str(script), "--lef", str(lef),
                         "--def", str(design), "--out", str(output)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    data = np.load(output, allow_pickle=False)
    names = [tuple(json.loads(str(n))) for n in data["pin_names"]]
    idx = names.index(("BTerm", "PORT0"))
    # Union is [100,100]-[1500,500], center (800,300) DBU. The
    # rectangle-center average would be (700,300); the first is (200,300).
    node = data["pin_node"][idx]
    absolute = data["pin_offset"][idx] + [data["node_x"][node], data["node_y"][node]]
    assert np.array_equal(absolute, [800., 300.])
    assert np.array_equal(data["pin_offset"][idx], [700., 200.])
    assert int((data["pin_names"] == data["pin_names"][idx]).sum()) == 1
    assert int((data["pin_node"] == data["pin_node"][idx]).sum()) == 1
