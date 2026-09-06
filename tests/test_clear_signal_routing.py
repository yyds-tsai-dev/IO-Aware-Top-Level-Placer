"""Native routing setup: inherited signal geometry must not bypass GRT."""
import os
from pathlib import Path
import subprocess
import pytest


@pytest.mark.skipif(not os.environ.get('OPENROAD_BIN'), reason='source scripts/openroad_env.sh')
def test_clear_stale_signal_wires_preserves_regular_and_special_supplies(tmp_path):
    lef=tmp_path/'tech.lef'
    lef.write_text('''VERSION 5.8 ;
UNITS DATABASE MICRONS 1000 ; END UNITS
LAYER metal1
 TYPE ROUTING ; DIRECTION HORIZONTAL ; PITCH 0.2 ; WIDTH 0.1 ;
END metal1
END LIBRARY
''')
    design=tmp_path/'stale.def'
    design.write_text('''VERSION 5.8 ;
DIVIDERCHAR "/" ; BUSBITCHARS "[]" ;
DESIGN stale ; UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 10000 ) ;
COMPONENTS 0 ; END COMPONENTS
NETS 3 ;
- empty_signal + ROUTED metal1 ( 100 100 ) ( * 100 ) ;
- old_signal + ROUTED metal1 ( 100 200 ) ( 1000 * ) ;
- regular_supply + USE POWER + ROUTED metal1 ( 100 300 ) ( 1000 * ) ;
END NETS
SPECIALNETS 1 ;
- special_supply + USE POWER + ROUTED metal1 100 ( 100 400 ) ( 1000 * ) ;
END SPECIALNETS
END DESIGN
''')
    helper=Path(__file__).resolve().parents[1]/'ioplace/route_eval/or_scripts/clear_signal_routing.tcl'
    ledger=tmp_path/'removed.txt'
    script=tmp_path/'check.tcl'
    script.write_text(f'''read_lef {{{lef}}}
read_def {{{design}}}
source {{{helper}}}
set block [ord::get_db_block]
set special [$block findNet special_supply]
set pg [llength [$special getSWires]]
if {{$pg == 0}} {{ error "missing special-wire fixture" }}
set regular [$block findNet regular_supply]
set supply_wire [$regular getWire]
if {{$supply_wire eq "NULL"}} {{ error "missing regular-supply fixture" }}
set removed [ioplace_clear_signal_routing $block {{{ledger}}}]
if {{$removed != 2}} {{ error "expected both empty and positive signal wires removed" }}
if {{[llength [$block getNets]] != 4}} {{ error "net identities changed" }}
if {{[$regular getWire] ne $supply_wire}} {{ error "regular supply changed" }}
if {{[llength [$special getSWires]] != $pg}} {{ error "special supply changed" }}
foreach name {{empty_signal old_signal}} {{
 if {{[[$block findNet $name] getWire] ne "NULL"}} {{ error "signal routing remains" }}
}}
if {{[ioplace_clear_signal_routing $block {{{tmp_path/'second.txt'}}}] != 0}} {{ error "not idempotent" }}
puts "CLEANUP_NATIVE_PASS"
exit
''')
    result=subprocess.run([os.environ['OPENROAD_BIN'],'-no_init',str(script)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'CLEANUP_NATIVE_PASS' in result.stdout
    assert set(ledger.read_text().splitlines())=={'empty_signal','old_signal'}
