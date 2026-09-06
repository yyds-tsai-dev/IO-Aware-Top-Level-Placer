set_thread_count 1
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/NangateOpenCellLibrary.tech.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/NangateOpenCellLibrary.macro.mod.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_128x116.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_128x256.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_128x32.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_256x16.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_256x32.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_256x48.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_256x64.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_32x32.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_512x64.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_64x124.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_64x256.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_64x62.lef}
read_lef {/ldaphome/yyds-tsai-dev/benchmarks/ispd25/NanGate45/lef/fakeram45_64x64.lef}
read_def {/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer/results/stage2_followup_20260906/mempool_tile_wrap__flat_k16/or_run/fixed.def}
source {/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer/ioplace/route_eval/or_scripts/clear_signal_routing.tcl}
set block [ord::get_db_block]
set n [ioplace_clear_signal_routing $block {/tmp/tile-clear-tcl-preflight/removed.txt}]
if {$n != 4360} {error "wrong inherited set size"}
write_def {/tmp/tile-clear-tcl-preflight/clean.def}
exit
