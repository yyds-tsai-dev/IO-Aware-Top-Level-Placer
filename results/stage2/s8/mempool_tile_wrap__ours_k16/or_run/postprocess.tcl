# S8 postprocess: V1/V2 invariant counts + wirelength report.
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/NangateOpenCellLibrary.tech.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/NangateOpenCellLibrary.macro.mod.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_128x116.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_128x256.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_128x32.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_256x16.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_256x32.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_256x48.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_256x64.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_32x32.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_512x64.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_64x124.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_64x256.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_64x62.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_64x64.lef
read_def /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/mempool_tile_wrap__ours_k16/or_run/fixed.def
set block_in [[[ord::get_db] getChip] getBlock]
set n_insts_in [llength [$block_in getInsts]]
set n_nets_in [llength [$block_in getNets]]
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/NangateOpenCellLibrary.tech.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/NangateOpenCellLibrary.macro.mod.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_128x116.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_128x256.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_128x32.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_256x16.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_256x32.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_256x48.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_256x64.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_32x32.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_512x64.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_64x124.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_64x256.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_64x62.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/benchmarks/ispd25/extracted/ISPD2025_benchmarks/NanGate45/lef/fakeram45_64x64.lef
read_def /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/mempool_tile_wrap__ours_k16/or_run/routed.def
set block_out [[[ord::get_db] getChip] getBlock]
set n_insts_out [llength [$block_out getInsts]]
set n_nets_out [llength [$block_out getNets]]
puts "V1_INSTS in=$n_insts_in out=$n_insts_out match=[expr {$n_insts_in == $n_insts_out}]"
puts "V2_NETS in=$n_nets_in out=$n_nets_out match=[expr {$n_nets_in == $n_nets_out}]"
report_wire_length -net [get_nets *] -detailed_route -file /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/mempool_tile_wrap__ours_k16/or_run/wirelength.rpt
puts "DONE_POSTPROCESS"
exit
