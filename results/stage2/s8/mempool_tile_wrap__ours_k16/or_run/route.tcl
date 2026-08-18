# S8 route: GR+DR mempool_tile_wrap__ours_k16 to produce a routed DEF for crossing extraction.
# Skeleton: results/stage2/rehearsal/mgc_fft_1/or_run/route.tcl / spec sec 6.1.
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
set block [[[ord::get_db] getChip] getBlock]
puts "COUNT_IN insts=[llength [$block getInsts]] nets=[llength [$block getNets]]"
set_thread_count 8
global_route -allow_congestion -congestion_report_file /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/mempool_tile_wrap__ours_k16/or_run/congestion.rpt -guide_file /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/mempool_tile_wrap__ours_k16/or_run/route.guide
detailed_route -droute_end_iter 5 -output_drc /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/mempool_tile_wrap__ours_k16/or_run/drc.rpt -verbose 1
write_def /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/mempool_tile_wrap__ours_k16/or_run/routed.def
puts "DONE_ROUTE"
exit
