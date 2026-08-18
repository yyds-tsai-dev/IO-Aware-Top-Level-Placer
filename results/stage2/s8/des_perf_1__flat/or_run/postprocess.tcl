# S8 postprocess: V1/V2 invariant counts + wirelength report.
read_lef /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_des_perf_1/tech.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_des_perf_1/cells.lef
read_def /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/des_perf_1__flat/or_run/fixed.def
set block_in [[[ord::get_db] getChip] getBlock]
set n_insts_in [llength [$block_in getInsts]]
set n_nets_in [llength [$block_in getNets]]
read_lef /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_des_perf_1/tech.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_des_perf_1/cells.lef
read_def /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/des_perf_1__flat/or_run/routed.def
set block_out [[[ord::get_db] getChip] getBlock]
set n_insts_out [llength [$block_out getInsts]]
set n_nets_out [llength [$block_out getNets]]
puts "V1_INSTS in=$n_insts_in out=$n_insts_out match=[expr {$n_insts_in == $n_insts_out}]"
puts "V2_NETS in=$n_nets_in out=$n_nets_out match=[expr {$n_nets_in == $n_nets_out}]"
report_wire_length -net [get_nets *] -detailed_route -file /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/des_perf_1__flat/or_run/wirelength.rpt
puts "DONE_POSTPROCESS"
exit
