# S4 postprocess: read the routed DEF back (fresh session, no detailed_route
# state) and produce the wirelength report + V1/V2 invariant counts that
# route.tcl itself didn't capture (spec sec 6.1/6.3, sec 10 S4/S2 rows).
set DES /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_fft_1
set OUT /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/rehearsal/mgc_fft_1/or_run
set OUR_DEF /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/rehearsal/mgc_fft_1/after_legalized.ntup.fix.def

read_lef $DES/tech.lef
read_lef $DES/cells.lef
read_def $OUR_DEF
set block_in [[[ord::get_db] getChip] getBlock]
set n_insts_in [llength [$block_in getInsts]]
set n_nets_in [llength [$block_in getNets]]

read_lef $DES/tech.lef
read_lef $DES/cells.lef
read_def $OUT/routed.def
set block_out [[[ord::get_db] getChip] getBlock]
set n_insts_out [llength [$block_out getInsts]]
set n_nets_out [llength [$block_out getNets]]

puts "V1_INSTS in=$n_insts_in out=$n_insts_out match=[expr {$n_insts_in == $n_insts_out}]"
puts "V2_NETS in=$n_nets_in out=$n_nets_out match=[expr {$n_nets_in == $n_nets_out}]"

report_wire_length -net [get_nets *] -detailed_route -file $OUT/wirelength.rpt

puts "DONE_POSTPROCESS"
exit
