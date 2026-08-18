set DES /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_fft_1
set OUT /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/rehearsal/mgc_fft_1/or_run
read_lef $DES/tech.lef
read_lef $DES/cells.lef
read_def $OUT/routed.def
set block [[[ord::get_db] getChip] getBlock]
puts "COUNT_OUT insts=[llength [$block getInsts]] nets=[llength [$block getNets]]"
set f [open $OUT/net_names_out.txt w]
foreach n [$block getNets] { puts $f [$n getName] }
close $f
report_wire_length -net [get_nets *] -detailed_route -file $OUT/wirelength.rpt
puts "DONE_COUNT_OUT"
exit
