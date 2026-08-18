set DES /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_fft_1
set OUR_DEF /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/rehearsal/mgc_fft_1/after_legalized.ntup.fix.def
read_lef $DES/tech.lef
read_lef $DES/cells.lef
read_def $OUR_DEF
set block [[[ord::get_db] getChip] getBlock]
puts "COUNT_IN insts=[llength [$block getInsts]] nets=[llength [$block getNets]]"
set f [open /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/rehearsal/mgc_fft_1/or_run/net_names_in.txt w]
foreach n [$block getNets] { puts $f [$n getName] }
close $f
exit
