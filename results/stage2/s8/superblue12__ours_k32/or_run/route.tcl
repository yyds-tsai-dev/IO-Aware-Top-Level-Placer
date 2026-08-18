# S8 route: GR+DR superblue12__ours_k32 to produce a routed DEF for crossing extraction.
# Skeleton: results/stage2/rehearsal/mgc_fft_1/or_run/route.tcl / spec sec 6.1.
read_lef /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_superblue12/tech.lef
read_lef /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_superblue12/cells.lef
read_def /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/superblue12__ours_k32/or_run/fixed.def
set block [[[ord::get_db] getChip] getBlock]
puts "COUNT_IN insts=[llength [$block getInsts]] nets=[llength [$block getNets]]"
set_thread_count 8
global_route -allow_congestion -congestion_report_file /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/superblue12__ours_k32/or_run/congestion.rpt -guide_file /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/superblue12__ours_k32/or_run/route.guide
detailed_route -droute_end_iter 5 -output_drc /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/superblue12__ours_k32/or_run/drc.rpt -verbose 1
write_def /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/s8/superblue12__ours_k32/or_run/routed.def
puts "DONE_ROUTE"
exit
