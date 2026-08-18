# S2 rehearsal: GR+DR mgc_fft_1 to produce a real routed DEF for
# ioplace/route_eval's acceptance check (spec sec 10 S2 row). Follows the
# F-OR skeleton in docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-plan.md
# sec 6.1. -allow_congestion is required (sec 6.1 note: without it GRT-0116
# aborts on this design). Threads capped at 8 per this task's red line.
set DES /nashome/NVL4/vdalab/yyds-dev/DREAMPlace/benchmarks/ispd2015/mgc_fft_1
set OUT /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/rehearsal/mgc_fft_1/or_run
set OUR_DEF /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/results/stage2/rehearsal/mgc_fft_1/after_legalized.ntup.fix.def

read_lef $DES/tech.lef
read_lef $DES/cells.lef
read_def $OUR_DEF

set block [[[ord::get_db] getChip] getBlock]
puts "ASSERT insts=[llength [$block getInsts]] nets=[llength [$block getNets]]"

set_thread_count 8

global_route -allow_congestion \
             -congestion_report_file $OUT/congestion.rpt \
             -guide_file $OUT/route.guide

# -droute_end_iter caps the optimization-iteration count: S2's acceptance
# only needs real routed WIRE geometry to sum/cross-check (spec sec 10 S2
# row), not a DRC-clean route (spec line ~307: DRC count is recorded, not a
# gate). Left unbounded, this design's violation count keeps decreasing very
# slowly past iteration 2 (observed: iter0 191333 -> iter1 70727 -> iter2
# 57709 viol, each iteration taking longer, 2.5-9 min) -- capping at 2
# bounds this rehearsal run to the already-observed ~10 minute window.
detailed_route -output_drc $OUT/drc.rpt -droute_end_iter 2 -verbose 1

write_def $OUT/routed.def
puts "DONE"
exit
