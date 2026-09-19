# A placement DEF can retain obsolete ROUTED records at the original pin
# coordinates. GRT skips nets with dbWire objects, including zero-length ones.
# Begin a fresh signal route while preserving special/POWER/GROUND geometry.
proc ioplace_clear_signal_routing {block report_path} {
    set inst_count [llength [$block getInsts]]
    set net_count [llength [$block getNets]]
    set removed 0
    set report [open $report_path w]
    foreach net [$block getNets] {
        if {[$net isSpecial] || [$net getSigType] in {POWER GROUND}} {
            continue
        }
        set wire [$net getWire]
        if {$wire ne "NULL"} {
            puts $report [$net getName]
            odb::dbWire_destroy $wire
            incr removed
            if {[$net getWire] ne "NULL"} {
                error "failed to remove inherited signal routing"
            }
        }
    }
    close $report
    if {$inst_count != [llength [$block getInsts]] || $net_count != [llength [$block getNets]]} {
        error "routing cleanup changed instance or net count"
    }
    puts "CLEARED_INHERITED_SIGNAL_WIRES $removed"
    return $removed
}
