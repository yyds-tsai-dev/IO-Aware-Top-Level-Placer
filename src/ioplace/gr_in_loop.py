"""The one gate for the retired GR-in-loop paths (design v2 sec 1, "Retired").

`src/scripts/run_route_gp.py` and `ioplace/ops/routing_gp_controller.py` are
not part of the v2 flow and are unmaintained. They stay in the tree (the
round-feedback results reference them) but must be opted into explicitly so
no v2 driver picks them up by accident.

The message below is P-H Task 8's, moved here verbatim from
`RoutingGPController.__init__`: one env var, one message, two call sites (the
controller at construction time, the CLI script at import time).
"""
import os

GR_IN_LOOP_ENV = "IOPLACE_ENABLE_GR_IN_LOOP"


def require_gr_in_loop():
    if os.environ.get(GR_IN_LOOP_ENV) != "1":
        raise RuntimeError(
            "in-loop GR is retired by the v2 design (sec 1): the final GRT "
            "protocol runs once, after placement. Set "
            "IOPLACE_ENABLE_GR_IN_LOOP=1 to use this unmaintained path.")
