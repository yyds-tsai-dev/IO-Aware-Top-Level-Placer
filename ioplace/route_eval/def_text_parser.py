"""Stage 2 S2 (`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-
plan.md` sec 7.1 / sec 10 S2 row): a standalone DEF `NETS ... END NETS`
ROUTED/FIXED wire-spec text parser, used *only* as an independent sampling
cross-check against `odb.dbWireDecoder` (`or_scripts/dump_segments.py`).

Sec 7.1 is explicit that `odb.dbWireDecoder` is the primary parser and this
module is not meant to replace it: odb already resolves the wire-tree
branch structure (`JUNCTION`), any non-default routing rule (`RULE`), and
knows each via's real layer pair from the LEF via library, none of which
this text-only parser can see. This module exists to answer a narrower
question -- "does the plain-text `+ ROUTED`/`+ FIXED` grammar decode to the
same segment/via geometry as odb, on the net's whose text I can read by
eye" -- for the handful of nets S2's acceptance check samples.

Grammar covered (LEF/DEF 5.7/5.8 NETS wire spec, the subset this repo's
benchmarks and OpenROAD's DEF writer actually use):

    - netName [ ( compName pinName ) ... ]
        + ROUTED <wireSpec> [+ ROUTED <wireSpec>]... [+ <otherStatement>]...
      | + FIXED <wireSpec> ...
      ;

    wireSpec := layerName [ TAPER | TAPERRULE ruleName ] point
                ( point | RECT ( dx1 dy1 dx2 dy2 ) | VIRTUAL )*
                ( NEW layerName [ TAPER | TAPERRULE ruleName ] point ... )*

    point := ( xval yval [ext] ) [ viaName [ orient ] ]
             xval, yval := integer | "*" (repeat the previous point's
                                           coordinate on that axis)

`SPECIALNETS ... END SPECIALNETS` is recognized (by keyword) and skipped
without attempting to parse its geometry, matching sec 7.2's "SPECIALNETS
(PG)...不算落差" -- PG nets are outside the `placedb` net set this whole
pipeline keys off of, so this parser never needs to understand their
(different, `+ SHAPE`/`+ USE`-flavoured) wire grammar.

`VIRTUAL` is this parser's own token for a virtual/no-metal segment
(the text-level counterpart of `odb.dbWireDecoder.VWIRE`). No routed DEF
seen while building this pipeline (ISPD2015/ISPD2025 corpora, sec 3.2)
contains one, so this token's exact spelling is a best-effort reading of
the LEF/DEF 5.8 reference rather than something cross-validated against a
real router's output -- flagged here so nobody mistakes "the golden test
covers it" for "this was checked against Innovus/OpenROAD's actual output".
"""
from dataclasses import dataclass, field

# --- segment kinds; same vocabulary as ioplace/route_eval/segments.py's
# KIND_* (kept as strings here, not the matching ints, since this module
# has no numpy/npz dependency of its own -- callers doing a cross-check
# against Segments.rows_for_net() already map KIND_NAMES -> these strings).
WIRE = "WIRE"
VIA = "VIA"
RECT = "RECT"
VWIRE = "VWIRE"


@dataclass
class Segment:
    kind: str          # WIRE | VIA | RECT | VWIRE
    layer: str          # the active path layer when this token was seen
    x0: int; y0: int; x1: int; y1: int
    via_name: str = ""  # VIA only


@dataclass
class ParsedNet:
    name: str
    segments: list = field(default_factory=list)


@dataclass
class ParsedDef:
    nets: dict = field(default_factory=dict)          # name -> ParsedNet
    special_net_names: list = field(default_factory=list)  # recognized, not parsed


class DefParseError(ValueError):
    pass


def _tokenize(text):
    # DEF is whitespace/`;`-delimited with parens as their own tokens; a
    # coordinate is always "( x y )" with a space-separated x/y in every
    # benchmark and every writer (DREAMPlace's passthrough writer, OpenROAD,
    # Innovus) this repo has seen, so a straight split() after padding the
    # punctuation is sufficient and keeps this parser trivially auditable.
    text = text.replace("(", " ( ").replace(")", " ) ").replace(";", " ; ")
    return text.split()


class _Cursor:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def peek(self, offset=0):
        j = self.i + offset
        return self.tokens[j] if j < len(self.tokens) else None

    def next(self):
        tok = self.peek()
        if tok is None:
            raise DefParseError("unexpected end of token stream")
        self.i += 1
        return tok

    def at_end(self):
        return self.i >= len(self.tokens)

    def expect(self, tok):
        got = self.next()
        if got != tok:
            raise DefParseError(f"expected {tok!r}, got {got!r} at token {self.i}")


def _find_matching_section(cur, header_kw, end_kw):
    """Advance `cur` to just past `header_kw` (e.g. 'NETS'), returning the
    index just after it; caller is expected to then parse entries until
    `end_kw` (e.g. 'END NETS', passed as ('END', 'NETS')). Returns False if
    `header_kw` isn't found before the token stream runs out.
    """
    while not cur.at_end():
        if cur.peek() == header_kw:
            cur.next()
            return True
        cur.next()
    return False


_NUM_OR_STAR = None  # placeholder for readability; see _parse_point below


def _parse_coord(tok, prev_val):
    if tok == "*":
        return prev_val
    return int(tok)


def _parse_point(cur, prev_pt):
    """Consume "( xval yval [ext] )" and return (x, y) as ints, resolving
    "*" continuation against `prev_pt` (may be None only if neither
    coordinate is "*", i.e. the net's very first point).
    """
    cur.expect("(")
    xtok = cur.next()
    ytok = cur.next()
    # optional extension value: another integer token before ")"
    if cur.peek() != ")":
        cur.next()  # extension value, not needed for segment geometry
    cur.expect(")")
    px, py = prev_pt if prev_pt is not None else (None, None)
    if xtok == "*" and px is None:
        raise DefParseError("'*' continuation with no previous point")
    if ytok == "*" and py is None:
        raise DefParseError("'*' continuation with no previous point")
    return _parse_coord(xtok, px), _parse_coord(ytok, py)


_ORIENTS = {"N", "S", "E", "W", "FN", "FS", "FE", "FW", "R0", "R90", "R180", "R270"}


def _parse_one_branch(cur, segments):
    """Parse a single "<layer> [TAPER[RULE] x] <point> (<point>|RECT|VIRTUAL)*"
    branch, appending Segment rows to `segments`, stopping at the first
    token that starts a new '+ <STATEMENT>', 'NEW', or ';' this function
    doesn't own (returned to the caller un-consumed). Does not itself
    consume a following "NEW <layer> ..." sibling branch -- see
    `_parse_wire_spec`, which drives that iteratively.
    """
    layer = cur.next()
    if cur.peek() in ("TAPER", "TAPERRULE"):
        cur.next()
        if cur.tokens[cur.i - 1] == "TAPERRULE":
            cur.next()  # rule name
    cur_pt = None
    virtual_next = False
    while True:
        tok = cur.peek()
        if tok is None or tok in (";", "+", "NEW"):
            break
        if tok == "(":
            x, y = _parse_point(cur, cur_pt)
            if cur_pt is not None and (x, y) != cur_pt:
                # A point identical to the previous one is a real, if
                # unusual, DEF construct -- real Innovus-produced routed
                # DEFs (sec 3.2's NanGate45 corpus) contain nets whose
                # entire "+ ROUTED" wire is a string of these (pre-CTS
                # pin-realization markers with zero physical length). odb's
                # dbWireDecoder emits no WIRE row for a degenerate
                # zero-length PATH/POINT/POINT triple either (see
                # or_scripts/dump_segments.py's matching `!= cur_pt` guard,
                # and evaluator_ref._walk_segment's own zero-length
                # early-return) -- mirror that here so the two parsers agree
                # net-for-net instead of the text parser inventing phantom
                # zero-length WIRE rows odb never produces.
                kind = VWIRE if virtual_next else WIRE
                segments.append(Segment(kind, layer, cur_pt[0], cur_pt[1], x, y))
            virtual_next = False
            cur_pt = (x, y)
            # optional via name (+ optional orient) right after a point
            nxt = cur.peek()
            if nxt is not None and nxt not in ("(", ";", "+", "NEW", "RECT", "VIRTUAL"):
                via_name = cur.next()
                if cur.peek() in _ORIENTS:
                    cur.next()
                segments.append(Segment(VIA, layer, x, y, x, y, via_name=via_name))
        elif tok == "RECT":
            cur.next()
            cur.expect("(")
            dx1 = int(cur.next()); dy1 = int(cur.next())
            dx2 = int(cur.next()); dy2 = int(cur.next())
            cur.expect(")")
            if cur_pt is None:
                raise DefParseError("RECT with no current point")
            x0, y0 = cur_pt
            segments.append(Segment(RECT, layer, x0 + dx1, y0 + dy1, x0 + dx2, y0 + dy2))
        elif tok == "VIRTUAL":
            cur.next()
            virtual_next = True
        else:
            raise DefParseError(f"unexpected token {tok!r} in wire spec at {cur.i}")
    return segments


def _parse_wire_spec(cur, segments):
    """Parse "<branch> (NEW <branch>)*" -- the full wire spec following a
    "+ ROUTED"/"+ FIXED" keyword, or a net's own initial branch plus any
    number of sibling "NEW <layer> ..." branches hanging off the same
    trunk. Iterative, not recursive: a real net's wire tree (e.g. a
    thousands-of-pins clock net) can have thousands of NEW branches, and
    the previous recursive-per-branch implementation blew Python's default
    recursion limit decoding one on a real routed DEF while building this
    pipeline (see this task's final report / results/stage2/rehearsal).
    """
    _parse_one_branch(cur, segments)
    while cur.peek() == "NEW":
        cur.next()
        _parse_one_branch(cur, segments)
    return segments


def _skip_to(cur, tok):
    while not cur.at_end() and cur.peek() != tok:
        cur.next()


def _parse_net_entry(cur):
    cur.expect("-")
    name = cur.next()
    segments = []
    while True:
        tok = cur.peek()
        if tok is None:
            raise DefParseError(f"net {name!r}: unterminated entry")
        if tok == ";":
            cur.next()
            break
        if tok == "(":
            # a ( compName pinName ) connection pair -- not a coordinate
            # here (those only ever appear inside a "+ ROUTED"/"+ FIXED"
            # wire spec, handled below), so just skip the triple.
            cur.next()
            while cur.peek() != ")":
                cur.next()
            cur.next()
            continue
        if tok == "+":
            cur.next()
            kw = cur.next()
            if kw in ("ROUTED", "FIXED"):
                _parse_wire_spec(cur, segments)
            else:
                # + USE, + WEIGHT, + PROPERTY, ... -- not part of the wire
                # geometry; skip tokens up to the next '+' or ';'.
                while cur.peek() not in (None, "+", ";"):
                    cur.next()
            continue
        # Unrecognized token outside of a '+' clause (defensive: shouldn't
        # happen for well-formed DEF) -- skip it rather than looping forever.
        cur.next()
    return ParsedNet(name=name, segments=segments)


def parse_def_routed(text):
    """Parse the `NETS ... END NETS` block of `text` into a `ParsedDef`.
    `SPECIALNETS ... END SPECIALNETS` is recognized and its net names are
    recorded in `special_net_names`, but its wire geometry is not parsed
    (see module docstring). Sections other than NETS/SPECIALNETS (ROW,
    TRACKS, VIAS, COMPONENTS, PINS, ...) are ignored entirely.
    """
    tokens = _tokenize(text)
    cur = _Cursor(tokens)
    result = ParsedDef()

    while not cur.at_end():
        tok = cur.next()
        if tok == "SPECIALNETS":
            cur.next()  # numNets
            cur.expect(";")
            while not (cur.peek() == "END" and cur.peek(1) == "SPECIALNETS"):
                if cur.at_end():
                    raise DefParseError("unterminated SPECIALNETS section")
                if cur.peek() == "-":
                    cur.next()
                    result.special_net_names.append(cur.next())
                _skip_to(cur, ";")
                if cur.peek() == ";":
                    cur.next()
            cur.next(); cur.next()  # END SPECIALNETS
        elif tok == "NETS":
            cur.next()  # numNets
            cur.expect(";")
            while not (cur.peek() == "END" and cur.peek(1) == "NETS"):
                if cur.at_end():
                    raise DefParseError("unterminated NETS section")
                net = _parse_net_entry(cur)
                result.nets[net.name] = net
            cur.next(); cur.next()  # END NETS
        # everything else: just keep scanning token by token

    return result
