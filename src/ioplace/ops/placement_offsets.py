"""Orient measurement pin offsets without mutating the fixed-offset GP model."""
from dataclasses import replace
import numpy as np


def oriented_netlist(nl, orientations):
    """Copy canonical-N movable offsets into actual ordinary-row orientations.

    DREAMPlace's importer keeps movable offsets canonical N, while fixed-pin
    offsets already reflect fixed orientations. General rotations are rejected;
    they require transformed master geometry instead of row-flip formulas.
    """
    values=np.asarray(orientations).astype(str)
    if values.shape!=(nl.num_movable,):
        raise ValueError('one orientation per movable node required')
    allowed=('N','R0','FS','MX','FN','MY','S','R180')
    if not np.isin(values,allowed).all():
        raise ValueError('unsupported movable orientation; transformed pin geometry required')
    flip_x=np.isin(values,('FN','MY','S','R180'))
    flip_y=np.isin(values,('FS','MX','S','R180'))
    x,y=nl.pin_offset_x.copy(),nl.pin_offset_y.copy()
    pins=np.flatnonzero(nl.pin2node<nl.num_movable)
    owners=nl.pin2node[pins]
    selected=pins[flip_x[owners]]
    x[selected]=nl.node_size_x[nl.pin2node[selected]]-x[selected]
    selected=pins[flip_y[owners]]
    y[selected]=nl.node_size_y[nl.pin2node[selected]]-y[selected]
    return replace(nl,pin_offset_x=x,pin_offset_y=y)
