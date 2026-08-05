from dataclasses import dataclass
import numpy as np

@dataclass
class Netlist:
    node_x: np.ndarray; node_y: np.ndarray
    node_size_x: np.ndarray; node_size_y: np.ndarray
    num_movable: int; num_terminals: int; num_terminal_NIs: int
    pin_offset_x: np.ndarray; pin_offset_y: np.ndarray
    pin2node: np.ndarray; pin2net: np.ndarray
    flat_net2pin: np.ndarray; flat_net2pin_start: np.ndarray
    xl: float; yl: float; xh: float; yh: float

    @property
    def num_physical(self):
        return len(self.node_x)

    @property
    def num_nets(self):
        return len(self.flat_net2pin_start) - 1

    @property
    def net_degrees(self):
        return np.diff(self.flat_net2pin_start)

def pin_positions(nl, node_x=None, node_y=None):
    nx = nl.node_x if node_x is None else node_x
    ny = nl.node_y if node_y is None else node_y
    return nx[nl.pin2node] + nl.pin_offset_x, ny[nl.pin2node] + nl.pin_offset_y

def netlist_from_placedb(placedb):
    n_phys = placedb.num_physical_nodes
    return Netlist(
        node_x=np.asarray(placedb.node_x[:n_phys], dtype=np.float64),
        node_y=np.asarray(placedb.node_y[:n_phys], dtype=np.float64),
        node_size_x=np.asarray(placedb.node_size_x[:n_phys], dtype=np.float64),
        node_size_y=np.asarray(placedb.node_size_y[:n_phys], dtype=np.float64),
        num_movable=placedb.num_movable_nodes,
        num_terminals=placedb.num_terminals,
        num_terminal_NIs=placedb.num_terminal_NIs,
        pin_offset_x=np.asarray(placedb.pin_offset_x, dtype=np.float64),
        pin_offset_y=np.asarray(placedb.pin_offset_y, dtype=np.float64),
        pin2node=np.asarray(placedb.pin2node_map, dtype=np.int32),
        pin2net=np.asarray(placedb.pin2net_map, dtype=np.int32),
        flat_net2pin=np.asarray(placedb.flat_net2pin_map, dtype=np.int32),
        flat_net2pin_start=np.asarray(placedb.flat_net2pin_start_map, dtype=np.int32),
        xl=float(placedb.xl), yl=float(placedb.yl),
        xh=float(placedb.xh), yh=float(placedb.yh))

def load_netlist(config_json):
    """Load netlist from DREAMPlace config JSON.

    Note: The config_json's aux_input paths are relative to $DREAMPLACE_ROOT/install.
    This function changes to that directory before loading to resolve relative paths correctly.
    The original working directory is restored after loading.

    Args:
        config_json: Path to the config JSON file

    Returns:
        tuple: (Netlist, placedb, params)
    """
    import os
    from ioplace.dreamplace_env import setup_dreamplace
    setup_dreamplace()
    import Params
    import PlaceDB

    # Save original cwd and change to install directory for relative path resolution
    orig_cwd = os.getcwd()
    root = setup_dreamplace()
    install_dir = os.path.join(root, "install")

    try:
        os.chdir(install_dir)
        params = Params.Params()
        params.load(config_json)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        return netlist_from_placedb(placedb), placedb, params
    finally:
        os.chdir(orig_cwd)
