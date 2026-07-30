# M0+M1 基建與 Evaluator 實作計畫(IO-Aware Top-Level Placer, Phase 1 前段)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可重現的基建(netlist 載入、region 定義、reference evaluator、兩條 baseline 流程),再交付 GPU evaluator 與 net-reweighting 閉環,產出 M0/M1 對照表。

**Architecture:** 我們的 code 全部放在本 repo 的 `ioplace/` Python package,以 import 方式驅動 DREAMPlace(位於 `/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`,Task 1 以新版 Python 重 build,從其 `install/` 執行);對 DREAMPlace 源碼的唯一改動是一個 iteration callback patch(存於本 repo 的 patch 檔)。Evaluator 先做 numpy 參考版(golden),M1 再做 torch GPU 版並以參考版驗證等價。

**Tech Stack:** Python 3.12(uv 管理;fallback 3.11 → 3.9)、PyTorch 2.8.0+cu128、numpy<2、pytest、DREAMPlace(Task 1 重 build)、Mt-KaHyPar(pip 或 source build)。

**對應 spec:** `docs/superpowers/specs/2026-07-30-io-aware-placer-phase1-design.md` 的 M0 與 M1(§9);範圍不含 M2+(可微項)。

## Global Constraints

- DREAMPlace 根目錄:`/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`(以下簡稱 `$DP`);一律從 `$DP/install` 執行/import(source tree 不含編譯出的 `*.so`)。
- 外部 import 需要**兩個** sys.path:`$DP/install` 與 `$DP/install/dreamplace`(DREAMPlace 混用 `import dreamplace.ops.*` 與 bare `import Params` 兩種風格)。
- Python:**3.12 優先**(Task 1 以 uv 裝 3.12 並重 build DREAMPlace,產生新 ABI 的 `*.so`);失敗依序退 3.11 → 修復舊 3.9 venv(現有 install 的 `*.so` 為 cp39 ABI,僅 fallback 時沿用)。torch 2.8.0+cu128;**numpy 必須 <2**(DREAMPlace 源碼用 `np.string_`,numpy 2.0 已移除)。以下命令中 `$PY` = 最終 venv 的 python(預期 `$DP/.venv312/bin/python`)。
- GPU:開發機為 NVIDIA L4 23GB;所有 dtype 主線 float32(config `"dtype": "float32"`)。
- 對 DREAMPlace 源碼的改動:**僅允許** Task 12 的 iteration callback(≤10 行),在 `$DP` 開 git branch `io-aware`,diff 同步存本 repo `ioplace/dp_patch/`。fence region 注入不改 DREAMPlace(在我們 driver 內於 `read()` 與 `initialize()` 之間注入)。
- Region 約束(spec D2/D3):K ≤ 64(FT bitmask 用 uint64);實驗 K ∈ {8,16,32};region = 矩形集合、全 die 分割、邊界對齊 **512×512 lattice**(產生器保證;`RegionGrid` 因此無 aliasing、計數精確)。
- 大 net 截斷:degree > 256 的 net 不建樹,用 region-presence 精確下界(spec §5.2/§6)。
- IO 計數語意(spec D4):tree edge 跨 region boundary 的 crossing 總數;pure FT = per-net「edges 經過但無 pin 的 region」數(net 層級去重)。
- TDD:每個 task 先寫測試;commit 訊息英文、結尾附 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- 單元測試不得依賴大 benchmark(用手工合成 case);integration test 用 `simple`(在 `$DP/install/benchmarks/simple/`)與 `adaptec1`,大 case(bigblue4)只在 report task 跑。
- 執行環境變數:`DREAMPLACE_ROOT`(預設 `/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`)。

## File Structure(本 repo 新增)

```
pyproject.toml                     # 最小 pytest/package 設定
ioplace/__init__.py
ioplace/dreamplace_env.py          # sys.path 設定(兩個路徑)
ioplace/netlist.py                 # Netlist dataclass + PlaceDB 載入 + pin_positions
ioplace/regions.py                 # RegionSpec/RegionSet + grid/slicing 產生器 + 驗證 + JSON
ioplace/region_grid.py             # RegionGrid(lattice rasterize + 點查詢)
ioplace/evaluator_ref.py           # 參考版 evaluator(numpy;golden)
ioplace/evaluator_gpu.py           # GPU 版 evaluator(torch)         [M1]
ioplace/fence_inject.py            # PlaceDB fence region 注入(two-stage baseline 用)
ioplace/reweight.py                # net-reweighting 權重公式          [M1]
ioplace/partition/__init__.py
ioplace/partition/hgr.py           # Netlist → hMETIS .hgr;讀 partition 檔
ioplace/partition/mtkahypar_runner.py
ioplace/drivers/run_placement.py   # 統一 driver:--mode {flat,two_stage,reweight}
ioplace/drivers/make_report.py     # 彙整 result JSON → markdown 對照表
ioplace/dp_patch/iteration-callback.patch          [M1]
tests/conftest.py                  # 手工合成 case fixtures
tests/test_netlist.py
tests/test_regions.py
tests/test_region_grid.py
tests/test_evaluator_ref.py
tests/test_evaluator_gpu.py        [M1]
tests/test_hgr.py
tests/test_fence_inject.py
tests/test_reweight.py             [M1]
configs/regions/                   # 產生的 region JSON
results/                           # 實驗輸出(gitignore 大檔,保留 JSON/md)
docs/dev-env.md                    # Task 1 產出:環境如何啟動
docs/results/m0-baseline-report.md # Task 10 產出
docs/results/m1-reweight-report.md # Task 13 產出
```

---

### Task 1: 以 Python 3.12 重建 DREAMPlace 環境 + repo 骨架

**Files:**
- Create: `pyproject.toml`, `ioplace/__init__.py`, `tests/__init__.py`, `docs/dev-env.md`, `.gitignore`
- (外部)`$DP/.venv312`(新 venv)、`$DP/build312`(新 build 目錄)、重灌 `$DP/install`(先備份)

**Interfaces:**
- Produces: 可用 venv `$DP/.venv312/bin/python`(**Python 3.12** + torch 2.8.0 cu128 + numpy<2 + pytest)與重 build 的 `$DP/install`(新 ABI 的 `*.so`);`docs/dev-env.md` 記載啟動方式。後續所有命令的 `$PY` 即此 interpreter。

**背景:** 舊 `.venv` 為 cp39 且 python 執行檔遺失;使用者要求不用 3.9。Python 版本綁定只存在於編譯出的 `*.so`(pybind/torch extension),對新直譯器重 build 即可;C++ parser 與 CUDA kernel 與 Python 版本無關。已知風險(upstream 僅宣稱支援 ≤3.9):(a) numpy 2.x 移除 `np.string_`(PlaceDB.py 有用)→ 必 pin `numpy<2`(1.26.x 支援 3.12);(b) Python 3.12 移除 distutils,build glue 若引用會炸 → 退 3.11;(c) 舊 pin 依賴(torch_optimizer==0.3.0 等,純 Python,預期可裝)。有利事實:此 checkout 已在 torch 2.8.0(遠新於官方宣稱的 ≥1.6)上跑通,堆疊現代化有前例。

- [ ] **Step 1: 準備 uv 與 Python 3.12,建 venv 並裝依賴**

```bash
DP=/nashome/NVL4/vdalab/yyds-dev/DREAMPlace
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh   # 無 uv 才裝
uv python install 3.12
cd $DP && uv venv --python 3.12 .venv312
PY=$DP/.venv312/bin/python
uv pip install --python $PY torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY "numpy<2" scipy matplotlib shapely cairocffi pyunpack patool pkgconfig setuptools "torch_optimizer==0.3.0" "ncg_optimizer==0.2.2" pytest
$PY -c "import torch, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
```
Expected: `2.8.0+cu128 True 1.26.x`。CUDA 不可用則停下回報(驅動問題,不要繼續)。

- [ ] **Step 2: 重 build DREAMPlace(先備份舊 install)**

```bash
cd $DP && cp -r install install.cp39.bak
cmake -B build312 -DPython_EXECUTABLE=$PY -DCMAKE_INSTALL_PREFIX=$DP/install -DCMAKE_BUILD_TYPE=Release
cmake --build build312 -j $(nproc)
cmake --build build312 --target install
```
Expected: configure 偵測到 CUDA 與 torch 2.8;build 約 10–40 分鐘。失敗處置(按錯誤類型):
1. 錯誤含 `distutils` → 3.12 特有:改用 3.11(`uv python install 3.11 && cd $DP && uv venv --python 3.11 .venv311`,重跑 Step 1–2,`$PY` 改指 `.venv311/bin/python`)。
2. 個別小型 API 相容錯(含 runtime 的 `np.string_` 類)→ 在 `$DP` 開(或沿用)branch `io-aware` 做最小修補並 commit,修補內容記入 `docs/dev-env.md`。
3. 多處大範圍失敗 → 停下回報;fallback = 修復舊 3.9 venv(`.venv` 的 site-packages 完整仍在:依序試 `pyvenv.cfg` 的 `home` 路徑重建 symlink → `uv venv --python 3.9 --allow-existing .venv` → 系統 python3.9),並還原 `install.cp39.bak` 為 `install`。

- [ ] **Step 3: Smoke run**

```bash
cd $DP/install && $PY dreamplace/Placer.py test/simple.json
```
Expected: 數秒完成,`install/results/simple/simple.gp.pl` 更新。runtime 才爆的相容性錯誤 → 回 Step 2 處置 2 修補後重跑 `cmake --build build312 --target install`。

- [ ] **Step 4: 建 repo 骨架**

`pyproject.toml`:
```toml
[project]
name = "ioplace"
version = "0.0.1"
requires-python = ">=3.9"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: integration tests that run real placements"]
```

`.gitignore`:
```
__pycache__/
*.pyc
results/**/*.pl
results/**/*.def
results/**/*.log
```

`ioplace/__init__.py` 與 `tests/__init__.py` 空檔。`docs/dev-env.md` 記錄:最終 Python 版本與 build 方式、interpreter 絕對路徑、smoke run 指令、`DREAMPLACE_ROOT` 用法、相容性修補清單。

- [ ] **Step 5: 驗證 pytest 可跑**

```bash
cd /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer
$PY -m pytest --collect-only -q
```
Expected: `no tests ran`(collect 成功、無錯誤)。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml ioplace/__init__.py tests/__init__.py docs/dev-env.md .gitignore
git commit -m "chore: repo skeleton and dev environment doc (Python 3.12 rebuild)"
```

---

### Task 2: Netlist 載入(dreamplace_env + netlist.py)

**Files:**
- Create: `ioplace/dreamplace_env.py`, `ioplace/netlist.py`
- Test: `tests/test_netlist.py`

**Interfaces:**
- Consumes: Task 1 的 venv。
- Produces:
  - `setup_dreamplace(root: str | None = None) -> str` — 把 `$root/install` 與 `$root/install/dreamplace` 插入 `sys.path`(root 預設取 env `DREAMPLACE_ROOT`,再預設 `/nashome/NVL4/vdalab/yyds-dev/DREAMPlace`);回傳 root。冪等。
  - `@dataclass Netlist` 欄位(numpy):`node_x, node_y, node_size_x, node_size_y`(float, len=num_physical)、`num_movable, num_terminals, num_terminal_NIs`(int)、`pin_offset_x, pin_offset_y`(float)、`pin2node, pin2net`(int32)、`flat_net2pin, flat_net2pin_start`(int32;start 長度 #nets+1)、`xl, yl, xh, yh`(float)。property:`num_physical`、`num_nets`、`net_degrees`(np.diff(start))。
  - `netlist_from_placedb(placedb) -> Netlist`(座標取 placedb 當前值 = scale 後座標系)。
  - `load_netlist(config_json: str) -> tuple[Netlist, placedb, params]` — 完整 `PlaceDB()(params)` 流程後轉出。
  - `pin_positions(nl: Netlist, node_x=None, node_y=None) -> tuple[np.ndarray, np.ndarray]` — `px = node_x[nl.pin2node] + nl.pin_offset_x`(node_x 預設 nl.node_x;傳入者長度須 ≥ num_physical,只取用 pin2node 索引到的部分)。

- [ ] **Step 1: Write the failing test**

`tests/test_netlist.py`:
```python
import numpy as np
import pytest
from ioplace.netlist import Netlist, pin_positions

def make_tiny_netlist():
    # 3 cells + 1 fixed, 2 nets: n0={c0,c1}, n1={c1,c2,f3}
    return Netlist(
        node_x=np.array([10., 30., 50., 90.]), node_y=np.array([10., 10., 40., 90.]),
        node_size_x=np.ones(4), node_size_y=np.ones(4),
        num_movable=3, num_terminals=1, num_terminal_NIs=0,
        pin_offset_x=np.zeros(5), pin_offset_y=np.zeros(5),
        pin2node=np.array([0, 1, 1, 2, 3], dtype=np.int32),
        pin2net=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        flat_net2pin=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        flat_net2pin_start=np.array([0, 2, 5], dtype=np.int32),
        xl=0., yl=0., xh=100., yh=100.)

def test_netlist_properties():
    nl = make_tiny_netlist()
    assert nl.num_physical == 4 and nl.num_nets == 2
    assert list(nl.net_degrees) == [2, 3]

def test_pin_positions_default_and_override():
    nl = make_tiny_netlist()
    px, py = pin_positions(nl)
    assert px[0] == 10. and px[2] == 30.
    px2, _ = pin_positions(nl, node_x=nl.node_x + 5., node_y=nl.node_y)
    assert px2[0] == 15.

@pytest.mark.slow
def test_load_netlist_simple():
    from ioplace.netlist import load_netlist
    import os
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    nl, placedb, params = load_netlist(os.path.join(root, "install/test/simple.json"))
    assert nl.num_movable > 0 and nl.num_nets > 0
    assert len(nl.flat_net2pin_start) == nl.num_nets + 1
    assert nl.net_degrees.min() >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
$PY -m pytest tests/test_netlist.py -v -m "not slow"
```
Expected: FAIL(`ModuleNotFoundError: ioplace.netlist`)。

- [ ] **Step 3: Write implementation**

`ioplace/dreamplace_env.py`:
```python
import os, sys

DEFAULT_ROOT = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"

def setup_dreamplace(root=None):
    root = root or os.environ.get("DREAMPLACE_ROOT", DEFAULT_ROOT)
    for p in (os.path.join(root, "install"), os.path.join(root, "install", "dreamplace")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
```

`ioplace/netlist.py`:
```python
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
    from ioplace.dreamplace_env import setup_dreamplace
    setup_dreamplace()
    import Params
    import PlaceDB
    params = Params.Params()
    params.load(config_json)
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    return netlist_from_placedb(placedb), placedb, params
```

- [ ] **Step 4: Run tests**

```bash
$PY -m pytest tests/test_netlist.py -v -m "not slow"
cd /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer && DREAMPLACE_ROOT=$DP $PY -m pytest tests/test_netlist.py -v -m slow
```
Expected: 前者 2 passed;後者 1 passed(slow test 會實際載入 simple case;注意 `load_netlist` 內 `params.load` 的相對路徑 —— simple.json 的 `aux_input` 是相對於 install 的路徑,若載入失敗,在 `load_netlist` 前 `os.chdir` 到 `$DP/install` 並於 test 中還原 cwd,把這個行為寫進 `load_netlist` docstring)。

- [ ] **Step 5: Commit**

```bash
git add ioplace/dreamplace_env.py ioplace/netlist.py tests/test_netlist.py
git commit -m "feat: netlist loading via DREAMPlace PlaceDB"
```

---

### Task 3: Region 定義與產生器(regions.py)

**Files:**
- Create: `ioplace/regions.py`
- Test: `tests/test_regions.py`

**Interfaces:**
- Consumes: 無(獨立)。
- Produces:
  - `@dataclass RegionSpec`:`name: str`、`rects: np.ndarray`(shape `(R,4)`,`[xl,yl,xh,yh]`,float64)。
  - `@dataclass RegionSet`:`die: tuple[float,float,float,float]`、`lattice: int`、`regions: list[RegionSpec]`。方法:`validate()`(邊界對齊 lattice、region 互不重疊、聯集覆蓋 die,violation 時 raise ValueError)、`to_json(path)`、`classmethod from_json(path)`、property `k`。
  - `make_grid_regions(die, nx, ny, lattice=512) -> RegionSet`(nx×ny 均勻格;K=nx*ny)。
  - `make_slicing_regions(die, k, seed=0, lattice=512, min_frac=0.1) -> RegionSet`(隨機 slicing tree:每次把面積最大的 region 沿長邊切一刀,切點在 [min_frac, 1-min_frac] 均勻取樣後 round 到 lattice;直到 k 個;每個 region 是單一矩形 —— rectilinear 的多矩形 region 由 grid 模式 + 後續手動合併支援,M0 實驗只需 grid 與 slicing)。

- [ ] **Step 1: Write the failing test**

`tests/test_regions.py`:
```python
import numpy as np
import pytest
from ioplace.regions import RegionSet, RegionSpec, make_grid_regions, make_slicing_regions

DIE = (0., 0., 1024., 1024.)

def test_grid_regions_cover_and_count():
    rs = make_grid_regions(DIE, 4, 2, lattice=512)
    assert rs.k == 8
    rs.validate()
    total = sum(((r.rects[:, 2]-r.rects[:, 0])*(r.rects[:, 3]-r.rects[:, 1])).sum()
                for r in rs.regions)
    assert total == pytest.approx(1024.*1024.)

def test_slicing_regions_valid_and_seeded():
    rs = make_slicing_regions(DIE, 16, seed=7)
    assert rs.k == 16
    rs.validate()
    rs2 = make_slicing_regions(DIE, 16, seed=7)
    assert all(np.array_equal(a.rects, b.rects) for a, b in zip(rs.regions, rs2.regions))

def test_validate_rejects_overlap():
    bad = RegionSet(die=DIE, lattice=512, regions=[
        RegionSpec("P0", np.array([[0., 0., 600., 1024.]])),
        RegionSpec("P1", np.array([[500., 0., 1024., 1024.]]))])
    with pytest.raises(ValueError):
        bad.validate()

def test_json_roundtrip(tmp_path):
    rs = make_grid_regions(DIE, 2, 2)
    p = tmp_path / "r.json"
    rs.to_json(str(p))
    rs2 = RegionSet.from_json(str(p))
    assert rs2.k == 4 and rs2.lattice == rs.lattice
    assert np.array_equal(rs2.regions[0].rects, rs.regions[0].rects)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
$PY -m pytest tests/test_regions.py -v
```
Expected: FAIL(module not found)。

- [ ] **Step 3: Write implementation**

`ioplace/regions.py`:
```python
from dataclasses import dataclass, field
import json
import numpy as np

@dataclass
class RegionSpec:
    name: str
    rects: np.ndarray  # (R,4) [xl,yl,xh,yh]

@dataclass
class RegionSet:
    die: tuple
    lattice: int
    regions: list

    @property
    def k(self):
        return len(self.regions)

    def _cell_wh(self):
        xl, yl, xh, yh = self.die
        return (xh - xl) / self.lattice, (yh - yl) / self.lattice

    def validate(self):
        xl, yl, xh, yh = self.die
        cw, ch = self._cell_wh()
        cover = np.zeros((self.lattice, self.lattice), dtype=np.int32)
        for r in self.regions:
            for (rxl, ryl, rxh, ryh) in r.rects:
                for v, lo, step in ((rxl, xl, cw), (rxh, xl, cw), (ryl, yl, ch), (ryh, yl, ch)):
                    idx = (v - lo) / step
                    if abs(idx - round(idx)) > 1e-6:
                        raise ValueError(f"region {r.name}: coord {v} not on lattice")
                ix0 = int(round((rxl - xl) / cw)); ix1 = int(round((rxh - xl) / cw))
                iy0 = int(round((ryl - yl) / ch)); iy1 = int(round((ryh - yl) / ch))
                cover[iy0:iy1, ix0:ix1] += 1
        if (cover > 1).any():
            raise ValueError("regions overlap")
        if (cover == 0).any():
            raise ValueError("regions do not cover die")

    def to_json(self, path):
        obj = {"die": list(self.die), "lattice": self.lattice,
               "regions": [{"name": r.name, "rects": np.asarray(r.rects).tolist()}
                           for r in self.regions]}
        with open(path, "w") as f:
            json.dump(obj, f, indent=1)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            obj = json.load(f)
        regs = [RegionSpec(d["name"], np.array(d["rects"], dtype=np.float64))
                for d in obj["regions"]]
        return cls(die=tuple(obj["die"]), lattice=int(obj["lattice"]), regions=regs)

def _snap(v, lo, step):
    return lo + round((v - lo) / step) * step

def make_grid_regions(die, nx, ny, lattice=512):
    xl, yl, xh, yh = die
    cw, ch = (xh - xl) / lattice, (yh - yl) / lattice
    xs = [_snap(xl + (xh - xl) * i / nx, xl, cw) for i in range(nx + 1)]
    ys = [_snap(yl + (yh - yl) * j / ny, yl, ch) for j in range(ny + 1)]
    regs = []
    for j in range(ny):
        for i in range(nx):
            regs.append(RegionSpec(f"P{j*nx+i}",
                np.array([[xs[i], ys[j], xs[i+1], ys[j+1]]], dtype=np.float64)))
    return RegionSet(die=die, lattice=lattice, regions=regs)

def make_slicing_regions(die, k, seed=0, lattice=512, min_frac=0.1):
    xl, yl, xh, yh = die
    cw, ch = (xh - xl) / lattice, (yh - yl) / lattice
    rng = np.random.default_rng(seed)
    boxes = [np.array([xl, yl, xh, yh], dtype=np.float64)]
    while len(boxes) < k:
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        b = boxes.pop(int(np.argmax(areas)))
        w, h = b[2]-b[0], b[3]-b[1]
        frac = rng.uniform(min_frac, 1.0 - min_frac)
        if w >= h:
            cut = _snap(b[0] + w * frac, xl, cw)
            if cut <= b[0] or cut >= b[2]:
                cut = _snap(b[0] + w * 0.5, xl, cw)
            boxes += [np.array([b[0], b[1], cut, b[3]]), np.array([cut, b[1], b[2], b[3]])]
        else:
            cut = _snap(b[1] + h * frac, yl, ch)
            if cut <= b[1] or cut >= b[3]:
                cut = _snap(b[1] + h * 0.5, yl, ch)
            boxes += [np.array([b[0], b[1], b[2], cut]), np.array([b[0], cut, b[2], b[3]])]
    regs = [RegionSpec(f"P{i}", b.reshape(1, 4)) for i, b in enumerate(boxes)]
    return RegionSet(die=die, lattice=lattice, regions=regs)
```

- [ ] **Step 4: Run tests**

```bash
$PY -m pytest tests/test_regions.py -v
```
Expected: 4 passed。

- [ ] **Step 5: Commit**

```bash
git add ioplace/regions.py tests/test_regions.py
git commit -m "feat: region schema, grid/slicing generators, validation"
```

---

### Task 4: Region-id grid(region_grid.py)

**Files:**
- Create: `ioplace/region_grid.py`
- Test: `tests/test_region_grid.py`

**Interfaces:**
- Consumes: `RegionSet`(Task 3)。
- Produces:
  - `class RegionGrid`:`__init__(self, rs: RegionSet)`;屬性 `grid: np.ndarray`(shape `(lattice, lattice)`,int16,值 = region id,row-major:`grid[iy, ix]`)、`die`、`nx = ny = rs.lattice`、`cell_w`、`cell_h`、`k`。
  - `region_of_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray(int16)` — 座標夾到 die 內(`np.clip`),映射到 lattice cell 後查表。
  - `pin_region_bitmask(self, nl: Netlist, node_x, node_y) -> np.ndarray(uint64, len=num_nets)` — 每 net 其 pins 所在 region 的 OR bitmask(K ≤ 64 斷言)。

- [ ] **Step 1: Write the failing test**

`tests/test_region_grid.py`:
```python
import numpy as np
import pytest
from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from tests.test_netlist import make_tiny_netlist

DIE = (0., 0., 100., 100.)

def test_grid_ids_match_geometry():
    rs = make_grid_regions(DIE, 2, 2, lattice=10)   # P0=左下 P1=右下 P2=左上 P3=右上
    rg = RegionGrid(rs)
    assert rg.grid.shape == (10, 10)
    ids = rg.region_of_points(np.array([10., 90., 10., 90.]),
                              np.array([10., 10., 90., 90.]))
    assert list(ids) == [0, 1, 2, 3]

def test_points_on_die_edge_clip():
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    rg = RegionGrid(rs)
    ids = rg.region_of_points(np.array([0., 100.]), np.array([0., 100.]))
    assert list(ids) == [0, 3]

def test_pin_region_bitmask():
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    rg = RegionGrid(rs)
    nl = make_tiny_netlist()
    # cells at (10,10)=P0,(30,10)=P0,(50,40)=P1(x=50 → 右半),(90,90)=P3
    bm = rg.pin_region_bitmask(nl, nl.node_x, nl.node_y)
    assert bm[0] == np.uint64(0b0001)            # n0: P0,P0
    assert bm[1] == np.uint64(0b1011)            # n1: P0,P1,P3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
$PY -m pytest tests/test_region_grid.py -v
```
Expected: FAIL(module not found)。

- [ ] **Step 3: Write implementation**

`ioplace/region_grid.py`:
```python
import numpy as np
from ioplace.netlist import pin_positions

class RegionGrid:
    def __init__(self, rs):
        rs.validate()
        self.die = rs.die
        self.k = rs.k
        n = rs.lattice
        self.nx = self.ny = n
        xl, yl, xh, yh = rs.die
        self.cell_w = (xh - xl) / n
        self.cell_h = (yh - yl) / n
        self.grid = np.full((n, n), -1, dtype=np.int16)
        for rid, r in enumerate(rs.regions):
            for (rxl, ryl, rxh, ryh) in r.rects:
                ix0 = int(round((rxl - xl) / self.cell_w))
                ix1 = int(round((rxh - xl) / self.cell_w))
                iy0 = int(round((ryl - yl) / self.cell_h))
                iy1 = int(round((ryh - yl) / self.cell_h))
                self.grid[iy0:iy1, ix0:ix1] = rid
        assert (self.grid >= 0).all()

    def _to_idx(self, x, y):
        xl, yl, xh, yh = self.die
        ix = np.clip(((x - xl) / self.cell_w).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(((y - yl) / self.cell_h).astype(np.int64), 0, self.ny - 1)
        return ix, iy

    def region_of_points(self, x, y):
        ix, iy = self._to_idx(np.asarray(x, dtype=np.float64),
                              np.asarray(y, dtype=np.float64))
        return self.grid[iy, ix]

    def pin_region_bitmask(self, nl, node_x, node_y):
        assert self.k <= 64
        px, py = pin_positions(nl, node_x, node_y)
        pin_rid = self.region_of_points(px, py).astype(np.uint64)
        bm = np.zeros(nl.num_nets, dtype=np.uint64)
        np.bitwise_or.at(bm, nl.pin2net, np.left_shift(np.uint64(1), pin_rid))
        return bm
```

- [ ] **Step 4: Run tests**

```bash
$PY -m pytest tests/test_region_grid.py -v
```
Expected: 3 passed。

- [ ] **Step 5: Commit**

```bash
git add ioplace/region_grid.py tests/test_region_grid.py
git commit -m "feat: region-id lattice grid with point query and net pin bitmask"
```

---

### Task 5: Reference evaluator — per-net MST

**Files:**
- Create: `ioplace/evaluator_ref.py`(本 task 只寫 MST 部分)
- Test: `tests/test_evaluator_ref.py`(本 task 的 MST 測試)

**Interfaces:**
- Consumes: `Netlist`、`pin_positions`。
- Produces:
  - `net_mst_edges(px: np.ndarray, py: np.ndarray) -> np.ndarray` — 單一 net 的 pin 座標(len d ≥ 1),回傳 MST 邊 `(d-1, 2)` int32(pin 索引對,d≤1 回空 `(0,2)`);Manhattan 距離;d=2 直連;d≥3 用 O(d²) Prim。
  - `net_mst_length(px, py) -> float` — 邊長總和(後續 task 用)。

- [ ] **Step 1: Write the failing test**

加入 `tests/test_evaluator_ref.py`:
```python
import numpy as np
from ioplace.evaluator_ref import net_mst_edges, net_mst_length

def _edge_set(edges):
    return {tuple(sorted(e)) for e in edges.tolist()}

def test_mst_two_pins():
    e = net_mst_edges(np.array([0., 10.]), np.array([0., 0.]))
    assert _edge_set(e) == {(0, 1)}

def test_mst_three_pins_line():
    # (0,0),(5,0),(20,0): MST = {0-1, 1-2},非 {0-2}
    e = net_mst_edges(np.array([0., 5., 20.]), np.array([0., 0., 0.]))
    assert _edge_set(e) == {(0, 1), (1, 2)}
    assert net_mst_length(np.array([0., 5., 20.]), np.array([0., 0., 0.])) == 20.

def test_mst_length_matches_bruteforce_random():
    rng = np.random.default_rng(3)
    for _ in range(20):
        d = rng.integers(2, 8)
        px, py = rng.uniform(0, 100, d), rng.uniform(0, 100, d)
        got = net_mst_length(px, py)
        # brute force: Prim 的另一實作 —— 逐步取最短跨集合邊
        import itertools
        dist = np.abs(px[:, None]-px[None, :]) + np.abs(py[:, None]-py[None, :])
        in_tree = {0}; total = 0.
        while len(in_tree) < d:
            cands = [(dist[i, j], j) for i in in_tree for j in range(d) if j not in in_tree]
            w, j = min(cands)
            total += w; in_tree.add(j)
        assert got == pytest.approx(total)

import pytest
```

- [ ] **Step 2: Run test to verify it fails**

```bash
$PY -m pytest tests/test_evaluator_ref.py -v
```
Expected: FAIL(module not found)。

- [ ] **Step 3: Write implementation**

`ioplace/evaluator_ref.py`(第一部分):
```python
import numpy as np

def net_mst_edges(px, py):
    d = len(px)
    if d <= 1:
        return np.empty((0, 2), dtype=np.int32)
    if d == 2:
        return np.array([[0, 1]], dtype=np.int32)
    dist = np.abs(px[:, None] - px[None, :]) + np.abs(py[:, None] - py[None, :])
    in_tree = np.zeros(d, dtype=bool)
    in_tree[0] = True
    best_cost = dist[0].copy()
    best_from = np.zeros(d, dtype=np.int32)
    edges = np.empty((d - 1, 2), dtype=np.int32)
    for t in range(d - 1):
        masked = np.where(in_tree, np.inf, best_cost)
        j = int(np.argmin(masked))
        edges[t] = (best_from[j], j)
        in_tree[j] = True
        upd = dist[j] < best_cost
        best_cost = np.where(upd, dist[j], best_cost)
        best_from = np.where(upd, j, best_from)
    return edges

def net_mst_length(px, py):
    e = net_mst_edges(px, py)
    if len(e) == 0:
        return 0.0
    return float(np.sum(np.abs(px[e[:, 0]] - px[e[:, 1]]) +
                        np.abs(py[e[:, 0]] - py[e[:, 1]])))
```

- [ ] **Step 4: Run tests**

```bash
$PY -m pytest tests/test_evaluator_ref.py -v
```
Expected: 3 passed。

- [ ] **Step 5: Commit**

```bash
git add ioplace/evaluator_ref.py tests/test_evaluator_ref.py
git commit -m "feat: per-net Manhattan MST (reference implementation)"
```

---

### Task 6: Reference evaluator — crossing / FT / 彙整

**Files:**
- Modify: `ioplace/evaluator_ref.py`(加入 crossing 與 `evaluate`)
- Test: `tests/test_evaluator_ref.py`(追加)

**Interfaces:**
- Consumes: Task 4 `RegionGrid`、Task 5 `net_mst_edges`。
- Produces:
  - `@dataclass EvalResult`:`io_count: int`、`ft_count: int`、`tree_wl: float`、`hpwl: float`、`per_net_crossings: np.ndarray(int32)`、`per_net_ft: np.ndarray(int32)`、`boundary_pair_demand: dict[tuple[int,int], int]`、`large_net_lb: int`(degree>max_degree nets 的 presence 下界總和,已含在 `io_count`)。
  - `edge_regions_and_crossings(rg: RegionGrid, x0, y0, x1, y1) -> tuple[set[int], int, list[tuple[int,int]]]` — 單條邊以 L-shape(先水平後垂直)沿 lattice 走:回傳(經過的 region id 集合、crossing 數、每次 crossing 的 (rid_a, rid_b) 有序化 tuple 清單)。
  - `evaluate(nl: Netlist, node_x, node_y, rg: RegionGrid, max_degree: int = 256) -> EvalResult`。語意:IO = Σ per-edge crossing(大 net 用 presence 下界);FT(per net)= |edges 經過的 region 聯集 −(該 net pin 所在 region 集合)|;`hpwl` = Σ per-net (span_x + span_y)。

- [ ] **Step 1: Write the failing test**

追加到 `tests/test_evaluator_ref.py`:
```python
from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import evaluate, edge_regions_and_crossings
from tests.test_netlist import make_tiny_netlist

DIE = (0., 0., 100., 100.)

def _rg22():
    return RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))

def test_edge_crossing_simple():
    rg = _rg22()
    regions, ncross, pairs = edge_regions_and_crossings(rg, 10., 10., 90., 10.)
    assert regions == {0, 1} and ncross == 1 and pairs == [(0, 1)]

def test_edge_crossing_L_through_third_region():
    rg = _rg22()
    # (10,10)→(90,90) L: 水平段經 P0→P1,垂直段經 P1→P3 → 2 crossings,經過 {0,1,3}
    regions, ncross, pairs = edge_regions_and_crossings(rg, 10., 10., 90., 90.)
    assert regions == {0, 1, 3} and ncross == 2
    assert pairs == [(0, 1), (1, 3)]

def test_evaluate_tiny_netlist():
    rg = _rg22()
    nl = make_tiny_netlist()
    # cells: c0(10,10)P0 c1(30,10)P0 c2(50,40)P1 f3(90,90)P3
    # n0={c0,c1}: 全在 P0 → 0 crossing, 0 FT
    # n1={c1,c2,f3}: MST edges = (c1,c2),(c2,f3)
    #   (30,10)-(50,40): L → (50,10) 經 P0→P1 = 1 crossing;再 (50,10)-(50,40) 留 P1
    #   (50,40)-(90,90): L → (90,40) 留 P1;再 (90,40)-(90,90) P1→P3 = 1 crossing
    #   → n1 crossings=2;經過 {0,1,3} 全有 pin → FT=0
    res = evaluate(nl, nl.node_x, nl.node_y, rg)
    assert res.io_count == 2
    assert list(res.per_net_crossings) == [0, 2]
    assert res.ft_count == 0
    assert res.boundary_pair_demand == {(0, 1): 1, (1, 3): 1}

def test_evaluate_pure_feedthrough():
    rg = _rg22()
    nl = make_tiny_netlist()
    # 搬位:c0(10,10)P0, c1(90,90)P3, c2(50,40)P1, f3(90,90)P3
    # n0={c0,c1}: L (10,10)→(90,10)→(90,90) 經 {P0,P1,P3},pins 只在 {P0,P3}
    #   → P1 是 pure feed-through;crossings = 2
    # n1={c1,c2,f3}: c1 與 f3 同點(邊長 0);(50,40)-(90,90) 的 L 經 {P1,P3} 皆有 pin
    #   → FT=0,crossing=1
    node_x = np.array([10., 90., 50., 90.]); node_y = np.array([10., 90., 40., 90.])
    res = evaluate(nl, node_x, node_y, rg)
    assert list(res.per_net_ft) == [1, 0]
    assert res.ft_count == 1
    assert list(res.per_net_crossings) == [2, 1]
    assert res.io_count == 3

def test_evaluate_bruteforce_random():
    rng = np.random.default_rng(11)
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    rg = RegionGrid(rs)
    for _ in range(10):
        n_cells, n_nets = 12, 6
        nx_, ny_ = rng.uniform(1, 99, n_cells), rng.uniform(1, 99, n_cells)
        pins, p2n = [], []
        for net in range(n_nets):
            d = int(rng.integers(2, 5))
            pins += list(rng.integers(0, n_cells, d)); p2n += [net]*d
        pins, p2n = np.array(pins, np.int32), np.array(p2n, np.int32)
        order = np.argsort(p2n, kind="stable")
        flat = pins[order]
        start = np.searchsorted(p2n[order], np.arange(n_nets + 1))
        from ioplace.netlist import Netlist
        nl = Netlist(node_x=nx_, node_y=ny_, node_size_x=np.ones(n_cells),
                     node_size_y=np.ones(n_cells), num_movable=n_cells,
                     num_terminals=0, num_terminal_NIs=0,
                     pin_offset_x=np.zeros(len(flat)), pin_offset_y=np.zeros(len(flat)),
                     pin2node=flat, pin2net=p2n[order].astype(np.int32),
                     flat_net2pin=np.arange(len(flat), dtype=np.int32),
                     flat_net2pin_start=start.astype(np.int32),
                     xl=0., yl=0., xh=100., yh=100.)
        res = evaluate(nl, nl.node_x, nl.node_y, rg)
        # brute force:逐 lattice 小步 walk 每條 MST 邊,數 id 變化
        from ioplace.evaluator_ref import net_mst_edges
        from ioplace.netlist import pin_positions
        px, py = pin_positions(nl)
        io_bf = 0
        for net in range(n_nets):
            s, e = start[net], start[net + 1]
            for (a, b) in net_mst_edges(px[s:e], py[s:e]):
                pa, pb = (px[s+a], py[s+a]), (px[s+b], py[s+b])
                for seg in (((pa[0], pa[1]), (pb[0], pa[1])), ((pb[0], pa[1]), (pb[0], pb[1]))):
                    (x0, y0), (x1, y1) = seg
                    steps = 400
                    xs = np.linspace(x0, x1, steps); ys = np.linspace(y0, y1, steps)
                    ids = rg.region_of_points(xs, ys)
                    io_bf += int(np.count_nonzero(np.diff(ids)))
        assert res.io_count == io_bf
```

- [ ] **Step 2: Run test to verify it fails**

```bash
$PY -m pytest tests/test_evaluator_ref.py -v
```
Expected: 新增測試 FAIL(`ImportError: evaluate`);Task 5 的測試仍 PASS。

- [ ] **Step 3: Write implementation**

追加到 `ioplace/evaluator_ref.py`:
```python
from dataclasses import dataclass, field
from ioplace.netlist import pin_positions

@dataclass
class EvalResult:
    io_count: int
    ft_count: int
    tree_wl: float
    hpwl: float
    per_net_crossings: np.ndarray
    per_net_ft: np.ndarray
    boundary_pair_demand: dict
    large_net_lb: int

def _walk_segment(rg, x0, y0, x1, y1, regions, pairs):
    """沿軸對齊線段在 lattice 上走,累計 region 集合與 crossing;回傳 crossing 數。"""
    ids0 = rg.region_of_points(np.array([x0]), np.array([y0]))[0]
    if x0 == x1 and y0 == y1:
        regions.add(int(ids0))
        return 0
    if y0 == y1:  # 水平
        ix0, iy = rg._to_idx(np.array([min(x0, x1)]), np.array([y0]))
        ix1, _ = rg._to_idx(np.array([max(x0, x1)]), np.array([y0]))
        row = rg.grid[iy[0], ix0[0]:ix1[0] + 1]
    else:         # 垂直
        ix, iy0 = rg._to_idx(np.array([x0]), np.array([min(y0, y1)]))
        _, iy1 = rg._to_idx(np.array([x0]), np.array([max(y0, y1)]))
        row = rg.grid[iy0[0]:iy1[0] + 1, ix[0]]
    regions.update(np.unique(row).tolist())
    diff_pos = np.nonzero(np.diff(row))[0]
    for p in diff_pos:
        a, b = int(row[p]), int(row[p + 1])
        pairs.append((min(a, b), max(a, b)))
    return len(diff_pos)

def edge_regions_and_crossings(rg, x0, y0, x1, y1):
    regions, pairs = set(), []
    n1 = _walk_segment(rg, x0, y0, x1, y0, regions, pairs)   # 水平段
    n2 = _walk_segment(rg, x1, y0, x1, y1, regions, pairs)   # 垂直段
    return regions, n1 + n2, pairs

def evaluate(nl, node_x, node_y, rg, max_degree=256):
    px, py = pin_positions(nl, node_x, node_y)
    start = nl.flat_net2pin_start
    n_nets = nl.num_nets
    per_net_crossings = np.zeros(n_nets, dtype=np.int32)
    per_net_ft = np.zeros(n_nets, dtype=np.int32)
    pair_demand = {}
    tree_wl = 0.0
    hpwl = 0.0
    large_lb = 0
    pin_rid_all = rg.region_of_points(px, py)
    for net in range(n_nets):
        s, e = start[net], start[net + 1]
        d = e - s
        if d <= 1:
            continue
        pin_idx = nl.flat_net2pin[s:e]
        nx_, ny_ = px[pin_idx], py[pin_idx]
        hpwl += (nx_.max() - nx_.min()) + (ny_.max() - ny_.min())
        pin_regions = set(pin_rid_all[pin_idx].tolist())
        if d > max_degree:
            lb = len(pin_regions) - 1
            per_net_crossings[net] = lb
            large_lb += lb
            continue
        edges = net_mst_edges(nx_, ny_)
        passed = set()
        ncross = 0
        for (a, b) in edges:
            regs, nc, pairs = edge_regions_and_crossings(
                rg, nx_[a], ny_[a], nx_[b], ny_[b])
            passed |= regs
            ncross += nc
            for pr in pairs:
                pair_demand[pr] = pair_demand.get(pr, 0) + 1
            tree_wl += abs(nx_[a] - nx_[b]) + abs(ny_[a] - ny_[b])
        per_net_crossings[net] = ncross
        per_net_ft[net] = len(passed - pin_regions)
    return EvalResult(
        io_count=int(per_net_crossings.sum()),
        ft_count=int(per_net_ft.sum()),
        tree_wl=float(tree_wl), hpwl=float(hpwl),
        per_net_crossings=per_net_crossings, per_net_ft=per_net_ft,
        boundary_pair_demand=pair_demand, large_net_lb=int(large_lb))
```

- [ ] **Step 4: Run tests**

```bash
$PY -m pytest tests/test_evaluator_ref.py -v
```
Expected: 全部 passed(含 brute-force 隨機 10 回合)。

- [ ] **Step 5: Commit**

```bash
git add ioplace/evaluator_ref.py tests/test_evaluator_ref.py
git commit -m "feat: reference evaluator with crossing count, pure feedthrough, boundary demand"
```

---

### Task 7: 統一 placement driver(flat 模式)

**Files:**
- Create: `ioplace/drivers/__init__.py`, `ioplace/drivers/run_placement.py`
- Test: `tests/test_driver.py`

**Interfaces:**
- Consumes: Tasks 2–6 全部。
- Produces:
  - `get_regions_for(die, k, rtype, seed, lattice=512) -> RegionSet` — rtype ∈ {"grid","slicing"};grid 時 k 必須 ∈ {4:2×2, 8:4×2, 16:4×4, 32:8×4} 的映射。**產生器只依 (die 比例, k, rtype, seed) 決定相對幾何**(座標系縮放下等比),這是 Task 9 座標系策略的基礎。
  - `run_flat(config_json, k, rtype, seed, out_json) -> dict` — 跑 DREAMPlace GP+LG(DP 關閉),回傳並寫出 result dict:`{mode, config, k, rtype, seed, runtime_s, peak_mem_mb, io_count, ft_count, tree_wl, hpwl, large_net_lb}`;座標存 `<out_json>.npz`(`node_x`,`node_y`)。
  - `extract_final_positions(placer, placedb) -> tuple[np.ndarray, np.ndarray]` — 取最終 (node_x, node_y)(len = num_physical,placedb 內部座標系)。
  - CLI:`python -m ioplace.drivers.run_placement --config C --mode flat --k 16 --rtype grid --seed 0 --out results/x.json`。

- [ ] **Step 1: 確認最終座標的取回機制(investigation,先做)**

讀 `$DP/install/dreamplace/NonLinearPlace.py` 的 `__call__` 尾端(legalization 之後)與 `$DP/install/dreamplace/PlaceDB.py` 的 `write`/`apply` 方法:確認 place 完成後最終解位於何處(候選 A:`placedb.node_x/node_y` 已被回寫;候選 B:需從 `placer.pos[0].data` 取,佈局為 `[x_0..x_{num_nodes-1}, y_0..y_{num_nodes-1}]`,取前 `num_physical` 段)。驗證方法:跑 simple case 後,比對兩個候選來源與 `install/results/simple/simple.gp.pl` 的前三個 node 座標(注意 write 可能 unscale)。把確認結果寫成 `extract_final_positions` 的實作與 docstring。

- [ ] **Step 2: Write the failing test**

`tests/test_driver.py`:
```python
import json, os
import numpy as np
import pytest
from ioplace.drivers.run_placement import get_regions_for

def test_get_regions_grid_k16():
    rs = get_regions_for((0., 0., 1000., 1000.), 16, "grid", 0)
    assert rs.k == 16
    rs.validate()

def test_get_regions_scale_equivariant():
    rs1 = get_regions_for((0., 0., 1000., 1000.), 8, "slicing", 5)
    rs2 = get_regions_for((0., 0., 2000., 2000.), 8, "slicing", 5)
    for a, b in zip(rs1.regions, rs2.regions):
        assert np.allclose(np.asarray(a.rects) * 2.0, np.asarray(b.rects))

@pytest.mark.slow
def test_run_flat_simple(tmp_path):
    from ioplace.drivers.run_placement import run_flat
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    out = str(tmp_path / "simple_flat.json")
    res = run_flat(os.path.join(root, "install/test/simple.json"), 4, "grid", 0, out)
    assert res["io_count"] >= 0 and res["hpwl"] > 0
    assert os.path.exists(out) and os.path.exists(out + ".npz")
    saved = json.load(open(out))
    assert saved["mode"] == "flat"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
$PY -m pytest tests/test_driver.py -v -m "not slow"
```
Expected: FAIL(module not found)。

- [ ] **Step 4: Write implementation**

`ioplace/drivers/run_placement.py`:
```python
import argparse, json, os, time
import numpy as np
from ioplace.dreamplace_env import setup_dreamplace
from ioplace.regions import make_grid_regions, make_slicing_regions
from ioplace.region_grid import RegionGrid
from ioplace.netlist import netlist_from_placedb
from ioplace.evaluator_ref import evaluate

GRID_SHAPES = {4: (2, 2), 8: (4, 2), 16: (4, 4), 32: (8, 4)}

def get_regions_for(die, k, rtype, seed, lattice=512):
    if rtype == "grid":
        nx, ny = GRID_SHAPES[k]
        return make_grid_regions(die, nx, ny, lattice=lattice)
    if rtype == "slicing":
        return make_slicing_regions(die, k, seed=seed, lattice=lattice)
    raise ValueError(rtype)

def _load_dreamplace(config_json):
    root = setup_dreamplace()
    import Params, PlaceDB
    params = Params.Params()
    cwd = os.getcwd()
    os.chdir(os.path.join(root, "install"))   # config 內是相對路徑
    try:
        params.load(config_json)
        params.detailed_place_engine = ""      # M0/M1: GP+LG only(spec §8 protocol)
        params.plot_flag = 0
        placedb = PlaceDB.PlaceDB()
        placedb.read(params)
        return params, placedb
    finally:
        os.chdir(cwd)

def _place(params, placedb):
    import NonLinearPlace
    lr = params.global_place_stages[0]["learning_rate"]
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    metrics = placer(params, placedb, lr)
    return placer, metrics

def extract_final_positions(placer, placedb):
    # 依 Task 7 Step 1 的確認結果實作;以下為候選 B 的形式(如候選 A 成立,改讀 placedb.node_x/node_y)
    pos = placer.pos[0].data.cpu().numpy()
    n_all = placedb.num_nodes
    n_phys = placedb.num_physical_nodes
    return pos[:n_phys].astype(np.float64), pos[n_all:n_all + n_phys].astype(np.float64)

def _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed):
    nl = netlist_from_placedb(placedb)
    nl.node_x, nl.node_y = node_x, node_y
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, seed))
    res = evaluate(nl, node_x, node_y, rg)
    return rg, {"io_count": res.io_count, "ft_count": res.ft_count,
                "tree_wl": res.tree_wl, "hpwl": res.hpwl,
                "large_net_lb": res.large_net_lb}

def run_flat(config_json, k, rtype, seed, out_json):
    import torch
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    placedb.initialize(params)
    placer, _ = _place(params, placedb)
    node_x, node_y = extract_final_positions(placer, placedb)
    _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)
    result = {"mode": "flat", "config": config_json, "k": k, "rtype": rtype,
              "seed": seed, "runtime_s": time.time() - t0,
              "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20
              if torch.cuda.is_available() else 0.0, **metrics}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=["flat", "two_stage", "reweight"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--rtype", default="grid", choices=["grid", "slicing"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reweight-every", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    if args.mode == "flat":
        run_flat(args.config, args.k, args.rtype, args.seed, args.out)
    elif args.mode == "two_stage":
        from ioplace.drivers.run_placement_two_stage import run_two_stage  # Task 9
        run_two_stage(args.config, args.k, args.rtype, args.seed, args.out)
    else:
        from ioplace.drivers.run_placement_reweight import run_reweight    # Task 12
        run_reweight(args.config, args.k, args.rtype, args.seed, args.out,
                     every=args.reweight_every, alpha=args.alpha)

if __name__ == "__main__":
    main()
```
(注:`nl.node_x, nl.node_y = node_x, node_y` 之後仍把 node_x/node_y 顯式傳入 `evaluate`,雙保險;`params.detailed_place_engine=""` 若該欄位名與 install 版不符,以 `$DP/install/test/ispd2005/adaptec1.json` 內實際欄位名為準修正。)

- [ ] **Step 5: Run tests + integration**

```bash
$PY -m pytest tests/test_driver.py -v -m "not slow"
DREAMPLACE_ROOT=$DP $PY -m pytest tests/test_driver.py -v -m slow
```
Expected: 2 passed;slow 1 passed(simple 全流程數十秒內)。

- [ ] **Step 6: Commit**

```bash
git add ioplace/drivers/__init__.py ioplace/drivers/run_placement.py tests/test_driver.py
git commit -m "feat: unified placement driver with flat baseline mode"
```

---

### Task 8: Mt-KaHyPar 整合(hgr + runner)

**Files:**
- Create: `ioplace/partition/__init__.py`, `ioplace/partition/hgr.py`, `ioplace/partition/mtkahypar_runner.py`
- Test: `tests/test_hgr.py`

**Interfaces:**
- Consumes: `Netlist`(Task 2)。
- Produces:
  - `write_hgr(nl: Netlist, path: str) -> int` — 輸出 hMETIS 格式(首行 `#nets #nodes`;每 net 一行 1-indexed unique node ids;全部 physical nodes 入圖);回傳寫出的 net 數。
  - `partition_netlist(nl: Netlist, k: int, epsilon: float = 0.03, seed: int = 0, threads: int = 8) -> np.ndarray(int32, len=num_physical)` — 呼叫 Mt-KaHyPar(objective km1),回傳每 node 的 part id。

- [ ] **Step 1: 安裝 Mt-KaHyPar 並確認 Python API 形態**

```bash
uv pip install --python $PY mtkahypar 2>/dev/null || \
  $PY -m pip install mtkahypar
$PY -c "import mtkahypar; print(mtkahypar.__version__ if hasattr(mtkahypar,'__version__') else 'ok')"
```
成功後,依安裝版本的官方 README(github.com/kahypar/mt-kahypar 的 Python interface 章節)確認 API 精確簽名 —— 預期形態(以此為底,依實際版本修正):
```python
import mtkahypar
mtk = mtkahypar.initialize(threads)
ctx = mtk.context_from_preset(mtkahypar.PresetType.DEFAULT)
ctx.set_partitioning_parameters(k, epsilon, mtkahypar.Objective.KM1)
hg = mtk.create_hypergraph(ctx, num_nodes, num_nets, hyperedge_indices, hyperedges)
part = hg.partition(ctx)
parts = np.array([part.block_id(v) for v in range(num_nodes)])
```
若 pip 安裝失敗(無 py3.9 wheel):fallback 為 source build CLI(`git clone --depth 1 https://github.com/kahypar/mt-kahypar && cmake -B build --preset default && cmake --build build --target mtkahypar -j16`),runner 改走 subprocess(`-h graph.hgr -k K -e 0.03 -o km1 --write-partition-file=true`),binary 路徑取 env `MTKAHYPAR_BIN`。兩條路都把最終採用方式記入 `docs/dev-env.md`。

- [ ] **Step 2: Write the failing test**

`tests/test_hgr.py`:
```python
import numpy as np
import pytest
from ioplace.partition.hgr import write_hgr
from tests.test_netlist import make_tiny_netlist

def test_write_hgr(tmp_path):
    nl = make_tiny_netlist()
    p = tmp_path / "t.hgr"
    n = write_hgr(nl, str(p))
    lines = p.read_text().strip().splitlines()
    assert lines[0] == "2 4"          # 2 nets, 4 nodes
    assert lines[1] == "1 2"          # n0 = {c0,c1} → 1-indexed
    assert lines[2] == "2 3 4"        # n1 = {c1,c2,f3}
    assert n == 2

def test_partition_two_clusters():
    mtk = pytest.importorskip("mtkahypar")
    from ioplace.partition.mtkahypar_runner import partition_netlist
    from ioplace.netlist import Netlist
    # 兩個 4-clique 群 + 1 條橋 net → k=2 應該把兩群分開
    p2n, pins = [], []
    nets = [[0,1],[1,2],[2,3],[0,3],[4,5],[5,6],[6,7],[4,7],[3,4]]
    for i, ns in enumerate(nets):
        pins += ns; p2n += [i]*len(ns)
    start = np.searchsorted(np.array(p2n), np.arange(len(nets)+1))
    nl = Netlist(node_x=np.zeros(8), node_y=np.zeros(8),
                 node_size_x=np.ones(8), node_size_y=np.ones(8),
                 num_movable=8, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                 pin2node=np.array(pins, np.int32), pin2net=np.array(p2n, np.int32),
                 flat_net2pin=np.array(pins, np.int32),
                 flat_net2pin_start=start.astype(np.int32),
                 xl=0., yl=0., xh=10., yh=10.)
    parts = partition_netlist(nl, 2, seed=1)
    assert len(parts) == 8 and set(parts) == {0, 1}
    assert len(set(parts[:4])) == 1 and len(set(parts[4:])) == 1
```
(注:`flat_net2pin` 此處直接等於 pins 序列,因為 pins 已按 net 排序。)

- [ ] **Step 3: Run test to verify it fails**

```bash
$PY -m pytest tests/test_hgr.py -v
```
Expected: FAIL(module not found)。

- [ ] **Step 4: Write implementation**

`ioplace/partition/hgr.py`:
```python
import numpy as np

def write_hgr(nl, path):
    lines = []
    n_written = 0
    for net in range(nl.num_nets):
        s, e = nl.flat_net2pin_start[net], nl.flat_net2pin_start[net + 1]
        nodes = np.unique(nl.pin2node[nl.flat_net2pin[s:e]])
        lines.append(" ".join(str(v + 1) for v in nodes))
        n_written += 1
    with open(path, "w") as f:
        f.write(f"{n_written} {nl.num_physical}\n")
        f.write("\n".join(lines) + "\n")
    return n_written
```

`ioplace/partition/mtkahypar_runner.py`(python-API 後端;依 Step 1 確認的簽名微調):
```python
import numpy as np

def _flat_hyperedges(nl):
    idx = [0]
    edges = []
    for net in range(nl.num_nets):
        s, e = nl.flat_net2pin_start[net], nl.flat_net2pin_start[net + 1]
        nodes = np.unique(nl.pin2node[nl.flat_net2pin[s:e]])
        edges.extend(int(v) for v in nodes)
        idx.append(len(edges))
    return idx, edges

def partition_netlist(nl, k, epsilon=0.03, seed=0, threads=8):
    import mtkahypar
    mtk = mtkahypar.initialize(threads)
    ctx = mtk.context_from_preset(mtkahypar.PresetType.DEFAULT)
    ctx.set_partitioning_parameters(k, epsilon, mtkahypar.Objective.KM1)
    ctx.set_seed(seed)
    idx, edges = _flat_hyperedges(nl)
    hg = mtk.create_hypergraph(ctx, nl.num_physical, nl.num_nets, idx, edges)
    part = hg.partition(ctx)
    return np.array([part.block_id(v) for v in range(nl.num_physical)],
                    dtype=np.int32)
```

- [ ] **Step 5: Run tests**

```bash
$PY -m pytest tests/test_hgr.py -v
```
Expected: 2 passed(若 mtkahypar API 與預期形態不符,以實際 README 修正 runner 後重跑至 pass;修正內容記入 commit message)。

- [ ] **Step 6: Commit**

```bash
git add ioplace/partition/ tests/test_hgr.py docs/dev-env.md
git commit -m "feat: hMETIS export and Mt-KaHyPar partition runner"
```

---

### Task 9: Fence 注入與 two-stage baseline

**Files:**
- Create: `ioplace/fence_inject.py`, `ioplace/drivers/run_placement_two_stage.py`
- Test: `tests/test_fence_inject.py`

**Interfaces:**
- Consumes: Tasks 3/4/7/8。
- Produces:
  - `inject_fence_regions(placedb, rs: RegionSet, parts: np.ndarray) -> None` — **必須在 `placedb.read(params)` 之後、`placedb.initialize(params)` 之前呼叫**。填四個欄位(偵察確認):`placedb.regions`(list of `(R,4)` np.ndarray, dtype=placedb.dtype)、`placedb.flat_region_boxes`(2D `(total,4)`)、`placedb.flat_region_boxes_start`(int32, cumsum, 首項 0)、`placedb.node2fence_region_map`(int32, len = num_movable + num_terminals;movable 取 `parts`,terminals 取其座標的幾何歸屬)。
  - `run_two_stage(config_json, k, rtype, seed, out_json) -> dict` — Mt-KaHyPar 切 → 注入 fence → DREAMPlace multi-electrostatics fence placement → evaluate;result dict 同 Task 7 格式(`mode="two_stage"`,另加 `initial_cut_io_lb`: Σ_e(λ_e−1) 以 assignment 計)。
- **座標系策略**:region 於 `read()` 後以「原始 die」產生並注入;`initialize()` 可能 shift/scale 座標(placedb.regions 會被一併 scale)。評測用 RegionGrid 於 place 完成後以「當時 die」+ 相同 (k, rtype, seed) 重新產生 —— Task 7 的 `test_get_regions_scale_equivariant` 保證兩者幾何等比,語意為同一組 region。

- [ ] **Step 1: Write the failing test**

`tests/test_fence_inject.py`:
```python
import numpy as np
import pytest
from types import SimpleNamespace
from ioplace.fence_inject import inject_fence_regions
from ioplace.regions import make_grid_regions

def _fake_placedb():
    return SimpleNamespace(
        dtype=np.float32,
        num_movable_nodes=3, num_terminals=1, num_terminal_NIs=0,
        node_x=np.array([10., 30., 50., 90.]), node_y=np.array([10., 10., 40., 90.]),
        xl=0., yl=0., xh=100., yh=100.)

def test_inject_fills_four_fields():
    db = _fake_placedb()
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    parts = np.array([0, 0, 1], dtype=np.int32)
    inject_fence_regions(db, rs, parts)
    assert len(db.regions) == 4
    assert db.flat_region_boxes.shape == (4, 4) and db.flat_region_boxes.ndim == 2
    assert db.flat_region_boxes.dtype == np.float32
    assert list(db.flat_region_boxes_start) == [0, 1, 2, 3, 4]
    assert db.node2fence_region_map.dtype == np.int32
    assert list(db.node2fence_region_map[:3]) == [0, 0, 1]
    assert db.node2fence_region_map[3] == 3     # terminal (90,90) → P3(幾何歸屬)

def test_inject_rejects_wrong_parts_length():
    db = _fake_placedb()
    rs = make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10)
    with pytest.raises(AssertionError):
        inject_fence_regions(db, rs, np.array([0, 1], dtype=np.int32))

@pytest.mark.slow
def test_two_stage_simple(tmp_path):
    import os
    from ioplace.drivers.run_placement_two_stage import run_two_stage
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    res = run_two_stage(os.path.join(root, "install/test/simple.json"),
                        4, "grid", 0, str(tmp_path / "s2.json"))
    assert res["mode"] == "two_stage" and res["io_count"] >= 0
    # fence placement 後,movable cells 幾何落點應與指派 region 一致(LG 後)
    assert res["fence_compliance"] >= 0.9
```

- [ ] **Step 2: Run test to verify it fails**

```bash
$PY -m pytest tests/test_fence_inject.py -v -m "not slow"
```
Expected: FAIL(module not found)。

- [ ] **Step 3: Write implementation**

`ioplace/fence_inject.py`:
```python
import numpy as np
from ioplace.region_grid import RegionGrid

INT32_MAX = np.iinfo(np.int32).max

def inject_fence_regions(placedb, rs, parts):
    assert not hasattr(placedb, "filler_start_map"), \
        "inject must happen after read() and BEFORE initialize()"
    n_assign = placedb.num_movable_nodes + placedb.num_terminals
    assert len(parts) == placedb.num_movable_nodes
    regions = [np.asarray(r.rects, dtype=placedb.dtype).reshape(-1, 4)
               for r in rs.regions]
    placedb.regions = regions
    placedb.flat_region_boxes = np.concatenate(regions, axis=0)
    counts = [len(r) for r in regions]
    placedb.flat_region_boxes_start = np.concatenate(
        [[0], np.cumsum(counts)]).astype(np.int32)
    rg = RegionGrid(rs)
    m = placedb.num_movable_nodes
    fence_map = np.full(n_assign, INT32_MAX, dtype=np.int32)
    fence_map[:m] = parts.astype(np.int32)
    term_sl = slice(m, n_assign)
    fence_map[m:] = rg.region_of_points(
        np.asarray(placedb.node_x[term_sl], dtype=np.float64),
        np.asarray(placedb.node_y[term_sl], dtype=np.float64)).astype(np.int32)
    placedb.node2fence_region_map = fence_map
```

`ioplace/drivers/run_placement_two_stage.py`:
```python
import json, os, time
import numpy as np
from ioplace.drivers.run_placement import (_load_dreamplace, _place,
    extract_final_positions, _evaluate_and_pack, get_regions_for)
from ioplace.fence_inject import inject_fence_regions
from ioplace.netlist import netlist_from_placedb
from ioplace.partition.mtkahypar_runner import partition_netlist
from ioplace.region_grid import RegionGrid

def run_two_stage(config_json, k, rtype, seed, out_json):
    import torch
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    assert params.enable_fillers == 1, "fence mode requires enable_fillers"
    nl0 = netlist_from_placedb(placedb)          # read 後的原始座標系
    parts = partition_netlist(nl0, k, seed=seed)[:placedb.num_movable_nodes]
    die0 = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rs0 = get_regions_for(die0, k, rtype, seed)
    inject_fence_regions(placedb, rs0, parts)
    # assignment 固有 IO 下界:Σ_e (touched parts − 1)
    node_part = np.concatenate([parts, placedb.node2fence_region_map[
        placedb.num_movable_nodes:]]).astype(np.int64)
    lam = 0
    for net in range(nl0.num_nets):
        s, e = nl0.flat_net2pin_start[net], nl0.flat_net2pin_start[net + 1]
        lam += len(np.unique(node_part[nl0.pin2node[nl0.flat_net2pin[s:e]]])) - 1
    placedb.initialize(params)
    placer, _ = _place(params, placedb)
    node_x, node_y = extract_final_positions(placer, placedb)
    _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)
    # fence compliance:LG 後 movable cells 落點 region == 指派 region 的比例
    die1 = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg1 = RegionGrid(get_regions_for(die1, k, rtype, seed))
    m = placedb.num_movable_nodes
    landed = rg1.region_of_points(node_x[:m], node_y[:m])
    compliance = float((landed == parts).mean())
    result = {"mode": "two_stage", "config": config_json, "k": k, "rtype": rtype,
              "seed": seed, "runtime_s": time.time() - t0,
              "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20
              if torch.cuda.is_available() else 0.0,
              "initial_cut_io_lb": int(lam), "fence_compliance": compliance,
              **metrics}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
    return result
```

- [ ] **Step 4: Run tests + integration**

```bash
$PY -m pytest tests/test_fence_inject.py -v -m "not slow"
DREAMPLACE_ROOT=$DP $PY -m pytest tests/test_fence_inject.py -v -m slow
```
Expected: 2 passed;slow 1 passed。若 multi-electrostatics 對 simple 這種小 case 數值不穩(density weight 向量化路徑),改用 `install/test/ispd2005/adaptec1.json` 作 integration case 並標記更長 timeout。

- [ ] **Step 5: Commit**

```bash
git add ioplace/fence_inject.py ioplace/drivers/run_placement_two_stage.py tests/test_fence_inject.py
git commit -m "feat: fence region injection and two-stage (partition-then-place) baseline"
```

---

### Task 10: M0 對照表(exit)

**Files:**
- Create: `ioplace/drivers/make_report.py`, `docs/results/m0-baseline-report.md`(產出)
- Test: `tests/test_make_report.py`

**Interfaces:**
- Consumes: Tasks 7/9 的 result JSON 格式。
- Produces: `collect_results(dir) -> list[dict]`、`render_markdown(rows) -> str`(欄位順序:case, mode, k, rtype, seed, io_count, ft_count, tree_wl, hpwl, runtime_s, peak_mem_mb;數字千分位;每 case 一小節);CLI `python -m ioplace.drivers.make_report --dir results/m0 --out docs/results/m0-baseline-report.md`。

- [ ] **Step 1: Write the failing test**

`tests/test_make_report.py`:
```python
import json
from ioplace.drivers.make_report import collect_results, render_markdown

def test_collect_and_render(tmp_path):
    r = {"mode": "flat", "config": "x/adaptec1.json", "k": 16, "rtype": "grid",
         "seed": 0, "io_count": 1234, "ft_count": 56, "tree_wl": 1.5e7,
         "hpwl": 1.2e7, "runtime_s": 100.0, "peak_mem_mb": 2048.0}
    (tmp_path / "a.json").write_text(json.dumps(r))
    rows = collect_results(str(tmp_path))
    assert len(rows) == 1 and rows[0]["case"] == "adaptec1"
    md = render_markdown(rows)
    assert "adaptec1" in md and "1,234" in md and "| flat |" in md
```

- [ ] **Step 2: Run → FAIL**(module not found),同前格式。

- [ ] **Step 3: Write implementation**

`ioplace/drivers/make_report.py`:
```python
import argparse, glob, json, os

COLS = ["mode", "k", "rtype", "seed", "io_count", "ft_count",
        "tree_wl", "hpwl", "runtime_s", "peak_mem_mb"]

def collect_results(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        r["case"] = os.path.basename(r["config"]).replace(".json", "")
        rows.append(r)
    return rows

def _fmt(v):
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.1f}"
    return str(v)

def render_markdown(rows):
    out = ["# Baseline 對照表\n"]
    for case in sorted({r["case"] for r in rows}):
        out.append(f"\n## {case}\n")
        out.append("| " + " | ".join(["case"] + COLS) + " |")
        out.append("|" + "---|" * (len(COLS) + 1))
        for r in [x for x in rows if x["case"] == case]:
            out.append("| " + " | ".join([case] + [_fmt(r.get(c, "")) for c in COLS]) + " |")
    return "\n".join(out) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    md = render_markdown(collect_results(a.dir))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests → PASS;跑 M0 實驗矩陣**

```bash
cd /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer
PY=$DP/.venv312/bin/python; CFG=$DP/install/test/ispd2005
for MODE in flat two_stage; do
  for K in 8 16; do
    DREAMPLACE_ROOT=$DP $PY -m ioplace.drivers.run_placement --config $CFG/adaptec1.json \
      --mode $MODE --k $K --rtype grid --seed 0 --out results/m0/adaptec1_${MODE}_k${K}_grid.json
  done
  DREAMPLACE_ROOT=$DP $PY -m ioplace.drivers.run_placement --config $CFG/adaptec1.json \
    --mode $MODE --k 16 --rtype slicing --seed 0 --out results/m0/adaptec1_${MODE}_k16_slicing.json
  DREAMPLACE_ROOT=$DP $PY -m ioplace.drivers.run_placement --config $CFG/bigblue4.json \
    --mode $MODE --k 16 --rtype grid --seed 0 --out results/m0/bigblue4_${MODE}_k16_grid.json
done
$PY -m ioplace.drivers.make_report --dir results/m0 --out docs/results/m0-baseline-report.md
```
Expected: adaptec1(211K cells)每 run 於 L4 上分鐘級;bigblue4(2.18M)GP 可能 30–90 分鐘/run(先跑 adaptec1 全組合確認無誤再跑 bigblue4)。報表產出後,人工檢查:two_stage 的 `io_count` 應明顯低於 flat(它為 cut 犧牲 WL),flat 的 `hpwl` 應低於 two_stage —— 這是 M0 的 sanity check;把觀察寫進報告尾註。

- [ ] **Step 5: Commit(M0 exit)**

```bash
git add ioplace/drivers/make_report.py tests/test_make_report.py docs/results/m0-baseline-report.md results/m0/*.json
git commit -m "feat: M0 baseline comparison report (flat vs two-stage on adaptec1/bigblue4)"
```

---

### Task 11: GPU evaluator(torch)

**Files:**
- Create: `ioplace/evaluator_gpu.py`
- Test: `tests/test_evaluator_gpu.py`

**Interfaces:**
- Consumes: `Netlist`、`RegionGrid`、`EvalResult`(欄位完全同 Task 6)。
- Produces: `evaluate_gpu(nl, node_x, node_y, rg, max_degree=256, device="cuda") -> EvalResult` — node_x/node_y 接受 np.ndarray 或 torch.Tensor;語意與 `evaluate` 完全一致。內部要點:
  - grid/前綴和預計算一次可重用:`class GpuEvalContext`(建構子吃 `nl`, `rg`, device;快取 grid tensor、`Ph`/`Pv` 前綴和、net 分桶索引),`evaluate_gpu` 為便利包裝,reweight 閉環(Task 12)直接持有 context 反覆呼叫 `context.evaluate(node_x_t, node_y_t)`。
  - **crossing O(1)/segment**:`Ph[iy, ix] = (grid[iy, :ix+1] 的相鄰相異數)` 沿 x 前綴和;水平段 crossing = `Ph[iy, ix1] - Ph[iy, ix0]`;垂直用 `Pv`。
  - **MST**:d==2 直接;3 ≤ d ≤ 32 dense batch Prim(`(B,d,d)` 距離張量,loop d−1 次,inf mask);33–256 逐 bucket chunk 處理(同法,B 較小);>256 用 presence 下界(`(net, rid)` 對 unique 計數)。
  - **FT / pair demand**:每段依 lattice 步長 bucketed padded gather(段最長 512 步)取得 id 序列 → per-net passed bitmask(int64 scatter OR)→ `ft = popcount(passed & ~pin_bm)`(K ≤ 64,popcount 用 K 次移位累加);pair demand 由 diff 位置的 (a,b) 對回 CPU 統計。

- [ ] **Step 1: Write the failing test**

`tests/test_evaluator_gpu.py`:
```python
import numpy as np
import pytest
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import evaluate
from ioplace.evaluator_gpu import evaluate_gpu
from tests.test_evaluator_ref import DIE  # (0,0,100,100)

def _random_case(rng, n_cells=40, n_nets=25, max_d=12):
    from ioplace.netlist import Netlist
    nx_, ny_ = rng.uniform(0.5, 99.5, n_cells), rng.uniform(0.5, 99.5, n_cells)
    pins, p2n = [], []
    for net in range(n_nets):
        d = int(rng.integers(2, max_d))
        pins += list(rng.choice(n_cells, d, replace=False)); p2n += [net] * d
    p2n = np.array(p2n, np.int32); pins = np.array(pins, np.int32)
    start = np.searchsorted(p2n, np.arange(n_nets + 1)).astype(np.int32)
    return Netlist(node_x=nx_, node_y=ny_, node_size_x=np.ones(n_cells),
                   node_size_y=np.ones(n_cells), num_movable=n_cells,
                   num_terminals=0, num_terminal_NIs=0,
                   pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                   pin2node=pins, pin2net=p2n,
                   flat_net2pin=np.arange(len(pins), dtype=np.int32),
                   flat_net2pin_start=start, xl=0., yl=0., xh=100., yh=100.)

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_gpu_matches_reference(seed):
    rng = np.random.default_rng(seed)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _random_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert gpu.io_count == ref.io_count
    assert gpu.ft_count == ref.ft_count
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.tree_wl == pytest.approx(ref.tree_wl, rel=1e-5)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand
```
(浮點座標隨機 → MST tie 機率趨零;若偶發 tie 造成 flake,將 `_random_case` 座標改為 `rng.uniform` 後加 `+ rng.normal(0, 1e-6)` 微擾並記錄。)

- [ ] **Step 2: Run → FAIL**(module not found)。

- [ ] **Step 3: 實作 `evaluator_gpu.py`(起步版 → 漸進向量化)**

先寫「委派 reference」的起步版讓 Step 1 測試立即全綠,之後逐步向量化(batch Prim → 前綴和 crossing → padded gather FT),**每完成一段向量化就重跑 Step 1 測試確保等價**;任何不一致以 reference 為準。效能目標(Step 4)驅動向量化完成度。

起步版:
```python
import numpy as np
from ioplace.evaluator_ref import evaluate as evaluate_ref

class GpuEvalContext:
    def __init__(self, nl, rg, device="cuda"):
        self.nl, self.rg, self.device = nl, rg, device
        # 向量化時在此快取:grid tensor、Ph/Pv 前綴和、net degree 分桶索引

    def evaluate(self, node_x, node_y):
        to_np = lambda t: (t.detach().cpu().numpy() if hasattr(t, "detach")
                           else np.asarray(t))
        return evaluate_ref(self.nl, to_np(node_x).astype(np.float64),
                            to_np(node_y).astype(np.float64), self.rg)

def evaluate_gpu(nl, node_x, node_y, rg, max_degree=256, device="cuda"):
    return GpuEvalContext(nl, rg, device).evaluate(node_x, node_y)
```

- [ ] **Step 4: Run tests + 效能量測**

```bash
DREAMPLACE_ROOT=$DP $PY -m pytest tests/test_evaluator_gpu.py -v
```
Expected: 5 passed。接著量測真實規模(寫成臨時腳本或 `python - <<EOF`,結果記入 Task 13 報告):
```bash
# adaptec1 與 bigblue4:載入 flat 結果 npz → GpuEvalContext.evaluate 計時(暖機後取 10 次中位數)
```
目標(非硬性,記錄實測):adaptec1 < 0.5s、bigblue4 < 5s(L4);若超出,記錄瓶頸段(建樹 vs FT gather)供後續優化,不阻塞 M1。

- [ ] **Step 5: Commit**

```bash
git add ioplace/evaluator_gpu.py tests/test_evaluator_gpu.py
git commit -m "feat: GPU evaluator equivalent to reference (batch Prim, prefix-sum crossings)"
```

---

### Task 12: Iteration callback patch 與 reweighting 閉環

**Files:**
- Create: `ioplace/reweight.py`, `ioplace/drivers/run_placement_reweight.py`, `ioplace/dp_patch/iteration-callback.patch`
- Modify(外部): `$DP/dreamplace/NonLinearPlace.py`(branch `io-aware`)與 `$DP/install/dreamplace/NonLinearPlace.py`(同步)
- Test: `tests/test_reweight.py`

**Interfaces:**
- Consumes: Task 11 `GpuEvalContext`;DREAMPlace `placer.data_collections.net_weights`(torch tensor,in-place 更新即生效 —— `WeightedAverageWirelength` 持 reference,偵察已確認)。
- Produces:
  - `update_net_weights(net_weights: torch.Tensor, per_net_crossings, alpha=0.5, cap=10.0) -> None` — `w = 1 + alpha * min(crossings, cap)`,in-place `copy_`(不得換 tensor 物件)。
  - callback 掛點:`NonLinearPlace` 實例屬性 `iteration_callback: Callable[[int, torch.Tensor], None] | None`(預設不存在;patch 在 `one_descent_step` 尾端呼叫)。
  - `run_reweight(config_json, k, rtype, seed, out_json, every=100, alpha=0.5) -> dict` — flat 流程 + callback;result dict 加 `reweight_every`, `alpha`, `num_reweights`。

- [ ] **Step 1: 打 patch(DREAMPlace 側)**

```bash
cd $DP && (git checkout io-aware 2>/dev/null || git checkout -b io-aware)
```
編輯 `$DP/dreamplace/NonLinearPlace.py`:在 `one_descent_step` 尾端、timing net weighting 區塊(搜尋 `enable_net_weighting`,約 462–497 行)**之後**、`logging.info(cur_metric)` 之前插入:
```python
            # io-aware: external per-iteration callback (see IO-Aware-Top-Level-Placer)
            cb = getattr(self, "iteration_callback", None)
            if cb is not None:
                cb(iteration, pos)
```
(縮排對齊該作用域;`iteration` 與 `pos` 皆為該作用域既有變數,若名稱不同以現場為準——`pos` 是 optimizer 的參數 tensor。)然後:
```bash
cd $DP && git add dreamplace/NonLinearPlace.py && git commit -m "io-aware: add per-iteration callback hook"
git diff main io-aware -- dreamplace/NonLinearPlace.py > /nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer/ioplace/dp_patch/iteration-callback.patch
cmake --build $DP/build312 --target install   # install 是複本,必須同步(fallback 3.9 環境無 build312,改用 cp 直接覆蓋 install/dreamplace/NonLinearPlace.py)
```
驗證同步:`diff $DP/dreamplace/NonLinearPlace.py $DP/install/dreamplace/NonLinearPlace.py` 無輸出。

- [ ] **Step 2: Write the failing test**

`tests/test_reweight.py`:
```python
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from ioplace.reweight import update_net_weights

def test_update_net_weights_formula_and_inplace():
    w = torch.ones(5)
    wid = id(w)
    update_net_weights(w, np.array([0, 1, 2, 20, 5]), alpha=0.5, cap=10.0)
    assert id(w) == wid
    assert torch.allclose(w, torch.tensor([1.0, 1.5, 2.0, 6.0, 3.5]))

@pytest.mark.slow
def test_reweight_simple_runs_and_calls_back(tmp_path):
    import os
    from ioplace.drivers.run_placement_reweight import run_reweight
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    res = run_reweight(os.path.join(root, "install/test/simple.json"),
                       4, "grid", 0, str(tmp_path / "rw.json"), every=20, alpha=0.5)
    assert res["mode"] == "reweight"
    assert res["num_reweights"] >= 1
```

- [ ] **Step 3: Run → FAIL**(module not found)。

- [ ] **Step 4: Write implementation**

`ioplace/reweight.py`:
```python
import numpy as np
import torch

def update_net_weights(net_weights, per_net_crossings, alpha=0.5, cap=10.0):
    c = torch.as_tensor(np.minimum(np.asarray(per_net_crossings, dtype=np.float32), cap),
                        device=net_weights.device, dtype=net_weights.dtype)
    net_weights.copy_(1.0 + alpha * c)
```

`ioplace/drivers/run_placement_reweight.py`:
```python
import json, os, time
import numpy as np
from ioplace.drivers.run_placement import (_load_dreamplace, extract_final_positions,
    _evaluate_and_pack, get_regions_for)
from ioplace.netlist import netlist_from_placedb
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_gpu import GpuEvalContext
from ioplace.reweight import update_net_weights

def run_reweight(config_json, k, rtype, seed, out_json, every=100, alpha=0.5):
    import torch, NonLinearPlace
    t0 = time.time()
    params, placedb = _load_dreamplace(config_json)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)           # initialize 後(scale 後)座標系
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, k, rtype, seed))
    ctx = GpuEvalContext(nl, rg, device="cuda")
    lr = params.global_place_stages[0]["learning_rate"]
    placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
    n_all, n_phys = placedb.num_nodes, placedb.num_physical_nodes
    state = {"count": 0}

    def cb(iteration, pos):
        if iteration == 0 or iteration % every != 0:
            return
        node_x = pos.data[:n_phys]
        node_y = pos.data[n_all:n_all + n_phys]
        res = ctx.evaluate(node_x, node_y)
        update_net_weights(placer.data_collections.net_weights,
                           res.per_net_crossings, alpha=alpha)
        state["count"] += 1

    placer.iteration_callback = cb
    placer(params, placedb, lr)
    node_x, node_y = extract_final_positions(placer, placedb)
    _, metrics = _evaluate_and_pack(placedb, node_x, node_y, k, rtype, seed)
    result = {"mode": "reweight", "config": config_json, "k": k, "rtype": rtype,
              "seed": seed, "reweight_every": every, "alpha": alpha,
              "num_reweights": state["count"], "runtime_s": time.time() - t0,
              "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, **metrics}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_json + ".npz", node_x=node_x, node_y=node_y)
    return result
```
(注 1:`GpuEvalContext.evaluate` 接受 GPU tensor 分段 view;preconditioner 也讀 `net_weights`(偵察 PlaceObj.py:76),in-place 更新兩者同步生效,這正是我們要的。注 2:cb 內 `pos.data[:n_phys]` / `pos.data[n_all:n_all+n_phys]` 的佈局假設與 `extract_final_positions` 相同 —— 以 Task 7 Step 1 的確認結果為準,若當時採候選 A,此處同步修改。)

- [ ] **Step 5: Run tests**

```bash
$PY -m pytest tests/test_reweight.py -v -m "not slow"
DREAMPLACE_ROOT=$DP $PY -m pytest tests/test_reweight.py -v -m slow
```
Expected: 全 passed;slow test 確認 callback 有被呼叫(`num_reweights >= 1`)。

- [ ] **Step 6: Commit**

```bash
git add ioplace/reweight.py ioplace/drivers/run_placement_reweight.py ioplace/dp_patch/iteration-callback.patch tests/test_reweight.py
git commit -m "feat: net-reweighting closed loop via DREAMPlace iteration callback"
```

---

### Task 13: M1 對照表(exit)

**Files:**
- Create: `docs/results/m1-reweight-report.md`(產出)
- Modify: `ioplace/drivers/make_report.py`(COLS 加入 `num_reweights`,已有欄位缺值顯示空白 —— `_fmt(r.get(c, ""))` 已處理)

**Interfaces:**
- Consumes: Tasks 10/11/12。

- [ ] **Step 1: 跑 M1 實驗矩陣**

```bash
PY=$DP/.venv312/bin/python; CFG=$DP/install/test/ispd2005
# adaptec1: 3 modes × K∈{8,16,32} grid + K=16 slicing;bigblue4: 3 modes × K=16 grid
for K in 8 16 32; do
  DREAMPLACE_ROOT=$DP $PY -m ioplace.drivers.run_placement --config $CFG/adaptec1.json \
    --mode reweight --k $K --rtype grid --seed 0 --reweight-every 100 --alpha 0.5 \
    --out results/m1/adaptec1_reweight_k${K}_grid.json
done
DREAMPLACE_ROOT=$DP $PY -m ioplace.drivers.run_placement --config $CFG/bigblue4.json \
  --mode reweight --k 16 --rtype grid --seed 0 --reweight-every 100 --alpha 0.5 \
  --out results/m1/bigblue4_reweight_k16_grid.json
cp results/m0/*.json results/m1/    # 併入 M0 的 flat/two_stage 數字同表呈現
$PY -m ioplace.drivers.make_report --dir results/m1 --out docs/results/m1-reweight-report.md
```

- [ ] **Step 2: 補充 evaluator 效能表與結論**

在 `docs/results/m1-reweight-report.md` 手動追加兩節:(1) **GPU evaluator runtime**(Task 11 Step 4 的實測:adaptec1 / bigblue4 單次評估時間、與 reference 版對照、reweight 閉環的總 evaluator 開銷佔比);(2) **M1 觀察**:reweight vs flat 的 io_count 變化(%)、hpwl 代價(%)、與 two_stage 的相對位置;α 與 every 的初步敏感度印象(若時間允許,加跑 alpha ∈ {0.2, 1.0} 各一組)。

- [ ] **Step 3: M1 exit 檢核(對照 spec §9)**

- [x] 條件 1:`mst_crossing_eval` 等價驗證(Task 11 tests)與 runtime 報告(本報告)
- [x] 條件 2:reweight 的 io_count 相對 flat 明顯下降 —— 若**沒有**下降,這本身是重要負面結果:記錄數字、檢查 per-net weight 分布與 crossing 分布,在報告寫下診斷假設(α 太小?every 太疏?WL 項壓制?),**不要**為了過關而只挑好看的 case。

- [ ] **Step 4: Commit(M1 exit)**

```bash
git add docs/results/m1-reweight-report.md results/m1/*.json ioplace/drivers/make_report.py
git commit -m "feat: M1 reweighting comparison report (flat vs two-stage vs reweight)"
```

---

## 附錄:已知不確定點(每項都有對應驗證步)

| # | 不確定點 | 對應驗證 |
|---|---|---|
| 1 | Python 3.12 重 build 的相容範圍(distutils、numpy 2.x、舊 pin 依賴) | Task 1 Step 2 失敗處置(3.12 → 3.11 → 修復 3.9 venv 的 fallback 鏈) |
| 2 | 最終座標取回機制(placedb 回寫 vs placer.pos) | Task 7 Step 1 investigation(比對 simple.gp.pl) |
| 3 | mtkahypar Python API 精確簽名(版本間變動) | Task 8 Step 1 小圖 smoke test;CLI fallback |
| 4 | `detailed_place_engine` 欄位名 | Task 7 Step 4 注記(以 install 的 test JSON 為準) |
| 5 | initialize 的 shift/scale 是否恆等 | Task 9 座標系策略(等比產生器 + `test_get_regions_scale_equivariant`) |
| 6 | multi-electrostatics 在 simple 小 case 的數值穩定性 | Task 9 Step 4(fallback 改 adaptec1) |
| 7 | callback 作用域內變數名(`iteration`/`pos`) | Task 12 Step 1 注記(以現場為準;timing 區塊是同作用域範本) |

## 附錄:M0/M1 完成後的下一步(不在本 plan)

- M2(可微 IO 項)另立 plan:先以純 PyTorch 實作 S1 soft membership + S2 期望 connectivity(spec §5.1–5.2),在 adaptec1 上與本 plan 的 reweight 做 ablation,再決定 CUDA 化。屆時 `GpuEvalContext` 的分桶結構與 `RegionGrid` 直接復用。
- 本 plan 刻意延後的 spec 項目:DEF REGION/GROUP 的 region 輸入(M0/M1 實驗全在 Bookshelf case,region 走 JSON sidecar;LEF/DEF 流程於 M4 接 mempool_cluster 時補)、多 seed(≥3)完整矩陣(M5)、5M/10M/30M benchmark 製作(M4)。
