# Benchmark sources for `NV_NVDLA_partition_*` and the TILOS trio (Ariane / NVDLA / BlackParrot) — fact-finding report

Date: 2026-09-18. Produced by an agent during the 2026-09-18 brainstorming
session to locate ready-to-use NVDLA/ASAP7 and TILOS benchmark sources for
the v2 Tier 3 benchmark plan; copied verbatim into the repository on
2026-09-19. Caveats the report itself states: nothing from any benchmark set
was downloaded, only page/API metadata was fetched (GitHub contents API +
raw.githubusercontent HEAD/GET, WebFetch of project pages and arXiv HTML,
plus text extracted locally from the ISPD24/25 contest intro PDFs, which were
deleted afterwards); every number is quoted from the cited page; items
explicitly marked **[unverified]** or **[inference]** in the body are not
independently confirmed and should be re-checked before being relied on.

Method: GitHub contents API + raw.githubusercontent HEAD/GET, WebFetch of project pages and arXiv HTML, text extracted locally from the ISPD24/25 contest intro PDFs (deleted afterwards). Nothing from any benchmark set was downloaded. Every number below is quoted from the cited page; items marked **[unverified]** or **[inference]** are not.

---

## 1. Where the literal names `NV_NVDLA_partition_*` occur

The names come from NVIDIA's `nvdla/hw` build: the Integrator's Manual lists the five physical partitions "NV_NVDLA_partition_a NV_NVDLA_partition_c NV_NVDLA_partition_o NV_NVDLA_partition_m NV_NVDLA_partition_p", states "The designs will be compiled at this hierarchy, and netlists will be generated for these designs. These are independent sub-designs for synthesis, which are instantiated in a top-level wrapper", and names the flow outputs `NV_NVDLA_partition*.gv` / `.full.def` / `.sdc` — but that flow needs commercial tools and your own technology (https://nvdla.org/hw/v2/integration_guide.html). RTL license: "NVIDIA Open NVDLA License and Agreement v1.0" (https://raw.githubusercontent.com/nvdla/hw/master/LICENSE).

### 1a. NVlabs/CircuitOps — the only source with ready-made per-partition DEF + netlist + tech LEF (ASAP7)

Repo: https://github.com/NVlabs/CircuitOps — LICENSE: Apache-2.0.

`designs/asap7/` contains directories **NV_NVDLA_partition_a, NV_NVDLA_partition_c, NV_NVDLA_partition_m, NV_NVDLA_partition_p** (no `partition_o`), plus aes_192, ariane133, ariane136, gcd, ibex, jpeg, mock-array_Element, uart. `designs/nangate45/`: aes, ariane133, bp_be, bp_fe, gcd, ibex, jpeg, swerv (no NVDLA). `designs/sky130hd/`: aes, gcd, ibex, jpeg, riscv32i.

Files per NVDLA partition (GitHub API sizes, bytes):

| dir | 6_final.def.gz | 6_final.v.gz | 6_final.sdc.gz | 6_final.spef.gz |
|---|---|---|---|---|
| NV_NVDLA_partition_a | 4,168,642 | 857,376 | 2,655 | 17,833,950 |
| NV_NVDLA_partition_c | 19,600,764 | 4,366,037 | 4,278 | 91,116,221 |
| NV_NVDLA_partition_m | 2,197,673 | 593,366 | 2,434 | 7,652,291 |
| NV_NVDLA_partition_p | 8,140,782 | 1,809,196 | 4,118 | 34,431,887 |
| ariane133 (asap7) | 16,852,220 | 3,469,968 | 2,451 | 86,706,318 |

`6_final.*` is ORFS's stage-6 ("finish") naming, i.e. post-route placed+routed DEF **[inference from naming; DeepWiki third-party summary calls them "fully processed form after place and route"]**. Instance counts from `IRs/README.md` ("post filler instance count", "Core utilisation"):

| Technode | Design | # of instances | Core utilisation |
|---|---|---|---|
| asap7 | NV_NVDLA_partition_m | 65,353 | 30 |
| asap7 | NV_NVDLA_partition_a | 111,207 | 30 |
| asap7 | NV_NVDLA_partition_p | 215,140 | 30 |
| asap7 | NV_NVDLA_partition_c | 499,581 | 30 |

(https://raw.githubusercontent.com/NVlabs/CircuitOps/main/IRs/README.md; the README says "Refer to the OpenROAD-flow-scripts git repo for default utilisation values".) IR tables for the same four partitions are in `IRs/asap7/`.

Technology files in the same repo, `platforms/asap7/lef/`: `asap7_tech_1x_201209_tech.lef`, `asap7sc7p5t_27_R_1x_201211.lef`, `sram_asap7_16x256_1rw.lef`, `sram_asap7_32x256_1rw.lef`, `sram_asap7_64x256_1rw.lef`, `sram_asap7_64x64_1rw.lef` (plus `lib/`, `rcx_patterns.rules`, `setRC.tcl`). These are the same cell-library revision (27_R_201211) and SRAM names as TILOS's ASAP7 enablement, not ORFS master's `28_R_220121a` — use CircuitOps'/TILOS's LEFs with these DEFs **[inference]**.

Caveats: no `partition_o`; util 30 % and filler cells included in counts; DEFs are post-route (strip PLACED/ROUTED sections to re-place); no RTL; per-design instance counts not given for ariane133/136 in the README.

### 1b. VLSIDA/HighTide — all five partitions on asap7 / nangate45 / sky130hd / gt2n, but you run the flow (bazel-orfs); no checked-in netlist/DEF

Repo: https://github.com/VLSIDA/HighTide — LICENSE: "BSD 3-Clause License, Copyright (c) 2024-2026, Matthew Guthaus". Paper: HighTide: An Agent-Curated Open-Source VLSI Benchmark Suite, arXiv 2606.04126 (CC BY 4.0). Site: https://vlsida.github.io/HighTide/ (design table lists "NVDLA | NVIDIA Deep Learning Accelerator (5 partitions) | Verilog | asap7, nangate45, sky130hd, gt2n"; "~150k cells (NVDLA partition)"). `hightide-benchmarks.dev/results.html` did not resolve (DNS) when fetched.

Layout: `designs/{asap7,nangate45,sky130hd,gt2n}/NVDLA/partition_{a,c,m,o,p}/{BUILD.bazel,constraint.sdc}` plus `designs/<plat>/NVDLA/sram/{lef,lib}` (bsg_fakeram macros). `partition_a/BUILD.bazel` is `hightide_design(name="partition_a", top="NV_NVDLA_partition_a", platform="asap7", verilog_files=["//designs/src/NVDLA:rtl"], ...)`. README: "Each design's upstream source is pinned in the root MODULE.bazel as a hermetic http_archive ... there is no git submodule and no vendored copy of the RTL in the repo"; DECISIONS.md: "@nvdla_hw_src (http_archive, pinned 771f20cc, nv_small) supplies the source; gen_vmod ... runs tmake -build vmod". Build targets: `//designs/asap7/NVDLA/partition_c:partition_c_{synth,floorplan,place,cts,route,final}`; `tools/bazel_to_orfs.sh designs/asap7/NVDLA/partition_a` exports a self-contained plain-ORFS bundle (config.mk + inputs + run.sh). Also present: `bp_processor/bp_quad` and `bp_uno` (asap7, nangate45, sky130hd). No Ariane or MemPool in HighTide.

Per-partition sizes as logged in `designs/src/NVDLA/DECISIONS.md` (HighTide's own build logs, "logic cells"):

| partition | asap7 | nangate45 | sky130hd | macros (gt2n section) |
|---|---|---|---|---|
| a | 62,350 (util 62.0 %) | 53,030 (util 42.7 %) | 35,825 | none ("all SRAMs FF") |
| c | 268,324 (util 30.6 %) | 250,898 (util 30.8 %) | ~265 k (util 25) | 65 SRAM instances, 2 types (64x256, 11x128) |
| m | 19,667 (util 56.8 %) | 13,053 (util 50.7 %) | 11,022 | none |
| o | 241,685 (util 42.5 %) | 189,268 (util 36.0 %) | 185,458 | 17 SRAM instances, 8 types |
| p | 98,602 (util 46.8 %) | 67,284 (util 35.6 %) | 64,868 | 6 SRAM instances, 4 types |

Paper Appendix A lists "NVDLA-small 500,000 cells, 110 macros" (whole design) and "BlackParrot v2 470,000 cells, 140 macros". Note HighTide's SRAM policy differs from TILOS (TILOS keeps macros only in partition c; HighTide keeps macros in c/o/p and converts partition a's SRAMs to flip-flops).

### 1c. TILOS-AI-Institute/MacroPlacement — RTL for all five partitions, but flows/netlists/DEF only for `NV_NVDLA_partition_c`

Repo: https://github.com/TILOS-AI-Institute/MacroPlacement — LICENSE: "BSD 3-Clause License, Copyright (c) 2018-2021, The Regents of the University of California".

`Testcases/nvdla/README.md`: RTL generated from the `nv_small` branch of nvdla/hw with `nv_small.spec`; "As NVDLA partition c has macros, our testcase only includes the partition c"; dual-port 256x64 SRAM replaced by two single-port fakeram wrappers (`fakeram<7|45|130>_256x64_dp.v`); "Clone the nv_small branch. For the default branch, generated RTL will have more than million instances." `Testcases/nvdla/rtl/` nevertheless contains `NV_NVDLA_partition_a.v`, `_c.v`, `_m.v`, `_o.v`, `_p.v` and `NV_nvdla.v` (HTTP 200 on raw.githubusercontent), so a/m/o/p RTL is present but unflowed.

Flow artefacts (`Flows/<enablement>/nvdla/`):

| enablement | netlist/ | def/ |
|---|---|---|
| NanGate45 | `NV_NVDLA_partition_c.v` (20,165,202 B, Genus Flow-1), `NV_NVDLA_partition_c.pb.txt.gz` (CT protobuf) | `NV_NVDLA_partition_c_fp.def` (271,187 B), `NV_NVDLA_partition_c_fp_placed_macros.def` (290,133 B) |
| ASAP7 | `NV_NVDLA_partition_c.pb.txt.gz` only (no `.v`: HTTP 404) | `NV_NVDLA_partition_c_fp.def` (240,852 B), `_fp_placed_macros.def` (248,293 B) |
| SKY130HD | `NV_NVDLA_partition_c.v` (20,165,202 B), `.pb.txt.gz` | `NV_NVDLA_partition_c_fp.def` (275,584 B), `_fp_placed_macros.def` (292,759 B) |

Sizes: README table "NVDLA | 45295 flip-flops | (256x64-bit SRAM) x 128"; Docs/OurProgress: "The number of hard macros in NVDLA is 128". Standard-cell count is not stated in the repo; the ISPD24 contest's NVDLA (128 macros, NanGate45) has 166K std cells (see §3) and is consistent with being this netlist **[inference]**.

### 1d. Places checked that do NOT contain `NV_NVDLA_partition_*`

- **OpenROAD-flow-scripts** (`flow/designs/`): asap7 = aes-block aes-mbff aes aes_lvt coralnpu cva6 ethmac ethmac_lvt gcd-ccs gcd ibex jpeg jpeg_lvt minimal mock-alu mock-cpu riscv32i-mock-sram riscv32i swerv_wrapper tinyRocket uart; nangate45 = aes ariane133 ariane136 black_parrot bp_be_top bp_fe_top bp_multi_top bp_quad cva6 dynamic_node gcd ibex jpeg mempool_group swerv swerv_wrapper tinyRocket; sky130hd = aes chameleon gcd ibex jpeg microwatt riscv32i. `flow/designs/src/` has ariane, ariane133, ariane136, black_parrot, bp_*_top, bp_quad, mempool_group — no nvdla.
- **ChiPBench** (https://github.com/MIRALab-USTC/ChiPBench, data at https://huggingface.co/datasets/MIRA-Lab/ChiPBench-D, 2.68 GB, metadata "bsd-3-clause"): 20 NanGate45/ORFS designs (ariane133 167,907 cells/132 macros; ariane136 171,347/136; bp 307,055/24; bp_multi 152,287/26; isa_npu 427,003/15; swerv_wrapper, vga_lcd, dft68, or1200, mor1kx, ethernet, VeriGPU, ariane81, bp_fe/bp_be variants). Provides pre_place.def, macro_placed.def, case-specific LEF/LIB, synthesized netlist, SDC. No NVDLA, bp_quad or mempool.
- **Hier-RTLMP** paper (arXiv 2304.11761) Table II: Ariane NG45 (133 macros, 118K std cells), BlackParrot NG45 (220, 769K), CA53/Ariane/BlackParrot/Tabla09/Tabla01/MemPool on GF12 (commercial). "No NVDLA design or NV_NVDLA_partition is mentioned anywhere in this paper." Runscripts live in OpenROAD and MacroPlacement.
- **MacroRank** (ASPDAC 2023): ISPD-2015 benchmarks placed with DREAMPlace — no NVDLA. DREAMPlace's own benchmark scripts are ISPD2005/2015-style **[from search summaries, not re-verified]**.
- **ICCAD CAD Contest 2022/2023**: 2022 Problem B "3D Placement with D2D Vertical Connections" (abstract: "movable macros were not included"); 2023 problems A "Multi-bit Large-scale Boolean Matching", B "3D Placement with Macros" (Synopsys), C "Static IR Drop Estimation Using ML" (https://www.iccad-contest.org/2023/). Problem pages returned 404; no evidence of NVDLA-named designs **[not verified beyond titles]**.

---

## 2. TILOS MacroPlacement layout and contents

Top-level: `Testcases/{ICCAD04, ariane133, ariane136, bp_quad, mempool, nvdla}` (RTL: `rtl/` folders; Ariane netlist from lowRISC; MemPool tile & group from pulp-platform; bp_quad "verilog netlist ... from the OpenROAD GitHub repo", `Testcases/bp_quad/rtl/bsg_chip_block.sv2v.v`); `Enablements/{NanGate45, ASAP7, SKY130HD}/{lib,lef,qrc}`; `Flows/<enablement>/<testcase>/{constraints,def,netlist,scripts,run}`.

Flows present (GitHub API): NanGate45: ariane133 ariane136 bp_quad mempool_group mempool_tile nvdla; ASAP7: ariane133 ariane136 bp_quad mempool_group mempool_tile nvdla; SKY130HD: ariane133 ariane136 mempool_tile nvdla. (Flows/README.md lists only four ASAP7 testcases; bp_quad and mempool_group were added later.)

What `def/` and `netlist/` are (Flows/README.md): "The def directory contains the floorplan DEF file used in the SP&R flow. We provide two DEF files, one with just the core area and pin placements that are used for the logical synthesis flow Flow-1 and another DEF file that also includes macro placements ... The netlist directory contains the synthesized netlist from Flow-1." Flow-1 floorplan rules: "aspect ratio 1 and a utilization value of 40% to 60%. All the pins are placed on the left side of the floorplan"; halos "2um for ASAP7 and 5um for NanGate45 and SKY130HD". So the DEFs carry die/core, pins and (optionally) macro placement — no standard-cell placement.

Pre-synthesized netlist availability (verified per directory):

| enablement / testcase | netlist/ | def/ |
|---|---|---|
| NG45 ariane133 | ariane.v (11.9 MB), ariane.pb.txt.gz, netlist.pb.txt.gz, legalized.plc | ariane133_fp.def, ariane133_fp_placed_macros.def, manual_floorplan.def, place_srams.tcl, Util_51/ |
| NG45 ariane136 | ariane.v (13.9 MB), ariane.pb.txt.gz | ariane136_fp.def, ariane136_fp_placed_macros.def |
| NG45 mempool_tile | mempool_tile_wrap.v (20.9 MB), .pb.txt.gz | mempool_tile_wrap_fp.def, _fp_placed_macros.def |
| NG45 nvdla | NV_NVDLA_partition_c.v (20.2 MB), .pb.txt.gz | NV_NVDLA_partition_c_fp.def, _fp_placed_macros.def |
| NG45 bp_quad | no netlist dir (404) | bsg_chip_fp.def (235 KB), bsg_chip_fp_placed_macros.def, manual_macro_placement/ |
| NG45 mempool_group | no netlist dir (404) | mempool_group_fp.def (2.07 MB), _fp_placed_macros.def |
| ASAP7 ariane136 | ariane.v (200 OK), ariane.pb.txt.gz | (not listed) |
| ASAP7 ariane133 | no netlist dir (404) | ariane_fp.def, ariane_fp_placed_macros.def |
| ASAP7 mempool_tile | mempool_tile_wrap.v (20.8 MB), .pb.txt.gz | mempool_tile_wrap_fp.def, _fp_placed_macros.def |
| ASAP7 nvdla | .pb.txt.gz only | NV_NVDLA_partition_c_fp.def, _fp_placed_macros.def |
| ASAP7 bp_quad | no netlist dir (404) | bsg_chip_fp.def (227 KB), bsg_chip_fp_placed_macros.def |
| ASAP7 mempool_group | no netlist dir (404) | mempool_group_fp.def (1.95 MB), _fp_placed_macros.def |
| SKY130HD ariane133 | ariane.v (200 OK) | (fp defs) |
| SKY130HD nvdla | NV_NVDLA_partition_c.v (20.2 MB), .pb.txt.gz | NV_NVDLA_partition_c_fp.def, _fp_placed_macros.def |

Where no `.v` is checked in, `scripts/OpenROAD/<design>.tar.gz` ("includes all the required files to run Flow-3 using OpenROAD-flow-scripts": `make DESIGN_CONFIG=./designs/<enablement>/<design>/config_hier.mk`) runs synthesis from RTL. (The tarball file names could not be listed: GitHub API rate-limited; `nvdla.tar.gz`/`bp_quad.tar.gz` guesses returned 404 — **[exact tarball names unverified]**.)

Sizes (official):
- README table (flip-flops / macros): Ariane133 19,807 FF, (256x16 SRAM) x 133; Ariane136 19,839 FF, x 136; MemPool tile 18,278 FF, (256x32) x 16 + (64x64) x 4; MemPool group 360,724 FF, (256x32) x 256 + (64x64) x 64 + (128x256) x 2 + (128x32) x 2; NVDLA 45,295 FF, (256x64) x 128; BlackParrot 214,441 FF, six SRAM types (32x32 … 512x64).
- ISPD'23 "Assessment of RL for Macro Placement" (arXiv 2302.11014, Table II, "#FFs and #Macros ... in both NanGate45 and ASAP7"): Ariane #StdCells 99–117 K, 20 K FFs, 133 macros, 1 type; BlackParrot 686–835 K, 214 K FFs, 220 macros, 6 types; MemPoolGroup 2529–2729 K, 361 K FFs, 324 macros, 4 types. (Range = NG45 vs ASAP7.)
- Hier-RTLMP Table II: Ariane NG45 118 K std cells / 133 macros; BlackParrot NG45 769 K / 220.
- Docs/OurProgress: hard macros NVDLA 128, bp_quad 220, MemPool Group 324.

ASAP7 technology LEF:
- TILOS `Enablements/ASAP7/lef/`: `asap7_tech_1x_201209.lef` (19,910 B), `asap7_tech_4x_201209.lef`, `asap7sc7p5t_27_R_1x_201211.lef` (+4x), and 12 FakeRAM2.0 SRAM LEFs `sram_asap7_{116x128,124x64,16x256,256x128,32x128,32x256,32x32,48x256,62x64,64x256,64x512,64x64}_1rw.lef`. README: "ASAP7 ... available under the BSD 3-Clause license"; "we use the FakeRAM2.0 memory generator ... commit tag MacroPlacement.ISPD23".
- ORFS `flow/platforms/asap7/`: `config.mk` sets `TECH_LEF = $(PLATFORM_DIR)/lef/asap7_tech_1x_201209.lef`, `SC_LEF = asap7sc7p5t_28_$(PRIMARY_VT_TAG)_1x_220121a.lef`; the lef dir also has `asap7_tech_1x_260907.lef`, R/L/SL cell LEFs, `asap7sc7p5t_28_SRAM_1x_220121a.lef`, and `fakeram7_*` / `fakeram_*` / `fakeregfile_*` SRAM LEFs. ORFS platforms: asap7 common gf180 gt2n ihp-sg13g2 nangate45 sky130hd sky130hs sky130io sky130ram.
- ORFS nangate45 `bp_quad/config.mk`: `DESIGN_NAME = bsg_chip`, `SYNTH_HIERARCHICAL = 1`, six `fakeram45_*` macro LEF/LIBs, `DIE_AREA 0 0 3600 3600`, `CORE_AREA 10 12 3590 3590`, `MACRO_PLACE_HALO 10 10`.

---

## 3. ISPD 2024 / 2025 global-routing contest sets (NanGate45; include an NVDLA)

**ISPD 2024** (https://liangrj2014.github.io/ISPD24_contest/ ; https://github.com/liangrj2014/ISPD24_contest): "First set of benchmarks with Nangate45 technology node" on Google Drive; "all the essential input information for global routing is contained within the .cap files and .net files located in the 'Simple_inputs' folder. We also release the LEF/DEF files of the circuits just for reference"; "We released the hidden benchmarks and contest results - March 26, 2024". Table 2 of the contest intro PDF (text extracted from the PDF; columns #stdcells / #macros / #nets / #pins / density % / GCell grid):

| public | std cells | macros | nets | pins | dens | grid |
|---|---|---|---|---|---|---|
| Ariane_sample | 122K | 133 | 129K | 420K | 51 | 844x1144 |
| MemPool-Tile_sample | 129K | 20 | 136K | 500K | 51 | 475x644 |
| NVDLA_sample | 166K | 128 | 177K | 630K | 51 | 1240x1682 |
| BlackParrot_sample | 715K | 220 | 770K | 2.9M | 68 | 1532x2077 |
| MemPool-Group_sample | 3.1M | 320 | 3.3M | 10.9M | 68 | 1782x2417 |
| MemPool-Cluster_sample | 9.9M | 1296 | 10.6M | 40.2M | 68 | 3511x4764 |
| TeraPool-Cluster_sample | 49.7M | 4192 | 59.3M | 213M | 68 | 7891x10708 |

Blind: Ariane_rank 121K/133/128K/435K/68/716x971; MemPool-Tile_rank 128K/20/136K/483K/68/429x581; NVDLA_rank 164K/128/174K/610K/68/908x1682; BlackParrot_rank 780K/220/825K/2.9M/68/1532x2077; MemPool-Group_rank 3.0M/320/3.2M/10.9M/68/1782x2417; MemPool-Cluster_rank 9.9M/1296/10.6M/40.2M/51/4113x5580; TeraPool-Cluster_rank 49.7M/4192/59.3M/213M/51/9245x12544. (Public-set naming on the leaderboard: Ariane133_51, Ariane133_68, NVDLA, BlackParrot, MemPool-Tile, MemPool-Group, MemPool-Cluster, TeraPool-Cluster.) The NVDLA rows carry 128 macros, i.e. TILOS `NV_NVDLA_partition_c` on NG45 **[inference]**. Corroboration: OpenROAD issue #4618 reports ISPD24 ariane133_68 "Created 120202 components ... 127026 nets" and "cluster-001 with 9.8M std cells".

**ISPD 2025** (https://github.com/liangrj2014/ISPD25_contest/blob/main/index.md ; https://www.ispd.cc/contests/25/index.html): "For each testcase, this contest provides two sets of input files: a) industry-standard LEF, DEF, LIB, and SDC files, and b) simplified rerouting resource and net information files" (.cap/.net). Intro PDF: "Each set will include 10 placed circuits designed using the Nangate45 technology nodes. Notably, the unpublished blind benchmark suite shares the same netlists as the public suite but features distinct placement solutions." Yes, NVDLA is included:

| testcase | N_endpoint | N_net (visible) | GCell graph | N_net (blind) | GCell (blind) |
|---|---|---|---|---|---|
| ariane | 20,218 | 123,900 | 10x761x761 | 105,924 | 10x646x646 |
| bsg (bsg_chip) | 214,821 | 736,883 | 10x1384x1384 | 768,239 | 10x1384x1384 |
| NVDLA | 45,925 | 199,481 | 10x1120x1120 | 157,744 | 10x1120x1120 |
| mempool_tile | 13,350 | 136,120 | 10x428x428 | 135,814 | 10x386x386 |
| mempool_group | 347,869 | 3,274,611 | 10x1611x1610 | 3,218,496 | 10x1611x1610 |
| mempool_cluster | 1,082,397 | 12,047,279 | 10x3175x3175 | 12,168,735 | 10x3719x3719 |

Cell counts are not given on the page; NVDLA's 45,925 endpoints match TILOS's 45,295 NVDLA flip-flops closely and bsg's 214,821 match bp_quad's 214,441 FFs **[inference: same TILOS netlists]**. Release log: "First set of testcases and example OpenROAD codes" Oct 7, 2024 (Drive folder `12ei9JOKaMeSPgc9CZOWrVyLJRgb0f2is`); "Hidden testcase" Mar 28, 2025 (Drive folder `1J3yoVZ07ifQiJ8l7SQ_J1Y5y0zXcEVmR`); evaluation uses a forked OpenROAD (https://github.com/liangrj2014/OpenROAD_ISPD25) and ignores DEF GCELLGRID ("use the Gcell definitions provided in the .cap files, ... fixed size of 4200 × 4200"). GAP-LA (arXiv 2507.13375) describes them as "industrial designs synthesized with the modified Nangate 45nm open library technology node" from "the ISPD25 contest released by NVIDIA". No license statement was found on either contest page **[license unverified]**.

---

## 4. Size summary (official numbers only)

| design | tech | instances / std cells | macros | source |
|---|---|---|---|---|
| Ariane133 / 136 | NG45 & ASAP7 | 99–117 K std cells, 20 K FFs | 133 / 136 | ISPD'23 Table II; README FFs 19,807 / 19,839 |
| Ariane NG45 | NG45 | 118 K std cells | 133 | Hier-RTLMP Table II |
| ariane133 / ariane136 | NG45 (ORFS) | 167,907 / 171,347 cells | 132 / 136 | ChiPBench README |
| Ariane (ISPD24) | NG45 | 122 K std cells, 129 K nets | 133 | ISPD24 intro Table 2 |
| NV_NVDLA_partition_c (TILOS) | NG45/ASAP7/SKY130HD | 45,295 FFs (std cells not stated) | 128 (256x64) | TILOS README |
| NVDLA (ISPD24) | NG45 | 166 K std cells, 177 K nets, 630 K pins | 128 | ISPD24 intro Table 2 |
| NVDLA (ISPD25) | NG45 | 45,925 endpoints, 199,481 nets | — | ISPD25 index.md |
| NV_NVDLA_partition_{m,a,p,c} (CircuitOps) | ASAP7 | 65,353 / 111,207 / 215,140 / 499,581 post-filler instances (util 30) | — | CircuitOps IRs/README |
| NVDLA partitions (HighTide) | ASAP7 | a 62,350; m 19,667; o 241,685; c 268,324; p 98,602 logic cells | c 65, o 17, p 6, a/m 0 | HighTide DECISIONS.md |
| NVDLA partitions (HighTide) | NG45 | a 53,030; m 13,053; o 189,268; c 250,898; p 67,284 | | HighTide DECISIONS.md |
| NVDLA-small (whole) | — | 500,000 cells | 110 | HighTide paper App. A |
| NVDLA nv_full | — | "more than million instances" (RTL) | — | TILOS Testcases/nvdla/README |
| bp_quad (BlackParrot) | NG45 & ASAP7 | 686–835 K std cells, 214 K FFs | 220 (6 types) | ISPD'23 Table II; README 214,441 FFs |
| BlackParrot NG45 | NG45 | 769 K std cells | 220 | Hier-RTLMP Table II |
| BlackParrot (ISPD24) | NG45 | 715 K (public) / 780 K (blind) std cells | 220 | ISPD24 intro Table 2 |
| bsg_chip (ISPD25) | NG45 | 214,821 endpoints, 736,883 nets | — | ISPD25 index.md |
| BlackParrot v2 (HighTide) | — | 470,000 cells | 140 | HighTide paper App. A |
| MemPool group | NG45 & ASAP7 | 2,529–2,729 K std cells, 361 K FFs | 324 | ISPD'23 Table II |
| MemPool-Group / -Cluster / TeraPool (ISPD24) | NG45 | 3.1 M / 9.9 M / 49.7 M std cells | 320 / 1296 / 4192 | ISPD24 intro Table 2 |

---

## 5. Recommendation

For designs literally named `NV_NVDLA_partition_*` with LEF/DEF and a tech LEF already in place, use **NVlabs/CircuitOps** (`designs/asap7/NV_NVDLA_partition_{a,c,m,p}` + `platforms/asap7/lef`, Apache-2.0): it is the only public source that ships per-partition DEF and gate-level netlists for four partitions in one technology (ASAP7, 65 K–500 K post-filler instances at 30 % utilisation), and the ASAP7 tech/cell/SRAM LEFs it needs are in the same repo; you strip the post-route placement/routing from `6_final.def` and re-place. Its gaps are no `partition_o`, no design at or above 1 M instances, and 30 % utilisation. If you need all five partitions, other nodes (NG45/SKY130HD/GT2N) or control over utilisation, use **VLSIDA/HighTide** as the secondary source (BSD-3; every partition on four platforms; `tools/bazel_to_orfs.sh` yields plain ORFS bundles, cell counts 11 K–268 K per partition) at the cost of running synthesis/floorplan yourself; TILOS only ever flowed `partition_c`. For the ASAP7 trio (Ariane, NVDLA, BlackParrot), use **TILOS MacroPlacement** (BSD-3) as the single source: its `Enablements/ASAP7` supplies `asap7_tech_1x_201209.lef` + `asap7sc7p5t_27_R` + FakeRAM SRAM LEFs, and `Flows/ASAP7/{ariane133,ariane136,nvdla,bp_quad,mempool_group,mempool_tile}/def` supplies pin/macro floorplan DEFs; pre-synthesized ASAP7 `.v` netlists exist only for ariane136 and mempool_tile, so nvdla(c), bp_quad, mempool_group and ariane133 on ASAP7 must be synthesized via the bundled ORFS tarballs (Yosys) — CircuitOps' ASAP7 ariane133/136 + NVDLA final DEFs are a ready-made stopgap for two of the three. None of the NVDLA partitions nor Ariane reaches 1 M instances in any source; the only ≥1 M designs with placed DEF + tech LEF are the NanGate45 ISPD 2024/2025 sets (MemPool-Group 3.1 M, MemPool-Cluster 9.9 M, TeraPool 49.7 M; bp_quad ~0.7–0.8 M), so plan the 1 M+ scaling runs on those and keep NVDLA/Ariane/bp_quad for the macro/process ablations.
