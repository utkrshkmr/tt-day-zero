# References

Upstream sources for the other documents. Pages marked **read** were opened on 2026-11-16 and are what the claims in this repository rest on. The rest are pointers that were not re-read on that date.

## Start here

| Source | Use it for |
| --- | --- |
| [Understanding the Tenstorrent Software Stack](https://docs.tenstorrent.com/getting-started/tt-software-stack.html) (read) | Tenstorrent's own one-page description of the layers |
| [Serve Models](https://docs.tenstorrent.com/getting-started/serve-models.html) (read) | `tt serve`, the catalog, image pins |
| [Deploy LLMs](https://docs.tenstorrent.com/getting-started/vLLM-servers.html) (read) | First vLLM deployment through tt-inference-server |
| [Getting started and TT-Installer](https://docs.tenstorrent.com/getting-started/README.html) | Driver, firmware, hugepages, tools |

## Serving (Layer 7)

| Source | Use it for |
| --- | --- |
| [tt-inference-server](https://github.com/tenstorrent/tt-inference-server) | `run.py`, workflows, MODEL_SPECS, `docs/model_support/models_by_hardware.md` |
| [tt-inference-server releases](https://github.com/tenstorrent/tt-inference-server/releases) (read) | Per-release model support changes and status levels |
| [tt-inference-server lesson](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/tt-inference-server/) (read) | Flags, device names, how model specs resolve |
| [tenstorrent/vllm](https://github.com/tenstorrent/vllm) (read) | vLLM fork; the TT plugin README; `dev` is the main branch |
| [Version compatibility matrix](https://docs.tenstorrent.com/tt-vscode-toolkit/version-compat/) (read) | Which versions go together; environment variables |
| [QwQ-32B catalog page](https://docs.tenstorrent.com/tt-awesome/models/qwq-32b/) (read) | Example of a per-model support page with pins |

## Model implementations (Layer 6)

| Source | Use it for |
| --- | --- |
| [tt-metal README](https://github.com/tenstorrent/tt-metal/blob/main/README.md) (read) | Model demo table with measured throughput; links to bring-up docs and tech reports |
| `models/docs/model_bring_up.md` and `models/docs/MODEL_UPDATES.md` in tt-metal | The upstream bring-up process; recent model changes |
| [Bounty program: model bring-up from scratch](https://docs.tenstorrent.com/tt-vscode-toolkit/lessons/bounty-program/) (read) | Worked example: Phi-3 on the Llama base, module by module |
| [tt-forge-models](https://github.com/tenstorrent/tt-forge-models) | Loaders for the compiler path |

## Compiler (Layer 5)

| Source | Use it for |
| --- | --- |
| [tt-forge](https://github.com/tenstorrent/tt-forge) (read) | The umbrella: frontends, sub-projects, how they connect |
| [Bringing up models with TT-Forge](https://docs.tenstorrent.com/tt-forge/model-bring-up-guide.html) (read) | `torch.compile(backend="tt")`, warm-up behavior, dtypes |
| [TT-Forge: the compiler pipeline](https://docs.tenstorrent.com/tt-quietbox2-guide/builder-hacker/06-tt-forge-compiler/) (read) | Stage-by-stage walk through tt-xla |
| [tt-xla](https://github.com/tenstorrent/tt-xla) | The PyTorch and JAX frontend |
| [tt-xla model auto-discovery tests](https://docs.tenstorrent.com/tt-xla/model_auto_discovery_tests.html) (read) | The test runner, `bringup_status` values, test config, adding a model |
| [tt-mlir overview](https://docs.tenstorrent.com/tt-mlir/overview.html) (read) | Dialects and the architecture diagram |
| [tt-mlir dialects overview](https://docs.tenstorrent.com/tt-mlir/dialects-overview.html) (read) | `ttcore`, `ttir`, `ttnn`, `ttkernel`, `ttmetal` |
| [tt-mlir project structure](https://docs.tenstorrent.com/tt-mlir/project-structure.html) (read) | Repository layout |
| [tt-mlir tools](https://docs.tenstorrent.com/tt-mlir/tools.html) | `ttmlir-opt`, `ttmlir-translate`, `ttrt`, tt-explorer |
| [Adding an op to tt-mlir](https://docs.tenstorrent.com/tt-mlir/adding-an-op.html) | The steps for a new lowering |
| [ttnn.jit](https://docs.tenstorrent.com/tt-mlir/ttnn-jit.html) (read) | Fusing TT-NN ops through the D2M pipeline; current limits |
| [PyKernel guide](https://docs.tenstorrent.com/tt-mlir/pykernel.html) | Kernels in Python through `ttkernel` |
| [tt-lang overview](https://docs.tenstorrent.com/tt-lang/overview.html) (read) | The kernel DSL |
| [tt-lang specification](https://docs.tenstorrent.com/tt-lang/specs/TTLangSpecification.html) | Language reference and support matrix |

## TT-NN and TT-Metalium (Layers 4 and 3)

| Source | Use it for |
| --- | --- |
| [tt-metal](https://github.com/tenstorrent/tt-metal) | Everything from TT-NN down |
| [tt-metal repository structure (DeepWiki)](https://deepwiki.com/tenstorrent/tt-metal/1.1-repository-structure) (read) | Directory map; third-party, auto-generated |
| [TT-NN documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) | Operations, tensors, memory configs |
| [TT-Metalium documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html) | Host API, kernel APIs, programming examples, tools |
| [The TT-Metal architecture](https://docs.tenstorrent.com/tt-quietbox2-guide/builder-hacker/01-tt-metal-architecture/) (read) | Short layered description |
| `METALIUM_GUIDE.md` in tt-metal | The architecture and programming model in one document |
| [Custom SFPI operations](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/custom_sfpi.html) | Writing a vector-unit function |

Tech reports, under `tech_reports/` in tt-metal:

| Report | Topic |
| --- | --- |
| `matrix_engine/` | Matrix engine, fidelity, FP32 accumulation |
| `GEMM_FLOPS/` | Matmul throughput: peak, automatic and hand-tuned |
| `AdvancedPerformanceOptimizationsForModels/` | Metal Trace and multiple command queues |
| `FlashAttention/` | A hand-written kernel worked through |
| `LLMs/` | LLM bring-up in TT-NN |
| `CNNs/` | Convolution mapping and sharding |
| `tensor_layouts/`, `data_formats/` | Tiles, layouts, dtypes |
| `Programming_Mesh_of_Devices/` | `MeshDevice` and multi-chip TT-NN |
| `CCL/` | Collective ops and tuning |
| `TT-Fabric/` | Multi-chip routing |
| `Saturating_DRAM_bandwidth/` | Memory-bound limits |
| `MetalProfiler/` | The device profiler |

## Kernel primitives and hardware (Layers 2 to 0)

| Source | Use it for |
| --- | --- |
| [tt-llk](https://github.com/tenstorrent/tt-llk) | Frozen repository; still the best written introduction to LLK |
| [sfpi](https://github.com/tenstorrent/sfpi) | Vector-unit compiler support |
| [tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation) | The hardware reference for a Tensix tile |
| [corsix.org Wormhole series, part 1](https://www.corsix.org/content/tt-wh-part1) through [part 5](https://www.corsix.org/content/tt-wh-part5) | Independent low-level description of the chip |
| [Wormhole specifications](https://docs.tenstorrent.com/aibs/wormhole/specifications.html) | Card-level numbers |
| [T3000 specifications](https://docs.tenstorrent.com/systems/t3000/specifications.html) | The eight-chip box |
| [tt-smi](https://github.com/tenstorrent/tt-smi), [tt-topology](https://github.com/tenstorrent/tt-topology), [tt-flash](https://github.com/tenstorrent/tt-flash), [tt-firmware](https://github.com/tenstorrent/tt-firmware), [tt-kmd](https://github.com/tenstorrent/tt-kmd), [tt-system-tools](https://github.com/tenstorrent/tt-system-tools), [tt-installer](https://github.com/tenstorrent/tt-installer), [tt-exalens](https://github.com/tenstorrent/tt-exalens) | System software |

## Community

| Source | Use it for |
| --- | --- |
| [tt-awesome](https://docs.tenstorrent.com/tt-awesome/) | Curated list of community projects, including a CPU emulator for developing kernels without hardware |
| [Tenstorrent Discord](https://discord.gg/tenstorrent) | Fastest route to an answer on a stuck bring-up |
| Issues on tt-metal, tt-mlir, tt-xla, tt-inference-server | Search for the model name before starting work; some carry bounties |

## Version caution

Sources read on 2026-11-16 did not agree on the current release numbers (different pages named different latest versions of tt-metal and tt-inference-server). Do not copy a version from this repository. Read the pin from the catalog entry for the model you are serving, and record what you actually ran in the run note.
