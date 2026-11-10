# dreamkernel

**Declare a brain; get a brain.**

`dreamkernel` is a small, zero-dependency toolkit for describing neural structures as
declarations — area shapes (topologies), projection maps, and cell models — and for
**building a runnable spiking network from that declaration**.

> Status: `0.1.x`, research prototype. The public API is not frozen yet.
>
> 中文版见 [`README.zh-CN.md`](README.zh-CN.md)。

---

## Three ideas

| Idea | What it means |
|---|---|
| **Shape** (`Topo`) | An area declares its geometry: `Point`, `Line`, `Ring`, `Grid`, `Modules`, a cortical `column(radial, tangential)`, or a recursive `tiled(base, nx, ny, within)`. |
| **Distance semantics** (`Metric`) | Whether coordinates may be read as *distance*: `None` (connection-only), `Collapsed` (column-internal "next to" is topological, not millimetres), `Millimetres` (1:1, distance → conduction delay). |
| **Alignment** (`Map`) | How two topologies line up: `Identity`, `Affine{scale, offset, mirror}`, `Permute` (control), `AllToAll` (broadcast — must be written **explicitly**). |

### Why a "collapsed" metric tier exists

Conduction delay comes from axon length, but **not every topology has millimetres**:
inside a cortical column, "next to" means *same column, adjacent layer*, not "0.3 mm away".
Treating it as metric fabricates a delay out of nothing. Hence three tiers instead of
"either fully abstract or fully 1:1".

---

## Declare → build → run

```rust
use dreamkernel::sim::{Brain, Config};

// ... declare SRC / MID / OUT (see examples/minimal.rs) ...

let mut brain = Brain::declare(&[SRC, MID, OUT], Config::default())?; // declaration → brain
brain.wire_local();                                                  // local recurrence
let mut ext = vec![0.0; brain.areas.len()];
ext[0] = 1.2;                                                        // drive the first area only
for _ in 0..300 { brain.step(&ext, 1.0); }
println!("{} Hz", brain.rate_hz(0));
```

Run the bundled example (three areas, two declared projections):

```bash
cargo run --example minimal
```

Actual output (`dt = 0.5 ms`):

```
神经元 180 个，共 2 条投射：
  SRC → MID（Feedforward）1152 条突触
  MID → OUT（Feedforward）1152 条突触
t=100ms  SRC   63.7 Hz   MID   27.7 Hz   OUT   37.7 Hz
t=300ms  SRC   68.2 Hz   MID   31.8 Hz   OUT   42.9 Hz
MID 的**跨区**入边延迟：4–5 tick（SRC 有米制：跨区 8mm + 区内位置差）
OUT 的**跨区**入边延迟：4–4 tick（MID 是坍缩档：区内位置不计入距离）
```

The last two lines are the point of the three tiers: for the same 8 mm, the metric side
spreads over 4–5 ticks while the collapsed side is a flat 4.

---

## Cell models are declarations too

```rust
pub struct Spec { model, tau_m, tau_a, theta, beta, gamma, refr_steps, dt, gain, act }

Spec::new(Model::AdLif)                       // leaky integrate-and-fire + adaptation
Spec::new(Model::Lif)                         // same integrator, beta = 0
Spec::new(Model::Rate).with_gain(0.05)        // rate unit (RNN-style), fires with p = act(v)
```

Constants, so they go straight into an area declaration:

```rust
const MID: AreaSpec = AreaSpec {
    topology: Topo::column(4, 8),
    neuron: Spec::new(Model::Rate).with_gain(0.01),   // this area is a rate population
    efferents: &MID_EFF,
    ..
};
```

`Model::Rate` is a **modelling choice, not a biological fact**: a real neuron does not emit
"probabilistic spikes". It exists so ANN-style units can be plugged into the same engine
for comparison, and therefore has to be declared explicitly.

---

## How similar frameworks do it

Rough comparison (from each project's papers/docs; orientation only):

| Framework | Cell abstraction | Topology / space | Extensibility |
|---|---|---|---|
| **NEST** | C++ "model" classes registered by name; params in a dict; `CopyModel` to derive | space is a plugin (`grid`, `free`, `3d`); positions are metadata, delays from connection rules | new models = C++ classes + recompile |
| **Arbor** | cell *catalogue* + mechanism catalogue; mechanism params per region | explicit morphology (cables); space **is** the morphology | mechanisms loaded by name |
| **Brian2** | equations → generated code (`NeuronGroup`); any ODE | no built-in space; distances are your arrays | equations are first class; codegen is the extension point |
| **Nengo** | `NeuronType` objects (`LIF`, `AdaptiveLIF`, …) | *ensembles* in a vector space; no anatomical map | neuron types are Python objects |
| **ANNarchy** | `Neuron` classes with ODE strings, code-generated | geometry optional (2-D/3-D), distance-dependent connectivity | ODE + projection patterns |
| **snnTorch / BindsNET** | `nn.Module` subclasses (`Leaky`, `LIF`) | none; tensors only | standard DL module registration |
| **Lava / Loihi** | process models, fixed-point arithmetic | chip mesh coordinates | vendor SDK |

Where `dreamkernel` differs — and where it does not:

* **Not novel**: a cell-model abstraction, "space as a plugin", declarative construction —
  all standard practice.
* **Different**: the same model switches **per area** between *no distance*, *collapsed
  column distance*, and *real millimetres*; projections are declared as
  "slot region → slot region + `Map`" and the builder generates synapses from that;
  construction-time validation reports geometry mismatches instead of silently broadcasting.
* **Deliberately missing** (for now): dendritic morphology, plasticity rules as plugins,
  distributed execution, hardware backends.

---

## Design principles

1. **Zero dependencies.** Geometry, maps, validation, and a tiny engine only.
2. **No implicit semantics.** `AllToAll` (broadcast) must be written explicitly; a collapsed
   topology must not fabricate distances.
3. **Report, don't panic.** Validation returns *all* problems; `Brain::declare` returns
   `Result` instead of aborting on a bad declaration.
4. **Incremental.** `Topo::POINT` is the default = no spatial structure = old behaviour;
   declaring something is opt-in.

## Licence

MIT OR Apache-2.0.
