# dreamkernel

**Declare a brain; get a brain.**

`dreamkernel` is a small, zero-dependency toolkit for describing neural structures as
declarations — area shapes (topologies), projection maps, and cell models — and for
**building a runnable spiking network from that declaration**.

> Status: `0.1.x`, research prototype. The public API is not frozen yet.
>
> 中文版见 [`README.zh-CN.md`](README.zh-CN.md)。

---

## Contents

- [Quick start](#declare--build--run) (three areas, running)
- [**Usage (complete)**](#usage-complete): field tables / **how to configure an area's topology** / slots & sub-regions / projections / cell templates / assembly / readout / validation errors / delay formula
  - [How to configure a topology](#2-how-to-configure-an-areas-topology-the-easy-place-to-get-wrong) ← **the easy place to get wrong**
- [One `Cell` template → a population](#one-cell-template--a-population-with-individual-differences)
- [Neuromodulators](#neuromodulators-are-named-not-a-single-arousal-scalar)
- [Assembly](#assembly-a-brain-is-put-together-from-parts)
- [How similar frameworks do it](#how-similar-frameworks-do-it)
- [Design principles](#design-principles)

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
use dreamkernel::core::cell::Modulators;
use dreamkernel::sim::{Brain, Config};

// ... declare SRC / MID / OUT (see examples/minimal.rs) ...

let mut brain = Brain::declare(&[SRC, MID, OUT], Config::default())?; // declaration → brain
brain.wire_local();                                                  // local recurrence
let mut ext = vec![0.0; brain.areas.len()];
ext[0] = 1.2;                                                        // drive the first area only
for _ in 0..300 { brain.step(&ext, &Modulators::default()); }
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

## Usage (complete)

From "runs in five minutes" to "configure the topology correctly", step by step.

### 0. Five minutes

```rust
use dreamkernel::sim::{Brain, Config};
use dreamkernel::core::cell::Modulators;

let mut brain = Brain::declare(&[SRC, MID], Config::default())?;   // declaration → brain
brain.wire_local();                                               // local recurrence
let ext = vec![1.2, 0.0];                                         // per-area drive
for _ in 0..300 { brain.step(&ext, &Modulators::default()); }
println!("SRC {} Hz", brain.rate_hz(0));
```

### 1. Fields of an area (`AreaSpec`)

| Field | Required | Meaning |
|---|---|---|
| `name` | ✓ | unique area name (instance-prefixed when assembled) |
| `class` | ✓ | `Cortex` / `ThalamicRelay{tonic}` / `SubcorticalRelay` / `PrimaryAfferent` / `Subcortical` / `Modulatory` / `Motor` — drives weight tier, background, homeostasis |
| `topo` | ✓ | modality label: `Visual` / `Somatic` / `Auditory` / `None` (only used to check who should be topographic with whom) |
| `micro` | ✓ | `Micro::cortical()/relay()/agranular()`: lamination + inhibitory fraction |
| `size` | ✓ | `Size::new(E, I, patches)` — **only `patches` is used**; E/I come from `Config` |
| `layout` | ✓ | `Layout::mni([x,y,z], width)`: centre and scale (width sets the half-extent used for distances) |
| `motif` | ✓ | local recurrence / feedforward / feedback strength |
| `slots` | ✓ | ports (see §3) |
| `topology` | ✓ | **the shape of the area** (see §2) |
| `efferents` | ✓ | outgoing projections (see §4); `&[]` = none |
| `neuron` | ✓ | cell template (see §5) |
| `type_specs` | ✓ | per-type templates `[P,PV,SOM,VIP]`; `&[]` = all four use `neuron` |

### 2. **How to configure an area's topology** (the easy place to get wrong)

A topology decides three things: (1) each neuron's coordinate on the "map"; (2) the index
order (`assign(n)` returns coordinates in index order, so index ↔ coordinate always agree);
(3) **whether coordinates may be read as distance** (`metric`, hence conduction delay).

| What you want to express | Write | `dims()` | `metric` | Behaviour |
|---|---|---|---|---|
| no space (modulatory / reticular nuclei) | `Topo::POINT` | 0 | `None` | every neuron at one point; no distance at all |
| one **ordered axis** (tonotopy, 1-D body map) | `Topo::LINE` | 1 | `Millimetres` | `u=(i+0.5)/n`; coordinates are millimetres |
| a **periodic** quantity (orientation 0°/180°, head direction) | `Topo::RING` | 1 | `Millimetres` | like `LINE` but `u` wraps (0 and 1 are neighbours) |
| a **2-D sheet** (retinotopy, 2-D body map) | `Topo::GRID` | 2 | `Millimetres` | `n` is laid out as a near-square grid |
| `n` **discrete modules** (barrel cortex) | `Topo::modules(n)` | 1 | `Millimetres` | one position per module, no structure inside |
| a **cortical column**: radial × tangential, distance **collapsed** | `Topo::column(radial, tangential)` | 2 | **`Collapsed`** | laid out as `tangential × radial`; **produces no conduction delay** |
| a **recursive grid** (hypercolumn: a sub-topology inside each tile) | `Topo::tiled(Kind::Grid, 8, 8, Kind::Ring)` | 2 | `Millimetres` | 8×8 outer tiles, each containing a sub-topology |

Three things to keep in mind:

1. **`LINE` / `GRID` turn positional differences into delay**; `column` does **not**
   (inside a column "next to" is topological, not millimetres — treating it as metric
   fabricates a delay). Getting this wrong produces no error, just a few silent milliseconds.
2. The argument order of `column(radial, tangential)` is **radial first**: `u` is tangential,
   `v` is radial, i.e. `column(6, 12)` = 6 layers deep × 12 tangential positions.
3. `assign(n)` lays `n` neurons onto the shape — **the shape comes first, the count second**:
   the same `GRID` works for 12 neurons or 12000.

```rust
topology: Topo::LINE,                        // S1 parcel: 1-D homunculus unfolding
topology: Topo::column(6, 12),               // V1: 6 layers × 12 orientation columns (collapsed)
topology: Topo::tiled(Kind::Grid, 8, 8, Kind::Ring),   // hypercolumn: position × orientation ring
topology: Topo::POINT,                       // relay / modulatory nuclei
```


### 2.1 The seven shapes, measured (`cargo run --example topologies`)

```
area       Topo       tiled dims       metric    shape  first…last coordinate
PointA     Point      -       0         None   (9, 1)  (0.50,0.50)…(0.50,0.50)
LineB      Line       -       1  Millimetres   (9, 1)  (0.06,0.50)…(0.94,0.50)
RingC      Ring       -       1  Millimetres   (9, 1)  (0.06,0.50)…(0.94,0.50)
GridD      Grid       -       2  Millimetres   (3, 3)  (0.17,0.17)…(0.83,0.83)
ModulesE   Modules    -       1  Millimetres   (9, 1)  (0.06,0.50)…(0.94,0.50)
ColumnF    Grid       -       2    Collapsed   (3, 3)  (0.12,0.08)…(0.12,0.42)
TiledG     Grid       yes     2  Millimetres   (3, 3)  (0.17,0.17)…(0.83,0.83)
```

Look at two things: **`ColumnF`'s metric is `Collapsed`** (positions inside a column produce
no delay), and **`TiledG`'s tiled flag is `yes`** (outer Grid, Ring inside each tile).

The same `GRID` holds 7 neurons or 48 — **shape first, count second**:

```
GRID with 7 neurons  → shape (3, 3), first coordinate (0.1667, 0.1667)
GRID with 12 neurons → shape (4, 3), first coordinate (0.1250, 0.1667)
GRID with 48 neurons → shape (7, 7), first coordinate (0.0714, 0.0714)
```

### 3. Slots and sub-regions (`Slot`)

A slot is a port of the area. **Sub-regions must be declared on slots**, otherwise a
projection cannot know which part to hit:

```rust
static SLOTS: [Slot; 2] = [
    // afferent slot receiving only the "lower body" half (`Rect` in this area's own frame)
    Slot::afferent("lower_body", TopoDomain::Somatic).with_region(Rect::of_u(0.0, 0.5)),
    Slot::efferent("out", TopoDomain::Somatic),
];
```

⚠️ **A `Rect` is expressed in the coordinate frame of the area that declares it.** The same
"lower body" has different numbers in different areas:

| Area | Its own axis | lower body |
|---|---|---|
| DRG (dermatome order) | lumbosacral → cervicothoracic | `[0.00, 0.50)` |
| VPL (body-map position) | lateral → medial | `[0.56, 1.00)` |

The two ends are aligned by `Map`, **not by equal numbers**.

### 4. Projections (`Efferent`)

```rust
Efferent {
    to: "MID", from_slot: "out", to_slot: "in",
    projection: Projection::Feedforward,
    map: Map::Identity, w: 2.0, fan_in: Some(24),
}
// shorthand:
Efferent::topo("MID", "out", "in", Projection::Feedforward, 2.0, 24)
```

| `Map` | Meaning | Used for |
|---|---|---|
| `Identity` | coordinates map directly (order preserving) | retina→V1, body map→VPL |
| `Affine{scale, offset, mirror}` | scale / shift / per-axis mirror | mirroring one side, local magnification |
| `Permute{seed}` | geometry unchanged, **source labels shuffled** | **control** (proves order preservation does something) |
| `AllToAll` | broadcast (non-topographic) | modulatory nuclei; must be written **explicitly** |

Wiring rule: both ends have a topology and `map` is order-preserving → each target picks
the nearest `fan_in` sources by coordinate; otherwise broadcast with `fan_in`.

### 5. Cell templates and types (`Spec`)

See the next section. In short: `neuron` is the template, `type_specs` gives per-type
parameters, `tau_spread` creates individual differences, `mod_gain` decides which
neuromodulators act on the cell.

### 6. Assembly (`Module` / `Assembly`)

```rust
let mut asm = Assembly::new();
asm.add(&RELAY_MOD, "L").add(&RELAY_MOD, "R")
   .wire("L.out", "R.in", Projection::Feedforward, Map::Identity, 2.0, Some(12));
let specs = asm.expand()?;                    // names become "L:RELAY" / "R:RELAY"
let mut brain = Brain::declare(&specs, Config::default())?;
```

Rules: projections inside a part may only target areas of the same part; crossing a part
boundary must go through a port; instance names and expanded names must be unique.

### 7. Running and reading out

| API | Meaning |
|---|---|
| `Brain::declare(&[AreaSpec], Config)` | declaration → brain; `Err(Vec<String>)` lists every problem (no panic) |
| `brain.wire_local()` | local recurrence inside each area (separate switch) |
| `brain.step(&ext, &Modulators)` | advance one step; `ext` is per-area drive |
| `brain.rate_hz(i)` | mean firing rate (Hz) of area `i` since construction |
| `brain.edges` | synapses generated per declared projection (observable) |
| `brain.incoming` / `delay` / `i_bias` / `mod_gain` | per-neuron views (debugging) |

### 8. What validation reports (cheat sheet)

| Message | Meaning |
|---|---|
| `[区重名]` | duplicate area name in one declaration |
| `[缺靶区]` / `[缺槽]` | an `Efferent` refers to a missing area/slot |
| `[区域越界]` / `[区域无形状]` | invalid slot sub-region, or a sub-region on an area with no topology |
| `[实例重名]` / `[连线无端口]` / `[连线方向不符]` / `[组内投射越界]` | assembly-level structural errors |

### 9. How delay is computed

```
delay = 1 ms (synaptic transmission) + distance / 10 mm·ms⁻¹
distance = area-centre distance + intra-area positional difference
           (the intra-area term is added **only for a side whose metric is Millimetres**)
then quantised by dt into ticks, clamped to 1..16
```

So at `dt = 0.5 ms`, two areas 8 mm apart give ≈4 ticks; the `column` side contributes
no positional term.

---

## One `Cell` template → a population with individual differences

Every knob that matters for "what kind of neuron is this" is a field on `Spec`, so a
**cell type is just a parameter set** — no engine change needed:

| Field | Meaning |
|---|---|
| `model` | `AdLif` / `Lif` (same integrator, β=0) / `Rate` |
| `tau_m`, `tau_a` | membrane / adaptation time constant (ms) |
| `theta`, `beta`, `gamma` | threshold, adaptation strength, threshold gain from adaptation |
| `refr_steps`, `dt` | refractory steps, timestep |
| `i_bias` | resting drive ("this neuron leans excitable/inhibitory") |
| `tau_spread` | **individual differences**: lognormal jitter of τ when one template builds a population |
| `mod_gain: [DA, NE, ACh, 5-HT]` | how much each neuromodulator feeds this neuron's input |
| `gain`, `act` | rate-unit slope / activation (`Relu` by default: silence at rest) |

Per-type templates go in the area declaration (`type_specs = [P, PV, SOM, VIP]`;
empty = the `neuron` template for all four):

```rust
static PV_FAST: Spec = Spec::new(Model::Lif).with_tau_m(8.0).with_theta(0.6);
// type_specs: &[P_SLOW, PV_FAST, SOM, VIP]   → 类型就是这四组参数
```

Measured (`type_specs_make_types_by_parameter`): identical input, PV fires
**4.3×** more spikes per neuron than pyramidal (1000 spikes / 10 PV vs 368 / 16 P).

## Neuromodulators are named, not a single "arousal" scalar

```rust
let mods = Modulators::new(/*DA*/ 0.0, /*NE*/ 0.0, /*ACh*/ 0.0, /*5-HT*/ 0.0);
brain.step(&ext, &mods);
```

Effect = `Σ mods[k] · Spec::mod_gain[k]` added to the input current. Measured
(`cargo run --example minimal`, SRC declares `mod_gain=[0.8,0,0,0]`):

```
DA=0.0  SRC  193.3 Hz   MID   83.0 Hz   OUT  326.2 Hz
DA=0.5  SRC  273.0 Hz   MID  115.5 Hz   OUT  395.7 Hz
DA=1.0  SRC  343.5 Hz   MID  141.7 Hz   OUT  433.8 Hz
```

A population with `mod_gain = 0` is bit-identical under any DA level (asserted).

> **Simplification, stated openly**: real dopamine acts through D1/D2 with opposite
> signs, and ACh changes the inhibitory network rather than "adding current". This layer
> only models *input gain per modulator per cell type*; receptor-level detail belongs to
> the concrete brain, not the framework.

---

## Assembly: a brain is put together from parts

A single `AreaSpec` is *one area*. What you usually want is a **reusable part** — e.g.
"thalamic relay = relay nucleus + local interneurons + one internal projection" —
instantiated several times (left/right, V1/V2) and then wired through **ports**.

```rust
let mut asm = Assembly::new();
asm.add(&RELAY_MOD, "L")
   .add(&RELAY_MOD, "R")
   .wire("L.out", "R.in", Projection::Feedforward, Map::Identity, 2.0, Some(12));

let specs = asm.expand()?;          // names become "L:RELAY" / "R:RELAY"
let mut brain = Brain::declare(&specs, Config::default())?;
```

```bash
cargo run --example assembly
# 组装：2 个实例、2 个区、1 条跨部件连线
# 展开后 2 个区： L:RELAY（出向 1 条） / R:RELAY（出向 0 条）
# L:RELAY → R:RELAY  288 条突触
# t=300ms  L  291.6 Hz   R  229.3 Hz
```

Two hard rules, both testable:

1. **A part's internal projections may only target areas of the same part.** Anything
   crossing a part boundary must go through a `Port` — otherwise instantiating the part
   twice silently points at the other instance.
2. Instance names must be unique, and so must the expanded area names (`instance:area`).

---

## Real fibre paths, and delays without a delay array

Two things the slot geometry cannot express on its own:

**1. A connection is not a straight line.** `Path` says how the fibre actually runs:

```rust
Path::Straight                     // default
Path::Tortuous(1.4)                // straight × tortuosity (axons are not straight; 1.2–1.6 typical)
Path::Via([0.0, 12.0, 0.0])        // through a waypoint: corpus callosum over the midline, around a nucleus
Path::Fixed(60.0)                  // measured fibre length in mm (geometry ignored)
```

Measured with two areas 8 mm apart (`dt = 0.5 ms`):

```
直 4 tick   迂曲(1.6) 5 tick   中转(上方 12mm) 8 tick   实测 60mm 14 tick
```

**2. Delays need not be stored.** Delay is a function of physical coordinates, so
`Config::delay_mode` offers two strategies:

| Mode | What is stored | Memory |
|---|---|---|
| `Stored` (default) | one `u8` per synapse | 1 byte / synapse |
| `Derived` | nothing; recomputed from `(area-pair distance, both coords)` each step | **0 bytes** |

```rust
Config { delay_mode: DelayMode::Derived, ..Default::default() }
brain.delay_bytes();   // 0   ← the whole delay array is gone
```

Measured (`derived_delay_matches_stored_bit_for_bit`): both modes produce **identical spike
counts** over 300 steps (a bit-for-bit assertion — otherwise "saving memory" would just be
changing the model), while the delay array drops from 384 B to 0 B on that small network.
At the scale of the full brain (7×10⁷ synapses) that is ~70 MB.

Two stated constraints: `Derived` assumes **straight** fibres (a projection declaring
`Via/Tortuous/Fixed` falls back to storing its delays), and same-area synapses are a single
synaptic delay in both modes.


---

## Synapses are declarations too

A projection says *where* (`Efferent`); a [`Synapse`] says *what kind of synapse*:

```rust
use dreamkernel::link::synapse::{Plasticity, Synapse, WeightDist};

Synapse::FIXED                                  // default: fixed weight, no plasticity
Synapse::noisy(2.0, 0.3)                        // lognormal jitter (relative sd 0.3)
Synapse::stdp(2.0, 0.02, 0.021)                 // fixed mean + pair-based STDP
Synapse::stdp(2.0, 0.02, 0.021)
    .with_dist(WeightDist::Lognormal(0.3))      // both
```

**Two deliberate choices:**

1. **Frozen by default.** Plasticity needs both a declaration *and* `Config::plasticity = true`,
   so adding one declaration can never silently change an existing model.
2. **Lognormal, not Gaussian.** Synaptic weights are positive and right-skewed; a Gaussian
   would sample negative weights, which means *inhibitory* — a sign change, not noise.
   (Measured eventually-mean of `Lognormal(0.3)` is `exp(σ²/2) ≈ 1.046`, not exactly 1.)

STDP is the standard pair rule: `Δw = +a₊·x_pre` when the post fires, `Δw = −a₋·x_post` when
the pre fires; traces decay by `exp(−dt/τ)`; weights are clamped to `[0, w_max]`.
Measured (`stdp_potentiates_when_pre_leads`): mean incoming weight to the downstream area
**1.0000 → 1.3006** over 300 steps when the pre side leads.

## Blueprint tools: validate / describe / manifest / fingerprint

Everything you need **without running dynamics** — usable in CI, before start-up, in review:

```rust
use dreamkernel::blueprint::Blueprint;

let bp = Blueprint::new(&areas);
let problems = bp.validate();     // ALL problems, not the first
let summary  = bp.describe();     // human-readable: classes, three metric tiers, slots, synapses
let json     = bp.manifest_json();// machine-readable (world / visualiser / external server)
let (ah, eh) = bp.fingerprint();  // structure fingerprint
```

`validate` reports duplicate names, missing targets, missing/wrong-direction slots,
sub-regions on shapeless areas, geometry mismatches (1-D target vs 2-D source), and
non-topographic broadcast over a modality slot. `fingerprint` is order-sensitive: a
representation-only refactor keeps it identical, any semantic change moves it.


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
