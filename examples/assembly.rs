//! **组装**：同一份部件实例化两次、连线、跑起来。
//!
//! ```text
//! cargo run --example assembly
//! ```
//!
//! 部件 `relay` 只有一个中继区（有入/出两个端口）。把它实例化成 `L` 和 `R`，
//! 再把 `L.out` 接到 `R.in` —— 展开后是两个区（`L:RELAY` / `R:RELAY`），
//! 跨部件连线变成 `L:RELAY → R:RELAY` 的一条出向投射。
//!
//! 这就是"脑是组装起来的"：**部件可复用、名字带实例前缀、连接走端口**。

use dreamkernel::assembly::{Assembly, Module, Port};
use dreamkernel::core::area::{AreaClass, AreaSpec, Layout, LocalMotif, Micro, Size, Slot, TopoDomain};
use dreamkernel::core::cell::{Model, Spec};
use dreamkernel::core::topo::{Map, Topo};
use dreamkernel::link::projection::{Efferent, Projection};
use dreamkernel::sim::{Brain, Config};

const SRC_DOSE: f32 = 6.0;

static SLOTS: [Slot; 2] = [
    Slot::afferent("in", TopoDomain::Somatic),
    Slot::efferent("out", TopoDomain::Somatic),
];

/// 组内不声明任何投射：这个部件只有一个区。
static NO_EFF: [Efferent; 0] = [];

const RELAY: AreaSpec = AreaSpec {
    name: "RELAY",
    class: AreaClass::ThalamicRelay { tonic: true },
    topo: TopoDomain::Somatic,
    micro: Micro::relay(),
    size: Size::new(24, 6, 1),
    layout: Layout::mni([0.0, 0.0, 0.0], 0.5),
    motif: LocalMotif::relay(),
    slots: &SLOTS,
    // ③ 一维米制：坐标就是毫米，位置差会变成传导延迟
    topology: Topo::LINE,
    efferents: &NO_EFF,
    neuron: Spec::new(Model::AdLif),
};

static AREAS: [AreaSpec; 1] = [RELAY];
static PORTS: [Port; 2] = [
    Port::inp("in", "RELAY", "in"),
    Port::out("out", "RELAY", "out"),
];
static RELAY_MOD: Module = Module { name: "relay", areas: &AREAS, ports: &PORTS };

fn main() {
    // ---- 1. 组装 ----
    let mut asm = Assembly::new();
    asm.add(&RELAY_MOD, "L")
        .add(&RELAY_MOD, "R")
        .wire("L.out", "R.in", Projection::Feedforward, Map::Identity, 2.0, Some(12));
    println!("{}", asm.summary());
    for e in asm.validate() {
        eprintln!("  {e}");
    }

    // ---- 2. 展开成扁平声明 ----
    let specs = match asm.expand() {
        Ok(v) => v,
        Err(errs) => {
            eprintln!("组装有 {} 条问题：", errs.len());
            for e in errs {
                eprintln!("  {e}");
            }
            std::process::exit(1);
        }
    };
    println!("展开后 {} 个区：", specs.len());
    for s in &specs {
        println!(
            "  {:<10} 拓扑 {:?}  出向 {} 条",
            s.name,
            s.topology.kind,
            s.efferents.len()
        );
    }

    // ---- 3. 跑 ----
    let cfg = Config { n_exc: 24, n_inh: 6, dt_ms: 1.0, seed: 5, ..Default::default() };
    let mut brain = Brain::declare(&specs, cfg).expect("展开出的声明应当合法");
    for e in &brain.edges {
        println!(
            "  {} → {}  {} 条突触",
            brain.areas[e.from_area].name, brain.areas[e.to_area].name, e.synapses
        );
    }
    let ext = vec![SRC_DOSE, 0.0];
    for t in 0..300 {
        brain.step(&ext, 1.0);
        if t % 100 == 99 {
            println!(
                "t={:>3}ms  L {:>6.1} Hz   R {:>6.1} Hz",
                t + 1,
                brain.rate_hz(0),
                brain.rate_hz(1)
            );
        }
    }
}
