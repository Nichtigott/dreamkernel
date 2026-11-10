//! **拓扑目录**：七种区形状各来一个，打印它们的元信息并真的接起来跑。
//!
//! ```text
//! cargo run --example topologies
//! ```
//!
//! 这是 README「区拓扑怎么设」那一节的可执行版本——如果这里跑不通，说明文档是假的。

use dreamkernel::assembly::{Assembly, Module, Port};
use dreamkernel::core::area::{
    AreaClass, AreaSpec, Layout, LocalMotif, Micro, Size, Slot, TopoDomain,
};
use dreamkernel::core::cell::{Model, Modulators, Spec};
use dreamkernel::core::topo::{Coord, Kind, Rect, Topo};
use dreamkernel::link::projection::{Efferent, Projection};
use dreamkernel::sim::{Brain, Config};

// ---- 每个区的槽：一个入、一个出 ----
static IN_OUT: [Slot; 2] = [
    Slot::afferent("in", TopoDomain::Somatic),
    Slot::efferent("out", TopoDomain::Somatic),
];
/// 只有 S1 parcel 需要"只接下半身"的子区域
static HALF: [Slot; 2] = [
    Slot::afferent("in", TopoDomain::Somatic).with_region(Rect::of_u(0.0, 0.5)),
    Slot::efferent("out", TopoDomain::Somatic),
];

/// 每个区都发往下一个区（链式）
macro_rules! chain_eff {
    ($next:literal) => {
        &[Efferent::topo($next, "out", "in", Projection::Feedforward, 2.0, 8)]
    };
}

const POINT_A: AreaSpec = AreaSpec {
    name: "PointA",
    class: AreaClass::Modulatory,
    topo: TopoDomain::None,
    micro: Micro::agranular(),
    size: Size::new(12, 4, 1),
    layout: Layout::mni([0.0, 0.0, 0.0], 0.3),
    motif: LocalMotif::relay(),
    slots: &IN_OUT,
    topology: Topo::POINT,
    efferents: chain_eff!("LineB"),
    neuron: Spec::new(Model::Lif).with_bias(1.2),
    type_specs: &[],
};

const LINE_B: AreaSpec = AreaSpec {
    name: "LineB",
    class: AreaClass::SubcorticalRelay,
    topo: TopoDomain::Somatic,
    micro: Micro::relay(),
    size: Size::new(12, 4, 1),
    layout: Layout::mni([4.0, 0.0, 0.0], 1.0),
    motif: LocalMotif::relay(),
    slots: &HALF, // 只接下半身（本区坐标系里的 [0,0.5)）
    topology: Topo::LINE,
    efferents: chain_eff!("RingC"),
    neuron: Spec::new(Model::AdLif),
    type_specs: &[],
};

const RING_C: AreaSpec = AreaSpec {
    name: "RingC",
    class: AreaClass::Cortex,
    topo: TopoDomain::Visual,
    micro: Micro::cortical(),
    size: Size::new(12, 4, 1),
    layout: Layout::mni([8.0, 0.0, 0.0], 1.0),
    motif: LocalMotif::cortical(),
    slots: &IN_OUT,
    topology: Topo::RING,
    efferents: chain_eff!("GridD"),
    neuron: Spec::new(Model::AdLif),
    type_specs: &[],
};

const GRID_D: AreaSpec = AreaSpec {
    name: "GridD",
    class: AreaClass::Cortex,
    topo: TopoDomain::Visual,
    micro: Micro::cortical(),
    size: Size::new(12, 4, 1),
    layout: Layout::mni([12.0, 0.0, 0.0], 1.0),
    motif: LocalMotif::cortical(),
    slots: &IN_OUT,
    topology: Topo::GRID,
    efferents: chain_eff!("ModulesE"),
    neuron: Spec::new(Model::AdLif),
    type_specs: &[],
};

const MODULES_E: AreaSpec = AreaSpec {
    name: "ModulesE",
    class: AreaClass::Cortex,
    topo: TopoDomain::Somatic,
    micro: Micro::cortical(),
    size: Size::new(12, 4, 1),
    layout: Layout::mni([16.0, 0.0, 0.0], 1.0),
    motif: LocalMotif::cortical(),
    slots: &IN_OUT,
    topology: Topo::modules(4),
    efferents: chain_eff!("ColumnF"),
    neuron: Spec::new(Model::AdLif),
    type_specs: &[],
};

const COLUMN_F: AreaSpec = AreaSpec {
    name: "ColumnF",
    class: AreaClass::Cortex,
    topo: TopoDomain::Visual,
    micro: Micro::cortical(),
    size: Size::new(12, 4, 1),
    layout: Layout::mni([20.0, 0.0, 0.0], 1.0),
    motif: LocalMotif::cortical(),
    slots: &IN_OUT,
    // 6 层深 × 4 个切向位置；距离坍缩（不产生传导延迟）
    topology: Topo::column(6, 4),
    efferents: chain_eff!("TiledG"),
    neuron: Spec::new(Model::AdLif),
    type_specs: &[],
};

const TILED_G: AreaSpec = AreaSpec {
    name: "TiledG",
    class: AreaClass::Cortex,
    topo: TopoDomain::Visual,
    micro: Micro::cortical(),
    size: Size::new(12, 4, 1),
    layout: Layout::mni([24.0, 0.0, 0.0], 1.0),
    motif: LocalMotif::cortical(),
    slots: &IN_OUT,
    // 外层 2×2 格，每格内再铺一个 Ring（位置 × 方位）
    topology: Topo::tiled(Kind::Grid, 2, 2, Kind::Ring),
    efferents: &[],
    neuron: Spec::new(Model::AdLif),
    type_specs: &[],
};

static AREAS: [AreaSpec; 7] = [
    POINT_A, LINE_B, RING_C, GRID_D, MODULES_E, COLUMN_F, TILED_G,
];
static PORTS: [Port; 2] = [
    Port::inp("in", "PointA", "in"),
    Port::out("out", "TiledG", "out"),
];
static CHAIN: Module = Module { name: "chain", areas: &AREAS, ports: &PORTS };

fn main() {
    // ---- 1. 拓扑元信息：七种形状各自长什么样 ----
    println!(
        "{:<10} {:<10} {:<4} {:>4} {:>12} {:>8}  首/末坐标",
        "区", "Topo", "tiled", "dims", "metric", "shape"
    );
    for a in AREAS.iter() {
        let t = a.topology;
        let c = t.assign(9);
        let first: Coord = c.first().copied().unwrap_or(Coord::new(0.0, 0.0));
        let last: Coord = c.last().copied().unwrap_or(Coord::new(0.0, 0.0));
        println!(
            "{:<10} {:<10} {:<4} {:>4} {:>12} {:>8}  ({:.2},{:.2})…({:.2},{:.2})",
            a.name,
            format!("{:?}", t.kind),
            if t.is_tiled() { "yes" } else { "-" },
            t.dims(),
            format!("{:?}", t.metric),
            format!("{:?}", t.shape(9)),
            first.u,
            first.v,
            last.u,
            last.v
        );
    }

    // ---- 2. 同一个形状，不同神经元数：形状先定、数量后配 ----
    for n in [7usize, 12, 48] {
        let t = Topo::GRID;
        println!(
            "GRID 铺 {n} 个神经元 → shape {:?}，首坐标 {:?}",
            t.shape(n),
            t.assign(n).first()
        );
    }

    // ---- 3. 真的接起来跑 ----
    let mut asm = Assembly::new();
    asm.add(&CHAIN, "c");
    if let Err(e) = asm.expand() {
        eprintln!("声明有 {} 条问题：{:?}", e.len(), e);
        std::process::exit(1);
    }
    // 单实例即可：链式投射已经在各区 efferents 里
    let specs: Vec<AreaSpec> = AREAS.to_vec();
    let cfg = Config { n_exc: 12, n_inh: 4, dt_ms: 1.0, seed: 11, ..Default::default() };
    let mut brain = Brain::declare(&specs, cfg).expect("七种拓扑都应当能建");
    brain.wire_local();
    let mut ext = vec![0.0f32; brain.areas.len()];
    ext[0] = 4.0;
    for _ in 0..300 {
        brain.step(&ext, &Modulators::default());
    }
    println!("\n跑了 300 步（只驱动 PointA）：");
    for i in 0..brain.areas.len() {
        println!("  {:<10} {:>7.1} Hz", brain.areas[i].name, brain.rate_hz(i));
    }
    println!("共 {} 条边（由 efferents 生成）", brain.edges.len());
}
