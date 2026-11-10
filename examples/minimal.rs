//! **最小示例：写一份声明，得到一个能跑的脑。**
//!
//! ```text
//! cargo run --example minimal
//! ```
//!
//! 三个区：`SRC`（一维拓扑的中继）→ `MID`（二维薄片）→ `OUT`，
//! 两条投射都**由声明生成**（源/靶子区域从槽的 `region` 取，两端用 `Map` 对齐）。
//! 然后给 `SRC` 一点驱动电流，看发放怎么传到下游、以及延迟怎么随距离变。

use dreamkernel::core::area::{AreaClass, AreaSpec, Layout, LocalMotif, Micro, Size, Slot, TopoDomain};
use dreamkernel::core::topo::Topo;
use dreamkernel::link::projection::{Efferent, Projection};
use dreamkernel::sim::{Brain, Config};

static SRC_SLOTS: [Slot; 1] = [Slot::efferent("out", TopoDomain::Somatic)];
static MID_SLOTS: [Slot; 2] = [
    Slot::afferent("in", TopoDomain::Somatic),
    Slot::efferent("out", TopoDomain::Visual),
];
static OUT_SLOTS: [Slot; 1] = [Slot::afferent("in", TopoDomain::Visual)];

static SRC_EFF: [Efferent; 1] =
    [Efferent::topo("MID", "out", "in", Projection::Feedforward, 2.0, 24)];
static MID_EFF: [Efferent; 1] = [Efferent::topo(
    "OUT",
    "out",
    "in",
    Projection::Feedforward,
    2.0,
    24,
)];

const SRC: AreaSpec = AreaSpec {
    name: "SRC",
    class: AreaClass::ThalamicRelay { tonic: true },
    topo: TopoDomain::Somatic,
    micro: Micro::relay(),
    size: Size::new(48, 12, 1),
    layout: Layout::mni([0.0, 0.0, 0.0], 1.0),
    motif: LocalMotif::relay(),
    slots: &SRC_SLOTS,
    // ③ 1:1 米制：坐标就是毫米，距离会变成传导延迟
    topology: Topo::LINE,
    efferents: &SRC_EFF,
};

const MID: AreaSpec = AreaSpec {
    name: "MID",
    class: AreaClass::Cortex,
    topo: TopoDomain::Visual,
    micro: Micro::cortical(),
    size: Size::new(48, 12, 1),
    layout: Layout::mni([8.0, 0.0, 0.0], 0.6),
    motif: LocalMotif::cortical(),
    slots: &MID_SLOTS,
    // ② 柱状：切向 8 × 径向 4，距离**坍缩**（不该凭空造出延迟）
    topology: Topo::column(4, 8),
    efferents: &MID_EFF,
};

const OUT: AreaSpec = AreaSpec {
    name: "OUT",
    class: AreaClass::Cortex,
    topo: TopoDomain::None,
    micro: Micro::cortical(),
    size: Size::new(48, 12, 1),
    layout: Layout::mni([16.0, 0.0, 0.0], 0.6),
    motif: LocalMotif::cortical(),
    slots: &OUT_SLOTS,
    topology: Topo::POINT,
    efferents: &[],
};

fn main() {
    let specs = [SRC, MID, OUT];
    let cfg = Config {
        n_exc: 48,
        n_inh: 12,
        local_frac: 0.2,
        seed: 7,
        dt_ms: 0.5,
        ext_base: 0.0,
        ..Default::default()
    };

    // ---- 1. 声明 → 脑 ----
    let mut brain = match Brain::declare(&specs, cfg.clone()) {
        Ok(b) => b,
        Err(errs) => {
            eprintln!("声明有 {} 条问题：", errs.len());
            for e in errs {
                eprintln!("  {e}");
            }
            std::process::exit(1);
        }
    };
    brain.wire_local(); // 同区局部递归（不属于"跨区声明"，单独开关）

    println!("神经元 {} 个，共 {} 条投射：", brain.n_neurons(), brain.edges.len());
    for e in &brain.edges {
        println!(
            "  {} → {}（{:?}）{} 条突触",
            brain.areas[e.from_area].name, brain.areas[e.to_area].name, e.projection, e.synapses
        );
    }

    // ---- 2. 跑起来 ----
    let mut ext = vec![0.0f32; brain.areas.len()];
    ext[0] = 1.2; // 只驱动 SRC
    for t in 0..300 {
        brain.step(&ext, 1.0);
        if t % 100 == 99 {
            println!(
                "t={:>3}ms  SRC {:>6.1} Hz   MID {:>6.1} Hz   OUT {:>6.1} Hz",
                t + 1,
                brain.rate_hz(0),
                brain.rate_hz(1),
                brain.rate_hz(2)
            );
        }
    }

    // ---- 3. 延迟确实来自几何（只看**跨区**突触，区内局部边恒 1 tick）----
    let cross = |brain: &Brain, area: usize| -> Vec<u8> {
        let (s, e) = (brain.area_start[area], brain.area_start[area + 1]);
        let mut v: Vec<u8> = Vec::new();
        for post in s..e {
            for (k, &(pre, _)) in brain.incoming[post].iter().enumerate() {
                if brain.area_of[pre] != area {
                    v.push(brain.delay[post][k]);
                }
            }
        }
        v.sort_unstable();
        v
    };
    for (name, idx, note) in [
        ("MID", 1usize, "SRC 有米制：跨区 8mm + 区内位置差"),
        ("OUT", 2usize, "MID 是坍缩档：区内位置不计入距离，只剩跨区 8mm"),
    ] {
        let d = cross(&brain, idx);
        if !d.is_empty() {
            println!(
                "{name} 的**跨区**入边延迟：{}–{} tick（{}）",
                d[0],
                d[d.len() - 1],
                note
            );
        }
    }
}
