//! **组装**：把"脑"拆成可复用的部件（module），再实例化、连线、展开成一份扁平声明。
//!
//! 单个 `AreaSpec` 只是"一个区"；真实建模时你想要的通常是"一个**部件**"——
//! 比如「丘脑中继模块 = 中继核 + 局部中间神经元 + 一条组内投射」，
//! 然后把它实例化两次（左/右、V1/V2），再用端口把实例连起来。
//!
//! ```text
//! Module{ areas, ports }  --instantiate("L")-->  area 名加前缀（"L:RELAY"）
//!        ↑                                              ↓
//!     Assembly::wire("L.out", "R.in")        expand() → Vec<AreaSpec> → Brain::declare
//! ```
//!
//! 两条硬规矩（都能变红）：
//! 1. **部件内部的投射只能指向同部件的区**——跨部件的连接必须走端口（`wire`），
//!    否则实例化两次时组内引用会指向另一个实例，静默接错。
//! 2. 实例名必须唯一；展开后区名（`实例:区`）也必须唯一。
//!
//! 名字要在运行时生成（`L:RELAY`），而 `AreaSpec` 的字段是 `&'static str`。
//! 这里用 [`Box::leak`] 泄漏一次性字符串：组装只在启动时发生一次、总量是几百个短串，
//! 换来的是**不用改 `AreaSpec` 的类型**（`const` 声明与运行时组装共用同一套类型）。

use crate::core::area::{AreaSpec, SlotKind};
#[cfg(test)]
use crate::core::area::TopoDomain;
use crate::core::topo::Map;
use crate::link::projection::{Efferent, Projection};

/// 端口方向。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    /// 入：外部可以接到这里（对应区的一个入向槽）。
    In,
    /// 出：这里的信号可以接出去（对应区的一个出向槽）。
    Out,
}

/// 一个**部件端口**：把「本部件里某个区的某个槽」暴露给外部。
#[derive(Clone, Copy, Debug)]
pub struct Port {
    /// 端口名（端口在部件内唯一）。
    pub name: &'static str,
    /// 本部件里的区名。
    pub area: &'static str,
    /// 该区的槽名。
    pub slot: &'static str,
    pub dir: Dir,
}

impl Port {
    pub const fn out(name: &'static str, area: &'static str, slot: &'static str) -> Self {
        Self { name, area, slot, dir: Dir::Out }
    }
    pub const fn inp(name: &'static str, area: &'static str, slot: &'static str) -> Self {
        Self { name, area, slot, dir: Dir::In }
    }
}

/// 一个可复用的部件：一组区 + 一组端口。
///
/// 组内投射**不**单独写：它就在各区的 `efferents` 里（指向同部件的区名）。
#[derive(Clone, Copy, Debug)]
pub struct Module {
    pub name: &'static str,
    pub areas: &'static [AreaSpec],
    pub ports: &'static [Port],
}

impl Module {
    fn area(&self, name: &str) -> Option<&AreaSpec> {
        self.areas.iter().find(|a| a.name == name)
    }
    fn port(&self, name: &str) -> Option<&Port> {
        self.ports.iter().find(|p| p.name == name)
    }
}

/// 一条**跨部件**连接：从某实例的出口到另一实例的入口。
#[derive(Clone, Copy, Debug)]
struct Wire {
    from: (usize, &'static str),
    to: (usize, &'static str),
    projection: Projection,
    map: Map,
    w: f32,
    fan_in: Option<usize>,
    synapse: crate::link::synapse::Synapse,
}

/// 组装：若干部件实例 + 它们之间的连线。
#[derive(Default, Debug)]
pub struct Assembly {
    /// `(module, 实例名)`。
    insts: Vec<(&'static Module, &'static str)>,
    wires: Vec<Wire>,
}

impl Assembly {
    pub fn new() -> Self {
        Self::default()
    }

    /// 实例化一个部件。`instance` 是实例名前缀（展开后区名 = `实例:区`）。
    pub fn add(&mut self, module: &'static Module, instance: &'static str) -> &mut Self {
        self.insts.push((module, instance));
        self
    }

    /// 把 `from`（`"实例.端口"`）接到 `to`（`"实例.端口"`）。
    ///
    /// `map` 决定两端拓扑怎么对齐；`fan_in = None` 用引擎默认。
    pub fn wire(
        &mut self,
        from: &str,
        to: &str,
        projection: Projection,
        map: Map,
        w: f32,
        fan_in: Option<usize>,
    ) -> &mut Self {
        // 字符串先泄漏成 'static，供 `Wire` 存（组装期一次性）
        let f = leak(from);
        let t = leak(to);
        self.wires.push(Wire {
            from: (0, f),
            to: (0, t),
            projection,
            map,
            w,
            fan_in,
            synapse: crate::link::synapse::Synapse::FIXED,
        });
        self
    }

    fn inst_index(&self, name: &str) -> Option<usize> {
        self.insts.iter().position(|(_, n)| *n == name)
    }

    /// 校验：实例名唯一、端口存在且方向对、部件内部投射不越界。
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        // ① 实例名唯一 + 展开后区名唯一
        let mut seen_inst: Vec<&str> = Vec::new();
        let mut seen_area: Vec<String> = Vec::new();
        for (m, inst) in &self.insts {
            if seen_inst.contains(inst) {
                errs.push(format!("[实例重名] {inst}"));
            }
            seen_inst.push(inst);
            for a in m.areas {
                let full = format!("{inst}:{}", a.name);
                if seen_area.contains(&full) {
                    errs.push(format!("[展开区名重复] {full}"));
                }
                seen_area.push(full);
            }
        }
        // ② 部件内部投射只能指向同部件
        for (m, inst) in &self.insts {
            for a in m.areas {
                for ef in a.efferents {
                    if m.area(ef.to).is_none() {
                        errs.push(format!(
                            "[组内投射越界] 部件 {} 的 {}::{} 想发到 {}（不在本部件内；跨部件请用端口）",
                            m.name, inst, a.name, ef.to
                        ));
                    }
                }
            }
        }
        // ③ 端口指向的区/槽必须存在，方向要与槽的方向一致
        for (m, inst) in &self.insts {
            for p in m.ports {
                let Some(a) = m.area(p.area) else {
                    errs.push(format!("[端口无区] {}::{} 指向不存在的区 {}", inst, p.name, p.area));
                    continue;
                };
                let Some(s) = a.slots.iter().find(|s| s.name == p.slot) else {
                    errs.push(format!(
                        "[端口无槽] {}::{} 指向 {} 没有的槽 {}",
                        inst, p.name, p.area, p.slot
                    ));
                    continue;
                };
                let want = match p.dir {
                    Dir::Out => SlotKind::Efferent,
                    Dir::In => SlotKind::Afferent,
                };
                if s.kind != want {
                    errs.push(format!(
                        "[端口方向不符] {}::{} 声明 {:?}，但槽 {} 是 {:?}",
                        inst, p.name, p.dir, p.slot, s.kind
                    ));
                }
            }
        }
        // ④ 连线：实例存在、端口存在、方向对
        for w in &self.wires {
            for (end, want) in [(w.from, Dir::Out), (w.to, Dir::In)] {
                let (inst_name, port_name) = split(end.1);
                let Some(idx) = self.inst_index(inst_name) else {
                    errs.push(format!("[连线无实例] {}", inst_name));
                    continue;
                };
                let (m, _) = self.insts[idx];
                match m.port(port_name) {
                    None => errs.push(format!("[连线无端口] {}.{}", inst_name, port_name)),
                    Some(p) if p.dir != want => errs.push(format!(
                        "[连线方向不符] {}.{} 应当是 {:?}",
                        inst_name, port_name, want
                    )),
                    Some(_) => {}
                }
            }
        }
        errs
    }

    /// **展开**成一份扁平区表（区名带实例前缀，组内投射改写成本实例，跨部件连线变成新的出向投射）。
    ///
    /// 返回 `Err` 时不做任何建脑动作（校验先跑）。
    pub fn expand(&self) -> Result<Vec<AreaSpec>, Vec<String>> {
        let errs = self.validate();
        if !errs.is_empty() {
            return Err(errs);
        }
        // 按 (实例, 区) 收集跨部件追加的出向投射
        let mut extra: Vec<(usize, &'static str, Vec<Efferent>)> = Vec::new();
        for w in &self.wires {
            let ((fi, fp), (ti, tp)) = (split(w.from.1), split(w.to.1));
            let (Some(fi), Some(ti)) = (self.inst_index(fi), self.inst_index(ti)) else { continue };
            let (fm, _fname) = self.insts[fi];
            let (tm, tname) = self.insts[ti];
            let (Some(fport), Some(tport)) = (fm.port(fp), tm.port(tp)) else { continue };
            let to_area = prefixed(tname, tm.area(tport.area).unwrap().name);
            let ef = Efferent {
                to: to_area,
                from_slot: fport.slot,
                to_slot: tport.slot,
                projection: w.projection,
                map: w.map,
                w: w.w,
                fan_in: w.fan_in,
                synapse: w.synapse,
            };
            match extra.iter_mut().find(|(i, n, _)| *i == fi && *n == fport.area) {
                Some((_, _, v)) => v.push(ef),
                None => extra.push((fi, fport.area, vec![ef])),
            }
        }
        // 展开每个部件实例
        let mut out = Vec::new();
        for (idx, (m, inst)) in self.insts.iter().enumerate() {
            for a in m.areas {
                let mut effs: Vec<Efferent> = a
                    .efferents
                    .iter()
                    .map(|ef| Efferent {
                        to: prefixed(inst, m.area(ef.to).unwrap().name),
                        ..*ef
                    })
                    .collect();
                if let Some((_, _, add)) = extra.iter().find(|(i, n, _)| *i == idx && *n == a.name) {
                    effs.extend(add.iter().copied());
                }
                out.push(AreaSpec {
                    name: prefixed(inst, a.name),
                    efferents: leak_slice(effs),
                    ..*a
                });
            }
        }
        Ok(out)
    }

    /// 组装的规模摘要（给人看）。
    pub fn summary(&self) -> String {
        let n_area: usize = self.insts.iter().map(|(m, _)| m.areas.len()).sum();
        format!(
            "组装：{} 个实例、{} 个区、{} 条跨部件连线",
            self.insts.len(),
            n_area,
            self.wires.len()
        )
    }
}

// ---- 名字工具：组装期一次性泄漏（见模块文档的说明）----
fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}
fn leak_slice(v: Vec<Efferent>) -> &'static [Efferent] {
    Box::leak(v.into_boxed_slice())
}
fn prefixed(inst: &str, name: &'static str) -> &'static str {
    leak(&format!("{inst}:{name}"))
}
fn split(end: &'static str) -> (&'static str, &'static str) {
    match end.split_once('.') {
        Some((a, b)) => (leak(a), leak(b)),
        None => (end, end),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::area::{AreaClass, Layout, LocalMotif, Micro, Size, Slot};
    use crate::core::cell::Modulators;
    use crate::sim::{Brain, Config};

    static RELAY_SLOTS: [Slot; 2] = [
        Slot::afferent("in", TopoDomain::Somatic),
        Slot::efferent("out", TopoDomain::Somatic),
    ];
    static NO_EFF: [Efferent; 0] = [];

    const RELAY: AreaSpec = AreaSpec {
        name: "RELAY",
        class: AreaClass::ThalamicRelay { tonic: true },
        topo: TopoDomain::Somatic,
        micro: Micro::relay(),
        size: Size::new(24, 6, 1),
        layout: Layout::mni([0.0, 0.0, 0.0], 0.5),
        motif: LocalMotif::relay(),
        slots: &RELAY_SLOTS,
        topology: crate::core::topo::Topo::LINE,
        efferents: &NO_EFF,
        neuron: crate::core::cell::Spec::new(crate::core::cell::Model::AdLif),
        type_specs: &[],
    };

    static RELAY_AREAS: [AreaSpec; 1] = [RELAY];
    static RELAY_PORTS: [Port; 2] = [
        Port::inp("in", "RELAY", "in"),
        Port::out("out", "RELAY", "out"),
    ];
    static RELAY_MOD: Module = Module {
        name: "relay",
        areas: &RELAY_AREAS,
        ports: &RELAY_PORTS,
    };

    fn chain() -> Assembly {
        let mut a = Assembly::new();
        a.add(&RELAY_MOD, "L")
            .add(&RELAY_MOD, "R")
            .wire("L.out", "R.in", Projection::Feedforward, Map::Identity, 2.0, Some(8));
        a
    }

    #[test]
    fn assembly_expands_with_prefixed_names_and_rewrites_targets() {
        let specs = chain().expand().unwrap();
        let names: Vec<&str> = specs.iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["L:RELAY", "R:RELAY"]);
        // 跨部件连线变成了 L 的出向投射，靶名带实例前缀
        let l = specs.iter().find(|s| s.name == "L:RELAY").unwrap();
        assert_eq!(l.efferents.len(), 1, "{:?}", l.efferents);
        assert_eq!(l.efferents[0].to, "R:RELAY");
        assert_eq!(l.efferents[0].to_slot, "in");
        // R 没有出向投射
        let r = specs.iter().find(|s| s.name == "R:RELAY").unwrap();
        assert!(r.efferents.is_empty());
    }

    #[test]
    fn assembly_reports_structural_mistakes() {
        // 实例重名
        let mut a = Assembly::new();
        a.add(&RELAY_MOD, "X").add(&RELAY_MOD, "X");
        assert!(a.validate().iter().any(|e| e.contains("[实例重名]")), "{:?}", a.validate());

        // 端口不存在 / 方向不符
        let mut b = chain();
        b.wire("L.nope", "R.in", Projection::Feedforward, Map::Identity, 1.0, None);
        assert!(b.validate().iter().any(|e| e.contains("[连线无端口]")));

        // 方向反了（把 in 当出口）
        let mut c = chain();
        c.wire("L.in", "R.in", Projection::Feedforward, Map::Identity, 1.0, None);
        assert!(c.validate().iter().any(|e| e.contains("[连线方向不符]")));
    }

    #[test]
    fn assembled_brain_actually_runs() {
        let specs = chain().expand().unwrap();
        let cfg = Config { n_exc: 24, n_inh: 6, dt_ms: 1.0, seed: 5, ..Default::default() };
        let mut brain = Brain::declare(&specs, cfg).expect("组装出的声明应当合法");
        let ext = vec![8.0f32, 0.0];
        for _ in 0..200 {
            brain.step(&ext, &Modulators::default());
        }
        println!("[assembly] {} L {:.1} Hz  R {:.1} Hz", brain.edges.len(), brain.rate_hz(0), brain.rate_hz(1));
        assert!(brain.rate_hz(0) > 0.0, "被驱动的实例应当发放");
        assert_eq!(brain.edges.len(), 1, "跨部件连线应当生成一条边");
        assert!(brain.rate_hz(1) > 0.0, "信号应当传到下游实例");
    }
}
