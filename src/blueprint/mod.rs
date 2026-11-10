//! **蓝图工具**：对一份"区声明"做校验、描述、导出清单、算指纹。
//!
//! 这四个动作都**不需要跑动力学**，所以能在 CI 里、在启动前、在评审时用：
//!
//! ```rust
//! use dreamkernel::blueprint::Blueprint;
//!
//! # let areas: Vec<dreamkernel::core::area::AreaSpec> = Vec::new();
//! let bp = Blueprint::new(&areas);
//! let problems = bp.validate();     // 全部问题，不是第一个
//! let summary = bp.describe();      // 人读
//! let json = bp.manifest_json();    // 机读（给世界/可视化/外部服务器）
//! let (a, e) = bp.fingerprint();    // 结构指纹（改动会变，重构不该变）
//! ```
//!
//! 与具体脑无关：输入是**声明**，所以任何一份组装出来的区表都能用。

use crate::core::area::{AreaClass, AreaSpec, SlotKind, TopoDomain};
use crate::link::synapse::Plasticity;

/// 一份区声明（切片包装，方法都在这里）。
#[derive(Clone, Copy, Debug)]
pub struct Blueprint<'a> {
    pub areas: &'a [AreaSpec],
}

impl<'a> Blueprint<'a> {
    pub const fn new(areas: &'a [AreaSpec]) -> Self {
        Self { areas }
    }

    fn find(&self, name: &str) -> Option<&'a AreaSpec> {
        self.areas.iter().find(|a| a.name == name)
    }

    /// 校验：**返回全部问题**（不 panic、不早退）。
    ///
    /// 覆盖四类：区名唯一、投射两端存在、槽存在且方向对、几何/地形自洽。
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        // ① 区名唯一
        for (i, a) in self.areas.iter().enumerate() {
            if self.areas[..i].iter().any(|b| b.name == a.name) {
                errs.push(format!("[区重名] {}", a.name));
            }
        }
        for a in self.areas {
            // ② 子区域必须要求区有形状
            for s in a.slots {
                if s.region.is_some() && a.topology.dims() == 0 {
                    errs.push(format!(
                        "[区域无形状] {} 的槽 {} 声明了子区域，但区的拓扑是 Point",
                        a.name, s.name
                    ));
                }
                if let Some(r) = s.region
                    && (r.u0 < 0.0 || r.u1 > 1.0 || r.u0 >= r.u1)
                {
                    errs.push(format!("[区域越界] {} 的槽 {} 子区域非法", a.name, s.name));
                }
            }
            // ③ 投射：靶区、两端槽、几何、地形
            for ef in a.efferents {
                let Some(b) = self.find(ef.to) else {
                    errs.push(format!("[缺靶区] {} → {}", a.name, ef.to));
                    continue;
                };
                let from_slot = a.slots.iter().find(|s| s.name == ef.from_slot);
                if from_slot.is_none() {
                    errs.push(format!("[缺槽] {} 没有出向槽 {}", a.name, ef.from_slot));
                } else if from_slot.is_some_and(|s| s.kind != SlotKind::Efferent) {
                    errs.push(format!("[槽方向不符] {}::{} 不是出向槽", a.name, ef.from_slot));
                }
                let to_slot = b.slots.iter().find(|s| s.name == ef.to_slot);
                if to_slot.is_none() {
                    errs.push(format!("[缺槽] {} 没有入向槽 {}", b.name, ef.to_slot));
                } else if to_slot.is_some_and(|s| s.kind != SlotKind::Afferent) {
                    errs.push(format!("[槽方向不符] {}::{} 不是入向槽", b.name, ef.to_slot));
                }
                // 两端都声明了子区域 → 维数必须拼得上
                if let (Some(fs), Some(ts)) = (from_slot, to_slot) {
                    let (da, db) = (a.topology.dims(), b.topology.dims());
                    if fs.region.is_some() && ts.region.is_some() && da > 0 && db > 0 && da != db {
                        errs.push(format!(
                            "[几何拼不上] {} → {}：源 {da} 维、靶 {db} 维",
                            a.name, ef.to
                        ));
                    }
                    if fs.topo != TopoDomain::None && !ef.map.is_topographic() {
                        errs.push(format!(
                            "[无地形] {} → {}：声明了 {:?} 槽却用 AllToAll",
                            a.name, ef.to, fs.topo
                        ));
                    }
                }
            }
        }
        errs
    }

    /// 人读摘要。
    pub fn describe(&self) -> String {
        let mut o = String::new();
        let n = self.areas.len();
        let n_eff: usize = self.areas.iter().map(|a| a.efferents.len()).sum();
        o.push_str(&format!("蓝图：{n} 个区，{n_eff} 条出向投射\n"));

        // 按类别
        let mut by_class: Vec<(String, usize)> = Vec::new();
        for a in self.areas {
            let k = format!("{:?}", a.class);
            match by_class.iter_mut().find(|(c, _)| *c == k) {
                Some((_, n)) => *n += 1,
                None => by_class.push((k, 1)),
            }
        }
        by_class.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        o.push_str("  类别：");
        for (k, c) in &by_class {
            o.push_str(&format!("{k}×{c} "));
        }
        o.push('\n');

        // 拓扑三档
        let (mut p0, mut p1, mut p2) = (0, 0, 0);
        let (mut collapsed, mut metric) = (0, 0);
        for a in self.areas {
            match a.topology.dims() {
                0 => p0 += 1,
                1 => p1 += 1,
                _ => p2 += 1,
            }
            if a.topology.has_metric() {
                metric += 1;
            } else if a.topology.dims() > 0 {
                collapsed += 1;
            }
        }
        o.push_str(&format!(
            "  拓扑：无结构 {p0} / 一维 {p1} / 二维 {p2}；有米制 {metric}，坍缩 {collapsed}\n"
        ));

        // 槽与突触
        let n_slots: usize = self.areas.iter().map(|a| a.slots.len()).sum();
        let n_sub = self
            .areas
            .iter()
            .filter(|a| a.slots.iter().any(|s| s.region.is_some()))
            .count();
        o.push_str(&format!("  槽：{n_slots} 个（{n_sub} 个区带子区域）\n"));

        let mut frozen = 0;
        let mut plastic = 0;
        let mut noisy = 0;
        for a in self.areas {
            for ef in a.efferents {
                if ef.synapse.rule == Plasticity::Frozen {
                    frozen += 1;
                } else {
                    plastic += 1;
                }
                if ef.synapse.dist != crate::link::synapse::WeightDist::Fixed {
                    noisy += 1;
                }
            }
        }
        o.push_str(&format!(
            "  突触：固定权重 {frozen} / 可塑性 {plastic}；带权重分布 {noisy}\n"
        ));

        // 细胞模型
        let mut by_model: Vec<(String, usize)> = Vec::new();
        for a in self.areas {
            let k = format!("{:?}", a.neuron.model);
            match by_model.iter_mut().find(|(c, _)| *c == k) {
                Some((_, n)) => *n += 1,
                None => by_model.push((k, 1)),
            }
        }
        o.push_str("  细胞模型：");
        for (k, c) in &by_model {
            o.push_str(&format!("{k}×{c} "));
        }
        o.push('\n');

        let (ah, eh) = self.fingerprint();
        o.push_str(&format!("  指纹：ahash={ah} ehash={eh}\n"));
        o
    }

    /// 机读清单（手写 JSON，零依赖）。
    pub fn manifest_json(&self) -> String {
        let mut o = String::from("{\n  \"schema\": 1,\n  \"areas\": [\n");
        for (i, a) in self.areas.iter().enumerate() {
            o.push_str(&format!(
                "    {{\"name\": \"{}\", \"class\": \"{:?}\", \"dims\": {}, \"metric\": \"{:?}\", \
                 \"patches\": {}, \"model\": \"{:?}\", \"slots\": [",
                a.name, a.class, a.topology.dims(), a.topology.metric, a.size.patches, a.neuron.model
            ));
            for (j, s) in a.slots.iter().enumerate() {
                if j > 0 {
                    o.push_str(", ");
                }
                o.push_str(&format!(
                    "{{\"name\": \"{}\", \"kind\": \"{:?}\", \"region\": {}}}",
                    s.name,
                    s.kind,
                    s.region
                        .map(|r| format!("[{:.3},{:.3},{:.3},{:.3}]", r.u0, r.u1, r.v0, r.v1))
                        .unwrap_or_else(|| "null".to_string())
                ));
            }
            o.push_str("], \"efferents\": [");
            for (j, ef) in a.efferents.iter().enumerate() {
                if j > 0 {
                    o.push_str(", ");
                }
                o.push_str(&format!(
                    "{{\"to\": \"{}\", \"from_slot\": \"{}\", \"to_slot\": \"{}\", \"w\": {:.4}, \
                     \"fan_in\": {}, \"topographic\": {}, \"plastic\": {}}}",
                    ef.to,
                    ef.from_slot,
                    ef.to_slot,
                    ef.w,
                    ef.fan_in.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
                    ef.map.is_topographic(),
                    ef.synapse.rule.is_plastic()
                ));
            }
            o.push_str("]}");
            if i + 1 < self.areas.len() {
                o.push(',');
            }
            o.push('\n');
        }
        o.push_str("  ],\n");
        let (ah, eh) = self.fingerprint();
        o.push_str(&format!(
            "  \"totals\": {{\"areas\": {}, \"efferents\": {}, \"ahash\": {ah}, \"ehash\": {eh}}}\n}}\n",
            self.areas.len(),
            self.areas.iter().map(|a| a.efferents.len()).sum::<usize>()
        ));
        o
    }

    /// 结构指纹：`(区+槽, 投射)`。顺序敏感——**重构不该变，语义改动一定会变**。
    pub fn fingerprint(&self) -> (u64, u64) {
        const FNV: u64 = 0xcbf2_9ce4_8422_2325;
        let mut ah = FNV;
        let mut eh = FNV;
        let mix = |h: &mut u64, s: &str| {
            for b in s.as_bytes() {
                *h ^= *b as u64;
                *h = h.wrapping_mul(0x100_0000_01b3);
            }
            *h ^= 0xff;
            *h = h.wrapping_mul(0x100_0000_01b3);
        };
        for a in self.areas {
            mix(&mut ah, a.name);
            mix(&mut ah, &format!("{:?}{:?}{}", a.class, a.topology, a.size.patches));
            for s in a.slots {
                mix(&mut ah, &format!("{}:{:?}", s.name, s.kind));
                if let Some(r) = s.region {
                    mix(&mut ah, &format!("{}", r.u0) as &str);
                }
            }
            for ef in a.efferents {
                mix(&mut eh, a.name);
                mix(&mut eh, ef.to);
                mix(&mut eh, &format!("{}:{}", ef.from_slot, ef.to_slot));
                mix(&mut eh, &format!("{}{:?}{}", ef.w, ef.map, ef.synapse.rule.is_plastic()));
            }
        }
        (ah, eh)
    }
}

/// 便利函数：直接对切片校验。
pub fn validate(areas: &[AreaSpec]) -> Vec<String> {
    Blueprint::new(areas).validate()
}

/// 便捷构造 `AreaClass` 的一句话说明（用于 CLI/文档）。
pub fn class_hint(c: AreaClass) -> &'static str {
    match c {
        AreaClass::Cortex => "皮层",
        AreaClass::ThalamicRelay { tonic: true } => "丘脑中继（紧张性）",
        AreaClass::ThalamicRelay { tonic: false } => "丘脑中继",
        AreaClass::SubcorticalRelay => "皮质下中继",
        AreaClass::PrimaryAfferent => "一级传入",
        AreaClass::Subcortical => "皮质下",
        AreaClass::Modulatory => "调质",
        AreaClass::Motor => "运动核",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::area::{Layout, LocalMotif, Micro, Size, Slot};
    use crate::core::topo::Topo;
    use crate::link::projection::{Efferent, Projection};

    static S1: [Slot; 2] = [
        Slot::afferent("in", TopoDomain::Somatic),
        Slot::efferent("out", TopoDomain::Somatic),
    ];
    static S2: [Slot; 2] = [
        Slot::afferent("in", TopoDomain::Somatic),
        Slot::efferent("out", TopoDomain::None),
    ];
    static EFF_OK: [Efferent; 1] =
        [Efferent::topo("B", "out", "in", Projection::Feedforward, 1.0, 8)];

    fn area(name: &'static str, slots: &'static [Slot], topology: Topo, eff: &'static [Efferent]) -> AreaSpec {
        AreaSpec {
            name,
            class: AreaClass::Cortex,
            topo: TopoDomain::Somatic,
            micro: Micro::cortical(),
            size: Size::new(8, 2, 1),
            layout: Layout::mni([0.0, 0.0, 0.0], 0.5),
            motif: LocalMotif::cortical(),
            slots,
            topology,
            efferents: eff,
            neuron: crate::core::cell::Spec::new(crate::core::cell::Model::AdLif),
            type_specs: &[],
        }
    }

    #[test]
    fn a_clean_blueprint_validates_and_describes() {
        let areas = vec![
            area("A", &S1, Topo::LINE, &EFF_OK),
            area("B", &S2, Topo::POINT, &[]),
        ];
        let bp = Blueprint::new(&areas);
        assert!(bp.validate().is_empty(), "{:?}", bp.validate());
        let d = bp.describe();
        assert!(d.contains("2 个区"), "{d}");
        assert!(d.contains("1 条出向投射"), "{d}");
        assert!(d.contains("固定权重 1"), "{d}");
        let m = bp.manifest_json();
        assert!(m.contains("\"schema\": 1"));
        assert!(m.contains("\"name\": \"A\""));
        assert!(m.contains("\"totals\": {\"areas\": 2"), "{m}");
        let (ah, eh) = bp.fingerprint();
        assert_ne!(ah, 0);
        assert_ne!(eh, 0);
    }

    #[test]
    fn problems_are_all_reported_not_just_the_first() {
        static BAD: [Efferent; 3] = [
            Efferent::topo("NOPE", "out", "in", Projection::Feedforward, 1.0, 8),
            Efferent::topo("B", "no_such_slot", "in", Projection::Feedforward, 1.0, 8),
            Efferent {
                to: "B",
                from_slot: "out",
                to_slot: "in",
                projection: Projection::Feedforward,
                map: crate::core::topo::Map::AllToAll,
                w: 1.0,
                fan_in: Some(8),
                synapse: crate::link::synapse::Synapse::FIXED,
            },
        ];
        // A 有三条坏声明；B 是合法靶；第二个 A 触发重名
        let areas = vec![
            area("A", &S1, Topo::LINE, &BAD),
            area("B", &S2, Topo::POINT, &[]),
            area("A", &S2, Topo::POINT, &[]),
        ];
        let errs = Blueprint::new(&areas).validate();
        for tag in ["[区重名]", "[缺靶区]", "[缺槽]", "[无地形]"] {
            assert!(errs.iter().any(|e| e.contains(tag)), "缺 {tag}：{errs:?}");
        }
    }

    #[test]
    fn fingerprint_changes_with_semantics_not_with_representation() {
        let base = vec![area("A", &S1, Topo::LINE, &EFF_OK), area("B", &S2, Topo::POINT, &[])];
        let again = base.clone();
        assert_eq!(
            Blueprint::new(&base).fingerprint(),
            Blueprint::new(&again).fingerprint(),
            "同样的声明必须同指纹"
        );
        // 改一处语义（权重）→ 投射指纹必须变
        let mut changed = base.clone();
        let mut eff = EFF_OK;
        eff[0] = Efferent { w: 2.0, ..EFF_OK[0] };
        changed[0].efferents = Box::leak(eff.to_vec().into_boxed_slice());
        assert_ne!(
            Blueprint::new(&base).fingerprint().1,
            Blueprint::new(&changed).fingerprint().1,
            "语义改动必须改变指纹"
        );
    }
}
