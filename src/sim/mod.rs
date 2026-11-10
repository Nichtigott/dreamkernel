//! **最小脉冲引擎**：从「声明」生成一个能跑的脑。
//!
//! 这是工具包的正题——`dreamkernel` 不只是类型库，它能把下面这份声明变成网络并跑起来：
//!
//! ```text
//! AreaSpec{ 区名, 细胞构成, 层构, 拓扑, 槽(子区域), 出向投射(Map/权重/会聚数) }
//!   → Areas → 边（由出向投射与槽的子区域生成）→ 神经元 + 突触 → step() → 发放率
//! ```
//!
//! 刻意**不做**的事（那属于实现层，见 docs/75）：
//! 感觉受体、世界接口、GPU、睡眠/重放、区特化的花活（LGN 的 ON/OFF 双图、上丘运动图…）。
//! 这里只保证"声明能长出网络"，让框架成为一个**可验证的最小闭环**。
//!
//! 规模口径：每区的 E / PV / SOM / VIP 数量由 [`Config`] 给（`--area-e` 那一套），
//! 层号由 `LaminarProfile` 的宽度比例分配——与实现层的约定一致，但不读那份数据。

use crate::core::area::{Area, AreaSpec};
use crate::core::cell::CellType;
use crate::core::layer::Layer;
use crate::core::neuron::{Neuron, Params};
use crate::core::topo::{Coord, Topo};
use crate::link::projection::Projection;

/// 建脑配置（框架层的默认值刻意保守：小、快、可复现）。
#[derive(Debug, Clone)]
pub struct Config {
    /// 每区兴奋性神经元数。
    pub n_exc: usize,
    /// 每区抑制性神经元数（按 PV/SOM/VIP = 0.6/0.25/0.15 分）。
    pub n_inh: usize,
    /// 覆盖特定区规模：`(区索引, E, I)`。
    pub overrides: Vec<(usize, usize, usize)>,
    /// 每神经元 máximo 区际扇入。
    pub fan_in: usize,
    /// 同区局部连边比例（E→E）。
    pub local_frac: f32,
    /// 随机种子（固定 → 结果可复现，这是判据的前提）。
    pub seed: u64,
    /// 仿真步长（ms）。影响 `Params::dt`。
    pub dt_ms: f32,
    /// 没有输入时每区的基础驱动电流。
    pub ext_base: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_exc: 24,
            n_inh: 6,
            overrides: Vec::new(),
            fan_in: 32,
            local_frac: 0.25,
            seed: 1,
            dt_ms: 1.0,
            ext_base: 0.0,
        }
    }
}

/// 一条**由声明生成**的边（建脑时的中间产物，公开是为了可观测）。
#[derive(Debug, Clone, Copy)]
pub struct BuiltEdge {
    pub from_area: usize,
    pub to_area: usize,
    pub projection: Projection,
    /// 这条边一共建了多少突触。
    pub synapses: usize,
}

/// 一个能跑的脑。
pub struct Brain {
    pub areas: Vec<Area>,
    /// 每神经元所属区。
    pub area_of: Vec<usize>,
    /// 每神经元细胞类型。
    pub cell_type: Vec<CellType>,
    /// 每神经元层（`Layer` 的 idx）。
    pub layer_of: Vec<u8>,
    /// 每神经元坐标（该区拓扑上的位置；`Point` 时全为 (0.5, 0.5)）。
    pub coord_of: Vec<Coord>,
    pub neurons: Vec<Neuron>,
    /// 逐神经元的入边 `(pre, w)`。
    pub incoming: Vec<Vec<(usize, f32)>>,
    /// 逐神经元的入边延迟（ms，量化到整数步）。
    pub delay: Vec<Vec<u8>>,
    /// 本帧外部输入电流（离散步之间由调用方设置）。
    pub input: Vec<f32>,
    /// 每区起点（`area_start[a]..area_start[a+1]`）。
    pub area_start: Vec<usize>,
    /// 每区四类细胞数量。
    pub counts: Vec<(usize, usize, usize, usize)>,
    /// 建边记录（可观测）。
    pub edges: Vec<BuiltEdge>,
    cfg: Config,
    /// 脉冲历史环（延迟线）：`hist[d * n + i]`。
    hist: Vec<u8>,
    steps: usize,
    max_delay: usize,
    /// 每神经元的脉冲计数（读速率用）。
    pub spike_count: Vec<u32>,
    /// 最近一步是否发放。
    pub spikes: Vec<bool>,
}

impl Brain {
    /// **从声明建脑**：`AreaSpec` 列表 → 区表 → 由各区的出向投射生成边 → 神经元 + 突触。
    ///
    /// 返回 `Err` 而不是 panic：声明写错（槽不存在 / 拓扑域不自洽 / 名字重复）应当被
    /// 调用方看到，而不是让进程崩在初始化里。
    pub fn declare(specs: &[AreaSpec], cfg: Config) -> Result<Self, Vec<String>> {
        let mut errs = Vec::new();
        // ① 名字唯一
        for (i, s) in specs.iter().enumerate() {
            if specs[..i].iter().any(|t| t.name == s.name) {
                errs.push(format!("[区重名] {}", s.name));
            }
        }
        // ② 出向投射的两端与槽必须存在
        let find = |n: &str| specs.iter().find(|s| s.name == n);
        for s in specs {
            for ef in s.efferents {
                if find(ef.to).is_none() {
                    errs.push(format!("[缺靶区] {} → {}", s.name, ef.to));
                    continue;
                }
                if !s.has_slot(ef.from_slot, crate::core::area::SlotKind::Efferent) {
                    errs.push(format!("[缺槽] {} 没有出向槽 {}", s.name, ef.from_slot));
                }
                let t = find(ef.to).unwrap();
                if !t.has_slot(ef.to_slot, crate::core::area::SlotKind::Afferent) {
                    errs.push(format!("[缺槽] {} 没有入向槽 {}", t.name, ef.to_slot));
                }
            }
        }
        if !errs.is_empty() {
            return Err(errs);
        }

        let areas: Vec<Area> = specs.iter().map(|s| s.to_area()).collect();
        let mut b = Self::skeleton(areas, cfg);
        b.wire_from_efferents();
        b.finish();
        Ok(b)
    }

    /// 空壳：只建神经元（不建突触）。给"我要自己接线"的场合用。
    pub fn from_areas(areas: Vec<Area>, cfg: Config) -> Self {
        let mut b = Self::skeleton(areas, cfg);
        b.finish();
        b
    }

    fn skeleton(areas: Vec<Area>, cfg: Config) -> Self {
        let n_areas = areas.len();
        let mut counts = Vec::with_capacity(n_areas);
        for (i, a) in areas.iter().enumerate() {
            let (e, inh) = cfg
                .overrides
                .iter()
                .find(|(idx, _, _)| *idx == i)
                .map(|(_, e, i2)| (*e, *i2))
                .unwrap_or((cfg.n_exc, cfg.n_inh));
            let pv = (inh as f32 * 0.6).round() as usize;
            let som = (inh as f32 * 0.25).round() as usize;
            let vip = inh.saturating_sub(pv + som);
            let _ = a;
            counts.push((e, pv, som, vip));
        }
        let mut area_start = Vec::with_capacity(n_areas);
        let mut total = 0usize;
        for &(e, p, s, v) in &counts {
            area_start.push(total);
            total += e + p + s + v;
        }
        area_start.push(total);

        let mut rng = crate::num::math::Lcg::new(cfg.seed);
        let mut area_of = Vec::with_capacity(total);
        let mut cell_type = Vec::with_capacity(total);
        let mut layer_of = Vec::with_capacity(total);
        let mut coord_of = Vec::with_capacity(total);
        let mut neurons = Vec::with_capacity(total);
        for (ai, a) in areas.iter().enumerate() {
            let (e, p, s, v) = counts[ai];
            let present: Vec<Layer> = a.profile.present_layers();
            let topo = a.topology;
            for (k, &(n, t)) in [(e, CellType::P), (p, CellType::Pv), (s, CellType::Som), (v, CellType::Vip)]
                .iter()
                .enumerate()
            {
                let coords = topo.assign(n);
                for (j, c) in coords.iter().enumerate() {
                    area_of.push(ai);
                    cell_type.push(t);
                    // 层：按区层构的宽度比例给（与实现层同一套：单层区统一 L3 槽位）
                    let frac = (j as f32 + 0.5) / n.max(1) as f32;
                    layer_of.push(layer_at(&present, &a.profile.width, frac));
                    coord_of.push(*c);
                    let prm = params_for(t, cfg.dt_ms);
                    neurons.push(Neuron::new(prm));
                    let _ = k;
                }
            }
            let _ = rng.next_f32();
        }
        let n = neurons.len();
        Self {
            areas,
            area_of,
            cell_type,
            layer_of,
            coord_of,
            neurons,
            incoming: vec![Vec::new(); n],
            delay: vec![Vec::new(); n],
            input: vec![0.0; n],
            area_start,
            counts,
            edges: Vec::new(),
            cfg,
            hist: vec![0u8; n],
            steps: 0,
            max_delay: 1,
            spike_count: vec![0u32; n],
            spikes: vec![false; n],
        }
    }

    /// 由各区声明的 `efferents` 生成突触。**这是"声明即生成"的地方**。
    pub fn wire_from_efferents(&mut self) {
        // 先快照（借 self 的可变方法时要读 areas）
        struct Snap {
            from: usize,
            to_name: &'static str,
            st: Topo,
            eff: crate::link::projection::Efferent,
            src_rect: Option<crate::core::topo::Rect>,
            dst_slot: &'static str,
        }
        let mut snaps: Vec<Snap> = Vec::new();
        for (i, a) in self.areas.iter().enumerate() {
            for ef in a.efferents {
                let src_rect = a
                    .slots
                    .iter()
                    .find(|s| s.name == ef.from_slot)
                    .and_then(|s| s.region);
                snaps.push(Snap {
                    from: i,
                    to_name: ef.to,
                    st: a.topology,
                    eff: *ef,
                    src_rect,
                    dst_slot: ef.to_slot,
                });
            }
        }
        for sn in snaps {
            let Some(to) = self.areas.iter().position(|a| a.name == sn.to_name) else { continue };
            let dt = self.areas[to].topology;
            let dst_rect = self.areas[to]
                .slots
                .iter()
                .find(|s| s.name == sn.dst_slot)
                .and_then(|s| s.region);
            let n_before: usize = self.incoming.iter().map(|v| v.len()).sum();
            self.wire_one(
                sn.from,
                to,
                sn.st,
                dt,
                sn.src_rect,
                dst_rect,
                sn.eff.map,
                sn.eff.fan_in.unwrap_or(self.cfg.fan_in),
                sn.eff.w,
                sn.eff.projection,
            );
            let n_after: usize = self.incoming.iter().map(|v| v.len()).sum();
            self.edges.push(BuiltEdge {
                from_area: sn.from,
                to_area: to,
                projection: sn.eff.projection,
                synapses: n_after - n_before,
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn wire_one(
        &mut self,
        a: usize,
        b: usize,
        st: Topo,
        dt: Topo,
        src_rect: Option<crate::core::topo::Rect>,
        dst_rect: Option<crate::core::topo::Rect>,
        map: crate::core::topo::Map,
        fan_in: usize,
        w_mul: f32,
        _proj: Projection,
    ) {
        let (sa, sb) = (self.area_start[a], self.area_start[b]);
        let ea = self.counts[a].0;
        let eb = self.counts[b].0;
        // 源/靶子群体：E 群内、被 Rect 选中的那些
        let cs = st.assign(ea);
        let cd = dt.assign(eb);
        let sel = |all: &[Coord], r: Option<crate::core::topo::Rect>| -> Vec<usize> {
            match r {
                None => (0..all.len()).collect(),
                Some(r) => (0..all.len()).filter(|&i| r.contains(all[i])).collect(),
            }
        };
        let s_sel = sel(&cs, src_rect);
        let d_sel = sel(&cd, dst_rect);
        if s_sel.is_empty() || d_sel.is_empty() {
            return;
        }
        // 权重：源细胞类型决定符号（框架不猜生物量，用 1.0 作基准）
        let w = 1.0 * w_mul;
        if map.is_topographic() && st.dims() > 0 && dt.dims() > 0 {
            // 按坐标配对（保序）：每个靶收最近的 fan_in 个源
            let sco: Vec<Coord> = s_sel.iter().map(|&i| cs[i]).collect();
            let dco: Vec<Coord> = d_sel.iter().map(|&i| cd[i]).collect();
            for (di, si, len) in map.pair_with_len(&sco, &dco, fan_in) {
                let post = sb + d_sel[di];
                let pre = sa + s_sel[si];
                self.incoming[post].push((pre, w));
                // 延迟：有米制的拓扑才把长度当距离（② 档坍缩，不产生延迟）
                let d = self.delay_for(a, b, len);
                self.delay[post].push(d);
            }
        } else {
            // 广播：每个靶收 fan_in 个（近似均匀）
            let mut rng = crate::num::math::Lcg::new(self.cfg.seed ^ (a as u64) << 32 ^ b as u64);
            let d_br = self.delay_for(a, b, 0.0);
            for &di in &d_sel {
                let post = sb + di;
                for _ in 0..fan_in.max(1) {
                    let si = s_sel[(rng.next_f32() * s_sel.len() as f32) as usize % s_sel.len()];
                    self.incoming[post].push((sa + si, w));
                    self.delay[post].push(d_br);
                }
            }
        }
    }

    /// 传导延迟（步）。长度 = **区中心距离** + 区内位置差（只有 `Metric::Millimetres`
    /// 的区才算区内那一项；② 坍缩档的"相邻"不是毫米，不能凭空造出距离）。
    fn delay_for(&self, a: usize, b: usize, len_norm: f32) -> u8 {
        let (pa, pb) = (self.areas[a].pos, self.areas[b].pos);
        let dc = ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt();
        // 区内位置差：**只要有一端是真实米制拓扑**就算——沿源图的展布本身是轴突长度。
        // 两端都坍缩/无结构时这一项为 0（"相邻"不是毫米，不许凭空造距离）。
        let (ta, tb) = (self.areas[a].topology, self.areas[b].topology);
        let intra = if ta.has_metric() || tb.has_metric() {
            let half = if ta.has_metric() {
                1.5 + 3.0 * self.areas[a].width.max(0.1)
            } else {
                1.5 + 3.0 * self.areas[b].width.max(0.1)
            };
            len_norm * 2.0 * half
        } else {
            0.0
        };
        let mm = dc + intra;
        // 10 mm/ms = 10 m/s（与实现层默认一致）；1 ms 是突触传递本身
        ((1.0 + mm / 10.0) / self.cfg.dt_ms).round().clamp(1.0, 16.0) as u8
    }

    /// 局部（同区）E→E 连边：`local_frac` 比例的随机连接。
    pub fn wire_local(&mut self) {
        let mut rng = crate::num::math::Lcg::new(self.cfg.seed ^ 0x5DEE_CE66);
        let n_areas = self.areas.len();
        for a in 0..n_areas {
            let base = self.area_start[a];
            let e = self.counts[a].0;
            if e == 0 {
                continue;
            }
            let k = ((e as f32 * self.cfg.local_frac).round() as usize).clamp(1, e);
            for j in 0..e {
                let post = base + j;
                for _ in 0..k {
                    let pre = base + (rng.next_f32() * e as f32) as usize % e;
                    if pre == post {
                        continue;
                    }
                    self.incoming[post].push((pre, 1.0));
                    self.delay[post].push(1);
                }
            }
        }
    }

    fn finish(&mut self) {
        self.max_delay = self.delay.iter().flatten().copied().max().unwrap_or(1) as usize;
        self.hist = vec![0u8; (self.max_delay + 1) * self.neurons.len().max(1)];
        self.steps = 0;
    }

    /// 走一步。`ext` 是每区的额外驱动（长度不足按 0 补齐）。
    ///
    /// 延迟实现：`hist[d * n + i]` 表示"神经元 i 在 d 步前是否发放"，按 `steps % D` 取模
    /// 成环（`D = max_delay + 1`）。写在第 0 槽，读时按延迟回看 —— 这就是传导延迟。
    pub fn step(&mut self, ext: &[f32], modulator: f32) {
        let n = self.neurons.len();
        if n == 0 {
            return;
        }
        let d_size = self.max_delay + 1;
        // 1) 收集本步到达的突触电流
        let mut syn = vec![0.0f32; n];
        // 下标有语义：同一个 post 要同时读 `incoming/delay/hist` 三个数组，zip 会丢信息
        #[allow(clippy::needless_range_loop)]
        for post in 0..n {
            let mut acc = 0.0f32;
            for (k, &(pre, w)) in self.incoming[post].iter().enumerate() {
                let d = (self.delay[post][k] as usize).min(self.max_delay);
                if d == 0 {
                    continue;
                }
                // 回看 d 步：ring 的写法是"第 s 步写入 saturating"，所以读 (steps - d) mod D
                let slot = (self.steps + d_size - d) % d_size;
                if self.hist[slot * n + pre] != 0 {
                    acc += w;
                }
            }
            syn[post] = acc * modulator;
        }
        // 2) 逐神经元积分
        let mut fired = vec![false; n];
        // 下标有语义：同时读 `area_of/neurons/spike_count`，且要写回 `fired`
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let a = self.area_of[i];
            let e = ext.get(a).copied().unwrap_or(0.0) + self.cfg.ext_base;
            let inp = syn[i] + self.input[i] + e;
            fired[i] = self.neurons[i].step(inp);
            if fired[i] {
                self.spike_count[i] += 1;
            }
        }
        // 3) 本步脉冲写进环
        self.steps += 1;
        let slot = self.steps % d_size;
        for (i, &f) in fired.iter().enumerate() {
            self.hist[slot * n + i] = u8::from(f);
        }
        self.spikes = fired;
    }

    /// 某区的平均发放率（Hz，按已跑的步数折算）。
    pub fn rate_hz(&self, area: usize) -> f32 {
        let (s, e) = (self.area_start[area], self.area_start[area + 1]);
        if e <= s || self.steps == 0 {
            return 0.0;
        }
        let c: u32 = self.spike_count[s..e].iter().sum();
        c as f32 / (e - s) as f32 / self.steps as f32 / self.cfg.dt_ms * 1000.0
    }

    pub fn n_neurons(&self) -> usize {
        self.neurons.len()
    }
}

/// 按层构宽度比例取层号（与实现层一致：单层区统一给 L3 槽位）。
fn layer_at(present: &[Layer], width: &[f32; 6], frac: f32) -> u8 {
    if present.is_empty() {
        return 2;
    }
    let total: f32 = present.iter().map(|l| width[l.idx()]).sum::<f32>().max(1e-6);
    let mut acc = 0.0;
    for &l in present {
        acc += width[l.idx()] / total;
        if frac <= acc {
            return l.idx() as u8;
        }
    }
    *present.last().unwrap() as u8
}

/// 四类细胞的电生理参数（与实现层同一套默认；框架不装"区特化"）。
fn params_for(t: CellType, dt_ms: f32) -> Params {
    // 校准口径：θ_eff = θ + β·a/(1−ρ) ≈ 2.1（见实现层的 AdLIF 说明）
    let (tau_m, tau_a, beta, theta) = match t {
        CellType::P => (20.0, 80.0, 0.18, 0.65),
        CellType::Pv => (10.0, 20.0, 0.10, 0.70),
        CellType::Som => (25.0, 150.0, 0.35, 0.65),
        CellType::Vip => (15.0, 60.0, 0.20, 0.70),
    };
    Params {
        tau_m,
        tau_a,
        refr_steps: 2,
        burst_steps: 2,
        t_inact_up: 0.02,
        t_inact_down: 0.98,
        beta,
        theta,
        gamma: 0.2,
        dt: dt_ms,
        burst: false,
    }
}

/// 声明式构造的便捷入口：`AreaSpec` 列表 + 默认配置 → 脑。
pub fn build(specs: &[AreaSpec]) -> Result<Brain, Vec<String>> {
    Brain::declare(specs, Config::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::area::{AreaClass, Layout, LocalMotif, Micro, Size, Slot};
    use crate::core::area::TopoDomain;

    static SRC_SLOTS: [Slot; 1] = [Slot::efferent("out", TopoDomain::Somatic)];
    static SRC_EFF: [crate::link::projection::Efferent; 1] = [crate::link::projection::Efferent::topo(
        "DST",
        "out",
        "in",
        Projection::Feedforward,
        1.0,
        8,
    )];
    static DST_SLOTS: [Slot; 1] = [Slot::afferent("in", TopoDomain::None)];

    fn src() -> AreaSpec {
        AreaSpec {
            name: "SRC",
            class: AreaClass::ThalamicRelay { tonic: true },
            topo: TopoDomain::Somatic,
            micro: Micro::relay(),
            size: Size::new(24, 6, 1),
            layout: Layout::mni([0.0, 0.0, 0.0], 0.5),
            motif: LocalMotif::relay(),
            slots: &SRC_SLOTS,
            topology: Topo::LINE,
            efferents: &SRC_EFF,
        }
    }

    fn dst() -> AreaSpec {
        AreaSpec {
            name: "DST",
            class: AreaClass::Cortex,
            topo: TopoDomain::None,
            micro: Micro::cortical(),
            size: Size::new(24, 6, 1),
            layout: Layout::mni([5.0, 0.0, 0.0], 0.5),
            motif: LocalMotif::cortical(),
            slots: &DST_SLOTS,
            topology: Topo::POINT,
            efferents: &[],
        }
    }

    #[test]
    fn declaration_builds_a_runnable_brain() {
        let mut b = build(&[src(), dst()]).unwrap();
        assert_eq!(b.n_neurons(), 2 * 30, "每区 24 E + 6 I");
        b.wire_local();
        // 给 SRC 一点驱动，跑 100 步
        for _ in 0..100 {
            let mut ext = vec![0.0; b.areas.len()];
            ext[0] = 8.0;
            b.step(&ext, 1.0);
        }
        let r_src = b.rate_hz(0);
        println!("[engine] SRC {r_src:.1} Hz  DST {:.1} Hz", b.rate_hz(1));
        assert!(r_src > 0.0, "被驱动的区必须发放");
        // 声明里有一条 SRC→DST 的投射，建边记录必须在
        assert_eq!(b.edges.len(), 1, "{:?}", b.edges);
        assert!(b.edges[0].synapses > 0, "声明应当生成突触");
    }

    static BAD_EFF: [crate::link::projection::Efferent; 1] = [crate::link::projection::Efferent::topo(
        "NOPE",
        "in",
        "in",
        Projection::Feedforward,
        1.0,
        4,
    )];

    #[test]
    fn bad_declaration_is_reported_not_panicked() {
        let mut bad = dst();
        bad.efferents = &BAD_EFF;
        match Brain::declare(&[src(), bad], Config::default()) {
            Ok(_) => panic!("坏声明不该建成"),
            Err(errs) => assert!(errs.iter().any(|e| e.contains("[缺靶区]")), "{errs:?}"),
        }
    }

    #[test]
    fn same_seed_is_deterministic() {
        let run = || {
            let mut b = build(&[src(), dst()]).unwrap();
            b.wire_local();
            for _ in 0..50 {
                let mut ext = vec![0.0; b.areas.len()];
                ext[0] = 8.0;
                b.step(&ext, 1.0);
            }
            b.spike_count.clone()
        };
        assert_eq!(run(), run(), "同一种子必须逐位可复现（判据的前提）");
    }
}
