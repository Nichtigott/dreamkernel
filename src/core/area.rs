//! 区类：一个脑区的**身份与类别**（唯一真值源）。
//!
//! 这里只有「一个区是什么」，没有「一个区怎么连」——连接在 `crate::link`，
//! 具体区的声明在 `crate::areas`。
//!
//! `AreaClass` 的四件事（驱动权重档 / 背景与稳态 / 拓扑域 / 紧张性驱动）都由它派生，
//! 以前这四件事是四个 `matches!(name, …)` 字符串名单散在三个文件里，
//! 加新核团（VPM）漏一处就被静默当成皮层区（docs/68 §12.6）。

use crate::core::layer::{Layer, LaminarProfile};
use crate::core::cell::CellType;
use crate::core::topo::{Rect, Topo};

/// 区内的连接 motif。四个参数描述"这个区的基础连接方式"。
#[derive(Debug, Clone, Copy)]
pub struct LocalMotif {
    /// L4→L3→L5 的前馈强度。
    pub feedforward: f32,
    /// L3 的局部递归强度（工作记忆/自联想）。
    pub recurrence: f32,
    /// L6 反馈到 L1/L3 的强度。
    pub feedback: f32,
}

impl LocalMotif {
    /// 一般皮层。
    pub const fn cortical() -> Self {
        Self {
            feedforward: 1.0,
            recurrence: 0.6,
            feedback: 0.5,
        }
    }

    /// 运动/前额：前馈入口弱，递归强。
    pub const fn motor() -> Self {
        Self {
            feedforward: 0.4,
            recurrence: 0.8,
            feedback: 0.4,
        }
    }

    /// CA3 式的自联想：递归极高。
    pub const fn autoassociative() -> Self {
        Self {
            feedforward: 0.5,
            recurrence: 1.5,
            feedback: 0.3,
        }
    }

    /// 枢纽（VTA、下丘脑）：强递归 + 中继。
    pub const fn hub() -> Self {
        Self {
            feedforward: 0.8,
            recurrence: 0.7,
            feedback: 0.6,
        }
    }

    /// **丘脑中继核**（LGN/MGN/VPL）：几乎没有局部递归兴奋（生物事实：中继细胞的兴奋来自
    /// 视网膜/外周传入，抑制来自 TRN 与局部中间神经元；不存在皮层式的 recurrent excitation）。
    /// 这是"视觉调制被中继自身的递归放大/自持"这一实现错误的修法（docs/63 §13）。
    pub const fn relay() -> Self {
        Self {
            feedforward: 1.0,
            recurrence: 0.0,
            feedback: 0.6,
        }
    }

    /// 门控（PAG、BNST）：低递归，强抑制。
    pub const fn gate() -> Self {
        Self {
            feedforward: 0.6,
            recurrence: 0.3,
            feedback: 0.5,
        }
    }

    /// DG 式的稀疏扩维：几乎不递归。
    pub const fn sparse() -> Self {
        Self {
            feedforward: 1.0,
            recurrence: 0.1,
            feedback: 0.2,
        }
    }

    /// **兴奋性前运动爆发神经元群（EBN，PPRF/riMLF）**（P0.5 S-1）。
    /// 事实：burst 细胞静息不放电（被 OPN 紧张性抑制钳住），扫视时成簇发放
    /// 300–600 Hz；池内相互兴奋是"点火后自持"的基底（J Neurophysiol 1998 的
    /// EBN 群体；OPN 撤钳后 burst 会持续 → 临床上 OPN 病变出现扫视振荡/凝视漂移）。
    /// 注：本文件的 motif 只在 `laminar_intra` 打开时生成区内边；生产配置（非 laminar）
    /// 下**池内**递归由 `spiking.rs` 的显式布线按对手池分开建（见 SC→EBN 段）。
    pub const fn burst() -> Self {
        Self {
            feedforward: 0.6,
            recurrence: 1.0,
            feedback: 0.3,
        }
    }

    /// **神经积分器（NI，NPH / INC）**（P0.5 S-2）：自持兴奋池。
    /// 事实：NI 神经元的活动 ∝ 眼位（位置阶跃），由 burst 的输入写入、由池内相互兴奋
    /// 维持、由慢适应缓慢泄放（Robinson 1972；Cannon & Robinson 1985；Leigh & Zee）。
    /// 同 `burst()`：池内递归在生产配置下由 `spiking.rs` 显式建。
    pub const fn integrator() -> Self {
        Self {
            feedforward: 1.0,
            recurrence: 1.8,
            feedback: 0.2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Area {
    pub name: &'static str,
    pub patches: usize,
    pub profile: LaminarProfile,
    pub motif: LocalMotif,
    /// 近似 MNI 坐标（mm）。右侧为正。
    pub pos: [f32; 3],
    /// 相对规模（0..1）：决定有效特征维数。
    pub width: f32,
    /// **显式声明**的类别（由 `AreaSpec::to_area()` 填入）。
    /// `None` = 手写字面量没声明，`class()` 会按名字表/层构回退。
    pub class: Option<AreaClass>,
    /// **显式声明**的拓扑域（同上）。
    pub topo: Option<TopoDomain>,
    /// 该区**自己声明**的出向投射（由 `AreaSpec::to_area()` 填入）。
    /// 建脑时从这里**自动生成边**——这就是"声明即生成"（用户 2026-11-10）。
    pub efferents: &'static [crate::link::projection::Efferent],
    /// 该区的**连接槽**（由 `AreaSpec::to_area()` 填入；手写字面量为空切片）。
    ///
    /// 建边层要靠它拿"这个端口暴露区的哪一块"（`Slot.region`），这样
    /// `add_edge` 就不用再手写 `DRG_LOWER` 这类常量（docs/74 §5）。
    pub slots: &'static [Slot],
    /// **显式声明**的形状（`Topo`）。默认 [`Topo::POINT`] = 无空间结构 = 旧的随机球布局。
    ///
    /// 只有非 `Point` 的区才会在建脑时按坐标铺开（并让距离→延迟生效）；
    /// **没声明的一律走旧路径**，所以加这个字段不会改变任何现有行为（docs/74、用户 2026-11-10）。
    pub topology: Topo,
}

// ============================================================================
// 脑区类别：**多态 + 派生的唯一真值源**（2026-11-09）
// ============================================================================

/// 拓扑域：一个区"映射的是什么"。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopoDomain {
    None,
    Visual,
    Auditory,
    Somatic,
}

/// 脑区类别。
///
/// **为什么要它**：丘脑感觉中继核与皮层在四件事上系统性不同，而这四件事原来是
/// **四个 `matches!(name, …)` 字符串名单散在三个文件里**：
/// ①丘脑-皮层是**驱动**（driver）、皮层-皮层是**调制**（modulator）⇒ 单突触权重档；
/// ②中继核静息放电 **1–5 Hz**、皮层 10–30 Hz ⇒ 背景倍率与稳态目标；
/// ③感觉中继**拓扑保序**投射（视网膜/躯体/音调拓扑）；
/// ④部分中继核有紧张性驱动。
///
/// 2026-11-09 实测事故：VPM（丘脑腹后内侧核，面部触觉中继）作为「丘脑补齐」加进区表后，
/// **四个名单一个都没进** ⇒ 被当成皮层区按 30 Hz 皮层式背景驱动、非拓扑广播到全身 S1，
/// 有效驱动 17,800 vs 身体中继 VPL 的 5,100，把躯体拓扑整个抹平（docs/68 §12.6）。
///
/// **规矩**：以后加核团只声明一次 `class`，四件事全部从它派生，不许再写名字名单。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AreaClass {
    /// 丘脑感觉中继。`tonic` = 是否给紧张性驱动。
    /// 生物：LGN/MGN/VPL/VPM 是感觉中继（有外周驱动）；VA/VL 是运动中继（无）。
    ThalamicRelay { tonic: bool },
    /// 皮层（含层构）。
    Cortex,
    /// 调质核团（LC/RAPHE/VTA/SNc/…）：慢、弥散、无拓扑。
    Modulatory,
    /// 其他皮质下核团（基底节、小脑、脑干、下丘脑、上丘…）。
    Subcortical,
    /// 皮质下**感觉中继**核团（薄束核 GR / 楔束核 CU / 三叉主感觉核 Pr5）。
    ///
    /// 生物：它们是被外周传入驱动的中继，静息 5–20 Hz 但**不是皮层式的背景轰击**；
    /// 与 `ThalamicRelay` 的区别只在于"不直接驱动皮层"（GR/CU 的靶是 VPL，Pr5 的靶是 VPM），
    /// 所以不算 driver（单突触权重档由 `Arbor.w` 显式给，不靠类别猜）。
    SubcorticalRelay,
    /// **一级传入**的胞体所在（背根神经节 DRG）。
    ///
    /// 生物事实：DRG 神经元是**机械/化学门控**的，没有自发发放——不刺激就完全静默
    /// （这点和所有中枢神经元都不同）。所以背景驱动与稳态目标都是 0：
    /// 给它皮层档的 30 Hz 背景，等于凭空造出"皮肤一直在被摸"的信号。
    PrimaryAfferent,
    /// 运动核团（TRIG/FAC/OMN…）。
    Motor,
}

impl AreaClass {
    /// 是否是丘脑-皮层**驱动**（生物：driver vs modulator）。决定单突触权重档。
    pub fn is_driver(self) -> bool {
        matches!(self, AreaClass::ThalamicRelay { .. })
    }
    /// 是否是**感觉中继**（丘脑中继 + 皮质下中继）。决定背景/稳态档。
    pub fn is_relay(self) -> bool {
        matches!(self, AreaClass::ThalamicRelay { .. } | AreaClass::SubcorticalRelay)
    }
    /// 背景驱动倍率（相对皮层）。中继核静息放电来自外周传入 + TRN，不是皮层式背景轰击。
    pub fn bg_scale(self) -> f32 {
        if self.is_relay() {
            0.15
        } else if self == AreaClass::PrimaryAfferent {
            0.0
        } else {
            1.0
        }
    }
    /// 稳态发放率目标（棘波/tick）。中继 0.006 = 3 Hz；皮层 0.06 = 30 Hz。
    pub fn homeo_target(self) -> f32 {
        if self.is_relay() {
            0.006
        } else if self == AreaClass::PrimaryAfferent {
            0.0
        } else {
            0.06
        }
    }
    /// 是否给紧张性驱动。
    pub fn has_tonic(self) -> bool {
        matches!(self, AreaClass::ThalamicRelay { tonic: true })
    }
}

impl Area {
    /// **显式声明**的类别；`None` = 没声明。回退规则（Glasser 表 / 名字表 / 层构）
    /// 属于**实现层**（哪些名字是皮层是这个脑的约定，不是通用规律），见 `docs/75`。
    pub fn class_declared(&self) -> Option<AreaClass> {
        self.class
    }

    /// **显式声明**的拓扑域；`None` = 没声明（回退同样在实现层）。
    pub fn topo_declared(&self) -> Option<TopoDomain> {
        self.topo
    }

    /// 该区是否声明了某个方向的槽。
    pub fn has_slot(&self, name: &str, kind: SlotKind) -> bool {
        self.slot(name).map(|s| s.kind == kind).unwrap_or(false)
    }

    /// 按名字取槽。
    pub fn slot(&self, name: &str) -> Option<&Slot> {
        self.slots.iter().find(|s| s.name == name)
    }
}

// ============================================================================
// 区对象（AreaSpec）：一个区的**完整声明**——微结构 / 规模 / 布置 / 连接槽。
// 与上面的 `Area` 的区别：`Area` 是运行时的区表条目，`AreaSpec` 是**构造它之前的声明**，
// `to_area()` 产出 `Area`。这里只说"这个区是什么"；"谁连谁"在 `crate::link`。
// ============================================================================
/// 连接槽的方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    /// 进（该区接收的输入）。
    Afferent,
    /// 出（该区发出的投射）。
    Efferent,
}

/// 连接槽：区对外的一个**具名接口**。
///
/// 生物依据：真实核团的输入/输出是按**来源与靶**区分的，不是"一个区一个端口"。
/// 例如 VPL 收身体（脊髓-内侧丘系）、VPM 收面部（三叉丘系），两者的输出都到 S1
/// 但占据不同的躯体拓扑域 —— 所以槽必须能带拓扑域。
#[derive(Debug, Clone, Copy)]
pub struct Slot {
    pub name: &'static str,
    pub kind: SlotKind,
    /// 该槽是否保序（拓扑）。`TopoDomain::None` = 非拓扑。
    pub topo: TopoDomain,
    /// **微观插槽**：这个端口暴露区的哪一块（`None` = 整区）。
    ///
    /// 这是"外部连接进来的微观插槽"（用户 2026-11-10）：一条投射可以只接到
    /// `DRG` 的腰骶段或 `S1_3b` 的臂带，而不是整个区。坐标与 `Topo` 同一套
    /// （归一化 `Rect`），所以声明层和建边层不需要各写一份几何。
    pub region: Option<Rect>,
    /// 落在哪些层（空 = 默认：入向 L4、出向 L2/3）。
    pub layers: &'static [Layer],
    /// 落在哪些细胞类型（空 = 默认：全部）。
    pub cells: &'static [CellType],
    /// 会聚数（`None` = 由 link/建边的 fan_in 决定）。
    pub fan_in: Option<usize>,
}

impl Slot {
    /// 普通入向槽（无子区域、无细胞/层限定）——旧行为。
    pub const fn afferent(name: &'static str, topo: TopoDomain) -> Self {
        Self::raw(name, SlotKind::Afferent, topo)
    }
    /// 普通出向槽——旧行为。
    pub const fn efferent(name: &'static str, topo: TopoDomain) -> Self {
        Self::raw(name, SlotKind::Efferent, topo)
    }

    const fn raw(name: &'static str, kind: SlotKind, topo: TopoDomain) -> Self {
        Self { name, kind, topo, region: None, layers: &[], cells: &[], fan_in: None }
    }

    /// **微观化**：把该槽限定到区的某个子区域（`Rect` 与该区的 `Topo` 同一套坐标）。
    pub const fn with_region(mut self, r: Rect) -> Self {
        self.region = Some(r);
        self
    }
    /// 限定落点在哪些层。
    pub const fn with_layers(mut self, l: &'static [Layer]) -> Self {
        self.layers = l;
        self
    }
    /// 限定落点在哪些细胞类型。
    pub const fn with_cells(mut self, c: &'static [CellType]) -> Self {
        self.cells = c;
        self
    }
    /// 声明会聚数。
    pub const fn with_fan_in(mut self, n: usize) -> Self {
        self.fan_in = Some(n);
        self
    }
}

/// 微结构：神经元构成。
#[derive(Debug, Clone, Copy)]
pub struct Micro {
    /// 层构（每层相对宽度；L4 = 0.05 表示无颗粒皮层，0 表示该层不存在）。
    pub profile: LaminarProfile,
    /// 抑制性占比（事实来源写在这里，不是可调参数）。
    /// 皮层 GABA 能 ~20%；上丘浅层 ~12%（docs/63 §13 用户要求按事实定 E/I）。
    pub inh_frac: f32,
}

impl Micro {
    /// 皮层（颗粒型感觉皮层）。
    pub const fn cortical() -> Self {
        Self { profile: LaminarProfile::sensory(), inh_frac: 0.20 }
    }
    /// 无颗粒皮层（运动/前额）。
    pub const fn agranular() -> Self {
        Self { profile: LaminarProfile::agranular(), inh_frac: 0.20 }
    }
    /// 丘脑中继：单层、无层构（profile 由 `subcortical()` 给）。
    pub const fn relay() -> Self {
        Self { profile: LaminarProfile::subcortical(), inh_frac: 0.20 }
    }
}

/// 规模。
#[derive(Debug, Clone, Copy)]
pub struct Size {
    pub e: usize,
    pub i: usize,
    pub patches: usize,
}

impl Size {
    pub const fn new(e: usize, i: usize, patches: usize) -> Self {
        Self { e, i, patches }
    }
}

/// 布置。
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    /// 近似 MNI 坐标（mm），右侧为正。
    pub mni: [f32; 3],
    /// 相对规模（0..1）：决定有效特征维数。
    pub width: f32,
}

impl Layout {
    pub const fn mni(mni: [f32; 3], width: f32) -> Self {
        Self { mni, width }
    }
}

/// **区对象**：一个区的完整声明。具体区用它构造，`to_area()` 产出运行时的 [`Area`]。
#[derive(Debug, Clone, Copy)]
pub struct AreaSpec {
    pub name: &'static str,
    pub class: AreaClass,
    pub topo: TopoDomain,
    pub micro: Micro,
    pub size: Size,
    pub layout: Layout,
    pub motif: LocalMotif,
    pub slots: &'static [Slot],
    /// 该区的**形状**。`Topo::POINT` = 无空间结构（旧行为）。
    pub topology: Topo,
    /// 该区**自己声明**的出向投射（见 [`crate::link::projection::Efferent`]）。
    /// 空切片 = 只靠中央 `add_edge` 声明（历史写法）。
    pub efferents: &'static [crate::link::projection::Efferent],
}

impl AreaSpec {
    /// 构造运行时的区表条目。
    pub fn to_area(&self) -> Area {
        Area {
            name: self.name,
            patches: self.size.patches,
            profile: self.micro.profile,
            motif: self.motif,
            pos: self.layout.mni,
            width: self.layout.width,
            class: Some(self.class),
            topo: Some(self.topo),
            efferents: self.efferents,
            slots: self.slots,
            topology: self.topology,
        }
    }

    pub fn slot(&self, name: &str) -> Option<&Slot> {
        self.slots.iter().find(|s| s.name == name)
    }

    /// 该区是否声明了某个方向的槽。
    pub fn has_slot(&self, name: &str, kind: SlotKind) -> bool {
        self.slot(name).map(|s| s.kind == kind).unwrap_or(false)
    }
}

// ============================================================================
// 第 3 层：脑部构造
// ============================================================================

