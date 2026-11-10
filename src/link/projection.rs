//! 投射与边的类型：区际投射方向、compartment、边、微观规格、实测连接组记录。
//!
//! 两层抽象：
//! - `Edge` 是**团→团**的骨架：区+patch+层 → 区+patch+层（`from`/`to` 两个 compartment）
//!   + 方向 `Projection` + 一个权重。
//! - `Arbor` 是挂在 `Edge` 上的**微观规格**：哪群细胞 → 哪群细胞、群体内的哪一段、
//!   保不保序、每个靶细胞会聚多少、权重倍率。没写（`Arbor::coarse()`）= 旧的团→团行为，
//!   于是"没声明的边逐字节不变"——改 link 层不会偷偷改动力学。

use crate::core::layer::Layer;
use crate::core::cell::CellType;
use crate::core::topo::{Map, Rect};

/// 区间投射方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    Feedforward,
    Feedback,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Compartment {
    pub area: usize,
    pub patch: usize,
    pub layer: Layer,
    pub width: f32,
}

/// 群体内的**一段**（归一化 0..1，按该细胞类型内的索引顺序，0 = 索引最小端）。
///
/// 为什么需要它：真实的投射不是"整区到整区"。DRG 的**下半身纤维**只上行到薄束核 GR、
/// **上半身纤维**只到楔束核 CU；VPL 的臂区只到 S1 的臂带。以前这些子群体在 link 层
/// 没有表示法，只能整区广播 ⇒ 臂的信号会从腿的落点出来（静默错位，没有任何机制报错）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub lo: f32,
    pub hi: f32,
}

impl Span {
    /// 整个群体。
    pub const ALL: Span = Span { lo: 0.0, hi: 1.0 };

    pub const fn new(lo: f32, hi: f32) -> Self {
        Self { lo, hi }
    }

    /// 占群体的比例。
    pub fn len(self) -> f32 {
        (self.hi - self.lo).max(0.0)
    }

    pub fn is_all(self) -> bool {
        self.lo <= 0.0 && self.hi >= 1.0
    }

    /// 一维子段 → `Rect`（`v` 全开）。`Arbor` 统一用 `Rect` 表达子区域，
    /// 但注入侧/旧常量仍然用 `Span`（一维更直观），两边靠这个互转。
    pub const fn to_rect(self) -> Rect {
        Rect::of_u(self.lo, self.hi)
    }

    /// 该 span 在群体里的 `[起, 止)` 索引（群体规模 `n`）。
    ///
    /// **两端都用四舍五入**，这样相邻的 span 在边界上取到同一个索引，不会重叠也不会漏一个：
    /// `[0,0.56)` 的止与 `[0.56,1)` 的起都是 `round(0.56·n)`。
    /// （原来止用 ceil、起用 floor，n=36 时 0.56 会同时属于两个 span。）
    /// 非空 span 至少给 1 个索引——否则小规模下整条投射会静默消失。
    pub fn index_range(self, n: usize) -> (usize, usize) {
        if n == 0 {
            return (0, 0);
        }
        let scale = |x: f32| ((x.clamp(0.0, 1.0) * n as f32).round() as usize).min(n);
        let (lo, hi) = (scale(self.lo), scale(self.hi));
        if hi <= lo && !self.is_empty() {
            (lo.min(n - 1), (lo.min(n - 1) + 1).min(n))
        } else {
            (lo, hi.max(lo))
        }
    }

    /// 空 span（hi <= lo）。
    pub fn is_empty(self) -> bool {
        self.hi <= self.lo
    }
}

/// 一条投射的**微观规格**：不只说"整区到整区"，还要说"哪群细胞到哪群细胞"。
///
/// 字段与"团→团"缺口的对应关系：
/// - `from`/`to`（区）、`from_layer`/`to_layer`（层）：已在 `Edge.from`/`Edge.to` 两个
///   compartment 里，这里不重复。
/// - `from_cell`/`to_cell`：**细胞类型**（锥体/PV/SOM/VIP）。`None` = 用现有默认
///   （源 = 锥体群；靶 = 该 compartment 的全部类型，E 全权重、I ×0.7）。
/// - `from_rect`/`to_rect`：群体内的子区域（`Rect`；一维拓扑只用 `u` 轴）。`None` = 整个群体。
/// - `topo`：保序（视网膜/躯体/音调拓扑）。在 span 内按**相对位置**对齐，不是整区对齐。
/// - `fan_in`：每个靶细胞收多少源细胞；`0` = 用 builder 的默认目标扇入。
/// - `w`：单突触权重倍率；`1.0` = 用方向档（丘脑-皮层 2.0 / 其余 0.25）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arbor {
    /// 两端拓扑怎么对齐。`AllToAll` = **没声明几何** → 走旧的 span 窗口。
    /// 只有两端都是二维拓扑（`dims()==2`）且 `map.is_topographic()` 时，
    /// 建边才走"按坐标配对"的路径。
    pub map: Map,
    pub from_cell: Option<CellType>,
    pub from_rect: Option<Rect>,
    pub to_cell: Option<CellType>,
    pub to_rect: Option<Rect>,
    pub topo: bool,
    pub fan_in: usize,
    pub w: f32,
}

impl Arbor {
    /// "什么都没说" = 旧的团→团行为。builder 见到它就跳过微观路径。
    pub const fn coarse() -> Self {
        Self {
            map: Map::AllToAll,
            from_cell: None,
            from_rect: None,
            to_cell: None,
            to_rect: None,
            topo: false,
            fan_in: 0,
            w: 1.0,
        }
    }

    pub fn is_coarse(self) -> bool {
        self == Self::coarse()
    }

    /// 拓扑投射：源 span → 靶 span 保序对齐，每个靶细胞 `fan_in` 条。
    ///
    /// 参数用一维 `Span`（最常见），内部存成 `Rect` 的 `u` 轴。
    pub const fn topo(from: Span, to: Span, fan_in: usize) -> Self {
        Self {
            map: Map::AllToAll,
            from_cell: None,
            from_rect: Some(from.to_rect()),
            to_cell: None,
            to_rect: Some(to.to_rect()),
            topo: true,
            fan_in,
            w: 1.0,
        }
    }

    /// 只限定源/靶子段（非拓扑，在段内随机落点）。
    pub const fn span(from: Span, to: Span) -> Self {
        Self {
            map: Map::AllToAll,
            from_cell: None,
            from_rect: Some(from.to_rect()),
            to_cell: None,
            to_rect: Some(to.to_rect()),
            topo: false,
            fan_in: 0,
            w: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeKind {
    /// 区内边：层 → 层。
    Intra { from_layer: Layer, to_layer: Layer },
    /// 区际边：由 connectome 决定的方向。
    Inter { projection: Projection },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
    pub kind: EdgeKind,
    /// 传导延迟，量化到 {0,1,2,4,8} 个 tick（1 tick = 1 ms 模型时间）。
    pub delay: u32,
    /// 微观规格。`Arbor::coarse()` = 这条边只声明了"区+层 → 区+层"（旧行为）。
    pub arbor: Arbor,
}

/// 一条**出向投射声明**：属于某个区（`AreaSpec.efferents`）。
///
/// 与 [`crate::link::validate::Link`] 的关系：`Link` 是装配层的"两边都写全"的写法
/// （用于 `BrainAssembly` 的独立校验）；`Efferent` 是"一个区自己声明我发到哪"的写法，
/// 建脑时由它**自动生成边**（不需要中央清单）。两者字段一一对应，可以互相转换。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Efferent {
    /// 靶区名。
    pub to: &'static str,
    /// 本区哪个**出向**槽（源子区域从它的 `region` 取）。
    pub from_slot: &'static str,
    /// 靶区哪个**入向**槽（靶子区域从它的 `region` 取）。
    pub to_slot: &'static str,
    pub projection: Projection,
    /// 两端拓扑怎么对齐。
    pub map: Map,
    /// 单突触权重倍率（乘在源区类别档位上；0 = 用默认）。
    pub w: f32,
    /// 会聚数；`None` = 用槽声明的，再没有用默认 32。
    pub fan_in: Option<usize>,
}

impl Efferent {
    /// 最常用的一种：拓扑投射、同一套对齐方式。
    pub const fn topo(
        to: &'static str,
        from_slot: &'static str,
        to_slot: &'static str,
        projection: Projection,
        w: f32,
        fan_in: usize,
    ) -> Self {
        Self { to, from_slot, to_slot, projection, map: Map::Identity, w, fan_in: Some(fan_in) }
    }
}

/// 实测的**皮层-皮层**投射（来自 world/connectome_rm41.csv，无向拓扑）。
///
/// 它**不进** `Graph::edges`：因为 a→b 的突触不是"整条投射一个权重"，而是要在
/// 靶区神经元上按 `w` 分配一个固定的区际扇入预算（见 spiking.rs 的
/// `cortico_fan` 两级分配与 ）。放进 edges 会让每条边按 patch 复制、
/// 突触数随区数线性膨胀（那是 v1 的老写法）。
///
/// 区表 v3 起 `src/dst` 存的是 **RM region 编号**（不是 180 个 Glasser parcel 索引）：
/// 实测表是 41 个 RM 区粒度的，一个 RM 区对应多个 parcel；两级分配先按 region 分预算，
/// 再在 region 内按距离抽 parcel（见 `Graph::regions`）。
#[derive(Debug, Clone, Copy)]
pub struct ConnEdge {
    pub src_region: usize,
    pub dst_region: usize,
    /// true = 上行（落 L4），false = 下行（落 L1/L6）。方向不是实测，是按推断层级分的，
    /// 可用 CORTICO_FLAT=1 全部按 ff 处理（消融）。
    pub ff: bool,
    /// 方向证据：该**有向边**的方向被人类证据支持且置信 ≥ medium
    /// （CSV `dir_inferred∈{ff,fb}`、`dir_conf∈{high,medium}`，且与 `ff` 一致）。
    /// 默认 false（无证据）。只在 `LAMINAR_INTER_EVID=1` 时参与层靶向（"扫雷"增量模式）。
    pub dir_evid: bool,
    /// 实测束长（mm）。本轮不用于延迟（Uniform 模式所有突触 = 1 tick），留给 W2b。
    pub dist_mm: f32,
    /// 实测权重（列归一化矩阵的份额，见 ）
    pub w_meas: f32,
    /// 模型权重 exp(-λ·dist_mm)，λ=0.150 mm^-1（Ercsey-Ravasz 2013），消融用
    pub w_edr: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_index_range_is_half_open_and_clamped() {
        assert_eq!(Span::ALL.index_range(10), (0, 10));
        assert_eq!(Span::new(0.0, 0.5).index_range(10), (0, 5));
        assert_eq!(Span::new(0.5, 1.0).index_range(10), (5, 10));
        // 相邻 span 在边界上取同一个索引（不重叠、不漏）：两端都四舍五入
        assert_eq!(Span::new(0.0, 0.56).index_range(36), (0, 20));
        assert_eq!(Span::new(0.56, 1.0).index_range(36), (20, 36));
        assert_eq!(Span::new(0.62, 1.0).index_range(10), (6, 10));
        // 非空 span 至少 1 个索引（小规模下不许静默消失）
        assert_eq!(Span::new(0.4, 0.6).index_range(2), (1, 2));
        // 越界参数被夹住，不会 panic，也不会返回 lo > hi
        assert_eq!(Span::new(-1.0, 2.0).index_range(4), (0, 4));
        let (lo, hi) = Span::new(0.8, 0.2).index_range(10);
        assert!(lo <= hi, "反序 span 也必须给出 lo<=hi（空区间）");
        // n = 0 不 panic
        assert_eq!(Span::ALL.index_range(0), (0, 0));
    }

    #[test]
    fn arbor_coarse_is_the_identity() {
        assert!(Arbor::coarse().is_coarse());
        // 任何一项改动都不再是 coarse（否则"没声明"和"声明了"会混淆）
        assert!(!Arbor { fan_in: 1, ..Arbor::coarse() }.is_coarse());
        assert!(!Arbor { w: 2.0, ..Arbor::coarse() }.is_coarse());
        assert!(!Arbor { topo: true, ..Arbor::coarse() }.is_coarse());
        assert!(!Arbor::topo(Span::ALL, Span::ALL, 32).is_coarse());
        assert!(!Arbor::span(Span::new(0.0, 0.5), Span::ALL).is_coarse());
    }
}
