//! 细胞类型的**词汇表**：锥体 / PV / SOM / VIP。
//!
//! 只有分类与索引约定，没有动力学——动力学属于实现层。
//! `link` 层要用它声明"源/靶是哪群细胞"，所以它必须在框架里（零依赖）。

/// 皮层四类细胞。当前只有电生理差异 + 类型特异的局部连接；
/// 层室/树突靶向还没接。
///
/// 住在 `core` 而不是 `sim::spiking`：`link` 层的微观投射要按**细胞类型**声明源/靶
/// （"L4 → L2/3 锥体"、"LGN 中继 → PV"），那是图结构的事，不该让 link 依赖 sim。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellType {
    /// 锥体（兴奋性）
    P,
    /// PV+ 快放电（胞体抑制，强前馈）
    Pv,
    /// SOM+（树突抑制，慢、弱）
    Som,
    /// VIP+（去抑制：抑制 SOM）
    Vip,
}

impl CellType {
    pub fn is_inh(self) -> bool {
        !matches!(self, CellType::P)
    }
    pub fn idx(self) -> usize {
        match self {
            CellType::P => 0,
            CellType::Pv => 1,
            CellType::Som => 2,
            CellType::Vip => 3,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            CellType::P => "P",
            CellType::Pv => "PV",
            CellType::Som => "SOM",
            CellType::Vip => "VIP",
        }
    }
}
