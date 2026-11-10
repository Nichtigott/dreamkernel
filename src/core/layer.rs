//! 层构：皮层六层的相对宽度与命名。
//!
//! 层是**结构事实**，不是可调参数：颗粒型感觉皮层 L4 完整；无颗粒皮层（运动/前额）
//! L4 压到 0.05 但不删除；海马只有部分层（width=0 表示该层不存在）。

use std::fmt;

/// 皮层 6 层。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layer {
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
}

impl Layer {
    pub const ALL: [Layer; 6] = [
        Layer::L1,
        Layer::L2,
        Layer::L3,
        Layer::L4,
        Layer::L5,
        Layer::L6,
    ];

    pub const fn idx(self) -> usize {
        match self {
            Layer::L1 => 0,
            Layer::L2 => 1,
            Layer::L3 => 2,
            Layer::L4 => 3,
            Layer::L5 => 4,
            Layer::L6 => 5,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Layer::L1 => "L1",
            Layer::L2 => "L2",
            Layer::L3 => "L3",
            Layer::L4 => "L4",
            Layer::L5 => "L5",
            Layer::L6 => "L6",
        }
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// 每层的相对宽度。0 表示该层不存在（只有海马会这样）。
#[derive(Debug, Clone, Copy)]
pub struct LaminarProfile {
    pub width: [f32; 6],
}

impl LaminarProfile {
    /// 颗粒型感觉皮层：L4 完整。
    pub const fn sensory() -> Self {
        Self {
            width: [0.25, 0.9, 1.0, 1.0, 1.0, 0.8],
        }
    }

    /// 无颗粒皮层（运动/前额）：L4 压到 0.05，但**不删除**。
    pub const fn agranular() -> Self {
        Self {
            width: [0.25, 0.9, 1.0, 0.05, 1.0, 0.8],
        }
    }

    /// 分化不良（dysgranular）：层 IV 存在但薄。**中间型**，不能挤到颗粒或无颗粒里。
    /// 解剖依据：运动前区/前额/部分扣带与颞极属此型（见 world/data/rm41_regions.csv
    /// 的 `laminar_profile` 列 + docs/53 §4.1）。0.45 这个具体宽度是**模型**选择（不是实测值），
    /// 不是测量值：它决定该区神经元摊到 L4 的比例（spiking.rs 按层宽比例分层）。
    pub const fn dysgranular() -> Self {
        Self {
            width: [0.25, 0.9, 1.0, 0.45, 1.0, 0.8],
        }
    }

    /// 海马亚区：单层（占 L3 槽位）。
    pub const fn hippocampal() -> Self {
        Self {
            width: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    }

    /// 皮质下核团：单层（同样占 L3 槽位）。
    pub const fn subcortical() -> Self {
        Self {
            width: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn is_present(&self, layer: Layer) -> bool {
        self.width[layer.idx()] > 0.01
    }

    pub fn nodes_per_patch(&self) -> usize {
        Layer::ALL
            .iter()
            .filter(|l| self.is_present(**l))
            .count()
    }

    pub fn present_layers(&self) -> Vec<Layer> {
        Layer::ALL
            .iter()
            .copied()
            .filter(|l| self.is_present(*l))
            .collect()
    }
}

