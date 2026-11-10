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

// ============================================================================
// 神经元模型的**声明格式**（固定格式 + builder）
// ============================================================================

/// 神经元模型（当前支持的几种）。
///
/// 都实现同一个接口 `step(input) -> bool`（"这一步是否发放"），
/// 所以引擎不需要为每种模型写一套循环。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Model {
    /// **AdLIF**：泄漏积分 + 适应电流（默认；有 spike-frequency adaptation）。
    AdLif,
    /// **LIF**：AdLIF 去适应（`β = 0`）——数学上是它的特例，不是另一套代码。
    Lif,
    /// **Rate**：连续发放率单元（RNN 的味道）。输出是 `[0,1]` 的激活，
    /// 引擎按"以激活值为概率发放脉冲"接到同一套接口上。
    ///
    /// ⚠️ **这是模型选择，不是生物事实**：真实神经元不发"概率脉冲"。
    /// 用途是把人工神经网络那套单元接进来做对照（RNN/GRU 式单元），
    /// 所以它必须显式声明，不能当默认。
    Rate,
}

/// `Rate` 单元的激活函数。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
    Tanh,
    Relu,
    Sigmoid,
}

/// 神经元模型的**规格**。`const` 可构造，所以能直接写进区声明。
///
/// ```rust
/// use dreamkernel::core::cell::{Model, Spec};
///
/// const FAST: Spec = Spec::new(Model::AdLif).with_tau_m(12.0).with_theta(0.7);
/// const UNIT: Spec = Spec::new(Model::Rate).with_gain(2.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spec {
    pub model: Model,
    /// 膜时间常数（ms）。
    pub tau_m: f32,
    /// 适应电流时间常数（ms）。`Model::Lif` 会忽略它。
    pub tau_a: f32,
    /// 基础阈值。
    pub theta: f32,
    /// 适应强度（`β`）。`Lif` 恒 0。
    pub beta: f32,
    /// 适应对阈值的增益（`γ`）。
    pub gamma: f32,
    /// 不应期（步）。
    pub refr_steps: u8,
    /// 步长（ms）。
    pub dt: f32,
    /// 输入增益（`Rate` 单元用；脉冲模型忽略）。
    pub gain: f32,
    /// `Rate` 单元的激活函数。
    pub act: Act,
}

impl Spec {
    /// 起手式：给定模型，其余用该模型的默认值。
    pub const fn new(model: Model) -> Self {
        match model {
            Model::AdLif => Self {
                model,
                tau_m: 20.0,
                tau_a: 80.0,
                theta: 0.65,
                beta: 0.18,
                gamma: 0.2,
                refr_steps: 2,
                dt: 1.0,
                gain: 1.0,
                act: Act::Tanh,
            },
            Model::Lif => Self {
                model,
                tau_m: 20.0,
                tau_a: 80.0,
                theta: 0.65,
                beta: 0.0,
                gamma: 0.0,
                refr_steps: 2,
                dt: 1.0,
                gain: 1.0,
                act: Act::Tanh,
            },
            Model::Rate => Self {
                model,
                tau_m: 10.0,
                tau_a: 0.0,
                theta: 0.5,
                beta: 0.0,
                gamma: 0.0,
                refr_steps: 0,
                dt: 1.0,
                gain: 1.0,
                // 默认 ReLU：**静息 0 输入 → 0 发放**。用 tanh 会让静止态发 50%
                // （tanh(0)=0 映射到 [0,1] 是 0.5），那不是"静息"。
                act: Act::Relu,
            },
        }
    }

    // ---- builder：链式设置，全部 `const`（能写进 `AreaSpec`）----
    pub const fn with_tau_m(mut self, v: f32) -> Self {
        self.tau_m = v;
        self
    }
    pub const fn with_tau_a(mut self, v: f32) -> Self {
        self.tau_a = v;
        self
    }
    pub const fn with_theta(mut self, v: f32) -> Self {
        self.theta = v;
        self
    }
    pub const fn with_beta(mut self, v: f32) -> Self {
        self.beta = v;
        self
    }
    pub const fn with_dt(mut self, v: f32) -> Self {
        self.dt = v;
        self
    }
    pub const fn with_gain(mut self, v: f32) -> Self {
        self.gain = v;
        self
    }
    pub const fn with_act(mut self, a: Act) -> Self {
        self.act = a;
        self
    }
}

impl Default for Spec {
    fn default() -> Self {
        Self::new(Model::AdLif)
    }
}

/// `Rate` 单元的运行时状态：一阶低通 + 激活。
///
/// `v_t = v_{t-1} + dt/τ·(gain·I − v_{t-1})`，`y = act(v_t)`；
/// 引擎把 `y` 当作**发放概率**（ANN→SNN 的标准桥接，见 [`Model::Rate`] 的说明）。
#[derive(Clone, Copy, Debug)]
pub struct RateUnit {
    pub v: f32,
    /// 上一步的激活值（读出用）。
    pub y: f32,
    spec: Spec,
}

impl RateUnit {
    pub fn new(spec: Spec) -> Self {
        Self { v: 0.0, y: 0.0, spec }
    }

    /// 推进一步，返回激活值 `y ∈ [0,1]`。
    pub fn forward(&mut self, input: f32) -> f32 {
        let tau = self.spec.tau_m.max(1e-3);
        let a = (self.spec.dt / tau).clamp(0.0, 1.0);
        self.v += a * (self.spec.gain * input - self.v);
        self.y = match self.spec.act {
            Act::Tanh => 0.5 * (crate::num::math::tanh32(self.v) + 1.0),
            Act::Sigmoid => crate::num::math::sigmoid(self.v),
            Act::Relu => crate::num::math::relu(self.v).min(1.0),
        };
        self.y
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.y = 0.0;
    }
}
