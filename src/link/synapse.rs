//! **突触的声明**：权重分布 + 可塑性规则。
//!
//! 一条投射（[`crate::link::projection::Efferent`]）声明"从哪到哪"，这里声明
//! **这条投射上的突触长什么样**：
//!
//! ```rust
//! use dreamkernel::link::projection::{Efferent, Projection};
//! use dreamkernel::link::synapse::{Plasticity, Synapse};
//!
//! // 固定权重（`Efferent::topo` 的默认就是它）
//! let plain = Efferent::topo("B", "out", "in", Projection::Feedforward, 2.0, 16);
//! assert_eq!(plain.synapse.rule, Plasticity::Frozen);
//!
//! // 权重有离散度 + 成对 STDP（把声明挂到投射上）
//! let declared = Synapse::stdp(2.0, 0.02, 0.021).with_dist(
//!     dreamkernel::link::synapse::WeightDist::Lognormal(0.3),
//! );
//! let link = Efferent { synapse: declared, ..plain };
//! assert!(link.synapse.rule.is_plastic());
//! ```
//!
//! 两个刻意的设计：
//! * **默认是固定的**（`Synapse::FIXED`）：可塑性必须显式声明，否则改引擎默认值会
//!   悄悄改掉所有已有模型的行为。
//! * 权重的离散度用 **lognormal**：突触权重是正数、右偏，lognormal 才对齐生物学；
//!   用正态会抽到负权重（那是"抑制"，会改变符号，不是噪声）。

use crate::num::math::Lcg;

/// 权重分布（围绕均值 `mean`）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightDist {
    /// 全部相同。
    Fixed,
    /// 乘性抖动：`w · exp(σ·N(0,1))`，`σ` 相对离散度。
    Lognormal(f32),
    /// 均匀分布：`mean · U(1−d, 1+d)`。
    Uniform(f32),
}

impl WeightDist {
    /// 采样一个权重倍率（均值≈1）。
    pub fn sample(self, rng: &mut Lcg) -> f32 {
        match self {
            WeightDist::Fixed => 1.0,
            WeightDist::Lognormal(sigma) if sigma > 0.0 => {
                // Box–Muller（用框架自带的 Lcg，保证同种子逐位可复现）
                let u1 = rng.next_f32().max(1e-7);
                let u2 = rng.next_f32();
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                (sigma * g).exp()
            }
            WeightDist::Lognormal(_) => 1.0,
            WeightDist::Uniform(d) if d > 0.0 => 1.0 + d * (2.0 * rng.next_f32() - 1.0),
            WeightDist::Uniform(_) => 1.0,
        }
    }
}

/// 可塑性规则。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Plasticity {
    /// 固定权重（默认）。
    Frozen,
    /// **成对 STDP**：pre 先于 post → 增强（LTP）；post 先于 pre → 削弱（LTD）。
    ///
    /// `Δw = +a_plus·x_pre`（post 发放时）、`Δw = −a_minus·x_post`（pre 发放时），
    /// 迹按 `exp(−dt/τ)` 衰减，权重夹在 `[0, w_max]`。
    Stdp {
        a_plus: f32,
        a_minus: f32,
        tau_plus_ms: f32,
        tau_minus_ms: f32,
        w_max: f32,
    },
}

impl Plasticity {
    pub fn is_plastic(self) -> bool {
        !matches!(self, Plasticity::Frozen)
    }
}

/// 一条投射上的**突触声明**。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Synapse {
    /// 均值权重倍率（乘在源区类别档位上）。
    pub mean: f32,
    /// 权重的离散度。
    pub dist: WeightDist,
    /// 可塑性规则。
    pub rule: Plasticity,
}

impl Synapse {
    /// 固定权重（最常用的默认）。
    pub const FIXED: Synapse = Synapse {
        mean: 1.0,
        dist: WeightDist::Fixed,
        rule: Plasticity::Frozen,
    };

    /// 固定权重 + 指定均值。
    pub const fn fixed(mean: f32) -> Self {
        Self { mean, dist: WeightDist::Fixed, rule: Plasticity::Frozen }
    }

    /// 权重带 lognormal 抖动（`sigma` = 相对离散度），可塑性仍关闭。
    pub const fn noisy(mean: f32, sigma: f32) -> Self {
        Self { mean, dist: WeightDist::Lognormal(sigma), rule: Plasticity::Frozen }
    }

    /// 固定均值 + 成对 STDP。
    pub const fn stdp(mean: f32, a_plus: f32, a_minus: f32) -> Self {
        Self {
            mean,
            dist: WeightDist::Fixed,
            rule: Plasticity::Stdp {
                a_plus,
                a_minus,
                tau_plus_ms: 20.0,
                tau_minus_ms: 20.0,
                w_max: mean * 4.0,
            },
        }
    }

    pub const fn with_dist(mut self, d: WeightDist) -> Self {
        self.dist = d;
        self
    }

    pub const fn with_rule(mut self, r: Plasticity) -> Self {
        self.rule = r;
        self
    }

    /// 这个突触的初始权重（按分布采一次）。
    pub fn sample(&self, rng: &mut Lcg) -> f32 {
        self.mean * self.dist.sample(rng)
    }
}

impl Default for Synapse {
    fn default() -> Self {
        Synapse::FIXED
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributions_are_centred_on_one_and_deterministic() {
        let mut a = Lcg::new(3);
        let mut b = Lcg::new(3);
        let dist = WeightDist::Lognormal(0.3);
        let (mut sum, mut n) = (0.0f64, 0);
        for _ in 0..4000 {
            let x = dist.sample(&mut a);
            assert_eq!(x, dist.sample(&mut b), "同种子必须逐位一致");
            assert!(x > 0.0, "lognormal 永远为正（负权重是抑制，不是噪声）");
            sum += x as f64;
            n += 1;
        }
        let mean = sum / n as f64;
        // 乘性抖动的期望是 exp(σ²/2)，σ=0.3 → 1.046，不是 1.0
        assert!((mean - (0.045f64).exp()).abs() < 0.06, "均值 {mean}");
        assert_eq!(WeightDist::Fixed.sample(&mut a), 1.0);
        let u = WeightDist::Uniform(0.2);
        for _ in 0..1000 {
            let x = u.sample(&mut a);
            assert!((0.79..1.21).contains(&x), "{x}");
        }
    }

    #[test]
    fn synapse_builders_are_const_and_sane() {
        assert_eq!(Synapse::FIXED.rule, Plasticity::Frozen);
        assert!(!Synapse::FIXED.rule.is_plastic());
        let s = Synapse::stdp(2.0, 0.02, 0.021);
        assert!(s.rule.is_plastic());
        assert_eq!(s.mean, 2.0);
        match s.rule {
            Plasticity::Stdp { w_max, .. } => assert!((w_max - 8.0).abs() < 1e-6),
            _ => panic!(),
        }
        let mut rng = Lcg::new(1);
        assert_eq!(s.sample(&mut rng), 2.0, "Fixed 分布下采样就是均值");
    }
}
