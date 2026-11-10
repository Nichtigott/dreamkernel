//! AdLIF 神经元：每神经元自己的时间常数、适应与阈值。
//!
//! 对应 docs/07 §6（AdLIF）与 docs/09 §13（细胞级记忆）。
//! 与文档的差别：参数从"每群体一套"改成"每神经元一套"——
//! H-N1 要检验异质时间常数是否优于同质。
//!
//! 动力学（`dt` 为步长，单位 ms）：
//!
//! ```text
//! v_t = α·v_{t-1} + (1-α)·I_t − β·a_{t-1}
//! a_t = ρ·a_{t-1} + (1-ρ)·s_{t-1}
//! s_t = Θ(v_t − θ₀ − γ·a_t)
//! ```
//!
//! 其中 `α = exp(−dt/τ_m)`，`ρ = exp(−dt/τ_a)`。
//! `a` 是"神经元自己的记忆"：它对最近发放积分，抬高有效阈值，
//! 于是同一个输入，这一次和上一次响应不同。

// `CellType` 住在框架 crate（`link` 层要按细胞类型声明投射），这里只转出。
pub use crate::core::cell::CellType;

/// 绝对不应期（步，dt=1ms）。生物：1–2 ms，对应最高发放率 500–1000 Hz。
/// 没有它神经元可以每步发放（1000 Hz），与生物事实不符。
pub const REFRACTORY_STEPS: u8 = 1;

/// 单个神经元的参数。比 docs/07 的"群体参数"多了两个自由度：
/// `tau_m`/`tau_a` 可以让每个神经元不同（H-N1 的自变量）。
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// 膜时间常数（ms）。
    pub tau_m: f32,
    /// 适应电流时间常数（ms），应大于 `tau_m`。
    pub tau_a: f32,
    /// 绝对不应期（步数，由 SpikingBrain 按 TimeBase 设定；dt=1ms 时 = 2）
    pub refr_steps: u8,
    /// burst 降阈窗口（步数）
    pub burst_steps: u8,
    /// T 型钙失活动力学（由 TimeBase 设定，避免这里再写 dt 公式）
    pub t_inact_up: f32,
    pub t_inact_down: f32,
    /// 适应强度：`a` 对膜电位的直接抑制。
    pub beta: f32,
    /// 基础阈值。
    pub theta: f32,
    /// 适应对阈值的抬升系数。
    pub gamma: f32,
    /// 仿真步长（ms）。
    pub dt: f32,
    /// 丘脑中继细胞的 T 型钙 burst 模式（低阈值反弹 → 2–3 棘波簇发）
    pub burst: bool,
}

impl Default for Params {
    fn default() -> Self {
        // 皮层锥体细胞量级：膜 20ms，适应 100ms，阈值 1.0
        Self {
            tau_m: 20.0,
            tau_a: 100.0,
            refr_steps: 2,
            burst_steps: 2,
            t_inact_up: 0.02,
            t_inact_down: 0.98,
            beta: 1.0,
            theta: 1.0,
            gamma: 0.5,
            dt: 1.0,
            burst: false,
        }
    }
}

impl Params {
    /// 由时间常数派生 `α`。
    pub fn alpha(&self) -> f32 {
        (-self.dt / self.tau_m).exp()
    }

    /// 由适应时间常数派生 `ρ`。
    pub fn rho(&self) -> f32 {
        (-self.dt / self.tau_a).exp()
    }
}

/// 神经元的状态。`a` 就是"细胞级记忆"的载体。
#[derive(Debug, Clone, Copy, Default)]
pub struct State {
    pub v: f32,
    pub a: f32,
    pub spike: bool,
    /// 重置前的膜电位（去极化阻滞用：发放后 v 归零，看不到峰值）
    pub v_peak: f32,
    /// 绝对不应期剩余步数（生物：1–2 ms，最高发放率 500–1000 Hz）
    pub refr: u8,
    /// T 型钙通道失活（0=去失活可反弹，1=失活）——丘脑中继的 burst 用
    pub t_inact: f32,
    /// 反弹 burst 剩余棘波数
    pub burst_left: u8,
}

#[derive(Debug, Clone)]
pub struct Neuron {
    pub params: Params,
    pub state: State,
    spikes: u32,
}

impl Neuron {
    pub fn new(params: Params) -> Self {
        Self {
            params,
            state: State::default(),
            spikes: 0,
        }
    }

    pub fn alpha(&self) -> f32 {
        self.params.alpha()
    }

    pub fn rho(&self) -> f32 {
        self.params.rho()
    }

    /// 推进一个 `dt`。返回本步是否发放。
    pub fn step(&mut self, input: f32) -> bool {
        let alpha = self.alpha();
        let rho = self.rho();

        // 膜电位：上一步的泄漏 + 本步输入 − 适应电流
        self.state.v = alpha * self.state.v + (1.0 - alpha) * input - self.params.beta * self.state.a;

        // 适应电流：对上一次发放积分（s 是上一步的脉冲）
        let prev_spike = if self.state.spike { 1.0 } else { 0.0 };
        self.state.a = rho * self.state.a + (1.0 - rho) * prev_spike;

        // 发放判定：阈值被适应抬高
        // 绝对不应期：期间膜电位照常整合，但不发放
        if self.state.refr > 0 {
            self.state.refr -= 1;
            self.state.spike = false;
            return false;
        }
        // T 型钙：去极化失活、超极化去失活；去失活后反弹 → 降低阈值 2 步
        if self.params.burst {
            let k_in = self.params.t_inact_up;
            let k_out = self.params.t_inact_down;
            if self.state.v > 0.5 {
                self.state.t_inact += (1.0 - self.state.t_inact) * k_in;
            } else if self.state.v < 0.3 {
                self.state.t_inact *= k_out;
            }
            if self.state.t_inact < 0.25 && self.state.v > 0.4 {
                self.state.burst_left = self.params.burst_steps;
                self.state.t_inact = 1.0;
            }
        }
        let mut threshold = self.params.theta + self.params.gamma * self.state.a;
        if self.state.burst_left > 0 {
            threshold -= 0.25;
            self.state.burst_left -= 1;
        }
        self.state.v_peak = self.state.v;
        let spike = self.state.v > threshold;

        if spike {
            self.state.v = 0.0; // 重置
            self.state.refr = self.params.refr_steps;
            self.spikes += 1;
        }
        self.state.spike = spike;
        spike
    }

    /// 电导型更新（分流抑制）：`g_e`/`g_i` 是兴奋/抑制电导，`i_extra` 是外部电流。
    ///
    /// `v ← α_eff·v + (1−α_eff)·E_eff − β·a`，其中
    /// `g_tot = g_L + g_e + g_i`（g_L=1）、`α_eff = exp(−dt·g_tot/τ_m)`、
    /// `E_eff = (g_e·E_E + g_i·E_I + i_extra)/g_tot`。
    /// 与电流型的区别：抑制是**分流**（除 g_tot）而不是减法，且膜时间常数随电导缩短。
    pub fn step_cond(&mut self, g_e: f32, g_i: f32, i_extra: f32) -> bool {
        const E_E: f32 = 1.0; // 兴奋反转（相对静息）
        const E_I: f32 = -0.3; // 抑制反转（相对静息）
        let g_tot = 1.0 + g_e + g_i;
        let e_eff = (g_e * E_E + g_i * E_I + i_extra) / g_tot;
        // 可移植 exp：与 CUDA 的 exp32 逐位相同（见 src/math.rs 的说明）
        let alpha = crate::num::math::exp32(-self.params.dt * g_tot / self.params.tau_m);
        self.state.v =
            alpha * self.state.v + (1.0 - alpha) * e_eff - self.params.beta * self.state.a;

        let prev_spike = if self.state.spike { 1.0 } else { 0.0 };
        let rho = self.rho();
        self.state.a = rho * self.state.a + (1.0 - rho) * prev_spike;
        if self.state.refr > 0 {
            self.state.refr -= 1;
            self.state.spike = false;
            return false;
        }
        if self.params.burst {
            let k_in = self.params.t_inact_up;
            let k_out = self.params.t_inact_down;
            if self.state.v > 0.5 {
                self.state.t_inact += (1.0 - self.state.t_inact) * k_in;
            } else if self.state.v < 0.3 {
                self.state.t_inact *= k_out;
            }
            if self.state.t_inact < 0.25 && self.state.v > 0.4 {
                self.state.burst_left = self.params.burst_steps;
                self.state.t_inact = 1.0;
            }
        }
        let mut threshold = self.params.theta + self.params.gamma * self.state.a;
        if self.state.burst_left > 0 {
            threshold -= 0.25;
            self.state.burst_left -= 1;
        }
        self.state.v_peak = self.state.v;
        let spike = self.state.v > threshold;
        if spike {
            self.state.v = 0.0;
            self.state.refr = self.params.refr_steps;
            self.spikes += 1;
        }
        self.state.spike = spike;
        spike
    }

    pub fn spike_count(&self) -> u32 {
        self.spikes
    }

    /// 清零状态（不清计数）。
    pub fn reset(&mut self) {
        self.state = State::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn drive(n: &mut Neuron, input: f32, steps: usize) -> usize {
        let before = n.spike_count();
        for _ in 0..steps {
            n.step(input);
        }
        (n.spike_count() - before) as usize
    }

    fn run_sequence(n: &mut Neuron, inputs: &[f32]) -> usize {
        let before = n.spike_count();
        for &i in inputs {
            n.step(i);
        }
        (n.spike_count() - before) as usize
    }

    #[test]
    fn default_params_are_sane() {
        let p = Params::default();
        assert!(p.alpha() > 0.0 && p.alpha() < 1.0);
        assert!(p.rho() > p.alpha(), "适应应比膜电位慢");
    }

    #[test]
    fn no_input_never_spikes() {
        let mut n = Neuron::new(Params::default());
        assert_eq!(drive(&mut n, 0.0, 1000), 0);
    }

    #[test]
    fn constant_input_adapts() {
        // 持续输入：前 200 步有发放，后 200 步被适应压下去
        let mut n = Neuron::new(Params::default());
        let early = drive(&mut n, 1.6, 200);
        let late = drive(&mut n, 1.6, 200);
        assert!(early > 0, "恒定输入应引发发放");
        assert!(late < early, "适应应降低发放率: early={early} late={late}");
    }

    #[test]
    fn adaptation_recovers_after_rest() {
        let mut n = Neuron::new(Params::default());
        let naive = drive(&mut n, 1.2, 200);
        n.reset();
        let _ = drive(&mut n, 1.6, 300); // 先适应
        let _ = drive(&mut n, 0.0, 800); // 休息，a 衰减
        let after = drive(&mut n, 1.2, 200);
        assert!(
            after >= naive,
            "休息后应恢复到接近基线: naive={naive} after={after}"
        );
    }

    #[test]
    fn history_changes_response() {
        // 同一个输入，不同的既往史 → 不同响应。这就是"细胞级记忆"。
        let mut fresh = Neuron::new(Params::default());
        let fresh_count = drive(&mut fresh, 1.2, 200);

        let mut primed = Neuron::new(Params::default());
        let _ = drive(&mut primed, 1.6, 300); // 先被"打动"过
        let primed_count = drive(&mut primed, 1.2, 200);

        assert!(
            primed_count < fresh_count,
            "被适应过的神经元对同一输入响应更少: fresh={fresh_count} primed={primed_count}"
        );
    }

    #[test]
    fn heterogeneous_time_constants_differ() {
        // 短脉冲：快神经元充得上，慢神经元充不上
        let fast = Params {
            tau_m: 5.0,
            ..Params::default()
        };
        let slow = Params {
            tau_m: 50.0,
            ..Params::default()
        };
        let mut a = Neuron::new(fast);
        let mut b = Neuron::new(slow);
        let pulse = vec![1.5_f32; 12];
        let ca = run_sequence(&mut a, &pulse);
        let cb = run_sequence(&mut b, &pulse);
        assert_ne!(ca, cb, "异质时间常数应对同一脉冲给出不同响应");
    }

    #[test]
    fn shunting_depends_on_membrane_potential() {
        // 判据（docs/19 §1 ④）：同输入、不同膜电位 → 抑制电流不同。
        // 电流型：抑制 = 常数减法；电导型：抑制电流 = g_i·(E_I − V)，随 V 变。
        let p = Params {
            theta: 10.0, // 不发放，只看亚阈值的稳态
            gamma: 0.0,
            beta: 0.0,
            ..Params::default()
        };
        // 同一 g_i=0.5、同一 i_extra=0.8，两种初始膜电位
        let mut lo = Neuron::new(p);
        lo.state.v = -0.2;
        let mut hi = Neuron::new(p);
        hi.state.v = 0.5;
        for _ in 0..500 {
            lo.step_cond(0.2, 0.5, 0.8);
            hi.step_cond(0.2, 0.5, 0.8);
        }
        // 电导型：抑制电流 I_inh = g_i·(E_I − V) 在 V 高时更负（分流更强）
        let i_lo = 0.5 * (-0.3 - lo.state.v);
        let i_hi = 0.5 * (-0.3 - hi.state.v);
        assert!(
            i_hi < i_lo,
            "去极化时抑制电流应更强（分流）: V_lo={:.3} I={:.3}, V_hi={:.3} I={:.3}",
            lo.state.v,
            i_lo,
            hi.state.v,
            i_hi
        );
        // 同一稳态：高电导把两种初始条件拉到同一 E_eff（分流 = 归一化）
        assert!(
            (lo.state.v - hi.state.v).abs() < 0.05,
            "分流应把不同初始 V 拉到同一稳态: {:.3} vs {:.3}",
            lo.state.v,
            hi.state.v
        );
    }

    #[test]
    fn threshold_and_reset_are_consistent() {
        // 单步超阈值必发；发后 v 必须归零
        let p = Params {
            theta: 0.5,
            gamma: 0.0,
            beta: 0.0,
            ..Params::default()
        };
        let mut n = Neuron::new(p);
        assert!(n.step(20.0), "单步输入 20 应越过阈值 0.5");
        assert_eq!(n.state.v, 0.0, "发放后膜电位应重置");
    }
}

