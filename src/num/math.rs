//! 可移植的初等函数：**CPU 与 GPU 必须逐位相同**。
//!
//! 为什么需要它：电导型神经元的 `alpha = exp(−dt·g_tot/τ_m)` 每步现算，
//! 而 glibc 的 `f32::exp` 与 CUDA 的 `expf` 在约 30% 的输入上差 1 ulp
//! （实测 `gpu_smoke` 的 exp 探测：200000 点里 60406 点不同）。
//! 1 ulp 的 alpha 会让 v 从**第 0 步**就差 1e-8，再被混沌放大——
//! 表现为"某一配置跑到第 393 步脉冲分叉"，看起来像移植 bug，其实不是。
//!
//! 解法：两边跑**同一份多项式**（Taylor 到 r^10/10!，|r| ≤ ln2/2）。
//! 只用到 + − × 和 floor/位操作，全部是 IEEE 严格定义的运算，
//! CUDA 侧再配合 `--fmad=false` 就与 Rust 逐位一致。
//! 精度：截断误差 < 1e-12 相对（|r| ≤ 0.347），加上一次乘 2^k 的舍入。

/// log2(e)
const LOG2E: f32 = f32::from_bits(0x3FB8_AA3B);
/// ln2 的高位（0.693359375，二进制可精确表示）
const LN2_HI: f32 = f32::from_bits(0x3F31_8000);
/// ln2 的余项（−2.12194440e-4），拆两段是为了减小 k·ln2 的舍入误差
const LN2_LO: f32 = f32::from_bits(0xB95E_8083);
/// 1/10!
const C10: f32 = f32::from_bits(0x3493_F27E);
/// 1/9!
const C9: f32 = f32::from_bits(0x3638_EF1D);
/// 1/8!
const C8: f32 = f32::from_bits(0x37D0_0D01);
/// 1/7!
const C7: f32 = f32::from_bits(0x3950_0D01);
/// 1/6!
const C6: f32 = f32::from_bits(0x3AB6_0B61);
/// 1/5!
const C5: f32 = f32::from_bits(0x3C08_8889);
/// 1/4!
const C4: f32 = f32::from_bits(0x3D2A_AAAB);
/// 1/3!
const C3: f32 = f32::from_bits(0x3E2A_AAAB);

/// `exp(x)`，与 `cuda/kernels.cu` 里的 `exp32` **逐位相同**。
#[inline]
pub fn exp32(x: f32) -> f32 {
    // 下溢/上溢边界：两边都返回常量，保证结果也一致
    if x < -87.0 {
        return 0.0;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    // x = k·ln2 + r，|r| ≤ ln2/2；ln2 拆两段减少舍入误差（Cephes 的做法）
    let kf = (x * LOG2E + 0.5).floor();
    let r = (x - kf * LN2_HI) - kf * LN2_LO;
    // exp(r) = Σ r^n/n!（Horner，顺序固定：先高次）
    let mut p = C10;
    p = p * r + C9;
    p = p * r + C8;
    p = p * r + C7;
    p = p * r + C6;
    p = p * r + C5;
    p = p * r + C4;
    p = p * r + C3;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    // 2^k 用位操作拼（不经过 libm 的 scalbn，避免实现差异）
    let k = (kf as i32).clamp(-126, 127);
    p * f32::from_bits(((k + 127) as u32) << 23)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp32_matches_libm_closely() {
        // 不是要求逐位等于 libm（本来就做不到），而是要求**数值上足够准**：
        // 相对误差 < 2e-7（≈ 1.7 ulp），覆盖 [-30, 20]
        let mut worst = 0.0f32;
        let mut x = -30.0f32;
        while x <= 20.0 {
            let a = exp32(x);
            let b = x.exp();
            let rel = ((a - b) / b).abs();
            assert!(rel < 2e-7, "x={x} exp32={a} libm={b} rel={rel}");
            worst = worst.max(rel);
            x += 0.0001;
        }
        assert!(worst < 2e-7, "worst rel {worst}");
    }

    #[test]
    fn exp32_edges() {
        assert_eq!(exp32(0.0), 1.0);
        assert!((exp32(1.0) - std::f32::consts::E).abs() < 1e-6);
        assert_eq!(exp32(-1000.0), 0.0);
        assert!(exp32(88.0).is_finite());
    }
}

/// 可复现的 64 位线性同余发生器（PCG 常数，够用且与平台无关）。
///
/// 框架需要自己的随机源：判据要求"同种子逐位可复现"，所以不能依赖 `rand` 的版本行为。
#[derive(Debug, Clone)]
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    /// `[0, 1)`。
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32 as f32) / (1u64 << 24) as f32
    }
    /// `[0, n)` 的整数。
    pub fn next_below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_f32() * n as f32) as usize % n }
    }
}

/// Logistic sigmoid：`1/(1+e^{-x})`。用 `exp32` 实现，跨平台可复现。
pub fn sigmoid(x: f32) -> f32 {
    // 分成两段避免 exp 溢出：x>=0 与 x<0 对称写法
    if x >= 0.0 {
        let e = exp32(-x);
        1.0 / (1.0 + e)
    } else {
        let e = exp32(x);
        e / (1.0 + e)
    }
}

/// `tanh` = 2·sigmoid(2x) − 1。**不要**用 `f32::tanh`：它在不同 libm 上末位可能不同，
/// 而本项目的判据要求"同种子逐位可复现"。
pub fn tanh32(x: f32) -> f32 {
    2.0 * sigmoid(2.0 * x) - 1.0
}

/// ReLU（负半轴截零）。
pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// 点积（等长取较短；长度不等是调用方的错，不在这里 panic）。
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 线性插值。
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// 就地 softmax（数值稳定版：先减最大值）。
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in x.iter_mut() {
        *v = exp32(*v - m);
        s += *v;
    }
    if s > 0.0 {
        for v in x.iter_mut() {
            *v /= s;
        }
    }
}

/// 最大值的下标（空切片返回 `None`）。
pub fn argmax(x: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in x.iter().enumerate() {
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod activation_tests {
    use super::*;

    #[test]
    fn sigmoid_and_tanh_are_consistent() {
        // tanh(x) = 2σ(2x) − 1 必须**逐位**成立（同一份实现推出来的）
        for x in [-6.0f32, -1.0, -0.1, 0.0, 0.3, 2.0, 9.0] {
            assert_eq!(tanh32(x), 2.0 * sigmoid(2.0 * x) - 1.0, "x={x}");
        }
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!(sigmoid(50.0) <= 1.0 && sigmoid(-50.0) >= 0.0);
        // 饱和：正值侧 f32 下就是 1.0；负值侧是小正数（exp32(-50)≈1.9e-22），不是精确 0
        assert_eq!(sigmoid(50.0), 1.0);
        assert!(sigmoid(-50.0) < 1e-20 && sigmoid(-50.0) >= 0.0);
    }

    #[test]
    fn softmax_is_a_distribution_and_shift_invariant() {
        let mut a = [1.0f32, 2.0, 3.0];
        softmax_inplace(&mut a);
        let s: f32 = a.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "{a:?}");
        assert!(a[2] > a[1] && a[1] > a[0]);
        // 平移不变（数值稳定性的关键）
        let mut b = [101.0f32, 102.0, 103.0];
        softmax_inplace(&mut b);
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-6, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn dot_lerp_argmax() {
        assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
        assert_eq!(dot(&[1.0], &[3.0, 4.0]), 3.0, "取较短");
        assert_eq!(lerp(0.0, 10.0, 0.25), 2.5);
        assert_eq!(lerp(0.0, 10.0, 2.0), 10.0, "t 被夹住");
        assert_eq!(argmax(&[1.0, 5.0, 5.0]), Some(1), "并列取第一个");
        assert_eq!(argmax(&[]), None);
        assert_eq!(relu(-3.0), 0.0);
    }

    #[test]
    fn lcg_is_reproducible_and_in_range() {
        let mut a = Lcg::new(7);
        let mut b = Lcg::new(7);
        for _ in 0..1000 {
            let x = a.next_f32();
            assert!((0.0..1.0).contains(&x), "{x}");
            assert_eq!(x, b.next_f32(), "同种子必须逐位一致");
        }
        assert!(Lcg::new(1).next_below(5) < 5);
        assert_eq!(Lcg::new(1).next_below(0), 0);
    }
}
