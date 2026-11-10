//! 拓扑：一个区的**形状**（声明）、把 n 个神经元铺到形状上（生成）、子区域与群体落点（查询）。
//!
//! 规格与动机见 `docs/74-topology.md`。要点：
//!
//! - 层（[`Layer`](super::layer::Layer)）/ patch / 细胞类型都是**解剖事实**，不动；
//!   拓扑是**正交的第四根轴**：给每个神经元一个 `(u, v) ∈ [0,1]²` 的坐标。
//! - 形状由区**声明**（[`Topo`]），神经元个数由预算决定（`--area-e`）——两者解耦：
//!   同一个 `Grid` 可以是 12 个神经元也可以是 12000 个，形状不变。
//! - 拓扑**不是**闭合枚举：`Topo::tiled` 把区切成网格、每格内部再铺一个子拓扑，
//!   可嵌套（V1 超柱 = 视网膜图 × 方位轴）。这是"元拓扑 + 递归生成网格"。
//! - 两个拓扑怎么对齐由 [`Map`] 显式声明（保序/仿射/折返/打乱/广播）。
//!   **`AllToAll` 是声明，不是默认**——把"几何不知道就退化成广播"那条路堵死。
//!
//! 这一层只做纯计算，不碰神经元、不碰突触，所以可以独立测。

/// 拓扑上的一个点，分量归一化到 `[0, 1]`。
///
/// 一维拓扑只用 `u`（`v` 恒 0.5）；二维拓扑两个都用。
/// **不存米制**：米制由区的 `Layout` 给（MNI 中心 + 外接半径），这里只存"在图上哪儿"。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coord {
    pub u: f32,
    pub v: f32,
}

impl Coord {
    pub const fn new(u: f32, v: f32) -> Self {
        Self { u, v }
    }
    /// 一维拓扑上的点。
    pub const fn at(u: f32) -> Self {
        Self { u, v: 0.5 }
    }

    /// 归一化坐标 → 区的**米制位置**（mm）。`half` = 该区的半边长（与旧的球半径同一把尺），
    /// `dims` = 该区拓扑的维数（0 = 全在中心，1 = 沿 x 铺开，2 = xy 平面）。
    ///
    /// 为什么需要米制：**空间拓扑不只是顺序**——两个神经元隔多远决定传导延迟，
    /// 而延迟决定"谁先到"。没有米制，拓扑就只剩一个序，时序结构藏不进去。
    pub fn to_mm(self, center: [f32; 3], half: f32, dims: usize) -> [f32; 3] {
        let h = half.max(0.0);
        match dims {
            0 => center,
            1 => [center[0] + (self.u - 0.5) * 2.0 * h, center[1], center[2]],
            _ => [
                center[0] + (self.u - 0.5) * 2.0 * h,
                center[1] + (self.v - 0.5) * 2.0 * h,
                center[2],
            ],
        }
    }
}

/// 两个坐标之间的**归一化长度**（`ring` 为真时 `u` 走最短环距）。
///
/// 输出是归一化长度（0..√2），乘上区的半边长（mm）就是米制距离。
pub fn norm_len(a: Coord, b: Coord, ring: bool) -> f32 {
    let mut du = (a.u - b.u).abs();
    if ring {
        du = du.min(1.0 - du);
    }
    let dv = a.v - b.v;
    (du * du + dv * dv).sqrt()
}

/// 传导延迟（ms）：`1 + 长度/速度 + 抖动`。
///
/// 速度单位 mm/ms（`Config::delay_velocity` 默认 10.0 = 10 m/s，有髓轴突量级）。
/// 这里的 1 ms 是**基础延迟**（突触传递本身，不是可调参数），
/// 抖动来自轴突直径差异（`delay_jitter`）。
pub fn conduction_delay_ms(len_mm: f32, velocity_mm_per_ms: f32, jitter: f32) -> f32 {
    let v = if velocity_mm_per_ms > 0.0 { velocity_mm_per_ms } else { 10.0 };
    1.0 + len_mm.max(0.0) / v + jitter.max(0.0)
}

/// 归一化矩形（子区域）。一维拓扑用 `v0..v1` 全开区间。
///
/// 取代了原先只在一维成立的 `Span{lo, hi}`：一维时退化成区间，二维时是矩形，
/// 折返/打乱由 [`Map`] 负责，不再混在"区间"里。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub u0: f32,
    pub u1: f32,
    pub v0: f32,
    pub v1: f32,
}

impl Rect {
    pub const ALL: Rect = Rect { u0: 0.0, u1: 1.0, v0: 0.0, v1: 1.0 };

    pub const fn new(u0: f32, u1: f32) -> Self {
        Self { u0, u1, v0: 0.0, v1: 1.0 }
    }

    /// 二维矩形（一维拓扑请用 [`Rect::new`]）。
    pub const fn sheet(u0: f32, u1: f32, v0: f32, v1: f32) -> Self {
        Self { u0, u1, v0, v1 }
    }

    pub fn contains(self, c: Coord) -> bool {
        c.u >= self.u0 && c.u < self.u1 && c.v >= self.v0 && c.v < self.v1
    }

    pub fn is_all(self) -> bool {
        self.u0 <= 0.0 && self.u1 >= 1.0 && self.v0 <= 0.0 && self.v1 >= 1.0
    }

    /// 一维子段（只取 `u` 轴，`v` 全开）。`Arbor` 的旧 `Span` 语义等价于这个。
    pub const fn of_u(lo: f32, hi: f32) -> Self {
        Self { u0: lo, u1: hi, v0: 0.0, v1: 1.0 }
    }

    /// `u` 轴上的 `[起, 止)` 索引（与 `link::projection::Span::index_range` 同一套规则：
    /// 两端四舍五入、非空至少 1 个）。一维拓扑只用这一半；二维拓扑用它做"列范围"。
    ///
    /// 这里自己实现而不调用 `Span`：`topo` 是更底层的模块，不该反向依赖 `link`。
    pub fn index_range_u(self, n: usize) -> (usize, usize) {
        if n == 0 {
            return (0, 0);
        }
        let scale = |x: f32| ((x.clamp(0.0, 1.0) * n as f32).round() as usize).min(n);
        let (lo, hi) = (scale(self.u0), scale(self.u1));
        if hi <= lo && self.u1 > self.u0 {
            (lo.min(n - 1), (lo.min(n - 1) + 1).min(n))
        } else {
            (lo, hi.max(lo))
        }
    }
}

/// 拓扑基元。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// 无空间结构（调质核团、网状核）。所有神经元同一个点。
    Point,
    /// 一维区间（音调轴、一维体图表）。不环绕。
    Line,
    /// 一维**周期**（朝向 0°/180° 等价、头方向）。`u = 0` 与 `u = 1` 是同一个点。
    Ring,
    /// 二维薄片（视网膜拓扑、躯体图、二维音调图）。长宽比按 `n` 自动取近正方形。
    Grid,
    /// `n` 个**离散模块**（桶状皮层：一根须一个桶）。模块内无结构。
    Modules,
}

/// **距离的语义**——对应"三种定义微观结构的方式"（用户 2026-11-10）。
///
/// | 变体 | 含义 | 距离 |
/// |---|---|---|
/// | `None` | ① 现在的建模：完全没有距离，全靠"谁连谁" | 不存在，延迟只能显式给 |
/// | `Collapsed` | ② 柱状结构：径向/切向两个轴，距离在这两个方向上**坍缩**（不是真实米制） | 归一化（层内/柱内位置），**不转成 mm、不产生传导延迟** |
/// | `Millimetres` | ③ 1:1 空间表征：每个神经元有真实坐标 | 米制 → 传导延迟 |
///
/// 关键区别不是"有没有坐标"，而是**坐标能不能当距离用**。②里"相邻"是拓扑意义上的
/// 相邻（同一柱内、同一层），不是"隔了几毫米"；把它当米制会凭空造出传导延迟。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    None,
    Collapsed,
    Millimetres,
}

/// 一个区的拓扑：基元 + 可选的**平铺递归**。
///
/// - `tiles = (1, 1)`：普通基元，坐标就是基元坐标。
/// - `tiles = (tx, ty)`：把区按 `tx × ty` 切成网格，**每格内部**再铺一个 `inner`。
///   于是 `u` 编码"在第几格里的哪一点"，`v` 编码"格内子拓扑上的哪一点"。
///   这就把"元拓扑 + 递归网格"表达出来了（V1 超柱：外层是视网膜图，格内是方位轴）。
///
/// 只允许一层嵌套：再深就需要三维坐标，而 `Coord` 是二维的（刻意保持简单）。
/// 需要"三层"时说明形状还没想清楚，应该先拆成两个区。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topo {
    pub kind: Kind,
    /// `Kind::Modules` 的模块数；其余基元忽略。
    pub modules: usize,
    /// 平铺网格 `(tx, ty)`；`(1,1)` = 不平铺。
    pub tiles: (u8, u8),
    /// 每格内部的子拓扑；`tiles == (1,1)` 时忽略。
    pub inner: Kind,
    pub inner_modules: usize,
    /// 距离的语义（见 [`Metric`]）。决定坐标能不能转成 mm / 传导延迟。
    pub metric: Metric,
    /// 显式网格形状 `(列数 = u 轴, 行数 = v 轴)`；`(0,0)` = 按 n 自动取近正方形。
    ///
    /// 柱状拓扑用它把"径向 × 切向"钉死：`column(6, 12)` = 6 层深 × 12 个切向位置。
    pub shape: (u16, u16),
}

impl Topo {
    /// 无结构。
    pub const POINT: Topo = Topo::flat(Kind::Point);
    /// 一维保序。
    pub const LINE: Topo = Topo::flat(Kind::Line);
    /// 一维周期。
    pub const RING: Topo = Topo::flat(Kind::Ring);
    /// 二维薄片。
    pub const GRID: Topo = Topo::flat(Kind::Grid);

    /// 普通基元（不平铺）。
    pub const fn flat(kind: Kind) -> Self {
        let metric = match kind {
            Kind::Point => Metric::None,
            _ => Metric::Millimetres,
        };
        Self { kind, modules: 0, tiles: (1, 1), inner: Kind::Point, inner_modules: 0, metric, shape: (0, 0) }
    }

    /// **② 柱状结构**（用户 2026-11-10）：径向 × 切向两个轴，距离在两者上**坍缩**。
    ///
    /// 与 ③（`LINE`/`GRID`）的区别：这里的"相邻"是拓扑相邻（同一柱/同一层），
    /// 不是"隔了几毫米"。所以 `metric = Collapsed`——坐标能用来定**谁连谁**，
    /// 但**不转成 mm、不产生传导延迟**（否则会凭空造出距离）。
    pub const fn column(radial: usize, tangential: usize) -> Self {
        Self {
            kind: Kind::Grid,
            modules: 0,
            tiles: (1, 1),
            inner: Kind::Point,
            inner_modules: 0,
            metric: Metric::Collapsed,
            // u = 切向（多少个柱）、v = 径向（柱内多少层）
            shape: (
                if tangential > u16::MAX as usize { u16::MAX } else { tangential as u16 },
                if radial > u16::MAX as usize { u16::MAX } else { radial as u16 },
            ),
        }
    }

    /// `n` 个离散模块。
    pub const fn modules(n: usize) -> Self {
        Self {
            kind: Kind::Modules,
            modules: n,
            tiles: (1, 1),
            inner: Kind::Point,
            inner_modules: 0,
            metric: Metric::Millimetres,
            shape: (0, 0),
        }
    }

    /// **递归网格**：外层 `base` 切成 `tx × ty` 格，每格内部铺一个 `within`。
    pub const fn tiled(base: Kind, tx: u8, ty: u8, within: Kind) -> Self {
        Self {
            kind: base,
            modules: 0,
            tiles: (tx, ty),
            inner: within,
            inner_modules: 0,
            metric: Metric::Millimetres,
            shape: (0, 0),
        }
    }

    /// 坐标能不能当**真实距离**用（②③ 的分界）。
    pub fn has_metric(self) -> bool {
        self.metric == Metric::Millimetres
    }

    pub fn is_tiled(self) -> bool {
        self.tiles.0 > 1 || self.tiles.1 > 1
    }

    /// 该拓扑是几维（0 = 无结构，1 = 一维，2 = 二维）。
    pub fn dims(self) -> usize {
        if self.is_tiled() {
            return 2;
        }
        match self.kind {
            Kind::Point => 0,
            Kind::Line | Kind::Ring | Kind::Modules => 1,
            Kind::Grid => 2,
        }
    }

    /// 把 `n` 个神经元**均匀铺到形状上**，返回每个神经元的坐标。
    ///
    /// 铺法是"形状先定、数量后配"：`n` 只决定分辨率，不改变形状。
    /// 顺序即神经元在该（层/patch/细胞群）内的索引顺序——这一条让"索引 ↔ 坐标"
    /// 双向可查（`assign` 之后索引就是坐标的下标）。
    pub fn assign(self, n: usize) -> Vec<Coord> {
        if n == 0 {
            return Vec::new();
        }
        // 显式形状（柱）：u 轴 = 切向位置、v 轴 = 径向位置；n 只决定填充多少个格
        if self.shape.0 > 0 && self.shape.1 > 0 && !self.is_tiled() {
            let (nx, ny) = (self.shape.0 as usize, self.shape.1 as usize);
            return (0..n)
                .map(|i| {
                    let (ix, iy) = (i % nx, (i / nx) % ny);
                    Coord::new((ix as f32 + 0.5) / nx as f32, (iy as f32 + 0.5) / ny as f32)
                })
                .collect();
        }
        if self.is_tiled() {
            // (tx, ty) 格 × 每格内 `within` 的 1 维位置：u 走格子、v 走格内
            let (tx, ty) = (self.tiles.0.max(1) as usize, self.tiles.1.max(1) as usize);
            let (nx, ny) = grid_shape(n, tx, ty);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let (ix, iy) = (i % nx, i / nx);
                let u = (ix as f32 + 0.5) / nx as f32;
                let v = (iy as f32 + 0.5) / ny as f32;
                out.push(Coord::new(u, v));
            }
            return out;
        }
        match self.kind {
            Kind::Point => vec![Coord::new(0.5, 0.5); n],
            Kind::Line | Kind::Ring | Kind::Modules => (0..n)
                .map(|i| Coord::at((i as f32 + 0.5) / n as f32))
                .collect(),
            Kind::Grid => {
                let (nx, ny) = grid_shape(n, 1, 1);
                (0..n)
                    .map(|i| {
                        let (ix, iy) = (i % nx, i / nx);
                        Coord::new((ix as f32 + 0.5) / nx as f32, (iy as f32 + 0.5) / ny as f32)
                    })
                    .collect()
            }
        }
    }

    /// `(nx, ny)`：把 `n` 点铺成尽量方的网格，且 `nx ≥ min_x`、`ny ≥ min_y`。
    pub fn shape(self, n: usize) -> (usize, usize) {
        if self.is_tiled() {
            grid_shape(n, self.tiles.0.max(1) as usize, self.tiles.1.max(1) as usize)
        } else if self.kind == Kind::Grid {
            grid_shape(n, 1, 1)
        } else {
            (n.max(1), 1)
        }
    }
}

/// `n` 个点铺成网格：列数取 `n` 的平方根附近，且不少于 `min_x`；行数不少于 `min_y`。
fn grid_shape(n: usize, min_x: usize, min_y: usize) -> (usize, usize) {
    let n = n.max(1);
    let mut nx = (n as f32).sqrt().ceil() as usize;
    nx = nx.max(min_x).max(1);
    let ny = n.div_ceil(nx).max(min_y).max(1);
    let nx = n.div_ceil(ny).max(nx).max(1);
    (nx, ny)
}

/// 坐标集合上的查询：子区域与群体落点。**只做纯函数**，不依赖神经元。
pub struct Coords<'a>(pub &'a [Coord]);

impl<'a> Coords<'a> {
    /// 落在矩形里的神经元索引（升序）。
    pub fn indices_in(&self, r: Rect) -> Vec<usize> {
        (0..self.0.len()).filter(|&i| r.contains(self.0[i])).collect()
    }

    /// 以 `c` 为中心的高斯群体落点，返回 `(索引, 权重)`；权重峰值 = 1，**不归一化**。
    ///
    /// 与 `sensory::touch_population` 同一约定（峰值 1、边缘按高斯衰减），
    /// 区别是这个版本对任何拓扑都成立，不看"哪个身体部位"。
    /// `sigma` 用归一化单位（`u/v` 的同一尺度）。
    pub fn population(&self, c: Coord, sigma: f32) -> Vec<(usize, f32)> {
        let s = sigma.max(1e-6);
        let mut out = Vec::new();
        for (i, p) in self.0.iter().enumerate() {
            let d = self.dist(*p, c) / s;
            let w = (-0.5 * d * d).exp();
            if w >= 0.02 {
                out.push((i, w));
            }
        }
        out
    }

    /// 拓扑距离。`Line`/`Grid` 直接欧氏；`Ring` 的 `u` 走最短环距。
    ///
    /// 注意：这里不知道拓扑类型，所以**上界取欧氏**——`Ring` 的最短环距由调用方
    /// 用 `Topo::Kind` 决定（见 [`map_dist`]）。
    fn dist(&self, a: Coord, b: Coord) -> f32 {
        let (du, dv) = (a.u - b.u, a.v - b.v);
        (du * du + dv * dv).sqrt()
    }

    /// 带周期处理的群体落点：`ring` 为真时 `u` 按最短环距算。
    pub fn population_on(&self, c: Coord, sigma: f32, ring: bool) -> Vec<(usize, f32)> {
        let s = sigma.max(1e-6);
        let mut out = Vec::new();
        for (i, p) in self.0.iter().enumerate() {
            let mut du = (p.u - c.u).abs();
            if ring {
                du = du.min(1.0 - du);
            }
            let dv = p.v - c.v;
            let d = (du * du + dv * dv).sqrt() / s;
            let w = (-0.5 * d * d).exp();
            if w >= 0.02 {
                out.push((i, w));
            }
        }
        out
    }

    /// 神经元 `a` 与 `b` 之间的归一化长度（`ring` 为真时 `u` 走最短环距）。
    pub fn len(&self, a: usize, b: usize, ring: bool) -> f32 {
        match (self.0.get(a), self.0.get(b)) {
            (Some(p), Some(q)) => norm_len(*p, *q, ring),
            _ => f32::INFINITY,
        }
    }

    /// 神经元 `a` 与 `b` 之间的**米制距离**（mm）。`half` 与 `to_mm` 同一把尺。
    pub fn len_mm(&self, a: usize, b: usize, ring: bool, half: f32) -> f32 {
        self.len(a, b, ring) * 2.0 * half.max(0.0)
    }

    /// 给定一个"身体/感觉空间"的点，落到这个拓扑上的坐标（`Map::apply` 的便捷包装）。
    pub fn nearest(&self, c: Coord) -> Option<usize> {
        let mut best = None;
        let mut bd = f32::INFINITY;
        for (i, p) in self.0.iter().enumerate() {
            let d = self.dist(*p, c);
            if d < bd {
                bd = d;
                best = Some(i);
            }
        }
        best
    }
}

/// 两个拓扑之间**怎么对齐**。一条区际 link 必须显式写一个 `Map`。
///
/// 设计原则：**广播（`AllToAll`）是声明，不是默认值**。以前"没写几何"就等于"整区广播"，
/// 于是 N-2026-005 那类"几何不知道、读数推不出来"的失败只能靠猜；现在必须显式写。
///
/// ⚠️ **没有 `Fold` 变体**（初版有，已删）。躯体图沿纵裂折成 U 形**不是把一个轴对折**，
/// 而是"上肢带与下肢带是两个不同的子区域（`Rect`），各自内部有自己的方向（`mirror`）"。
/// 对折算子会把两侧映射到同一半轴上，丢掉"哪一侧"的信息——实测写出来两条肢体落在同一端。
/// 正确表达：`Slot.region` 声明两个带 + 各自的 `Map::Affine{mirror}`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Map {
    /// 保序：坐标直接对应（视网膜图→V1、躯体图→VPL）。
    Identity,
    /// 仿射：缩放 + 平移 + 逐轴镜像（`mirror[0]` = u 轴翻转）。
    Affine { scale: [f32; 2], offset: [f32; 2], mirror: [bool; 2] },
    /// 打乱（**对照**：拓扑若真在起作用，换这个必须让保序判据变红）。
    Permute { seed: u64 },
    /// 广播（无地形：调质核团）。
    AllToAll,
}

impl Map {
    /// 把一个坐标映射到另一侧；`AllToAll` 返回 `None`（没有对应点）。
    pub fn apply(&self, c: Coord) -> Option<Coord> {
        match *self {
            Map::Identity => Some(c),
            Map::Affine { scale, offset, mirror } => Some(Coord::new(
                (if mirror[0] { 1.0 - c.u } else { c.u }) * scale[0] + offset[0],
                (if mirror[1] { 1.0 - c.v } else { c.v }) * scale[1] + offset[1],
            )),
            Map::Permute { .. } => None, // 由 `permute` 单独实现（需要种子 + 规模）
            Map::AllToAll => None,
        }
    }

    pub fn is_topographic(self) -> bool {
        !matches!(self, Map::AllToAll)
    }

    /// 打乱映射的确定性置换（同一 `(seed, n)` 必定同一结果）。
    pub fn permute(seed: u64, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        // 先混合再置奇：直接 `seed | 1` 会让 42 与 43 落成同一个种子（实测过）。
        let mut s = (seed ^ 0x9E37_79B9_7F4A_7C15).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for i in (1..n).rev() {
            // xorshift64
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let j = (s % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        idx
    }

    /// 与 [`Map::pair`] 相同，但额外返回**归一化长度**——链路的"长短"要一直带到
    /// 建延迟那一步（`coord::norm_len` → mm → `conduction_delay_ms`）。
    pub fn pair_with_len(
        &self,
        src: &[Coord],
        dst: &[Coord],
        fan_in: usize,
    ) -> Vec<(usize, usize, f32)> {
        if !self.is_topographic() || src.is_empty() || dst.is_empty() {
            return Vec::new();
        }
        let fan = fan_in.max(1);
        // `Permute`：几何不变、**源标签打乱**——这是"拓扑到底在不在起作用"的对照。
        // 注意不能返回空：空配对等于"这条投射没有突触"，对照就变成测另一件事了。
        let perm = match *self {
            Map::Permute { seed } => Some(Map::permute(seed, src.len())),
            _ => None,
        };
        let mut out = Vec::new();
        for (di, d) in dst.iter().enumerate() {
            let Some(target) = self.apply_for_pairs(*d) else { continue };
            let mut ds: Vec<(f32, usize)> = src
                .iter()
                .enumerate()
                .map(|(si, s)| {
                    let (du, dv) = (s.u - target.u, s.v - target.v);
                    ((du * du + dv * dv).sqrt(), si)
                })
                .collect();
            ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for &(len, si) in ds.iter().take(fan) {
                // 打乱时把"选中的源"换成它被打乱后指向的那个源
                let mapped = perm.as_ref().map(|p| p[si]).unwrap_or(si);
                out.push((di, mapped, len));
            }
        }
        out
    }

    /// 配对用的目标坐标：`Permute` 与 `Identity` 几何相同（打乱的是源标签）。
    fn apply_for_pairs(&self, c: Coord) -> Option<Coord> {
        match *self {
            Map::Permute { .. } => Some(c),
            _ => self.apply(c),
        }
    }

    /// 按 `map` 把源坐标序列连到靶坐标序列：每个靶神经元收最近的 `fan_in` 个源。
    ///
    /// 与 [`Map::pair_with_len`] 同一实现，只是丢掉长度。
    pub fn pair(&self, src: &[Coord], dst: &[Coord], fan_in: usize) -> Vec<(usize, usize)> {
        self.pair_with_len(src, dst, fan_in)
            .into_iter()
            .map(|(d, s, _)| (d, s))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_assigns_evenly_and_is_order_preserving() {
        let c = Topo::LINE.assign(10);
        assert_eq!(c.len(), 10);
        assert!((c[0].u - 0.05).abs() < 1e-6);
        assert!((c[9].u - 0.95).abs() < 1e-6);
        for w in c.windows(2) {
            assert!(w[0].u < w[1].u, "一维拓扑的索引必须与坐标同向");
        }
    }

    #[test]
    fn grid_assigns_near_square_and_covers_sheet() {
        let c = Topo::GRID.assign(12);
        assert_eq!(c.len(), 12);
        // 3×4 或 4×3，两轴都覆盖 (0,1)
        let us: Vec<f32> = c.iter().map(|p| p.u).collect();
        let vs: Vec<f32> = c.iter().map(|p| p.v).collect();
        assert!(us.iter().cloned().fold(0.0, f32::max) < 1.0);
        assert!(vs.iter().cloned().fold(0.0, f32::max) < 1.0);
        // 不同神经元不同坐标（没有塌缩）
        let mut uniq: Vec<(i32, i32)> =
            c.iter().map(|p| ((p.u * 100.0) as i32, (p.v * 100.0) as i32)).collect();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 12, "12 个神经元应有 12 个不同坐标");
    }

    #[test]
    fn point_and_modules() {
        let p = Topo::POINT.assign(5);
        assert!(p.iter().all(|c| *c == Coord::new(0.5, 0.5)));
        assert_eq!(Topo::POINT.dims(), 0);
        let m = Topo::modules(4).assign(8);
        assert_eq!(m.len(), 8);
        assert_eq!(Topo::modules(4).dims(), 1);
    }

    #[test]
    fn tiled_is_recursive_and_two_dimensional() {
        // 外层 Line 切成 4×4 格，每格内部再铺一个一维子拓扑
        let t = Topo::tiled(Kind::Line, 4, 4, Kind::Ring);
        assert!(t.is_tiled());
        assert_eq!(t.dims(), 2, "递归网格是二维的（外层位置 × 格内子拓扑）");
        let c = t.assign(64);
        assert_eq!(c.len(), 64);
        assert!(c.iter().all(|p| p.u > 0.0 && p.u < 1.0 && p.v > 0.0 && p.v < 1.0));
    }

    #[test]
    fn rect_selects_sub_region() {
        let c = Topo::GRID.assign(100);
        let co = Coords(&c);
        let left = co.indices_in(Rect::new(0.0, 0.5));
        let right = co.indices_in(Rect::new(0.5, 1.0));
        assert_eq!(left.len() + right.len(), 100, "两半必须正好覆盖");
        assert!(left.iter().all(|&i| c[i].u < 0.5));
        assert!(right.iter().all(|&i| c[i].u >= 0.5));
    }

    #[test]
    fn population_peaks_at_one_and_decays() {
        let c = Topo::LINE.assign(20);
        let co = Coords(&c);
        let pop = co.population(Coord::at(0.525), 0.05);
        assert!(!pop.is_empty());
        let (peak_idx, peak_w) = pop
            .iter()
            .cloned()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert!((peak_w - 1.0).abs() < 1e-6, "峰值必须是 1（与 touch_population 同约定）");
        assert_eq!(peak_idx, 10, "0.525 应该落在第 10 个神经元上");
    }

    #[test]
    fn ring_wraps_around_in_population() {
        let c = Topo::RING.assign(10);
        let co = Coords(&c);
        // 朝向 0.05（= 第 0 个）用环距应当能吃到 0.95（= 第 9 个）的邻居
        let ring = co.population_on(Coord::at(0.05), 0.12, true);
        let linear = co.population_on(Coord::at(0.05), 0.12, false);
        let near9_ring = ring.iter().any(|&(i, _)| i == 9);
        let near9_lin = linear.iter().any(|&(i, _)| i == 9);
        assert!(near9_ring, "环上 0.05 与 0.95 是邻居");
        assert!(!near9_lin, "线性轴上它们不是邻居");
    }

    #[test]
    fn bands_plus_mirror_express_the_u_fold() {
        // 躯体图的 U 形折返 = **两个 band** + **各自的镜像方向**，不是一个轴对折：
        //   Arm 带 [0.20,0.50)：distal 在**低**索引端（mirror）
        //   Leg 带 [0.62,1.00)：distal 在**高**索引端（不 mirror）
        // 判据：两条肢体的远端必须分居索引轴两端（Penfield & Boldrey 1937）。
        let arm = Rect::new(0.20, 0.50);
        let leg = Rect::new(0.62, 1.00);
        let map = |r: Rect, pos: f32, mirror: bool| -> f32 {
            let p = if mirror { 1.0 - pos } else { pos };
            r.u0 + p * (r.u1 - r.u0)
        };
        let arm_distal = map(arm, 1.0, true);
        let leg_distal = map(leg, 1.0, false);
        assert!(arm_distal <= arm.u0 + 1e-6, "上肢远端应落在带的外侧端（低索引）");
        assert!(leg_distal >= leg.u1 - 1e-6, "下肢远端应落在带的内侧端（高索引）");
        assert!(arm_distal < leg_distal, "两条肢体远端必须分居两端");
        // 近端（靠躯干）反过来：arm 近端在带的高端、leg 近端在带的低端
        assert!(map(arm, 0.0, true) >= arm.u1 - 1e-6);
        assert!(map(leg, 0.0, false) <= leg.u0 + 1e-6);
    }

    #[test]
    fn permute_is_deterministic_and_breaks_order() {
        let a = Map::permute(42, 100);
        let b = Map::permute(42, 100);
        assert_eq!(a, b, "同一 seed 必须同一置换");
        let identity: Vec<usize> = (0..100).collect();
        assert_ne!(a, identity, "置换不能是恒等（否则对照无效）");
        let c = Map::permute(43, 100);
        assert_ne!(a, c, "不同 seed 必须不同置换");
    }

    #[test]
    fn pair_is_topographic_under_identity() {
        let src = Topo::LINE.assign(50);
        let dst = Topo::LINE.assign(20);
        let pairs = Map::Identity.pair(&src, &dst, 3);
        assert_eq!(pairs.len(), 60, "每个靶 3 条 × 20 个靶");
        // 保序：靶索引越大，源索引的均值越大
        let mut mean_by_dst = [0.0f32; 20];
        let mut cnt = [0usize; 20];
        for &(d, s) in &pairs {
            mean_by_dst[d] += s as f32;
            cnt[d] += 1;
        }
        for d in 0..20 {
            mean_by_dst[d] /= cnt[d].max(1) as f32;
        }
        assert!(
            mean_by_dst.windows(2).all(|w| w[0] <= w[1] + 1e-6),
            "Identity 下源的均值必须随靶位置单调不减：{mean_by_dst:?}"
        );
    }

    #[test]
    fn pair_is_empty_for_all_to_all() {
        let src = Topo::LINE.assign(10);
        let dst = Topo::LINE.assign(5);
        assert!(Map::AllToAll.pair(&src, &dst, 4).is_empty(), "广播不是拓扑配对，必须显式处理");
        assert!(!Map::AllToAll.is_topographic());
        assert!(Map::Identity.is_topographic());
    }

    #[test]
    fn affine_mirror_reverses_the_source_order() {
        // 镜像抹平时序：靶的低索引端必须接源的高索引端
        let src = Topo::LINE.assign(40);
        let dst = Topo::LINE.assign(40);
        let m = Map::Affine { scale: [1.0, 1.0], offset: [0.0, 0.0], mirror: [true, false] };
        let pairs = m.pair(&src, &dst, 1);
        assert!(!pairs.is_empty());
        let src_at = |d: usize| -> f32 {
            pairs.iter().filter(|(dd, _)| *dd == d).map(|(_, s)| *s as f32).sum::<f32>()
        };
        let low: f32 = (0..5).map(src_at).sum::<f32>() / 5.0;
        let high: f32 = (35..40).map(src_at).sum::<f32>() / 5.0;
        assert!(low > high, "镜像后低索引靶应接高索引源：{low} vs {high}");
    }
}

    // ================= 米制与**时序**（2026-11-10，用户要求）=================
    //
    // 空间拓扑不只是顺序：两点的**距离**决定传导延迟，延迟决定谁先到。
    // 没有米制，拓扑就只剩一个序，时长/间隔这些时序结构无处可藏。

    #[test]
    fn line_lays_out_on_one_axis_in_mm() {
        let c = Topo::LINE.assign(5);
        let center = [10.0, -20.0, 30.0];
        let half = 3.0;
        let mm: Vec<[f32; 3]> = c.iter().map(|q| q.to_mm(center, half, 1)).collect();
        // 一维：只有 x 变，y/z 全部等于中心
        assert!(mm.iter().all(|p| (p[1] - center[1]).abs() < 1e-6 && (p[2] - center[2]).abs() < 1e-6));
        // 单调 + 落在 ±half 之内、且接近两端（神经元在格心，不是端点本身）
        assert!(mm.windows(2).all(|w| w[0][0] < w[1][0]));
        assert!(mm[0][0] > center[0] - half && mm[0][0] < center[0]);
        assert!(mm[4][0] < center[0] + half && mm[4][0] > center[0]);
        assert!((mm[0][0] - (center[0] - half * 0.8)).abs() < 1e-5, "n=5 时首格心在 0.1");
    }

    #[test]
    fn grid_lays_out_on_a_sheet_in_mm() {
        let c = Topo::GRID.assign(16);
        let center = [0.0, 0.0, 0.0];
        let mm: Vec<[f32; 3]> = c.iter().map(|q| q.to_mm(center, 2.0, 2)).collect();
        assert!(mm.iter().all(|p| p[2].abs() < 1e-6), "薄片没有厚度");
        // 两轴都被用到（不是一条线）
        let nx = mm.iter().map(|p| p[0].to_bits()).collect::<std::collections::BTreeSet<_>>().len();
        let ny = mm.iter().map(|p| p[1].to_bits()).collect::<std::collections::BTreeSet<_>>().len();
        assert!(nx > 1 && ny > 1, "二维拓扑必须两个轴都用上：{nx}×{ny}");
    }

    #[test]
    fn length_grows_with_separation() {
        let c = Topo::LINE.assign(10);
        let co = Coords(&c);
        let near = co.len(0, 1, false);
        let far = co.len(0, 9, false);
        assert!(far > near, "越远越长：{far} vs {near}");
        assert!((far - 0.9).abs() < 1e-5, "端点相距 0.9（归一化）");
        // 米制 = 归一化 × 2 × half
        assert!((co.len_mm(0, 9, false, 5.0) - 9.0).abs() < 1e-4);
    }

    #[test]
    fn ring_length_wraps() {
        let c = Topo::RING.assign(10);
        let co = Coords(&c);
        assert!((co.len(0, 9, false) - 0.9).abs() < 1e-5, "线性距离是 0.9");
        assert!((co.len(0, 9, true) - 0.1).abs() < 1e-5, "环距应当绕过去，只差 0.1");
    }

    /// **时序结构**判据：同一个源区，靶越远 → 延迟越长 → 到达越晚。
    ///
    /// 这条测的是"空间距离最终变成时间差"，不是"坐标算得对"。
    /// 构造用的是真实场景：源只是靶区的**一小段**（局部凸起 / 拓扑中继的一部分），
    /// 于是靶区远端离源越来越远、到达越来越晚。Identity 下每个靶取最近的源，
    /// 源铺满整个靶区时长度全是 0（那就测不出东西）——所以必须用子段。
    #[test]
    fn distance_becomes_conduction_delay() {
        let src = Topo::LINE.assign(40);
        let dst = Topo::LINE.assign(40);
        let src_sub = &src[0..16]; // 源只占靶区左侧 40%
        let pairs = Map::Identity.pair_with_len(src_sub, &dst, 1);
        assert_eq!(pairs.len(), 40);
        let half = 10.0; // mm
        let vel = 10.0; // mm/ms（10 m/s）
        let len_of = |d: usize| -> f32 {
            pairs.iter().find(|(dd, _, _)| *dd == d).map(|(_, _, l)| *l).unwrap()
        };
        let delay = |d: usize| -> f32 { conduction_delay_ms(len_of(d) * 2.0 * half, vel, 0.0) };
        assert!(len_of(0) < 1e-5, "靶 0 就在源上，长度应为 0");
        assert!(len_of(39) > 0.5, "靶远端离源很远：{}", len_of(39));
        assert!(delay(0) < delay(20), "中间的靶应该比近端晚到");
        assert!(delay(20) < delay(39), "最远的靶应该最晚到");
        // 归一化长度 0.6 × 2 × 10mm = 12mm ⇒ 10 m/s 下 1.2 ms
        assert!((delay(39) - (1.0 + 1.2)).abs() < 1e-3, "延迟 = 1 + 12mm/10 = {}", delay(39));
    }

    /// 延迟量化到 tick 之后仍然保序：长度差足够大时，tick 序列也单调。
    #[test]
    fn delay_order_survives_tick_quantisation() {
        let dt = 2.0f32; // ms/tick（本项目的 dt）
        let tick = |mm: f32| -> u8 {
            (conduction_delay_ms(mm, 10.0, 0.0) / dt).round().clamp(1.0, 16.0) as u8
        };
        let ticks: Vec<u8> = [0.0f32, 10.0, 30.0, 80.0, 140.0].iter().map(|m| tick(*m)).collect();
        assert!(
            ticks.windows(2).all(|w| w[0] <= w[1]),
            "tick 之后必须还是单调：{ticks:?}"
        );
        assert!(ticks[4] > ticks[0], "远近必须能分出不同 tick：{ticks:?}");
    }

    // ================= 三种定义微观结构的方式（用户 2026-11-10）=================
    //
    // ①`POINT`  完全没有距离，全靠"谁连谁"（现在的脑）
    // ②`column` 径向×切向、距离**坍缩**（不是真实米制，不产生传导延迟）
    // ③`LINE/GRID/RING` 1:1 空间表征，坐标就是 mm

    #[test]
    fn three_modes_have_different_metrics() {
        assert_eq!(Topo::POINT.metric, Metric::None);
        assert!(!Topo::POINT.has_metric());
        assert_eq!(Topo::LINE.metric, Metric::Millimetres);
        assert!(Topo::LINE.has_metric());
        let col = Topo::column(6, 12);
        assert_eq!(col.metric, Metric::Collapsed);
        assert!(!col.has_metric(), "② 的距离是拓扑相邻，不是毫米");
        assert_eq!(col.dims(), 2, "柱有两根轴（径向 × 切向）");
        // 平铺（超柱）是 1:1 的，有米制
        assert!(Topo::tiled(Kind::Grid, 4, 4, Kind::Ring).has_metric());
    }

    #[test]
    fn collapsed_topology_still_answers_positions() {
        // ② 不提供 mm，但**仍然要能回答"谁跟谁相邻"**——那是它的全部用途。
        let c = Topo::column(3, 4).assign(12);
        assert_eq!(c.len(), 12);
        let co = Coords(&c);
        let near = co.len(0, 1, false);
        let far = co.len(0, 11, false);
        assert!(far > near, "坍缩拓扑也必须能算拓扑距离：{far} vs {near}");
        // 但不许当米制用
        assert!(!Topo::column(3, 4).has_metric());
    }