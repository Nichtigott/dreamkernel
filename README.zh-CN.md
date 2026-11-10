# dreamkernel

**写一份声明，得到一个能跑的脑。**

`dreamkernel` 是一个**零依赖**的小工具包：把神经结构写成声明——区的形状（拓扑）、
投射的对齐方式（映射）、神经元模型——然后**由声明生成一个可运行的脉冲网络**。

> 状态：`0.1.x`，研究原型，公共 API 不保证兼容。
>
> English: [`README.md`](README.md)

---

## 目录

- [快速上手](#声明--建脑--跑)（三区跑起来）
- [**用法（完整）**](#用法完整)：字段表 / **区拓扑怎么设** / 槽与子区域 / 投射 / 细胞模板 / 组装 / 读数 / 校验报错 / 延迟算法
  - [区拓扑怎么设](#2-区拓扑怎么设这是最容易踩的地方) ← **最容易踩的地方，必看**
- [一个 Cell 模板 → 一群神经元](#一个-cell-模板--一群各有不同的神经元)
- [调质（多巴胺等）](#调质是有名字的不是一个兴奋度标量)
- [组装：脑是"拼起来"的](#组装脑是拼起来的)
- [别的框架是怎么做的](#别的框架是怎么做的)
- [设计原则](#设计原则)

---

## 三个概念

| 概念 | 说明 |
|---|---|
| **形状** `Topo` | 区自己声明几何：`Point` / `Line` / `Ring` / `Grid` / `Modules` / 柱状 `column(径向, 切向)` / 递归 `tiled(基元, nx, ny, 格内)` |
| **距离语义** `Metric` | 坐标**能不能当距离用**：`None`（全靠连接）/ `Collapsed`（柱内"相邻"是拓扑相邻，不是毫米）/ `Millimetres`（1:1，距离→传导延迟） |
| **对齐** `Map` | 两个拓扑怎么对：`Identity` / `Affine{scale, offset, mirror}` / `Permute`（对照）/ `AllToAll`（广播，必须**显式**写） |

### 为什么需要"坍缩"这一档

传导延迟来自轴突长度，但**不是所有拓扑都有毫米**：皮层柱内部"相邻"是指同一柱的相邻层，
不是"隔了 0.3 mm"。把它当米制会凭空造出延迟。所以是三档，而不是"要么全抽象、要么全 1:1"。

---

## 声明 → 建脑 → 跑

```rust
use dreamkernel::core::cell::Modulators;
use dreamkernel::sim::{Brain, Config};

// …声明 SRC / MID / OUT（见 examples/minimal.rs）…

let mut brain = Brain::declare(&[SRC, MID, OUT], Config::default())?; // 声明 → 脑
brain.wire_local();                                                  // 同区局部递归
let mut ext = vec![0.0; brain.areas.len()];
ext[0] = 1.2;                                                        // 只驱动第一个区
for _ in 0..300 { brain.step(&ext, &Modulators::default()); }
println!("{} Hz", brain.rate_hz(0));
```

跑附带的示例（三个区、两条由声明生成的投射）：

```bash
cargo run --example minimal
```

实测输出（`dt = 0.5 ms`）：

```
神经元 180 个，共 2 条投射：
  SRC → MID（Feedforward）1152 条突触
  MID → OUT（Feedforward）1152 条突触
t=100ms  SRC   63.7 Hz   MID   27.7 Hz   OUT   37.7 Hz
t=300ms  SRC   68.2 Hz   MID   31.8 Hz   OUT   42.9 Hz
MID 的**跨区**入边延迟：4–5 tick（SRC 有米制：跨区 8mm + 区内位置差）
OUT 的**跨区**入边延迟：4–4 tick（MID 是坍缩档：区内位置不计入距离）
```

最后两行就是三档精度的用处：同样 8 mm，**有米制的一侧散成 4–5 tick，坍缩的一侧是平的 4**。

---

## 神经元模型也是声明

```rust
pub struct Spec { model, tau_m, tau_a, theta, beta, gamma, refr_steps, dt, gain, act }

Spec::new(Model::AdLif)                       // 泄漏积分 + 适应电流
Spec::new(Model::Lif)                         // 同一个积分器，β = 0
Spec::new(Model::Rate).with_gain(0.05)        // 发放率单元（RNN 味道），以 act(v) 为概率发放
```

`const` 可构造，所以能直接写进区声明：

```rust
const MID: AreaSpec = AreaSpec {
    topology: Topo::column(4, 8),
    neuron: Spec::new(Model::Rate).with_gain(0.01),   // 这个区是发放率群体
    efferents: &MID_EFF,
    ..
};
```

`Model::Rate` 是**模型选择，不是生物事实**：真实神经元不发"概率脉冲"。
它的用途是把人工神经网络那套单元接进同一个引擎做对照，所以必须显式声明，
而且默认激活用 ReLU（0 输入 → 0 发放；用 tanh 会让静止态发 50%）。

---

## 用法（完整）

下面从"最小能跑"到"把区拓扑设对"，按步骤写。

### 0. 五分钟

```rust
use dreamkernel::sim::{Brain, Config};
use dreamkernel::core::cell::Modulators;

let mut brain = Brain::declare(&[SRC, MID], Config::default())?;   // 声明 → 脑
brain.wire_local();                                               // 同区局部递归
let ext = vec![1.2, 0.0];                                         // 每区的外部驱动
for _ in 0..300 { brain.step(&ext, &Modulators::default()); }
println!("SRC {} Hz", brain.rate_hz(0));
```

### 1. 一个区有哪些字段（`AreaSpec`）

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | ✓ | 区名（全局唯一；组装时会被加实例前缀） |
| `class` | ✓ | `Cortex` / `ThalamicRelay{tonic}` / `SubcorticalRelay` / `PrimaryAfferent` / `Subcortical` / `Modulatory` / `Motor`。**决定权重档、背景、稳态目标** |
| `topo` | ✓ | 模态标签：`Visual` / `Somatic` / `Auditory` / `None`（只用于"谁跟谁该保序"的配对检查） |
| `micro` | ✓ | `Micro::cortical()/relay()/agranular()`：层构 + 抑制占比 |
| `size` | ✓ | `Size::new(E, I, patches)`——**只用到 `patches`**；E/I 由 `Config` 决定 |
| `layout` | ✓ | `Layout::mni([x,y,z], width)`：区中心与尺度（width 影响半边长=距离换算） |
| `motif` | ✓ | 局部递归/前馈/反馈强度 |
| `slots` | ✓ | 端口（见 §3） |
| `topology` | ✓ | **区的形状**（见 §2） |
| `efferents` | ✓ | 出向投射（见 §4）；`&[]` = 不声明 |
| `neuron` | ✓ | 神经元模板（见 §5） |
| `type_specs` | ✓ | 按类型的模板 `[P,PV,SOM,VIP]`；`&[]` = 四类都用 `neuron` |

### 2. **区拓扑怎么设**（这是最容易踩的地方）

拓扑决定三件事：①神经元在"图"上的坐标；②索引顺序（`assign(n)` 的顺序就是索引顺序，
所以"索引 ↔ 坐标"永远同向）；③**坐标能不能当距离用**（`metric`，进而决定传导延迟）。

| 你想表达 | 写法 | `dims()` | `metric` | 行为 |
|---|---|---|---|---|
| 没有空间（调质核、网状核） | `Topo::POINT` | 0 | `None` | 所有神经元同一个点；不产生任何距离 |
| 一条**保序轴**（音调、一维体图表） | `Topo::LINE` | 1 | `Millimetres` | `u=(i+0.5)/n`；坐标就是毫米 |
| **周期**量（朝向 0°/180°、头方向） | `Topo::RING` | 1 | `Millimetres` | 同上但 `u` 走最短环距（0 与 1 是邻居） |
| **二维薄片**（视网膜图、二维体图） | `Topo::GRID` | 2 | `Millimetres` | `n` 自动铺成近正方形网格 |
| `n` 个**离散模块**（桶状皮层） | `Topo::modules(n)` | 1 | `Millimetres` | 每模块一个位置，模块内无结构 |
| **皮层柱**：径向×切向，距离**坍缩** | `Topo::column(radial, tangential)` | 2 | **`Collapsed`** | 铺成 `切向 × 径向` 网格；**不产生传导延迟** |
| **递归网格**（超柱：格内再放子拓扑） | `Topo::tiled(Kind::Grid, 8, 8, Kind::Ring)` | 2 | `Millimetres` | 外层 8×8 格，每格内部再铺一个子拓扑 |

三条要点：

1. **`LINE` / `GRID` 会让"位置差"变成延迟**；`column` **不会**（柱内"相邻"是拓扑相邻，
   不是毫米——把它当米制会凭空造出延迟）。选错了不会报错，只会悄悄多出几毫秒。
2. `column(radial, tangential)` 的参数顺序是 **径向在前**：`u` 轴 = 切向、`v` 轴 = 径向，
   即 `column(6, 12)` = 6 层深 × 12 个切向位置。
3. `assign(n)` 把 n 个神经元铺到形状上——**形状先定、数量后配**，
   同一个 `GRID` 可以是 12 个神经元也可以是 12000 个。

```rust
topology: Topo::LINE,                        // S1 的 parcel：一维 homunculus 展开
topology: Topo::column(6, 12),               // V1：6 层深 × 12 个方位柱（距离坍缩）
topology: Topo::tiled(Kind::Grid, 8, 8, Kind::Ring),   // 超柱：位置 × 方位环
topology: Topo::POINT,                       // 中继核 / 调质核
```


### 2.1 七种形状的实测（`cargo run --example topologies`）

```
区          Topo       tiled dims       metric    shape  首/末坐标
PointA     Point      -       0         None   (9, 1)  (0.50,0.50)…(0.50,0.50)
LineB      Line       -       1  Millimetres   (9, 1)  (0.06,0.50)…(0.94,0.50)
RingC      Ring       -       1  Millimetres   (9, 1)  (0.06,0.50)…(0.94,0.50)
GridD      Grid       -       2  Millimetres   (3, 3)  (0.17,0.17)…(0.83,0.83)
ModulesE   Modules    -       1  Millimetres   (9, 1)  (0.06,0.50)…(0.94,0.50)
ColumnF    Grid       -       2    Collapsed   (3, 3)  (0.12,0.08)…(0.12,0.42)
TiledG     Grid       yes     2  Millimetres   (3, 3)  (0.17,0.17)…(0.83,0.83)
```

看两处：**`ColumnF` 的 metric 是 `Collapsed`**（柱内位置不产生延迟），
**`TiledG` 的 tiled 是 `yes`**（外层 Grid、格内 Ring）。

同一个 `GRID` 可以是 7 个神经元也可以是 48 个——**形状先定，数量后配**：

```
GRID 铺 7 个神经元 → shape (3, 3)，首坐标 (0.1667, 0.1667)
GRID 铺 12 个神经元 → shape (4, 3)，首坐标 (0.1250, 0.1667)
GRID 铺 48 个神经元 → shape (7, 7)，首坐标 (0.0714, 0.0714)
```

### 3. 槽与子区域（`Slot`）

槽 = 区的端口。**子区域必须写在槽上**，投射才知道接哪一块：

```rust
static SLOTS: [Slot; 2] = [
    // 入向槽：只接"下半身"那一半（`Rect` 在本区自己的坐标系里）
    Slot::afferent("lower_body", TopoDomain::Somatic).with_region(Rect::of_u(0.0, 0.5)),
    Slot::efferent("out", TopoDomain::Somatic),
];
```

⚠️ **`Rect` 是在声明它的区自己的坐标系里说的**。同一个"下半身"，在两个区里数值不同：

| 区 | 自己的轴 | 下半身 |
|---|---|---|
| DRG（皮节段序） | 腰骶 → 颈胸 | `[0.00, 0.50)` |
| VPL（体图位置） | 外侧 → 内侧 | `[0.56, 1.00)` |

两端靠 `Map` 对齐，**不靠数值相等**。

### 4. 投射（`Efferent`）

```rust
Efferent {
    to: "MID",              // 靶区名（同部件内）
    from_slot: "out",       // 本区哪个出向槽
    to_slot: "in",          // 靶区哪个入向槽
    projection: Projection::Feedforward,
    map: Map::Identity,     // 两端拓扑怎么对齐
    w: 2.0,                 // 单突触权重倍率
    fan_in: Some(24),       // 每个靶细胞收多少源细胞（None = 引擎默认）
}
// 常用简写：
Efferent::topo("MID", "out", "in", Projection::Feedforward, 2.0, 24)
```

`map` 的四个选择：

| `Map` | 含义 | 用在哪 |
|---|---|---|
| `Identity` | 坐标直接对应（保序） | 视网膜图→V1、体图→VPL |
| `Affine{scale, offset, mirror}` | 缩放/平移/逐轴镜像 | 一侧镜像到另一侧、局部放大 |
| `Permute{seed}` | 几何不变、**源标签打乱** | **对照**（证明保序真的在起作用） |
| `AllToAll` | 广播（无地形） | 调质核团；**必须显式写**，不写等于没声明 |

建边规则：两端都有拓扑且 `map` 保序 → 按坐标找最近的 `fan_in` 个源；
否则按 `fan_in` 广播。

### 5. 细胞模板与类型（`Spec`）

见下一节"一个 Cell 模板 → 一群各有不同的神经元"。要点：`neuron` 是模板，
`type_specs` 给四类各自的参数，`tau_spread` 造个体差异，`mod_gain` 决定受哪种调质影响。

### 6. 组装（`Module` / `Assembly`）

```rust
let mut asm = Assembly::new();
asm.add(&RELAY_MOD, "L").add(&RELAY_MOD, "R")
   .wire("L.out", "R.in", Projection::Feedforward, Map::Identity, 2.0, Some(12));
let specs = asm.expand()?;                    // 区名变成 "L:RELAY" / "R:RELAY"
let mut brain = Brain::declare(&specs, Config::default())?;
```

规矩：部件内部投射只能指向同部件的区；跨部件必须走端口；实例名与展开名都唯一。

### 7. 运行与读数

| API | 说明 |
|---|---|
| `Brain::declare(&[AreaSpec], Config)` | 声明 → 脑；`Err(Vec<String>)` 列出全部声明问题（不 panic） |
| `brain.wire_local()` | 同区局部递归（不属于跨区声明，单独开关） |
| `brain.step(&ext, &Modulators)` | 走一步；`ext` 是每区驱动 |
| `brain.rate_hz(i)` | 第 i 个区自建脑以来的平均发放率（Hz） |
| `brain.edges` | 每条声明生成了多少突触（可观测） |
| `brain.incoming` / `delay` / `i_bias` / `mod_gain` | 逐神经元视图（调试用） |

### 8. 校验会报什么（速查）

| 错误串 | 含义 |
|---|---|
| `[区重名]` | 同一份声明里区名重复 |
| `[缺靶区]` / `[缺槽]` | `Efferent` 指向不存在的区/槽 |
| `[区域越界]` / `[区域无形状]` | 槽的子区域非法，或区没声明拓扑却要子区域 |
| `[实例重名]` / `[连线无端口]` / `[连线方向不符]` / `[组内投射越界]` | 组装层的结构错误 |

### 9. 延迟怎么算

```
delay = 1ms（突触传递本身） + 距离/10mm·ms⁻¹
距离 = 区中心距离 + 区内位置差（**只在一侧是 Millimetres 时才算那侧的区内位置差**）
再按 dt 量化成 tick，夹在 1..16
```

所以：`dt=0.5ms`、两区相距 8mm ⇒ 约 4 tick；`column` 那一侧不贡献位置差。

---

## 一个 `Cell` 模板 → 一群"各有不同"的神经元

"这个神经元是什么类型"，需要的每一个旋钮都是 `Spec` 的字段，所以**类型 = 一组参数**，
不用改引擎：

| 字段 | 含义 |
|---|---|
| `model` | `AdLif` / `Lif`（同一积分器，β=0）/ `Rate` |
| `tau_m`, `tau_a` | 膜 / 适应时间常数（ms） |
| `theta`, `beta`, `gamma` | 阈值、适应强度、适应对阈值的增益 |
| `refr_steps`, `dt` | 不应期步数、步长 |
| `i_bias` | 静息驱动（"这个神经元偏兴奋/偏抑制"） |
| `tau_spread` | **个体差异**：一个模板组装一群神经元时 τ 的 lognormal 抖动 |
| `mod_gain: [DA, NE, ACh, 5-HT]` | 各调质对这个神经元输入的增益 |
| `gain`, `act` | 发放率单元的斜率 / 激活（默认 `Relu`：静息即沉默） |

按类型的模板写在区声明里（`type_specs = [P, PV, SOM, VIP]`；空 = 四类都用 `neuron`）：

```rust
static PV_FAST: Spec = Spec::new(Model::Lif).with_tau_m(8.0).with_theta(0.6);
// type_specs: &[P_SLOW, PV_FAST, SOM, VIP]   → 类型就是这四组参数
```

实测（`type_specs_make_types_by_parameter`）：同样输入下，PV 的单神经元脉冲数是锥体的
**4.3 倍**（1000 脉冲 / 10 个 PV vs 368 / 16 个 P）。

## 调质是有名字的，不是一个"兴奋度"标量

```rust
let mods = Modulators::new(/*多巴胺*/ 0.0, /*去甲肾上腺素*/ 0.0, /*乙酰胆碱*/ 0.0, /*5-羟色胺*/ 0.0);
brain.step(&ext, &mods);
```

作用 = `Σ mods[k] · Spec::mod_gain[k]`，加在输入电流上。实测（`cargo run --example minimal`，
SRC 声明了 `mod_gain=[0.8,0,0,0]`）：

```
DA=0.0  SRC  193.3 Hz   MID   83.0 Hz   OUT  326.2 Hz
DA=0.5  SRC  273.0 Hz   MID  115.5 Hz   OUT  395.7 Hz
DA=1.0  SRC  343.5 Hz   MID  141.7 Hz   OUT  433.8 Hz
```

`mod_gain = 0` 的群体在任意 DA 水平下**逐位不变**（有断言）。

> **简化的地方直说**：真实多巴胺经 D1/D2 起反向作用，ACh 改的是抑制网络，不是"加电流"。
> 这一层只建"每种调质 × 每种细胞类型的输入增益"；受体级细节属于具体的脑，不属于框架。

---

## 组装：脑是"拼起来"的

单个 `AreaSpec` 只是"一个区"。真实建模时你要的通常是**可复用的部件**——比如
「丘脑中继部件 = 中继核 + 局部中间神经元 + 一条组内投射」——把它实例化多次
（左/右、V1/V2），再用**端口**连起来。

```rust
let mut asm = Assembly::new();
asm.add(&RELAY_MOD, "L")
   .add(&RELAY_MOD, "R")
   .wire("L.out", "R.in", Projection::Feedforward, Map::Identity, 2.0, Some(12));

let specs = asm.expand()?;          // 区名变成 "L:RELAY" / "R:RELAY"
let mut brain = Brain::declare(&specs, Config::default())?;
```

```bash
cargo run --example assembly
# 组装：2 个实例、2 个区、1 条跨部件连线
# 展开后 2 个区： L:RELAY（出向 1 条） / R:RELAY（出向 0 条）
# L:RELAY → R:RELAY  288 条突触
# t=300ms  L  291.6 Hz   R  229.3 Hz
```

两条硬规矩，都能变红：

1. **部件内部的投射只能指向同部件的区**——跨部件必须走端口，否则实例化两次时
   组内引用会指向另一个实例，静默接错。
2. 实例名必须唯一；展开后的区名（`实例:区`）也必须唯一。

---

## 别的框架是怎么做的

粗略对比（按各项目的论文/文档；只作方向性参考）：

| 框架 | 细胞模型抽象 | 拓扑/空间 | 扩展方式 |
|---|---|---|---|
| **NEST** | C++ 模型类按名字注册，参数用字典；`CopyModel` 派生 | 空间是**插件**（`grid`/`free`/`3d`），位置是元数据，延迟由连接规则算 | 新模型 = C++ 类 + 重编译 |
| **Arbor** | 细胞目录 + 机制目录；机制参数按区域给 | 显式形态学（cable），空间**就是**形态 | 机制按名字加载 |
| **Brian2** | 方程 → 生成代码（`NeuronGroup`），任意 ODE | 无内建空间，距离是你自己的数组 | 方程是一等公民，扩展点是代码生成 |
| **Nengo** | `NeuronType` 对象（LIF/AdaptiveLIF…） | 向量空间里的 **ensemble**，没有解剖图 | 神经元类型是 Python 对象 |
| **ANNarchy** | `Neuron` 类 + ODE 字符串，代码生成 | 可选几何（2D/3D），距离依赖连接 | ODE + 投射模式 |
| **snnTorch / BindsNET** | `nn.Module` 子类（`Leaky`/`LIF`） | 无，纯张量 | 标准 DL 模块注册 |
| **Lava / Loihi** | 进程模型 + 定点算术 | 芯片网格坐标 | 厂商 SDK |

`dreamkernel` 的差别（以及不差别的地方）：

* **不新颖**：细胞模型抽象、"空间即插件"、声明式构造——都是标准做法。
* **不一样**：同一个模型可以**按区**在"无距离 / 坍缩柱距离 / 真实毫米"之间切换；
  投射写成"槽的子区域 → 槽的子区域 + `Map`"，建边由声明生成；
  构造期校验会报几何拼不上，而不是静默广播。
* **故意没有**（目前）：树突形态学、可插拔的可塑性规则、分布式执行、硬件后端。

---

## 设计原则

1. **零依赖**：只有几何、映射、校验和一个小引擎。
2. **不藏语义**：广播（`AllToAll`）必须显式写；坍缩拓扑不许凭空造距离。
3. **报错而不是崩**：校验返回**全部**问题；`Brain::declare` 返回 `Result` 而不是中断。
4. **增量**：`Topo::POINT` 是默认 = 无空间结构 = 旧行为；声明了才生效。

## 许可

MIT OR Apache-2.0。
