# dreamkernel

**写一份声明，得到一个能跑的脑。**

`dreamkernel` 是一个**零依赖**的小工具包：把神经结构写成声明——区的形状（拓扑）、
投射的对齐方式（映射）、神经元模型——然后**由声明生成一个可运行的脉冲网络**。

> 状态：`0.1.x`，研究原型，公共 API 不保证兼容。
>
> English: [`README.md`](README.md)

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
use dreamkernel::sim::{Brain, Config};

// …声明 SRC / MID / OUT（见 examples/minimal.rs）…

let mut brain = Brain::declare(&[SRC, MID, OUT], Config::default())?; // 声明 → 脑
brain.wire_local();                                                  // 同区局部递归
let mut ext = vec![0.0; brain.areas.len()];
ext[0] = 1.2;                                                        // 只驱动第一个区
for _ in 0..300 { brain.step(&ext, 1.0); }
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
