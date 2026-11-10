# dreamkernel

**把「一个神经网络结构长什么样」变成可声明、可校验、可导出的东西。**

这个 crate 只有**框架层**：形状（拓扑）、对齐（映射）、声明与校验的机制。
"是哪个脑"（哪些区、哪些通路、什么动力学）属于**实现层**，不在本 crate 里。

> 状态：`0.1.x`，研究原型。公共 API 不保证兼容。

---

## 三个概念

| 概念 | 说明 |
|---|---|
| **形状 `Topo`** | 一个区声明自己的几何：`Point`（无结构）/ `Line` / `Ring` / `Grid` / `Modules` / 柱状 `column(径向, 切向)` / 递归 `tiled(基元, nx, ny, 格内)` |
| **距离语义 `Metric`** | 坐标**能不能当距离用**：`None`（全靠连接）/ `Collapsed`（柱内"相邻"是拓扑相邻，不产生传导延迟）/ `Millimetres`（1:1，距离→延迟） |
| **对齐 `Map`** | 两个拓扑怎么对：`Identity` / `Affine{scale,offset,mirror}` / `Permute`（对照）/ `AllToAll`（广播，必须**显式**写） |

### 为什么需要"距离坍缩"这一档

真实传导延迟取决于轴突长度，但**不是所有拓扑都有毫米**：皮层柱内部"相邻"是拓扑相邻，
按毫米铺开会凭空造出 4 mm ⇒ 0.4 ms 的延迟。所以三档是必要的，
而不是"要么全抽象、要么全 1:1"。

```rust
use dreamkernel::core::topo::{Coord, Coords, Map, Metric, Rect, Topo};

let c = Topo::LINE.assign(40);            // 40 个神经元铺到一维
let co = Coords(&c);
let pop = co.population(Coord::at(0.5), 0.05);  // 群体落点（峰值 1，不归一化）
let pairs = Map::Identity.pair_with_len(&c, &c, 8); // 按坐标配对，带长度

let col = Topo::column(6, 12);            // 径向 6 层 × 切向 12 柱
assert_eq!(col.dims(), 2);
assert_eq!(col.metric, Metric::Collapsed); // 坍缩：不转毫米
```

---

## 设计原则

1. **零依赖**：不拖张量、GPU、文件格式。它要在任何环境里都能编译。
2. **不猜语义**：`Topo` 只说形状，`Map` 只说对齐；区名/通路是实现层的事。
3. **能不崩就不崩**：校验返回**全部**问题，而不是第一个就 panic。
4. **增量**：`Topo::POINT` 是默认值 = 没有空间结构 = 旧行为。声明了才生效。

---

## 规格

* `docs/74-topology.md`：拓扑代数、连接映射、三档精度、迁移三步。
* `docs/75-framework-vs-implementation.md`：框架层与实现层的边界。

## 许可

MIT OR Apache-2.0。
