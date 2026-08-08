# Au = Pd = 2 nm，Au/Pd PZC 差值 0.30 V

本目录完整复现 `Figures/Figure_Au2nm_Pd2nm/` 的两类电静力模型、文件结构和图件风格，只改变 Au 与 Pd 的 PZC。所有结果均由本目录脚本独立重算；原目录未被覆盖。

## PZC 换算与固定参数

项目采用 `pH = 7.0` 和既有换算斜率 `0.059126500015748 V/pH`：

`E_RHE = E_SHE + 0.059126500015748 × pH`

- Au：`0.51 V vs. SHE → 0.923885500110236 V vs. RHE`
- Pd：`0.21 V vs. SHE → 0.623885500110236 V vs. RHE`
- `PZC_Au - PZC_Pd = 0.300000000000000 V`

原参数文件实际温度为 `298.0 K`；本研究保持该温度及其他参数不变。`PZC_support = 0.50 V vs. RHE`，`L_Au = L_Pd = 2 nm`，`C_H,Au = C_H,Pd = 0.50 F/m²`，`C_H,support = 0.20 F/m²`，`C_tot = 10 mM`，`it0_1 = it0_2 = 1.852573885166257e-4 A/m²`，`alpha1 = alpha2 = 0.5`。

换算的机器可读记录位于 `pzc_conversion.json` 和 `pzc_conversion.csv`。各正式构型的完整 RHE 参数、overrides 与 summary 位于 `Au_C_Pd/inputs/`。

## Au｜C｜Pd 的 L_support 结果

| `L_support` (nm) | `E_mix` with EDL (V vs. RHE) | `i_mix_avg` with EDL (A/m²) |
|---:|---:|---:|
| 0 | 0.591897995202 | 0.03910120258 |
| 1 | 0.581156362680 | 0.03041079263 |
| 2 | 0.576301356538 | 0.02788794333 |
| 3 | 0.573683855477 | 0.02668980084 |
| 10 | 0.570021665723 | 0.02509690740 |
| 1000 | 0.569830007729 | 0.02504680111 |

Janus 构型（`L_support = 0 nm`）与 `1000 nm` 构型的差值为：

`E_mix(0 nm) - E_mix(1000 nm) = 0.022067987473 V = 22.067987473 mV`

因此，在只把 Au/Pd PZC 差扩大到 `0.30 V`、其余参数不变时，两种构型的 `E_mix` 差值没有达到 `60 mV`；约为目标的 `36.8%`。

`L_support` OFAT 的 5–10 nm 指数拟合衰减长度为 `2.1437 nm`，拟合 `R² = 0.9999877`；相对平台的 `0.1 mV` 位置为 `10.6941 nm`。

## 独立 Au｜Pd 模型

| 条件 | `E_mix` (V vs. RHE) | `i_mix_avg` (A/m²) | `I_mix` (A) |
|---|---:|---:|---:|
| with EDL | 0.591897995202 | 0.01583843811 | 6.335375245e-13 |
| w/o EDL | 0.467000000000 | 0.1175603541 | 4.702414163e-12 |

独立模型的 `C_tot` 扫描保持原范围 `10^-4–10^3 M`。代表点的 with-EDL 结果为：

| `C_tot` | `E_mix` (V vs. RHE) | `i_mix_avg` (A/m²) | with/w/o ratio |
|---:|---:|---:|---:|
| 0.01 M | 0.591897995202 | 0.01583843811 | 0.1347260 |
| 1 M | 0.513703789411 | 0.06959297241 | 0.5919765 |
| 10 M | 0.485650635563 | 0.09731571755 | 0.8277937 |
| 1000 M | 0.469094148915 | 0.1152246909 | 0.9801322 |

## 输出结构与验证

- `Au_C_Pd/Case_Figures/Figure_3/`：六个正式构型的 RP potential 与 local-current panels，含 1000 nm active-window 图。
- `Au_C_Pd/Case_Figures/Figure_RP/`：2D solution potential 与 surface-charge 图。
- `Au_C_Pd/Figure_L_support/`：18 点主 OFAT、overlap 诊断、1000 nm 参考及收敛记录。
- `Au_C_Pd/PZC_support_study/`：164 点 support-PZC 主扫描、锚点、收敛审计和 12 组代表性二维场。
- `Au_Pd_independent/`：独立平面 EDL 的 Figure 3、RP 2D、polarization、uniform-bar 与 `C_tot` 专题。

脚本内验证通过：绝对电流平衡、Fourier mode/GL 收敛、2D `y=0` 表面回代、SVG 可编辑文字及文件数量检查均通过。最终输出为 `78 PNG + 78 SVG + 0 PDF`。

所有图件沿用原目录的 Helvetica-first 字体栈、Au/C/Pd 材料色、with/w/o EDL 配色和线型。当前机器 PNG 字体回退为 DejaVu Sans；SVG 保留 Helvetica-first 可编辑文字。

## 适用范围

本研究仍使用 linearized Poisson–Boltzmann / Debye–Hückel。正式 Au｜C｜Pd 构型的 `max |phi_tilde|` 约为 `7.85–8.09`，独立模型为 `8.87`，均超过弱场阈值 1。数值收敛验证通过不消除这一物理适用性限制；材料边界附近也应保留 Fourier/Gibbs caveat。

## 一键生成

在 `2026/` 工作区根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME=/private/tmp/au2_pzc03_cache \
MPLCONFIGDIR=/private/tmp/au2_pzc03_mpl \
Mixed_Potential_Electrical_Double_Layer/.venv_macos/bin/python \
Figures/Figure_Au2nm_Pd2nm_PZC_03V/make_all_au2nm_pd2nm.py
```

生成器遵循非覆盖策略；若目标结果已存在，会校验并复用专题结果或拒绝混写。
