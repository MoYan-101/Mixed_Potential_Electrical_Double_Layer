# Hutchings 2022：Au–Pd mixed-potential/EDL 对比汇总

本目录集中整理四种 Au–Pd 模型的 mixed potential、绝对 mixed current、
Figure 3 和 reaction-plane (RP) 图件，用于讨论 Huang/Hutchings 等人在
*Nature* 2022 论文 *Au–Pd separation enhances bimetallic catalysis of alcohol
oxidation* 中提出的 cooperative redox enhancement (CORE)。论文原文位于
[`MS/s41586-022-04397-7.pdf`](../../MS/s41586-022-04397-7.pdf)。

目录名称 `Explantion_Hutchings` 保留用户指定的拼写。

## 四种情况

1. `01_Au_Pd_independent`：两个电静力独立、但通过理想导线共享
   `E_mix` 的 Au/Pd planar half-spaces。
2. `02_Janus_Au2_Pd2`：直接显示 legacy Neumann 计算半胞
   `Au(2)|Pd(2) nm`。其偶延拓后的完整材料周期仍可写成
   `Au(2)|Pd(2)|Pd(2)|Au(2)`，合并相邻同材料段后包含 4 nm 条纹，但该
   完整周期不再作为本目录的空间图横轴。
3. `03_Physical_mixture_C10`：legacy Neumann 半胞
   `Au(2)|C(10)|Pd(2) nm`，本目录直接显示该 14 nm 计算域；完整偶延拓
   周期为 28 nm。它是 physical mixture 的共面条纹替代模型，不是真实
   三维颗粒/碳网络。
4. `04_Janus_on_C_support`：`C(5)|Au(4)|Pd(4)|C(5) nm` 计算半胞，
   表示 C-supported Janus 的齐平共面条纹替代模型。

前三案的 case 文件夹、空间图及 Figure 3 panel a 均保留原生 Au(2)+Pd(2)
计算域/反应面积。Case 1 的解析场不含 Neumann 横向反射；只有 Summary 为了
四案绝对电流公平比较，才把前三案的原生电流线性换算到 Au(4)+Pd(4)。该
面积换算不能称为边界延拓。

## 绝对 mixed current 的共同口径

所有汇总柱状图固定使用：

```text
out-of-plane width W = 0.01 m
Au reactive width = 4 nm
Pd reactive width = 4 nm
common reactive area = W * (4 + 4) nm = 8.0e-11 m2
I_mix = |I_Au| = |I_Pd| at I_Au + I_Pd = 0
```

因此 `I_mix` 是正的 half-reaction magnitude，不是恒为零的净电流，也不是
current density。

`Summary/figures/` 的每张柱状图只画一个共同的 `w/o EDL` 基线柱，随后画
四个模型各自的 `with EDL` 柱；不会把相同的 w/o-EDL 数值重复画四次。原始
四案数值仍逐案保存在下表、CSV 和 JSON 中，便于追溯。这里的表和 Summary
图使用上面的共同 4 nm+4 nm 面积，不是前三案 case 文件夹的原生 2 nm 电流。

`Summary/figures/summary_half_reaction_polarization_overlay.png/svg` 与
`summary_half_reaction_polarization_overlay_xmin045V.png/svg` 使用同一组曲线数据，
分别显示 `0.40--0.64 V` 和 `0.45--0.75 V` 视窗。两版都在同一张图中画四种
`with EDL` 工况和一个共同 `w/o EDL` 参考，共五种 polarization condition；
横轴统一为 `Electrode potential (V vs. RHE)`。左上图例只保留
`Oxidation on Au` 和 `Reduction on Pd`，不显示 `Half reaction` 标题；
纵轴单位 `µA` 使用正体。Au oxidation 为正、Pd reduction 为负；
每一对交点均
按绝对电流判据 `I_Au + I_Pd = 0` 确定。曲线和交点都使用共同 Au(4 nm)+Pd(4 nm)
面积，不使用 Au/Pd 各自面积归一化后的 current density。逐点支撑数据位于
`Summary/csv/four_case_polarization_curves.csv`，计算与交点回归记录位于
`Summary/polarization_summary.json`。

| Model | E_mix, w/o EDL (V) | E_mix, with EDL (V) | I_mix, w/o EDL (pA) | I_mix, with EDL (pA) |
|---|---:|---:|---:|---:|
| Independent Au/Pd | 0.467000000 | 0.624910433 | 9.404828326 | 3.452046250 |
| Janus Au\|Pd | 0.467000000 | 0.624910433 | 9.404828326 | 5.319995190 |
| Physical mixture, C10 | 0.467000000 | 0.597600469 | 9.404828326 | 4.521278834 |
| C-supported Janus | 0.467000000 | 0.610999760 | 9.404828326 | 5.166437459 |

Summary 中，Legacy Case 2/3 的报告电流是原计算半胞 `i_mix_abs` 的两倍；
Case 4 使用 18 nm 半胞原生电流；Case 1 使用面积等效的两倍原始电流。
各 case 的 Figure 3 panel a 则使用原生计算单元电流：with EDL 依次为
`1.726023/2.659998/2.260639/5.166437 pA`，w/o EDL 依次为
`4.702414/4.702414/4.702414/9.404828 pA`。输出 JSON 明确区分
`native_I_mix_*` 与 `comparison_I_mix_*`。

## 输出结构

```text
Summary/
  figures/
    summary_E_mix_comparison.png/svg
    summary_I_mix_absolute_comparison.png/svg
    summary_half_reaction_polarization_overlay.png/svg
    summary_half_reaction_polarization_overlay_xmin045V.png/svg
  csv/
    four_case_comparison.csv
    four_case_polarization_curves.csv
  polarization_summary.json
  summary.json

01_Au_Pd_independent/
02_Janus_Au2_Pd2/
03_Physical_mixture_C10/
04_Janus_on_C_support/
  Figure_3/                 # 每案 6 PNG + 6 SVG
  Figure_RP/                # 每案 3 PNG + 3 SVG
  data/                     # summary、surface/charge、2D field

inputs/source_registry.json
manifest.json
validation.json
artifacts.json
environment.json
checksums.sha256
```

每个 case 的 Figure 3 包含：

- `E_mix` 与绝对 `I_mix`；
- reaction-plane potential；
- reactant concentration；
- overpotential；
- current density；
- PZC/potential reference map。

每个 case 的 Figure RP 包含：

- solution-phase potential 2D；
- solution-phase potential + `Red_1^-`/`Ox_2^+` concentration 2D；
- surface-charge distribution。

总输出为 `40 PNG + 40 editable-text SVG + 0 PDF`：Summary 两组柱状图和
两个视窗的五工况 polarization overlay，加上四个 case 各九组图。

## 当前正式发布

- 生成时间：`2026-08-08T22:47:10+09:00`。
- 本次发布将前三案恢复为原生 Au/Pd 2 nm 空间域，并把第二案目录改为
  `02_Janus_Au2_Pd2`；Summary 仍保持共同 Au/Pd 4/4 nm 面积。两张柱状图
  均为一个共同 w/o-EDL 基线柱加四个 case-specific with-EDL 柱；polarization
  overlay 的两个视窗均包含四个 with-EDL case 和一个共同 w/o-EDL reference。
- 正式输出位于本目录的 `Summary/`、`01_...` 至 `04_...`、`inputs/`
  及根目录追溯文件；不是仅存在于临时烟测目录。
- [validation.json](validation.json) 已通过共同面积、电流换算/平衡、C 无
  Faradaic current、C10/C1000 平台、图件数量、editable SVG 和可见文字中
  无 `Local` 等检查。
- [checksums.sha256](checksums.sha256) 覆盖 129 个生成文件；图件实测为
  `40 PNG + 40 SVG + 0 PDF`，23 个 JSON 均可严格解析。
- [manifest.json](manifest.json) 记录的 collection source aggregate SHA-256 为
  `39cf9fe2d6778f98c698ac1666ec767bd52997329d67e4c2251941f5680b99cc`。
- collection 单元测试结果为 `9 passed`；另外已用四个真实上游 case 完成整套
  renderer/pipeline 和正式图面抽查。

## 图中文字与符号规范

- Figure 3 的可见标题统一为 `Reactant concentration at RP`、
  `Overpotential at RP` 和 `Current density at RP`，不使用 `Local`；文件名和
  CSV 字段只按追溯需要命名。
- 物理量/变量 `E`、`I`、`phi/Phi`、`eta`、`i`、`sigma`、`c`、`x`、`y`
  使用斜体；说明性下标 `mix`、`RP`、`bulk`、`eq`、`s` 使用正体。
- 化学物种 `Red_1^-`、`Ox_2^+` 的名称、元素符号、EDL/PZC/RHE 缩写、
  SI 单位及前缀均为正体。solution-phase potential 色标使用斜体 `Phi` 配
  正体下标 `s`；surface charge 单位写作 `µC cm^-2`；无量纲浓度比不附
  `(-)`。
- Figure 3 panel a 和 Summary 使用绝对 `I_mix`；Figure 3 的空间曲线 panel
  仍按定义使用 local quantity `i(x)`，但可见标题只写 `Current density at RP`。
- 独立 Au/Pd 案的 Figure 3 b--e 和 surface-charge 图以 `Au interface` /
  `Pd interface` 子图标题区分材料，不再重复显示右上角 Au/Pd 色块角标。
- 其余三案的 Figure 3 b--e 和 surface-charge 图不显示底部 Au/Pd/C 材料条；
  材料边界仍由竖虚线和曲线配色表达，`x (nm)` 直接标在主坐标轴下方。
- Polarization curve 的横轴统一写作
  `Electrode potential (V vs. RHE)`。

## 一键重建

从 `2026/` 工作区根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/private/tmp/hutchings_explanation_mpl \
Mixed_Potential_Electrical_Double_Layer/.venv_macos/bin/python \
ALL/Explantion_Hutchings/make_all_explantion_hutchings.py
```

生成器默认拒绝覆盖既有正式输出。图件为 600 dpi PNG 和保留可编辑文字的
SVG，字体栈为 Helvetica/Nimbus Sans/Arial/DejaVu Sans，不生成 PDF。

## C(10 nm) 与 C(1000 nm)

在匹配的 legacy `Au|C|Pd` scan 中，C10 已达到当前解释和作图精度下的
远场平台。C10 相对 C1000 的 with-EDL 差异约为 `0.2391 mV` 和
`0.02613%`（共同 Au4+Pd4 口径下约 `0.00118 pA`）。因此可把两者解释为
相同的弱-overlap/separated limit，但不得写成机器精度逐位完全相同。

## 与 Hutchings 2022 的解释边界

Hutchings 论文实验建立的是电子侧 CORE：Au 上的 alcohol oxidation 产生
电子，导电支撑把电子传到空间分离的 Pd，并在 Pd 上驱动 ORR。这里的四个
模型都预先假定理想电子连接和共同 `E_mix`，只能研究 CORE 已成立后的
solution-side EDL correction。

因此可以检验的命题是：Janus 接触界面的 lateral EDL overlap 是否会改变
reaction-plane fields 并减弱 EDL-induced current suppression。不能用这些
结果单独解释 alloy 的 ligand/strain/Pd-redox 改变，也不能把模型的 pA
电流直接等同于论文的 thermocatalytic rate 或 TOF。

所有 with-EDL case 的 `max |phi_tilde|` 都明显大于 1，超出 linear
Debye–Hückel 的弱电位定量范围；legacy Fourier/cosine 材料边界还需保留
Gibbs/ringing caveat。数值收敛不消除这些物理适用性限制。
