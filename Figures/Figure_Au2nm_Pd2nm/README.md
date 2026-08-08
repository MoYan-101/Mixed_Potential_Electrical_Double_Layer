# Au = Pd = 2 nm 双模型图件集

本目录比较两种不同的电静力拓扑。两者使用相同的 Au/Pd 平面内活性长度、动力学参数、电解液浓度和 Au/Pd Helmholtz 电容；区别在于电解液空间是否连续。

## 固定参数

- `L_Au = L_Pd = 2 nm`（平面内活性长度，不是金属厚度）
- `C_H,Au = C_H,Pd = 50 μF/cm² = 0.50 F/m²`
- 旧 Au｜C｜Pd 模型：`C_H,C = 20 μF/cm² = 0.20 F/m²`
- `C_tot = 10 mol/m³ = 10 mM`，`lambda_D` 由模型计算
- `it0_1 = it0_2 = 1.852573885166257e-4 A/m²`
- `alpha1 = alpha2 = 0.5`
- `out_of_plane_width = 0.01 m`
- 独立模型中每个电极只有一个浸液活性面

旧模型计算 `L_support = 0, 1, 2, 3, 10, 1000 nm`。独立 Au｜Pd 模型没有 support 或电极间距坐标：两个局部平面半空间在电静力上相互独立，只通过共同金属电位与绝对电流平衡 `I_Au + I_Pd = 0` 耦合。

## 一键生成

在 `2026/` 工作区根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/private/tmp/au2_pd2_mpl \
Mixed_Potential_Electrical_Double_Layer/.venv_macos/bin/python \
Figures/Figure_Au2nm_Pd2nm/make_all_au2nm_pd2nm.py
```

脚本只生成 600 dpi PNG 和可编辑文字 SVG，不生成 PDF。当前入口同时生成六个正式 case、`L_support` OFAT、support-PZC 参数研究、独立 Au｜Pd 的 Figure 3/2D 图、均匀界面变量柱状图、独立模型基准 polarization curve，以及该独立模型的 `C_tot` 参数研究。`C_tot` 研究中的 EDL profile 和 half-reaction polarization 是该专题的独立结果。

若只想跳过 support-PZC 专题，可加 `--skip-support-pzc-study`。该开关独立于 `--skip-legacy`，便于用 `--skip-legacy --skip-independent` 单独构建或复用 support-PZC 专题。

## 结果

旧 Au｜C｜Pd 模型的 with-EDL 结果：

| `L_support` (nm) | `E_mix` (V) | `i_mix_avg` (A/m²) |
|---:|---:|---:|
| 0 | 0.624910432787 | 0.06649993988 |
| 1 | 0.611904400435 | 0.06037129666 |
| 2 | 0.605705732607 | 0.05861936926 |
| 3 | 0.602335075272 | 0.05775933090 |
| 10 | 0.597600468608 | 0.05651598543 |
| 1000 | 0.597361386234 | 0.05650122135 |

独立 Au｜Pd 模型：

| 条件 | `E_mix` (V) | `i_mix_avg` (A/m²) | `I_mix` (A) |
|---|---:|---:|---:|
| with EDL | 0.624910432787 | 0.04315057813 | 1.726023125e-12 |
| w/o EDL | 0.467000000000 | 0.1175603541 | 4.702414163e-12 |

基准 polarization curve 使用绝对半反应电流，`with EDL` 为实线、`w/o EDL` 为虚线。EDL 使 `E_mix` 从 `0.467 V` 升至 `0.624910 V`（`+0.157910 V`），同时使 `I_mix` 从 `4.7024 pA` 降至 `1.7260 pA`（降低约 `63.3%`）。纵轴 `Current (10^-3 μA)` 中 `10^-3 μA = 10^-9 A`。

独立 Au｜Pd 的 `C_tot` 研究扫描 `10^-4–10^3 M` 共 36 点，并精确包含 `0.01/1/10/1000 M`。代表点如下：

| `C_tot` | `E_mix` with EDL (V) | `i_mix_avg` with EDL (A/m²) | `I_mix` with EDL (A) | with/w/o ratio |
|---:|---:|---:|---:|---:|
| 0.01 M | 0.624910432787 | 0.04315057813 | 1.726023125e-12 | 0.3670504 |
| 1 M | 0.526048310477 | 0.09045095068 | 3.618038027e-12 | 0.7694001 |
| 10 M | 0.490580281883 | 0.1069601338 | 4.278405350e-12 | 0.9098317 |
| 1000 M | 0.469647664287 | 0.1163866636 | 4.655466544e-12 | 0.9900163 |

与旧参考参数不同，本组 `C_H,Au=C_H,Pd=50 μF/cm²` 的 with-EDL 电流在全部采样点都低于 w/o-EDL 值，并从下方单调趋近；因此图中没有沿用旧参考的 “overshoot” 结论。`10–1000 M` 只用于展示数学高盐极限，尤其 `10^3 M` 不代表物理可实现浓度。

## 输出结构

- `Au_C_Pd/Case_Figures/Figure_3/`：reaction-plane potential（panel b）和 local current density（panel e）；六个正式 case 各两组全图，1000 nm 另有两组双侧 active-window 图，共 14 PNG + 14 SVG。
  - panel b 沿用 Figure 3 的 `with EDL` 橙色与 `w/o EDL` 深蓝色。
  - panel e 与独立模型 Figure 3 一致：Au 为绿色、Pd 为蓝色；`with EDL` 为实线，`w/o EDL` 为同色虚线。
- `Au_C_Pd/Case_Figures/Figure_RP/`：2D solution potential 和 surface charge distribution；六个正式 case 各两组全图，1000 nm 另有两组双侧 active-window 图，共 14 PNG + 14 SVG。材料显示色为 Au `#E4C133`、C `#8C8C8C`、Pd `#5A90C8`。
- `Au_C_Pd/Figure_L_support/OFAT/`：8 组 support-length OFAT 图；`csv/`、`inputs/`、`manifest.json` 和 `validation.json` 保存扫描结果与追溯信息。
- `Au_C_Pd/PZC_support_study/`：主扫描仍为 `L_support = 1, 3, 6, 15 nm` 与 `PZC_support = 0.10–0.90 V`（41 个等间距点，164 个结果），另算 `L_support = 1000 nm` 在 `0.10/0.50/0.90 V` 的远场锚点，并以 `L_support = 0 nm` 验证 support PZC 不应影响结果。发布版长度趋势另外直接求解 `L_support = 2, 10 nm`，因此使用 `1, 2, 3, 6, 10, 1000 nm` 共六个横坐标和 `PZC_C = 0.10/0.50/0.90 V` 三条曲线；没有用插值替代直接求解。除趋势、reaction-plane potential profiles 和 surface-charge profiles 三组 PNG/SVG 外，`figures/Figure_RP/` 还为 `4` 个长度 × `3` 个代表 PZC 保存 12 组 2D solution-phase-potential 图和 12 组 potential+Red1+Ox2 复合图；同时保存主扫描、发布版长度趋势及其收敛表、12 个代表 profiles、2D summary、锚点、生产收敛表、原计划分辨率审计、inputs/config、summary、validation、manifest 和独立 checksums。
  - 发布版趋势图以对数 `L_support` 为横坐标，只显示 `E_mix`、`i_mix_avg` 和 support 平均有符号电荷；三条线分别代表 `PZC_C = 0.10/0.50/0.90 V`。图中同时标出 separated electrodes 的 `E_mix = 0.624910432787 V`；它与对应 1000 nm 构型的差值依次为 `52.442/27.533/3.724 mV`。RP-overlap 指标继续保存在主扫描 CSV 与 validation 中，不在趋势图显示。Profile 中 `PZC_support = 0.10/0.50/0.90 V` 分别使用灰色点线、黑色实线和蓝色虚线；材料区保持 Au `#E4C133`、C `#8C8C8C`、Pd `#5A90C8`。
  - 可见变量使用斜体，说明性下标 `support/mix/RP` 和单位使用正体；字体为 Helvetica-first，单位统一放在圆括号中。GL 使用 128 点；为满足原定收敛阈值，生产 `N_modes` 按 `L_support=1/3/6/15/1000 nm` 分别提高为 `960/960/1920/3840/11520`，对应低阶检查为 `480/480/960/1920/9600`。
  - 12 张 potential-only 图和 12 张 potential+reactants 复合图的电势面板共用以 0 mV 为中心的对称色标，动态范围为 `-220` 至 `+220 mV`；复合图中的 Red1/Ox2 由 `c_i/c_bulk = exp(-z_i F Phi_s/RT)` 得到，并在全部 12 个 case、两种反应物之间共用同一个对数色标。纵向均为 `0–5 lambda_D`、`Ny=320`。SVG 仅将密集 pcolormesh 栅格化，文字和标注仍可编辑；`csv/support_pzc_2d_summary.csv` 保存网格、场极值、共同色标、reactant reciprocity 及 `y=0` 回代检查。
  - 原计划的短 support `480→960` 和 1000 nm `5760→7680` mode-pair 结果保存在 `csv/support_pzc_original_resolution_audit.csv`；GL64→128 则在实际生产分辨率下检查。15 组 mode-pair 中有 4 组超阈值：`6 nm, 0.10 V`、`15 nm, 0.10/0.50 V`、`1000 nm, 0.10 V`；因此没有把原计划阶数误报为收敛。

support-PZC 主扫描的端点结果如下；每行依次给出 `PZC_support=0.10→0.90 V`：

| `L_support` (nm) | `E_mix` (V) | `i_mix_avg` (A/m²) | support 平均有符号电荷 (μC/cm²) |
|---:|---:|---:|---:|
| 1 | 0.600357 → 0.623144 | 0.064278 → 0.058898 | 10.5455 → -2.5713 |
| 3 | 0.582001 → 0.621853 | 0.065762 → 0.054874 | 8.2090 → -2.7420 |
| 6 | 0.574784 → 0.621347 | 0.066241 → 0.053434 | 6.8617 → -2.8405 |
| 15 | 0.572566 → 0.621191 | 0.066285 → 0.052988 | 5.7921 → -2.9188 |

15 nm 与 1000 nm 锚点的差异仍很小但不为零：在 `PZC_support=0.10/0.50/0.90 V` 下，`E_mix(1000)-E_mix(15)` 分别为 `-0.0975/-0.0409/-0.0050 mV`，active-RP RMS 差分别为 `0.0378/0.0284/0.0094 mV`。
- `support_pzc_trends_au2_pd2` 的 mixed-potential panel 使用 `0.460–0.645 V` y 范围，在 separated-electrodes 虚线上方保留专用空间；顶部数值标注使用不透明白底，避免与 separated reference 和 `PZC_C=0.90 V` 分支重合。
- `Au_C_Pd/inputs/`：每个 case 的完整参数、可重放 overrides 和 CSV/JSON summary。
- `Au_C_Pd/csv/`：每个 case 的独立 profile CSV、汇总 profile、case summary 和收敛检查。
- `Au_C_Pd/validation.json`：电流平衡、2D 表面回代、图件数量和数值收敛验证。
- `Au_Pd_independent/figures/Figure_3/`：Figure 3 a–f 六组图。
- `Au_Pd_independent/figures/Figure_3/Uniform_Bar_Comparison/`：保留原 2×3 均匀界面变量柱状图，并新增只包含 reaction-plane potential、带正确上下标/电荷的 RP reactant concentration、以及以 mV 表示的 RP overpotential 三栏版本；发表版使用 `8.0 × 3.60 in` 的单排窄 panel 版式，每个 panel 相对旧版约收窄 29%。第一、第三个标题保持单行，只有中间标题分两行；极小浓度柱值竖排，负 overpotential 柱值使用较小的柱外间距，避免与柱体或横轴重叠。标题、轴标签、刻度、柱值和图例字号均使用放大版；共 2 PNG + 2 SVG，并保存 bar-value CSV、summary、validation、manifest 和独立 checksums。
- `Au_Pd_independent/figures/rp_2d/`：potential-only 与 potential+reactants 两组 2D 图；其中 potential-only 发布版在本 study wrapper 内使用 `2026 × 1852 px` 半宽画布，Au/Pd 两个窄 panel 保持横向并列并共用右侧 colorbar，不做整图水平压缩；共享 independent-EDL 绘图模块仍保留原默认版式。
- `Au_Pd_independent/figures/Polarization_Scheme/`：10 mM 基准条件的 signed half-reaction polarization curve，Au 氧化为绿色、Pd 还原为红色，1 PNG + 1 SVG；同时保存 curve CSV、summary、validation、manifest 和独立 checksums。
- `Au_Pd_independent/`：`params`、`derived`、`summary`、`validation`、`manifest`、profile CSV 和 SHA-256 checksums。
- `Au_Pd_independent/C_tot_study/figures/Figure_4/`：`E_mix`、`i_mix_avg` 浓度趋势和 half-reaction polarization overlay，共 3 PNG + 3 SVG；两张趋势图统一使用 `3 × 3.45 in`（600 dpi 为 `1800 × 2070 px`）的固定窄画幅，导出明确使用 `bbox_inches=None`，不允许 tight bounding box 改变画布。两张趋势图不显示顶部解释性标题，并保留 `10^3 M` 端点与曲线但不显示 `10^3 M` 文字标注。`E_mix` 范围为 `0.40–0.67 V`；`i_mix_avg` 按 `1 A/m² = 0.1 mA/cm²` 换算，纵轴单位为 `mA/cm²` 且从 0 开始。polarization overlay 使用带符号的绝对 Au/Pd 半反应电流，纵轴写作 `Current (10^-3 uA)` 并关于 0 对称，不使用 current density 或非负 magnitude。
- `Au_Pd_independent/C_tot_study/figures/Mechanism/`：Au/Pd 两侧的 `sigma → phi_RP → eta_RP → kinetic/concentration weight → i_mix_avg` 因果链，1 PNG + 1 SVG。
- `Au_Pd_independent/C_tot_study/figures/EDL_scheme/`：0.01/1 M 解析电位 profile 与 w/o-EDL 零线；保留原 `2×1` 版本，并新增 Au/Pd 横向并列的 `1×2` 版本，共 2 PNG + 2 SVG。横向版采用固定画幅，其 600 dpi PNG 宽度严格匹配 half-reaction polarization overlay，导出不使用 tight bounding box。
- `Au_Pd_independent/C_tot_study/`：共 6 PNG + 6 SVG + 0 PDF；另含 5 个 CSV、完整参数/derived/scan config/summary/validation/manifest、artifact hashes 和独立 SHA-256 checksums。方法和版式参考 `20260803_153355_ctot_study`，数值由本组 2 nm、`C_H=50/50` 参数重新计算。

`L_support` 主扫描使用 18 个点：`0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10 nm`；overlap 诊断另算 `11, 12, 15 nm`，并以 `1000 nm` 为独立 EDL 参考。短扫描使用 `N_modes=960`、每侧 128 点 Gauss–Legendre 积分，1000 nm 参考使用 `N_modes=7680`。

本组 OFAT 的长度尺度结果：

- `E_mix` 的 5–10 nm 指数拟合衰减长度为 `2.1444 nm`，`R² = 0.9999879`。
- `L_support = L_GC,C` 的交点为 `0.8638 nm`。
- 相对 1000 nm 参考，三项 overlap 指标同时衰减到接触值的 5% 时，连续插值为 `5.7396 nm`，首个采样点为 `6 nm`。
- 以 `0.1 mV / 0.1 mV / 0.1%` 的三项严格阈值判定时，连续边界为 `12.7567 nm`，首个采样点为 `15 nm`。
- 指数拟合的 `0.1 mV` 平台位置为 `11.2489 nm`，位于 0–10 nm 主扫描之外，因此没有主扫描内的首个平台采样点。

当前总输出为 80 PNG + 80 SVG + 0 PDF。

## 适用范围

两种模型均使用 linearized Poisson–Boltzmann / Debye–Hückel。保存的 validation 保留 `max |phi_tilde|`；本组结果明显超过 1，因此这是线性模型内部比较，不应视为已定量验证的强场结果。

`C_tot` 扫描中 36 个点有 27 个超过 `max |phi_tilde| = 1` 的弱场阈值，扫描最大值约为 `10.179`；该项作为 applicability caveat 写入 validation，但不与已经通过的求根、电流平衡、Stern 电荷关系和动力学重构数值检查混为一谈。

旧 Au｜C｜Pd 模型使用 cosine/Fourier 展开，材料边界附近仍可能出现 Gibbs 特征。独立模型的 2 nm 只用于活性面积，不解析有限电极的边缘场。旧模型的 `L_support = 0` 是连续电解液边界上的相邻 Au｜Pd，不等同于两个独立半空间电极。

support-PZC 专题把 15 nm 用作有限长度的 RP-overlap 参考，并通过 1000 nm 三个锚点量化其与远场极限的差异；不能把 15 nm 宣称为普适的无 overlap 边界。专题保留 Debye–Hückel 超限和 Fourier/Gibbs caveat，但这些适用性提醒与已通过的求根、电流平衡和数值收敛检查分开记录。新增的二维输出只覆盖 12 个代表点；每点同时提供 potential-only 和 potential+Red1+Ox2 两种版式，不把 164 点主扫描全部扩成二维图。

support-PZC 主扫描 164 个点全部超过 `max |phi_tilde| = 1` 的弱场阈值，扫描最大值约为 `8.3666`。生产收敛的最大变化为 `0.01946 mV / 0.06275%`，GL64→128 的最大变化为 `0.000253 mV / 0.000742%`，最大相对电流平衡残差为 `7.88e-13`；这些数值检查通过不消除 Debye–Hückel 的物理适用性限制。
