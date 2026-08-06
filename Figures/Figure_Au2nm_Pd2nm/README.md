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

脚本只生成 600 dpi PNG 和可编辑文字 SVG，不生成 PDF。当前入口同时生成六个正式 case、`L_support` OFAT，以及独立 Au｜Pd 的 Figure 3/2D 图；仍不生成 EDL scheme 或 polarization scheme。

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

## 输出结构

- `Au_C_Pd/Case_Figures/Figure_3/`：reaction-plane potential（panel b）和 local current density（panel e）；六个正式 case 各两组全图，1000 nm 另有两组双侧 active-window 图，共 14 PNG + 14 SVG。
  - panel b 沿用 Figure 3 的 `with EDL` 橙色与 `w/o EDL` 深蓝色。
  - panel e 与独立模型 Figure 3 一致：Au 为绿色、Pd 为蓝色；`with EDL` 为实线，`w/o EDL` 为同色虚线。
- `Au_C_Pd/Case_Figures/Figure_RP/`：2D solution potential 和 surface charge distribution；六个正式 case 各两组全图，1000 nm 另有两组双侧 active-window 图，共 14 PNG + 14 SVG。材料显示色为 Au `#E4C133`、C `#8C8C8C`、Pd `#5A90C8`。
- `Au_C_Pd/Figure_L_support/OFAT/`：8 组 support-length OFAT 图；`csv/`、`inputs/`、`manifest.json` 和 `validation.json` 保存扫描结果与追溯信息。
- `Au_C_Pd/inputs/`：每个 case 的完整参数、可重放 overrides 和 CSV/JSON summary。
- `Au_C_Pd/csv/`：每个 case 的独立 profile CSV、汇总 profile、case summary 和收敛检查。
- `Au_C_Pd/validation.json`：电流平衡、2D 表面回代、图件数量和数值收敛验证。
- `Au_Pd_independent/figures/Figure_3/`：Figure 3 a–f 六组图。
- `Au_Pd_independent/figures/rp_2d/`：potential-only 与 potential+reactants 两组 2D 图。
- `Au_Pd_independent/`：`params`、`derived`、`summary`、`validation`、`manifest`、profile CSV 和 SHA-256 checksums。

`L_support` 主扫描使用 18 个点：`0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10 nm`；overlap 诊断另算 `11, 12, 15 nm`，并以 `1000 nm` 为独立 EDL 参考。短扫描使用 `N_modes=960`、每侧 128 点 Gauss–Legendre 积分，1000 nm 参考使用 `N_modes=7680`。

本组 OFAT 的长度尺度结果：

- `E_mix` 的 5–10 nm 指数拟合衰减长度为 `2.1444 nm`，`R² = 0.9999879`。
- `L_support = L_GC,C` 的交点为 `0.8638 nm`。
- 相对 1000 nm 参考，三项 overlap 指标同时衰减到接触值的 5% 时，连续插值为 `5.7396 nm`，首个采样点为 `6 nm`。
- 以 `0.1 mV / 0.1 mV / 0.1%` 的三项严格阈值判定时，连续边界为 `12.7567 nm`，首个采样点为 `15 nm`。
- 指数拟合的 `0.1 mV` 平台位置为 `11.2489 nm`，位于 0–10 nm 主扫描之外，因此没有主扫描内的首个平台采样点。

当前总输出为 44 PNG + 44 SVG + 0 PDF。

## 适用范围

两种模型均使用 linearized Poisson–Boltzmann / Debye–Hückel。保存的 validation 保留 `max |phi_tilde|`；本组结果明显超过 1，因此这是线性模型内部比较，不应视为已定量验证的强场结果。

旧 Au｜C｜Pd 模型使用 cosine/Fourier 展开，材料边界附近仍可能出现 Gibbs 特征。独立模型的 2 nm 只用于活性面积，不解析有限电极的边缘场。旧模型的 `L_support = 0` 是连续电解液边界上的相邻 Au｜Pd，不等同于两个独立半空间电极。
