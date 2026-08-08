# Support \(C_H\) study

本目录使用 Figure_same_length_i0_alpha 基线，单独扫描非反应性 support 的
Helmholtz 电容 \(C_{\mathrm{H},\mathrm{support}}\)。内部 solver 兼容键仍为
Cdl_C，图面和新文件名只使用 \(C_H\)。

## 设置

- \(C_{\mathrm{H},\mathrm{support}}=0\)–\(1.0\ \mathrm{F\,m^{-2}}\)
  （0–100 \(\mu\mathrm{F\,cm^{-2}}\)），41 个线性点。
- 基线为 \(0.10\ \mathrm{F\,m^{-2}}=10\ \mu\mathrm{F\,cm^{-2}}\)。
- 其他物理参数保持
  same_length_i0_alpha050_au25_pd25_20260528_111255 不变。
- 数值分辨率：N_modes=960、Nx=5000，FULL lateral EDL model。
- \(C_{\mathrm{H},\mathrm{support}}=0\) 表示 support 上
  \(g_{\mathrm{support}}=0\) 且局部 compact-layer charge 为零；10 nm support
  几何仍然存在，因此它既不等于 L_gap=0，也不等于全体系 w/o EDL。

## 主要结果

在当前线性模型内部，随 support \(C_H\) 从 0 增至
100 \(\mu\mathrm{F\,cm^{-2}}\)：

- \(E_{\mathrm{mix}}\) 从 0.598833846 V 单调降至 0.595288805 V，
  总变化为 -3.545 mV。
- \(\bar{i}_{\mathrm{mix}}\) 从 0.08248951 增至
  0.08928160 \(\mathrm{A\,m^{-2}}\)，增加 8.234%。
- support 区平均 \(\phi_{\mathrm{RP}}\) 从 -29.60 mV 变为 +65.13 mV；
  同期 Au/Pd 区平均 \(\phi_{\mathrm{RP}}\) 只分别变化约 +2.26/+0.49 mV。

这说明 support 本身虽然不发生反应，其 \(C_H\) 仍可通过横向 EDL
耦合间接改变 Au/Pd 动力学；在本基线下，该影响存在但明显小于
active-side 界面参数的影响。

## 输出

- 本目录的 TAG 为 same_length_i0_alpha050_au25_pd25_20260528_111255。
- 综合图：
  support_ch_effect_same_length_i0_alpha050_au25_pd25_20260528_111255.png/svg
- 41 点扫描数据：
  csv/support_ch_sweep_same_length_i0_alpha050_au25_pd25_20260528_111255.csv
- 0/10/100 \(\mu\mathrm{F\,cm^{-2}}\) 空间剖面：
  csv/support_ch_phi_rp_profiles_same_length_i0_alpha050_au25_pd25_20260528_111255.csv
  其中材料交界点单独标为 boundary，is_support_mask 保留 support 积分所用的
  共享端点定义。
- inputs/：完整参数、overrides、高分辨率 baseline summary、扫描配置和
  关键结果 summary。

扫描 CSV 中 I_Au_abs_A 与 I_Pd_abs_A 的 abs 表示已按实际宽度换算成安培，
不是对带符号电流取数学绝对值；Pd 阴极电流仍为负。mixed potential 的判据
始终是 I_Au_abs_A + I_Pd_abs_A = 0，mixed current density 使用同一个总
reactive area 归一化。

重新生成：

    PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=.matplotlib-cache \
      .venv/bin/python \
      Figures/Figure_support_CH/make_support_ch_study_same_length_i0_alpha.py

只导出 PNG/SVG，不导出 PDF。

## 模型适用性提醒

全扫描的 max_abs_phi_tilde 为 6.019–6.083，均高于线性
Debye–Hückel 阈值 1。因此这里的结果应表述为当前线性模型内部的参数敏感性，
不应把静电势的绝对量级表述为已经定量验证。
