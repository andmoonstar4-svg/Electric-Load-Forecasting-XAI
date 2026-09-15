# 基于 iTransformer 与 XAI 的极端天气电力负荷预测

> 常规与极端天气下的日前电力负荷预测：多模型对比 + SHAP 可解释性分析。
> 仓库自带完整可复现管线；**原始数据不入库**（Elia 数据不可再分发，见 [数据许可](#数据许可)）。

![tests](https://img.shields.io/badge/tests-150%20passed-brightgreen)
![python](https://img.shields.io/badge/python-3.11-blue)

---

## 这个仓库解决什么问题

论文对比了 **iTransformer / DLinear / Autoformer / 随机森林** 在常规与极端天气下的日前负荷预测，并用 SHAP 分析特征归因。

但原仓库（1 个 commit、180 个文件）里**只有两份上游项目的整包拷贝（LTSF-Linear + Time-Series-Library），自研代码为 0 行**——论文第 2～6 章描述的每一步都不在里面。别人 clone 之后无法复现任何一张表。

本次重构把仓库变成"论文的代码化版本"：**每张表、每张图都能由一条命令重新生成，并带完整溯源**。

| 论文产物 | 生成方式 |
|---|---|
| 表 3-1 数据集 1 常规天气 | `elxai compare --models iTransformer,DLinear,Autoformer,RandomForest` |
| 表 3-2 数据集 2 外部验证 | 同上，换 `configs/data/tetuan.yaml` |
| 表 4-1 极端天气 | 同上，加 `configs/experiment/extreme.yaml` |
| 表 6-1 损失函数消融 | `elxai ablation --losses mae,mse,exp` |
| 图 5-1～5-4 SHAP 归因 | `elxai xai`（`xai.*` 控制采样与分辨率） |
| 数据来源与许可 | `runs/<id>/provenance.json` + `docs/provenance.md` |

---

## 架构

单向分层：上层可依赖下层，下层永不反向依赖。

```
configs/                 YAML 实验配置（data / model / experiment 可组合）
   │
src/elxai/
   ├── config.py         配置 dataclass、组合、校验、内容指纹
   ├── data/             ★ 数据层（不依赖 torch）
   │   ├── schema.py       数据契约：特征顺序、TaskSpec、帧校验
   │   ├── provenance.py   数据源许可登记与运行溯源
   │   ├── cleaning.py     Elia/ERA5 清洗与对齐
   │   ├── labels.py       极端天气判定与子集构造
   │   ├── windows.py      时序切分（含 purge）、窗口化、标准化
   │   └── datasets.py     唯一入口：Config → DatasetSplits
   ├── models/           统一 Forecaster 接口
   │   ├── torch_adapter.py  上游 iTransformer/DLinear/Autoformer 适配
   │   └── forest.py         随机森林基线（同接口，可参与同一评价与归因）
   ├── training/         损失（含指数惩罚）、指标、训练编排、run 目录
   ├── xai/              SHAP 归因（重写）
   ├── experiments/      每个论文产物一个函数
   └── cli.py            dataset / train / compare / ablation / xai / summarize
   │
third_party/             上游代码（只保留实际使用的 4 个模型，16 文件）+ PROVENANCE.json
tests/                   150 个测试，全部使用合成数据
docs/                    架构审计、迁移指南、数据许可
```

设计要点见 [`docs/architecture.md`](docs/architecture.md)，逐条列了发现的问题与对应修复。

### 三条最重要的设计决定

1. **数据契约是代码，不是散文。** 论文 S5.1 的 `feature0~6 = u10,t2m,msl,ssrd,tcc,tp,Total Load` 固化在 `schema.PREDICTORS` 与 `TARGET_INDEX`，并有校验与测试。静默重排一列会让所有 SHAP 图失效，所以顺序由常量和断言保护。
2. **损失函数是 run 标识的一部分。** 原实现的 `setting` 字符串不含 `--loss`，导致表 6-1 的三行写进同一目录互相覆盖。现在 `run_id = <name>__<fingerprint>`，指纹覆盖整份配置。
3. **溯源是产物，不是段落。** 每次运行写出 `provenance.json`（各源文件 SHA-256、train/val/test 的具体时间范围、git commit 与是否 dirty）与 `dataset_report.json`（清洗过程的每一个判决：丢了哪些列、丢了多少行、极端天气命中多少小时）。论文的 100994 / 52417 / 8689 因此可核对。

---

## 安装

```bash
# 只跑数据与指标层（不需要 GPU、不需要 torch）
pip install -e .

# 训练上游神经模型
pip install -e ".[torch]"

# SHAP 归因
pip install -e ".[xai]"

# 全部 + 测试
pip install -e ".[torch,xai,dev]"
```

上游模型代码已经在 `third_party/`，无需额外下载。若想从上游重新取回：

```bash
python scripts/fetch_third_party.py --clone --ref <commit>
```

---

## 数据准备

原始数据不入库。设一个环境变量指向你的本地副本：

```bash
export ELXAI_DATA_DIR=/path/to/raw        # Windows: $env:ELXAI_DATA_DIR="D:\data"
```

需要的文件（路径可在 config 里覆盖）：

| 文件 | 来源 |
|---|---|
| `ods001.csv` | Elia 电网负荷导出（`;` 分隔，含 Total Load 与若干预测列） |
| `reanalysis-era5-single-levels-timeseries-*.csv` | ERA5 单点时间序列（50.0N, 4.0E） |
| Tetuan 数据 | 数据集 2 用；配置里指向合并好的 CSV |

先确认数据规模与论文一致：

```bash
elxai dataset \
  --config configs/data/belgium.yaml --config configs/split.yaml \
  --out runs/dataset1_check

# 读 runs/dataset1_check/dataset_report.json：
#   range.n_rows                              对照论文的 100994
#   cleaning.load_rows_dropped_by_tolerance   若 >0，说明原预处理丢过观测
#   cleaning.dropped_forecast_columns         应为 3 个模型生成的预测列
#   frame.n_gap_steps                         时间网格缺口数
```

极端天气子集规模：加 `--config configs/experiment/extreme.yaml`，看 `extreme.n_subset`（对照 8689）。

---

## 复现论文结果

```bash
# 表 3-1：数据集 1 常规天气（每个模型用它自己的配置，参数才会正确）
for m in itransformer dlinear autoformer randomforest; do
  elxai train \
    --config configs/data/belgium.yaml --config configs/split.yaml \
    --config configs/experiment/normal.yaml --config configs/model/compare_$m.yaml
done
elxai summarize runs/normal-weather-belgium/ --out results/table3-1.csv

# 表 3-2：数据集 2 外部验证（换成 tetuan 配置，seq_len 自动变为 72）
elxai train --config configs/data/tetuan.yaml --config configs/split.yaml \
  --config configs/experiment/normal.yaml --config configs/model/compare_itransformer.yaml

# 表 4-1：极端天气（batch_size 8、极端子集都在 experiment/extreme.yaml 里）
elxai compare --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/experiment/extreme.yaml --config configs/model/compare_itransformer.yaml \
  --models iTransformer,DLinear,Autoformer,RandomForest --out results/table4-1.csv

# 表 6-1：损失函数消融
elxai ablation --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/model/compare_autoformer.yaml --config configs/experiment/loss_ablation.yaml \
  --models Autoformer,iTransformer --losses mae,mse,exp --out results/table6-1.csv

# 图 5-1～5-4：SHAP 归因
elxai xai --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/model/compare_itransformer.yaml --config configs/experiment/xai.yaml \
  --set data.extreme.enabled=true --set training.loss=mse
```

任何字段都可以用 `--set section.field=value` 覆盖，例如：

```bash
--set training.batch_size=8 --set xai.target_steps=all --set xai.resolution=variable_time
```

`elxai config` 会打印组合后的配置与内容指纹，用来确认你跑的到底是什么。

### 指数损失的两个参数

论文正文写 `2^(2|e|)−1` 且"取 β = 2"，但代码是 `EMEX(pred, true, X=4)`，即 `2^(4|e|)−1`——`X=4` 从未出现在论文里。配置里两个参数都显式：

```yaml
training:
  loss: exp
  exp_loss_beta: 2.0     # 底数 β
  exp_loss_alpha: 4.0    # 指数系数，对齐产生已发布表格的那份代码
```

要复现正文的 `2^(2|e|)` 形式，设 `exp_loss_alpha: 1.0`。

---

## 测试

```bash
pytest -q          # 150 passed
```

全部使用**合成数据**。测试本身锁定了若干具体缺陷，例如：

| 测试 | 防止的问题 |
|---|---|
| `test_no_window_overlaps_the_next_split` | 训练窗口的预测目标落进验证块（原实现无 purge） |
| `test_clean_time_index_rejects_unparseable` | 不可解析时间戳变成 NaT 后排序到最前，**使所有窗口整体错位一行** |
| `test_readings_beyond_the_legacy_tolerance_are_reported` | 15 分钟负荷与小时天气对齐时静默丢行 |
| `test_completeness_residual_is_small_for_a_linear_model` | SHAP 值必须满足 `Σφ + E[f] = f(x)` |
| `test_variable_resolution_sums_over_lags_and_matches_ground_truth` | 归因必须能识别出"模型只依赖目标通道" |
| `test_fingerprint_changes_with_the_loss` | 表 6-1 的三行不能互相覆盖 |
| `test_scaler_is_fitted_on_training_rows_only` | 标准化统计量不得用测试期数据拟合 |
| `test_auto_nsamples_scales_with_dimension` | 对 1176 维输入不能只用 100 次采样 |

端到端冒烟（不需要真实数据，用合成 CSV，约几分钟）：

```bash
elxai compare --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/experiment/smoke.yaml --config configs/model/compare_dlinear.yaml \
  --models DLinear
```

---

## run 目录结构

每次运行产出完整可审计的记录：

```
runs/<experiment>__<dataset>__<model>__L<seq>H<pred>__<loss>__<fingerprint>/
├── config.yaml               本次运行使用的完整配置
├── provenance.json           数据源 SHA-256、切分时间范围、git commit/dirty
├── dataset_report.json       清洗与极端天气子集的每一个判决
├── splits.json               train/val/test 的行号与时间范围
├── scaler.json               标准化参数（仅 fit 训练集）
├── metrics.json              MSE/MAE/RMSE/分步/峰值段/偏差/环境
├── metrics_by_horizon.csv    24 个预测步的误差
├── training.json             逐轮 loss、最佳轮次、参数量、设备
├── predictions.npz           预测与真值（可重算任何指标）
├── model/                    模型权重与配置
├── shap/                     SHAP 值、报告（采样数、分辨率、完备性残差）
└── figures/                  SHAP 图
```

**同一配置重复运行不会覆盖既有记录**，而是追加时间戳——幂等可查，有损操作被禁止。

---

## 数据许可

| 数据 | 许可 | 可否再分发 |
|---|---|---|
| Elia 比利时负荷 | Elia 开放数据条款 | **否** |
| ERA5 | Copernicus 许可（需署名） | 需署名 |
| Tetuan City | CC BY 4.0 | 需署名 |

机器可读的登记表在 `src/elxai/data/provenance.py`，可用 `elxai dataset --check-redistribution` 自动校验。仓库里**不含任何原始数据文件**。

给论文补充的数据许可说明文本见 [`docs/provenance.md`](docs/provenance.md) §3（中文）与 §4（英文）。

---

## 从原仓库迁移

原仓库的每个文件去了哪里、每个 bug 怎么修的、为什么某些数值会与原结果不同，见 [`docs/migration.md`](docs/migration.md)。

改动指标的四项（均有理由，且可切换到原行为做对照）：

1. **purge**：训练块末尾不再把验证期观测当预测目标 → 原数字略乐观；`--set split.purge=0` 可恢复。
2. **亚小时负荷对齐**：不再丢行 → 训练数据变多。
3. **极端子集**：明确为"极端日 + 之前 7 整天"，跨缺口窗口被丢弃。
4. **标准化口径**：scaler 只 fit 训练集，且模型内部 instance 标准化被记录。

---

## 引用

若使用本项目的代码，请引用原论文所依据工作：

- LIU Y, HU T, ZHANG H, et al. *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.* ICLR, 2024.
- ZENG A, CHEN M, ZHANG L, et al. *Are Transformers Effective for Time Series Forecasting?* AAAI, 2023, 37(9): 11121-11128.
- WU H, XU J, WANG J, et al. *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS, 2021, 34.
- LUNDBERG S M, LEE S I. *A Unified Approach to Interpreting Model Predictions.* NeurIPS, 2017, 30.
- BREIMAN L. *Random Forests.* Machine Learning, 2001, 45(1): 5-32.

数据来源引用见 [`docs/provenance.md`](docs/provenance.md)。

上游模型实现来自 [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)（MIT）与 [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)，感谢作者开源。
