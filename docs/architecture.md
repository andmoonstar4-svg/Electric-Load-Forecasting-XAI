# 架构审计与目标设计

本文档记录一次针对 `Electric-Load-Forecasting-XAI` 的架构重构：**审计了什么、发现了什么、改成了什么、以及为什么这样改**。

重构的目标不是"代码更好看"，而是让论文里的每一张表、每一张 SHAP 图都能从仓库里的代码重新跑出来。

---

## 1. 审计结论：原仓库的状态

| 项 | 事实 |
|---|---|
| 提交数 | 1（`1367ae9`） |
| 跟踪文件 | 180 |
| 自研代码 | **0 行** |
| `DLinear-main/` | cure-lab/LTSF-Linear 整包 vendored（约 60 文件，含 `pics/` 宣传大图、`scripts/*.sh`、`__pycache__/*.pyc`） |
| `Time-Series-Library-main/` | thuml/Time-Series-Library 整包 vendored（约 120 文件，含 60+ 未使用架构、Dockerfile、CI） |
| `report/` | 论文 PDF |
| `README.md` | 唯一自研产物 |

论文第 2～6 章描述的**每一步都不在仓库里**：数据集构建、随机森林基线、SHAP 归因、指数惩罚损失。别人 clone 之后无法复现任何一张表。

调研中还发现，作者本机的 `C:\Users\27247\Downloads\Load-Forecasting-iTransformer`（`origin` 指向同一 GitHub 仓库）**包含远端没有的自研改动**：

- `run.py` 新增 `--do_shap` / `--shap_background`
- `exp/exp_long_term_forecasting.py` 新增 `shap_analysis()`（461 行文件里的第 283–461 行）
- `data_provider/data_loader.py` 有一处改动

也就是说：**能跑的代码只存在于本机，仓库里只有两份上游拷贝。** 这是本次重构要解决的首要问题。

---

## 2. 发现的具体缺陷（每条都可验证）

### 2.1 可复现性

| # | 缺陷 | 证据 | 后果 |
|---|---|---|---|
| R1 | 损失函数不在 run 标识里 | `run.py` 的 `setting` 字符串由 `task/model_id/model/data/features/sl/ll/pl/dm/nh/el/dl/df/expand/dc/fc/eb/dt/des/ii` 拼成，**不含 `--loss`** | 表 6-1 的三行（MAE/MSE/指数）写进**同一个** `./results/<setting>/`，互相覆盖。已发布的表格无法从产物重建 |
| R2 | SHAP 输出路径同样漏掉 loss | `folder = './shap_results/' + args.model + '_' + setting + '/'` | 图 5-1 与 5-2（同一模型、不同损失）只有靠别的超参不同才不冲突 |
| R3 | 标准化的口径没有记录 | `Dataset_Custom` 用 `StandardScaler` 做数据集级逐列标准化；iTransformer/DLinear 内部**又**做了一次 instance 级标准化 | 论文说"MSE/MAE 均作了标准化处理"，但没有说清是哪种；两次归一化叠加后指标不可跨实现比较 |
| R4 | 超参数散落在 shell/argparse | 常规 `batch_size=48`、极端 `8`，只写在论文正文和命令行里 | 无法从配置回溯某次实验 |
| R5 | 时序列切分**没有 purge** | `data_loader.py` 的 `border1s/border2s` 直接按 0.7/0.1/0.2 切 | 训练块末尾的窗口，其**预测目标**落进了验证块 |

### 2.2 数据管线（这是最严重的一类）

| # | 缺陷 | 证据 | 后果 |
|---|---|---|---|
| D1 | 15 分钟负荷与小时天气用 `merge_asof(tolerance='30min')` 对齐 | `新建 文本文档.py` 预处理脚本 | 时间戳若与整点有偏移，落在容差外的读数被静默丢弃；幸存行仍构成"干净"小时序列，所以看不出来 |
| D2 | 时间戳校验形同虚设 | `_clean_time_index` 检查的是**原始列**的 `isna()`；原始列是字符串，`isna()` 恒为 False | 不可解析的时间戳不会报错，会变成 `NaT`，而 `NaT` 排序在**最前**，导致**所有窗口和切分边界整体错位一行** |
| D3 | 极端天气阈值与论文不一致 | 脚本用 `tp > 0.01`、`u10 > 10`、`msl < 100000`；论文写"小时降水量超过 5??/h、风速超过 15?/?" | 论文的阈值单位是 OCR 残缺（`5??/h`）；且脚本多了一个论文没有的低气压条件 |
| D4 | 极端天气子集不连续，但直接窗口化 | "保留极端事件所在完整日期及之前 7 天" | 帧内存在跨月的跳跃，跨跳跃的窗口含虚假阶跃，且没有任何机制拦截 |

### 2.3 SHAP 归因（`shap_analysis()`）

| # | 缺陷 | 原实现 | 后果 |
|---|---|---|---|
| X1 | 背景样本只有 5 个 | `background_x_np = batch_x[:5]` | `E[f]` 估计几乎无精度 |
| X2 | 1176 维输入只采样 100 次 | `explainer.shap_values(test_flat, nsamples=100)` | 对 1176 维合作博弈采样 100 次，SHAP 值可信度极低，完备性不成立 |
| X3 | 只解释 3 个样本 | `test_x_np = batch_x[5:8]` | 归因结果方差大 |
| X4 | 把 `(168, 7)` 展平成 1176 维再按列平均 | `np.mean(np.abs(shap_values), axis=(0,1))` | 在时间轴上直接平均隐含了 product kernel 假设，逐变量得分**不保证**能加总回预测值 |
| X5 | 只解释未来第 1 小时 | `return outputs[:, 0, 0]` | 图注暗示解释整个 24 小时预测，实际只有 h+1 |
| X6 | 异常被吞掉 | `except Exception as e: print(...)` | 图可能静默缺失，却看起来像正常完成 |
| X7 | 函数体后半段是死代码 | 第 394–461 行重复定义了 `model_predict` 和第二段 SHAP 块，永不执行 | 维护陷阱：改错位置不会生效 |
| X8 | 背景样本取自**测试集** | `test_data, test_loader = self._get_data(flag='test')` | 基线 `E[f]` 含测试期信息 |

### 2.4 论文与代码的其他不一致

| # | 论文 | 代码 |
|---|---|---|
| P1 | 指数损失写作 `2^(2\|y-ŷ\|) − 1`，正文说 `取β = 2` | `EMEX(pred, true, X=4)` 实为 `mean(2^(4\|e\|) − 1)`；`X=4` 从未出现在论文里 |
| P2 | `feature0~6 = u10,t2m,msl,ssrd,tcc,tp,Total Load` | 该映射只存在于论文正文和 `cols` 属性，没有任何断言保护 |
| P3 | 极端天气降水阈值 `5??/h` | 脚本用 `0.01`（ERA5 `tp` 单位是米，即 10 mm/h） |
| P4 | 早停"轮数为 3" | `--patience` 默认 3，但未写进论文的实验设置 |

---

## 3. 目标架构

分层单向依赖：`data → models → training → xai → experiments`。上层可以依赖下层，下层永不反向依赖。

```
repo/
├── pyproject.toml              可 pip install -e .；torch/shap 作为可选 extra
├── src/elxai/
│   ├── config.py               配置 dataclass + YAML 组合 + 指纹
│   ├── cli.py                  dataset/train/compare/ablation/xai/summarize
│   ├── third_party.py          上游代码路径注册 + 源码内省
│   ├── data/
│   │   ├── schema.py           ★ 数据契约：特征顺序、TaskSpec、校验
│   │   ├── provenance.py       ★ 数据溯源与许可登记
│   │   ├── cleaning.py         Elia/ERA5 清洗与对齐
│   │   ├── labels.py           极端天气判定与子集构造
│   │   ├── windows.py          切分（含 purge）、窗口化、标准化
│   │   └── datasets.py         单一入口：Config -> DatasetSplits
│   ├── models/
│   │   ├── base.py             Forecaster 接口 + FitReport
│   │   ├── torch_adapter.py    上游模型适配 + config 属性内省校验
│   │   └── forest.py           随机森林基线（同一接口）
│   ├── training/
│   │   ├── losses.py           三种损失，含指数惩罚
│   │   ├── metrics.py          指标 + 分步 + 峰值段 + 自助置信区间 + DM 检验
│   │   └── trainer.py          ★ run 目录布局与训练编排
│   ├── xai/shap_explainer.py   ★ 重写的 SHAP 层
│   └── experiments/runner.py   每个论文产物一个函数
├── configs/{data,model,experiment}/
├── third_party/                上游代码 + PROVENANCE.json
├── scripts/fetch_third_party.py
├── tests/                      111 个测试，全部使用合成数据
└── docs/
```

### 关键设计决策

**D-1：数据契约是代码，不是散文。**
`schema.py` 把 `feature0..6` 固化成 `PREDICTORS` 元组与 `TARGET_INDEX` 常量，并配 `validate_frame()`。任何列缺失、时间戳重复、时间网格不规则、NaN，都会在数据边界直接抛错。`tests/test_schema.py` 锁死这个顺序——静默重排会让所有 SHAP 图失效。

**D-2：溯源是一等产物，不是 README 段落。**
`provenance.py` 为每个数据源声明 `licence` / `redistribution` / `citation`，每次运行写出 `provenance.json`（含每个源文件的 SHA-256、实际的 train/val/test 时间范围、git commit、是否 dirty）。比利时 Elia 数据标记为 `redistribution="no"`，`assert_redistributable()` 让打包步骤可以拒绝。原始数据永不入库。

**D-3：run 目录由配置指纹决定，损失函数是标识的一部分。**
`run_id = <name>__<fingerprint>`，指纹覆盖整份配置。因此表 6-1 的三行天然落在三个目录。同一配置重复运行不会覆盖既有记录，而是追加时间戳——**幂等可查，但有损操作被禁止**。

**D-4：purge 是默认行为，放宽必须显式且留痕。**
`chronological_split` 默认在验证块与测试块前各丢弃 `seq_len + pred_len - 1` 行，这是保证"训练窗口的输入与目标都不越过边界"的最小值。`purge=0` 会被解释为"使用安全默认值"；显式设成比安全值小会记 WARNING。

**D-5：上游代码是依赖，不是你的代码。**
`third_party/` 只保留实际使用的 4 个模型与它们 import 的层，共 16 个文件，并记录 URL / commit / LICENSE。`third_party.py` 负责把它挂到 `sys.path`，而不是依赖"从仓库根目录启动进程"这一隐式前提。

**D-6：上游 config 需求由源码解析得出。**
`required_config_attrs()` 用正则从上游模型文件里抽出所有 `configs.<attr>` 访问，再与适配器提供的命名空间取差集。缺默认值时**立刻报出属性名**，而不是训练 40 分钟后 `AttributeError`。

**D-7：SHAP 的每个可疑默认值都改为显式参数，并输出残差。**
`nsamples='auto'` 按特征维度推导（`4×d`，下限 2000，上限 20000）；`resolution` 显式选择是否保留 lag 轴；`target_steps` 可选 `"all"`；背景样本取自**训练集**。每次归因都计算**完备性残差** `|Σφ + E[f] − f(x)|` 并写入 JSON——它让"归因不可信"变成一个数字，而不是一种猜测。

**D-8：模型内部归一化被标记，因为它改变 SHAP 的含义。**
iTransformer/DLinear 在 `forward` 内对每个窗口做标准化。扰动某个输入会同时改变窗口统计量。`uses_internal_normalisation` 标志与 `model_normalises_windows` 字段把这一点带到产物里。

---

## 4. 缺陷 → 修复对照

| 缺陷 | 修复位置 | 守护测试 |
|---|---|---|
| R1/R2 | `config.Config.fingerprint()`（含 loss）、`trainer.RunPaths` | `test_fingerprint_changes_with_the_loss` |
| R3 | `ScalingConfig` + `Scaler` 只 fit 训练集 + `scaler.json` | `test_scaler_is_fitted_on_training_rows_only` |
| R4 | `configs/experiment/*.yaml` | `test_shipped_experiment_configs_compose` |
| R5 | `windows.chronological_split` 的 purge | `test_no_window_overlaps_the_next_split` |
| D1 | `cleaning.merge_load_weather` 按实测间距选策略 + 报告旧行为丢了多少 | `test_readings_beyond_the_legacy_tolerance_are_reported` |
| D2 | `cleaning._clean_time_index` 校验**解析后**的序列 | `test_clean_time_index_rejects_unparseable` |
| D3 | `ExtremeConfig` 显式阈值 + 单位；`labels.flag_extreme_weather` | `test_tp_threshold_unit_is_respected` |
| D4 | `labels.select_extreme_subset` 三种模式 + `make_windows` 丢弃跨缺口窗口 | `test_windows_spanning_a_gap_are_dropped` |
| X1/X3/X8 | `XAIConfig.n_background/n_eval`，背景取 `data["train"]` | `test_background_and_eval_counts_are_respected` |
| X2 | `auto_nsamples` / `resolve_nsamples` | `test_auto_nsamples_scales_with_dimension` |
| X4 | `resolution` 参数 | `test_variable_resolution_sums_over_lags_and_matches_ground_truth` |
| X5 | `target_steps` 参数 | `test_original_explained_only_step_zero_this_allows_the_whole_horizon` |
| X6 | 不再吞异常；残差写入 JSON | `test_completeness_residual_is_small_for_a_linear_model` |
| X7 | 重写为单一函数 | — |
| P1 | `ExpPenaltyLoss(beta, alpha)` 双参数并写入产物 | `test_loss_metadata_records_the_exponent` |
| P2 | `schema.PREDICTORS` / `TARGET_INDEX` | `test_canonical_order_matches_paper_feature_indices` |
| P3 | `tp_min_m` + `tp_unit` | 同上 D3 |
| P4 | `TrainingConfig.patience` + 写入 `metrics.json` | `test_early_stopping_can_end_before_the_epoch_limit` |

---

## 5. 有意**不**改的部分

- **不引入 Hydra / Lightning。** 组合式 YAML + dataclass 校验已足够，且没有把配置变成框架魔法。上游代码本身不依赖 Hydra。
- **不用 git submodule。** vendored 目录 + `PROVENANCE.json` + 取回脚本对使用者更省事，也便于离线复现。
- **不重写上游模型。** iTransformer/DLinear/Autoformer 的实现保持逐字不变，只在外面包一层适配器。这样论文的结论仍然归因于上游架构本身。
- **不假装解决了因果问题。** SHAP 报告里明确写入"SHAP 衡量的是模型对输入的依赖程度，不是变量对负荷的物理因果影响"（论文 S5.1 已正确指出这点）。
- **不改论文已经说明的结论。** 例如 S4.3 拒绝用常规/极端误差增长率衡量鲁棒性——`error_growth_ratio()` 仍然返回该比值，但带 `interpretable: False` 与理由。

---

## 6. 落地状态

| 项 | 状态 |
|---|---|
| 分层包与配置体系 | 完成 |
| 数据层（含 4 个已修复缺陷） | 完成，已用合成数据验证 |
| 模型层（上游适配 + 随机森林） | 完成，接口与训练已测 |
| 损失与指标层 | 完成，数值已对手算与 sklearn 复核 |
| XAI 层 | 完成，含完备性残差 |
| CLI 与实验编排 | 完成 |
| 测试 | **150 个**，全部使用合成数据（不含任何真实数据集） |
| 端到端冒烟 | 完成：合成 CSV → 构建 → 训练（iTransformer/DLinear/RandomForest）→ 消融 → SHAP |
| **在真实数据上复现论文数值** | **未做** —— 需要作者本地的 Elia/ERA5/Tetuan 数据 |
