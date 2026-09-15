# 迁移指南：从原仓库到 elxai

本文件说明原仓库的每个文件去了哪里，以及论文的每个产物现在由什么命令生成。

---

## 1. 目录映射

| 原路径 | 现在 | 说明 |
|---|---|---|
| `DLinear-main/`（整包） | **删除** | 与 `Time-Series-Library-main/` 重复；DLinear 从后者取 |
| `Time-Series-Library-main/`（整包，120 文件） | `third_party/Time-Series-Library/`（16 文件） | 只保留 4 个模型 + 它们 import 的层；见 `third_party/PROVENANCE.json` |
| `Time-Series-Library-main/exp/exp_long_term_forecasting.py` | `src/elxai/training/trainer.py` + `src/elxai/models/torch_adapter.py` | 训练循环拆成"编排"与"模型适配"两层 |
| └ 其中的 `shap_analysis()`（L283–461） | `src/elxai/xai/shap_explainer.py` | 完全重写，见下 |
| `Time-Series-Library-main/data_provider/data_loader.py` | `src/elxai/data/{cleaning,windows,datasets}.py` | 拆成清洗 / 窗口化 / 编排 |
| `Time-Series-Library-main/run.py`（argparse） | `src/elxai/config.py` + `configs/*.yaml` + `src/elxai/cli.py` | 配置从命令行移入 YAML，可组合、可校验、可指纹 |
| `新建 文本文档.py`（散落在 Downloads） | `src/elxai/data/cleaning.py` + `labels.py` | 数据构建成为包的一部分，有测试 |
| 随机森林（框架外、无脚本） | `src/elxai/models/forest.py` | 实现同一 `Forecaster` 接口，可参与同一评价与同一归因 |
| `DLinear-main/pics/`（4.5 MB 宣传图） | **删除** | 与项目无关的上游素材 |
| `**/__pycache__/*.pyc` | **不再跟踪** | 并入 `.gitignore` |
| `report/*.pdf` | `report/*.pdf` | 保留 |
| `README.md` | 重写 | 见根目录 |

---

## 2. SHAP：从 `shap_analysis()` 到 `elxai.xai`

原函数 179 行，其中约 68 行是死代码（第 394–461 行重复定义了 `model_predict` 和第二段 SHAP 块，永不执行）。下表逐项对照。

| 原实现 | 现在 | 为什么 |
|---|---|---|
| `background_x_np = batch_x[:5]` | `xai.n_background`（默认 100），取自 `data["train"]` | 5 个样本估计 `E[f]` 精度极低；且原来取自**测试集**，基线含测试期信息 |
| `nsamples=100`（对 1176 维） | `xai.nsamples: auto` → `4 × n_features`，下限 2000 | 对 1176 维合作博弈采样 100 次，完备性不成立 |
| `test_x_np = batch_x[5:8]` | `xai.n_eval`（默认 50） | 3 个样本的归因方差过大 |
| `reshape(-1)` 展平后 `mean(axis=(0,1))` | `xai.resolution: variable \| variable_time` | 在时间轴直接平均隐含 product kernel 假设，逐变量得分不保证可加 |
| `outputs[:, 0, 0]` | `xai.target_steps: [0] \| "all" \| [...]` | 原来只解释未来第 1 小时，图注却暗示整个 24 小时 |
| `except Exception: print(...)` | 异常向上抛出；另写入**完备性残差** | 静默失败会产出"看起来正常但缺失"的图 |
| `folder = './shap_results/' + model + '_' + setting` | `runs/<run_id>/shap/` | `setting` 不含 loss，图 5-1/5-2 会互相覆盖 |
| 无 | `model_normalises_windows` 标志 + 残差 | iTransformer/DLinear 在 `forward` 内做 instance 标准化，扰动输入会同时改变窗口统计量 |

### 等价调用

原来：

```bash
python run.py --task_name long_term_forecast --model Autoformer --data custom \
  --root_path ./data --data_path elia.csv --seq_len 168 --pred_len 24 \
  --do_shap --shap_background 100
```

现在：

```bash
elxai xai \
  --config configs/data/belgium.yaml \
  --config configs/split.yaml \
  --config configs/model/autoformer.yaml \
  --config configs/experiment/xai.yaml \
  --set data.extreme.enabled=true \
  --set xai.n_background=100 \
  --set xai.resolution=variable \
  --set xai.target_steps=all
```

产物落在 `runs/<run_id>/`：`shap/shap_report.json`、`shap/shap_values.npz`、`figures/shap_importance_*.png`。

---

## 3. 损失消融：表 6-1 现在可复现

原来三次运行写进同一个 `./results/<setting>/`，因为 `setting` 不含 `--loss`。

```bash
elxai ablation \
  --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/model/autoformer.yaml --config configs/experiment/loss_ablation.yaml \
  --models Autoformer,iTransformer --losses mae,mse,exp \
  --out results/table6-1.csv
```

每次运行有独立目录（`fingerprint` 覆盖整份配置），`metrics.json` 里记录 `loss` / `loss_beta` / `loss_alpha` / `best_epoch`。

**注意指数损失的两个参数**：论文正文写 `2^(2|e|)−1` 且"取 β = 2"，但代码是 `EMEX(pred, true, X=4)`，即 `2^(4|e|)−1`。配置里两个都写明了：

```yaml
training:
  loss: exp
  exp_loss_beta: 2.0    # 底数 β
  exp_loss_alpha: 4.0   # 指数系数，对齐产生已发布表格的那份代码
```

若要复现论文正文的 `2^(2|e|)` 形式，设 `exp_loss_alpha: 1.0`（因为 `β^(α·e)` 在 `β=2, α=1` 时即 `2^e`）。

---

## 4. 数据管线：修掉的三个 bug

### 4.1 时间戳校验失效（最严重）

原 `_clean_time_index` 检查的是**原始列**的 `isna()`。原始列是字符串，`isna()` 恒为 False，所以不可解析的时间戳：
1. 通过校验；
2. 在后续解析中变成 `NaT`；
3. `NaT` 排序在**最前**；
4. **所有窗口和切分边界整体错位一行**，且没有任何报错。

现在检查的是解析后的序列（`cleaning.py`），并有测试 `test_clean_time_index_rejects_unparseable`。

### 4.2 15 分钟负荷与小时天气的对齐

原做法 `merge_asof(direction='nearest', tolerance='30min')`：当负荷时间戳与整点有偏移、且落在容差外时，读数被静默丢弃。现在按**实测时间间隔**选策略：

- 亚小时负荷 → 重采样为小时均值，再按整点精确连接（15 分钟功率读数取小时均值即该小时电量，量纲正确）；
- 整点负荷 → 用 `merge_asof` 容忍时间戳抖动。

同时把"旧做法会丢多少行"作为诊断量写进 `dataset_report.json` 的 `load_rows_dropped_by_tolerance`。

### 4.3 极端天气阈值

原脚本：`tp > 0.01`、`u10 > 10`、`msl < 100000`（多了一个论文没有的低气压条件）。
论文："小时降水量超过 5??/h、风速超过 15?/?"（单位是 OCR 残缺）。

现在全部是配置项，且**单位显式**：

```yaml
extreme:
  t2m_min_c: 0.0
  t2m_max_c: 35.0
  tp_min_m: 0.005   # 5 mm/h；ERA5 tp 单位是米
  tp_unit: m
  u10_min_ms: 15.0  # 对齐论文的 15
  mode: events_only # 论文的 8689 行构造
  context_days: 7
```

`dataset_report.json` 的 `extreme` 段记录实际命中数（`n_events_only` / `n_extreme_days` / `n_subset`）与所用阈值，使论文的 8689 这个数字可核对。

---

## 5. 无法自动完成的部分

以下需要作者本地数据，本次重构**没有**做：

| 项 | 原因 |
|---|---|
| 在真实数据上复现表 3-1 / 3-2 / 4-1 / 6-1 | 需要 Elia / ERA5 / Tetuan 原始数据；Elia 许可不允许再分发（见 `docs/provenance.md`） |
| 核对"数据集 1 共 100994 条、数据集 2 共 52417 条、极端集 8689 条" | 同上；现在这些数字会由 `dataset_report.json` 自动给出，可直接对照 |
| 用真实数据重跑图 5-1～5-4 | 同上 |

### 建议的核对顺序

```bash
export ELXAI_DATA_DIR="<你的数据目录>"

# 1) 先确认数据规模与论文一致
elxai dataset --config configs/data/belgium.yaml --config configs/split.yaml \
  --out runs/dataset1_check
# 读 runs/dataset1_check/dataset_report.json:
#   range.n_rows            对照 100994
#   cleaning.load_rows_dropped_by_tolerance  旧做法丢了多少（若 >0，说明存在 D1 bug）
#   frame.n_gap_steps       时间网格缺口数

# 2) 极端子集规模
elxai dataset --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/experiment/extreme.yaml --out runs/extreme_check
# 读 extreme.n_subset 对照 8689

# 3) 常规天气模型对比（表 3-1）
elxai compare --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/model/itransformer.yaml --config configs/experiment/normal.yaml \
  --models iTransformer,DLinear,Autoformer,RandomForest --out results/table3-1.csv

# 4) 损失消融（表 6-1）
elxai ablation --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/model/autoformer.yaml --config configs/experiment/loss_ablation.yaml \
  --models Autoformer,iTransformer --losses mae,mse,exp --out results/table6-1.csv

# 5) SHAP（图 5-1～5-4）
elxai xai --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/model/autoformer.yaml --config configs/experiment/xai.yaml \
  --set data.extreme.enabled=true --set training.loss=exp
elxai xai ... --config configs/model/itransformer.yaml --set training.loss=mse
```

### 数值可能变化，这是预期的

与原实现相比，以下改动**会**让指标变化，且都是有理由的：

1. **purge**：训练块末尾的窗口不再把验证期观测当作预测目标。原实现没有 purge，所以原数字略微乐观。
2. **对齐策略**：亚小时负荷不再丢行（若旧做法确实丢了行），训练数据变多。
3. **极端子集的时间范围**：现在是"极端日 + 之前 7 整天"，且跨缺口窗口被丢弃；原先是否如此取决于脚本细节。
4. **标准化口径**：数据集级 scaler 只 fit 训练集（原先也是，但值得复核），叠加模型内部 instance 标准化的情况现在被记录。

若需要**逐字**复现原数字以便对照，可用：
- `--set split.purge=0`（关闭 purge，并会记 WARNING）
- `--set data.merge_tolerance=30min`（恢复旧对齐的诊断，但策略仍按实测间距选择）

---

## 6. 快速验证

```bash
pip install -e ".[torch,xai,dev]"
pytest -q                       # 147 个测试，全部使用合成数据

# 用合成数据跑通全链路（不需要真实数据集）
python - <<'PY'
import numpy as np, pandas as pd
rng = np.random.default_rng(3); n = 24*400
q = pd.date_range('2019-01-01', periods=n*4, freq='15min')
load = 10000 + 1500*np.sin(2*np.pi*np.arange(n*4)/96) + rng.normal(0,150,n*4)
pd.DataFrame({'Datetime': q.tz_localize('UTC'), 'Total Load': load,
              'Most recent forecast': load*0.98}).to_csv('elia.csv', index=False, sep=';')
t = pd.date_range('2019-01-01', periods=n, freq='1h')
pd.DataFrame({'valid_time': t, 'u10': np.clip(rng.gamma(2,2,n),0,None),
              't2m': 283+rng.normal(0,2,n), 'msl': 101000+rng.normal(0,500,n),
              'ssrd': np.clip(rng.normal(5e5,2e5,n),0,None), 'tcc': rng.random(n),
              'tp': np.clip(rng.exponential(0.001,n),0,None)}).to_csv('era5.csv', index=False)
PY

elxai compare --config configs/data/belgium.yaml --config configs/split.yaml \
  --config configs/model/itransformer.yaml --config configs/experiment/smoke.yaml \
  --set data.load_csv=elia.csv --set data.weather_csv=era5.csv \
  --models iTransformer,DLinear,RandomForest
```
