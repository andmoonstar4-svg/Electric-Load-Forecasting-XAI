# 数据来源、许可与溯源

原项目无法公开数据，因此论文的数值无法由他人重新推导。本文件给出**架构层面**的解决办法：把溯源变成产物，而不是一段散文。

---

## 1. 数据源登记表

代码里的机器可读版本在 `src/elxai/data/provenance.py` 的 `SOURCES`。

| key | 数据 | 发布方 | 许可 | 可否再分发 |
|---|---|---|---|---|
| `elia` | 比利时电网 Total Load | Elia Transmission Belgium | Elia 开放数据条款 / 电网数据免责声明 | **否** |
| `era5` | ERA5 单层小时再分析 | Copernicus C3S / ECMWF | Copernicus 许可（免费，需署名，不得暗示背书） | 需署名 |
| `tetuan` | Tetuan 市三区功耗 + 气象 | UCI / Kaggle | CC BY 4.0 | 需署名 |
| `ijeie33` | IEEE 33 节点配电网 | — | — | 未知（仅与 S7.3 调度方向相关） |

### 对 Elia 数据的具体处理

`redistribution="no"` 意味着：

- 原始 `ods001.csv` **不进入版本库**（`.gitignore` 已排除 `data/` 与 `*.csv`）；
- `scripts/fetch_third_party.py` 之类的工具**不会**下载它；
- `assert_redistributable(["elia"])` 会在打包/发布步骤直接抛 `PermissionError`：
  ```
  PermissionError: refusing to package data from source(s) that are not
  redistributable: ['elia']. Ship the fetch script and this provenance record instead.
  ```
- 仓库里只保留**获取方式 + 清洗脚本 + 校验和**，让别人用自己的凭据取同一份数据。

`elxai dataset --check-redistribution` 就是这个断言的 CLI 入口。

---

## 2. 每次运行写出的溯源记录

`runs/<run_id>/provenance.json` 包含：

```json
{
  "dataset": "belgium",
  "sources": [
    {
      "key": "elia",
      "path": "<你本地的路径>",
      "sha256": "…",
      "n_rows": 403176,
      "start": "2014-12-31 23:00:00+00:00",
      "end": "…"
    },
    { "key": "era5", "…": "…" }
  ],
  "source_declarations": { "elia": { "licence": "…", "redistribution": "no", "citation": "…" } },
  "built_dataset_hash": "…",
  "built_dataset_rows": 100994,
  "splits": {
    "train": { "start_index": 0, "end_index": 70695, "n_rows": 70695, "start": "…", "end": "…" },
    "val":   { "…": "…" },
    "test":  { "…": "…" }
  },
  "seed": 2021,
  "git_commit": "…",
  "git_dirty": false,
  "python": "3.11.15",
  "platform": "Windows-…"
}
```

要点：

- **`splits` 给出的是具体时间范围与行号区间**，而不只是比例。任何"这份结果对应哪段时间"的疑问都能直接回答。
- `git_dirty` 让"结果到底对应哪份代码"变得可判定。原仓库无法回答这个问题。
- 每个源文件都有 SHA-256，所以"是不是同一份数据"不再靠记忆。

配套的 `dataset_report.json` 记录数据构建过程的**每一个判决**：

```json
{
  "cleaning": {
    "load_rows": 403176,
    "weather_rows": 105192,
    "merged_rows": 100994,
    "dropped_forecast_columns": ["Most recent forecast", "Day-ahead 6PM forecast", "Week-ahead forecast"],
    "load_rows_dropped_by_tolerance": 0,
    "load_resampled_to_hourly": true,
    "join_strategy": "resample+exact",
    "weather_gap_hours": 1
  },
  "clipped_values": { "tcc": 0, "msl": 0 },
  "extreme": { "n_total": 100994, "n_events_only": 1234, "n_extreme_days": 321, "n_subset": 8689,
               "context_days": 7, "thresholds": { "t2m_max_c": 35.0, "tp_min_m": 0.005, "u10_min_ms": 15.0 } },
  "window_counts": { "train": 70695, "val": 9880, "test": 19760 },
  "windows_dropped_to_gaps": 0,
  "frame": { "n_rows": 100994, "inferred_freq": "1h", "n_gap_steps": 0 }
}
```

这张表就是**回答"论文的 100994 / 52417 / 8689 是怎么来的"的凭据**，而且能被重新计算。注意 `load_rows_dropped_by_tolerance` —— 若该值大于 0，说明原始预处理确实丢过观测（见 `docs/migration.md` §4.2）。

---

## 3. 论文里需要补的一段

论文的"数据来源"节只列了引用。建议补充如下说明（可直接用于修订稿）：

> 本文使用的比利时负荷数据来自 Elia 公开数据接口，该数据受 Elia 开放数据条款约束，**不在本文项目中再分发**；可复现代码通过 `src/elxai/data/cleaning.py` 中的清洗流程从原始导出文件构建建模数据集，原始文件的 SHA-256 与所覆盖的时间范围记录在每次实验的 `provenance.json` 中。ERA5 数据来自 Copernicus Climate Change Service，依 Copernicus 许可使用并在此致谢。Tetuan 数据来自 UCI 机器学习库，依 CC BY 4.0 使用。三者的许可与引用信息由 `src/elxai/data/provenance.py` 统一登记，并可通过 `elxai dataset --check-redistribution` 自动校验再分发条件。

（英文版见下一节。）

---

## 4. English text for the paper

> The Belgian load series is obtained from Elia open grid data and is subject to
> Elia's open-data terms; it is **not redistributed** with this project. The
> reproducibility code reconstructs the modelling dataset from a raw export via
> `src/elxai/data/cleaning.py`; the SHA-256 of each raw file and the exact date
> ranges used for training, validation and testing are recorded in the
> `provenance.json` of every run. ERA5 data are provided by the Copernicus
> Climate Change Service and used under the Copernicus licence; the Tetuan City
> dataset is provided by the UCI Machine Learning Repository under CC BY 4.0. All
> licences, citations and redistribution conditions are registered in
> `src/elxai/data/provenance.py` and can be checked with
> `elxai dataset --check-redistribution`.

---

## 5. 上游代码的许可

`third_party/Time-Series-Library/` 来自 [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)（MIT）。`third_party/PROVENANCE.json` 记录：

- 上游 URL 与许可；
- **实际 vendored 的文件清单**（16 个，而不是整包 120 个）；
- 来源 commit（若通过 `scripts/fetch_third_party.py --clone` 取回则自动写入）；
- 原项目中的本地改动说明。

`third_party/LTSF-Linear`（cure-lab/LTSF-Linear，DLinear 的另一来源）在原仓库中整包存在，但与 `Time-Series-Library` 重复。本仓库只使用后者，前者的位置保留为空占位并在 PROVENANCE 中说明，避免"同一份代码两份拷贝、改一处漏一处"。

---

## 6. 复现性检查清单

发布前逐项确认：

- [ ] `git status` 干净，或已知 `provenance.json` 中记录为 `git_dirty: true`
- [ ] 每个 run 目录含 `config.yaml` + `provenance.json` + `metrics.json`
- [ ] 论文表格中的每个数值都能在某个 `metrics.json` 里找到，或由 `elxai summarize` 重建
- [ ] 仓库内不含任何原始数据文件（`git ls-files | findstr /i "csv parquet xlsx"` 应为空）
- [ ] `elxai dataset --check-redistribution` 通过（在排除不可再分发源的前提下）
- [ ] 测试通过：`pytest -q`
- [ ] 所有 SHAP 图的 run 目录里有 `shap/shap_report.json`，且记录了采样数与完备性残差
