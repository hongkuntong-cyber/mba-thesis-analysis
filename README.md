# 德国 Amazon SKU 需求聚类与轻量预测验证

本仓库是 MBA 论文实证工程，用于复现德国 Amazon SKU 的数据审计、V2 需求代理修正、Ward 聚类、K 值验证、Rolling-Origin 回测，以及 MA4_proxy、Naive、SES、ADIDA2 的轻量化预测比较。

研究以“严格、可验证、可复现”为原则。正式方法已冻结在 `protocol/`，任何协议调整均以带版本号的修订文件保留，不能根据结果反向改写清洗规则、候选特征、K 值范围或评价指标。

## 当前结论

- V4.0 已将“不同簇绑定不同模型池”改为“聚类只负责预测管理分层”；p×q、
  MA4_proxy、Naive、SES和ADIDA2在所有可评价SKU上统一比较。
- p×q在4/9/13周共同样本的周期总量WAPE为0.6749/0.6550/0.7035；各周期总量
  最优模型分别为SES/p×q/MA4_proxy。预冻结的p×q替代门槛在全体、两个簇及
  低信息层级均未通过，因此暂不替代MA4_proxy，只作为周期总量影子估计。
- 最新 V3.0 业务导向特征筛选枚举了固定包含 `ADI + CV2` 的 64 个组合；建议完整期间主聚类继续使用 `ADI + CV2 + nonzero_mean + acf1`，Ward `K=2`，主样本 N=197。
- V3.0 的第二个 Pareto 候选为 `ADI + CV2 + peak_ratio`，因重抽样、样本门槛和算法一致性更弱，仅保留为敏感性对照。
- `K=3` 未通过预定稳定性、Raw/V2 一致性和独立画像条件，因此 SBA/TSB 未进入正式模型池。
- 首个预测起点前冻结方案：`ADI + CV2 + std_sales`，Ward `K=2`，N=130。
- ADIDA2 已作为正式核心模型检验，但未通过生产路由门槛；建议对 10 个重复改善候选做影子运行。
- 最终留出中，统一 MA4_proxy 的销量加权 WAPE 为 0.7760，轻量分层为 0.7745；差异很小，当前不支持全面部署复杂路由。

V3.0 聚类证据见 [`reports/business_feature_review_v3.md`](reports/business_feature_review_v3.md)；既有预测、路由和最终留出证据见 [`reports/final_report.md`](reports/final_report.md)。V3.0 使用的是已经查看过的历史数据，属于回顾性方法开发，不应表述为新增样本确认。

## 复现环境

- Python 3.12
- 全局随机种子：42
- 输入文件：`01_原始数据/德国Amazon_SKU周度数据_原始合并版_未清洗 (1).xlsx`
- 期望 SHA256：`ece008d42c9dd6ea11e4a0c8f6d828c2cb037df1d837ea233587f115491786b6`

建议使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

## 一条命令运行

正式端到端运行：

```bash
python -m src.pipeline --config config/analysis.yaml
```

若正式结果已存在，只重建报告和图形：

```bash
python -m src.pipeline --config config/analysis.yaml --reuse-existing
```

运行测试：

```bash
pytest -q
```

探索性比较簇专属轻量算法池（新增 SBA，不覆盖 V1 正式结论）：

```bash
python -m src.forecast_pool_v2 --config config/analysis.yaml
python -m src.validate_forecast_pool_v2
```

其方法在运行前冻结于 `protocol/forecast_pool_v2_exploratory_protocol.md`。由于 V1 的最终窗口已经查看，V2 输出只能作为探索性证据，不能重新称为独立确认性留出。

运行 V3.0 业务导向候选池、后选择敏感性和图形：

```bash
python -m src.business_feature_pipeline --config config/analysis_business_features.yaml
python -m src.business_feature_diagnostics --config config/analysis_business_features.yaml
python -m src.business_feature_plots --output-root outputs/business_features_v3
```

运行 V4.0 统一p×q三周期历史验证与独立校验：

```bash
python -m src.pxq_validation_v4 --config config/pxq_validation_v4.yaml
python -m src.validate_pxq_v4 --config config/pxq_validation_v4.yaml
```

V4.0协议为`protocol/amendment_v4.0_unified_pxq_validation.md`，报告为
`reports/pxq_validation_v4.md`。4/9/13周只是28/63/91天代理；所有结果属于
回顾性方法开发，不是新增数据确认。

V3.0 正式运行前冻结的协议、字典和证据注册表分别为：

- `protocol/amendment_v3.0_business_aware_feature_screening.md`
- `protocol/feature_dictionary_v3.md`
- `protocol/feature_evidence_registry_v3.csv`

端到端正式运行会执行 500 次 80% 无放回子样本稳定性计算，耗时明显高于测试。原始文件只读，正式输出写入 `outputs/` 和 `reports/`。

## 工程结构

```text
config/                  冻结配置
protocol/                研究协议、特征字典和版本化修订
src/                     审计、清洗、聚类、预测、评价和流水线代码
tests/                   边界、恒等式、泄漏和复现测试
outputs/audit/           审计与 V2 清洗结果
outputs/clustering/      特征网格、K 验证、画像和稳健性结果
outputs/forecast/        回测、路由、ADIDA2 和最终留出结果
reports/                 最终报告、实施矩阵与图表索引
```

## 关键研究口径

- V2 仅修正满足固定概率条件的内部连续零值区间，结果称为“V2 需求代理值”，不是恢复的真实销量。
- 主样本要求正销量周数不少于 5；严格稳健性样本不少于 10。
- 正式聚类固定使用 Yeo–Johnson、标准化和 Ward，验证 `K=2–6`。
- 预测采用 6 个互不重叠的 8 周窗口；前 5 个用于冻结路由，第 6 个只做一次最终留出测试。
- 正式模型池固定为 MA4_proxy、Naive、SES、ADIDA2；Zero 仅用于预测价值与规则管理基准。
- 测试数据不得参与清洗、特征计算、变换拟合、聚类或模型参数估计。

## 主要交付文件

- 研究协议：[`protocol/research_protocol.md`](protocol/research_protocol.md)
- 特征字典：[`protocol/feature_dictionary.md`](protocol/feature_dictionary.md)
- 分析配置：[`config/analysis.yaml`](config/analysis.yaml)
- 最终报告：[`reports/final_report.md`](reports/final_report.md)
- 实施矩阵：[`reports/implementation_matrix.md`](reports/implementation_matrix.md)
- 完整聚类网格：[`outputs/clustering/feature_k_grid_500.csv`](outputs/clustering/feature_k_grid_500.csv)
- Rolling-Origin 明细：[`outputs/forecast/rolling_origin_predictions.csv`](outputs/forecast/rolling_origin_predictions.csv)
- 冻结路由：[`outputs/forecast/frozen_routes_before_holdout.csv`](outputs/forecast/frozen_routes_before_holdout.csv)

## 数据与解释限制

当前数据只包含销量，没有库存、缺货、采购提前期、MOQ、服务水平、持有成本、采购价或毛利。因此结果只支持预测误差和实施复杂度判断，不能外推为利润、库存成本或缺货成本改善。2024–2025 与 2026 的 SKU 宇宙也存在明显断裂，相关排除与可评价样本数均在报告中单独披露。
