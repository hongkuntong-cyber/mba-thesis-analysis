"""Build and execute the bounded, reader-facing V4.4 experiment notebook."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from tempfile import TemporaryDirectory
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parent.parent


def build():
    md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
    cells=[
md('''# 固定120个SKU：简单概率与MA4公平比较（V4.4）

## tl;dr

本轮已补算，数据截至 **2026-08-02**。研究性质是**补充探索性分析**。

- 主候选为“周期发生频率 × 正需求周期均量”，与早期“周发生率 × 正需求周均量”区分。
- 全体固定120个SKU保留。三个周期均有MA4可评价案例；周期MASE有效覆盖分别为120、118、96个SKU。
- 主候选周期MASE分别为 **1.842、2.803、4.460**；同样本MA4为 **1.661、2.606、3.766**。
  28和91天的探索性配对区间指向误差更大，63天区间跨零。未证明替代MA4。
- 概率能描述需求差异，但其总体排序AUC低于MA4数量排序；MA4=0子样本的概率AUC约0.5，尚无可靠增量证据。
- 当期动态活跃层误差与MA4接近，信息不足层误差很大；这属于适用范围的描述性证据，**不是分层带来因果改善**。
- 同一批历史周期上P×Q恰等于历史周期均值；已逐条验证，不能包装成新数量算法。

9725个既有预测值核对后复用，只补充5134个既有方法在新案例上的预测值；另计算已冻结的均值机制对照。
没有重跑聚类特征搜索、K值搜索或500次稳定性实验。
'''),
md('''## Context & Methods

研究路径：**聚类 → 分层管理 → 所有SKU采用统一简单预测**。

### Key Assumptions

1. 完整期V3名单固定为120个相对活跃SKU与77个低频对照。名单属于事后完整期分组。
2. 每个SKU以自身最后完整观测周为尾部，向前最多六个不重叠的H周测试窗口；4/9/13周即28/63/91天代理。
3. 每一配对使用同一个SKU、起点、未来目标、起点前V2修正和相同训练缩放分母。未来只用原始销量。
4. MA4固定最近4周；周期频率候选固定最多52周。窗口长度不同是模型定义差异，额外用相同历史均值对照检验机制。
5. 历史不足、不存在正需求周期、周期MASE尺度为零均保留记录，不加epsilon，不伪造条件数量。
6. 先逐案例配对，再在SKU内跨起点平均，最后对SKU等权汇总。置信区间为SKU层1000次bootstrap（seed=42）。
7. 配对区间不包含共同时间冲击、多重比较和事后样本选择的不确定性；不能称为独立确认或非劣效检验。

|方法|公式/定义|定位|
|---|---|---|
|BlockFrequencyMean|P=s/n，Q=正周期总量均值，E=P×Q|本轮主候选|
|MA4_proxy|最近4周V2销量均值×H|企业数量基准|
|MatchedHistoryMean|同一批H周块总量均值|机制对照；有正块时等于主候选|
|LaplaceMean|(s+1)/(n+2)×Q|已有V4.3平滑对照，未重新选参数|
|LegacyPXQ|最近H周正周比例×连续训练期正周均值×H|早期周发生率方案|
|Naive|最近一周销量×H|朴素对照|
|FullHistoryMean|全部连续训练期完整H周块均值|预设历史窗口对照|

周期MASE：测试周期总量绝对误差 ÷ 最近52周内相邻H周历史块总量平均绝对差。
至少两个块且分母>0才计算。它不是旧的周路径MASE，两者不能混表。
RMSSE补充均值预测的平方损失，避免只用偏向中位数的绝对损失评价期望量。

来源：[MASE与预测目标](https://otexts.com/fpp3/accuracy.html)。
'''),
md('''## Data

源码与历史基础：[GitHub项目](https://github.com/hongkuntong-cyber/mba-thesis-analysis)。
基线提交 `3ff25df`；本轮协议在结果生成前提交于 `0bf11a8`。

原始Excel：`01_原始数据/德国Amazon_SKU周度数据_原始合并版_未清洗 (1).xlsx`。
SHA256：`ece008d42c9dd6ea11e4a0c8f6d828c2cb037df1d837ea233587f115491786b6`。
242个原始SKU，197个完整期聚类主样本；本轮主对象120个均保留。

V4.3完整结果包的58项SHA256全部匹配。GitHub旧V4预测CSV存在损坏，本轮不读取该损坏文件。
'''),
md('### 1. 读取参数与验证状态'),
code('''from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "outputs/pxq_fair_v4_4").exists():
    ROOT = ROOT.parent
OUT = ROOT / "outputs/pxq_fair_v4_4"
assert OUT.exists(), "请在仓库或notebooks目录运行"
validation = json.loads((OUT / "validation_summary.json").read_text())
assert validation["failed"] == 0
for name, expected in json.loads((OUT / "manifest_sha256.json").read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_columns", 15)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(f"独立验证：{validation['passed']}项通过；{validation['validated_cases']}个案例；{validation['validated_forecast_rows']}条预测。")
'''),
md('### 2. 样本身份、覆盖与一个可核对的案例'),
code('''ids = pd.read_csv(OUT/"cohort_membership.csv")
display(ids.groupby(["full_cluster", "calendar_group"]).size().rename("SKU数").to_frame())
coverage = pd.read_csv(OUT/"coverage_summary.csv")
display(coverage.loc[coverage.cohort.eq("active120"), ["horizon_weeks","planned_skus","planned_cases","ma4_cases","pure_quantity_skus","pure_quantity_cases","mase_skus","mase_cases"]])
components = pd.read_csv(OUT/"components.csv")
example = components.loc[components.full_cluster.eq(1) & components.s.gt(0)].sort_values(["horizon_weeks","origin","sku"]).head(1)
display(example[["sku","origin","horizon_weeks","n","s","probability","conditional_mean","ma4_forecast","actual_sum"]])
'''),
md('''完整期活跃120个包含66个跨2024—2026、30个仅2024—2025、24个仅2026。
90天代理并不是丢掉24个SKU：120个均有MA4案例、120个均有至少一个概率案例；
115个有可计算的条件数量预测，96个有可计算周期MASE。不可计算案例仍在逐SKU覆盖表中。
'''),
md('''## Results

### 3. 主比较：相同案例下的周期MASE

以下均值按SKU等权。差值为主候选减MA4，正值表示主候选误差更大。
表中的区间仅为探索性证据；不同周期可用SKU与日历期不同，不比较跨周期绝对分数高低来推断期限效应。
'''),
code('''comparisons = pd.read_csv(OUT/"quantity_comparisons.csv")
main = comparisons.loc[comparisons.cohort.eq("active120") & comparisons.method.eq("BlockFrequencyMean") & comparisons.metric.eq("mase_52")]
display(main[["horizon_weeks","n_skus","n_pairs","mean_model","mean_ma4","delta","ci_low","ci_high","pair_win_share"]])
'''),
md('### 4. 全部有数量预测的配对案例：高估与低估'),
code('''ae = comparisons.loc[comparisons.cohort.eq("active120") & comparisons.method.eq("BlockFrequencyMean") & comparisons.metric.eq("ae")]
display(ae[["horizon_weeks","n_skus","n_pairs","mean_model","mean_ma4","bias_model","bias_ma4","under_units_model","under_units_ma4","over_units_model","over_units_ma4"]])
'''),
md('''这里是预测误差，不是实际缺货或实际积压。63天的原始AE点估计略低，但探索性区间跨零；
91天候选减少高估量，同时增加低估量。没有库存、成本、提前期数据，不能称这种取舍为企业收益改善。
'''),
md('### 5. 同历史机制对照和既有概率方案'),
code('''all_methods = comparisons.loc[comparisons.cohort.eq("active120") & comparisons.metric.eq("mase_52")]
display(all_methods[["horizon_weeks","method","n_pairs","mean_model","mean_ma4","delta"]])
q = pd.read_csv(OUT/"quantity_predictions.csv")
wide = q.pivot(index=["horizon_weeks","sku","origin"], columns="method", values="forecast_sum")
valid = wide[["BlockFrequencyMean","MatchedHistoryMean"]].dropna()
assert np.allclose(valid.BlockFrequencyMean, valid.MatchedHistoryMean)
print("同历史P×Q与历史均量恒等式：全部有效案例通过。")
'''),
md('''这说明分解增加了“会不会发生、发生时多少”的解释，而没有凭空产生新的数量信息。
早期LegacyPXQ保留为独立对照，不与本轮周期概率混称为一个方法。
'''),
md('### 6. RMSSE与尺度敏感性'),
code('''squared = comparisons.loc[comparisons.cohort.eq("active120") & comparisons.method.eq("BlockFrequencyMean") & comparisons.metric.eq("scaled_squared_error")]
display(squared[["horizon_weeks","n_pairs","rmsse_model","rmsse_ma4"]])
scale_check = comparisons.loc[comparisons.cohort.eq("active120") & comparisons.method.eq("BlockFrequencyMean") & comparisons.metric.eq("mase_full")]
display(scale_check[["horizon_weeks","mean_model","mean_ma4","delta"]])
'''),
md('''RMSSE也未改善，因此本次数量结果不能单纯归因于MASE不适合评价均值。
改用全训练历史的缩放分母后，主候选相对MA4的平均差值方向不变。
'''),
md('### 7. 概率可靠性与相对于MA4的排序信息'),
code('''prob = pd.read_csv(OUT/"probability_summary.csv")
display(prob.loc[prob.cohort.eq("active120") & prob.scope.eq("all"), ["horizon_weeks","method","n","mean_probability","actual_rate","brier","ece","auc","ma4_score_auc"]])
display(prob.loc[prob.cohort.eq("active120") & prob.scope.eq("ma4_zero") & prob.method.eq("BlockFrequency"), ["horizon_weeks","n","n_skus","mean_probability","actual_rate","auc"]])
'''),
md('''概率主表及AUC使用SKU等权。MA4仅作为数量排序分数计算AUC，不作为概率参与Brier。
主候选全样本AUC约0.793、0.686、0.707；同样本MA4数量排序约0.858、0.823、0.833。
这些是描述性对比，并未对AUC差做确认性推断。MA4=0子样本的主候选AUC约0.528、0.474、0.468，
尚不能证明概率可靠地识别MA4遗漏的需求恢复。

旧Laplace平滑方案改善Brier，但长期周期低估仍明显；不能以28天改善推断三周期都可靠。
'''),
md('### 8. 概率档：历史高概率是否对应实际需求'),
code('''bins = pd.read_csv(OUT/"reliability_bins.csv")
display(bins.loc[bins.cohort.eq("active120") & bins.method.eq("BlockFrequency"), ["horizon_weeks","bin","n","n_skus","mean_probability","actual_rate"]])
'''),
md('''分箱采用案例等权，与上面的SKU等权主概率表不同；bin=0对应[0,20%)，bin=4对应[80%,100%]。
28天最高概率档实际发生率约89.8%，但这不能单独证明对MA4的增量；低概率档也仍有需求发生。
分箱只用于诊断，不作为采购阈值。
'''),
md('### 9. 分层适用性：完整120名单中的当期动态画像'),
code('''layers = pd.read_csv(OUT/"quantity_by_layer.csv")
display(layers.loc[layers.full_cluster.eq(1) & layers.dimension.eq("dynamic_cluster") & layers.metric.eq("mase_52"), ["horizon_weeks","layer","n","n_skus","mean_model","mean_ma4","delta","ci_low","ci_high"]])
'''),
md('''当期layer=1为活跃层，layer=2为稀疏层，layer=0表示当期聚类信息不足。
同一个SKU可在不同起点处于不同层，因此各层SKU数不能相加当成总SKU数。
活跃层的数值接近不等于已通过非劣效检验，低信息层的结果也不能作为事后删样本理由。
'''),
md('### 10. 原日历安排的结果保留为独立对照'),
code('''old = pd.read_csv(OUT/"old_calendar_quantity_comparisons.csv")
display(old.loc[old.cohort.eq("active120") & old.method.eq("BlockFrequencyMean") & old.metric.eq("mase_52"), ["horizon_weeks","n_pairs","n_skus","mean_model","mean_ma4","delta"]])
'''),
md('''## Takeaways

1. **数量预测尚未证明优于MA4。** 周期概率分解与同历史均值相等，不能以形式上的分解宣称算法精度创新。
2. **概率信息具有解释性，可靠性和增量仍需证据。** 高概率档确实更常发生需求，但当前排序未超过MA4，
   需求恢复子样本也没有稳定区分能力；不能宣称已改善企业决策。
3. **分层揭示了适用范围差异。** 当期活跃层与MA4接近、信息不足层误差较大，为重点复核历史不足对象提供研究线索；
   这不是随机试验，也没有验证复核规则或库存收益。
4. **本轮没有按结果改方法。** 未平滑主候选、既有平滑和旧周分解均完整报告；不把表现更好的某一层或方法事后升级为主结论。

### 复现与验证

在仓库根目录：
```bash
python -m pip install -r requirements-notebook.txt
python -m src.pxq_fair_v4_4
python -m src.validate_pxq_fair_v4_4
python -m unittest tests.test_pxq_fair_v4_4 -v
```

已有准备结果时只重建汇总；新工作副本移走V4.4输出目录才会从原始数据重建。
独立校验包括原始未来目标、全部公式、配对集合、SKU等权分母、bootstrap区间、覆盖及输出指纹。
''')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    path=ROOT/'notebooks/fair_ma4_fixed120_v4_4.ipynb'
    path.parent.mkdir(exist_ok=True)
    # These cells use plain Python only. Execute them in order without opening
    # sockets: this environment rejects both TCP and Unix Jupyter kernel binds.
    shell=InteractiveShell.instance()
    shell.history_manager.enabled=False
    for index,cell in enumerate((c for c in nb.cells if c.cell_type=='code'),1):
        with capture_output() as captured:
            result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Cell {index} failed') from (result.error_before_exec or result.error_in_exec)
        cell.execution_count=index
        cell.outputs=[]
        if captured.stdout:
            cell.outputs.append(nbf.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbf.v4.new_output('stream',name='stderr',text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbf.v4.new_output('display_data',data=output.data,metadata=output.metadata))
    nb.metadata['execution_engine']='Sequential in-process IPython; Jupyter socket binding is unavailable in this runtime.'
    nb.cells.append(md('执行说明：本环境不允许Jupyter内核绑定通信套接字；本笔记本的全部普通Python单元已在同一个进程内IPython会话中按顺序实际执行，输出由执行捕获，未手工伪造。常规Jupyter内核运行需在本地环境完成。'))
    nbf.validate(nb)
    nbf.write(nb,path)
    print(f'Executed {sum(x.cell_type=="code" for x in nb.cells)} cells: {path}')


if __name__=='__main__': build()
