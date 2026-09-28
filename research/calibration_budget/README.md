# 第二轮：校准数据量实验

本轮已经运行完成。先看[中文结果说明](../../docs/calibration_budget/SUMMARY_ZH.md)，
细节见[英文报告](../../docs/calibration_budget/REPORT.md)。

## 这次研究什么

保持上一轮的分类模型与评估歌曲不变，只改变用于校准的歌曲数量。
按艺人抽取，保留同一艺人的所有歌曲。24、47、71 位艺人各抽样 30 次，
94 位艺人的全部校准池只计算一次。小样本嵌套在大样本中，便于成对比较。
五个原始划分均保留，方法仍为 sigmoid 和 binary temperature，与不校准比较。

规则在运行前保存在 `protocol.md` 和 `config.json`。没有重抽不利样本，也没有
利用评估指标挑选校准子集。评估集上一轮已经看过，所以本轮是探索性延伸。

## 复现命令

在仓库根目录执行，复用已验证的 `.venv-improve` 环境。需要上一轮完整输出
`outputs/calibration/20260926_v2/`。当前源代码不修改上一轮冻结文件。
选择一个尚不存在的输出目录和报告目录，按顺序运行：

```bash
.venv-improve/bin/python -m unittest research.calibration_budget.test_budget -v
.venv-improve/bin/python -m research.calibration_budget.run freeze --out outputs/calibration_budget/reproduction_01
.venv-improve/bin/python -m research.calibration_budget.run fit --out outputs/calibration_budget/reproduction_01
.venv-improve/bin/python -m research.calibration_budget.run score --out outputs/calibration_budget/reproduction_01
.venv-improve/bin/python -m research.calibration_budget.verify_results --out outputs/calibration_budget/reproduction_01
.venv-improve/bin/python -m research.calibration_budget.report --out outputs/calibration_budget/reproduction_01 --destination outputs/calibration_budget/reproduction_report_01
```

`freeze` 先保存所有抽样 IDs、艺人、标签计数和文件哈希。`fit` 完成全部校准拟合，
不会读取评估标签；全部拟合完成后，`score` 才计算新指标。
`verify_results` 从保存的参数重建预测，并以 scikit-learn 独立核对指标。
全量校准端点必须复现上一轮保存的预测。

报告程序生成表格和图形；中文摘要和英文报告的解释段落是在查看结果后人工组织的
文字，不参与实验计算，也不会被报告程序自动生成。使用不同路径复现可保留原报告。

## 实际输出

本轮完整记录在 `outputs/calibration_budget/20260926_v1/`：

- `plans.json`：455 组校准子集的歌曲、艺人和正负例计数。
- `fits/`：所有校准参数、失败状态及完整精度预测。
- `scores/`：逐次、逐标签指标，以及不同数据量之间的配对差值。
- `verification.json`：数值重建与交叉检查。
- `source/`：运行时源代码快照。

共计划 910 次“子集 × 方法”比较，其中 908 次有完整的四标签结果。
有一个小子集缺少 ambient 正例，导致两种方法的该标签无法拟合；两条记录均保留。
另有一个 sigmoid/rock 拟合达到截距上界，仍保留其结果。
4 项新增测试通过。所有成功预测都能从参数精确重建，指标核对最大差异约为 3.33e-16。

图中的阴影只是重复抽样结果的 10%–90% 范围，不是置信区间；全量校准只有一次拟合。
不能把这轮得到的“改善次数”理解成面对新音乐时的成功概率。
原有研究代码、特征、分类模型与发布结果均保持不变。
