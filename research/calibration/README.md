# 音乐多标签概率校准：第一轮探索实验

已完成本地资产审计、历史预测诊断、五次艺人隔离划分、两种校准方法及主划分的
2,000 次艺人聚类 bootstrap。正式结果见 [英文报告](../../docs/calibration/REPORT.md)。
这是对历史开发池的探索性研究，不是全新数据上的确认性检验。

## 研究在做什么

原模型给每个标签输出一个概率。这里检查这些概率和观察到的标签是否匹配，并比较：

- identity：保留原始概率。
- sigmoid：每标签分别拟合 `sigmoid(a*z+b)`，可改变尺度和整体位置。
- binary temperature：每标签分别拟合 `sigmoid(z/T)`，只能改变尺度。

`z` 为重新训练的分类头输出的 logit。四个标签允许同时存在，因此不用 softmax。
分类头只见训练组，校准器只见校准组，评估组只用于算指标。三组按艺人隔离。
Brier 可以理解为概率与 0/1 标签的平均平方误差，越低越好；它衡量整体概率质量，
不能单凭下降就证明每个概率区间都校准好了。

## 文件与运行记录

- `protocol.md`、`config.json`：查看新实验结果前固定的规则。
- `audit_inputs.py`：特征 SHA-256、来源、ID、标签、数据角色与本地资产检查。
- `splits.py`：固定随机种子和仅按标签/艺人数平衡的划分规则。
- `calibrators.py`、`metrics.py`：二元校准、概率指标、成对聚类 bootstrap。
- `run_experiment.py`：分步冻结和执行；输出目录存在时拒绝覆盖。
- `report.py`：固定 5/10 等宽分箱的科学图表。
- `test_calibration.py`：数据泄漏、行序、指标定义及极端数值检查。
- `EXECUTION_NOTES.md`：首次精度检查失败及修复的完整说明。
- `../../outputs/calibration/20260926_v2/`：本次完整日志、源代码快照、参数和预测。
- `../../docs/calibration/results/`：可随代码分享的派生预测和汇总，不含音频或上游权重。

本轮在原缓存所在工作副本中使用独立研究目录，未切换分支、提交或推送。
原训练脚本、发布模型、安装包和历史实验结果未修改。

## 复现

从仓库根目录执行。已验证的解释器为 `.venv-improve/bin/python`，Python 3.13.15，
依赖版本见本目录 `requirements.txt`。没有全局升级或修改历史依赖锁。
该环境还含历史深度学习包，但本研究只导入 numpy/scipy/scikit-learn/matplotlib。
新机器可自行建虚拟环境并安装此独立 requirements；还需原开发特征及配套 JSON。
当前 runner 严格验证原缓存哈希，若使用新提取特征，需要另建有明确来源的协议版本。

```bash
.venv-improve/bin/python -m unittest research.calibration.test_calibration -v
.venv-improve/bin/python -m research.calibration.run_experiment freeze --out outputs/calibration/reproduction_01
.venv-improve/bin/python -m research.calibration.run_experiment stage-a --out outputs/calibration/reproduction_01
.venv-improve/bin/python -m research.calibration.run_experiment primary --out outputs/calibration/reproduction_01
.venv-improve/bin/python -m research.calibration.run_experiment repeats --out outputs/calibration/reproduction_01
.venv-improve/bin/python -m research.calibration.run_experiment bootstrap --out outputs/calibration/reproduction_01
```

固定顺序执行：审计通过后冻结全部划分，再做历史诊断、主实验、重复划分和区间。
源代码、输入、配置或依赖版本改变后，旧冻结目录会拒绝继续运行。创建新目录并记录
原因，不覆盖既有结果。严禁以改善幅度选择保留的种子或方法。

## 当前证据与下一步

sigmoid 在五次划分中的四次降低了平均 Brier，一次变差；改善主要来自 ambient。
温度缩放在五次划分中平均 Brier 都略升。报告同时保留标签级负面结果。
主划分区间只反映固定训练和校准结果下的评估样本不确定性，不包含重新训练的变化。

第一轮无需补数据。若继续研究“校准数据量”的影响，应在新协议中先规定嵌套的
艺人子样本与数据量梯度，再执行；本轮只研究约 20% 校准集，尚无学习曲线。
若需要确认性结论，应另建此前未使用且与历史全部使用艺人隔离的数据集，固定抽样总体。
新增多少数据应由效应大小、艺人数、标签稀疏程度和精度目标共同决定。

初查文献支持方法背景，但尚未完成系统综述或证明研究新颖性。可按可复现项目和探索
报告展示；不应描述为已发表论文或已获得独立验证。
