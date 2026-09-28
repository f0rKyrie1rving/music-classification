# 第三轮：解释校准失败的一个数据划分

已经完成。先看[中文解释](../../docs/calibration_diagnostics/SUMMARY_ZH.md)，
详细数字、图表与限制见[英文报告](../../docs/calibration_diagnostics/REPORT.md)。

## 做了什么

重点检查上一轮已知变差的划分 20260927，同时保留五个划分作对照。
只读取原有 logits、标签和校准参数，没有重新训练或修改原模型。

- 对比校准组与评估组的标签比例、平均分数和分正负标签的分数分布。
- 精确分解每个标签对总体 Brier 变化的贡献，核对各项加总。
- 固定五个原始分数区间，比较每个区间的观察标签频率，保留样本数。
- 汇总每位评估艺人的误差贡献，并逐一计算“移除这一位后”的结果。
  这只是敏感性检查，正式结果没有删除任何艺人。

结论是：摇滚和流行音乐的调整在评估组上增加了损失，ambient 的收益几乎消失。
去掉任意单个评估艺人后，这个划分仍然变差。数据构成与分数行为存在差别，
但不能仅据此认定某一种分布漂移就是唯一原因。

## 复现

在仓库根目录、已验证的 `.venv-improve` 环境运行。需要上一轮完整输出
`outputs/calibration/20260926_v2/`。使用尚不存在的输出目录：

```bash
.venv-improve/bin/python -m unittest research.calibration_diagnostics.test_diagnostics -v
.venv-improve/bin/python -m research.calibration_diagnostics.run freeze --out outputs/calibration_diagnostics/reproduction_01
.venv-improve/bin/python -m research.calibration_diagnostics.run run --out outputs/calibration_diagnostics/reproduction_01
.venv-improve/bin/python -m research.calibration_diagnostics.verify_results --out outputs/calibration_diagnostics/reproduction_01
.venv-improve/bin/python -m research.calibration_diagnostics.report --out outputs/calibration_diagnostics/reproduction_01 --destination outputs/calibration_diagnostics/reproduction_report_01
```

本次输出在 `outputs/calibration_diagnostics/20260926_v1/`。
文件哈希、源代码快照和依赖版本保存在 freeze/source 中；全部统计表在 results 中。
`verify_results.py` 独立调用 scikit-learn 核对指标，直接重算每次艺人删除结果，
并核对所有分解项。4 项新增测试通过，原始预测重建误差为零。
测试是单独执行的：验证器不会代替 unittest 运行测试，复现时必须执行上面的第一条命令。

这是一轮事后诊断，因为先看到了失败结果才选择这个划分。协议用于约束诊断范围，
不冒充前瞻性预注册。各表格不能用来宣称因果关系或直接挑选更好的部署策略。
报告的解释段落和中文摘要是在数值计算完成后整理的，不由报告脚本自动生成。

## 可继续研究的方向

下一轮可测试“保守校准”：只使用校准组内部的艺人分组验证，判断调整是否值得；
证据不足时保留原分数或减小调整。所有选择只能在校准组内完成，原评估组不能
参与挑选参数。由于这一路线受到已有结果启发，沿用当前评估组的结论仍然是
探索性的，真正确认仍需要新的独立歌曲和艺人。
