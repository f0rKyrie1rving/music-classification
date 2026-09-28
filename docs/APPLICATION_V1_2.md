# 应用 v1.2：已接入通过新歌验收的固定模型

2026年9月28日。当前源码桌面应用默认使用 **1.2.0**，模型就是上一轮通过验收的
`music-candidate-20260927-v1`。本轮完成应用接入，没有重新训练、调门槛或新增考试歌曲。

打开桌面程序后，直接选择音频并点击 **Analyze audio** 即可使用新版。
**Compare v1.1** 可以查看同一段音频的旧版结果；分数、门槛、是否选中和顶部标签列表
一起切换，不会把旧版分数与新版标签混在一起，也不会重复分析音频。

## 实际验证结果

| 检查 | 结果 |
|---|---|
| 整个 `tests/` 测试集 | 215项通过；研究环境和不含Torch的基础测试环境均通过 |
| 372首验收歌曲、250位艺人的缓存回归 | 两种策略共744次新应用入口调用，与原验收预测及标签一致 |
| 独立接入核验 | 24,576项检查通过；完整精度数组、显示分数、判断均一致 |
| 真实音频的命令行与桌面分析入口 | 都成功运行，模型标识、权重校验值、门槛和旧版对比通过核对 |
| 本机真实Tk窗口（macOS） | 示例按钮、分析按钮、切换旧版及切回新版通过，分析函数只调用一次 |
| 深色模式显示 | 已查看真实窗口，说明文字改为系统配色，避免深色背景下看不清 |
| 安装包目录布局 | 将打包清单中的文件复制到独立目录，从含空格的另一工作目录运行通过 |
| 冻结研究文件 | 接入前后校验通过，模型、旧版源码和旧验收记录保持不变 |
| Windows安装、运行及卸载 | 构建流程和验收脚本已更新；本轮未在Windows运行，因此尚未生成或发布新安装包 |

仓库自带示例在新版中输出 `pop, rock`，旧版输出 `rock`。这是固定音频的运行核对，
不能把示例标签当作人工认定的音乐类别。

| 示例标签 | 新版分数／门槛／选中 | v1.1分数／门槛／选中 |
|---|---|---|
| electronic | 0.0699／0.4000／否 | 0.0699／0.4000／否 |
| pop | 0.3219／0.2750／是 | 0.2483／0.2750／否 |
| ambient | 0.0272／0.2000／否 | 0.0078／0.1321／否 |
| rock | 0.6237／0.2750／是 | 0.6237／0.2750／是 |

显示值保留四位小数；判断仍使用未经四舍五入的分数和门槛。新版氛围分类器没有类别
权重，因此没有再次应用旧版的氛围概率修正。

## 怎样运行

在现有项目目录中，用已配置好的环境打开桌面窗口：

```sh
.venv-improve/bin/python desktop_app.py
```

命令行分析及旧版对比：

```sh
.venv-improve/bin/python application_predict.py data/previews/track_0207501_30s.wav --compare-baseline
```

也可以把音频路径替换成自己的文件。仍只读取前30秒，输入必须不少于30秒。
新环境的依赖安装方式见[项目README](../README.md)，Windows安装说明见[安装指南](WINDOWS_INSTALLER.md)。

## 接入时保留了哪些边界

新入口为 [`application_predict.py`](../application_predict.py)，版本与发布绑定记录由
[`application_release.py`](../application_release.py) 和
[`artifacts/application_release.json`](../artifacts/application_release.json) 管理。
后者记录了本次应用采用的候选文件、权重与已通过验收的证据校验值。文件缺失或被改变时，
应用会报告错误，不会悄悄换回旧模型。

候选目录中的六个文件保持原样。里面的历史字段 `candidate_not_fresh_validated`
记录的是候选刚构建时的状态；新的应用发布记录单独绑定后续通过的确认结果。
旧 `predict_app.py`、`app_version.txt` 中的1.1.0、旧权重和概率修正文件继续作为冻结对照。
**今后不要通过修改旧 `app_version.txt` 升级当前应用版本**；当前版本来自
`application_release.APP_VERSION`，并须与发布记录、安装器和构建检查一起更新。

PyInstaller打包定义已加入全部候选文件及新的应用入口。Windows工作流会核对版本标签，
分别验证便携目录和安装后的程序，确认实际使用的是这份候选、正确门槛和同一次音频的
v1.1对比结果。源码接入、本机检查通过不等于Windows安装包已发布。

上一轮[新歌验收](application_candidate_validation/SUMMARY_ZH.md)的结果仍为：流行、氛围
误报152→124，漏报59→61，总体F1为0.619910→0.637002，Brier误差降低2.72%。本轮缓存回归
验证这些行为在接入后没有改变，**不能另算一次独立效果验证**。原有标签、同数据源及
预训练接触不明等限制仍保留，尤其氛围误报仍较多。

检查记录：[独立接入核验](application_integration/verification.json)、
[真实音频CLI](application_integration/cli_sample.json)、
[桌面分析入口](application_integration/desktop_smoke.json)、
[本机窗口操作](application_integration/native_ui.json)。

候选权重SHA-256：`70ceca1ab53b9b8d968d6e1ca4664bc78e22b78e2c70def772d495cbbf208036`。

应用发布记录SHA-256：`f1729a461f17c0b87a7668f8e189e35e4f5c5d2f5928ae8d631fc1a97b7d4a00`。
