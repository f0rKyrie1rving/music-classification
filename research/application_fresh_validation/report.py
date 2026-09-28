"""Build a plain-language report from the complete fixed application evaluation."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .evaluate import ROOT, LABELS, read, save, digest
from .pipeline import verify_frozen

NAMES = dict(zip(LABELS, ['电子', '流行', '氛围', '摇滚']))


def report(out, destination):
    verify_frozen(out)
    status = read(out/'evaluation_status.json')
    if status['state'] != 'complete':
        raise ValueError('Cannot report an incomplete evaluation')
    for name, expected in status['hashes'].items():
        if digest(out/name) != expected:
            raise ValueError('Evaluation result changed: '+name)
    summary, metrics = read(out/'summary.json'), read(out/'metrics.json')
    intervals = read(out/'intervals.json')
    public = read(out/'public_predictions.json')
    extraction = read(out/'extraction_status.json')
    selected = read(out/'selected_manifest.json')['tracks']
    by_id = {r['track_id']: r for r in selected}
    rows = public['rows']
    y = np.array([r['targets'] for r in rows])
    raw = np.array([r['raw_scores'] for r in rows])
    corrected = np.array([r['corrected_scores'] for r in rows])
    pred = np.array([r['selected']['f1'] for r in rows])
    coverage = summary['coverage']
    verdict = summary['assessment']['verdict']
    conclusions = {
        'go': '通过本轮预先规定的概率检验；可以进入新版应用的构建与安装测试。',
        'promising': '误差有所下降，但不确定性仍较大；暂时只能说有希望，不能宣布验证通过。',
        'inconclusive': '本轮证据不足以确认概率修正有效；不应把它宣传成已验证的提升。',
        'harmful': '观察到变差信号；不建议按当前修正方案发布，应保留结果并重新研究原因。',
        'incomplete_evidence': '成功获取的歌曲或艺人不足预定比例；本轮不能作通过结论。',
    }
    conclusion = conclusions[verdict]
    destination.mkdir(parents=True, exist_ok=True)
    result_dir = destination/'results'; result_dir.mkdir(exist_ok=True)
    for name in ['summary.json', 'metrics.json', 'intervals.json', 'public_predictions.json',
                 'predictions.npz', 'evaluation_status.json', 'extraction_status.json',
                 'download_status.json', 'features.json', 'runtime_control.json',
                 'selected_manifest.json', 'manifest.json', 'exclusions.json', 'data_audit.json',
                 'metadata_amendment.json', 'independent_audit.json', 'config.json', 'freeze.json']:
        shutil.copy2(out/name, result_dir/name)
    shutil.copy2(out/'protocol.md', destination/'PROTOCOL.md')
    # Diagnosis is descriptive and performed only after the fixed comparison.
    errors = []
    for j, label in enumerate(LABELS):
        for kind, mask in [('false_positive', pred[:, j] & (y[:, j] == 0)),
                           ('false_negative', ~pred[:, j] & (y[:, j] == 1))]:
            candidates = np.flatnonzero(mask)
            ordered = sorted(candidates, key=lambda i: (-float((corrected[i,j]-y[i,j])**2), rows[i]['track_id']))
            for i in ordered[:5]:
                r = rows[i]; source = by_id[r['track_id']]
                errors.append({'label':label, 'kind':kind, 'track_id':r['track_id'],
                    'artist_id':r['artist_id'], 'source_tags':r['source_tags'],
                    'raw_score':float(raw[i,j]), 'corrected_score':float(corrected[i,j]),
                    'source_target':int(y[i,j]), 'selected':bool(pred[i,j]),
                    'attribution_verbatim':source['attribution_verbatim'],
                    'license_code':source['license_code'],
                    'audio_file':source['audio_file'],
                    'scope':'post-evaluation highest squared-error examples among source-label disagreements; not human-adjudicated mistakes'})
    save(result_dir/'error_examples.json', errors)
    table = []
    old, new = metrics['raw'], metrics['weight_corrected']
    for label in LABELS:
        a, b = old['per_label'][label], new['per_label'][label]
        d = old['policies']['f1']['per_label'][label]
        table.append(f"| {NAMES[label]} | {a['positive_tracks']} | {a['brier']:.6f} | {b['brier']:.6f} | {d['false_positives']} | {d['false_negatives']} |")
    weak = max(LABELS, key=lambda label: old['policies']['f1']['per_label'][label]['false_positives'])
    missing = max(LABELS, key=lambda label: old['policies']['f1']['per_label'][label]['false_negatives'])
    rel = summary['relative_brier_reduction']
    change = f"相对下降 **{rel*100:.2f}%**" if rel is not None and rel >= 0 else f"相对上升 **{-rel*100:.2f}%**" if rel is not None else '原始误差为零，比例不可计算'
    ci = intervals['metrics']['brier']['ci95']
    llci = intervals['metrics']['log_loss']['ci95']
    policy = old['policies']['f1']
    text = f'''# 新歌验收：应用 v1.1

**结论：{conclusion}** 判定标记为 `{verdict}`。

## 这次实际做了什么

固定应用版本 `931c58b`，让旧版和新版分析同一批新歌曲，没有重新训练或调整参数。
从新的归档 08–11 找到 171 位合规、与历史记录不重叠的艺人，每位最多两首，
共选定 **{coverage['selected_tracks']} 首**。最后成功评估 **{coverage['observed_tracks']} 首、
{coverage['observed_artists']} 位艺人**，覆盖歌曲 {coverage['track_fraction']:.1%}、艺人 {coverage['artist_fraction']:.1%}。
下载失败 {len(extraction['download_failures'])} 首，特征提取失败 {len(extraction['extraction_failures'])} 首；全部记录，没有补换歌曲。

此前使用和考察过的 1,929 首歌、837 位艺人都被排除，包含上一轮的全部候选歌曲。
独立审计核对了本地记录与音频/特征缓存。最初计划 200 位艺人，元数据检查只有 171 位
符合条件，因此在预测前固定纳入全部 171 位；没有为了凑数扩大范围。

## 概率数字有没有更靠谱

| 检查项，越小越好 | 旧版 | 新版 |
| --- | ---: | ---: |
| 平均概率误差，Brier | {old['macro_brier']:.6f} | {new['macro_brier']:.6f} |
| 另一项概率误差，log loss | {old['macro_log_loss']:.6f} | {new['macro_log_loss']:.6f} |

平均 Brier 误差{change}。它衡量模型报出的概率与原始标签有多大偏差，
**不是识别准确率变化了同样的百分比**。

把同一艺人的歌作为一组，重复抽样 2,000 次，Brier“新版减旧版”的 95% 区间为
**[{ci[0]:.6f}, {ci[1]:.6f}]**；log loss 对应区间为
**[{llci[0]:.6f}, {llci[1]:.6f}]**。负数支持新版误差较小，跨零代表仍有方向不确定性。
模型和修正规则在整个检查中保持固定。这些区间是对艺人重采样的条件性估计，
不包括重新训练、标签错误或选方法带来的不确定性，也不是整个音乐世界的保证。

## 仍然容易错在哪里

| 风格 | 原始标签阳性歌曲 | 旧版概率误差 | 新版概率误差 | 误报数 | 漏报数 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

在默认策略下，误报最多的是**{NAMES[weak]}**，漏报最多的是**{NAMES[missing]}**。
“误报”指模型贴了该标签而数据源没有；“漏报”指数据源有该标签而模型没贴。
这些计数依据上传者的原始标签，标签可能遗漏，而且整首歌的标签不一定在前 30 秒可听见。
因此它们是需要检查的分歧，不能全部当作人工确认的错误。

**两种判断策略下，新旧版标签变化都是 {summary['interface_checks']['changed_tag_decisions']}。**
默认策略的精确率 {policy['micro_precision']:.1%}、召回率 {policy['micro_recall']:.1%}、
F1 {policy['micro_f1']:.3f} 均保持相同。这次修正改善的是报分方式，尚未修复上述识别分歧。
各风格的严重误报/漏报示例已完整保留在 [error_examples.json](results/error_examples.json)，
方便后续人工听辨；本轮没有用这些例子回头调模型。

## 是否值得发布，后面做什么

{conclusion}
本次只完成验证，没有推送 GitHub、修改应用模型或生成新的 Windows 安装包。
若继续发布，应先跑 Windows 构建、安装后分析和卸载检查，再提供安装包。
若继续改善识别能力，应在后续开发数据上处理标签质量和分类错误，另留新的验收集。

## 这份结论的边界

新歌曲来自同一个 MTG-Jamendo 数据源，不是另一个完全独立的数据集；抽样还受归档、
授权、艺人和片段长度限制。本轮“新”只相对于核查过的项目记录，不能排除编码器预训练、
艺人别名或未记录的外部接触。此次看过结果后，这批歌已被使用，不能再称作下一版的
未见测试。数据许可、逐曲归属与原始标签保留在 [选曲清单](results/selected_manifest.json)。
MTG-Jamendo 元数据源采用 CC BY-NC-SA 4.0；音频保留各自的许可证并仅在本地使用。

完整固定规则见 [PROTOCOL.md](PROTOCOL.md)，逐曲新旧数值见
[public_predictions.json](results/public_predictions.json)，全部重采样结果保留在
[predictions.npz](results/predictions.npz)。这份报告不把已有的权重修正方法宣传为新算法。
'''
    (destination/'SUMMARY_ZH.md').write_text(text, encoding='utf-8')
    print(destination/'SUMMARY_ZH.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    report(args.out.resolve(), args.destination.resolve())
