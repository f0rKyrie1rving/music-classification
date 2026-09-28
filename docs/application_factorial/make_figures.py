"""Render the prespecified cells and all learning-curve seeds without refitting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
study = ROOT/'outputs/application_factorial/20260927_v1'
summary = json.loads((study/'summary.json').read_text())
config = json.loads((study/'config.json').read_text())
font = Path('/System/Library/Fonts/STHeiti Medium.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 11, 'axes.titlesize': 15, 'figure.facecolor': 'white'})
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.3))
for recipe, color, label in [('baseline', '#73777e', '旧分类器'), ('new', '#176caa', '新分类器')]:
    curves = np.asarray([[summary['cells'][f'f{fraction}_{seed}'][recipe+'_original']['pooled']['macro_brier']
                          for fraction in (40, 70)] + [summary['cells']['full'][recipe+'_original']['pooled']['macro_brier']]
                         for seed in config['sampling_seeds']])
    for curve in curves:
        left.plot([40, 70, 100], curve, color=color, alpha=.3, linewidth=1)
    average = curves.mean(0)
    left.plot([40, 70, 100], average, color=color, marker='o', label=label, linewidth=2.5)
    left.annotate(f'{average[-1]:.5f}', (100, average[-1]), xytext=(-3, 10),
                  textcoords='offset points', ha='right', color=color)
left.set(title='增加训练素材：概率误差下降', xlabel='每个来源中参与训练的艺人比例',
         ylabel='四类平均 Brier 误差（越低越好）', xticks=[40, 70, 100], xticklabels=['40%', '70%', '100%'])
left.legend(frameon=False); left.grid(axis='y', alpha=.17)
cells = ['baseline_original', 'new_original', 'baseline_tuned', 'new_tuned']
labels = ['旧方法', '只改\n分类器', '只调\n门槛', '两项\n一起改']
for offset, metric, color, label in [(-.18, 'fp', '#277fba', '误报'), (.18, 'fn', '#c88a38', '漏报')]:
    values = [sum(summary['cells']['full'][cell]['pooled']['per_label'][l][metric]
                  for l in ('pop', 'ambient')) for cell in cells]
    bars = right.bar(np.arange(4)+offset, values, width=.34, color=color, label=label)
    right.bar_label(bars, padding=4, fontsize=11)
right.set(title='配套调整分类器和门槛，取舍更好', ylabel='流行＋氛围的标签次数',
          xticks=np.arange(4), xticklabels=labels, ylim=(0, 1080))
right.legend(frameon=False); right.grid(axis='y', alpha=.17); right.set_axisbelow(True)
fig.suptitle('1,915 首已有歌曲的开发检查', fontsize=18, y=.99)
fig.text(.05, .022, '左图细线为全部 3 个固定抽样种子，粗线为均值；100% 共用一组结果。均在相同检查歌曲上比较。', fontsize=10, color='#45474a')
fig.text(.05, -.014, '来源标签作为参考；本轮不是新歌确认，也未替换应用。', fontsize=10, color='#45474a')
fig.tight_layout(rect=(0, .05, 1, .92))
fig.savefig(OUT/'results_overview.png', dpi=160, bbox_inches='tight')
plt.close(fig)
print(OUT/'results_overview.png')
