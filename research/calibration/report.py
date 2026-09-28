"""Static scientific plots for the historical audit and primary evaluation."""
import os
from pathlib import Path
from .common import ROOT, LABELS

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/calibration/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .metrics import bins


def plots(directory, y, probabilities, title):
    directory = Path(directory)
    for count in (5,10):
        fig, axes = plt.subplots(2,4,figsize=(15,7), layout='constrained')
        for j,label in enumerate(LABELS):
            axes[0,j].plot([0,1],[0,1],'--',color='0.6',lw=1)
            for method,p in probabilities.items():
                bs = bins(y[:,j],p[:,j],count); nonempty = [b for b in bs if b['n']]
                line, = axes[0,j].plot([b['mean_score'] for b in nonempty],
                    [b['positive_fraction'] for b in nonempty], 'o-',label=method,ms=4)
                axes[1,j].stairs([b['n'] for b in bs],np.linspace(0,1,count+1),
                                  color=line.get_color(),label=method,lw=1.7)
                if len(probabilities)==1:
                    for b in nonempty:
                        axes[0,j].annotate(str(b['n']),(b['mean_score'],b['positive_fraction']),
                                           xytext=(3,5),textcoords='offset points',fontsize=8)
            axes[0,j].set(title=label,xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel='Mean predicted probability')
            axes[1,j].set(xlim=(0,1),xlabel='Predicted probability')
            axes[0,j].grid(alpha=.2); axes[1,j].grid(axis='y',alpha=.2)
        axes[0,0].set_ylabel('Observed positive fraction'); axes[1,0].set_ylabel('Tracks per bin')
        axes[0,0].legend(loc='upper left',fontsize=8)
        fig.suptitle(f'{title}\n{count} fixed equal-width bins; counts in lower panels; no pointwise curve intervals',fontsize=13)
        fig.savefig(directory / f'reliability_{count}.png',dpi=160)
        fig.savefig(directory / f'reliability_{count}.pdf')
        plt.close(fig)
