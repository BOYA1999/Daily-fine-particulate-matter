from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_statistical_figure import (
    CITIES, NAMES, ROWS, BLUE, TEAL, OCHRE, INK, GRAY, GRID,
    city_rows, horizontal, legend, panel_layout, read, save, set_style,
)
from plot_expanded_figures import BUDGETS, COLORS, CITY_MARKS, trend


def geometry():
    titles = ['Matching the closest retained station', 'Sensitivity to group size',
              *[f'{name} covariance fits' for name in NAMES]]
    fig, axes = panel_layout(titles, (.123, .419, .743))
    compare = city_rows(read('distance_conditioned').query('reference == "all"'))
    axes[0].scatter(compare.ordinary_delta, ROWS-.13, marker='s', color=OCHRE, s=25, zorder=3)
    m = compare.conditioned_delta
    axes[0].errorbar(m, ROWS+.13, xerr=[m-compare.lo14,compare.hi14-m], fmt='o',
                     color=BLUE, markersize=4, linewidth=.9, capsize=2)
    horizontal(axes[0], [0,.1,.2,.3], (-.012,.345), 'Shared group − reference MAE (µg/m³)')
    axes[0].axvline(0, color=GRAY, linewidth=.7, linestyle='--')
    legend(fig,.023,.902,[('Matched count','s',OCHRE,OCHRE), ('Count and closest station','o',BLUE,BLUE)])

    choices = read('group_strength').query('size in [3,6,9]')
    for city, color, mark in zip(CITIES, COLORS, CITY_MARKS):
        s = choices[choices.metro.eq(city)].sort_values('size')
        axes[1].errorbar(100*s.fraction,s.delta,yerr=[s.delta-s.lo14,s.hi14-s.delta],fmt=mark+'-',
                         color=color,linewidth=.95,capsize=2,markersize=3.7)
    axes[1].set(xlabel='Mean fraction of available donors removed (%)',
                ylabel='MAE increase (µg/m³)', xlim=(6,52),ylim=(0,.61))
    axes[1].grid(axis='y',color=GRID,linewidth=.65)
    axes[1].tick_params(axis='y',length=0)
    legend(fig,.518,.902,[(n,m,c,c) for n,m,c in zip(NAMES,CITY_MARKS,COLORS)])

    variants = read('covariance_stability')
    fit_order = ['full','early','late','overlap120','floor1e3','shrink01','shrink30']
    descriptions = ['Full fit','2017–18','2019–20','Overlap ≥120','Floor 10⁻³','Shrink 1%','Shrink 30%']
    for ax, city in zip(axes[2:], CITIES):
        row = variants[variants.metro.eq(city)].set_index('fit').loc[fit_order]
        y = np.arange(len(row))
        ax.errorbar(row.delta,y,xerr=[row.delta-row.lo14,row.hi14-row.delta],
                    fmt='o',color=BLUE,markersize=3,linewidth=.8,capsize=1.8)
        ax.axvline(0,color=GRAY,linestyle='--',linewidth=.65)
        ax.set(ylim=(6.55,-.55),xlim=(-.014,.405),xticks=[0,.1,.2,.3,.4],
               xlabel='MAE increase (µg/m³)')
        ax.set_yticks(y, descriptions if city == 'LA' else [])
        ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',color=GRID,linewidth=.6)
    fig.text(.025,.400,'Original group sizes and matched uniform reference',fontsize=7.2,color=GRAY)
    save(fig,7)


def screening():
    fig, axes = panel_layout(['Decisions for observed highs', 'False high flags and review workload',
                              *[f'{name} review budgets' for name in NAMES]], (.110,.410,.735))
    out = read('screening_confusion').query('geometry == "footprint" and threshold == 35')
    fields = [('high_low','#A85D44','Classified low'),('high_flag','#397C76','Flagged high'),
              ('high_review','#AFC7D4','Referred')]
    for method, offset in [('constant',-.16),('combined',.16)]:
        rows = city_rows(out[out.scale.eq(method)])
        start = np.zeros(len(CITIES))
        for key, color, _ in fields:
            val = 100*rows[key]/rows.high
            axes[0].barh(ROWS+offset,val,left=start,height=.27,color=color,zorder=3)
            start += val.to_numpy()
    horizontal(axes[0], [0,25,50,75,100],(-1,103),'Observed high site-days (%)')
    fig.legend(handles=[Patch(facecolor=color,label=label) for _,color,label in fields],
               loc='upper left',bbox_to_anchor=(.023,.902),borderaxespad=0,frameon=False,ncols=3,
               fontsize=7.2,handlelength=.9,handletextpad=.4,columnspacing=1.)

    for city, name, color in zip(CITIES,NAMES,COLORS):
        rows = out[out.metro.eq(city)].set_index('scale').loc[['constant','combined']]
        x,y = rows.review_count.to_numpy(), rows.nonhigh_flag.to_numpy()
        axes[1].plot(x,y,color=color,linewidth=1.25)
        axes[1].scatter(x[0],y[0],s=25,facecolor='white',edgecolor=color,zorder=3)
        axes[1].scatter(x[1],y[1],s=25,color=color,zorder=3)
        axes[1].annotate(name,(x[1],y[1]),xytext=(3,6),textcoords='offset points',color=color,fontsize=7.2)
    axes[1].set(xlabel='Site-days referred for review',ylabel='False high flags',
                xlim=(0,295),ylim=(-1.65,14),yticks=[0,4,8,12])
    axes[1].grid(axis='y',color=GRID,linewidth=.65)
    axes[1].tick_params(axis='y',length=0)
    legend(fig,.518,.902,[('Constant width','o','white',GRAY),('Combined scale','o',GRAY,GRAY)])

    cases = read('screening_budget_summary').query('geometry=="footprint" and threshold==35')
    yhi = max(1, cases.recovered.max()*1.1)
    for city, ax in zip(CITIES,axes[2:]):
        for scale, mark, color, style in [('constant','o',GRAY,'--'),('combined','D',BLUE,'-')]:
            rows = cases[cases.metro.eq(city)&cases.scale.eq(scale)].sort_values('fraction')
            trend(ax,rows.fraction*100,rows.recovered,color,mark,linestyle=style,
                  markerfacecolor='white' if scale=='constant' else color)
        ax.set_xscale('log',base=2)
        ax.set(xlim=(.217,2.3),ylim=(0,yhi),xticks=BUDGETS,xticklabels=['0.25','0.5','1','2'],
               xlabel='Candidates reviewed (%)')
        ax.set_xticks([],minor=True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
    axes[2].set_ylabel('High cases recovered')
    legend(fig,.023,.416,[('Constant width','o','white',GRAY),('Combined scale','D',BLUE,BLUE)])
    fig.text(.518,.399,'Means across 64 group schedules',fontsize=7.2,color=GRAY)
    save(fig,10)


if __name__ == '__main__':
    set_style()
    geometry()
    screening()
