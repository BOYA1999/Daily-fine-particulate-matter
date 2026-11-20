import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_statistical_figure import (
    ROOT, CITIES, NAMES, ROWS, BLUE, TEAL, OCHRE, GRAY, GRID, INK,
    city_rows, horizontal, legend, paired_points, panel_layout, read, save, set_style,
)

COLORS = [BLUE, TEAL, OCHRE]
CITY_MARKS = ['o', 's', '^']
BUDGETS = [.25, .5, 1, 2]
DATA = ROOT / 'data'


def trend(ax, x, values, color, marker='o', **kwargs):
    ax.plot(x, values, color=color, marker=marker, linewidth=1.25, markersize=3.6, **kwargs)
    ax.grid(axis='y', color=GRID, linewidth=.63)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)


def workflow():
    fig = plt.figure(figsize=(7.08, 3.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    def box(x, y, w, head, body, fill):
        h = .205
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.013',
                                    edgecolor='#AEBFC5', facecolor=fill, linewidth=.75))
        ax.text(x+.014, y+h-.034, head, fontsize=9.5, weight='bold', va='top')
        ax.text(x+.014, y+h-.093, body, fontsize=7.7, linespacing=1.22, va='top')

    def arrow(x, y, xx, yy):
        ax.annotate('', (xx, yy), (x, y), arrowprops=dict(arrowstyle='-|>', lw=.95, color=GRAY))

    pale, accent = '#F4F7F7', '#DFECEF'
    box(.031,.748,.436,'Source observations', 'EPA AQS daily PM₂.₅ at local conditions\n2017–2025 · 36 archived query responses', pale)
    box(.531,.748,.436,'Monitoring panel', 'Valid daily data · one value per site-day\nRoster defined in 2017–2020 · 59 sites', accent)
    arrow(.481,.854,.516,.854)
    arrow(.75,.734,.75,.681)
    ax.plot([.173,.827],[.675,.675],color=GRAY,linewidth=.85)
    for x in [.173,.50,.827]:
        arrow(x,.675,x,.634)
    box(.031,.417,.284,'2017–2020 · Fit', 'Station–month means and covariance\nEarlier-year residual predictions', pale)
    box(.358,.417,.284,'2021–2022 · Calibrate', 'Use masked historical observations\nSet lower and upper quantiles', pale)
    box(.685,.417,.284,'2023–2025 · Evaluate', 'Hide target and match donor counts\nCompare three removal patterns', accent)
    arrow(.325,.523,.347,.523)
    arrow(.651,.523,.675,.523)
    arrow(.827,.403,.827,.350)
    box(.031,.070,.436,'Within-network comparisons', 'Paired errors, interval scores and review workload\nDate-block uncertainty within each network', pale)
    box(.531,.070,.436,'Chicago transfer experiment', 'Hold Chicago point predictions fixed\nVary the sources for scales and quantiles', pale)
    ax.plot([.248,.827],[.344,.344],color=GRAY,linewidth=.85)
    arrow(.248,.344,.248,.287)
    arrow(.75,.344,.75,.287)
    save(fig, 2)


def availability():
    fig, axes = panel_layout(['Dates with at least nine reporting sites', 'Observations retained by the nine-site rule',
                              *[f'{name} roster' for name in NAMES]], (.110, .410, .735))
    years = read('year_support')
    for city, color, marker in zip(CITIES, COLORS, CITY_MARKS):
        rows = years[years.metro.eq(city)]
        trend(axes[0], rows.year, rows.eligible_dates, color, marker)
        test = rows[rows.year.ge(2023)]
        trend(axes[1], test.year, 100*test.eligible_site_days/test.reporting_site_days, color, marker)
    axes[0].set(xlabel='Calendar year', ylabel='Eligible dates', ylim=(0, 379),
                xticks=[2017,2019,2021,2023,2025], yticks=[0,100,200,300])
    for cut in [2020.5, 2022.5]:
        axes[0].axvline(cut, linestyle='--', color=GRID, linewidth=.9)
    axes[1].set(xlabel='Evaluation year', ylabel='Retained site-days (%)',
                ylim=(70, 103), yticks=[70,80,90,100], xticks=[2023,2024,2025])
    legend(fig,.023,.902,[(n,m,c,c) for n,m,c in zip(NAMES,CITY_MARKS,COLORS)])
    cmap = LinearSegmentedColormap.from_list('reporting', ['#F2F4F2', '#ADCFCB', '#225F6A'])
    for city, ax in zip(CITIES, axes[2:]):
        frame = pd.read_csv(DATA / f'{city}_panel.csv', index_col='date', parse_dates=['date'])
        rates = frame.notna().groupby(frame.index.year).mean().T
        ax.pcolormesh(rates.to_numpy(), cmap=cmap, vmin=0, vmax=1, edgecolors='white', linewidth=.3)
        m = len(rates)
        ax.set(ylim=(m,0), xticks=[.5,4.5,8.5], xticklabels=['2017','2021','2025'],
               yticks=[.5,m-.5], yticklabels=['1',str(m)], xlabel='Calendar year')
        ax.tick_params(axis='both',length=0)
        ax.spines['bottom'].set_visible(False)
    axes[2].set_ylabel('Historical roster site')
    fig.text(.025,.399,'Share of calendar days with a valid observation', fontsize=7.2, color=GRAY)
    fig.legend(handles=[Patch(facecolor=cmap(s),label=f'{100*s:.0f}%') for s in [0.,.5,1.]],
               loc='upper left', bbox_to_anchor=(.52,.419), ncol=3, fontsize=7.2,
               frameon=False, borderaxespad=0, handlelength=.95, handletextpad=.4, columnspacing=.9)
    save(fig,4)


def error_sensitivity():
    fig, axes = panel_layout(['Nearest versus shared-group removal', 'Sensitivity to evaluation weights',
                              *[f'{name} stations' for name in NAMES]], (.110,.410,.735))
    combined = read('review_geometry_summary').query('design == "footprint"')
    extra = combined.query('pattern == "nearest" and reference == "footprint"')
    paired_points(axes[0], extra.rename(columns={'method':'series'}),
                  [('covariance','o',BLUE),('fixed_idw','^',OCHRE)], 'delta_mae')
    bounds = extra[extra.method.isin(['covariance','fixed_idw'])]
    left = min(-.01, bounds.lo14.min()-.025)
    right = bounds.hi14.max()+.025
    ticks = [x for x in MaxNLocator(nbins=5).tick_values(left,right) if left <= x <= right]
    horizontal(axes[0],ticks,(left,right),'MAE change (µg/m³)')
    axes[0].axvline(0,color=GRAY,linestyle='--',linewidth=.75)
    legend(fig,.023,.902,[('Covariance','o',BLUE,BLUE),('Fixed IDW','^',OCHRE,OCHRE)])

    cov = city_rows(combined.query('method == "covariance" and pattern == "footprint" and reference == "random"'))
    symbols = [('delta_mae','o',BLUE),('date_balanced','s',TEAL),('site_balanced','^',OCHRE)]
    for (key, marker, color), offset in zip(symbols, [-.16,0,.16]):
        axes[1].scatter(cov[key], ROWS+offset, marker=marker, color=color, s=23, zorder=3)
    xmax = .38
    horizontal(axes[1],[0,.1,.2,.3],(-.01,xmax),'Grouped − random MAE (µg/m³)')
    axes[1].axvline(0,color=GRAY,linestyle='--',linewidth=.75)
    legend(fig,.518,.902,[('Equal site-day','o',BLUE,BLUE),('Equal date','s',TEAL,TEAL),
                           ('Equal station','^',OCHRE,OCHRE)])
    profiles = read('station_pointwise_intervals').query('method == "covariance"')
    lo_bound = min(-.05,profiles.lo14.min()-.03)
    hi_bound = profiles.hi14.max()+.06
    for city, ax in zip(CITIES, axes[2:]):
        station = profiles[profiles.metro.eq(city)].sort_values('delta')
        scores = station.delta.to_numpy()
        rank = np.arange(1, len(scores)+1)
        ax.axvline(0,color=GRAY,linestyle='--',linewidth=.75)
        ax.hlines(rank,station.lo14,station.hi14,color='#ACBFC5',linewidth=.7)
        ax.scatter(scores,rank,color=np.where(scores >= 0,BLUE,OCHRE),s=15,zorder=3)
        ax.set(xlim=(lo_bound,hi_bound), ylim=(0,len(scores)+1), yticks=[1,len(scores)],
               xlabel='Grouped − random MAE (µg/m³)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis='x',color=GRID,linewidth=.63)
        ax.tick_params(axis='y',length=0)
    axes[2].set_ylabel('Rank within network')
    fig.text(.025,.399,'Station means for the covariance predictor', fontsize=7.2, color=GRAY)
    save(fig,6)


def interval_diagnostics():
    fig, axes = panel_layout(['Coverage by observed concentration', 'Interval width by observed concentration',
                              *[f'{name} by year' for name in NAMES]], (.110,.410,.735))
    profiles = read('interval_profiles')
    bands = profiles.query('slice_type == "observed_band" and scale == "combined"')
    tests = ['le25','25to35','gt35']
    labels = ['≤25', '>25 to 35', '>35']
    for i, (city,color,marker) in enumerate(zip(CITIES,COLORS,CITY_MARKS)):
        rows = bands.loc[bands.metro.eq(city)].set_index('stratum').loc[tests]
        y = ROWS + (i-1)*.15
        axes[0].scatter(100*rows.coverage,y,color=color,s=23,marker=marker,zorder=3)
        axes[1].scatter(rows.width,y,color=color,s=23,marker=marker,zorder=3)
    horizontal(axes[0], [0,25,50,75,100],(-2,103),'Coverage (%)',labels=labels)
    axes[0].axvline(90,color=GRAY,linestyle='--',linewidth=.75)
    max_width = np.ceil(bands.width.max())+1
    ticks = [x for x in MaxNLocator(nbins=5).tick_values(0,max_width) if x <= max_width]
    horizontal(axes[1],ticks,(-.5,max_width),'Mean interval width (µg/m³)',labels=labels)
    legend(fig,.023,.902,[(n,m,c,c) for n,m,c in zip(NAMES,CITY_MARKS,COLORS)])
    fig.text(.52,.889,'Observed concentration bands in µg/m³',fontsize=7.2,color=GRAY)
    annual = profiles.query('slice_type == "test_year"')
    low = np.floor(annual.coverage.min()*100)-2
    high = np.ceil(annual.coverage.max()*100)+2
    for city,ax in zip(CITIES,axes[2:]):
        for scale,mark,color,linestyle in [('constant','o',GRAY,'--'),('combined','D',BLUE,'-')]:
            a = annual[annual.metro.eq(city)&annual.scale.eq(scale)].sort_values('stratum')
            trend(ax,a.stratum.astype(int),100*a.coverage,color,mark,linestyle=linestyle,
                  markerfacecolor='white' if scale == 'constant' else color)
        ax.axhline(90,color=GRAY,linestyle=':',linewidth=.75)
        ax.set(xticks=[2023,2024,2025],xlabel='Evaluation year',ylim=(low,high))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
    axes[2].set_ylabel('Coverage (%)')
    legend(fig,.023,.416,[('Constant width','o','white',GRAY),('Combined scale','D',BLUE,BLUE)])
    save(fig,9)


def threshold_sensitivity():
    fig, axes = panel_layout(['Observed high site-days', 'High site-days missed by the point estimate',
                              *[f'{name} review budgets' for name in NAMES]], (.110,.410,.735))
    queue = read('review_queue').query('scale in ["constant", "combined"]')
    reference = read('review_interval_summary').query(
        'center == "covariance" and calibration == "local" and test_design == "local" and scale == "constant"')
    misses = queue.query('scale == "constant" and budget_fraction == .01')
    for ax, table, metric, xlabel in [
        (axes[0],reference,'high','Observed high site-days'),
        (axes[1],misses,'point_miss','High site-days missed by point estimate'),
    ]:
        for t, color, symbol, offset in [(25, BLUE,'o',-.13),(35,OCHRE,'^',.13)]:
            ax.scatter(city_rows(table[table.threshold.eq(t)])[metric],ROWS+offset,
                       color=color,marker=symbol,s=23,zorder=3)
        upper = table[metric].max()*1.13
        ticks = [x for x in MaxNLocator(nbins=5,integer=True).tick_values(0,upper) if x <= upper]
        horizontal(ax,ticks,(-.02*upper,upper),xlabel)
    labels = [('25 µg/m³','o',BLUE,BLUE),('35 µg/m³','^',OCHRE,OCHRE)]
    legend(fig,.023,.902,labels)
    legend(fig,.023,.416,labels)
    pairing = queue.pivot(index=['metro','threshold','budget_fraction','reviewed'],
                          columns='scale',values='confirmed_high')
    gains = (pairing.combined-pairing.constant).rename('gain').reset_index()
    assert gains.gain.notna().all()
    ylo, yhi = min(-1, gains.gain.min())-.45, max(1,gains.gain.max())+.5
    for city, ax in zip(CITIES,axes[2:]):
        ax.axhline(0,color=GRAY,linestyle=':',linewidth=.8)
        for t,color,mark,style in [(25,BLUE,'o','-'),(35,OCHRE,'^','--')]:
            pairs = gains[gains.metro.eq(city)&gains.threshold.eq(t)].sort_values('budget_fraction')
            trend(ax,100*pairs.budget_fraction,pairs.gain,color,mark,linestyle=style)
        ax.set_xscale('log',base=2)
        ax.set(xticks=BUDGETS,xticklabels=['0.25','0.5','1','2'],xlim=(.217,2.3),ylim=(ylo,yhi),
               xlabel='Cases reviewed (%)')
        ax.set_xticks([],minor=True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
    axes[2].set_ylabel('Additional high cases referred')
    save(fig,12)


if __name__ == '__main__':
    set_style()
    workflow()
    availability()
    error_sensitivity()
    interval_diagnostics()
    threshold_sensitivity()
