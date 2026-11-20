from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / 'results', ROOT / 'manuscript' / 'figures'
CITIES = ['LA', 'NY', 'CH']
NAMES = ['South Coast', 'New York', 'Chicago']
INK, GRID, BLUE = '#253746', '#DCE4E7', '#225C7A'
TEAL, OCHRE, GRAY = '#428077', '#A16138', '#64727B'
ROWS = np.arange(len(CITIES))


def read(name):
    return pd.read_csv(RESULTS / f'{name}.csv')


def city_rows(frame):
    return frame.set_index('metro', verify_integrity=True).loc[CITIES]


def horizontal(ax, ticks, bounds, xlabel, labels=NAMES):
    ax.set(xlim=bounds, ylim=(2.55, -0.55), xticks=ticks, xlabel=xlabel)
    ax.set_yticks(ROWS, labels)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.grid(axis='x', color=GRID, linewidth=.63)
    ax.set_axisbelow(True)


def legend(fig, x, y, entries):
    handles = [Line2D([], [], linestyle='none', marker=mark, markerfacecolor=color,
                      markeredgecolor=edge, markersize=4.8, label=name)
               for name, mark, color, edge in entries]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(x, y),
               borderaxespad=0, frameon=False, ncols=len(handles), handlelength=.9,
               handletextpad=.42, columnspacing=1.0, fontsize=7.2)


def spatial_support(fig, ax):
    data = city_rows(read('review_spatial_support'))
    ax.hlines(ROWS, data.median_random_km, data.median_nearest_km,
              color='#BBCAD0', linewidth=1.65, zorder=2)
    for col, mark, fill, edge in [
        ('random', 'o', 'white', GRAY), ('grouped', 'o', BLUE, BLUE),
        ('nearest', 'D', OCHRE, OCHRE),
    ]:
        ax.scatter(data[f'median_{col}_km'], ROWS, s=22, marker=mark, facecolor=fill,
                   edgecolor=edge, linewidth=.9, zorder=3)
    horizontal(ax, [0, 10, 20, 30, 40], (-.65, 41), 'Distance to closest remaining monitor (km)')
    legend(fig, .023, .902, [('Random', 'o', 'white', GRAY),
                           ('Shared group', 'o', BLUE, BLUE),
                           ('Target-nearest', 'D', OCHRE, OCHRE)])


def paired_points(ax, frame, choices, field, sign=1):
    for i, (key, mark, color) in enumerate(choices):
        rows = city_rows(frame.loc[frame.series.eq(key)])
        value = rows[field]
        lower, upper = value - rows.lo14, rows.hi14 - value
        error = [lower, upper] if sign == 1 else [upper, lower]
        ax.errorbar(sign*value, ROWS + (-.14 if i == 0 else .14),
                    xerr=error, fmt=mark, color=color, markersize=4.4,
                    linewidth=1, capsize=2.4, capthick=.8, zorder=3)


def reconstruction(fig, ax):
    data = read('review_geometry_summary').query(
        'design == "footprint" and pattern == "footprint" and reference == "random"')
    paired_points(ax, data.rename(columns={'method': 'series'}),
                  [('covariance', 'o', BLUE), ('fixed_idw', '^', OCHRE)], 'delta_mae')
    horizontal(ax, [0, .2, .4, .6, .8], (-.035, .81), 'MAE change (µg/m³)')
    ax.axvline(0, color=GRAY, linestyle='--', linewidth=.75)
    legend(fig, .518, .902, [('Covariance', 'o', BLUE, BLUE), ('Fixed IDW', '^', OCHRE, OCHRE)])


def annual_errors(fig, axes):
    data = read('review_geometry_by_year').query(
        'design == "footprint" and method == "covariance" and pattern == "footprint" and reference == "random"')
    fig.text(.025, .399, 'Covariance predictor', fontsize=7.2, color=GRAY)
    for ax, city in zip(axes, CITIES):
        rows = data[data.metro.eq(city)].sort_values('year')
        ax.plot(rows.year, rows.delta_mae, color=BLUE, marker='o', linewidth=1.35, markersize=4.2)
        for x, y in zip(rows.year, rows.delta_mae):
            ax.annotate(f'{y:.3f}', (x, y), xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=7, color=INK)
        ax.set(xlim=(2022.75, 2025.25), xticks=[2023, 2024, 2025], ylim=(0, .355),
               yticks=[0, .1, .2, .3], xlabel='Evaluation year')
        ax.grid(axis='y', color=GRID, linewidth=.63)
        ax.tick_params(axis='y', length=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('MAE change (µg/m³)')


def local_intervals(fig, ax):
    data = read('review_interval_paired').query(
        'center == "covariance" and calibration == "footprint" and test_design == "footprint" '
        'and reference == "constant" and metric == "score"')
    paired_points(ax, data.rename(columns={'scale': 'series'}),
                  [('concentration', 's', TEAL), ('combined', 'o', BLUE)],
                  'difference', sign=-1)
    horizontal(ax, [0, 1, 2, 3, 4], (-.09, 4.10), 'Interval score reduction (µg/m³)')
    ax.axvline(0, color=GRAY, linestyle='--', linewidth=.75)
    legend(fig, .518, .902, [('Concentration', 's', TEAL, TEAL), ('Combined', 'o', BLUE, BLUE)])


def matching_calibration(fig, ax):
    data = read('review_interval_summary').query(
        'center == "covariance" and test_design == "footprint" and scale == "constant" and threshold == 35')
    random = city_rows(data[data.calibration.eq('random')]).coverage * 100
    grouped = city_rows(data[data.calibration.eq('footprint')]).coverage * 100
    ax.hlines(ROWS, random, grouped, color='#BBCAD0', linewidth=1.65, zorder=2)
    ax.scatter(random, ROWS, s=23, marker='s', facecolor='white', edgecolor=GRAY, linewidth=.9, zorder=3)
    ax.scatter(grouped, ROWS, s=23, marker='o', color=BLUE, zorder=3)
    horizontal(ax, [84, 86, 88, 90, 92], (84, 93), 'Coverage under shared-group loss (%)')
    ax.axvline(90, color=GRAY, linestyle='--', linewidth=.75)
    legend(fig, .023, .902, [('Random calibration', 's', 'white', GRAY),
                           ('Shared-group calibration', 'o', BLUE, BLUE)])


def chicago_transfer(fig, axes):
    data = read('CH_transfer_intervals').query('scale == "combined" and threshold == 35')
    data = data.set_index('regime', verify_integrity=True).loc[
        ['source_only', 'target_recalibrated', 'target_only']]
    fig.text(.025, .399, 'Combined interval scale', fontsize=7.2, color=GRAY)
    metrics = [
        (100*data.coverage, [90, 92, 94], (89.35, 95.40), 'Coverage (%)', 1),
        (data.width, [6.2, 6.4, 6.6, 6.8], (6.02, 7.03), 'Mean width (µg/m³)', 2),
        (data.interval_score, [9.2, 9.4, 9.6, 9.8], (9.15, 9.96), 'Mean score (µg/m³)', 2),
    ]
    for ax, (values, ticks, bounds, caption, precision) in zip(axes, metrics):
        horizontal(ax, ticks, bounds, caption, labels=['Source only', 'Local quantiles', 'Fully local'])
        ax.scatter(values, ROWS, color=[OCHRE, BLUE, TEAL], s=23, zorder=3)
        for y, val in zip(ROWS, values):
            ax.annotate(f'{val:.{precision}f}', (val, y), xytext=(5, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=7.2, color=INK)
    axes[0].axvline(90, color=GRAY, linestyle='--', linewidth=.75)


def screening_tradeoff(fig, ax, other):
    data = read('review_interval_summary').query(
        'center == "covariance" and calibration == "local" and test_design == "local" and '
        'threshold == 35 and scale in ["constant","combined"]')
    data = data.assign(classified_low=100*data.high_below/data.high)
    rates = data.pivot(index='metro', columns='scale', values='classified_low').loc[CITIES]
    workload = data.pivot(index='metro', columns='scale', values='reviewed').loc[CITIES]
    for city, name, color in zip(CITIES, NAMES, [BLUE, TEAL, OCHRE]):
        x = workload.loc[city, ['constant', 'combined']]
        y = rates.loc[city, ['constant', 'combined']]
        ax.plot(x, y, color=color, linewidth=1.35, zorder=2)
        ax.scatter(x.iloc[0], y.iloc[0], s=25, facecolors='white', edgecolors=color, zorder=3)
        ax.scatter(x.iloc[1], y.iloc[1], s=24, color=color, zorder=3)
        ax.annotate(name, (x.iloc[1], y.iloc[1]), xytext=(2, -12 if city == 'NY' else 6),
                    textcoords='offset points',
                    fontsize=7.3, color=color)
    ax.set(xlabel='Site-days referred for review', ylabel='High cases classified low (%)',
           xlim=(0, 305), xticks=[0, 50, 100, 150, 200, 250, 300],
           ylim=(0, 50), yticks=[0, 10, 20, 30, 40, 50])
    ax.grid(axis='y', color=GRID, linewidth=.63)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    legend(fig, .023, .902, [('Constant width', 'o', 'white', GRAY),
                           ('Combined scale', 'o', GRAY, GRAY)])

    episodes = read('review_episodes')
    omitted = episodes.assign(reduction=-100*episodes.leave_out_delta)
    spread = omitted.groupby('metro').reduction.agg(['min', 'max']).loc[CITIES]
    observed = rates.constant - rates.combined
    other.hlines(ROWS, spread['min'], spread['max'], color='#A8BDC6', linewidth=2.2, zorder=2)
    other.scatter(observed, ROWS, color=BLUE, s=25, zorder=3)
    horizontal(other, [0, 5, 10, 15, 20, 25, 30], (-.75, 30.75), 'Reduction in high cases classified low (pp)')
    other.axvline(0, color=GRAY, linestyle='--', linewidth=.75)
    fig.text(.521, .89, 'Lines show the range after omitting one episode', fontsize=7.2, color=GRAY)


def review_workload(fig, axes, data):
    legend(fig, .023, .416, [('Constant width', 'o', 'white', GRAY),
                           ('Combined scale', 'D', BLUE, BLUE)])
    for ax, city in zip(axes, CITIES):
        rows = data[data.metro.eq(city)]
        for method, symbol, color, face, linestyle, size in [
            ('constant', 'o', GRAY, 'white', '--', 5.1),
            ('combined', 'D', BLUE, BLUE, '-', 3.1),
        ]:
            s = rows[rows.scale.eq(method)].sort_values('budget_fraction')
            ax.plot(100*s.budget_fraction, 100*s.confirmed_high/s.point_miss, marker=symbol,
                    color=color, markerfacecolor=face, markersize=size, linestyle=linestyle, linewidth=1.2)
        ax.set_xscale('log', base=2)
        ax.set_xticks([.25, .5, 1, 2], ['0.25', '0.5', '1', '2'])
        ax.set(xlim=(.217, 2.30), ylim=(0, 100), yticks=[0, 25, 50, 75, 100],
               xlabel='Cases reviewed (%)')
        ax.set_xticks([], minor=True)
        ax.grid(axis='y', color=GRID, linewidth=.63)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
    axes[0].set_ylabel('Misses recovered (%)')


def panel_layout(titles, lower_left=(.110, .456, .766)):
    fig = plt.figure(figsize=(7.08, 4.95))
    boxes = [[.110, .570, .364, .265], [.609, .570, .365, .265]]
    boxes += [[left, .108, right-left, .254] for left, right in zip(lower_left, [.314, .640, .975])]
    locations = [(.025, .962), (.520, .962), (.025, .471), (.351, .471), (.677, .471)]
    axes = []
    for tag, title, (x, y), box in zip('abcde', titles, locations, boxes):
        fig.text(x, y, f'({tag}) {title}', fontsize=9.65, fontweight='bold', va='top')
        axes.append(fig.add_axes(box))
    return fig, axes


def save(fig, number):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f'Fig{number}.pdf')
    fig.savefig(FIGURES / f'Fig{number}.png', dpi=600)
    plt.close(fig)


def set_style():
    plt.rcParams.update({
        'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 7.7,
        'pdf.fonttype': 42, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': '#8899A1',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.linewidth': .66, 'axes.labelpad': 5, 'xtick.major.size': 2.8,
        'figure.facecolor': 'white',
    })


def main():
    set_style()
    fig, axes = panel_layout(['Spatial support after monitor loss', 'Error penalty from grouped loss',
                              *[f'{name} by year' for name in NAMES]], (.110, .410, .735))
    spatial_support(fig, axes[0])
    reconstruction(fig, axes[1])
    annual_errors(fig, axes[2:])
    save(fig, 5)
    fig, axes = panel_layout(['Matching the calibration mask', 'Interval score improvement',
                              'Chicago coverage', 'Chicago mean width', 'Chicago interval score'],
                             (.121, .456, .785))
    matching_calibration(fig, axes[0])
    local_intervals(fig, axes[1])
    chicago_transfer(fig, axes[2:])
    save(fig, 8)
    queue = read('review_queue').query('threshold == 35 and scale in ["constant", "combined"]')
    misses = queue.groupby('metro').point_miss.first()
    fig, axes = panel_layout(['Review workload and missed highs', 'Sensitivity to high-pollution episodes',
                              *[f'{name} ({misses[city]} misses)' for city, name in zip(CITIES, NAMES)]],
                             (.110, .410, .735))
    screening_tradeoff(fig, *axes[:2])
    review_workload(fig, axes[2:], queue)
    save(fig, 11)


if __name__ == '__main__':
    main()
