#!/usr/bin/env python3
"""Recalculate and validate the additional article Source Data effect intervals."""

from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import os
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, ks_2samp, mannwhitneyu, rankdata, spearmanr, t

from ci_source_data_io import read_sections

# Fixed seeds keep label-only renumbering numerically identical.
SORA_BOOTSTRAP_SEEDS = {'nanog_h3k27ac_rep1/SNAPtag / H3K27ac mintbody': 3904273537, 'nanog_h3k27ac_rep1/MCP': 2962495082, 'nanog_h3k27ac_rep1/mTetR': 752966123, 'nanog_ser5ph_rep1/SNAPtag / Ser5ph mintbody': 2119468077, 'nanog_ser5ph_rep1/MCP': 3425656785, 'nanog_ser5ph_rep1/mTetR': 708550852, 'sox2_h3k27ac_rep1/SNAPtag / H3K27ac mintbody': 607597237, 'sox2_h3k27ac_rep1/MCP': 2350469021, 'sox2_h3k27ac_rep1/mTetR': 274597188, 'sox2_ser5ph_rep1/SNAPtag / Ser5ph mintbody': 1780564731, 'sox2_ser5ph_rep1/MCP': 3965088427, 'sox2_ser5ph_rep1/mTetR': 1264458621}

B = 9999
TABLES = []
DATA_ROOT = DATA = PKL = WORK = None
sys.modules.setdefault('numpy._core', np.core)
sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
RESULTS, CHECKS, COVERAGE, INPUTS = [], [], [], {}


def same(x, y, key, rtol=3e-10, atol=3e-13):
    ok = bool(np.isclose(x, y, rtol=rtol, atol=atol))
    CHECKS.append({'key': key, 'actual': float(x), 'saved': float(y), 'matches': ok})
    if not ok:
        raise ValueError((key, x, y))


def seed(key):
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:4], 'little')


def finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def median_resamples(x, rng, size=B):
    x = np.sort(finite(x))
    n = len(x)
    if n < 2:
        raise ValueError('A median interval requires at least two finite observations')
    # Uniform order statistics exactly reproduce the iid empirical-bootstrap
    # median, including the joint central order statistics for even n and ties.
    k = (n + 1) // 2
    u = rng.beta(k, n + 1 - k, size)
    first = x[np.minimum((n * u).astype(int), n - 1)]
    if n % 2:
        return first
    v = u + (1 - u) * rng.beta(1, n - k, size)
    second = x[np.minimum((n * v).astype(int), n - 1)]
    return (first + second) / 2


def median_ci(a, b=None, key='', random_seed=None):
    rng = np.random.default_rng(seed(key) if random_seed is None else random_seed)
    samples = median_resamples(a, rng)
    if b is not None:
        samples -= median_resamples(b, rng)
    bounds = np.quantile(samples, [0.025, 0.975])
    estimate = float(np.median(a) - (np.median(b) if b is not None else 0))
    return [estimate, float(bounds[0]), float(bounds[1])]


def mean_ci(a, b):
    a, b = finite(a), finite(b)
    if min(len(a), len(b)) < 2:
        raise ValueError('A mean-difference interval requires two observations per group')
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb)**2 / (va**2 / (len(a)-1) + vb**2 / (len(b)-1))
    estimate = float(np.mean(a) - np.mean(b))
    margin = float(t.ppf(0.975, df) * se) if se else 0
    return [estimate, estimate - margin, estimate + margin]


def auc_ci(a, b):
    a, b = np.sort(finite(a)), np.sort(finite(b))
    if min(len(a), len(b)) < 2:
        raise ValueError('An AUC interval requires two observations per group')
    px = (np.searchsorted(b, a, 'left') + np.searchsorted(b, a, 'right')) / (2 * len(b))
    py = 1 - (np.searchsorted(a, b, 'left') + np.searchsorted(a, b, 'right')) / (2 * len(a))
    estimate = float(np.mean(px))
    se = np.sqrt(np.var(px, ddof=1)/len(a) + np.var(py, ddof=1)/len(b))
    return [estimate, max(0, estimate - 1.959963984540054*se), min(1, estimate + 1.959963984540054*se)]


def add(tab, labels, records, method):
    if len(records) != len(tab['rows']):
        raise ValueError(f"Comparison row count changed: {tab['sheet']}/{tab['header_row']}")
    for row, values in zip(tab['rows'], records):
        if len(values) != len(labels):
            raise ValueError("Effect output width does not match its labels")
        identifiers = {k: v for k, v in row['values'].items() if k in (
            'dataset_id', 'source_file', 'gene', 'factor', 'marker', 'replicate',
            'original_replicate', 'target', 'quartile', 'channel', 'radius_px',
            'panel', 'metric', 'time_min', 'window', 'win_idx', 'compare',
            'comparison', 'treatment', 'group_1', 'group_2', 'bin_idx')}
        for label, actual in zip(labels, values):
            if label not in row['values']:
                raise ValueError(f"Source Data does not contain {label}: {tab['sheet']}")
            saved = row['values'][label]
            same(actual, saved, f"{tab['sheet']}/{row['row']}/{label}")
            RESULTS.append({
                'sheet': tab['sheet'], 'table_title': tab['title'],
                'header_row': tab['header_row'], 'source_row': row['row'],
                'comparison': json.dumps(identifiers, sort_keys=True),
                'effect_column': label, 'recalculated_value': float(actual),
                'article_value': float(saved), 'method': method,
            })
    COVERAGE.append({'sheet': tab['sheet'], 'header_row': tab['header_row'],
                     'n_rows': len(records), 'effect_columns': labels, 'method': method})


def remember_input(path):
    INPUTS[path.relative_to(DATA_ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()


def table(sheet, header):
    return next(x for x in TABLES if x['sheet'] == sheet and x['header_row'] == header)


def frame(tab):
    return pd.DataFrame([x['values'] for x in tab['rows']])


def load_csv(path):
    remember_input(path)
    return pd.read_csv(path, low_memory=False)


def bootstrap_self_test():
    tests = []
    for i, values in enumerate(([0, 2], [0, 1, 8], [0, 1, 2, 20], [0, 0, 0, 5], [0, 0, 2, 3, 10])):
        x = np.asarray(values, float)
        fast = median_resamples(x, np.random.default_rng(10+i), 200000)
        brute = np.median(np.random.default_rng(20+i).choice(x, (200000, len(x))), axis=1)
        discrepancy = ks_2samp(fast, brute).statistic
        assert discrepancy < 0.009, (values, discrepancy)
        tests.append({'sample': values, 'empirical_CDF_max_difference': discrepancy})
    (WORK / 'median_bootstrap_self_test.json').write_text(json.dumps(tests, indent=2))


def radial():
    requests = defaultdict(list)
    tabs = [x for x in TABLES if 'p_value_active_vs_inactive' in x['columns']]
    for tab in tabs:
        for row in tab['rows']:
            requests[row['values']['dataset_id']].append(row['values'])
    yy, xx = np.indices((19, 19))
    radius_map = np.sqrt((xx-9)**2 + (yy-9)**2).astype(int)
    results = {}
    for count, (dataset, rows) in enumerate(requests.items(), 1):
        path = PKL / f'{dataset}.pkl'
        remember_input(path)
        with path.open('rb') as handle:
            payload = pickle.load(handle)
        rand = np.asarray(payload['on_images_random'], np.float32)
        for channel in sorted({x['channel'] for x in rows}):
            idx = {'SNAPtag': 0, 'MCP': 1, 'mTetR': 2}[channel]
            a = np.asarray(payload['on_images'][idx], np.float32)
            b = np.asarray(payload['off_images'][idx], np.float32)
            bg = np.nanmean(rand[:, :, :, idx], axis=0)
            peak = float(np.nanmax([np.nanmax(np.nanmean(a, axis=0)-bg), np.nanmax(np.nanmean(b, axis=0)-bg)]))
            scale = 100/peak if np.isfinite(peak) and peak > 0 else 1
            for r in sorted({int(x['radius_px']) for x in rows if x['channel'] == channel}):
                aa = np.nanmean(((a-bg[None, :, :])*scale)[:, radius_map == r], axis=1)
                bb = np.nanmean(((b-bg[None, :, :])*scale)[:, radius_map == r], axis=1)
                s = next(x for x in rows if x['channel'] == channel and int(x['radius_px']) == r)
                u = mannwhitneyu(aa, bb).statistic
                same(u, s['mannwhitney_U_Active'], f'{dataset}/{channel}/{r}/U')
                same(len(aa), s['n_Active'], f'{dataset}/{channel}/{r}/nA')
                same(len(bb), s['n_Inactive'], f'{dataset}/{channel}/{r}/nI')
                results[(dataset, channel, r)] = mean_ci(aa, bb)
        if count % 10 == 0:
            print('Radial datasets', count, '/', len(requests), flush=True)
    for tab in tabs:
        records = [results[(x['values']['dataset_id'], x['values']['channel'], int(x['values']['radius_px']))] for x in tab['rows']]
        add(tab, ['mean_difference_Active_minus_Inactive', 'mean_difference_CI95_low', 'mean_difference_CI95_high'], records,
            'Pointwise Welch mean-difference 95% CI, original float32 normalized radial locus observations; no pooling across radii or replicates; original MW/BH unchanged.')


def snapshots():
    path = DATA_ROOT / '01_Snapshot_analysis/ci_inputs/mean_intensity_long_from_csv.csv'
    long = load_csv(path)
    long['replicate'] = long.replicate.astype(str)
    long['well'] = long.well.astype(str)
    groups = dict(tuple(long.groupby(['target', 'factor', 'replicate', 'well', 'channel'])))
    for tab in [table('Sup Fig 3', 2), table('Sup Fig 9', 11)]:
        records = []
        for row in tab['rows']:
            s = row['values']
            key = (s['target'], s['factor'], str(s['replicate']), str(s['well']), s['channel'])
            subset = groups[key]
            a = finite(subset.loc[subset.state == 'Active', 'value'])
            b = finite(subset.loc[subset.state == 'Inactive', 'value'])
            same(len(a), s['n_active'], f'{key}/nA')
            same(len(b), s['n_inactive'], f'{key}/nI')
            ci = median_ci(np.log2(a+1e-6), np.log2(b+1e-6), key)
            same(ci[0], s['delta_median_log2'], f'{key}/median')
            aci = auc_ci(a, b)
            same(aci[0], s['auc_signed'], f'{key}/AUC')
            # Best-direction AUC is max(AUC,1-AUC); include both possible directions.
            low = 0.5 if aci[1] <= 0.5 <= aci[2] else min(max(aci[1], 1-aci[1]), max(aci[2], 1-aci[2]))
            high = max(max(aci[1], 1-aci[1]), max(aci[2], 1-aci[2]))
            records.append([ci[1], ci[2], 2**ci[1], 2**ci[2], aci[1], aci[2], low, high])
        add(tab, ['delta_median_log2_CI95_low', 'delta_median_log2_CI95_high', 'fold_change_CI95_low', 'fold_change_CI95_high',
                  'auc_signed_CI95_low', 'auc_signed_CI95_high', 'auc_best_CI95_low', 'auc_best_CI95_high'], records,
            '9999 iid observation bootstrap draws for difference of log2 medians; DeLong placement-variance normal CI for signed AUC, transformed CI for best-direction AUC; group-specific, original q unchanged.')
    center = load_csv(DATA_ROOT/'01_Snapshot_analysis/ci_inputs/snapshot_cell_measurements.csv')
    # These legacy additional summaries are nuclear CSV means, not the panel b
    # locus-centered violin values. Verify that distinction against saved medians.
    tab = table('Sup Fig 4', 3850)
    records = []
    for row in tab['rows']:
        s = row['values']
        v = center.loc[center.dataset_id == s['dataset_id']]
        a = finite(v.loc[v.state.isin(['Active', 'ON']), s['channel']+'_mean_intensity'])
        b = finite(v.loc[v.state.isin(['Inactive', 'OFF']), s['channel']+'_mean_intensity'])
        same(len(a), s['n_active'], f'{s["dataset_id"]}/{s["channel"]}/center_nA')
        same(np.median(a), s['median_active'], f'{s["dataset_id"]}/{s["channel"]}/centerA')
        same(np.median(b), s['median_inactive'], f'{s["dataset_id"]}/{s["channel"]}/centerI')
        records.append(median_ci(a, b, ('center', s['dataset_id'], s['channel'])))
    add(tab, ['median_difference_Active_minus_Inactive', 'median_difference_CI95_low', 'median_difference_CI95_high'], records, '9999 independent original CSV-observation bootstrap draws; nuclear mean intensity; original P/q unchanged.')
    raw = pd.concat([frame(table('Sup Fig 4', 23371)), frame(table('Sup Fig 4', 105698))])
    tab = table('Sup Fig 4', 105593)
    records = []
    for row in tab['rows']:
        s = row['values']
        v = raw.loc[(raw.gene == s['gene']) & (raw.factor == s['factor']) & (raw.replicate == s['original_replicate'])]
        a = finite(v.loc[v.state == 'Active', 'log10_raw_SNAP_3x3_mean'])
        b = finite(v.loc[v.state == 'Inactive', 'log10_raw_SNAP_3x3_mean'])
        same(len(a), s['n_Active'], f'violin/{s["gene"]}/{s["factor"]}/nA')
        same(mannwhitneyu(a,b).statistic, s['mannwhitney_U_Active'], f'violin/{s["gene"]}/{s["factor"]}/U')
        records.append(median_ci(a, b, ('violin',s['gene'],s['factor'],s['original_replicate'])))
    add(tab, ['median_log10_difference_Active_minus_Inactive', 'median_log10_difference_CI95_low', 'median_log10_difference_CI95_high'], records, '9999 locus-instance bootstrap draws; original displayed replicate and log10 transform; original P unchanged.')
    pairs = load_csv(DATA_ROOT/'01_Snapshot_analysis/ci_inputs/snapshot_snap_mcp_center_observations.csv')
    tab = table('Sup Fig 4', 3950)
    records = []
    def rho(x, y, axis=-1):
        rx, ry = rankdata(x, axis=axis), rankdata(y, axis=axis)
        rx -= np.mean(rx, axis=axis, keepdims=True)
        ry -= np.mean(ry, axis=axis, keepdims=True)
        return np.sum(rx*ry, axis=axis)/np.sqrt(np.sum(rx*rx, axis=axis)*np.sum(ry*ry, axis=axis))
    for row in tab['rows']:
        s = row['values']
        g = pairs.loc[pairs.dataset_id == s['dataset_id']]
        x, y = g.snap_center_mean_3x3.to_numpy(float), g.relative_mcp_at_center.to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        same(len(x), s['n_pooled_valid'], s['dataset_id']+'/rho_n')
        same(spearmanr(x,y).statistic, s['spearman_rho'], s['dataset_id']+'/rho')
        ci = bootstrap((x,y), rho, paired=True, vectorized=True, method='percentile', n_resamples=1999, batch=32, random_state=seed(('rho',s['dataset_id']))).confidence_interval
        records.append([float(ci.low), float(ci.high)])
        print('Spearman CI', s['dataset_id'], flush=True)
    add(tab, ['spearman_rho_CI95_low', 'spearman_rho_CI95_high'], records, '1999 paired locus-observation bootstrap draws; SNAP/MCP pairing retained; percentile pointwise 95% CI; original P/q unchanged.')
    values = frame(table('Sup Fig 4',5920))
    tab = table('Sup Fig 4',5902)
    records = []
    for row in tab['rows']:
        s = row['values']
        dataset = s['source_file'].split('/')[4]
        v = values.loc[(values.dataset == dataset) & (values.channel == s['channel'])]
        a, b = (finite(v.loc[v.state == state,'core_minus_annulus']) for state in ['Active','Inactive'])
        same(len(a),s['n_active'],f'SoRa/{dataset}/{s["channel"]}/nA')
        same(np.median(a),s['active_median_core_minus_annulus'],f'SoRa/{dataset}/{s["channel"]}/medA')
        same(mannwhitneyu(a,b).statistic,s['mannwhitney_u'],f'SoRa/{dataset}/{s["channel"]}/U')
        records.append(median_ci(a,b,random_seed=SORA_BOOTSTRAP_SEEDS[dataset + '/' + s['channel']]))
    add(tab,['median_difference_Active_minus_Inactive','median_difference_CI95_low','median_difference_CI95_high'],records,'9999 quality-filtered locus-instance bootstrap draws; displayed experiment (public replicate 1) only; original MW P unchanged.')


def seqfish():
    values = frame(table('Fig 2',15184))
    for header in (87,15904,131):
        tab = table('Fig 2',header)
        records = []
        for row in tab['rows']:
            s = row['values']
            v = values.loc[values.marker == s['marker']]
            if header != 131:
                v = v.loc[v.activity_rank_quartile == s['quartile']]
                same(len(v),s['n'],str(('seqfish',s['marker'],s['quartile'],'n')))
            x = finite(v.delta_z_Active_minus_Inactive)
            ci = median_ci(x,key=('seqfish',header,s['marker'],s.get('quartile','all')))
            if header == 131:
                same(ci[0],s['median_delta'],f'seqfish/{s["marker"]}/median')
            records.append(ci)
        add(tab,['median_delta_z_Active_minus_Inactive','median_delta_z_CI95_low','median_delta_z_CI95_high'],records,'9999 gene-level bootstrap draws, resampling precomputed paired Active/Inactive gene differences; original Wilcoxon P/BH unchanged.')
    values = frame(table('Fig 2',145))
    tab = table('Fig 2',15149)
    records = []
    for row in tab['rows']:
        s=row['values']
        x=finite(values.loc[(values.target==s['target'])&(values.bin_idx==s['bin_idx']),'log_ratio'])
        ci=median_ci(x,key=('sci',s['target'],s['bin_idx']))
        same(2**ci[0],s['median_ratio'],f'sci/{s["target"]}/{s["bin_idx"]}/median')
        records.append([ci[0],ci[1],ci[2],2**ci[1],2**ci[2]])
    add(tab,['median_log2_ratio','median_log2_ratio_CI95_low','median_log2_ratio_CI95_high','median_ratio_CI95_low','median_ratio_CI95_high'],records,'9999 gene-level bootstrap draws of paired state log2 ratios; ratio CI is exp2(log2 CI); original Wilcoxon P/BH unchanged.')


def inhibitors():
    tab=table('Sup Fig 12',22757)
    sources={gene:load_csv(DATA/f'10_HDAC_time_windows/Fig5_{gene}_duty_cycle_by_window_cell.csv') for gene in ['Nanog','Sox2']}
    records=[]
    for row in tab['rows']:
        s=row['values'];v=sources[s['gene']];v=v.loc[v.win_idx==s['win_idx']]
        treatment=s['compare'].split(' vs ')[1]
        a=finite(v.loc[v.group==treatment,'active_fraction']);b=finite(v.loc[v.group=='DMSO','active_fraction'])
        same(mannwhitneyu(b,a).statistic,s['mannwhitney_U_DMSO'],f'window/{s["gene"]}/{s["window"]}/{treatment}/U')
        records.append(median_ci(a,b,('window',s['gene'],s['window'],treatment)))
    add(tab,['median_duty_difference_treatment_minus_DMSO','median_duty_difference_CI95_low','median_duty_difference_CI95_high'],records,'9999 cell-level bootstrap draws separately within each window; no frame resampling or pooling windows; original MW/Holm unchanged.')
    values=frame(table('Sup Fig 13',2));values=values.loc[values.included_in_plot.eq(True)]
    tab=table('Sup Fig 13',13620);records=[]
    for row in tab['rows']:
        s=row['values'];v=values
        for key in ['panel','gene','display_snap','comparison','metric','time_min']:
            v=v.loc[v[key]==s[key]]
        a=finite(v.loc[v.group==s['group_2'],'log2_fold_change']);b=finite(v.loc[v.group==s['group_1'],'log2_fold_change'])
        same(len(a),s['n_2'],f'acute/{row["row"]}/n2');same(len(b),s['n_1'],f'acute/{row["row"]}/n1')
        same(mannwhitneyu(b,a).statistic,s['mannwhitney_U_group_1'],f'acute/{row["row"]}/U')
        ci=median_ci(a,b,('acute',s['panel'],s['metric'],s['time_min'],s['group_2']))
        same(ci[0],s['median_difference_log2'],f'acute/{row["row"]}/median')
        records.append(ci[1:])
    add(tab,['median_difference_log2_CI95_low','median_difference_log2_CI95_high'],records,'9999 cell-level bootstrap draws, separately by displayed panel, metric and time point after experiment-specific normalization; group_2 minus group_1; conditional on included pooled imaging series, not a biological-replicate CI; original MW/Holm unchanged.')


def fig5():
    frames, runs = [], []
    for gene in ['Nanog', 'Sox2']:
        root = DATA_ROOT / '12_Fig5_integrated' / gene / 'tetr_mcp_hybrid_analysis'
        for name, target in [('aggregated_perframe.csv', frames), ('aggregated_runs_by_cell.csv', runs)]:
            values = load_csv(root / name)
            values = values.loc[values.group.isin(['DMSO', 'TSA', 'RGFP966_0h'])].copy()
            values['group'] = values.group.str.replace('_0h', '', regex=False)
            values['gene'] = gene
            target.extend(values.to_dict('records'))
    observed = pd.DataFrame(frames)
    original_frames = frame(table('Fig 5', 2))
    original_runs = frame(table('Fig 5', 11962))
    same(len(frames), len(original_frames), 'Fig5/frame-count')
    same(len(runs), len(original_runs), 'Fig5/run-count')
    duty = frame(table('Fig 5', 14008))
    duty['group'] = duty.group.str.replace('_0h', '', regex=False)
    observed['cell_uid'] = observed.file + '_c' + observed.cell_id.astype(str)
    derived = observed.groupby(['gene', 'group', 'cell_uid']).on_off.mean()
    for record in duty.itertuples():
        same(derived.loc[(record.gene, record.group, record.cell_uid)], record.duty_cycle,
             'Fig5/' + record.cell_uid)
    tab = table('Fig 5', 14240)
    records = []
    for row in tab['rows']:
        saved = row['values']
        gene, treatment = saved['gene'], saved['treatment']
        a = finite(duty.loc[(duty.gene == gene) & (duty.group == treatment), 'duty_cycle'])
        b = finite(duty.loc[(duty.gene == gene) & (duty.group == 'DMSO'), 'duty_cycle'])
        u, pvalue = mannwhitneyu(b, a)
        same(len(a), saved['n_treatment'], f'Fig5/{gene}/{treatment}/n')
        same(len(b), saved['n_DMSO'], f'Fig5/{gene}/{treatment}/control_n')
        same(u, saved['mannwhitney_U_DMSO'], f'Fig5/{gene}/{treatment}/U')
        same(pvalue, saved['P_two_sided'], f'Fig5/{gene}/{treatment}/P')
        records.append(median_ci(a, b, ('Fig5', gene, treatment)))
    add(tab, ['median_duty_difference_treatment_minus_DMSO',
              'median_duty_difference_CI95_low', 'median_duty_difference_CI95_high'],
        records, '9999 cell-level percentile bootstrap draws; treatment minus DMSO; original MW/Holm unchanged.')


def main():
    global TABLES, DATA_ROOT, DATA, PKL, WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path(os.environ.get('GAO2026_DATA_ROOT', 'data')))
    parser.add_argument('--source-data', type=Path, required=True,
                        help='The article Source_Data.xlsx workbook, distributed with the paper.')
    parser.add_argument('--output-root', type=Path,
                        default=Path(os.environ.get('GAO2026_OUTPUT_ROOT', 'outputs')) / 'source_data_ci')
    parser.add_argument('--sections', nargs='+',
                        choices=['radial', 'snapshots', 'seqfish', 'inhibitors', 'fig5'],
                        default=['radial', 'snapshots', 'seqfish', 'inhibitors', 'fig5'])
    parser.add_argument('--self-test', action='store_true',
                        help='Also check the bootstrap-median sampling distribution against direct resampling.')
    args = parser.parse_args()
    DATA_ROOT = args.data_root.expanduser().resolve()
    if not (DATA_ROOT / '01_Snapshot_analysis').is_dir():
        raise FileNotFoundError('The data root must contain 01_Snapshot_analysis through 16_ChromHMM_Fig3c')
    DATA = DATA_ROOT
    PKL = DATA_ROOT / '01_Snapshot_analysis/pkl'
    WORK = args.output_root.expanduser().resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    workbook = args.source_data.expanduser().resolve()
    before_hash = hashlib.sha256(workbook.read_bytes()).hexdigest()
    INPUTS['article/Source_Data.xlsx'] = before_hash
    TABLES = read_sections(workbook)
    if args.self_test:
        bootstrap_self_test()
    functions = {'radial': radial, 'snapshots': snapshots, 'seqfish': seqfish,
                 'inhibitors': inhibitors, 'fig5': fig5}
    for section in args.sections:
        print('Calculating:', section, flush=True)
        functions[section]()
    if before_hash != hashlib.sha256(workbook.read_bytes()).hexdigest():
        raise RuntimeError('Source Data changed during the calculation')
    with (WORK / 'comparison_effect_intervals.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS[0]))
        writer.writeheader()
        writer.writerows(RESULTS)
    report = {'all_match': all(x['matches'] for x in CHECKS), 'checks': len(CHECKS),
              'effect_rows': sum(x['n_rows'] for x in COVERAGE),
              'effect_values': len(RESULTS), 'median_bootstrap_draws': B,
              'spearman_bootstrap_draws': 1999, 'inputs': INPUTS,
              'source_data_sha256': before_hash, 'source_data_unchanged': True,
              'existing_P_q_U_W_unchanged': True,
              'conditional_on_analyzed_series': True, 'coverage': COVERAGE,
              'software': {'numpy': np.__version__, 'pandas': pd.__version__,
                           'scipy': __import__('scipy').__version__}}
    (WORK / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    (WORK / 'numeric_checks.json').write_text(json.dumps(CHECKS, indent=2) + '\n')
    print(f"PASS: {report['effect_rows']} comparison rows; {report['effect_values']} effect/CI values", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
