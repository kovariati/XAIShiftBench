from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (14, 28, 42, 56)
KEY = ['code_module', 'code_presentation', 'id_student']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_zip_csv(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f, **kwargs)


def clean_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def combine_additive(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=KEY)
    x = pd.concat(parts, ignore_index=True)
    value_cols = [c for c in x.columns if c not in KEY]
    return x.groupby(KEY, as_index=False, observed=True)[value_cols].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--chunksize', type=int, default=750_000)
    args = ap.parse_args()
    t0 = time.time()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f'CRC error: {bad}')
        expected = {
            'assessments.csv', 'courses.csv', 'studentAssessment.csv', 'studentInfo.csv',
            'studentRegistration.csv', 'studentVle.csv', 'vle.csv'
        }
        names = set(zf.namelist())
        missing = expected - names
        if missing:
            raise ValueError(f'Missing OULAD members: {sorted(missing)}')

        courses = read_zip_csv(zf, 'courses.csv')
        info = read_zip_csv(zf, 'studentInfo.csv', na_values=['?'])
        reg = read_zip_csv(zf, 'studentRegistration.csv', na_values=['?'])
        vle = read_zip_csv(zf, 'vle.csv', na_values=['?'])

        if info.duplicated(KEY).any() or reg.duplicated(KEY).any():
            raise ValueError('Student key is not unique in studentInfo or studentRegistration')

        base = info.merge(reg, on=KEY, how='left', validate='one_to_one')
        base['target_unsuccessful'] = base['final_result'].isin(['Fail', 'Withdrawn']).astype('int8')
        base['date_registration'] = pd.to_numeric(base['date_registration'], errors='coerce')
        base['date_unregistration'] = pd.to_numeric(base['date_unregistration'], errors='coerce')

        course_keys = courses[['code_module', 'code_presentation']].drop_duplicates().reset_index(drop=True)
        course_keys['course_id'] = np.arange(len(course_keys), dtype=np.int16)
        course_map = {(r.code_module, r.code_presentation): int(r.course_id) for r in course_keys.itertuples()}
        vle = vle.merge(course_keys, on=['code_module', 'code_presentation'], how='left', validate='many_to_one')
        if vle['course_id'].isna().any():
            raise ValueError('VLE site metadata has unknown course key')
        vle['activity_feature'] = 'clicks_' + vle['activity_type'].map(clean_name)
        site_meta = vle[['course_id', 'id_site', 'activity_feature']].copy()

        additive: dict[int, list[pd.DataFrame]] = {h: [] for h in HORIZONS}
        active_parts: list[pd.DataFrame] = []
        site_first_parts: list[pd.DataFrame] = []
        n_vle_rows = 0
        n_vle_upto56 = 0

        with zf.open('studentVle.csv') as f:
            reader = pd.read_csv(
                f,
                chunksize=args.chunksize,
                usecols=['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click'],
                dtype={
                    'code_module': 'string', 'code_presentation': 'string',
                    'id_student': 'int32', 'id_site': 'int32', 'date': 'int16', 'sum_click': 'int32'
                },
            )
            for chunk_idx, chunk in enumerate(reader):
                n_vle_rows += len(chunk)
                chunk['course_id'] = [course_map[(m, p)] for m, p in zip(chunk['code_module'], chunk['code_presentation'])]
                chunk['course_id'] = chunk['course_id'].astype('int16')
                chunk = chunk[chunk['date'] <= max(HORIZONS)].copy()
                n_vle_upto56 += len(chunk)
                if chunk.empty:
                    continue
                chunk = chunk.merge(site_meta, on=['course_id', 'id_site'], how='left', validate='many_to_one')
                if chunk['activity_feature'].isna().any():
                    raise ValueError('studentVle references site missing from vle.csv')

                active_parts.append(chunk[['course_id', 'id_student', 'date']].drop_duplicates())
                sf = chunk.groupby(['course_id', 'id_student', 'id_site'], as_index=False, observed=True)['date'].min()
                site_first_parts.append(sf)

                for h in HORIZONS:
                    sub = chunk[chunk['date'] <= h]
                    if sub.empty:
                        continue
                    basic = sub.groupby(['course_id', 'id_student'], as_index=False, observed=True).agg(
                        total_clicks=('sum_click', 'sum'),
                        interaction_rows=('sum_click', 'size'),
                    )
                    act = sub.pivot_table(
                        index=['course_id', 'id_student'], columns='activity_feature',
                        values='sum_click', aggfunc='sum', fill_value=0, observed=True
                    ).reset_index()
                    merged = basic.merge(act, on=['course_id', 'id_student'], how='left', validate='one_to_one')
                    merged = merged.merge(course_keys, on='course_id', how='left', validate='many_to_one').drop(columns='course_id')
                    additive[h].append(merged)
                print(f'processed VLE chunk {chunk_idx + 1}; total rows={n_vle_rows:,}; <=56={n_vle_upto56:,}', flush=True)

    # Exact active-day and unique-site cumulative counts.
    active = pd.concat(active_parts, ignore_index=True).drop_duplicates(['course_id', 'id_student', 'date'])
    site_first = pd.concat(site_first_parts, ignore_index=True)
    site_first = site_first.groupby(['course_id', 'id_student', 'id_site'], as_index=False, observed=True)['date'].min()

    horizon_tables = []
    audit_rows = []
    activity_cols: set[str] = set()
    for h in HORIZONS:
        agg = combine_additive(additive[h])
        activity_cols.update(c for c in agg.columns if c.startswith('clicks_'))

        ad = active[active['date'] <= h].groupby(['course_id', 'id_student'], as_index=False, observed=True).size()
        ad = ad.rename(columns={'size': 'active_days'})
        us = site_first[site_first['date'] <= h].groupby(['course_id', 'id_student'], as_index=False, observed=True).size()
        us = us.rename(columns={'size': 'unique_sites'})
        aux = ad.merge(us, on=['course_id', 'id_student'], how='outer')
        aux = aux.merge(course_keys, on='course_id', how='left', validate='many_to_one').drop(columns='course_id')
        agg = agg.merge(aux, on=KEY, how='outer')

        risk = base[(base['date_registration'].isna() | (base['date_registration'] <= h)) &
                    (base['date_unregistration'].isna() | (base['date_unregistration'] > h))].copy()
        risk['horizon_day'] = h
        table = risk.merge(agg, on=KEY, how='left', validate='one_to_one')
        numeric_vle = ['total_clicks', 'interaction_rows', 'active_days', 'unique_sites'] + [c for c in table.columns if c.startswith('clicks_')]
        for c in numeric_vle:
            table[c] = table[c].fillna(0)
        table['clicks_per_active_day'] = np.where(table['active_days'] > 0, table['total_clicks'] / table['active_days'], 0.0)
        table['days_registered_before_start'] = -table['date_registration']
        table['presentation_year'] = table['code_presentation'].str[:4].astype('int16')
        table['presentation_period'] = table['code_presentation'].str[-1]
        horizon_tables.append(table)

        for (m, p), g in table.groupby(['code_module', 'code_presentation'], observed=True):
            audit_rows.append({
                'horizon_day': h,
                'code_module': m,
                'code_presentation': p,
                'n_risk_set': int(len(g)),
                'n_unsuccessful': int(g['target_unsuccessful'].sum()),
                'prevalence_unsuccessful': float(g['target_unsuccessful'].mean()),
                'zero_vle_share': float((g['total_clicks'] == 0).mean()),
            })

    model = pd.concat(horizon_tables, ignore_index=True)
    # Activity types may appear only at later horizons; concatenation can therefore
    # introduce NaN columns for earlier horizons. These are structural zeroes.
    all_vle_cols = ['total_clicks', 'interaction_rows', 'active_days', 'unique_sites', 'clicks_per_active_day'] + [c for c in model.columns if c.startswith('clicks_')]
    for c in all_vle_cols:
        model[c] = model[c].fillna(0)

    # Reproduction-compatible cohort used in the earlier CRIT-AID preparation: it
    # excluded already-unregistered students but did not exclude registrations after
    # the prediction horizon. The prospective primary cohort below is stricter.
    cohort_rows = []
    for h in HORIZONS:
        primary_n = int(((base['date_registration'].isna() | (base['date_registration'] <= h)) & (base['date_unregistration'].isna() | (base['date_unregistration'] > h))).sum())
        comparison_n = int((base['date_unregistration'].isna() | (base['date_unregistration'] > h)).sum())
        cohort_rows.append({'horizon_day': h, 'prospective_primary_n': primary_n, 'comparison_reproduction_n': comparison_n, 'late_registration_excluded': comparison_n - primary_n})
    pd.DataFrame(cohort_rows).to_csv(out / 'oulad_cohort_policy_comparison.csv', index=False)

    # Keep final_result only for auditability; model adapters must explicitly exclude it.
    model.to_csv(out / 'oulad_score_free_horizons.csv.gz', index=False, compression='gzip')
    pd.DataFrame(audit_rows).to_csv(out / 'oulad_risk_set_audit.csv', index=False)

    # Exact same-module and same-period temporal pairs.
    available = set(zip(courses['code_module'], courses['code_presentation']))
    pairs = []
    for module in sorted(courses['code_module'].unique()):
        for period in ['B', 'J']:
            src, tgt = f'2013{period}', f'2014{period}'
            if (module, src) in available and (module, tgt) in available:
                pairs.append({'code_module': module, 'period': period, 'source_presentation': src, 'target_presentation': tgt})
    pair_df = pd.DataFrame(pairs)
    if len(pair_df) != 9:
        raise ValueError(f'Expected 9 temporal pairs, found {len(pair_df)}')
    pair_df.to_csv(out / 'oulad_temporal_pairs.csv', index=False)

    feature_cols = [
        'gender', 'region', 'highest_education', 'imd_band', 'age_band',
        'num_of_prev_attempts', 'studied_credits', 'disability',
        'days_registered_before_start', 'total_clicks', 'interaction_rows',
        'active_days', 'unique_sites', 'clicks_per_active_day'
    ] + sorted(activity_cols)
    feature_registry = pd.DataFrame({
        'feature': feature_cols,
        'role': ['static_or_registration' if c in {
            'gender', 'region', 'highest_education', 'imd_band', 'age_band',
            'num_of_prev_attempts', 'studied_credits', 'disability', 'days_registered_before_start'
        } else 'vle_cumulative' for c in feature_cols],
        'available_by_horizon': True,
        'score_free': True,
    })
    feature_registry.to_csv(out / 'oulad_feature_registry.csv', index=False)

    audit = {
        'source_zip': str(args.zip),
        'source_sha256': sha256(args.zip),
        'zip_crc_pass': True,
        'n_student_info_rows': int(len(info)),
        'n_courses': int(len(courses)),
        'n_vle_rows': int(n_vle_rows),
        'n_vle_rows_date_le_56': int(n_vle_upto56),
        'n_modeling_rows': int(len(model)),
        'n_comparison_reproduction_rows': int(sum(r['comparison_reproduction_n'] for r in cohort_rows)),
        'n_late_registration_rows_excluded': int(sum(r['late_registration_excluded'] for r in cohort_rows)),
        'n_temporal_pairs': int(len(pair_df)),
        'horizons': list(HORIZONS),
        'target_definition': '1=Fail or Withdrawn; 0=Pass or Distinction',
        'risk_set_definition': 'registered by horizon and not unregistered on or before horizon',
        'assessment_scores_used': False,
        'date_unregistration_used_as_feature': False,
        'runtime_seconds': time.time() - t0,
    }
    (out / 'oulad_preparation_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
