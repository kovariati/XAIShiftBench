from __future__ import annotations
import argparse, json
from pathlib import Path
from xaishiftbench.oulad_temporal_pilot import run_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', type=int, required=True)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    out = args.root / 'outputs' / 'oulad_fixed_cohort' / 'pair_runs'
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f'pair_{args.pair:02d}_rows.csv'
    feat_path = out / f'pair_{args.pair:02d}_features.csv'
    meta_path = out / f'pair_{args.pair:02d}_metadata.json'
    if rows_path.exists() and feat_path.exists() and meta_path.exists():
        print('existing')
        return
    rows, feats, meta = run_pair(
        args.root / 'outputs' / 'oulad_fixed_cohort' / 'oulad_fixed_cohort_h14_h56.csv.gz',
        args.root / 'outputs' / 'oulad_prepared' / 'oulad_temporal_pairs.csv',
        args.pair,
    )
    rows.to_csv(rows_path, index=False)
    feats.to_csv(feat_path, index=False)
    meta['cohort_policy'] = 'fixed_intersection_day14_day56'
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))

if __name__ == '__main__': main()
