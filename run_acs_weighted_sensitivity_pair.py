from __future__ import annotations
import argparse, json
from pathlib import Path
from xaishiftbench.acs_temporal_pilot import run_pair


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--pair',type=int,required=True)
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    out=args.root/'outputs'/'acs_weighted_sensitivity'/'pair_runs'
    out.mkdir(parents=True,exist_ok=True)
    rows_path=out/f'pair_{args.pair:02d}_rows.csv'
    features_path=out/f'pair_{args.pair:02d}_features.csv'
    subgroups_path=out/f'pair_{args.pair:02d}_subgroups.csv'
    metadata_path=out/f'pair_{args.pair:02d}_metadata.json'
    if rows_path.exists() and metadata_path.exists():
        print('existing'); return
    rows, features, subgroups, meta=run_pair(args.root,args.pair)
    rows.to_csv(rows_path,index=False)
    features.to_csv(features_path,index=False)
    subgroups.to_csv(subgroups_path,index=False)
    meta['survey_weight_sensitivity']='PWGTP-weighted energy on release-defined matched explanation rows'
    metadata_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
