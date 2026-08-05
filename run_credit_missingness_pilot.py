from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
from xaishiftbench.credit_missingness_pilot import run_credit_missingness_pilot

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--pairs',type=int,default=10); ap.add_argument('--seed',type=int,default=20260730); ns=ap.parse_args()
 data=ns.root/'data/south_german_credit/raw/SouthGermanCredit.asc'; out=ns.root/'outputs/credit_missingness_pilot'; out.mkdir(parents=True,exist_ok=True)
 rows,features,meta=run_credit_missingness_pilot(data,seed_base=ns.seed,n_pairs=ns.pairs,models=('logistic','lightgbm'),indicator_modes=('none','all'))
 rows.to_csv(out/'credit_missingness_refits.csv',index=False); features.to_csv(out/'credit_missingness_feature_profiles.csv',index=False)
 (out/'run_metadata.json').write_text(json.dumps(asdict(meta),indent=2),encoding='utf-8'); print(json.dumps(asdict(meta),indent=2))
if __name__=='__main__': main()
