from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP_HARM', 'WKHP', 'SEX', 'RAC1P']
CATEGORICAL = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP_HARM', 'SEX', 'RAC1P']
NUMERIC = ['AGEP', 'WKHP']
CPI_2018 = 251.107
CPI_2024 = 313.689
REAL_2024_THRESHOLD = 50_000 * CPI_2024 / CPI_2018

REL_2018 = {
    0: 'reference_person', 1: 'spouse_or_partner', 2: 'child', 3: 'child', 4: 'child',
    5: 'sibling', 6: 'parent', 7: 'grandchild', 8: 'in_law', 9: 'in_law',
    10: 'other_relative', 11: 'roommate_housemate_boarder', 12: 'roommate_housemate_boarder',
    13: 'spouse_or_partner', 14: 'foster_child', 15: 'other_nonrelative',
    16: 'institutionalized_group_quarters', 17: 'noninstitutionalized_group_quarters',
}
REL_2024 = {
    20: 'reference_person', 21: 'spouse_or_partner', 22: 'spouse_or_partner',
    23: 'spouse_or_partner', 24: 'spouse_or_partner', 25: 'child', 26: 'child', 27: 'child',
    28: 'sibling', 29: 'parent', 30: 'grandchild', 31: 'in_law', 32: 'in_law',
    33: 'other_relative', 34: 'roommate_housemate_boarder', 35: 'foster_child',
    36: 'other_nonrelative', 37: 'institutionalized_group_quarters',
    38: 'noninstitutionalized_group_quarters',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_dictionary(path: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            out.setdefault(row[1], []).append({
                'kind': row[0], 'type': row[2] if len(row)>2 else '',
                'length': row[3] if len(row)>3 else '', 'start': row[4] if len(row)>4 else '',
                'end': row[5] if len(row)>5 else '',
                'description': row[6] if len(row)>6 else (row[4] if len(row)>4 else ''),
            })
    return out


def dictionary_audit(d18: dict, d24: dict) -> pd.DataFrame:
    rows=[]
    for v18,v24 in [('AGEP','AGEP'),('COW','COW'),('SCHL','SCHL'),('MAR','MAR'),('OCCP','OCCP'),
                    ('POBP','POBP'),('RELP','RELSHIPP'),('WKHP','WKHP'),('SEX','SEX'),
                    ('RAC1P','RAC1P'),('PINCP','PINCP'),('ST','STATE')]:
        r18=d18.get(v18,[]); r24=d24.get(v24,[])
        vals18={(r['start'],r['end'],r['description']) for r in r18 if r['kind']=='VAL'}
        vals24={(r['start'],r['end'],r['description']) for r in r24 if r['kind']=='VAL'}
        rows.append({'variable_2018':v18,'variable_2024':v24,
                     'label_2018':next((r['description'] for r in r18 if r['kind']=='NAME'),''),
                     'label_2024':next((r['description'] for r in r24 if r['kind']=='NAME'),''),
                     'n_value_rows_2018':len(vals18),'n_value_rows_2024':len(vals24),
                     'value_definitions_identical':vals18==vals24,
                     'requires_harmonization':(v18!=v24) or (vals18!=vals24)})
    return pd.DataFrame(rows)



def adjust_pincp(pincp: pd.Series, adjinc: pd.Series) -> pd.Series:
    """Convert raw PUMS income to survey-year constant dollars."""
    income = pd.to_numeric(pincp, errors='coerce')
    factor = pd.to_numeric(adjinc, errors='coerce') / 1_000_000.0
    return income * factor

def process_zip(zip_path: Path, year: int, part: str, output_path: Path, chunksize: int) -> dict:
    state_col, rel_col, rel_map = ('ST','RELP',REL_2018) if year==2018 else ('STATE','RELSHIPP',REL_2024)
    usecols=['SERIALNO','SPORDER',state_col,'PWGTP','ADJINC','AGEP','COW','SCHL','MAR','OCCP','POBP',rel_col,'WKHP','SEX','RAC1P','PINCP']
    n_raw=n_filtered=pos_nom=pos_real=0
    weight_sum=weight_pos_nom=weight_pos_real=0.0
    categories={c:set() for c in ['COW','SCHL','MAR','OCCP','POBP',rel_col,'SEX','RAC1P']}
    if output_path.exists(): output_path.unlink()
    first=True
    with zipfile.ZipFile(zip_path) as zf:
        bad=zf.testzip()
        if bad: raise RuntimeError(f'CRC error in {zip_path}: {bad}')
        names=zf.namelist()
        if len(names)!=1: raise ValueError(f'Expected one member in {zip_path}: {names}')
        with zf.open(names[0]) as raw, gzip.open(output_path,'wt',encoding='utf-8',newline='',compresslevel=1) as dest:
            for chunk in pd.read_csv(raw,usecols=usecols,chunksize=chunksize,low_memory=False):
                n_raw += len(chunk)
                for c in categories:
                    categories[c].update(pd.to_numeric(chunk[c],errors='coerce').dropna().astype(int).unique().tolist())
                for c in ['SPORDER',state_col,'PWGTP','ADJINC','AGEP','COW','SCHL','MAR','OCCP','POBP',rel_col,'WKHP','SEX','RAC1P','PINCP']:
                    chunk[c]=pd.to_numeric(chunk[c],errors='coerce')
                chunk['PINCP_RAW']=chunk['PINCP']
                chunk['ADJINC_FACTOR']=chunk['ADJINC']/1_000_000.0
                chunk['PINCP_ADJ']=adjust_pincp(chunk['PINCP_RAW'],chunk['ADJINC'])
                chunk=chunk[(chunk.AGEP>16)&(chunk.PINCP_ADJ>100)&(chunk.WKHP>0)&(chunk.PWGTP>=1)].copy()
                n_filtered += len(chunk)
                chunk['YEAR']=year; chunk['SOURCE_PART']=part
                chunk['STATE']=chunk[state_col].astype('Int16')
                chunk['RELP_RAW']=chunk[rel_col].astype('Int16')
                chunk['RELP_HARM']=chunk[rel_col].map(rel_map).fillna('unknown_relationship')
                chunk['PINCP']=chunk['PINCP_ADJ']
                chunk['TARGET_NOMINAL_50K']=(chunk.PINCP_ADJ>50_000).astype('int8')
                real_thr=50_000 if year==2018 else REAL_2024_THRESHOLD
                chunk['TARGET_REAL_2018_50K']=(chunk.PINCP_ADJ>real_thr).astype('int8')
                outcols=['YEAR','SOURCE_PART','SERIALNO','SPORDER','STATE','PWGTP','ADJINC','ADJINC_FACTOR']+FEATURES+['RELP_RAW','PINCP_RAW','PINCP_ADJ','PINCP','TARGET_NOMINAL_50K','TARGET_REAL_2018_50K']
                chunk[outcols].to_csv(dest,index=False,header=first)
                first=False
                pos_nom += int(chunk.TARGET_NOMINAL_50K.sum()); pos_real += int(chunk.TARGET_REAL_2018_50K.sum())
                weight_sum += float(chunk.PWGTP.sum())
                weight_pos_nom += float(chunk.loc[chunk.TARGET_NOMINAL_50K==1,'PWGTP'].sum())
                weight_pos_real += float(chunk.loc[chunk.TARGET_REAL_2018_50K==1,'PWGTP'].sum())
                print(f'{zip_path.name}: raw={n_raw:,} filtered={n_filtered:,}',flush=True)
    return {'zip':str(zip_path),'zip_sha256':sha256(zip_path),'year':year,'part':part,
            'output_file':output_path.name,'output_sha256':sha256(output_path),'n_raw':n_raw,'n_filtered':n_filtered,
            'positive_nominal':pos_nom,'prevalence_nominal':pos_nom/n_filtered,'positive_real':pos_real,
            'prevalence_real':pos_real/n_filtered,'weight_sum':weight_sum,
            'weighted_prevalence_nominal':weight_pos_nom/weight_sum,'weighted_prevalence_real':weight_pos_real/weight_sum,
            'raw_categories':{k:sorted(v) for k,v in categories.items()}}


def combine(parts:list[dict],year:int)->dict:
    x=[r for r in parts if r['year']==year]; n=sum(r['n_filtered'] for r in x); raw=sum(r['n_raw'] for r in x)
    ws=sum(r['weight_sum'] for r in x)
    return {'year':year,'n_raw':raw,'n_filtered':n,
            'positive_nominal':sum(r['positive_nominal'] for r in x),
            'prevalence_nominal':sum(r['positive_nominal'] for r in x)/n,
            'positive_real':sum(r['positive_real'] for r in x),'prevalence_real':sum(r['positive_real'] for r in x)/n,
            'weight_sum':ws,
            'weighted_prevalence_nominal':sum(r['weighted_prevalence_nominal']*r['weight_sum'] for r in x)/ws,
            'weighted_prevalence_real':sum(r['weighted_prevalence_real']*r['weight_sum'] for r in x)/ws}


def main()->None:
    ap=argparse.ArgumentParser()
    for name in ['zip_2018_a','zip_2018_b','zip_2024_a','zip_2024_b','dict_2018','dict_2024']:
        ap.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--chunksize',type=int,default=500000)
    args=ap.parse_args(); t0=time.time(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    dictionary_audit(parse_dictionary(args.dict_2018),parse_dictionary(args.dict_2024)).to_csv(out/'acs_income_dictionary_harmonization_audit.csv',index=False)
    specs=[(args.zip_2018_a,2018,'A'),(args.zip_2018_b,2018,'B'),(args.zip_2024_a,2024,'A'),(args.zip_2024_b,2024,'B')]
    parts=[]
    for z,year,part in specs:
        parts.append(process_zip(z,year,part,out/f'acs_income_{year}_{part.lower()}_harmonized.csv.gz',args.chunksize))
    years=[combine(parts,2018),combine(parts,2024)]
    pd.DataFrame(years).to_csv(out/'acs_income_year_summary.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='raw_categories'} for r in parts]).to_csv(out/'acs_income_part_audit.csv',index=False)
    ca=[]
    for v18,v24 in [('COW','COW'),('SCHL','SCHL'),('MAR','MAR'),('OCCP','OCCP'),('POBP','POBP'),('RELP','RELSHIPP'),('SEX','SEX'),('RAC1P','RAC1P')]:
        s18=set().union(*[set(r['raw_categories'].get(v18,[])) for r in parts if r['year']==2018])
        s24=set().union(*[set(r['raw_categories'].get(v24,[])) for r in parts if r['year']==2024])
        ca.append({'variable_2018':v18,'variable_2024':v24,'n_codes_2018':len(s18),'n_codes_2024':len(s24),
                   'n_shared_codes':len(s18&s24),'n_2018_only':len(s18-s24),'n_2024_only':len(s24-s18),
                   'codes_2018_only':' '.join(map(str,sorted(s18-s24))),
                   'codes_2024_only':' '.join(map(str,sorted(s24-s18)))})
    pd.DataFrame(ca).to_csv(out/'acs_income_observed_code_audit.csv',index=False)
    pd.DataFrame({'feature':FEATURES,'type':['numeric' if f in NUMERIC else 'categorical' for f in FEATURES],
                  'semantic_harmonization':['coarse RELP/RELSHIPP crosswalk' if f=='RELP_HARM' else 'same concept; dictionary and observed-code audit retained' for f in FEATURES]}).to_csv(out/'acs_income_feature_registry.csv',index=False)
    audit={'features':FEATURES,'numeric_features':NUMERIC,'categorical_features':CATEGORICAL,
           'adult_filter':'AGEP > 16; ADJINC-adjusted PINCP > 100; WKHP > 0; PWGTP >= 1',
           'income_adjustment':'PINCP_ADJ = PINCP_RAW * ADJINC / 1,000,000 before filtering and target construction','nominal_target':'PINCP_ADJ > 50000 in each survey year dollars','real_target_2018':'PINCP_ADJ > 50000',
           'real_target_2024':f'PINCP_ADJ > {REAL_2024_THRESHOLD:.6f}',
           'cpi_u_annual_average_2018':CPI_2018,'cpi_u_annual_average_2024':CPI_2024,
           'relationship_harmonization':'coarse semantic crosswalk from 2018 RELP to 2024 RELSHIPP',
           'part_summaries':[{k:v for k,v in r.items() if k!='raw_categories'} for r in parts],
           'year_summaries':years,'runtime_seconds':time.time()-t0}
    (out/'acs_income_preparation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
