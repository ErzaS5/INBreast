"""Read-only source hashes and comparison to the completed DICOM audit."""
from __future__ import annotations
import argparse
from pathlib import Path, PureWindowsPath
import pandas as pd
from dicom_audit import sha256
from reporting import write_json


def verify_sources(root,output,previous_audit=None):
    files=sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.dcm','.dicom','.xml','.xls','.xlsx','.csv'})
    rows=[{'relative_path':str(p.relative_to(root)),'size':p.stat().st_size,'sha256':sha256(p)} for p in files]
    output.mkdir(parents=True,exist_ok=True)
    target=output/'source_hashes.csv'
    unchanged=None
    if target.is_file():
        old=pd.read_csv(target).set_index('relative_path').sha256.to_dict()
        current={row['relative_path']:row['sha256'] for row in rows}
        unchanged=old==current
        if not unchanged:
            raise ValueError('Originalni source fajlovi/hash manifest se razlikuju od prethodne provere.')
    pd.DataFrame(rows).to_csv(target,index=False)
    matching=0
    if previous_audit and previous_audit.is_file():
        previous=pd.read_csv(previous_audit)
        known={PureWindowsPath(str(row.path)).name:str(row.sha256) for row in previous.itertuples() if str(row.sha256)!='nan'}
        for row in rows:
            filename=Path(row['relative_path']).name
            if filename in known:
                if known[filename]!=row['sha256']:
                    raise ValueError(f'Kanonski DICOM ne odgovara prethodnom auditu: {filename}')
                matching+=1
    summary={'data_root':str(root.resolve()),'source_files':len(rows),'unchanged_since_manifest':unchanged,
             'dicoms_matching_prior_audit':matching,'read_only':True,'original_dataset_modified':False}
    write_json(output/'source_verification.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/verification'))
    parser.add_argument('--previous-audit',type=Path,default=Path('artifacts/dicom_audit/duplicate_candidates.csv'))
    args=parser.parse_args()
    print(verify_sources(args.data_root,args.output_dir,args.previous_audit))
