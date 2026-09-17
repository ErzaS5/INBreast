"""Validate and cache deterministic model inputs without touching source images."""
from pathlib import Path
import argparse
from data import load_metadata_pairs, load_prepared_view
from reporting import write_json


def prepare_cache(metadata, cache, size):
    pairs=load_metadata_pairs(metadata,check_files=True)
    images=roi_views=0
    for row in pairs.to_dict('records'):
        for view in ('cc','mlo'):
            image,mask,geometry=load_prepared_view(row[f'{view}_dicom_path'],row[f'{view}_xml_path'],size,
                                                   return_geometry=True,cache_dir=cache)
            if row[f'{view}_has_roi'] and not mask.any():
                raise ValueError(f'ROI izgubljen nakon preprocessinga: {row["pair_id"]}/{view}')
            images+=1;roi_views+=int(mask.any())
    summary={'pairs':len(pairs),'views':images,'nonempty_roi_views':roi_views,'size':size,'all_inputs_valid':True}
    write_json(cache/f'validation_size_{size}.json',summary)
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--metadata-pairs',type=Path,default=Path('artifacts/metadata_pairs.csv'))
    parser.add_argument('--cache-dir',type=Path,default=Path('artifacts/preprocessed'))
    parser.add_argument('--size',type=int,default=384)
    args=parser.parse_args()
    prepare_cache(args.metadata_pairs,args.cache_dir,args.size)
