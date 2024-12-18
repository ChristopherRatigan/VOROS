import argparse
import os
import re
import numpy as np
import pandas as pd

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='label_BUSI.py')
    parser.add_argument('--directory', type=str, help='Directory where encoded images are saved', required=True)
    args = parser.parse_args()
    file_index = 0
    folder_df = pd.DataFrame(columns=['label', 'person_id', 'path'])
    for subdir, dirs, files in os.walk(args.directory):
        for file in files:
            if re.search('.pt$', file):
                label = re.split('\/', subdir)[-1]
                assert label in ['benign', 'malignant', 'normal'], 'Unexpected label: {}'.format(label)
                path = os.path.join(subdir, file)
                person_id = re.findall('\((\d+)\)', file)
                assert len(person_id) == 1, 'Unexpected file: {}'.format(file)
                folder_df.loc[file_index] = [label, person_id[0], path]
                file_index += 1
    # Sort df and add 'study_id' column
    folder_df['person_id'] = pd.to_numeric(folder_df['person_id'])
    folder_df = folder_df.sort_values(['label', 'person_id'])
    mask = folder_df.duplicated(['label', 'person_id'])
    study_id = mask.copy()
    study_id[~mask] = np.arange(sum(~mask))
    study_id[mask] = np.nan
    folder_df['study_id'] = study_id
    folder_df.study_id = folder_df.study_id.ffill()
    # Write .csv to directory
    temp_df = folder_df[['study_id', 'label', 'path']].set_index('study_id')
    temp_df.label = temp_df.label.apply(lambda label: [1,0,0] if label == 'normal' else [0,1,0] if label == 'benign' else [0,0,1])
    temp_df.to_csv(os.path.join(args.directory, 'labels.csv'))