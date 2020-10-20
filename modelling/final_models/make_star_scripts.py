import pandas as pd
import os
import warnings

path = '/rds/projects/d/daviesgr-alex-phd/kepler-dwarfs/data/stellr/inputs_DR14_ASPC.csv'

df = pd.read_csv(path)
df = df.loc[(df['bad_data']==0) & (df['on_grid_1e']==1)]  # Run only good data and on_grid stars
n_stars = len(df)

with open('template.sh') as fin:
    template = fin.read()

save_dir = 'star_scripts'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
elif len(os.listdir(save_dir)) > 0:
    warnings.warn(f'Directory {save_dir} is not empty, consider removing ' +
                  'unwanted files and rerunning this script.')

for i, row in df.iterrows():
    sr = template.replace('NAME', row['name']) 
    sr = sr.replace('PY_PATH', os.path.join(os.getcwd(), 'run_star.py'))

    with open(f'{save_dir}/star_{row["name"]}.sh', 'w') as file:
        file.write(sr)
