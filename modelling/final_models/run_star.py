# import matplotlib
# matplotlib.use('Agg')

import pandas as pd
import sys
import os
import logging

from stellr import Star

path = '/rds/projects/d/daviesgr-alex-phd/kepler-dwarfs/data/stellr/inputs_DR14_ASPC.csv'

print(sys.argv)
name = sys.argv[1]

df = pd.read_csv(path)

if len(sys.argv) == 2:
    df = df.loc[df['name']==name]

observed = {
    'star': df
}

print(observed)

star = Star(name=name, path='star_results/DR14_ASPC', observed=observed)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', filename=os.path.join(star.savedir, 'star.log'), level=logging.INFO)

sample_kwargs = {
    'burn_in': 1000,
    'num_samples': 1000,
    'num_chains': 10,
    'xla': True,                     # Accelerated linear algebra enabled
    'adaptation_kwargs': {
        'target_accept_prob': 0.98,  # Target acceptance probability
    },
}

star.fit(sample_kwargs=sample_kwargs)
star.plot_diagnostics(save=True)
star.plot_corners(save=True)

print('Finished.')

