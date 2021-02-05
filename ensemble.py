import numpy as np
import pandas as pd
from tqdm import tqdm

results = [pd.read_csv("results/predict-{}.csv".format(i+1)) for i in range(5)]
header = ['Id', 'THEORETICAL', 'ENGINEERING', 'EMPIRICAL', 'OTHERS']
for i in tqdm(range(len(results[0]))):
    row = np.mean([results[j].iloc[i, 1:] for j in range(5)], axis=0)
    row = np.where(row > 0.5, np.ones((1, 4)), np.zeros((1, 4)))
    if i == 0:
        ensemble = row
    else:
        ensemble = np.concatenate((ensemble, row), axis=0)
ensemble = np.concatenate((np.expand_dims(np.arange(1, 40001), axis=1), ensemble), axis=1)
ensemble = pd.DataFrame(ensemble.astype(int))
ensemble.to_csv('results/ensemble.csv', header=header, index=False)
