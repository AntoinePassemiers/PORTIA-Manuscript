# -*- coding: utf-8 -*-
# infer-dream4-mf.py
# author: Antoine Passemiers

import argparse
import io
import time
import zipfile

import portia as pt
import synapseclient

from evalportia.data import get_synapse_credentials
from evalportia.tools import *


METHODS = [
    'portia', 'eteportia', 'ntportia', 'genie3', 'ennet',
    'nimefi', 'tigress', 'aracneap', 'plsnet']


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=METHODS,
                        help='GRN inference method')
    args = parser.parse_args()

    synapse = synapseclient.Synapse()
    username, password = get_synapse_credentials()
    synapse.login(username, password)

    entity = synapse.get('syn3049734')
    zip_obj = zipfile.ZipFile(entity.path, 'r')

    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FOLDER = os.path.join(ROOT, 'inferred-networks')
    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    running_times = []
    for net_id in ['1', '2', '3', '4', '5']:

        content = zip_obj.read(f'insilico_size100_{net_id}_multifactorial.tsv')
        df = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        gene_names = df.columns.to_numpy()
        X = df.to_numpy()

        dataset = pt.GeneExpressionDataset()
        for i in range(len(X)):
            dataset.add(pt.Experiment(i, X[i, :]))

        t0 = time.time()

        if args.method == 'portia':
            M_bar = pt.run(dataset, method='fast', lambda2=0.02)
        elif args.method == 'eteportia':
            M_bar = pt.run(dataset, method='end-to-end', verbose=False)
        elif args.method == 'ntportia':
            M_bar = pt.run(dataset, method='no-transform')
        elif args.method == 'genie3':
            X = np.asarray(dataset.X)
            M_bar = get_genie3().GENIE3(X, regulators='all', ntrees=1000, nthreads=4)
        elif args.method == 'ennet':
            X = np.asarray(dataset.X)
            M_bar = get_ennet()(X)
        elif args.method == 'tigress':
            M_bar = get_tigress()(df, nsplit=4000)
        elif args.method == 'nimefi':
            X = np.asarray(dataset.X)
            M_bar = get_nimefi()(X)
        elif args.method == 'aracneap':
            X = np.asarray(dataset.X)
            M_bar = get_aracne_ap()(X)
        elif args.method == 'plsnet':
            X = np.asarray(dataset.X)
            M_bar = get_plsnet()(X)
        else:
            raise NotImplementedError()

        running_times.append(time.time() - t0)
        print(f'Running time: {running_times[-1]} seconds')

        # Rank and store results
        folder = os.path.join(OUTPUT_FOLDER, 'dream4-mf', args.method)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, f'{net_id}.txt')
        with open(filepath, 'w') as f:
            for gene_a, gene_b, score in pt.rank_scores(M_bar, gene_names):
                f.write(f'{gene_a}\t{gene_b}\t{score}\n')

    print('Average running time: %f seconds' % np.mean(running_times))


if __name__ == '__main__':
    main()
