# -*- coding: utf-8 -*-
# infer-dream4.py
# author: Antoine Passemiers

import argparse
import zipfile
import time
import io

import numpy as np
import synapseclient

import portia as pt
from evalportia.data import get_synapse_credentials
from evalportia.tools import *


METHODS = [
    'portia', 'eteportia', 'ntportia', 'zscores', 'genie3',
    'ennet', 'tigress', 'nimefi', 'aracneap', 'plsnet']


def main():
    synapse = synapseclient.Synapse()
    username, password = get_synapse_credentials()
    synapse.login(username, password)

    entity = synapse.get('syn3049733')
    zip_obj = zipfile.ZipFile(entity.path, 'r')

    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=METHODS,
                        help='GRN inference method')
    parser.add_argument('-noko', action='store_true',
                    help='Whether to remove KO experiments')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FOLDER = os.path.join(ROOT, 'inferred-networks')
    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    running_times = []
    for net_id in ['1', '2', '3', '4', '5']:

        content = zip_obj.read(f'insilico_size100_{net_id}/insilico_size100_{net_id}_knockdowns.tsv')
        df1 = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        gene_names = df1.columns.to_numpy()
        knockdowns = df1.to_numpy()

        content = zip_obj.read(f'insilico_size100_{net_id}/insilico_size100_{net_id}_knockouts.tsv')
        df2 = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        knockouts = df2.to_numpy()

        content = zip_obj.read(f'insilico_size100_{net_id}/insilico_size100_{net_id}_timeseries.tsv')
        df3 = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        t = df3['Time'].to_numpy()
        df3.drop(['Time'], axis=1, inplace=True)
        timeseries = df3.to_numpy()
        idx = np.where(np.diff(t) < 0)[0] + 1
        timeseries = np.split(timeseries, idx, axis=0)
        times = np.split(t, idx, axis=0)

        content = zip_obj.read(f'insilico_size100_{net_id}/insilico_size100_{net_id}_wildtype.tsv')
        df4 = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        wildtype = np.squeeze(df4.to_numpy())

        t0 = time.time()

        dataset = pt.GeneExpressionDataset()
        k = 0
        dataset.add(pt.Experiment(k, wildtype))
        k += 1
        for t, series in zip(times, timeseries):
            for i in range(len(t)):
               dataset.add(pt.Experiment(k, series[i], time=t[i]))
            k += 1
        for i in range(len(knockdowns)):
            dataset.add(pt.Experiment(k, knockdowns[i], knockdown=[i]))
            k += 1
        if not args.noko:
            for i in range(len(knockouts)):
                dataset.add(pt.Experiment(k, knockouts[i], knockout=[i]))
                k += 1

        if args.method == 'portia':
            M_bar = pt.run(dataset, method='fast')
        elif args.method == 'eteportia':
            M_bar = pt.run(dataset, method='end-to-end', verbose=True)
        elif args.method == 'ntportia':
            M_bar = pt.run(dataset, method='no-transform')
        elif args.method == 'zscores':
            M_bar = dataset.compute_null_mutant_zscores()
        elif args.method == 'genie3':
            X = np.asarray(dataset.X)
            M_bar = get_genie3()(X)
        elif args.method == 'ennet':
            X = np.asarray(dataset.X)
            K = np.asarray(dataset.K)
            M_bar = get_ennet()(X, K=K)
        elif args.method == 'tigress':
            df = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
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
        print('Running time: %f seconds' % running_times[-1])

        # Rank and store results
        folder = os.path.join(OUTPUT_FOLDER, 'dream4', args.method)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if args.noko:
            filepath = os.path.join(folder, f'noko.{net_id}.txt')
        else:
            filepath = os.path.join(folder, f'{net_id}.txt')
        with open(filepath, 'w') as f:
            for gene_a, gene_b, score in pt.rank_scores(M_bar, gene_names):
                f.write(f'{gene_a}\t{gene_b}\t{score}\n')

    print('Average running time: %f seconds' % np.mean(running_times))


if __name__ == '__main__':
    main()
