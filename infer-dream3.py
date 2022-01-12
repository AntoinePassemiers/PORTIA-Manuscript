# -*- coding: utf-8 -*-
# infer-dream3.py
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
    'portia', 'eteportia', 'ntportia', 'zscores',
    'genie3', 'ennet', 'tigress', 'nimefi', 'aracneap', 'plsnet']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=METHODS,
                        help='GRN inference method')
    parser.add_argument('-noko', action='store_true',
                    help='Whether to remove KO experiments')
    args = parser.parse_args()

    synapse = synapseclient.Synapse()
    username, password = get_synapse_credentials()
    synapse.login(username, password)

    entity = synapse.get('syn2853601')
    zip_obj = zipfile.ZipFile(entity.path, 'r')

    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FOLDER = os.path.join(ROOT, 'inferred-networks')
    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for net_id in ['Ecoli1', 'Ecoli2', 'Yeast1', 'Yeast2', 'Yeast3']:

        content = zip_obj.read(f'InSilicoSize100/InSilicoSize100-{net_id}-heterozygous.tsv')
        df1 = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        df1.drop(['strain'], axis=1, inplace=True)
        gene_names = df1.columns.to_numpy()
        heterozygous = df1.to_numpy()

        content = zip_obj.read(f'InSilicoSize100/InSilicoSize100-{net_id}-trajectories.tsv')
        df2 = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        t = df2['Time'].to_numpy()
        df2.drop(['Time'], axis=1, inplace=True)
        trajectories = df2.to_numpy()
        idx = np.where(np.diff(t) < 0)[0] + 1
        timeseries = np.split(trajectories, idx, axis=0)
        times = np.split(t, idx, axis=0)

        content = zip_obj.read(f'InSilicoSize100/InSilicoSize100-{net_id}-null-mutants.tsv')
        df3 = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        df3.drop(['strain'], axis=1, inplace=True)
        null_mutants = df3.to_numpy()

        dataset = pt.GeneExpressionDataset()
        k = 0
        dataset.add(pt.Experiment(k, null_mutants[0]))
        k += 1
        dataset.add(pt.Experiment(k, heterozygous[0]))
        k += 1
        for t, series in zip(times, timeseries):
            for i in range(len(t)):
                dataset.add(pt.Experiment(k, series[i], time=t[i]))
            k += 1
        if not args.noko:
            for i in range(1, len(null_mutants)):
                dataset.add(pt.Experiment(k, null_mutants[i], knockout=[i - 1]))
                k += 1
        for i in range(1, len(heterozygous)):
            dataset.add(pt.Experiment(k, heterozygous[i], knockdown=[i - 1]))
            k += 1

        t0 = time.time()

        if args.method == 'portia':
            M_bar = pt.run(dataset, method='fast', lambda2=0.005)
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
            df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
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
        np.fill_diagonal(M_bar, 0)
        assert not np.any(M_bar < 0)

        print('Running time: %f seconds' % (time.time() - t0))

        # Rank and store results
        folder = os.path.join(OUTPUT_FOLDER, 'dream3', args.method)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if args.noko:
            filepath = os.path.join(folder, f'noko.{net_id}.txt')
        else:
            filepath = os.path.join(folder, f'{net_id}.txt')
        print(f'Writing output to {filepath}')
        with open(filepath, 'w') as f:
            for gene_a, gene_b, score in pt.rank_scores(M_bar, gene_names):
                f.write(f'{gene_a}\t{gene_b}\t{score}\n')

    zip_obj.close()


if __name__ == '__main__':
    main()
