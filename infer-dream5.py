# -*- coding: utf-8 -*-
# infer-dream5.py
# author: Antoine Passemiers

import argparse
import time

import portia as pt
import synapseclient

from evalportia.data import get_synapse_credentials
from evalportia.tools import *


METHODS = [
    'portia', 'eteportia', 'ntportia', 'genie3', 'ennet',
    'tigress', 'aracneap', 'plsnet']


def main():
    synapse = synapseclient.Synapse()
    username, password = get_synapse_credentials()
    synapse.login(username, password)

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

    synapse_ids = {
        '1': {
            'expression': 'syn2787226',
            'tfs': 'syn2787227',
            'metadata': 'syn2787225'
        },
        '3': {
            'expression': 'syn2787234',
            'tfs': 'syn2787235',
            'metadata': 'syn2787233'
        },
        '4': {
            'expression': 'syn2787238',
            'tfs': 'syn2787239',
            'metadata': 'syn2787237'
        }
    }

    for net_id in ['1', '3', '4']:  # TODO

        synapse_id_expression = synapse_ids[net_id]['expression']
        synapse_id_tfs = synapse_ids[net_id]['tfs']
        synapse_id_metadata = synapse_ids[net_id]['metadata']

        entity = synapse.get(synapse_id_tfs)
        with open(entity.path, 'r') as f:
            tfs = set()
            for line in f.readlines():
                line = line.rstrip().lstrip()
                if len(line) > 0:
                    tfs.add(line)

        entity = synapse.get(synapse_id_expression)
        df = pd.read_csv(entity.path, delimiter='\t', header='infer')
        gene_names = df.columns.to_numpy()
        gene_dict = {name: i for i, name in enumerate(gene_names)}
        X = df.to_numpy()
        n_samples = X.shape[0]
        tf_idx = np.asarray([i for i in range(len(gene_names)) if gene_names[i] in tfs])
        tf_names = gene_names[tf_idx]

        dataset = pt.GeneExpressionDataset()
        entity = synapse.get(synapse_id_metadata)
        df2 = pd.read_csv(entity.path, delimiter='\t', header='infer', na_values='NA')
        for i in range(n_samples):
            row = df2.iloc[i]
            _id = str(row['#Experiment']) + '-' + str(row['Repeat']) + '-' + str(row['Perturbations']) + '-' + str(row['PerturbationLevels']) + '-' + str(row['Treatment']) + '-' + str(row['DeletedGenes']) + '-' + str(row['OverexpressedGenes'])
            expression = X[i, :]
            knockout = [] if not isinstance(row['DeletedGenes'], str) else row['DeletedGenes'].split(',')
            knockout = [gene_dict[name] for name in knockout]
            t = 0 if np.isnan(row['Time']) else float(row['Time'])
            if (not args.noko) or (len(knockout) == 0):
                dataset.add(pt.Experiment(_id, expression, knockout=knockout, time=t))

        t0 = time.time()

        if args.method == 'portia':
            M_bar = pt.run(dataset, tf_idx=tf_idx, method='fast')
        elif args.method == 'eteportia':
            M_bar = pt.run(dataset, tf_idx=tf_idx, method='end-to-end', verbose=True)
        elif args.method == 'ntportia':
            M_bar = pt.run(dataset, tf_idx=tf_idx, method='no-transform')
        elif args.method == 'genie3':
            X = np.asarray(dataset.X)
            M_bar = get_genie3()(X, tf_idx=tf_idx)
        elif args.method == 'ennet':
            X = np.asarray(dataset.X)
            M_bar = get_ennet()(X, tf_idx=tf_idx)
        elif args.method == 'tigress':
            n_split = 4000 if (net_id == '1') else 1000
            M_bar = get_tigress()(df, tf_names=tf_names, nsplit=n_split)
        elif args.method == 'aracneap':
            X = np.asarray(dataset.X)
            M_bar = get_aracne_ap()(X, tf_idx=tf_idx)
        elif args.method == 'plsnet':
            X = np.asarray(dataset.X)
            M_bar = get_plsnet()(X, tf_idx=tf_idx)
        else:
            raise NotImplementedError()

        print('Running time: %f seconds' % (time.time() - t0))

        # Rank and store results
        folder = os.path.join(OUTPUT_FOLDER, 'dream5', args.method)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if args.noko:
            filepath = os.path.join(folder, f'noko.{net_id}.txt')
        else:
            filepath = os.path.join(folder, f'{net_id}.txt')
        with open(filepath, 'w') as f:
            for gene_a, gene_b, score in pt.rank_scores(M_bar, gene_names, limit=100000):
                f.write(f'{gene_a}\t{gene_b}\t{score}\n')


if __name__ == '__main__':
    main()
