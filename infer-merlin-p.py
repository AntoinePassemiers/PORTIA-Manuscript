# -*- coding: utf-8 -*-
# run-portia-on-merlin-p.py
# author: Antoine Passemiers

import argparse
import time

import portia as pt

from portia.gt.grn import GRN
from evalportia.tools import *

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data', 'merlin-p_inferred_networks-master')
OUTPUT_FOLDER = os.path.join(ROOT, 'inferred-networks')


METHODS = ['portia', 'eteportia', 'zscores', 'genie3', 'ennet', 'tigress', 'nimefi', 'aracneap', 'plsnet']


DATASETS = [
    {
        'name': 'Geuvadis',
        'expr-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'expression', 'Geuvadis.txt'),
        'gs': 'Cusanovich',
        'gs-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'gold', 'Cusanovich_gold.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'expression', 'TF_names.txt')
    },
    {
        'name': 'niu',
        'expr-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'expression', 'Niu.txt'),
        'gs': 'Cusanovich',
        'gs-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'gold', 'Cusanovich_gold.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'expression', 'TF_names.txt')
    },
    {
        'name': 'NatVar',
        'expr-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'NatVar.txt'),
        'gs': 'MacIsaac2',
        'gs-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'gold', 'MacIsaac2.NatVar.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'NatVar_TF_names.txt')
    },
    {
        'name': 'NatVar',
        'expr-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'NatVar.txt'),
        'gs': 'YEASTRACT_Count3',
        'gs-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'gold', 'YEASTRACT_Count3.NatVar.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'NatVar_TF_names.txt')
    },
    {
        'name': 'NatVar',
        'expr-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'NatVar.txt'),
        'gs': 'YEASTRACT_Type2',
        'gs-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'gold', 'YEASTRACT_Type2.NatVar.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'NatVar_TF_names.txt')
    },
    {
        'name': 'KO',
        'expr-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'KO.txt'),
        'gs': 'MacIsaac2',
        'gs-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'gold', 'MacIsaac2.KO.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'KO_TF_names.txt')
    },
    {
        'name': 'KO',
        'expr-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'KO.txt'),
        'gs': 'YEASTRACT_Count3',
        'gs-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'gold', 'YEASTRACT_Count3.KO.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'KO_TF_names.txt')
    },
    {
        'name': 'KO',
        'expr-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'KO.txt'),
        'gs': 'YEASTRACT_Type2',
        'gs-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'gold', 'YEASTRACT_Type2.KO.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'KO_TF_names.txt')
    },
    {
        'name': 'Stress',
        'expr-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'Stress.txt'),
        'gs': 'MacIsaac2',
        'gs-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'gold', 'MacIsaac2.Stress.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'Stress_TF_names.txt')
    },
    {
        'name': 'Stress',
        'expr-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'Stress.txt'),
        'gs': 'YEASTRACT_Count3',
        'gs-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'gold', 'YEASTRACT_Count3.Stress.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'Stress_TF_names.txt')
    },
    {
        'name': 'Stress',
        'expr-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'Stress.txt'),
        'gs': 'YEASTRACT_Type2',
        'gs-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'gold', 'YEASTRACT_Type2.Stress.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'yeast_networks', 'expression', 'Stress_TF_names.txt')
    },
]


def load_tf_names(filepath):
    tf_names = set()
    with open(filepath, 'r') as f:
        for line in f.readlines():
            tf_names.add(line.rstrip())
    return tf_names


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=METHODS, help='GRN inference method')
    args = parser.parse_args()

    running_times = []
    for dataset_info in DATASETS:

        dataset_name = dataset_info['name']
        goldstandard = dataset_info['gs']
        net_id = dataset_name

        # Load expression data
        df = None
        filepath = dataset_info['expr-location']
        for index_col in ['Gene', 'TargetID', 'Name']:
            try:
                df = pd.read_csv(filepath, sep='\t', index_col=index_col).transpose()
                break
            except ValueError:
                pass
        assert df is not None
        gene_names = df.columns
        X = df.to_numpy()

        # Load goldstandard network
        filepath = dataset_info['gs-location']
        A = GRN.load_goldstandard(filepath, gene_names).asarray()

        # Subset
        mask = np.logical_or(np.any(A == 1, axis=0), np.any(A == 1, axis=1))
        A = A[:, mask][mask, :]
        X = X[:, mask]
        gene_names = gene_names[mask]
        df = df[gene_names]

        print(f'Number of experimentally verified interactions: {int(np.nansum(A))}')

        # Load TF names
        filepath = dataset_info['tf-location']
        tf_names = set(load_tf_names(filepath)).intersection(gene_names)
        for tf_name in tf_names:
            assert tf_name in gene_names

        n_samples = X.shape[0]
        n_genes = X.shape[1]
        print(f'Number of observations: {n_samples}')
        print(f'Number of genes: {n_genes}')
        tf_idx = np.where([gene_name in tf_names for gene_name in gene_names])[0]
        assert len(tf_names) == len(tf_idx)
        print(f'Number of regulators: {len(tf_idx)}')

        dataset = pt.GeneExpressionDataset()
        for exp_id, data in enumerate(X):
            dataset.add(pt.Experiment(exp_id, data))

        t0 = time.time()

        if args.method == 'portia':
            M_bar = pt.run(dataset, tf_idx=tf_idx, method='fast', normalize=False)
        elif args.method == 'eteportia':
            M_bar = pt.run(dataset, tf_idx=tf_idx, method='end-to-end', verbose=True)
        elif args.method == 'genie3':
            X = np.asarray(dataset.X)
            M_bar = get_genie3()(X, tf_idx=tf_idx)
        elif args.method == 'ennet':
            X = np.asarray(dataset.X)
            M_bar = get_ennet()(X, tf_idx=tf_idx)
        elif args.method == 'tigress':
            res = get_tigress()(df, tf_names=tf_names, nsplit=1000)
            M_bar = np.zeros((n_genes, n_genes))
            M_bar[tf_idx, :] = res
        elif args.method == 'aracneap':
            X = np.asarray(dataset.X)
            M_bar = get_aracne_ap()(X, tf_idx=tf_idx)
        elif args.method == 'plsnet':
            X = np.asarray(dataset.X)
            M_bar = get_plsnet()(X, tf_idx=tf_idx)
        else:
            raise NotImplementedError()

        running_times.append(time.time() - t0)
        print('Running time: %f seconds' % running_times[-1])

        # Rank and store results
        folder = os.path.join(OUTPUT_FOLDER, 'merlin-p', args.method)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, f'{net_id}-{goldstandard}.txt')
        with open(filepath, 'w') as f:
            for gene_a, gene_b, score in pt.rank_scores(M_bar, gene_names, limit=100000):
                f.write(f'{gene_a}\t{gene_b}\t{score}\n')

    print(f'Running times: {running_times}')


if __name__ == '__main__':
    main()
