# -*- coding: utf-8 -*-
# evaluate-merlin-p.py
# author: Antoine Passemiers

import matplotlib.pyplot as plt

from evalportia.gt import graph_theoretic_evaluation
from evalportia.metrics import *
from evalportia.plot import plot_fp_types, plot_matrix_symmetry
from evalportia.tools import *
from evalportia.utils.latex import *


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data', 'merlin-p_inferred_networks-master')
NETWORKS_PATH = os.path.join(ROOT, 'inferred-networks', 'merlin-p')
EVAL_TMP_PATH = os.path.join(ROOT, 'eval-tmp', 'merlin-p')
if not os.path.isdir(EVAL_TMP_PATH):
    os.makedirs(EVAL_TMP_PATH)
TABLES_PATH = os.path.join(ROOT, 'tables')
if not os.path.isdir(TABLES_PATH):
    os.makedirs(TABLES_PATH)
FIGURES_PATH = os.path.join(ROOT, 'figures', 'merlin-p')
if not os.path.isdir(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)
PDF_DATA_PATH = os.path.join(ROOT, 'pdf-data', 'merlin-p')
if not os.path.isdir(PDF_DATA_PATH):
    os.makedirs(PDF_DATA_PATH)


DATASETS = [
    {
        'name': 'niu',
        'expr-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'expression', 'Niu.txt'),
        'gs': 'Cusanovich',
        'gs-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'gold', 'Cusanovich_gold.txt'),
        'tf-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'expression', 'TF_names.txt')
    },
    {
        'name': 'Geuvadis',
        'expr-location': os.path.join(DATA_FOLDER, 'LCL_networks', 'expression', 'Geuvadis.txt'),
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

    method_names = os.listdir(NETWORKS_PATH)
    evaluations = {method_name: {'gt': [], 'scores': [], 'aurocs': [], 'auprcs': [], 'symmetry': [], 'target-symmetry': []} for method_name in method_names}

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
        gs_filepath = dataset_info['gs-location']
        A = GRN.load_goldstandard(gs_filepath, gene_names).asarray()

        # Subset
        mask = np.logical_or(np.any(A == 1, axis=0), np.any(A == 1, axis=1))
        A = A[:, mask][mask, :]
        X = X[:, mask]
        gene_names = gene_names[mask]
        df = df[gene_names]

        print(f'Number of experimentally verified interactions: {int(np.nansum(A))}')

        # Load TF names
        filepath = dataset_info['tf-location']
        _tf_names = set(load_tf_names(filepath)).intersection(gene_names)
        for tf_name in _tf_names:
            assert tf_name in gene_names
        tf_names = []
        _gene_names = set(gene_names)
        for tf_name in _tf_names:
            if tf_name in _gene_names:
                tf_names.append(tf_name)

        n_samples = X.shape[0]
        n_genes = X.shape[1]
        print(f'Number of observations: {n_samples}')
        print(f'Number of genes: {n_genes}')
        tf_idx = np.where([gene_name in tf_names for gene_name in gene_names])[0]
        assert len(tf_names) == len(tf_idx)
        print(f'Number of regulators: {len(tf_idx)}')

        G_target = GRN(A, tf_idx)
        G_target.add_negatives_inplace()

        filepath = os.path.join(PDF_DATA_PATH, f'{net_id}-{goldstandard}.npz')
        if not os.path.exists(filepath):
            pdf_data = compute_area_distributions(G_target, n_networks=25000)
            np.savez(filepath, **pdf_data)
        else:
            pdf_data = np.load(filepath)

        for method_name in method_names:
            filepath = os.path.join(NETWORKS_PATH, method_name, f'{net_id}-{goldstandard}.txt')
            G_pred = GRN.load_network(filepath, gene_names, tf_idx)
            metrics = compute_metrics(G_target, G_pred, pdf_data)
            evaluations[method_name]['scores'].append(metrics['score'])
            evaluations[method_name]['aurocs'].append(metrics['auroc'])
            evaluations[method_name]['auprcs'].append(metrics['auprc'])
            # print(method_name, evaluations[method_name])

            tf_mask = np.zeros(n_genes, dtype=bool)
            tf_mask[tf_idx] = 1
            tmp_filepath = os.path.join(EVAL_TMP_PATH, f'{net_id}-{goldstandard}.npz')
            res = graph_theoretic_evaluation(tmp_filepath, G_target, G_pred, tf_mask)
            res['G-target'] = G_target
            res['G-pred'] = G_pred
            res['net-id'] = net_id
            res['goldstandard'] = goldstandard
            evaluations[method_name]['gt'].append(res)
            evaluations[method_name]['symmetry'].append(matrix_symmetry(G_pred))
            evaluations[method_name]['target-symmetry'].append(matrix_symmetry(G_target))

    for method_name in evaluations.keys():
        evaluations[method_name]['overall-score'] = np.mean(evaluations[method_name]['scores'])

    def compute_table_row(method_name, key):
        values = [method_name]
        values.append(np.mean(evaluations[key]['auprcs'][0]))
        values.append(np.mean(evaluations[key]['aurocs'][0]))
        values.append(np.mean(evaluations[key]['auprcs'][1]))
        values.append(np.mean(evaluations[key]['aurocs'][1]))
        values.append(np.mean(evaluations[key]['auprcs'][2:5]))
        values.append(np.mean(evaluations[key]['aurocs'][2:5]))
        values.append(np.mean(evaluations[key]['auprcs'][5:8]))
        values.append(np.mean(evaluations[key]['aurocs'][5:8]))
        values.append(np.mean(evaluations[key]['auprcs'][8:]))
        values.append(np.mean(evaluations[key]['aurocs'][8:]))
        values.append(evaluations[key]['overall-score'])
        return values

    # Generate performance table
    print('Generating LaTeX table...')
    caption = 'ROC-AUC scores of different GRN inference methods on 3 yeast expression datasets and a LCL dataset from MERLIN-P, evaluated on 3 and 2 goldstandard networks, respectively.'
    table = LaTeXTable(caption, 'tab:merlin-p-benchmark')
    table.add_column(MultiColumn('Method', dtype=str, alignment='l'))
    table.add_column(MultiColumn(f'LCL (Niu)', ['AUPR', 'AUROC'], dtype=float))
    table.add_column(MultiColumn(f'LCL (Geuvadis)', ['AUPR', 'AUROC'], dtype=float))
    table.add_column(MultiColumn(f'NatVar (Average)', ['AUPR', 'AUROC'], dtype=float))
    table.add_column(MultiColumn(f'KO (Average)', ['AUPR', 'AUROC'], dtype=float))
    table.add_column(MultiColumn(f'Stress (Average)', ['AUPR', 'AUROC'], dtype=float))
    table.add_column(MultiColumn('Overall score', dtype=float, alignment='r'))
    table.add_row_values(compute_table_row('ARACNe-AP', 'aracneap'))
    table.add_row_values(compute_table_row('GENIE3', 'genie3'))
    table.add_row_values(compute_table_row('PLSNET', 'plsnet'))
    table.add_row_values(compute_table_row('TIGRESS', 'tigress'))
    table.add_row_values(compute_table_row('ENNET', 'ennet'))
    table.add_midrule()
    table.add_row_values(compute_table_row('\\fastmethodname', 'portia'))
    table.add_row_values(compute_table_row('\\methodname', 'eteportia'))
    with open(os.path.join(TABLES_PATH, 'merlin-p.tex'), 'w') as f:
        f.write(str(table))
    print(table)

    caption = 'Proportions of false positives made on the MERLIN-P datasets, categorised according to the local causal structure in which they occured, for all methods.'
    method_keys = ['aracneap', 'genie3', 'plsnet', 'tigress', 'ennet', 'portia', 'eteportia']
    _method_names = ['ARACNe-AP', 'GENIE3', 'PLSNET', 'TIGRESS', 'ENNET', 'PORTIA', 'etePORTIA']
    net_names = []
    for dataset_info in DATASETS:
        s = dataset_info["gs"].replace("_", "\\_")
        net_names.append(f'{dataset_info["name"]} ({s})')
    label = 'tab:fp-categories-merlinp'
    table = create_fp_table(evaluations, _method_names, method_keys, net_names, label, caption)
    with open(os.path.join(TABLES_PATH, 'merlinp-fp-categories.tex'), 'w') as f:
        f.write(str(table))

    for method_name in method_names:
        ndcgs = []
        ndcgs.append(np.mean([evaluations[method_name]['gt'][i]['score'] for i in [0, 1]]))
        print(f'LCL - NDCG of {method_name}: {ndcgs}')
        ndcgs = []
        ndcgs.append(np.mean([evaluations[method_name]['gt'][i]['score'] for i in [2, 5, 8]]))
        ndcgs.append(np.mean([evaluations[method_name]['gt'][i]['score'] for i in [3, 6, 9]]))
        ndcgs.append(np.mean([evaluations[method_name]['gt'][i]['score'] for i in [4, 7, 10]]))
        print(f'Yeast - NDCG of {method_name}: {ndcgs}')

    # Generate figures
    print('Generating figures...')

    values = []
    for method_name in ['genie3', 'aracneap', 'tigress', 'plsnet', 'ennet', 'portia', 'eteportia']:
        values.append(evaluations[method_name]['symmetry'])
    values.append(evaluations['portia']['target-symmetry'])
    _method_names = ['GENIE3', 'ARACNe-AP', 'TIGRESS', 'PLSNET', 'ENNET', 'PORTIA', 'etePORTIA', 'Goldstandard']
    title = 'Symmetry of inferred GRNs (MERLIN-P)'
    color_idx = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    net_names = ['LCL', None, 'NatVar', None, None, 'KO', None, None, 'StressResp', None, None]
    plot_matrix_symmetry(values, _method_names, title=title, network_names=net_names, color_idx=color_idx)
    filepath = os.path.join(ROOT, 'figures', f'symmetry-merlin-p.eps')
    plt.savefig(filepath, dpi=300, transparent=True)
    plt.close()

    for method_name in method_names:
        for res in evaluations[method_name]['gt']:
            net_id = res['net-id']
            goldstandard = res['goldstandard']
            ax = plt.subplot(1, 1, 1)
            plot_fp_types(ax, res['G-target'], res['G-pred'], res['T'], n_pred=res['G-target'].n_edges)
            plt.title(f'Score: {res["score"]}')
            filepath = os.path.join(FIGURES_PATH, f'{method_name}-{net_id}-{goldstandard}.png')
            plt.savefig(filepath)
            plt.close()


if __name__ == '__main__':
    main()
