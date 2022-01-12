# -*- coding: utf-8 -*-
# evaluate-dream3.py
# author: Antoine Passemiers

import io
import os
import zipfile

import matplotlib

from evalportia.grn import GRN

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import synapseclient

from evalportia.data import get_synapse_credentials
from evalportia.metrics import *
from evalportia.gt import graph_theoretic_evaluation
from evalportia.utils.latex import *
from evalportia.plot import plot_fp_types, plot_matrix_symmetry


ROOT = os.path.dirname(os.path.abspath(__file__))
NETWORKS_PATH = os.path.join(ROOT, 'inferred-networks', 'dream3')
EVAL_TMP_PATH = os.path.join(ROOT, 'eval-tmp', 'dream3')
if not os.path.isdir(EVAL_TMP_PATH):
    os.makedirs(EVAL_TMP_PATH)
TABLES_PATH = os.path.join(ROOT, 'tables')
if not os.path.isdir(TABLES_PATH):
    os.makedirs(TABLES_PATH)
FIGURES_PATH = os.path.join(ROOT, 'figures', 'dream3')
if not os.path.isdir(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

synapse_ids = {
    'Ecoli1': {
        'goldstandard': 'syn2853610',
        'pdf': 'syn4558484'
    },
    'Ecoli2': {
        'goldstandard': 'syn2853609',
        'pdf': 'syn4558485'
    },
    'Yeast1': {
        'goldstandard': 'syn2853608',
        'pdf': 'syn4558486'
    },
    'Yeast2': {
        'goldstandard': 'syn2853607',
        'pdf': 'syn4558487'
    },
    'Yeast3': {
        'goldstandard': 'syn2853606',
        'pdf': 'syn4558488'
    }
}


def main():

    synapse = synapseclient.Synapse()
    username, password = get_synapse_credentials()
    synapse.login(username, password)

    entity = synapse.get('syn2853601')
    zip_obj = zipfile.ZipFile(entity.path, 'r')

    method_names = os.listdir(NETWORKS_PATH)
    evaluations = {method_name: {'gt': [], 'scores': [], 'aurocs': [], 'auprcs': [], 'symmetry': [], 'target-symmetry': [], 'scores-noko': [], 'aurocs-noko': [], 'auprcs-noko': []} for method_name in method_names}

    for net_id in ['Ecoli1', 'Ecoli2', 'Yeast1', 'Yeast2', 'Yeast3']:
        content = zip_obj.read(f'InSilicoSize100/InSilicoSize100-{net_id}-heterozygous.tsv')
        df1 = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        df1.drop(['strain'], axis=1, inplace=True)
        gene_names = df1.columns.to_numpy()
        n_genes = len(gene_names)

        entity = synapse.get(synapse_ids[net_id]['goldstandard'])
        gs_filepath = entity.path
        tf_idx = np.arange(n_genes)
        G_target = GRN.load_goldstandard(gs_filepath, gene_names, tf_idx)
        A = G_target.asarray()

        entity = synapse.get(synapse_ids[net_id]['pdf'])
        pdf_data = scipy.io.loadmat(entity.path)
        pdf_data = {
            'x_auroc': pdf_data['x_auroc'],
            'y_auroc': pdf_data['y_auroc'],
            'x_auprc': pdf_data['x_aupr'],
            'y_auprc': pdf_data['y_aupr']
        }

        for method_name in method_names:
            filepath = os.path.join(NETWORKS_PATH, method_name, f'{net_id}.txt')
            G_pred = GRN.load_network(filepath, gene_names, tf_idx)
            M_bar = G_pred.asarray()
            metrics = score_dream_prediction(gs_filepath, filepath, pdf_data, use_test=False)
            evaluations[method_name]['scores'].append(metrics['score'])
            evaluations[method_name]['aurocs'].append(metrics['auroc'])
            evaluations[method_name]['auprcs'].append(metrics['auprc'])
            evaluations[method_name]['symmetry'].append(matrix_symmetry(M_bar))
            evaluations[method_name]['target-symmetry'].append(matrix_symmetry(A))

            tf_mask = np.ones(n_genes, dtype=bool)
            tmp_filepath = os.path.join(EVAL_TMP_PATH, f'{net_id}.npz')
            res = graph_theoretic_evaluation(tmp_filepath, G_target, G_pred, tf_mask)
            res['G-target'] = G_target
            res['G-pred'] = G_pred
            res['net-id'] = net_id
            evaluations[method_name]['gt'].append(res)

            filepath = os.path.join(NETWORKS_PATH, method_name, f'noko.{net_id}.txt')
            if os.path.exists(filepath):
                metrics = score_dream_prediction(gs_filepath, filepath, pdf_data, use_test=False)
                evaluations[method_name]['scores-noko'].append(metrics['score'])
                evaluations[method_name]['aurocs-noko'].append(metrics['auroc'])
                evaluations[method_name]['auprcs-noko'].append(metrics['auprc'])

    for method_name in evaluations.keys():
        evaluations[method_name]['overall-score'] = np.mean(evaluations[method_name]['scores'])
        if len(evaluations[method_name]['scores-noko']) > 0:
            evaluations[method_name]['overall-score-noko'] = np.mean(evaluations[method_name]['scores-noko'])
        else:
            evaluations[method_name]['overall-score-noko'] = 0

    # Generate performance table
    print('Generating LaTeX tables...')
    def compute_table_row(method_name, key):
        values = [method_name]
        for i in range(5):
            values.append(evaluations[key]['auprcs'][i])
            values.append(evaluations[key]['aurocs'][i])
        values.append(evaluations[key]['overall-score-noko'])
        values.append(evaluations[key]['overall-score'])
        return values
    caption = 'AUROC, AUPR and overall scores of different GRN inference methods, evaluated on the 5 networks from DREAM3.'
    table = LaTeXTable(caption, 'tab:dream3-benchmark')
    table.add_column(MultiColumn('Method', dtype=str, alignment='l'))
    for i in range(5):
        table.add_column(MultiColumn(f'Net{i+1}', ['AUPR', 'AUROC'], dtype=float))
    table.add_column(MultiColumn('Overall score (no KO)', dtype=float, alignment='r'))
    table.add_column(MultiColumn('Overall score', dtype=float, alignment='r'))
    table.add_row_values(compute_table_row('ARACNe-AP', 'aracneap'))
    table.add_row_values(compute_table_row('GENIE3', 'genie3'))
    table.add_row_values(compute_table_row('PLSNET', 'plsnet'))
    table.add_row_values(compute_table_row('TIGRESS', 'tigress'))
    table.add_row_values(compute_table_row('ENNET', 'ennet'))
    table.add_midrule()
    table.add_row_values(compute_table_row('Z-scores', 'zscores'))
    table.add_row_values(compute_table_row('\\fastmethodname', 'portia'))
    table.add_row_values(compute_table_row('\\methodname', 'eteportia'))
    with open(os.path.join(TABLES_PATH, 'dream3.tex'), 'w') as f:
        f.write(str(table))
    print(table)

    # Generate performance table (no KO)
    def compute_table_row(method_name, key):
        values = [method_name]
        for i in range(5):
            values.append(evaluations[key]['auprcs-noko'][i])
            values.append(evaluations[key]['aurocs-noko'][i])
        values.append(evaluations[key]['overall-score-noko'])
        return values
    caption = 'AUROC, AUPR and overall scores of different GRN inference methods, evaluated on the 5 networks from DREAM3 (no KO experiment).'
    table = LaTeXTable(caption, 'tab:dream3-noko-benchmark')
    table.add_column(MultiColumn('Method', dtype=str, alignment='l'))
    for i in range(5):
        table.add_column(MultiColumn(f'Net{i+1}', ['AUPR', 'AUROC'], dtype=float))
    table.add_column(MultiColumn('Overall score', dtype=float, alignment='r'))
    table.add_row_values(compute_table_row('ARACNe-AP', 'aracneap'))
    table.add_row_values(compute_table_row('GENIE3', 'genie3'))
    table.add_row_values(compute_table_row('PLSNET', 'plsnet'))
    table.add_row_values(compute_table_row('TIGRESS', 'tigress'))
    table.add_row_values(compute_table_row('ENNET', 'ennet'))
    table.add_midrule()
    table.add_row_values(compute_table_row('\\fastmethodname', 'portia'))
    table.add_row_values(compute_table_row('\\methodname', 'eteportia'))
    with open(os.path.join(TABLES_PATH, 'dream3-noko.tex'), 'w') as f:
        f.write(str(table))

    caption = 'False positives made on \\dreamthree, categorised according to the local causal structure in which they occured, for each method.'
    method_keys = ['aracneap', 'genie3', 'plsnet', 'tigress', 'ennet', 'portia', 'eteportia']
    _method_names = ['ARACNe-AP', 'GENIE3', 'PLSNET', 'TIGRESS', 'ENNET', 'PORTIA', 'etePORTIA']
    net_names = [f'Net{j + 1}' for j in range(5)]
    label = 'tab:fp-categories-dream3'
    table = create_fp_table(evaluations, _method_names, method_keys, net_names, label, caption)
    with open(os.path.join(TABLES_PATH, 'dream3-fp-categories.tex'), 'w') as f:
        f.write(str(table))

    for method_name in method_names:
        ndcg = 0.
        for res in evaluations[method_name]['gt']:
            ndcg += res['score'] / len(evaluations[method_name]['gt'])
        print(f'NDCG of {method_name}: {ndcg}')

    # Generate figures
    print('Generating figures...')
    for method_name in method_names:
        for res in evaluations[method_name]['gt']:
            net_id = res['net-id']
            ax = plt.subplot(1, 1, 1)
            plot_fp_types(ax, res['G-target'], res['G-pred'], res['T'], n_pred=150)
            filepath = os.path.join(FIGURES_PATH, f'{method_name}-{net_id}.png')
            plt.savefig(filepath)
            plt.close()

    values = []
    for method_name in ['genie3', 'aracneap', 'tigress', 'plsnet', 'ennet', 'portia', 'eteportia', 'zscores']:
        values.append(evaluations[method_name]['symmetry'])
    values.append(evaluations['portia']['target-symmetry'])
    method_names = ['GENIE3', 'ARACNe-AP', 'TIGRESS', 'PLSNET', 'ENNET', 'PORTIA', 'etePORTIA', 'Z-scores', 'Goldstandard']
    title = 'Symmetry of inferred GRNs (DREAM3)'
    plot_matrix_symmetry(values, method_names, title=title)
    filepath = os.path.join(ROOT, 'figures', f'symmetry-dream3.eps')
    plt.savefig(filepath, dpi=300, transparent=True)
    plt.close()

    print('Finished.')


if __name__ == '__main__':
    main()
