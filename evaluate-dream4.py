# -*- coding: utf-8 -*-
# evaluate-dream4.py
# author: Antoine Passemiers

import io
import os
import zipfile

import scipy.io
import matplotlib.pyplot as plt
import synapseclient

from portia.gt import graph_theoretic_evaluation, plot_fp_types
from portia.gt.grn import GRN
from portia.gt.symmetry import matrix_symmetry
from evalportia.data import get_synapse_credentials
from evalportia.metrics import *
from evalportia.utils.latex import *
from evalportia.plot import plot_matrix_symmetry


ROOT = os.path.dirname(os.path.abspath(__file__))
NETWORKS_PATH = os.path.join(ROOT, 'inferred-networks', 'dream4')
EVAL_TMP_PATH = os.path.join(ROOT, 'eval-tmp', 'dream4')
if not os.path.isdir(EVAL_TMP_PATH):
    os.makedirs(EVAL_TMP_PATH)
TABLES_PATH = os.path.join(ROOT, 'tables')
if not os.path.isdir(TABLES_PATH):
    os.makedirs(TABLES_PATH)
FIGURES_PATH = os.path.join(ROOT, 'figures', 'dream4')
if not os.path.isdir(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

synapse_ids = {
    '1': {
        'pdf': 'syn4558445'
    },
    '2': {
        'pdf': 'syn4558446'
    },
    '3': {
        'pdf': 'syn4558447'
    },
    '4': {
        'pdf': 'syn4558448'
    },
    '5': {
        'pdf': 'syn4558449'
    }
}


def main():

    synapse = synapseclient.Synapse()
    username, password = get_synapse_credentials()
    synapse.login(username, password)

    entity = synapse.get('syn3049733')
    zip_obj = zipfile.ZipFile(entity.path, 'r')

    entity = synapse.get('syn3049736')
    zip_obj2 = zipfile.ZipFile(entity.path, 'r')

    method_names = os.listdir(NETWORKS_PATH)
    evaluations = {method_name: {'gt': [], 'scores': [], 'aurocs': [], 'auprcs': [], 'symmetry': [], 'target-symmetry': [], 'scores-noko': [], 'aurocs-noko': [], 'auprcs-noko': []} for method_name in method_names}

    for net_id in ['1', '2', '3', '4', '5']:
        content = zip_obj.read(f'insilico_size100_{net_id}/insilico_size100_{net_id}_knockdowns.tsv')
        df1 = pd.read_csv(io.BytesIO(content), delimiter='\t', header='infer')
        gene_names = df1.columns.to_numpy()
        n_genes = len(gene_names)

        tf_idx = np.arange(n_genes)
        content = zip_obj2.read(f'DREAM4_Challenge2_GoldStandards/Size 100/DREAM4_GoldStandard_InSilico_Size100_{net_id}.tsv')
        G_target = GRN.load_goldstandard(content.decode('ascii'), gene_names, tf_idx, from_string=True)
        A = G_target.asarray()

        entity = synapse.get(synapse_ids[net_id]['pdf'])
        pdf_data = scipy.io.loadmat(entity.path)
        pdf_data = {
            'x_auroc': pdf_data['auroc_X'],
            'y_auroc': pdf_data['auroc_Y'],
            'x_auprc': pdf_data['aupr_X'],
            'y_auprc': pdf_data['aupr_Y']
        }

        for method_name in method_names:
            filepath = os.path.join(NETWORKS_PATH, method_name, f'{net_id}.txt')
            G_pred = GRN.load_network(filepath, gene_names, tf_idx)
            M_bar = G_pred.asarray()
            metrics = score_dream_prediction(io.StringIO(content.decode('ascii')), filepath, pdf_data)
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
                metrics = score_dream_prediction(io.StringIO(content.decode('ascii')), filepath, pdf_data)
                evaluations[method_name]['scores-noko'].append(metrics['score'])
                evaluations[method_name]['aurocs-noko'].append(metrics['auroc'])
                evaluations[method_name]['auprcs-noko'].append(metrics['auprc'])

    for method_name in evaluations.keys():
        print(method_name, evaluations[method_name])
        evaluations[method_name]['overall-score'] = np.mean(evaluations[method_name]['scores'])
        if len(evaluations[method_name]['scores-noko']) > 0:
            evaluations[method_name]['overall-score-noko'] = np.mean(evaluations[method_name]['scores-noko'])
        else:
            evaluations[method_name]['overall-score-noko'] = 0

    # Generate performance table
    print('Generating LaTeX table...')
    def compute_table_row(method_name, key):
        values = [method_name]
        for i in range(5):
            values.append(evaluations[key]['auprcs'][i])
            values.append(evaluations[key]['aurocs'][i])
        values.append(evaluations[key]['overall-score-noko'])
        values.append(evaluations[key]['overall-score'])
        return values
    caption = 'AUROC, AUPR and overall scores of different GRN inference methods, evaluated on the 5 networks from DREAM4.'
    table = LaTeXTable(caption, 'tab:dream4-benchmark')
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
    with open(os.path.join(TABLES_PATH, 'dream4.tex'), 'w') as f:
        f.write(str(table))
    print(table)

    # Generate performance table (no KO)
    print('Generating LaTeX table...')
    def compute_table_row(method_name, key):
        values = [method_name]
        for i in range(5):
            values.append(evaluations[key]['auprcs-noko'][i])
            values.append(evaluations[key]['aurocs-noko'][i])
        values.append(evaluations[key]['overall-score-noko'])
        return values
    caption = 'AUROC, AUPR and overall scores of different GRN inference methods, evaluated on the 5 networks from DREAM4 (with no KO experiment).'
    table = LaTeXTable(caption, 'tab:dream4-noko-benchmark')
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
    with open(os.path.join(TABLES_PATH, 'dream4-noko.tex'), 'w') as f:
        f.write(str(table))

    caption = 'Proportions of false positives made on \\dreamfour, categorised according to the local causal structure in which they occured, for all methods.'
    method_keys = ['aracneap', 'genie3', 'plsnet', 'tigress', 'ennet', 'portia', 'eteportia']
    _method_names = ['ARACNe-AP', 'GENIE3', 'PLSNET', 'TIGRESS', 'ENNET', 'PORTIA', 'etePORTIA']
    net_names = [f'Net{j + 1}' for j in range(5)]
    label = 'tab:fp-categories-dream4'
    table = create_fp_table(evaluations, _method_names, method_keys, net_names, label, caption)
    with open(os.path.join(TABLES_PATH, 'dream4-fp-categories.tex'), 'w') as f:
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
            plot_fp_types(ax, res['G-target'], res['G-pred'], res['T'], n_pred=250)
            filepath = os.path.join(FIGURES_PATH, f'{method_name}-{net_id}.png')
            plt.savefig(filepath)
            plt.close()

    values = []
    for method_name in ['genie3', 'aracneap', 'tigress', 'plsnet', 'ennet', 'portia', 'eteportia', 'zscores']:
        values.append(evaluations[method_name]['symmetry'])
    values.append(evaluations['portia']['target-symmetry'])
    method_names = ['GENIE3', 'ARACNe-AP', 'TIGRESS', 'PLSNET', 'ENNET', 'PORTIA', 'etePORTIA', 'Z-scores', 'Goldstandard']
    title = 'Symmetry of inferred GRNs (DREAM4)'
    plot_matrix_symmetry(values, method_names, title=title)
    filepath = os.path.join(ROOT, 'figures', f'symmetry-dream4.eps')
    plt.savefig(filepath, dpi=300, transparent=True)
    plt.close()


if __name__ == '__main__':
    main()
