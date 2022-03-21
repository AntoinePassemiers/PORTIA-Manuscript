# -*- coding: utf-8 -*-
# evaluate-dream5.py
# author: Antoine Passemiers

import argparse
import os
import zipfile
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import synapseclient

from portia.gt.evaluation import graph_theoretic_evaluation, plot_fp_types
from portia.gt.symmetry import matrix_symmetry
from portia.gt.grn import GRN
from evalportia.data import get_synapse_credentials
from evalportia.metrics import *
from evalportia.utils.latex import *
from evalportia.plot import plot_matrix_symmetry


ROOT = os.path.dirname(os.path.abspath(__file__))
NETWORKS_PATH = os.path.join(ROOT, 'inferred-networks', 'dream5')
EVAL_TMP_PATH = os.path.join(ROOT, 'eval-tmp', 'dream5')
if not os.path.isdir(EVAL_TMP_PATH):
    os.makedirs(EVAL_TMP_PATH)
TABLES_PATH = os.path.join(ROOT, 'tables')
if not os.path.isdir(TABLES_PATH):
    os.makedirs(TABLES_PATH)
FIGURES_PATH = os.path.join(ROOT, 'figures', 'dream5')
if not os.path.isdir(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)


synapse_ids = {
    '1': {
        'expression': 'syn2787226',
        'tfs': 'syn2787227',
        'metadata': 'syn2787225',
        'goldstandard': 'syn2787240'
    },
    '3': {
        'expression': 'syn2787234',
        'tfs': 'syn2787235',
        'metadata': 'syn2787233',
        'goldstandard': 'syn2787243'
    },
    '4': {
        'expression': 'syn2787238',
        'tfs': 'syn2787239',
        'metadata': 'syn2787237',
        'goldstandard': 'syn2787244'
    }
}


def parse_output(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.rstrip()
        try:
            elements = [float(el) for el in line.split()]
            if len(elements) == 3:
                data.append(elements)
            else:
                continue
        except:
            continue
    assert len(data) == 5
    return {
        'overall-score': data[0][0],
        'auprc': data[3],
        'auroc': data[4]
    }


def main():

    synapse = synapseclient.Synapse()
    username, password = get_synapse_credentials()
    synapse.login(username, password)

    entity = synapse.get('syn2787219')
    with zipfile.ZipFile(entity.path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(ROOT, 'data'))

    script_location = os.path.join(ROOT, 'data', 'eval', 'matlab')

    method_names = os.listdir(NETWORKS_PATH)
    evaluations = {method_name: {'gt': [], 'scores': [], 'aurocs': [], 'auprcs': [], 'symmetry': [], 'target-symmetry': [], 'scores-noko': [], 'aurocs-noko': [], 'auprcs-noko': []} for method_name in method_names}

    for noko in [True, False]:

        for method_name in method_names:
            for net_id in ['1', '3', '4']:
                synapse_id_expression = synapse_ids[net_id]['expression']
                synapse_id_tfs = synapse_ids[net_id]['tfs']
                synapse_id_goldstandard = synapse_ids[net_id]['goldstandard']
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
                n_genes = X.shape[1]
                tf_idx = np.asarray([i for i in range(len(gene_names)) if gene_names[i] in tfs])

                filename = f'noko.{net_id}.txt' if noko else f'{net_id}.txt'
                filepath = os.path.join(NETWORKS_PATH, method_name, filename)
                dest_folder = os.path.join(script_location, '..', 'INPUT', 'predictions', 'myteam')
                if not os.path.isdir(dest_folder):
                    os.makedirs(dest_folder)
                dest_filepath = os.path.join(dest_folder, f'DREAM5_NetworkInference_myteam_Network{net_id}.txt')
                shutil.copyfile(filepath, dest_filepath)

                entity = synapse.get(synapse_id_goldstandard)
                G_target = GRN.load_goldstandard(entity.path, gene_names)
                A = G_target.asarray()
                print(f'Proportion of positives: {float(np.nanmean(A))}')
                G_pred = GRN.load_network(filepath, gene_names, tf_idx)
                M_bar = G_pred.asarray()
                tf_mask = np.zeros(n_genes, dtype=bool)
                tf_mask[tf_idx] = 1
                tmp_filepath = os.path.join(EVAL_TMP_PATH, f'{net_id}.npz')
                res = graph_theoretic_evaluation(tmp_filepath, G_target, G_pred, tf_mask)
                res['G-target'] = G_target
                res['G-pred'] = G_pred
                res['net-id'] = net_id
                evaluations[method_name]['gt'].append(res)
                evaluations[method_name]['symmetry'].append(matrix_symmetry(G_pred))
                evaluations[method_name]['target-symmetry'].append(matrix_symmetry(G_target))

            os.system(f'cd {script_location}; matlab -nosplash -nodisplay -r "run \'go_all\', exit\"')

            filepath = os.path.join(script_location, '..', 'OUTPUT', 'myteam.txt')
            results = parse_output(filepath)
            if not noko:
                evaluations[method_name]['overall-score'] = results['overall-score']
                evaluations[method_name]['aurocs'] = results['auroc']
                evaluations[method_name]['auprcs'] = results['auprc']
            else:
                evaluations[method_name]['overall-score-noko'] = results['overall-score']
                evaluations[method_name]['aurocs-noko'] = results['auroc']
                evaluations[method_name]['auprcs-noko'] = results['auprc']

        for method_name in evaluations.keys():
            print(method_name, evaluations[method_name])

    # Generate performance table
    print('Generating LaTeX tables...')
    def compute_table_row(method_name, key):
        values = [method_name]
        for i in range(3):
            values.append(evaluations[key]['auprcs-noko'][i])
            values.append(evaluations[key]['aurocs-noko'][i])
        values.append(evaluations[key]['overall-score-noko'])
        return values
    caption = 'ROC-AUC scores of different GRN inference methods, evaluated on the 4 networks proposed in the \\dreamfive GRN sub-challenge (no KO experiment).'
    table = LaTeXTable(caption, 'tab:dream5-benchmark', bioinformatics=(not noko))
    table.add_column(MultiColumn('Method', dtype=str, alignment='l'))
    for i in [1, 3, 4]:
        table.add_column(MultiColumn(f'Net{i}', ['AUPR', 'AUROC'], dtype=float))
    table.add_column(MultiColumn('Overall score', dtype=float, alignment='r'))
    table.add_row_values(compute_table_row('ARACNe-AP', 'aracneap'))
    table.add_row_values(compute_table_row('GENIE3', 'genie3'))
    table.add_row_values(compute_table_row('PLSNET', 'plsnet'))
    table.add_row_values(compute_table_row('TIGRESS', 'tigress'))
    table.add_row_values(compute_table_row('ENNET', 'ennet'))
    table.add_midrule()
    table.add_row_values(compute_table_row('\\fastmethodname', 'portia'))
    table.add_row_values(compute_table_row('\\methodname', 'eteportia'))
    with open(os.path.join(TABLES_PATH, 'dream5-noko.tex'), 'w') as f:
        f.write(str(table))

    def compute_table_row(method_name, key):
        values = [method_name]
        for i in range(3):
            values.append(evaluations[key]['auprcs'][i])
            values.append(evaluations[key]['aurocs'][i])
        values.append(evaluations[key]['overall-score-noko'])
        values.append(evaluations[key]['overall-score'])
        return values
    caption = 'ROC-AUC scores of different GRN inference methods, evaluated on the 4 networks proposed in the \\dreamfive \\ GRN inference sub-challenge.'
    table = LaTeXTable(caption, 'tab:dream5-noko-benchmark', double_column=True)
    table.add_column(MultiColumn('Method', dtype=str, alignment='l'))
    for i in [1, 3, 4]:
        table.add_column(MultiColumn(f'Net{i}', ['AUPR', 'AUROC'], dtype=float))
    table.add_column(MultiColumn('Overall score (no KO)', dtype=float, alignment='r'))
    table.add_column(MultiColumn('Overall score', dtype=float, alignment='r'))
    table.add_row_values(compute_table_row('ARACNe-AP', 'aracneap'))
    table.add_row_values(compute_table_row('GENIE3', 'genie3'))
    table.add_row_values(compute_table_row('PLSNET', 'plsnet'))
    table.add_row_values(compute_table_row('TIGRESS', 'tigress'))
    table.add_row_values(compute_table_row('ENNET', 'ennet'))
    table.add_midrule()
    table.add_row_values(compute_table_row('\\fastmethodname', 'portia'))
    table.add_row_values(compute_table_row('\\methodname', 'eteportia'))
    with open(os.path.join(TABLES_PATH, 'dream5.tex'), 'w') as f:
        f.write(str(table))

    caption = 'Proportions of false positives made on \\dreamfive, categorised according to the local causal structure in which they occured, for all methods.'
    method_keys = ['aracneap', 'genie3', 'plsnet', 'tigress', 'ennet', 'portia', 'eteportia']
    _method_names = ['ARACNe-AP', 'GENIE3', 'PLSNET', 'TIGRESS', 'ENNET', 'PORTIA', 'etePORTIA']
    net_names = [f'Net{j}' for j in [1, 3, 4]]
    label = 'tab:fp-categories-dream5'
    table = create_fp_table(evaluations, _method_names, method_keys, net_names, label, caption)
    with open(os.path.join(TABLES_PATH, 'dream5-fp-categories.tex'), 'w') as f:
        f.write(str(table))

    print(table)
    for method_name in method_names:
        ndcgs = []
        for res in evaluations[method_name]['gt']:
            ndcgs.append(res['score'])
        print(f'NDCG of {method_name}: {np.mean(ndcgs)}')

    # Generate figures
    print('Generating figures...')
    for method_name in method_names:
        for res in evaluations[method_name]['gt']:
            net_id = res['net-id']
            ax = plt.subplot(1, 1, 1)
            plot_fp_types(ax, res['G-target'], res['G-pred'], res['T'], n_pred=300)
            filepath = os.path.join(FIGURES_PATH, f'{method_name}-{net_id}.png')
            plt.savefig(filepath)
            plt.close()

    values = []
    for method_name in ['genie3', 'aracneap', 'tigress', 'plsnet', 'ennet', 'portia', 'eteportia']:
        values.append(evaluations[method_name]['symmetry'])
    values.append(evaluations['portia']['target-symmetry'])
    method_names = ['GENIE3', 'ARACNe-AP', 'TIGRESS', 'PLSNET', 'ENNET', 'PORTIA', 'etePORTIA', 'Goldstandard']
    title = 'Symmetry of inferred GRNs (DREAM5)'
    plot_matrix_symmetry(values, method_names, title=title)
    filepath = os.path.join(ROOT, 'figures', f'symmetry-dream5.eps')
    plt.savefig(filepath, dpi=300, transparent=True)
    plt.close()

    print('Finished.')


if __name__ == '__main__':
    main()
