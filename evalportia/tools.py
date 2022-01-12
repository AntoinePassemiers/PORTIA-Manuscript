# -*- coding: utf-8 -*-
# tools.py
# author: Antoine Passemiers

import os
import sys
import shutil
import importlib.util
import urllib.request

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri


rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

ROOT = os.path.dirname(os.path.abspath(__file__))
TMP_FOLDER = os.path.join(ROOT, '..', 'tmp')
if not os.path.isdir(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)

ENNET_PATH = os.path.join(ROOT, 'ennet.R')
TIGRESS_PATH = os.path.join(ROOT, 'tigress.R')
NIMEFI_PATH = os.path.join(ROOT, 'nimefi.R')
GENIE3_PATH = os.path.join(ROOT, 'genie3.R')


def get_ennet():
    r = rpy2.robjects.r
    r['source'](ENNET_PATH)
    ennet = rpy2.robjects.globalenv['call_ennet']
    def ennet_func(X, K=None, tf_idx=None):
        nrow = X.shape[0]
        ncol = X.shape[1]
        kwargs = {}
        if K is not None:
            kwargs['K'] = rpy2.robjects.r.matrix(K, nrow=K.shape[0], ncol=K.shape[1])
        if tf_idx is not None:
            kwargs['Tf'] = rpy2.robjects.vectors.IntVector(tf_idx + 1)
        X = rpy2.robjects.r.matrix(X, nrow=nrow, ncol=ncol)
        G = ennet(X, **kwargs)
        assert isinstance(G, np.ndarray)
        assert len(G.shape) == 2
        return G
    return ennet_func


def get_tigress():
    r = rpy2.robjects.r
    r['source'](TIGRESS_PATH)
    tigress = rpy2.robjects.globalenv['call_tigress']
    def tigress_func(df, tf_names=None, nsplit=1000):
        kwargs = {'nsplit': nsplit}
        if tf_names is not None:
            kwargs['tflist'] = rpy2.robjects.vectors.StrVector(list(tf_names))
        assert isinstance(df, pd.DataFrame)
        df = rpy2.robjects.pandas2ri.py2rpy(df)
        G = tigress(df, **kwargs)
        assert isinstance(G, np.ndarray)
        assert len(G.shape) == 2
        return G
    return tigress_func


def get_nimefi():
    nimefi_location = os.environ.get('NIMEFI_LOCATION')
    r = rpy2.robjects.r
    r(f'setwd("{nimefi_location}")')
    r['source'](NIMEFI_PATH)
    nimefi = rpy2.robjects.globalenv['call_nimefi']
    def nimefi_func(X, tf_idx=None):
        kwargs = {}
        if tf_idx is not None:
            kwargs['predictorIndices'] = rpy2.robjects.vectors.IntVector(tf_idx + 1)
        df = nimefi(X, **kwargs)
        genes_a = df['firstCol'].to_numpy()
        genes_b = df['secondCol'].to_numpy()
        scores = df['thirdCol'].to_numpy()
        n_genes = X.shape[1]
        G = np.zeros((n_genes, n_genes))
        for k in range(len(genes_a)):
            i = int(genes_a[k][1:]) - 1
            j = int(genes_b[k][1:]) - 1
            G[i, j] = float(scores[k])
        return G
    return nimefi_func


def get_plsnet():
    plsnet_location = os.environ.get('PLSNET_LOCATION')
    tmp_folder = os.path.join(TMP_FOLDER, 'plsnet')
    if os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    def plsnet_func(X, tf_idx=None):
        if tf_idx is None:
            tf_idx = np.arange(X.shape[1])
        n_genes = X.shape[1]
        expr_filepath = os.path.join(tmp_folder, 'expression.txt')
        with open(expr_filepath, 'w') as f:
            for i in range(X.shape[0]):
                row = ','.join([str(X[i, j]) for j in range(X.shape[1])])
                f.write(f'{row}\n')
        out_filepath = os.path.join(tmp_folder, 'network.txt')
        matlab_code = f'userpath(\'{plsnet_location}\'); '
        matlab_code += f'expr_matrix = readmatrix(\'{expr_filepath}\'); '
        s = ' '.join([str(i + 1) for i in tf_idx])
        matlab_code += f'input_idx = [{s}]; '
        matlab_code += f'VIM = plsnet(expr_matrix, input_idx, 5, 30, 1000); '
        # matlab_code += f'dlmwrite(\'{out_filepath}\', VIM); '
        matlab_code += f'writematrix(VIM, \'{out_filepath}\'); '
        matlab_code += f'exit'
        cmd = f'matlab -nodisplay -nosplash -nodesktop -r "{matlab_code}"'
        os.system(cmd)
        M_bar = []
        with open(out_filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            elements = line.split(',')
            if len(elements) > 1:
                M_bar.append([float(el) for el in elements])
        M_bar = np.asarray(M_bar)

        return M_bar
    return plsnet_func


def get_genie3():
    r = rpy2.robjects.r
    r['source'](GENIE3_PATH)
    genie3 = rpy2.robjects.globalenv['call_genie3']
    def genie3_func(X, tf_idx=None):
        kwargs = {}
        if tf_idx is not None:
            kwargs['regulators'] = rpy2.robjects.vectors.IntVector(tf_idx + 1)
        X = X.T
        X = rpy2.robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        G = genie3(X, **kwargs)
        assert isinstance(G, np.ndarray)
        assert len(G.shape) == 2
        return G
    return genie3_func


def get_aracne_ap():
    tmp_folder = os.path.join(TMP_FOLDER, 'aracne-ap')
    if os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    def aracne_ap_func(X, tf_idx=None):
        n_genes = X.shape[1]
        expr_filepath = os.path.join(tmp_folder, 'expression.txt')
        with open(expr_filepath, 'w') as f:
            row = '\t'.join([f"Sample{i+1}" for i in range(X.shape[0])])
            f.write(f'gene\t{row}\n')
            for j in range(X.shape[1]):
                row = '\t'.join([str(X[i, j]) for i in range(X.shape[0])])
                f.write(f'g{j+1}\t{row}\n')
        tf_filepath = os.path.join(tmp_folder, 'tfs.txt')
        with open(tf_filepath, 'w') as f:
            if tf_idx is None:
                for j in range(n_genes):
                    f.write(f'g{j+1}\n')
            else:
                for j in tf_idx:
                    f.write(f'g{j+1}\n')
        aracne_ap_location = os.environ.get('ARACNE_AP_LOCATION')
        out_filepath = os.path.join(tmp_folder, 'network.txt')
        if os.path.exists(out_filepath):
            os.remove(out_filepath)
        cmd = f'java -Xmx5G -jar {aracne_ap_location} -e {expr_filepath} -o {tmp_folder} --tfs {tf_filepath} --pvalue 1E-8 --seed 1 --calculateThreshold'
        os.system(cmd)
        for i in range(100):
            cmd = f'java -Xmx5G -jar {aracne_ap_location} -e {expr_filepath} -o {tmp_folder} --tfs {tf_filepath} --pvalue 1E-8 --seed {i+1}'
            os.system(cmd)
        cmd = f'java -Xmx5G -jar {aracne_ap_location} -o {tmp_folder} --consolidate'
        os.system(cmd)
        with open(out_filepath, 'r') as f:
            lines = f.readlines()[1:]
        M = np.zeros((n_genes, n_genes))
        for line in lines:
            line = line.rstrip()
            elements = line.split('\t')
            if len(elements) == 4:
                i = int(elements[0][1:]) - 1
                j = int(elements[1][1:]) - 1
                M[i, j] = float(elements[2])
        return M
    return aracne_ap_func
