# -*- coding: utf-8 -*-
# latex.py
# author: Antoine Passemiers

import enum

import numpy as np

from evalportia.causal_structure import CausalStructure


class MultiColumn:

    def __init__(self, name, colnames=[], alignment='c', dtype=float):
        self.name = name
        self.colnames = colnames
        if len(self.colnames) <= 1:
            self.n_columns = 1
        else:
            self.n_columns = len(self.colnames)
        self.data = [[] for _ in range(self.n_columns)]
        self.alignment = alignment
        self.dtype = dtype
        self.max_values = [-np.inf for _ in range(self.n_columns)]
        self.start = 1

    def set_start(self, start):
        self.start = start

    def get_header_top(self):
        if self.n_columns > 1:
            return f'\\multicolumn{{{self.n_columns}}}{{c}}{{{self.name}}}'
        else:
            return self.name

    def get_header_middle(self):
        start = self.start
        end = self.start + self.n_columns - 1
        if self.n_columns == 1:
            return ''
        else:
            return f'\\cmidrule(lr){{{start}-{end}}}'

    def get_header_bottom(self):
        if self.n_columns == 1:
            return ''
        else:
            return ' & '.join(self.colnames)

    def consume_data(self, values):
        values, remaining = values[:self.n_columns], values[self.n_columns:]
        for i, value in enumerate(values):
            if self.dtype == float:
                self.max_values[i] = max(self.max_values[i], value)
            self.data[i].append(value)
        return remaining

    def format_value(self, value, row_id):
        if self.dtype == float:
            max_value = self.max_values[row_id]
            if value + 1e-15 >= max_value:
                value = f'\\textbf{{{value:.3f}}}'
            else:
                value = f'{value:.3f}'
        else:
            value = str(value)
        return value

    def get_data(self, row_id):
        if self.n_columns == 1:
            return self.format_value(self.data[0][row_id], 0)
        else:
            elements = []
            for i in range(self.n_columns):
                elements.append(self.format_value(self.data[i][row_id], i))
            return ' & '.join(elements)


class LaTeXTable:

    class Row(enum.Enum):

        MIDRULE = enum.auto()
        DATA = enum.auto()

    def __init__(self, caption, label, double_column=True, text_width=False, bioinformatics=True):
        self.caption = caption
        self.label = label
        self.rows = []
        self.multi_columns = []
        self.column_map = {}
        self.n_columns = 0
        self.double_column = double_column
        self.text_width = text_width
        self.bioinformatics = bioinformatics
        if self.bioinformatics:
            self.text_width = False

    def add_midrule(self):
        self.rows.append(LaTeXTable.Row.MIDRULE)

    def add_column(self, multi_column):
        self.multi_columns.append(multi_column)
        multi_column.set_start(self.n_columns + 1)
        for j in range(multi_column.n_columns):
            self.column_map[self.n_columns] = (multi_column, j)
            self.n_columns += 1

    def add_row_values(self, values):
        for multi_column in self.multi_columns:
            values = multi_column.consume_data(values)
        self.rows.append(LaTeXTable.Row.DATA)

    def __str__(self):
        s = ''
        if self.double_column:
            s += '\\begin{table*}[t]\n'
        else:
            s += '\\begin{table}[t]\n'
        if self.bioinformatics:
            s += f'\\processtable{{{self.caption}\\label{{{self.label}}}}}{{'
        else:
            s += f'\\caption{{{self.caption}\\label{{{self.label}}}}}'
        if self.text_width:
            s += '\\begin{adjustbox}{max width=\\textwidth}'
        s += '\\begin{tabular}'
        s += '{'
        for multi_column in self.multi_columns:
            s += multi_column.alignment * multi_column.n_columns
        s += '} \\\\ '
        s += '\\toprule\n'
        for j, multi_column in enumerate(self.multi_columns):
            s += multi_column.get_header_top()
            if j == len(self.multi_columns) - 1:
                s += ' \\\\\n'
            else:
                s += ' & '
        for multi_column in self.multi_columns:
            s += multi_column.get_header_middle()
            s += ' '
        s += '\n'
        for j, multi_column in enumerate(self.multi_columns):
            s += multi_column.get_header_bottom()
            if j == len(self.multi_columns) - 1:
                s += ' \\\\ \\midrule\n'
            else:
                s += ' & '
        row_id = 0
        for row in self.rows:
            if row == LaTeXTable.Row.MIDRULE:
                s += '\\midrule\n'
            else:
                s += ' & '.join([column.get_data(row_id) for column in self.multi_columns])
                s += ' \\\\\n'
                row_id += 1
        s += '\\botrule\n'
        s += '\\end{tabular}'
        if self.bioinformatics:
            s += '}}'
        if self.text_width:
            s += '\\end{adjustbox}'
        s += '{}\n'
        s += '\\end{table*}\n'
        return s

    def __repr__(self):
        return self.__str__()


def create_fp_table(evaluations, method_names, method_keys, net_names, label, caption):

    CATEGORIES = [
        (CausalStructure.TRUE_POSITIVE, None),
        (CausalStructure.CHAIN, 'figures/graph-n-chain.eps'),
        (CausalStructure.FORK, 'figures/graph-n-fork.eps'),
        (CausalStructure.COLLIDER, 'figures/graph-n-collider.eps'),
        (CausalStructure.CHAIN_REVERSED, 'figures/graph-n-chain-reversed.eps'),
        (CausalStructure.UNDIRECTED, 'figures/graph-n-indirect.eps'),
        (CausalStructure.SPURIOUS_CORRELATION, 'figures/graph-n-spurious.eps')
    ]
    
    table = LaTeXTable(caption, label, text_width=True)
    table.add_column(MultiColumn('Structure', dtype=str, alignment='c'))
    table.add_column(MultiColumn('Illustration', dtype=str, alignment='c'))
    table.add_column(MultiColumn('Network', dtype=str, alignment='c'))
    for method_name in method_names:
        table.add_column(MultiColumn(method_name, dtype=str, alignment='r'))

    n_networks = len(net_names)
    for cat_info in CATEGORIES:
        i, filepath = cat_info
        for j in range(n_networks):
            values = []
            if j == 0:
                s = f"\\multirow{{{n_networks}}}{{*}}{{``{CausalStructure.to_string(i)}''}}"
                values.append(s)
                if filepath is not None:
                    s = f"\\multirow{{{n_networks}}}{{*}}{{\\includegraphics[width=0.1\\textwidth]{{{filepath}}}}}"
                else:
                    s = f"\\multirow{{{n_networks}}}{{*}}{{-}}"
                values.append(s)
            else:
                values.append(' ')
                values.append(' ')
            values.append(net_names[j])
            counts = []
            total_counts = []
            for method_key, method_name in zip(method_keys, method_names):
                counts.append(np.sum(evaluations[method_key]['gt'][j]['T'] == i))
                total_counts.append(np.sum(evaluations[method_key]['gt'][j]['T'] <= CausalStructure.SPURIOUS_CORRELATION))
            ub = max(counts)
            for count in counts:
                if count >= ub:
                    values.append(f'\\textbf{{{count}}}')
                else:
                    values.append(count)
            table.add_row_values(values)
        table.add_midrule()
    return table
