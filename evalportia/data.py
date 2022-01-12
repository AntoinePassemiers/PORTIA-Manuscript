# -*- coding: utf-8 -*-
# data.py
# author: Antoine Passemiers

import os
import getpass


def get_synapse_credentials():
    username = os.environ.get('SYNAPSE_USERNAME')
    if username is None:
        username = input('Synapse username: ')
        os.environ['SYNAPSE_USERNAME'] = username
    password = os.environ.get('SYNAPSE_PASSWORD')
    if password is None:
        password = getpass.getpass('Synapse password: ')
    return username, password
