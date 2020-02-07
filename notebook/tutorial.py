# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Quantum Walk Graph Classifier

# 今回の成果物である量子ウォークにおけるグラフの分類器に関するチュートリアル

# +
import numpy as np
import random
import copy

from numpy import pi
from tqdm import trange
from grakel import datasets
from sklearn.model_selection import KFold

from classifier.qcircuit import ClassifierCircuit
from preprocess.qwfilter import QWfilter
