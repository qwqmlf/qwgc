import numpy as np
import random
import matplotlib.pyplot as plt

from numpy import pi
from tqdm import tqdm, trange
from sklearn.model_selection import KFold
from grakel import datasets
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from sklearn import svm

from preprocess.qwfilter import QWfilter

try:
    from utils.notification import Notify
    notify = True
except Exception:
    notify = False

step = 5

THETA_MIN, THETA_MAX = -pi, pi
C = 10
iteration = 100
backend = Aer.get_backend('qasm_simulator')
shots = 1024

'''
This is just very prototype code
'''


def qw_kernel(train_data, train_label, lam=1):
    '''
    Input:
        train_data: 2dim array (a series of training data)
        train_label: 2dim array (a series of label, one hot)
    Output:
        theta: array
        coin_param: ?
    '''
    ld = len(train_data)

    # start training
    # to check convergence of error, prepare this list

    weights = np.zeros(ld)
    print('training start!')
    for _ in trange(iteration):
        it = random.randint(0, ld)
        decision = 0
        for j in range(it):
            decision += weights[j] * train_label[it] * _kernel_function(train_data[it], train_data[j], 7)
        decision *= train_label[it]/lam
        if decision < 1:
            weights[it] += 1
    return weights


def test(x_train, y_train, x_test, y_test, weights):
    print('test start!')
    errors = 0
    for ila, lb_test in enumerate(y_test):
        decision = 0
        for ilb, lb_train in enumerate(y_train):
            decision += weights[ilb]*y_train[ilb]*_kernel_function(x_train[ilb], x_test[ila], 7)
        if decision < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != y_test[ila]:
            errors += 1
    return 1 - errors/len(y_test)


def _kernel_function(x, y, qsize):
    # definition of coin
    coin_u3s = np.array([pi/2, pi/step, pi])

    ampdata_x = QWfilter(coin_u3s, step, 'super').single_amplitude(x)
    x_amp = _zero_fill(ampdata_x, 2**qsize)

    q1 = QuantumRegister(qsize)
    qc1 = QuantumCircuit(q1, name='QW1')
    qc1.initialize(x_amp, q1)
    qw1 = qc1.to_instruction()

    ampdata_y = QWfilter(coin_u3s, step, 'super').single_amplitude(y)
    y_amp = _zero_fill(ampdata_y, 2**qsize)

    q2 = QuantumRegister(qsize)
    qc2 = QuantumCircuit(q2)
    qc2.initialize(y_amp, q2)
    qw2 = qc2.to_instruction()

    kq = QuantumRegister(qsize)
    c = ClassicalRegister(qsize)
    kqc = QuantumCircuit(kq, c)
    kqc.append(qw1, qargs=kq)
    kqc.append(qw2, qargs=kq)
    kqc.measure(kq, c)
    # calc prob '000...0'
    job = execute(kqc, backend=backend, shots=shots)
    count = job.result().get_counts(kqc)
    return count.get('0'*qsize, 0)/shots


def ceilog(x):
    return int(np.ceil(np.log2(x)))


def _zero_fill(x, base, array=True):
    # FIXME efficiently
    xl = list(x)
    x_len = len(xl)
    if base - x_len < 0:
        raise ValueError('Error')
    xs = xl + [0 for _ in range(base-x_len)]
    if array:
        return np.array(xs)
    else:
        return xs


if __name__ == '__main__':
    data_name = 'MUTAG'
    Data = datasets.fetch_dataset(data_name, verbose=False)
    data_x, data_y = np.array(Data.data), np.array(Data.target)

    k = 5
    kf = KFold(n_splits=k, shuffle=True)
    accuracy = []
    for train_index, test_index in kf.split(data_x):
        # preprocessing for generating data.
        x_train, y_train = data_x[train_index], data_y[train_index]
        x_test, y_test = data_x[test_index], data_y[test_index]
        weight = qw_kernel(x_train, y_train)
        accs = test(x_train, y_train, x_test, y_test, weight)
        print(accs)
        if notify:
            Notify.notify_accs(accs, 'svm')
        accuracy.append(accs)
    Notify.notify_accs(accuracy, 'K5 result')
    Notify.notify_accs(np.mean(accuracy), 'K5 result mean')
    print(accuracy)
    print(np.mean(accuracy))

    
