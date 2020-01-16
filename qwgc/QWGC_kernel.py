import numpy as np
import random

from numpy import pi
from tqdm import tqdm
from grakel import datasets
from sklearn.model_selection import KFold
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute

from preprocess.qwfilter import QWfilter

try:
    from utils.notification import Notify
    notify = True
except Exception:
    notify = False

THETA_MIN, THETA_MAX = -pi, pi


class QWKernel:
    '''
    In this package, we selected particle swarm optimzier as a optimizer
    for tuning vector theta.
    You can choose any of optimization way, but gradient way might not
    suit for this model.
    FIXME: more customizable
    In this scheme, the type of coin is constantly changing in each iteraions.
    '''

    def __init__(self, n_steps=5, initial='super', **kwargs):
        '''
        Hyper parameters of model.
        Input:
            FIXME:
            encoder: list of binary
                    e.g. ['01', '10']
                    length of encoder must be the same as the number of class
            n_steps: int (the number of steps of Quantum walk)
        '''
        # FIXME more efficient
        if n_steps < 0:
            raise ValueError('The number of steps must be\
                              zero or over')

        self.step = n_steps
        self.initial = initial

    def create_kernel(self, train_data, train_label):
        '''
        Input:
            train_data: 2dim array (a series of training data)
            train_label: 2dim array (a series of label, one hot)
        Output:
            theta: array
            coin_param: ?
        '''
        ld = len(train_data)
        coin_u3s = np.array([random.uniform(THETA_MIN, THETA_MAX),
                             pi/self.step, pi/self.step])

        ampdata = QWfilter(coin_u3s, self.step,
                           self.initial).amplitude(train_data)

        qsize = self.ceilog(max(set(map(len, ampdata))))

        # FIXME
        n_amp = np.array([self._zero_fill(amp, 2**qsize)
                          for amp in ampdata])

        print('Creating kernel...')
        # start training
        # to check convergence of error, prepare this list
        kernel = np.zeros((ld, ld))
        backend = Aer.get_backend('qasm_simulator')
        shots = 1024
        print(qsize)
        for id1, d1 in tqdm(enumerate(n_amp)):
            q1 = QuantumRegister(qsize)
            qc1 = QuantumCircuit(q1, name='QW1')
            qc1.initialize(d1, q1)
            qw1 = qc1.to_instruction()
            for id2, d2 in enumerate(n_amp):
                # create operations
                q2 = QuantumRegister(qsize)
                qc2 = QuantumCircuit(q2)
                qc2.initialize(d2, q2)
                qw2 = qc2.to_instruction()
                # circuit for calc inner product
                kq = QuantumRegister(qsize)
                c = ClassicalRegister(qsize)
                kqc = QuantumCircuit(kq, c)
                kqc.append(qw1, qargs=kq)
                kqc.append(qw2, qargs=kq)
                kqc.measure(kq, c)
                # calc prob '000...0'
                job = execute(kqc, backend=backend, shots=shots)
                count = job.result().get_counts(kqc)
                kernel[id1][id2] = count.get('0'*qsize, 0)/shots
        return kernel

    @staticmethod
    def ceilog(x):
        return int(np.ceil(np.log2(x)))

    def test(self, test_data, kernel, coin_param):
        '''
        Test function to evaluate the performance
        Input:
            test_data: 2dim array (a series of test data)
            kernel: 2dim array
        Output:
            answers: 2dim array (answers for each data)
        '''
        ampdata = QWfilter(coin_param, self.step,
                           self.initial).amplitude(test_data)
        n_amp = np.array([self._zero_fill(amp, 2**self.qsize)
                         for amp in ampdata])
        # TODO Inplement the case that the number of class is unknown
        answers = []
        return answers

    def _accs(self, ans, label):
        count = 0
        for i, j in zip(ans, label):
            if np.argmax(i) == np.argmax(j):
                count += 1
        # print('answer ', [np.argmax(i) for i in ans])
        # print('label ', [np.argmax(i) for i in label])
        return count/len(label)

    @staticmethod
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

    def summary(self, answer, label, printer=False):
        accs = self._accs(answer, label)
        if printer:
            print('The accuracy of this model is ', accs)
        return accs


def one_hot_encoder(label, n_class):
    enc_label = [np.zeros(n_class) for _ in label]
    for ilb, lb in enumerate(label):
        if lb == -1:
            enc_label[ilb][0] = 1
        else:
            enc_label[ilb][lb] = 1
    return enc_label


if __name__ == '__main__':
    # parsing parameters from toml
    data_name = 'MUTAG'
    Data = datasets.fetch_dataset(data_name, verbose=False)
    data_x, data_y = np.array(Data.data), np.array(Data.target)

    acclist = []
    k = 5
    kf = KFold(n_splits=k, shuffle=True)
    qwk = QWKernel()
    for train_index, test_index in kf.split(data_x):
        # preprocessing for generating data.
        x_train, y_train = data_x[train_index], data_y[train_index]
        x_test, y_test = data_x[test_index], data_y[test_index]

        # one hot encoding
        y_train = one_hot_encoder(y_train, 2)
        y_test = one_hot_encoder(y_test, 2)
        kernel = qwk.create_kernel(x_train, y_train)
        print(kernel)

        answer = test()

        if notify:
            Notify.notify_accs(accs, conv)
    print('acclist', acclist)
    print('mean', np.mean(acclist))
