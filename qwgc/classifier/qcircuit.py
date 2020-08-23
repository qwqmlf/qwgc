from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from scipy.special import softmax

from utils.notification import Notify
import numpy as np
from numpy import pi

QASM = Aer.get_backend('qasm_simulator')


class ClassifierCircuit:
    '''
    This class is the core part of this algorithm.
    FIXME inherit QWGC and take super constructor.
    '''

    def __init__(self, data, label, theta, n_class,
                 layers, encoder, shots=1024):
        '''
        Input:
            encoding: list (discribe the correspondence of
                            measurement basis and label)
                    e.g. ['01', '10']
        '''
        self.data = data
        if label is None:
            # FIXME this is for test
            self.label = 0
        else:
            if len(label[0]) != len(encoder):
                raise ValueError('dimension of label and \
                                  encoder must be the same')
            self.label = label
        # FIXME
        self.n_class = n_class
        self.theta = theta
        self.qsize = len(theta)
        self.layers = layers
        self.encoder = encoder
        self.shots = shots
        if layers > self.qsize:
            raise Warning('The length of theta is shorter \
                           than the number of layers')

    def _circuit_constructor(self, visualize=False, notify=False):
        '''
        returning circuits
        TODO
        if visualize got the list of integers,
        then, print out the circuit of certain index
        '''
        qcs = []
        nq = self.qsize
        layer = self.layers
        if notify:
            Notify.notify_accs(list(self.data[0]), list(self.data[-1]))
        for index, d in enumerate(self.data):
            # qubits for representing data
            qr = QuantumRegister(nq)
            # qubits for mapping data
            mp = QuantumRegister(layer)
            c = ClassicalRegister(layer)
            qc = QuantumCircuit(qr, mp, c, name='data%d' % index)
            qc.initialize(d, qr)
            qc.h(qr)
            qc = self._map(qc, qr, mp)
            qc.measure(mp, c)
            qcs.append(qc)
        return qcs

    def _map(self, qc, qr, mp):
        # counter = 0
        for ith, theta in enumerate(self.theta):
            # if ith % self.layers == 0:
            qc.cry(theta, qr[ith], mp[ith % self.layers])
            # else:
            #     qc.h(mp[ith%self.layers])
            #     qc.cu3(theta, pi/2, pi/2, qr[ith], mp[ith % self.layers])
            #     qc.h(mp[ith%self.layers])
        return qc

    def cost(self, notify=False):
        '''
        This function is the interface to pass through
        the result of measurement
        Input:
            data: amplitude vector
        Output:
            result of measurement
        '''
        qcs = self._circuit_constructor(notify=notify)
        probs = self._get_result(qcs, notify=notify)
        cross = np.mean([self._cross_entropy_error(pb, lb)
                        for pb, lb in zip(probs, self.label)])
        return cross

    def answers(self):
        qcs = self._circuit_constructor()
        probs = self._get_result(qcs)
        answers = [np.zeros(self.n_class) for _ in probs]
        for ipb, pb in enumerate(probs):
            ind = np.argmax(pb)
            answers[ipb][ind] = 1
        return answers

    def _get_result(self, qcs, notify=False):
        '''
        returning probabilities of estimation
        '''
        job = execute(qcs, backend=QASM, shots=self.shots)
        counts = [job.result().get_counts(qc) for qc in qcs]
        dinom = [sum([cs.get(i, 0) for i in self.encoder]) for cs in counts]
        if notify:
            bins = [format(i, "02b") for i in range(4)]
            Notify.notify_accs("After Classify data0", [counts[0].get(b, 0)/sum(counts[0].values()) for b in bins])
            Notify.notify_accs("After Classify data last", [counts[-1].get(b, 0)/sum(counts[-1].values()) for b in bins])
        enc_probs = [np.array([cs.get(i, 0)/(din+1e-10) for i in self.encoder])
                     for cs, din in zip(counts, dinom)]
        return enc_probs

    @staticmethod
    def ceilog(x):
        return int(np.ceil(np.log2(x)))

    @staticmethod
    def _cross_entropy_error(y, t, delta=1e-7):
        return -np.sum(t * np.log(y + delta))
