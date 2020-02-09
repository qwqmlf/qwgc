"""
This is archive version of previous qmnist project.
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute, Aer
from numpy import pi
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm import trange

import numpy as np
import random
import copy
import umap.umap_ as umap

THETA_MIN, THETA_MAX = -pi, pi
BACKEND = Aer.get_backend('qasm_simulator')
SHOTS = 2048


class QMNIST:
    """
    QMMIST is for quantum classifier for MNIST

    This
    """
    def __init__(self, encoder, alpha=0.0001,
                 n_particle=20, iteration=50, w=0.8, Cp=1.3, Cg=1.1):
        self.encoder = encoder
        self.alpha = alpha
        self.n_particle = n_particle
        self.iteration = iteration
        self.w = w
        self.Cp = Cp
        self.Cg = Cg

    def fit(self, x=None, y=None):
        dim = len(x[0])
        particles = np.array([[random.uniform(THETA_MIN, THETA_MAX)
                              for j in range(dim)]
                              for n in range(self.n_particle)])
        velocities = np.array([[0 for i in range(dim)] for n in range(self.n_particle)])
        personal_best_pos = copy.copy(particles)
        personal_best_score = [self._get_error(x, y, theta)
                               for theta in particles]

        best_particle = np.argmin(personal_best_score)
        grobal_best_pos = personal_best_pos[best_particle]

        errors = []
        accs = []
        print("Training start!")
        for t in trange(self.iteration):
            for n in range(self.n_particle):
                ran_p = random.uniform(0, 1)
                ran_g = random.uniform(0, 1)
                particles[n] = particles[n] + velocities[n]
                velocities[n] = (self.w*velocities[n] +
                                 self.Cp*ran_p*(personal_best_pos[n]-particles[n]) +
                                 self.Cg*ran_g*(grobal_best_pos-particles[n]))
                score = self._get_error(x, y, particles[n])
                if score < personal_best_score[n]:
                    personal_best_score[n] = score
                    personal_best_pos[n] = particles[n]

            best_particle = np.argmin(personal_best_score)
            grobal_best_pos = personal_best_pos[best_particle]

            error = self._get_error(x, y, grobal_best_pos)
            acc = self._get_accuracy(x, y, grobal_best_pos)
            print(error, acc, grobal_best_pos)
            errors.append(error)
            accs.append(acc)
            converg = [errors, accs]
        return grobal_best_pos, converg

    def _get_error(self, x, y, theta):
        counts = self._get_counts(x, theta)
        errors = []
        for ct, lb in zip(counts, y):
            emp_x = np.array([ct.get(b, 0)/SHOTS for b in self.encoder])
            err = self._error_func(emp_x, np.array(y))
            errors.append(err)
        return np.mean(errors)

    def _get_counts(self, x, theta):
        qcs = self._classifier(x, theta)
        job = execute(qcs, backend=BACKEND, shots=SHOTS)
        result = job.result()
        counts = [result.get_counts(qc) for qc in qcs]
        # print(counts)
        return counts

    @staticmethod
    def _error_func(x, bx, delta=1e-9):
        return -np.sum(bx * np.log(x + delta))

    @staticmethod
    def _map_func(x):
        val = x/np.arcsinh(x)
        # print(val, x)
        return val

    def _classifier(self, x, theta):
        qcs = []
        ld = len(x[0])
        for xt in x:
            dataq = QuantumRegister(ld)
            c = ClassicalRegister(ld)
            qc = QuantumCircuit(dataq, c)
            for xd, qr in zip(xt, dataq):
                qc.h(qr)
                qc.rz(self._map_func(xd), qr)
                qc.h(qr)
                qc.rz(self._map_func(xd), qr)
            # anzatz
            for r in range(4):
                for ith, th in enumerate(theta):
                    qc.ry(th, dataq[ith])
                for ids, d in enumerate(dataq[:-1]):
                    qc.cz(d, dataq[ids+1])
            qc.cz(dataq[0], dataq[-1])
            qc.measure(dataq, c)
            qcs.append(qc)
        return qcs

    def _get_accuracy(self, x, y, theta):
        counts = self._get_counts(x, theta)
        answers = self._get_answer(counts)
        count = 0
        for ans, lb in zip(answers, y):
            if ans == np.argmax(lb):
                count += 1
        accuracy = count/len(y)
        return accuracy

    def _get_answer(self, counts):
        answers = []
        for cs in counts:
            answer = np.argmax([cs.get(b, 0) for b in self.encoder])
            answers.append(answer)
        return answers

    def test(self, x, param):
        counts = self._get_counts(x, param)
        answer = self._get_answer(counts)
        return answer

    def performance(self, ans, label):
        count = 0
        for an, lb in zip(ans, label):
            if np.argmax(lb) == an:
                count += 1
        return count/len(label)


if __name__ == '__main__':
    digits = load_digits()
    # qmnist = QMNIST(['0000000001', '0000000010',
    #                  '0000000100', '0000001000',
    #                  '0000010000', '0000100000',
    #                  '0001000000', '0010000000',
    #                  '0100000000', '1000000000'])

    # qmnist = QMNIST(['0000', '1000',
    #                  '0001', '1001',
    #                  '0010', '1010',
    #                  '0100', '1101',
    #                  '0101', '1111'])

    qmnist = QMNIST(['01', '10'])
    onhot_labels = []
    for i in digits.target:
        lab = [0 for _ in range(10)]
        lab[int(i)] = 1
        onhot_labels.append(lab)
    reducer = umap.UMAP(n_components=2)
    reducer.fit(digits.data)
    embedding = reducer.transform(digits.data)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))
    x_train, x_test, y_train, y_test = train_test_split(embedding[:1000], onhot_labels[:1000])
    param, conv = qmnist.fit(x_train, y_train)
    answers = qmnist.test(x_test, param)
    accs = qmnist.performance(answers, y_test)
    print(param, conv)
    print(accs)
