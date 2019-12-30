import numpy as np
import random
import copy

from numpy import pi
from tqdm import trange
from grakel import datasets
from sklearn.model_selection import KFold

from classifier.qcircuit import ClassifierCircuit
from preprocess.qwfilter import QWfilter

try:
    from utils.notification import Notify
    notify = True
except Exception:
    notify = False

THETA_MIN, THETA_MAX = -pi, pi


class QWGC:
    '''
    In this package, we selected particle swarm optimzier as a optimizer
    for tuning vector theta.
    You can choose any of optimization way, but gradient way might not
    suit for this model.
    FIXME: more customizable
    In this scheme, the type of coin is constantly changing in each iteraions.
    '''

    def __init__(self, encoder, Cp=1.5, Cg=1.3, n_particle=20, T=100, w=0.8,
                 ro_max=1.0, n_layer=2, lamb=0.005, n_steps=5, initial='super'):
        '''
        Hyper parameters of model.
        Input:
            FIXME:
            encoder: list of binary
                    e.g. ['01', '10']
                    length of encoder must be the same as the number of class
            Cp: float (constant value of coefficient
                        for personal best position)
            Cg: float (constant value of coefficient
                        for grobal best position)
            n_particle: int (the number of particles
                        for searching better parameters)
            T: int (the number of iterations of how many times
                        this model learns)
            w: float (constant value of coefficient
                        for previous particle directions)
            ro_max: float (maximum number of random number
                        which is used for updating parameter)
            n_layer: int (the number of layers of mapping circuit)
            lamb: float (constant value of coefficient
                        for the sum of square in error)
            n_steps: int (the number of steps of Quantum walk)

        '''
        self.encoder = encoder
        self.Cp = Cp
        self.Cg = Cg
        self.w = w
        self.ro_max = ro_max
        # FIXME more efficient
        if T < 0:
            raise ValueError('The number of iterations must be \
                             non negative value')
        if n_particle < 0:
            raise ValueError('The number of particles must be\
                             non negative value')
        if n_layer <= 0:
            raise ValueError('The number of layers must be\
                             one or over')
        if n_steps < 0:
            raise ValueError('The number of steps must be\
                              zero or over')
        self.T = T
        self.n_particle = n_particle
        self.layers = n_layer
        self.step = n_steps
        self.lamb = lamb
        self.initial = initial

    def optimize(self, train_data, train_label):
        '''
        Input:
            train_data: 2dim array (a series of training data)
            train_label: 2dim array (a series of label, one hot)
        Output:
            theta: array
            coin_param: ?
        '''
        self.n_class = len(train_label[0])
        if self.n_class > 2**self.layers:
            raise ValueError('the number of class must be less than 2^layers')

        # initial parameter for Quantum Walk
        # each particle has [theta, phi, lambda]
        coin_u3s = np.array([[random.uniform(THETA_MIN, THETA_MAX)
                            for i in range(3)]
                            for n in range(self.n_particle)])

        ampdata = [QWfilter(coin_u3s[n], self.step,
                            self.initial).amplitude(train_data)
                   for n in range(self.n_particle)]
        theta_size = self.ceilog(max(set(map(len, ampdata[0]))))

        # FIXME
        n_amp = np.array([[self._zero_fill(amp, 2**theta_size)
                          for amp in ampdata[n]]
                          for n in range(self.n_particle)])

        # initialize particles
        particles = np.array([[random.uniform(THETA_MIN, THETA_MAX)
                             for i in range(theta_size)]
                             for n in range(self.n_particle)])
        velocities = np.array([np.zeros(theta_size)
                              for n in range(self.n_particle)])
        coin_v = np.array([np.zeros(3) for n in range(self.n_particle)])

        # for recording best positions of each particle
        personal_bpos = copy.copy(particles)
        personal_cbpos = copy.copy(coin_u3s)
        personal_best_scores = [self._get_cost(amps, train_label, theta)
                                for amps, theta in zip(n_amp, particles)]

        # recording best position of all particles
        best_particle = np.argmin(personal_best_scores)
        grobal_best_pos = personal_bpos[best_particle]
        grobal_best_coin = coin_u3s[best_particle]

        print('training start!')
        # start training
        # to check convergence of error, prepare this list
        errors = []
        accuracy = []
        for t in trange(self.T, desc='training'):
            ampdata = [QWfilter(coin_u3s[n], self.step,
                       self.initial).amplitude(train_data)
                       for n in range(self.n_particle)]
            n_amp = np.array([[self._zero_fill(amp, 2**theta_size)
                             for amp in ampdata[n]]
                             for n in range(self.n_particle)])
            for n in range(self.n_particle):
                amp = n_amp[n]
                # random number for personal best pos
                rnp = random.uniform(0, self.ro_max)
                # random number for grobal best
                rng = random.uniform(0, self.ro_max)

                # update position
                particles[n] = particles[n] + velocities[n]
                # update coin pos

                coin_u3s[n] = coin_u3s[n] + coin_v[n]

                velocities[n] = (self.w*velocities[n] +
                                 self.Cp*rnp*(personal_bpos[n]-particles[n]) +
                                 self.Cg*rng*(grobal_best_pos-particles[n]))
                coin_v[n] = (self.w*coin_v[n] +
                             self.Cp*rnp*(personal_cbpos[n]-coin_u3s[n]) +
                             self.Cg*rng*(grobal_best_coin-coin_u3s[n]))

                # calculation cost with updated parameters
                # and update best position and score
                score = self._get_cost(amp, train_label, particles[n])
                if score < personal_best_scores[n]:
                    personal_best_scores[n] = score
                    personal_bpos[n] = particles[n]
                    personal_cbpos[n] = coin_u3s[n]

            # in all particles, calculate which is the best particle
            # and coin parameters
            best_particle = np.argmin(personal_best_scores)
            grobal_best_coin = coin_u3s[best_particle]
            grobal_best_pos = personal_bpos[best_particle]

            best_amp_data = QWfilter(grobal_best_coin, self.step,
                                     self.initial).amplitude(train_data)
            n_best_amp = [self._zero_fill(amp, 2**theta_size)
                          for amp in best_amp_data]
            error = self._get_cost(n_best_amp, train_label, grobal_best_pos)
            accs = self._get_accuracy(n_best_amp, train_label,
                                      grobal_best_pos)
            errors.append(error)
            accuracy.append(accs)
            if t % 10 == 0 and notify:
                Notify.notify_error(t, error, accs)
        convergence = [errors, accuracy]
        return grobal_best_pos, grobal_best_coin

    def _get_cost(self, data, label, theta):
        cost = ClassifierCircuit(data, label, theta, self.n_class,
                                 self.layers, self.encoder).cost()

        error = cost + self.lamb*np.sum([i**2 for i in theta])
        return error

    def _get_accuracy(self, data, label, theta):
        answers = ClassifierCircuit(data, label, theta, self.n_class,
                                    self.layers, self.encoder).answers()
        acc = self._accs(answers, label)
        return acc

    @staticmethod
    def ceilog(x):
        return int(np.ceil(np.log2(x)))

    def test(self, test_data, theta, coin_param):
        '''
        Test function to evaluate the performance
        Input:
            test_data: 2dim array (a series of test data)
            theta: tuned parameters theta
            coin_param: optimzed coin parameters
        Output:
            answers: 2dim array (answers for each data)
        '''
        theta_size = len(theta)
        ampdata = QWfilter(coin_param, self.step,
                           self.initial).amplitude(test_data)
        n_amp = np.array([self._zero_fill(amp, 2**theta_size)
                         for amp in ampdata])
        # TODO Inplement the case that the number of class is unknown
        answers = ClassifierCircuit(n_amp, None, theta, 2,
                                    self.layers, self.encoder).answers()
        return answers

    def _accs(self, ans, label):
        count = 0
        for i, j in zip(ans, label):
            if np.argmax(i) == np.argmax(j):
                count += 1
        print('answer ', [np.argmax(i) for i in ans])
        print('label ', [np.argmax(i) for i in label])
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
    # prepare dataset
    data_name = 'MUTAG'
    Data = datasets.fetch_dataset(data_name, verbose=False)
    data_x, data_y = np.array(Data.data), np.array(Data.target)

    acclist = []
    k = 5
    kf = KFold(n_splits=k, shuffle=True)
    qwgc = QWGC(['01', '10'])
    for train_index, test_index in kf.split(data_x):
        # preprocessing for generating data.
        x_train, y_train = data_x[train_index], data_y[train_index]
        x_test, y_test = data_x[test_index], data_y[test_index]

        # one hot encoding
        y_train = one_hot_encoder(y_train, 2)
        y_test = one_hot_encoder(y_test, 2)

        theta, coin_param = qwgc.optimize(x_train, y_train)
        # test
        ans = qwgc.test(x_test, theta, coin_param)
        # evaluate
        accs = qwgc.summary(ans, y_test)

        acclist.append(accs)
        print(accs)
        if notify:
            Notify.notify_accs(accs)
    print('acclist', acclist)
    print('mean', np.mean(acclist))
