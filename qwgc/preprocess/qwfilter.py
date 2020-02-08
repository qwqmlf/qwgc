import numpy as np
import itertools

from grakel import Graph
from .qwalk import QuantumWalk


class QWfilter:
    def __init__(self, u3param, step, initial):
        self.u3p = u3param
        self.step = step
        self.initial = initial

    def amplitude(self, data, normalize=False):
        '''
        interface
        Output:
            amplitude (no normalize)
        '''
        adjacency = [Graph(d[0]).get_adjacency_matrix() for d in data]
        amplitude = []
        for ad in adjacency:
            count = np.count_nonzero(ad)//2
            nad = len(ad)

            # prepare coin with u3 parameter
            coin = self._compose_coins(count, ad)
            # prepare initial state
            initial_state = self._initial(count, nad)

            # construct quantum walk and go n steps
            qwalk = QuantumWalk(initial_state, coin, ad)
            qwalk.n_steps(self.step)

            amplitude.append(qwalk.calc_amp())
        return amplitude

    def single_amplitude(self, d):
        ad = Graph(d[0]).get_adjacency_matrix()
        count = np.count_nonzero(ad)//2
        nad = len(ad)

        # prepare coin with u3 parameter
        coin = self._compose_coins(count, ad)
        # prepare initial state
        initial_state = self._initial(count, nad)

        # construct quantum walk and go n steps
        qwalk = QuantumWalk(initial_state, coin, ad)
        qwalk.n_steps(self.step)
        amplitude = qwalk.calc_amp()
        return amplitude
    
    def single_prob(self, d):
        ad = Graph(d[0]).get_adjacency_matrix()
        count = np.count_nonzero(ad)//2
        nad = len(ad)

        # prepare coin with u3 parameter
        coin = self._compose_coins(count, ad)
        # prepare initial state
        initial_state = self._initial(count, nad)

        # construct quantum walk and go n steps
        qwalk = QuantumWalk(initial_state, coin, ad)
        qwalk.n_steps(self.step)
        probability = qwalk.calc_probs()
        return probability

    def _compose_coins(self, count, adja):
        '''
        Input:
            count: the number of non-zero elements in adjacency matrix
            adja: adjacency matrix of data
            pahse: FIXME default True
                useing quantum coin with phase
        Output:
            coin: 2d matrix (unitary)
        '''
        co = []
        elcoin = []
        coin = np.array(np.diag(np.zeros(count*2)), dtype=np.complex)
        for lad in adja:
            s = int(sum(lad))
            co.append(s)
        section = [0] + list(itertools.accumulate(co))[0:-1]
        for c in co:
            coin_element = self._coin(c)
            for ce in coin_element:
                elcoin.append(ce)
        counter = 0
        for ic, coinel in enumerate(zip(coin, elcoin)):
            ci, ce = coinel[0], coinel[1]
            nce = len(ce)
            ci[counter:counter+nce] = ce
            if ic+1 in section:
                counter = ic+1
        if not is_unitary(coin):
            raise Exception('coin operator must be unitary')
        return coin

    def _coin(self, num):
        if num == 2:
            # testing if this is good enough to make good amp or not
            coin = self.U3(self.u3p[0], self.u3p[1], self.u3p[2])
        else:
            coin = np.array([[2/num for k in range(num)]
                            for i in range(num)] - np.identity(num))

        if not is_unitary(coin):
            raise Exception("elementary operator must be unitary.")
        return coin

    def _initial(self, count, nad):
        if self.initial is None:
            initial_state = None
        elif isinstance(self.initial, list or np.ndarray):
            initial_state = self.initial
        else:
            initial_state = [1/np.sqrt(nad) for i in range(nad)] + \
                            [0 for i in range(2*count-nad)]
            # FIXME check threshold
            assert(np.sum(i**2 for i in initial_state)-1 < 1e-5)
        return initial_state

    @staticmethod
    def U3(theta, phi, lamb):
        return np.array([[np.cos(theta/2),
                         -np.exp(1j*lamb)*np.sin(theta/2)],
                         [np.exp(1j*phi)*np.sin(theta/2),
                         np.exp(1j*lamb+1j*phi)*np.cos(theta/2)]])


def is_unitary(operator, tolerance=0.0001):
    h, w = operator.shape
    if not h == w:
        return False
    adjoint = np.conjugate(operator.transpose())
    product1 = np.dot(operator, adjoint)
    product2 = np.dot(adjoint, operator)
    ida = np.eye(h)
    return np.allclose(product1, ida) & np.allclose(product2, ida)
