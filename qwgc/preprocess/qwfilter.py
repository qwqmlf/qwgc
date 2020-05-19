import numpy as np
import random
import copy
import itertools

from grakel import Graph
from .qwalk import QuantumWalk

np.set_printoptions(linewidth=100000)


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
            # amp = self.coin_walk(ad)
            amp = self.szegedy_google(ad)
            amplitude.append(amp)
        return amplitude
    
    def coin_walk(self, ad):
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

    def szegedy_google(self, adjacency):
        """
        quantum pagerank
        """
        G = google_matrix(adjacency)
        initial = 1/2*np.array([np.sqrt(G[j][i])
                                for i, _ in enumerate(G)
                                for j, _ in enumerate(G)])
        print(sum([abs(i)**2 for i in initial]))
        Pi_op = self._Pi_operator(G)
        print(is_unitary(Pi_op))
        swap = self._swap_operator(len(G))
        print(is_unitary(swap))
        operator = (2*Pi_op) - np.identity(len(Pi_op))
        Szegedy = np.dot(operator, swap)
        Szegedy_n = copy.deepcopy(Szegedy)
        if self.step == 0:
            return initial
        elif self.step == 1:
            amp = np.dot(Szegedy, self.initial)
            return amp
        else:
            for n in range(self.step-1):
                Szegedy_n = np.dot(Szegedy_n, Szegedy)
            amp = np.dot(Szegedy_n, initial)
            return amp

    def _Pi_operator(self, ptran):
        '''
        This is not a quantum operation,
        just returning matrix
        '''
        lg = len(ptran)
        psi_op = []
        count = 0
        for i in range(lg):
            psi_vec = [0 for _ in range(lg**2)]
            for j in range(lg):
                psi_vec[count] = np.sqrt(ptran[j][i])
                count += 1
            psi_op.append(np.kron(np.array(psi_vec).T,
                          np.conjugate(psi_vec)).reshape((lg**2, lg**2)))
        Pi = psi_op[0]
        for i in psi_op[1:]:
            Pi = np.add(Pi, i)
        return Pi

    def _swap_operator(self, lad):
        # find closest 2 pow
        base = int(np.ceil(np.log2((1 << int(np.ceil(np.log2(lad)))))))
        swap = np.zeros((lad**2, lad**2))
        for i in range(lad):
            # ibin = format(i, '0%db' % base)
            for j in range(lad):
                # jbin = format(j, '0%db' % base)
                # a = int(ibin + jbin, 2)
                # b = int(jbin + ibin, 2)
                ai = np.array([1 if t == i else 0 for t in range(lad)])
                bi = np.conjugate(np.array([1 if k == j else 0 for k in range(lad)]).T)
                swap += np.kron(ai, bi)
        # raise Exception("")
        return swap

    def _reform(self, ad):
        for il, ln in enumerate(ad):
            rd = random.choice([i for i, _ in enumerate(ln) if i != il])
            if sum(ln) == 1:
                ad[il][rd] = 1
        return ad

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


def prob_transition(graph, gtype='google'):
    if gtype == 'google':
        return google_matrix(graph)
    else:
        pmatrix = np.zeros(graph.shape)
        indegrees = np.sum(graph, axis=0)
        for ix, indeg in enumerate(indegrees):
            if indeg == 0:
                pmatrix[:, ix] = graph[:, ix]
            else:
                pmatrix[:, ix] = graph[:, ix]/indeg
        return pmatrix


def google_matrix(graph, alpha=0.85):
    E = np.zeros((len(graph), len(graph)))
    for i, t in enumerate(graph):
        if sum(graph[:, i]) == 0:
            E[:, i] = np.array([1/len(graph) for _ in graph])
        else:
            for ij, j in enumerate(t):
                E[ij, i] = j/sum(graph[:, i])
    G = alpha*E + (1-alpha)/(len(graph)) * np.ones((len(graph), len(graph)))
    return G
