'''
Thank you Barr Katie for providing your code.
I reffered this paper and code.
Barr, Katie, Toby Fleming, and Viv Kendon.
"Simulation methods for quantum walks on
graphs applied to perfect state transfer and
formal language recognition." CoSMoS 2013 (2013): 1.
'''
import numpy as np
import numpy.linalg as la
from .gparse import GraphInfo


class QuantumWalk:

    def __init__(self, initial_state, coin_operator, adjacency_matrix,
                 tolerance=1e-5):
        '''
        initial_state: None or array like
        coin_operator: array like
        adjacency_matrix: array like
        '''
        gf = GraphInfo(adjacency_matrix)
        if not is_unitary(coin_operator):
            raise ValueError('Coin operator must be unitary')
        else:
            self.coin_operator = coin_operator
        self.shift_operator = gf.shift()
        self.time_ev = np.dot(self.shift_operator, coin_operator)
        eigs, _ = la.eig(self.time_ev)
        if initial_state is None:
            self.create_default_initial()
        else:
            self.initial_state = initial_state
        self.current_state = initial_state
        self.adjacency = adjacency_matrix
        self.dim = len(adjacency_matrix)
        self.tolerance = tolerance

    def create_default_initial(self):
        basis_state = self.time_ev.shape[0]
        vec = np.zeros((basis_state))
        vec[0] = 1
        self.initial_state = vec

    def step(self):
        self.current_state = np.dot(self.time_ev, self.current_state)

    def step_back(self):
        self.current_state = np.dot(np.concatenate(self.time_ev.transpose()),
                                    self.current_state)

    def steps(self, n):
        for i in range(n):
            state = np.dot(self.time_ev, self.current_state)
        self.current_state = state

    @property
    def node_deg(self):
        return [int(np.sum(self.adjacency[i])) for i in range(self.dim)]

    def prob_at_node(self, index):
        if index > self.dim:
            raise ValueError('Graph does not have %d nodes' % index)
        probs = self.calc_probs
        return probs[index]

    def calc_probs(self):
        probs = np.zeros(self.dim)
        ind = 0
        for i in range(self.dim):
            for j in range(self.node_deg[i]):
                amps_at_j = self.current_state[ind]
                probs[i] += amps_at_j * np.conjugate(amps_at_j)
                ind += 1
        assert np.isclose(np.sum(probs), 1, atol=self.tolerance)
        return probs

    def calc_amp(self):
        # FIXME in this implementation, we can't get amplitudes of each node
        return self.current_state

    def n_steps(self, steps):
        if steps < 0:
            raise ValueError('steps must be 0 or over')
        elif steps == 0:
            self.current_state = self.initial_state
            return self.current_state

        eig = la.eig(self.time_ev)[1]
        inverse = la.inv(eig)
        diag = np.dot(np.dot(inverse, self.time_ev), eig)

        for i, _ in enumerate(diag):
            transition = diag[i][i]
            x = transition.real
            y = transition.imag
            theta = np.arctan2(y, x)
            diag[i][i] = np.cos(steps*theta) + complex(0, np.sin(steps*theta))
        transform_bvec = np.dot(inverse, self.initial_state)
        evolved = np.dot(diag, transform_bvec)
        trans_back = np.dot(eig, evolved)
        self.current_state = trans_back
        return trans_back


def is_unitary(operator):
    h, w = operator.shape
    if not h == w:
        return False
    adjoint = np.conjugate(operator.transpose())
    product1 = np.dot(operator, adjoint)
    product2 = np.dot(adjoint, operator)
    ida = np.eye(h)
    return np.allclose(product1, ida) & np.allclose(product2, ida)
