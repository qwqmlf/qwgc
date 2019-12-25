import numpy as np


class GraphInfo:
    def __init__(self, adjacency_matrix):
        self.adja = adjacency_matrix
        self.dim = len(adjacency_matrix)

    def shift(self):
        deg = self._degrees
        node_index = self._first_node()
        ndim = self.dim
        size = node_index[ndim-1] + len(deg[ndim-1])
        deg2 = [[0] for _ in deg]
        for i, v in enumerate(deg):
            deg2[i] = list(set(v))
        array = np.zeros((size, size), dtype=int)
        for i, val in enumerate(deg):
            index1 = node_index[i]
            n = 0
            for j, jval in enumerate(deg2[i]):
                nolinks = self.adja[jval][i]
                n = self._coinst(i, val[j])
                coinst = index1 + n
                node = val[j]
                for k in deg2[node]:
                    if k == i:
                        coinst2 = node_index[node] + self._coinst(node, i)
                for k in range(int(nolinks)):
                    array[coinst+k][coinst2+k] = 1
        # FIXME catch before
        for i, v in enumerate(array):
            for j, w in enumerate(array):
                if array[i][j] == 1:
                    array[j][i] = 1
        return array

    @property
    def _degrees(self):
        degrees = [[] for _ in range(self.dim)]
        for iv, v in enumerate(self.adja):
            for ix, x in enumerate(v):
                if x == 1:
                    degrees[iv].append(ix)
        return degrees

    def _first_node(self):
        n = 0
        array = np.zeros(len(self._degrees), dtype=int)
        for i, v in enumerate(self._degrees):
            array[i] = n
            n += len(v)
        return array

    def _coinst(self, i, j):
        n1 = 0
        n2 = 0
        idim = len(self._degrees[i])
        for k in range(idim):
            if n2 == 0:
                if self._degrees[i][k] != j:
                    n1 += 1
                else:
                    n2 = 1
        if n1 == idim:
            return 'no link'
        else:
            return n1
