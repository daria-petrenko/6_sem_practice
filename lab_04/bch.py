import numpy as np
import gf

class BCH:
    def __init__(self, n, t):
        self.n = n
        self.t = t
        q = -1
        n_copy = n + 1
        while n_copy:
            n_copy = n_copy // 2
            q += 1
        if 2 ** q - 1 != n:
            raise ValueError('n is not 2^{q} - 1')
        with open('primpoly.txt', 'r') as file:
            primpoly_list = file.read().split(', ')
        for primpoly in primpoly_list:
            if int(primpoly) // 2 ** (q):  # if degree == q
                break
        primpoly = int(primpoly)
        self.pm = gf.gen_pow_matrix(primpoly)
        alpha = np.array([2], np.int)
        curr_poly = alpha
        degrees_list = [alpha[0]]
        for i in range(2 * t - 1):
            curr_poly = gf.prod(curr_poly, alpha, self.pm)
            degrees_list.append(curr_poly[0])
        self.g, self.R = gf.minpoly(np.array(degrees_list), self.pm)
        self.deg_g = self.g.shape[0] - 1
        
    def encode(self, U):
        result_matr = np.zeros((U.shape[0], U.shape[1] + self.deg_g), np.int)
        for i in range(U.shape[0]):
            result_matr[i, : U.shape[1]] = U[i, :]
            _, d = gf.polydivmod(result_matr[i], self.g, self.pm)
            result_matr[i, - d.shape[0]:] = d
        return result_matr
    
    def dist(self):
        k = self.n - self.deg_g
        min_dist = self.n
        for i in range(1, 2 ** k):
            u = np.array([[int(x) for x in np.binary_repr(i)]])
            u = np.concatenate((np.zeros((1, k - u.shape[1]), np.int), u), axis=1)
            v = self.encode(u)
            min_dist = min(min_dist, np.sum(v))
        return min_dist
    
    def decode(self, W, method='euclid'):
        alpha = np.array([2], np.int)
        curr_deg = alpha
        alpha_list_t = [alpha[0]]
        for i in range(2 * self.t - 1):
            curr_deg = gf.prod(curr_deg, alpha, self.pm)
            alpha_list_t.append(curr_deg[0])
        alpha_list_n = alpha_list_t.copy()
        for i in range(self.n - 2 * self.t):
            curr_deg = gf.prod(curr_deg, alpha, self.pm)
            alpha_list_n.append(curr_deg[0])
        alpha_list_t = np.array(alpha_list_t)
        alpha_list_n = np.array(alpha_list_n)
        result = np.zeros(W.shape, np.int)
        if method == 'pgz':
            for row in range(W.shape[0]):
                deg_list = gf.polyval(W[row], alpha_list_t, self.pm)
                if np.all(deg_list == 0):
                    result[row] = W[row]
                else:
                    curr_dim = self.t 
                    while(curr_dim > 0):
                        A = np.zeros((curr_dim, curr_dim), np.int)
                        for i in range(curr_dim):
                            A[i] = deg_list[i: curr_dim + i]
                        b = deg_list[curr_dim: 2 * curr_dim]
                        value = gf.linsolve(A, b, self.pm)
                        if value is not np.nan:
                            value = np.concatenate((value, np.ones(1, np.int)))
                            break
                        curr_dim -= 1
                    if curr_dim == 0:
                        result[row] = np.nan
                    else:
                        roots = gf.polyval(value, alpha_list_n, self.pm)
                        roots = np.nonzero(np.logical_not(roots))
                        err_pos = (roots[0]) % self.n
                        result[row] = W[row]
                        result[row, err_pos] = np.logical_not(result[row, err_pos]).astype(np.int)
                        if np.any(gf.polyval(result[row], alpha_list_t, self.pm) != 0):
                             result[row, :] = np.nan
        elif method == 'euclid':
            alpha_list_t = alpha_list_t[::-1]
            for row in range(W.shape[0]):
                S = gf.polyval(W[row], alpha_list_t, self.pm)
                S = np.concatenate((S, np.ones(1, np.int)))
                z = np.array([1] + [0 for x in range(2 * self.t + 1)], np.int)
                r, A, Lambda = gf.euclid(z, S, self.pm, self.t)
                roots = gf.polyval(Lambda, alpha_list_n, self.pm)
                roots = np.nonzero(np.logical_not(roots))
                err_pos = (roots[0]) % self.n
                result[row] = W[row]
                result[row, err_pos] = np.logical_not(result[row, err_pos]).astype(np.int)
                if np.any(gf.polyval(result[row], alpha_list_t, self.pm) != 0):
                     result[row, :] = np.nan
        return result     