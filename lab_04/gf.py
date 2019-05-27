import numpy as np

def gen_pow_matrix(primpoly):
    poly_deg = -1
    curr_poly = primpoly
    while(curr_poly):
        curr_poly = curr_poly // 2
        poly_deg += 1
    if(poly_deg == -1):  # in case of zero polynom
        poly_deg = 0 
    
    table = np.zeros((2 ** poly_deg - 1, 2), np.int)
    alpha = 2
    for i in range(2 ** poly_deg - 1):
        table[i, 1] = alpha
        alpha *= 2
        if alpha >= 2 ** poly_deg:
            alpha = alpha ^ primpoly
    for i in range(2 ** poly_deg - 1):
        table[table[i, 1] - 1, 0] = i + 1
    return table


def add(X, Y):
    return np.bitwise_xor(X, Y)


def sum(X, axis=0):
    return np.bitwise_xor.reduce(X, axis=axis)


def prod(X, Y, pm):
    zeros_matrix = np.logical_and(X, Y).astype(np.int)
    degree_matrix = (pm[X - 1, 0] + pm[Y - 1, 0]) % pm.shape[0]
    return zeros_matrix * pm[degree_matrix - 1, 1]


def divide(X, Y, pm):
    if np.sum(Y == 0) > 0:
         raise ValueError('Division by zero')
    zeros_matrix = X.astype(np.bool).astype(np.int)
    degree_matrix = (pm[X - 1, 0] - pm[Y - 1, 0]) % pm.shape[0]
    return zeros_matrix * pm[degree_matrix - 1, 1]
    

def linsolve(A, b, pm):
    matr_shape = A.shape[0]
    A_full = np.concatenate((A, b[:, np.newaxis]), axis=1)
    for i in range(matr_shape):
        curr_coeff = A_full[i, i]
        k = i + 1
        while(curr_coeff == 0 and k < matr_shape):
            curr_coeff = A_full[k, i]
            if curr_coeff != 0:
                A_full[[i, k], :] = A_full[[k, i], :]
                break
            k += 1
        if curr_coeff == 0:
            return np.nan
        A_full[i, i:] = divide(
                A_full[i, i:], 
                np.ones(matr_shape - i + 1, np.int) * A_full[i, i], 
                pm
                )
        for j in range(i + 1, matr_shape):
            A_full[j, i:] = add(
                    A_full[j, i:], 
                    prod(
                            np.ones(matr_shape - i + 1, np.int) * A_full[j, i],
                            A_full[i, i:], 
                            pm
                            )
                    )
    for i in range(matr_shape - 2, -1, -1):
        A_full[i, -1] = sum(prod(A_full[i:, -1], A_full[i, i:-1], pm))
    return A_full[:, -1]
     
       
def minpoly(x, pm):
    root_set = set()
    for i in range(x.shape[0]):
        x_deg = x[i]
        while not x_deg in root_set:
            root_set.add(x_deg)
            x_deg = prod(x_deg, x_deg, pm)
    result_poly = None
    for elem in root_set:
        if result_poly is None:
            result_poly = np.array([1, elem], np.int)
        else:
            result_poly = polyprod(result_poly, np.array([1, elem], np.int), pm)
    return result_poly, np.array(list(root_set))
      
        
def polyval(p, x, pm):
    p = p[np.argmax(p != 0):]
    deg_array = np.zeros((x.shape[0], p.shape[0]), np.int)
    deg_array[:, -1] = 1
    for i in range(p.shape[0] - 2, -1, -1):
        deg_array[:, i] = prod(deg_array[:, i + 1] , x, pm)
    deg_array = prod(deg_array, p[np.newaxis, :], pm)
    return sum(deg_array, axis=1)


def polyprod(p1, p2, pm):
    p1 = p1.copy()
    p2 = p2.copy()
    if np.all(p1 == 0):
        return np.array([0], np.int)
    if np.all(p2 == 0):
        return np.array([0], np.int)
    p1 = p1[np.argmax(p1 != 0):]
    p2 = p2[np.argmax(p2 != 0):]
    if p1.shape[0] < p2.shape[0]:
        (p1, p2) = (p2, p1)
    p1_shape = p1.shape[0]
    p2_shape = p2.shape[0]
    result_poly_len = p1_shape + p2_shape - 1
    result_poly = np.zeros((p2_shape, result_poly_len), np.int)
    for i in range(p2_shape):
        result_poly[i, result_poly_len - p1_shape - i:result_poly_len - i] = prod(
                p1,
                np.ones(p1_shape, np.int) * p2[p2_shape - 1 - i],
                pm
                )
    result_poly = sum(result_poly, axis=0)
    if np.all(result_poly == 0):
        return np.array([0], np.int)
    result_poly = result_poly[np.argmax(result_poly != 0):]
    return result_poly


def polydivmod(p1, p2, pm):
    p1 = p1.copy()
    p2 = p2.copy()
    if np.all(p2 == 0):
        raise ValueError('division by zero')
    p1 = p1[np.argmax(p1 != 0):]
    p2 = p2[np.argmax(p2 != 0):]
    p1_shape = p1.shape[0]
    p2_shape = p2.shape[0]
    if p1_shape < p2_shape:
        return np.array([0], np.int), p1
    first_coeff = p2[0]
    p2 = divide(p2, np.ones(p2.shape, np.int) * first_coeff, pm)
    result_poly = np.zeros(p1_shape - p2_shape + 1, np.int)
    for i in range(result_poly.shape[0]):
        result_poly[i] = p1[i]
        p1[i: i + p2_shape] = add(
                p1[i: i + p2_shape], 
                prod(p2, np.ones(p2_shape, np.int) * p1[i], pm)
                )
    p1 = p1[np.argmax(p1 != 0):]
    if np.argmax(p1 != 0) == 0 and p1[0] == 0:
        p1 = np.array([0], np.int)
    result_poly = divide(result_poly, np.ones(result_poly.shape[0], np.int) * first_coeff, pm)
    return result_poly, p1


def polyadd(p1, p2):
    max_deg = max(p1.shape[0], p2.shape[0])
    p1 = p1.copy()
    p2 = p2.copy()
    p1 = np.concatenate((np.zeros(max_deg - p1.shape[0], np.int), p1))
    p2 = np.concatenate((np.zeros(max_deg - p2.shape[0], np.int), p2))
    result = add(p1, p2)
    result = result[np.argmax(result != 0):]
    return result
    

def euclid(p1, p2, pm, max_deg=0):
    p1 = p1.copy()
    p2 = p2.copy()
    x_0 = np.array([1], np.int)
    y_0 = np.array([0], np.int)
    x_1 = np.array([0], np.int)
    y_1 = np.array([1], np.int)
    if  p1.shape[0] < p2.shape[0]:
        (p1, p2) = (p2, p1)
    while p2.shape[0] > max_deg + 1:
        q, d = polydivmod(p1, p2, pm)
        (p1, p2) = (p2, d)
        (x_0, x_1) = (x_1, polyadd(x_0, polyprod(x_1, q, pm)))
        (y_0, y_1) = (y_1, polyadd(y_0, polyprod(y_1, q, pm)))
    return p2, x_1, y_1