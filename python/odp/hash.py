import random
import numpy as np

PRIME = 105943

def H(label, a, b, B, p=PRIME):
    # label is an int and B number of partitions
    return ((a * label + b) % p) % B

def hash_factory(a, b, B, p=PRIME):
    """
    generate a hash function based on parameters a, b, B and p
    """
    return lambda label : ((a * label + b) % p) % B

def generate_ab(R, p=PRIME):
    a = []
    b = []
    for i in range(R):
        tmp = random.randrange(1, p)
        while tmp in a:
            tmp = random.randrange(1, p)
        a.append(tmp)
        tmp = random.randrange(1, p)
        while tmp in b:
            tmp = random.randrange(0, p)
        b.append(tmp)

    return a, b

def hash_vector(y, a, b, B, p):
    mapper = lambda x: ((a * x + b) % p) % B
    vfunc = np.vectorize(mapper)
    y_h = vfunc(y)
    return y_h

def hash_and_test(y_train, R, B, p=PRIME):
    # print("in hash_and_test")
    a, b = generate_ab(R, p)
    y = np.zeros((R, y_train.shape[0]))
    for i in range(R):
        y[i] = hash_vector(y_train, a[i], b[i], B, p)
        # print("in model %d the number of unique labels after hash is %d" % (i, len(np.unique(y[i]))))
        # hash function may lead to less than B categories
        while len(np.unique(y[i])) != B:
            # print("hash failed, rehashing")
            tmp_a = get_a(p)
            tmp_b = get_b(p)
            while tmp_a in a:
                tmp_a = get_a(p)
            while tmp_b in b:
                tmp_b = get_b(p)
            a[i] = tmp_a
            b[i] = tmp_b
            y[i] = hash_vector(y_train, a[i], b[i], B, p)
            # print("in model %d the number of unique labels after hash is %d" % (i, len(np.unique(y[i]))))
    return a, b, y

def get_hash_params(num_labels, R, B, p=PRIME):
    """
    generate R pairs of hash parameters a and b and ensure that the hashed
    result will have B different kinds.

    params:
    num_labels - total number of classes
    R - paris of hash parameters
    B - number of buckets
    p - a large prime number

    returns:
    a, b - hash parameters
    """
    a, b = generate_ab(R, p)
    y = np.array(list(range(num_labels)))
    for i in range(R):
        y_hash = hash_vector(y, a[i], b[i], B, p)
        # hash function may lead to less than B categories
        while len(np.unique(y_hash)) != B:
            print("hash failed, rehashing")
            tmp_a = get_a(p)
            tmp_b = get_b(p)
            while tmp_a in a:
                tmp_a = get_a(p)
            while tmp_b in b:
                tmp_b = get_b(p)
            a[i] = tmp_a
            b[i] = tmp_b
        # print("in model %d the number of unique labels after hash is %d" % (i, len(np.unique(y_hash))))
    return a, b

def get_a(p=PRIME):
    return random.randrange(1, p)

def get_b(p=PRIME):
    return random.randrange(0, p)
