import numpy as np
import math

# helper functions
def binomial_prod(M):
    """
    find the product of multiple binomials of the form
    (x + a)(x + b)(x + c)...
    expressed as an array M
    [a, 1
     b, 1
     c, 1
     ...]

    the output will be the resulting polynome
    e x^0 + f x^1 + g x^2 + ...
    expressed as an array polynome
    [e, f, g, ...]
    """
    polynome = M[0, :]
    for i in range(1, M.shape[0]):
        step1 = np.append(0, polynome)  # multiply by x
        step2 = np.append(polynome * M[i, 0], 0)  # multiply by coefficient
        # include 0 for the highest degree of x
        polynome = step1 + step2
    return polynome


def lagrange(x, i):
    """
    find the ith Lagrange polynome for f with data at x
    l = a x^0 + b x^1 + c x^2 + ...
    expressed as an array
    [a, b, c, ...]
    """
    denominator = 1
    M = np.empty((0, 2))  # [null, null]
    for j in range(len(x)):
        if i != j:
            denominator *= x[i] - x[j]
            M = np.vstack((M, np.array([-x[j], 1])))
    return binomial_prod(M) / denominator


def poly_prime(polynome):
    """
    return the derivative of a polynome
    a x^0 + b x^1 + c x^2 + ...
    expressed as an array
    [a, b, c, ...].
    the derivative is also expressed as an array and will contain one less element
    """
    return (polynome * np.array(list(range(len(polynome)))))[1:]


def poly_eval(polynome, x):
    """
    evaluate f(x) for a polynomial f
    a x^0 + b x^1 + c x^2 + ...
    expressed as an array
    [a, b, c, ...]
    """
    f = 0
    for i in range(len(polynome)):
        f = f + polynome[i] * x**i
    return f


# classes
class Kernel:
    """
    provide information about a finite volume scheme based based on a kernel
    definition
    """

    def __init__(self, left, right, u_index_at_center=0):
        """
        left:   number of cells left of center in the kernel
        right:  number of cells right of center
        u_index_at_center:      what is the subscript of u at the central cell
        """
        self.center = left  # index of the center cell
        self.size = left + right + 1  # number of cells in the kernel
        self.u_index_at_center = u_index_at_center

        # kernel step size
        self.h = 1

        # generate an array of x-values for the cell centers
        x = np.arange(-left, right + self.h, self.h)
        self.x_cell_centers = x

        # generate an array of x-values for the cell faces
        self.x_cell_faces = np.append(x[0] - self.h / 2, x + self.h / 2)

    def kernel_print(self):
        """
        give printout of the kernel showing cell interfaces and indeces
        """
        for i in range(self.size):
            print("| " + str(i - self.center + self.u_index_at_center), end=" ")
        print("|")

    def solve_u_at_interface(self, face="right"):
        """
        find the weighted average of the cells values u_0, u_1, etc that find
        the actual value at the inteface

        note that Teyssier uses a slighlty different notation. u represents the
        actual value at the interface and \bar{u} represents the cell value
        """

        x = self.x_cell_faces
        c = self.center
        n = len(x)

        if face == "right" or face == "r":
            x_eval = x[c + 1]
        elif face == "left" or face == "l":
            x_eval = x[c]
        else:
            fprintf("ERROR! No interface x-value provided.")
            return

        u_weights_polynomes = np.zeros([n - 1, n])  # each row contains a polynome
        u_weights_polynomes_prime = np.zeros([n - 1, n - 1])
        u_weights = np.zeros(n - 1)  # scalar values
        u_weights_dict = {}

        # find the polynomial expression being multiplied to each u term
        for i in range(1, n):  # skip first polynome because m = 0
            lagrange_i = lagrange(x, i)
            for j in range(i):
                u_weights_polynomes[j, :] = u_weights_polynomes[j, :] + lagrange_i

        # find the derivative of each of these polynomials and evaluate at the
        # specified point
        for i in range(n - 1):
            u_weights_polynomes_prime[i, :] = poly_prime(u_weights_polynomes[i, :])
            u_weights[i] = self.h * poly_eval(u_weights_polynomes_prime[i, :], x_eval)
            u_weights_dict[i - c + self.u_index_at_center] = u_weights[i]

        return u_weights_dict


class Scheme:
    """
    enable addition and subtraction of interface schemes
    """

    def __init__(self, kernel, face="right"):
        """
        kernel: object
        face:   which face of the central cell is the scheme evaluating
        """
        self.u_weights = kernel.solve_u_at_interface(face)

    def __add__(self, other):
        dict1 = self.u_weights
        dict2 = other.u_weights
        newdict = {}

        # add weights that exist for the same u
        for i in set(dict1.keys()).intersection(dict2.keys()):
            # POTENTIAL BUG what if the difference is almost 0
            if dict1[i] + dict2[i] != 0:  # don't include if 0
                newdict[i] = dict1[i] + dict2[i]

        for i in dict1.keys():
            if i not in dict2.keys():
                newdict[i] = dict1[i]

        for i in dict2.keys():
            if i not in dict1.keys():
                newdict[i] = dict2[i]

        # create new Scheme object out of a meaningless Kernel object
        mysum = Scheme(Kernel(0,0), 'l')
        mysum.u_weights = dict(sorted(newdict.items()))
        return mysum


    def __sub__(self, other):
        dict1 = self.u_weights
        dict2 = other.u_weights
        newdict = {}

        # add weights that exist for the same u
        for i in set(dict1.keys()).intersection(dict2.keys()):
            # POTENTIAL BUG what if the difference is almost 0
            if dict1[i] - dict2[i] != 0:  # don't include if 0
                newdict[i] = dict1[i] - dict2[i]

        for i in dict1.keys():
            if i not in dict2.keys():
                newdict[i] = dict1[i]

        for i in dict2.keys():
            if i not in dict1.keys():
                newdict[i] = -dict2[i]


        # create new Scheme object out of a meaningless Kernel object
        mydifference = Scheme(Kernel(0,0), 'l')
        mydifference.u_weights = dict(sorted(newdict.items()))
        return mydifference

    def __mul__(self, other):
        mydict = self.u_weights
        newdict = {}
        for i in mydict.keys():
            newdict[i] = other*mydict[i]
        # create new Scheme object out of a meaningless Kernel object
        product = Scheme(Kernel(0,0), 'l')
        product.u_weights = newdict
        return product

    def __rmul__(self, other):
        mydict = self.u_weights
        newdict = {}
        for i in mydict.keys():
            newdict[i] = other*mydict[i]
        # create new Scheme object out of a meaningless Kernel object
        product = Scheme(Kernel(0,0), 'l')
        product.u_weights = newdict
        return product

    def __truediv__(self, other):
        mydict = self.u_weights
        newdict = {}
        for i in mydict.keys():
            newdict[i] = mydict[i]/other
        # create new Scheme object out of a meaningless Kernel object
        quotient = Scheme(Kernel(0,0), 'l')
        quotient.u_weights = newdict
        return quotient

    def scheme_print(self):
        """
        give printout of the weighted average
        """
        mydict = self.u_weights
        for i in list(mydict.keys())[:-1]:
            print(str(mydict[i]) + " * u_" + str(i), end=" + ")
        i = list(mydict.keys())[-1]
        print(str(mydict[i]) + " * u_" + str(i))
