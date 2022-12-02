import numpy as np

def binomial_prod(M):
    '''
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
    '''
    polynome = M[0,:]
    for i in range(1,M.shape[0]):
        step1 = np.append(0, polynome) # multiply by x
        step2 = np.append(polynome*M[i,0],0) # multiply by coefficient
        # include 0 for the highest degree of x
        polynome = step1 + step2
    return polynome

def lagrange(x, i):
    '''
    find the ith Lagrange polynome for f with data at x
    l = a x^0 + b x^1 + c x^2 + ...
    expressed as an array
    [a, b, c, ...]
    '''
    denominator = 1
    M = np.empty((0,2)) # [null, null]
    for j in range(len(x)):
        if i != j:
            denominator *= x[i] - x[j]
            M = np.vstack((M, np.array([-x[j], 1])))
    return binomial_prod(M) / denominator

def poly_prime(polynome):
    '''
    return the derivative of a polynome
    a x^0 + b x^1 + c x^2 + ...
    expressed as an array
    [a, b, c, ...].
    the derivative is also expressed as an array and will contain one less element
    '''
    return (polynome*np.array(list(range(len(polynome)))))[1:]

def poly_eval(polynome,x):
    '''
    evaluate f(x) for a polynomial f
    a x^0 + b x^1 + c x^2 + ...
    expressed as an array
    [a, b, c, ...]
    '''
    f = 0
    for i in range(len(polynome)):
        f = f + polynome[i] * x**i
    return f

def FV_scheme(x_ep, h, c, interface = 'r'):
    n = len(x_ep)
    if interface == 'r':
        x_eval = x_ep[c + 1]
    elif interface == 'l':
        x_eval = x_ep[c]
    else:
        fprintf('No interface x-value provided.')
        return

    u_weights_polynomes = np.zeros([n - 1, n]) # each row contains a polynome
    u_weights_polynomes_prime = np.zeros([n - 1, n - 1])
    u_weights = np.zeros(n - 1) # scalar values
    u_weights_dict = {}

    for i in range(1, n): # skip first polynome because m = 0
        lagrange_i = lagrange(x_ep, i)
        for j in range(i):
            u_weights_polynomes[j,:] = u_weights_polynomes[j,:] + lagrange_i

    for i in range(n - 1):
        u_weights_polynomes_prime[i,:] = poly_prime(u_weights_polynomes[i,:])
        u_weights[i] = h * poly_eval(u_weights_polynomes_prime[i,:], x_eval)
        u_weights_dict[i - c] = u_weights[i]

    return u_weights_dict


n = 2
x = np.linspace(0,2,n)      # locations of the cells i have data for
h = x[1] - x[0]             # mesh size in x
x_ep = np.append(x[0] - h / 2, x + h / 2)   # locations of cell end points


print(FV_scheme(x_ep, h, 0, 'r'))
# l = 1       # number of kernel points to the left of center
# r = 1       # number of kernel points to the right of center
#
# x_plot = 1  # finer x mesh for plotting
#
#
# order = l + r
#
# x_ref = x[:order+1]
