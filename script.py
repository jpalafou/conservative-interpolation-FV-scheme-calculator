# for development purposes

from classes import Kernel
from classes import Scheme

a = Scheme(Kernel(0,1),'r')
print(a.u_weights)
# print((2*a).u_weights)
print(((a/2) + a).u_weights)
print(a.u_weights)
