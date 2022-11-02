import pytest
from classes import Kernel
from classes import Scheme

def test_2nd_order_central_difference():
    d_dx = {-1:-1/2, 1:1/2}
    right = Scheme(Kernel(0,1),'r')
    left = Scheme(Kernel(1,0),'l')
    assert (right - left).u_weights == d_dx


def test_2nd_order_Fromm():
    right_solution = {-1:-1/4, 0:1, 1:1/4}
    right_average = (Scheme(Kernel(0,1),'r') + Scheme(Kernel(1,0),'r'))/2
    assert right_average.u_weights == right_solution

    left_solution = {-1:1/4, 0:1, 1:-1/4}
    left_average = (Scheme(Kernel(0,1),'l') + Scheme(Kernel(1,0),'l'))/2
    assert left_average.u_weights == left_solution

    d_dx_solution = {-2:1/4, -1:-5/4, 0:3/4, 1:1/4}
    d_dx = (Scheme(Kernel(0,1),'r') + Scheme(Kernel(1,0),'r'))/2 - \
    (Scheme(Kernel(0,1,-1),'r') + Scheme(Kernel(1,0,-1),'r'))/2
