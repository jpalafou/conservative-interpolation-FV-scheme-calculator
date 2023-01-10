import pytest  # run > pytest -s -q -rA
import numpy as np
import math
from util.fvscheme import Kernel

n_tests = 5
max_left_cells = 20
max_right_cells = 20

# helper functions
@pytest.fixture
def kern():
    """
    generate a random kernel with l cells to the left and r cells to the right
    of center
    """
    l = np.random.randint(max_left_cells + 1)
    r = np.random.randint(max_right_cells + 1)
    adj_index_at_center = np.random.randint(-l - 1, r + 1)
    return Kernel(l, r, adj_index_at_center)


# tests
def test_kernel_setup_SingleCellKernel():
    """
    make sure it is possible to create a single-celled kernel
    """
    assert Kernel(0, 0).size == 1
    assert len(Kernel(0, 0).x_cell_centers) == 1
    assert len(Kernel(0, 0).x_cell_faces) - 1 == 1


@pytest.mark.parametrize("unused_parameter", range(n_tests))  # test n_tests times
def test_kernel_setup_random(unused_parameter, kern):
    """
    test that kernel properties are consistent when a kernel is set up randomly
    """
    assert kern.size == len(kern.x_cell_centers)
    assert kern.size == len(kern.x_cell_faces) - 1
