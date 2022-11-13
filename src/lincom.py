import dataclasses

@dataclasses.dataclass
class LinearCombination:
    """
    class that describes linear combinations of terms
    a u_0 + b u_1 + c u_2 + ...
    as a dictionary
    {0: a, 1: b, 2: c, ...}
    enables addition and subtraction between linear combinations
    """

    coeffs: dict # {int: int}

    def __post_init__(self):
        """
        linear combinations should be sorted and should not contain
        coefficients of 0 unless they are the zero instance {0: 0}
        """
        # if coeffs is empty or if all its values are zero and 0 is not the only
        # index
        if self.coeffs == {} or (all(j == 0 for j in self.coeffs.values()) and \
        list(self.coeffs.keys()) != [0]):
            object.__setattr__(self, "coeffs", {0: 0})
        else: # coeffs is nonempty and contains at least one nonzero item
            # sort by degree
            sorted_coeffs = dict(sorted(self.coeffs.items()))
            # remove 0 coefficients if they unless 0x^0 is the only term
            new_coeffs = {}
            for deg, coeff in sorted_coeffs.items():
                if coeff != 0 or (deg == 0 and len(new_coeffs) == 0):
                    new_coeffs[deg] = sorted_coeffs[deg]
            object.__setattr__(self, "coeffs", new_coeffs)

    def __str__(self):
        strings = [f"{coeff} u_{ind}" for (ind, coeff) in self.coeffs.items()]
        return " + ".join(strings) if strings else str(self.zero())

    def zero():
        return LinearCombination({0: 0})

    def __add__(self, other):
        coeffs_sum = {}
        for i in self.coeffs.keys():
            if i in other.coeffs.keys():
                coeffs_sum[i] = self.coeffs[i] + other.coeffs[i]
            else:
                coeffs_sum[i] = self.coeffs[i]
        for i in other.coeffs.keys():
            if i not in self.coeffs.keys():
                coeffs_sum[i] = other.coeffs[i]
        return self.__class__(coeffs_sum)

    def __neg__(self):
        return self.__class__(dict([(i, -j) for i, j in self.coeffs.items()]))

    def __sub__(self, other):
        return self + -other
