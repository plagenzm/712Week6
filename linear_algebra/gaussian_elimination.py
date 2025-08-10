"""
| Gaussian elimination method for solving a system of linear equations.
| Gaussian elimination - https://en.wikipedia.org/wiki/Gaussian_elimination
"""

import numpy as np
from numpy import float64
from numpy.typing import NDArray


def retroactive_resolution(
    coefficients: NDArray[float64], vector: NDArray[float64]
) -> NDArray[float64]:
    """
    This function performs a retroactive linear system resolution
    for triangular matrix

    Examples:
        1.
            * 2x1 + 2x2 - 1x3 = 5
            * 0x1 - 2x2 - 1x3 = -7
            * 0x1 + 0x2 + 5x3 = 15
        2.
            * 2x1 + 2x2 = -1
            * 0x1 - 2x2 = -1

    >>> gaussian_elimination([[2, 2, -1], [0, -2, -1], [0, 0, 5]], [[5], [-7], [15]])
    array([[2.],
           [2.],
           [3.]])
    >>> gaussian_elimination([[2, 2], [0, -2]], [[-1], [-1]])
    array([[-1. ],
           [ 0.5]])
    """

    rows, columns = np.shape(coefficients)

    x: NDArray[float64] = np.zeros((rows, 1), dtype=float)
    for row in reversed(range(rows)):
        total = np.dot(coefficients[row, row + 1 :], x[row + 1 :])
        x[row, 0] = (vector[row][0] - total[0]) / coefficients[row, row]

    return x


#MP note: seperated augmented matrix code to further encapsulate for clean code, to help debug, and reusability. Copilot confirmed it was a good decision to encapsulate and code was from Copilot
def create_augmented_matrix(coefficients: NDArray[float64], vector: NDArray[float64]) -> NDArray[float64]:
    augmented_mat = np.concatenate((coefficients, vector), axis=1)
    return augmented_mat.astype("float64")


# MP note: wanted to add error handler to this function. The error handlers are from Copilot and check inputs to ensure they can be converted to float64
def gaussian_elimination(
    coefficients, vector
) -> NDArray[float64]:
    try:
        coefficients = np.array(coefficients, dtype=float64)
        vector = np.array(vector, dtype=float64)
    except (TypeError, ValueError):
        raise TypeError("Inputs must be convertible to float64 NumPy arrays.")
    #MP note: extra check for irregular inputs
    if not (np.issubdtype(coefficients.dtype, np.number) and np.issubdtype(vector.dtype, np.number)):
        raise TypeError("Please ensure all inputs are numbers")

    """
    This function performs Gaussian elimination method

    Examples:
        1.
            * 1x1 - 4x2 - 2x3 = -2
            * 5x1 + 2x2 - 2x3 = -3
            * 1x1 - 1x2 + 0x3 = 4
        2.
            * 1x1 + 2x2 = 5
            * 5x1 + 2x2 = 5

    >>> gaussian_elimination([[1, -4, -2], [5, 2, -2], [1, -1, 0]], [[-2], [-3], [4]])
    array([[ 2.3 ],
           [-1.7 ],
           [ 5.55]])
    >>> gaussian_elimination([[1, 2], [5, 2]], [[5], [5]])
    array([[0. ],
           [2.5]])
    """
    # coefficients must to be a square matrix so we need to check first
    rows, columns = np.shape(coefficients)
    if rows != columns:
        return np.array((), dtype=float)
    #MP note: added error message. If this wasn't here the program would run and then finish with no messages. Copilot mentioned using it only temporarily for debugging as it can clutter output later
    else:
        print("Matrix is square")

    #MP note: calling the create_augmented_matrix function. Code was from Copilot. 
    augmented_mat = create_augmented_matrix(coefficients, vector)

    # scale the matrix leaving it triangular
    for row in range(rows - 1):
        pivot = augmented_mat[row, row]
        for col in range(row + 1, columns):
            factor = augmented_mat[col, row] / pivot
            augmented_mat[col, :] -= factor * augmented_mat[row, :]

    x = retroactive_resolution(
        augmented_mat[:, 0:columns], augmented_mat[:, columns : columns + 1]
    )

    return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
