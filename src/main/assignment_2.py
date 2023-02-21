import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)


def neville_method(x_points, y_points, x):
    size: int = len(x_points)

    # create square matrix size x size
    matrix = np.zeros((size, size))

    # fill first column with y values
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]

    # populate final matrix
    for i in range(1, size):
        for j in range(1, i + 1):
            first_multiplication: float = (x - x_points[i - j]) * matrix[i][j - 1]
            second_multiplication: float = (x - x_points[i]) * matrix[i - 1][j - 1]
            denominator: float = x_points[i] - x_points[i - j]
            coefficient: float = (first_multiplication - second_multiplication) / denominator
            matrix[i][j] = coefficient

    # print(matrix)
    print(matrix[size - 1][size - 1])
    print()

    return None


def divided_difference_table(x_points, y_points):
    size: int = len(x_points)

    # create square matrix size x size
    matrix = np.zeros((size, size))

    # fill first column with y values
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]

    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        for j in range(1, i + 1):
            # the numerator is the immediate left and diagonal left indices...
            numerator: float = matrix[i][j - 1] - matrix[i - 1][j - 1]

            # the denominator is the X-SPAN.
            denominator: float = x_points[i] - x_points[i - j]

            operation: float = numerator / denominator

            matrix[i][j] = operation

    # print(matrix)
    return matrix


def get_approximate_result(matrix, x_points, value):
    size: int = len(x_points)

    # initialize x_span to 1, initialize p0 to y0
    reoccurring_x_span = 1
    reoccurring_px_result = matrix[0][0]

    # we only need the diagonals for polynomial coefficient
    for index in range(1, size):
        polynomial_coefficient = matrix[index][index]

        # we use the previous index for x_points....
        reoccurring_x_span *= (value - x_points[index - 1])

        # get a(x) * the x_span
        mult_operation = polynomial_coefficient * reoccurring_x_span

        # add the reoccurring px result
        reoccurring_px_result += mult_operation

    # final result
    return reoccurring_px_result


def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):
            # skip if value is prefilled
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # get left cell entry
            left: float = matrix[i][j - 1]

            # get diagonal left cell entry
            diagonal_left: float = matrix[i - 1][j - 1]

            numerator: float = left - diagonal_left
            denominator: float = matrix[i][0] - matrix[i - (j - 1)][0]
            operation = numerator / denominator

            # fill matrix
            matrix[i][j] = operation

    return matrix


def hermite_interpolation(x_points, y_points, slopes):
    num_of_points = len(x_points)
    matrix = np.zeros((2 * num_of_points, 2 * num_of_points))

    # populate x values
    for index in range(0, 2 * num_of_points, 2):
        matrix[index][0] = x_points[int(index / 2)]
        matrix[index+1][0] = x_points[int(index / 2)]

    # populate y values
    for index in range(0, 2 * num_of_points, 2):
        matrix[index][1] = y_points[int(index / 2)]
        matrix[index + 1][1] = y_points[int(index / 2)]

    # populate with derivatives
    for index in range(0, 2 * num_of_points, 2):
        matrix[index + 1][2] = slopes[int(index / 2)]

    filled_matrix = apply_div_dif(matrix)

    print(filled_matrix)
    print()


if __name__ == "__main__":
    # (1) Using Neville’s method, find the 2nd degree interpolating value for f(3.7)
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximating_value = 3.7
    neville_method(x_points, y_points, approximating_value)

    # (2) Using Newton’s forward method, print out the polynomial approximations for degrees 1, 2, and 3
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    divided_table = divided_difference_table(x_points, y_points)
    NUM_DEGREES: int = 3
    polynomial_coefficient_array = np.zeros(NUM_DEGREES)
    for index in range(0, NUM_DEGREES):
        polynomial_coefficient_array[index] = divided_table[index + 1][index + 1]
    print(list(polynomial_coefficient_array))
    print()

    # (3) Using the results from 2, approximate f(7.3)
    approximating_x = 7.3
    final_approximation = get_approximate_result(divided_table, x_points, approximating_x)
    print(final_approximation)
    print()

    # (4) Using the divided difference method, print out the Hermite polynomial approximation matrix
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    hermite_interpolation(x_points, y_points, slopes)

    # (5) Using cubic spline interpolation, solve for the following
    x_points = [2, 5, 8, 10]
    y_points = [3, 5, 7, 9]
    size = len(x_points)
    delta_x = np.diff(x_points)
    delta_y = np.diff(y_points)

    # (a) Matrix A
    A = np.zeros(shape=(size, size))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, size - 1):
        A[i, i - 1] = delta_x[i - 1]
        A[i, i + 1] = delta_x[i]
        A[i, i] = 2 * (delta_x[i - 1] + delta_x[i])

    print(A)
    print()

    # (b) Vector b
    b = np.zeros(shape=(size, 1))
    for i in range(1, size - 1):
        b[i, 0] = 3 * (delta_y[i] / delta_x[i] - delta_y[i - 1] / delta_x[i - 1])

    print(b.reshape(-1))
    print()

    # (c) Vector x
    x = np.dot(np.linalg.inv(A), b)

    print(x.reshape(-1))
