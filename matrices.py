import numpy as np


def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m.

    Args:
        n (int): number of rows.
        m (int): number of columns.

    Returns:
        np.ndarray: matrix n*m.
    """
    return np.random.rand(n,m)

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: matrix sum.
    """
    return np.add(x,y)


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar.

    Args:
        x (np.ndarray): matrix.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied matrix.
    """
    return np.multiply(x,a)


def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix or vector.

    Returns:
        np.ndarray: dot product.
    """
    return np.dot(x, y)


def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`. 

    Args:
        dim (int): matrix dimension.

    Returns:
        np.ndarray: identity matrix.
    """
    return np.identity(dim)


def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: inverse matrix.
    """
    return np.linalg.inv(x)


def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: transosed matrix.
    """
    return np.matrix_transpose(x)

def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product.

    Args:
        x (np.ndarray): 1th matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: hadamard produc
    """
    return np.multiply(x,y)


def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple[int]: indexes of basis columns.
    """
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    rank = np.sum(s > 1e-10)
    return tuple(range(rank))


def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 'fro', 2 or inf.

    Returns:
        float: vector norm
    """
    if order == 'fro':
        return np.linalg.norm(x, 'fro')
    elif order == 2:
        return np.linalg.norm(x, 2)
    elif order == np.inf:
        return np.linalg.norm(x, np.inf)
    else:
        raise ValueError("Order must be one of 'fro', 2, or np.inf")
