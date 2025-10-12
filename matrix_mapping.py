import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    return np.negative(x)


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    return np.flip(x)


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    alpha_rad = np.deg2rad(alpha_deg)
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)

    rotation_matrix = np.array([
        [cos_alpha, -sin_alpha],
        [sin_alpha, cos_alpha]
    ])
    scale_matrix = np.array([
        [scale[0], 0],
        [0, scale[1]]
    ])
    shear_matrix = np.array([
        [1, shear[0]],
        [shear[1], 1]
    ])

    transform_matrix = shear_matrix @ rotation_matrix @ scale_matrix
    transformed_points = np.dot(x, transform_matrix.T) + np.array(translate)
    
    return transformed_points
