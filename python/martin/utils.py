import numpy as np
from scipy.linalg import expm

def randlangevin(mode, kappa):
    """
    Sample from the Langevin distribution in SO(3) with a given mode and concentration parameter.
    
    Parameters:
    -----------
    mode : array_like, shape (3, 3)
        The mode rotation matrix.
    kappa : float
        The concentration parameter. If kappa <= 0, the function returns the identity matrix.
        
    Returns:
    --------
    Re : ndarray, shape (3, 3)
        A random rotation matrix sampled from the Langevin distribution.
    """
    if kappa <= 0:
        return np.eye(3)
    
    # 1) Sample theta from the von Mises distribution
    theta = np.random.vonmises(0, 2 * kappa)
    
    # 2) Sample an axis of rotation uniformly from the sphere.
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    
    # 3) Construct the skew-symmetric matrix corresponding to the rotation axis.
    axis_skew = np.array([[0,         -axis[2],  axis[1]],
                          [axis[2],    0,       -axis[0]],
                          [-axis[1],   axis[0],  0]])
    
    # Compute the rotation matrix P via the matrix exponential.
    P = expm(theta * axis_skew)
    
    # Multiply the mode by the perturbation P.
    Re = mode @ P
    return Re

if __name__ == "__main__":
    # Define a mode rotation matrix (for instance, the identity matrix)
    mode = np.eye(3)
    kappa = 5.0  # example concentration parameter
    R_sample = randlangevin(mode, kappa)
    print("Random rotation sample from the Langevin distribution:")
    print(R_sample)