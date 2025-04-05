import numpy as np
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D

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
    kappa = 50.0  # example concentration parameter
    R_sample = randlangevin(mode, kappa)
    print("Random rotation sample from the Langevin distribution:")
    print(R_sample)

    import matplotlib.pyplot as plt

    # Create a 3D plot to visualize the camera frames.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define origins and colors for each axis
    origin = [0, 0, 0]
    colors = ['r', 'g', 'b']

    # Plot the original (mode) coordinate frame with solid lines
    for i, color in enumerate(colors):
        ax.quiver(origin[0], origin[1], origin[2],
                  mode[0, i], mode[1, i], mode[2, i],
                  color=color, arrow_length_ratio=0.1, label=f"Original axis {i+1}")

    # Plot the noisy (R_sample) coordinate frame with dashed lines
    for i, color in enumerate(colors):
        ax.quiver(origin[0], origin[1], origin[2],
                  R_sample[0, i], R_sample[1, i], R_sample[2, i],
                  color=color, linestyle='dashed', arrow_length_ratio=0.1, label=f"Noisy axis {i+1}")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Original (solid) vs Noisy (dashed) Rotation Frames")
    ax.legend()
    plt.show()