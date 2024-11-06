import numpy as np
from scipy.spatial.transform import Rotation as R

def skewd(x):
    # Create a 3x3 zero matrix
    skew = np.zeros((3, 3))
    
    # Fill in the skew-symmetric matrix
    skew[0, 1] = -x[2]
    skew[0, 2] = x[1]
    skew[1, 0] = x[2]
    skew[1, 2] = -x[0]
    skew[2, 0] = -x[1]
    skew[2, 1] = x[0]
    
    return skew

def so3_exp(omega):
    omega = omega.flatten()  # Ensure omega is a 1D array
    # Compute the square of the magnitude (theta^2)
    theta_sq = np.dot(omega, omega)
    
    # Variables to hold the real and imaginary factors
    if theta_sq < 1e-10:
        theta = 0
        theta_quad = theta_sq * theta_sq
        imag_factor = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_quad
        real_factor = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_quad
    else:
        theta = np.sqrt(theta_sq)
        half_theta = 0.5 * theta
        imag_factor = np.sin(half_theta) / theta
        real_factor = np.cos(half_theta)

    # Return the quaternion (real, imag_factor * omega.x, imag_factor * omega.y, imag_factor * omega.z)
    return R.from_quat([imag_factor * omega[0], imag_factor * omega[1], imag_factor * omega[2], real_factor])

# Define the se3_exp function (equivalent to the C++ version)
def se3_exp(a):
    omega = a[:3].flatten()  # Angular velocity (first 3 components)
    theta = np.linalg.norm(omega)
    
    # Get SO(3) part (rotation)
    so3 = so3_exp(omega)
    
    # Compute the skew-symmetric matrix of omega
    Omega = skewd(omega)
    Omega_sq = np.dot(Omega, Omega)
    
    if theta < 1e-10:
        V = so3.as_matrix()  # Identity rotation matrix for small theta
    else:
        theta_sq = theta ** 2
        V = np.eye(3) + (1.0 - np.cos(theta)) / theta_sq * Omega + (theta - np.sin(theta)) / (theta_sq * theta) * Omega_sq

    # Translation vector (last 3 components)
    translation = V @ a[3:]
    # Construct the transformation (Isometry)
    se3 = np.eye(4)
    se3[:3, :3] = so3.as_matrix()  # Rotation matrix part
    se3[:3, 3] = translation.flatten()  # Translation vector part
    
    return se3