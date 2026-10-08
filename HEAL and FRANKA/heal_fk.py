import numpy as np


# ============================================================
# STANDARD DENAVIT-HARTENBERG TRANSFORMATION
# ============================================================

def dh_matrix(a, d, alpha, theta):
    """
    Standard Denavit-Hartenberg homogeneous
    transformation matrix.

    Parameters
    ----------
    a : float
        Link length [m]

    d : float
        Link offset [m]

    alpha : float
        Link twist [rad]

    theta : float
        Joint angle [rad]

    Returns
    -------
    T : (4,4) numpy array
        Homogeneous transformation matrix
    """

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    ct = np.cos(theta)
    st = np.sin(theta)

    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,       ca,      d],
        [0.0,    0.0,      0.0,    1.0]
    ])

    return T


# ============================================================
# HEAL 6-DOF FORWARD KINEMATICS
# ============================================================

def heal_fk(q):
    """
    Forward kinematics of the HEAL 6-DOF manipulator
    using standard Denavit-Hartenberg parameters.

    Input
    -----
    q : array-like, length 6
        Joint angles [rad]

    Output
    ------
    T06 : (4,4) numpy array
        Transformation from DH frame 0
        to the HEAL end-effector frame.
    """

    q = np.asarray(q, dtype=float)

    if q.shape != (6,):
        raise ValueError(
            "q must contain exactly 6 joint angles."
        )

    q1, q2, q3, q4, q5, q6 = q

    # ========================================================
    # D-H PARAMETERS
    # ========================================================

    # Joint 1
    a1 = 0.0
    d1 = 0.149799698
    alpha1 = np.deg2rad(89.9998025)
    theta1 = q1

    # Joint 2
    a2 = 0.299999761
    d2 = 0.087500000
    alpha2 = np.deg2rad(180.0)
    theta2 = q2 + np.deg2rad(90.0)

    # Joint 3
    a3 = 0.0
    d3 = 0.087373076
    alpha3 = np.deg2rad(89.9543739)
    theta3 = q3 + np.deg2rad(0.0456261)

    # Joint 4
    a4 = 0.0
    d4 = 0.377355799
    alpha4 = np.deg2rad(29.1927532)
    theta4 = q4 + np.deg2rad(180.0)

    # Joint 5
    a5 = 0.0
    d5 = 0.000100516
    alpha5 = np.deg2rad(89.9998025)
    theta5 = q5 - np.deg2rad(90.0456261)

    # Joint 6
    a6 = 0.0
    d6 = -0.122699902
    alpha6 = np.deg2rad(180.0)
    theta6 = q6 + np.deg2rad(180.0)

    # ========================================================
    # INDIVIDUAL D-H TRANSFORMATIONS
    # ========================================================

    A1 = dh_matrix(a1, d1, alpha1, theta1)

    A2 = dh_matrix(a2, d2, alpha2, theta2)

    A3 = dh_matrix(a3, d3, alpha3, theta3)

    A4 = dh_matrix(a4, d4, alpha4, theta4)

    A5 = dh_matrix(a5, d5, alpha5, theta5)

    A6 = dh_matrix(a6, d6, alpha6, theta6)

    # ========================================================
    # FORWARD KINEMATICS
    # ========================================================

    T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

    return T06


# ============================================================
# WORLD -> D-H FRAME 0
# ============================================================

def world_to_dh0():
    """
    Fixed transformation from the MuJoCo world frame
    to our D-H frame 0.

    The first joint is located at z = 0.171 m.
    Our D-H x0/y0 axes are rotated by 180 degrees
    about world Z.
    """

    T = np.eye(4)

    T[:3, :3] = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    T[:3, 3] = [0.0, 0.0, 0.171]

    return T


# ============================================================
# WORLD -> END EFFECTOR
# ============================================================

def heal_fk_world(q):
    """
    Calculate the HEAL end-effector pose
    expressed in the MuJoCo world frame.
    """

    T_world_0 = world_to_dh0()

    T_0_6 = heal_fk(q)

    T_world_6 = T_world_0 @ T_0_6

    return T_world_6


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Test joint configuration
    q_deg = np.array([
        30.0,
        -20.0,
        40.0,
        15.0,
        25.0,
        -30.0
    ])

    # Convert degrees -> radians
    q = np.deg2rad(q_deg)

    # Calculate FK
    T06 = heal_fk(q)

    # Calculate world-frame pose
    T_world_ee = heal_fk_world(q)

    print()
    print("==========================================")
    print("HEAL 6-DOF D-H FORWARD KINEMATICS")
    print("==========================================")

    print("\nJoint angles [degrees]:")
    print(q_deg)

    print("\nJoint angles [radians]:")
    print(q)

    print("\nT06:")
    print(T06)

    print("\nEnd-effector position in world frame [m]:")
    print(T_world_ee[:3, 3])

    print("\nEnd-effector rotation in world frame:")
    print(T_world_ee[:3, :3])