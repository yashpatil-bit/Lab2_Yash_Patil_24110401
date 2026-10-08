import numpy as np


# ============================================================
# STANDARD DENAVIT-HARTENBERG TRANSFORMATION
# ============================================================

def dh_matrix(a, d, alpha, theta):
    """
    Standard D-H homogeneous transformation matrix.

    Convention:

        A_i = Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha)

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
    T : 4x4 numpy array
    """

    ct = np.cos(theta)
    st = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,    sa,       ca,       d],
        [0.0,   0.0,      0.0,     1.0]
    ])

    return T


# ============================================================
# FIXED TOOL TRANSFORMATION
# ============================================================

def tool_transform():
    """
    Fixed transformation from the flange to the hand.

    The supplied MuJoCo model has:
        translation = (0, 0, 0.107) m
        rotation    = Rz(-45 degrees)

    This is fixed and does NOT add another DOF.
    """

    theta = -np.pi / 4.0

    c = np.cos(theta)
    s = np.sin(theta)

    T = np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.107],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return T


# ============================================================
# FRANKA 7-DOF FORWARD KINEMATICS
# ============================================================

def franka_fk(q):
    """
    Forward kinematics of the 7-DOF Franka Panda.

    Parameters
    ----------
    q : array-like
        [q1, q2, q3, q4, q5, q6, q7]
        Joint angles in radians.

    Returns
    -------
    T_hand : 4x4 numpy array
        End-effector pose of the hand.
    """

    q = np.asarray(q, dtype=float)

    if q.shape != (7,):
        raise ValueError(
            "q must contain exactly 7 joint angles."
        )

    q1, q2, q3, q4, q5, q6, q7 = q


    # ========================================================
    # D-H PARAMETERS
    # ========================================================

    # Joint 1
    a1 = 0.0
    d1 = 0.333
    alpha1 = 0.0
    theta1 = q1

    # Joint 2
    a2 = 0.0
    d2 = 0.0
    alpha2 = -np.pi / 2
    theta2 = q2

    # Joint 3
    a3 = 0.0
    d3 = 0.316
    alpha3 = np.pi / 2
    theta3 = q3

    # Joint 4
    a4 = 0.0825
    d4 = 0.0
    alpha4 = np.pi / 2
    theta4 = q4

    # Joint 5
    a5 = -0.0825
    d5 = 0.384
    alpha5 = -np.pi / 2
    theta5 = q5

    # Joint 6
    a6 = 0.0
    d6 = 0.0
    alpha6 = np.pi / 2
    theta6 = q6

    # Joint 7
    a7 = 0.088
    d7 = 0.0
    alpha7 = np.pi / 2
    theta7 = q7


    # ========================================================
    # INDIVIDUAL D-H MATRICES
    # ========================================================

    A1 = dh_matrix(
        a1, d1, alpha1, theta1
    )

    A2 = dh_matrix(
        a2, d2, alpha2, theta2
    )

    A3 = dh_matrix(
        a3, d3, alpha3, theta3
    )

    A4 = dh_matrix(
        a4, d4, alpha4, theta4
    )

    A5 = dh_matrix(
        a5, d5, alpha5, theta5
    )

    A6 = dh_matrix(
        a6, d6, alpha6, theta6
    )

    A7 = dh_matrix(
        a7, d7, alpha7, theta7
    )


    # ========================================================
    # FORWARD KINEMATICS
    # ========================================================

    T07 = (
        A1
        @ A2
        @ A3
        @ A4
        @ A5
        @ A6
        @ A7
    )


    # ========================================================
    # ADD FIXED HAND TRANSFORMATION
    # ========================================================

    T_hand = T07 @ tool_transform()


    return T_hand


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Test configuration
    # --------------------------------------------------------

    q_deg = np.array([
        20.0,
        -20.0,
        30.0,
        -70.0,
        20.0,
        40.0,
        30.0
    ])

    # Convert degrees to radians
    q = np.deg2rad(q_deg)


    # --------------------------------------------------------
    # Calculate FK
    # --------------------------------------------------------

    T_hand = franka_fk(q)


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("==========================================")
    print("FRANKA 7-DOF D-H FORWARD KINEMATICS")
    print("==========================================")

    print("\nJoint angles [degrees]:")
    print(q_deg)

    print("\nJoint angles [radians]:")
    print(q)

    print("\nEnd-effector transformation matrix:")
    print(T_hand)

    print("\nEnd-effector position [m]:")
    print(T_hand[:3, 3])

    print("\nEnd-effector rotation matrix:")
    print(T_hand[:3, :3])