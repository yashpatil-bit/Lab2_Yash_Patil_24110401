"""
Franka Panda (7-DOF) - MuJoCo forward-kinematics viewer with keyboard joint control.

Uses its OWN GLFW window (glfw is already installed with mujoco), so the
keyboard is read directly with glfw.get_key(). There are no MuJoCo viewer
hotkeys in this window, so no key can clash with anything. Works on
Windows / Linux / macOS with plain `python franka_mujoco.py`.

CONTROLS (hold the key to move, release to stop) - click the window first
    joint 1 :  Q (+)   A (-)
    joint 2 :  W (+)   S (-)
    joint 3 :  E (+)   D (-)
    joint 4 :  R (+)   F (-)
    joint 5 :  T (+)   G (-)
    joint 6 :  Y (+)   H (-)
    joint 7 :  U (+)   J (-)

    UP / DOWN arrow : joint speed + / -
    BACKSPACE       : reset to the initial pose
    P               : print the full D-H vs MuJoCo comparison
    ESC             : quit
    Mouse           : left-drag rotate, right-drag pan, scroll zoom
                      (hold SHIFT while dragging to change the drag mode)
"""

import os
import time

import numpy as np
import glfw
import mujoco

try:
    from franka_fk import franka_fk
    HAVE_DH = True
except Exception as e:
    HAVE_DH = False
    print(f"[warning] could not import franka_fk ({e}). "
          "Viewer will run without the D-H comparison.")


# ============================================================
# SETTINGS
# ============================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_REL = os.path.join("robot_descriptions", "franka", "panda.xml")

XML_PATH = os.path.join(_HERE, _REL)
if not os.path.exists(XML_PATH):
    XML_PATH = _REL                      # fall back to current directory

N_JOINTS = 7

JOINT_SPEED_DEG_PER_SEC = 30.0           # starting speed
SPEED_STEP_DEG_PER_SEC = 5.0             # UP / DOWN arrow change
SPEED_MIN, SPEED_MAX = 5.0, 180.0

PRINT_PERIOD = 0.25                      # terminal print rate while moving [s]

# (forward key, backward key) for joints 1..7
KEY_MAP = [
    (glfw.KEY_Q, glfw.KEY_A),
    (glfw.KEY_W, glfw.KEY_S),
    (glfw.KEY_E, glfw.KEY_D),
    (glfw.KEY_R, glfw.KEY_F),
    (glfw.KEY_T, glfw.KEY_G),
    (glfw.KEY_Y, glfw.KEY_H),
    (glfw.KEY_U, glfw.KEY_J),
]
KEY_NAMES = [("Q", "A"), ("W", "S"), ("E", "D"),
             ("R", "F"), ("T", "G"), ("Y", "H"), ("U", "J")]


# ============================================================
# INITIAL JOINT CONFIGURATION
# ============================================================

Q_INIT_DEG = np.array([
    20.0,     # q1
    -20.0,    # q2
    30.0,     # q3
    -70.0,    # q4
    20.0,     # q5
    40.0,     # q6
    30.0      # q7
])

q_init = np.deg2rad(Q_INIT_DEG)
q = q_init.copy()


# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

hand_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "hand"
)

# Joint limits straight from the XML (radians)
joint_min = np.array([model.jnt_range[i, 0] for i in range(N_JOINTS)])
joint_max = np.array([model.jnt_range[i, 1] for i in range(N_JOINTS)])


def set_joints(q_rad):
    """Write the seven arm joint values into MuJoCo and run forward kinematics."""
    for i in range(N_JOINTS):
        data.qpos[model.jnt_qposadr[i]] = q_rad[i]
    mujoco.mj_forward(model, data)


def compare_dh_and_mujoco(q_rad, verbose=True):
    """Return (dh_pos, mujoco_pos, error_norm_m); optionally print everything."""

    mujoco_position = data.xpos[hand_id].copy()
    mujoco_rotation = data.xmat[hand_id].reshape(3, 3).copy()

    if not HAVE_DH:
        if verbose:
            print("\nMuJoCo hand position [m]:", mujoco_position)
            print("MuJoCo hand rotation:\n", mujoco_rotation)
        return None, mujoco_position, None

    T_hand = franka_fk(q_rad)
    dh_position = T_hand[:3, 3]
    dh_rotation = T_hand[:3, :3]

    position_error = dh_position - mujoco_position
    err = np.linalg.norm(position_error)

    if verbose:
        print()
        print("==========================================")
        print("FRANKA D-H FORWARD KINEMATICS")
        print("==========================================")
        print("\nJoint angles [degrees]:")
        print(np.rad2deg(q_rad))
        print("\nJoint angles [radians]:")
        print(q_rad)
        print("\nEnd-effector transformation matrix:")
        print(T_hand)
        print("\nEnd-effector position from D-H [m]:")
        print(dh_position)
        print("\nEnd-effector rotation from D-H:")
        print(dh_rotation)

        print()
        print("==========================================")
        print("MUJOCO")
        print("==========================================")
        print("\n7-DOF arm joint values [degrees]:")
        print(np.rad2deg(data.qpos[:N_JOINTS]))
        print("\nMuJoCo hand position [m]:")
        print(mujoco_position)
        print("\nMuJoCo hand rotation:")
        print(mujoco_rotation)

        print()
        print("==========================================")
        print("D-H vs MUJOCO")
        print("==========================================")
        print("\nPosition error:")
        print(position_error)
        print("\nPosition error magnitude [m]:")
        print(err)
        print("\nPosition error magnitude [mm]:")
        print(err * 1000)

    return dh_position, mujoco_position, err


# ============================================================
# GLFW WINDOW + MUJOCO RENDERING OBJECTS
# ============================================================

if not glfw.init():
    raise RuntimeError("Could not initialise GLFW")

glfw.window_hint(glfw.SAMPLES, 4)
window = glfw.create_window(1280, 800, "Franka Panda - MuJoCo keyboard control",
                            None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("Could not create a GLFW window")

glfw.make_context_current(window)
glfw.swap_interval(1)                    # vsync -> ~60 fps

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)

cam.lookat[:] = [-0.15, 0.0, 1.15]
cam.distance = 1.6
cam.azimuth = 135
cam.elevation = -15

scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)


# ============================================================
# MOUSE (camera) CALLBACKS
# ============================================================

mouse = {"left": False, "right": False, "x": 0.0, "y": 0.0}


def mouse_button_cb(win, button, action, mods):
    mouse["left"] = glfw.get_mouse_button(
        win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    mouse["right"] = glfw.get_mouse_button(
        win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    mouse["x"], mouse["y"] = glfw.get_cursor_pos(win)


def cursor_pos_cb(win, x, y):
    dx = x - mouse["x"]
    dy = y - mouse["y"]
    mouse["x"], mouse["y"] = x, y

    if not (mouse["left"] or mouse["right"]):
        return

    _, height = glfw.get_window_size(win)
    shift = (glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
             glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

    if mouse["right"]:
        action = (mujoco.mjtMouse.mjMOUSE_MOVE_H if shift
                  else mujoco.mjtMouse.mjMOUSE_MOVE_V)
    else:
        action = (mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift
                  else mujoco.mjtMouse.mjMOUSE_ROTATE_V)

    mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll_cb(win, xoffset, yoffset):
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                          0.0, -0.05 * yoffset, scene, cam)


glfw.set_mouse_button_callback(window, mouse_button_cb)
glfw.set_cursor_pos_callback(window, cursor_pos_cb)
glfw.set_scroll_callback(window, scroll_cb)


# ============================================================
# ONE-SHOT KEYS (reset, speed, print, quit)
# ============================================================

speed_deg = JOINT_SPEED_DEG_PER_SEC
print_full_requested = False


def key_cb(win, key, scancode, action, mods):
    global speed_deg, print_full_requested, q

    if action != glfw.PRESS:
        return

    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(win, True)

    elif key == glfw.KEY_BACKSPACE:
        q = q_init.copy()
        print("\n[reset] back to initial pose")

    elif key == glfw.KEY_UP:
        speed_deg = min(SPEED_MAX, speed_deg + SPEED_STEP_DEG_PER_SEC)
        print(f"[speed] {speed_deg:.0f} deg/s")

    elif key == glfw.KEY_DOWN:
        speed_deg = max(SPEED_MIN, speed_deg - SPEED_STEP_DEG_PER_SEC)
        print(f"[speed] {speed_deg:.0f} deg/s")

    elif key == glfw.KEY_P:
        print_full_requested = True


glfw.set_key_callback(window, key_cb)


# ============================================================
# STARTUP MESSAGE
# ============================================================

set_joints(q)
compare_dh_and_mujoco(q, verbose=True)

print()
print("==========================================")
print("MUJOCO VIEWER - KEYBOARD CONTROL  (click the window first)")
print("==========================================")
print("\n  Joint   Forward (+)   Backward (-)")
for i, (kf, kb) in enumerate(KEY_NAMES):
    print(f"  q{i + 1}      {kf}             {kb}")
print("\n  UP/DOWN : speed   BACKSPACE : reset   P : print   ESC : quit\n")


# ============================================================
# MAIN LOOP
# ============================================================

viewport = mujoco.MjrRect(0, 0, 0, 0)
last_time = time.perf_counter()
last_print = 0.0

while not glfw.window_should_close(window):

    now = time.perf_counter()
    dt = min(now - last_time, 0.1)       # clamp after any stall
    last_time = now

    # ---- read held keys directly from GLFW --------------------
    moved = False
    step = np.deg2rad(speed_deg) * dt

    for i, (k_fwd, k_bwd) in enumerate(KEY_MAP):

        direction = 0.0
        if glfw.get_key(window, k_fwd) == glfw.PRESS:
            direction += 1.0
        if glfw.get_key(window, k_bwd) == glfw.PRESS:
            direction -= 1.0

        if direction != 0.0:
            q[i] = np.clip(q[i] + direction * step,
                           joint_min[i], joint_max[i])
            moved = True

    # ---- kinematics -------------------------------------------
    set_joints(q)

    # ---- terminal output --------------------------------------
    if print_full_requested:
        print_full_requested = False
        compare_dh_and_mujoco(q, verbose=True)

    if moved and (now - last_print) > PRINT_PERIOD:
        last_print = now
        _, mj_pos, err = compare_dh_and_mujoco(q, verbose=False)
        msg = (f"q [deg]: {np.round(np.rad2deg(q), 1)} | "
               f"Hand (MuJoCo) [m]: {np.round(mj_pos, 4)}")
        if err is not None:
            msg += f" | D-H vs MuJoCo error [mm]: {err * 1000:.3f}"
        print(msg)

    # ---- render -----------------------------------------------
    w, h = glfw.get_framebuffer_size(window)
    viewport.width, viewport.height = w, h

    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)

    left_text = "\n".join(
        [f"q{i + 1}  [{KEY_NAMES[i][0]}/{KEY_NAMES[i][1]}]"
         for i in range(N_JOINTS)] + ["Speed  [UP/DOWN]", "Hand position [m]"]
    )
    hp = data.xpos[hand_id]
    right_text = "\n".join(
        [f"{np.rad2deg(q[i]):8.2f} deg" for i in range(N_JOINTS)] +
        [f"{speed_deg:.0f} deg/s",
         f"{hp[0]:.3f} {hp[1]:.3f} {hp[2]:.3f}"]
    )
    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                       mujoco.mjtGridPos.mjGRID_TOPLEFT,
                       viewport, left_text, right_text, context)

    glfw.swap_buffers(window)
    glfw.poll_events()


# ============================================================
# EXIT
# ============================================================

glfw.terminate()

print()
print("==========================================")
print("FINAL POSE")
print("==========================================")
compare_dh_and_mujoco(q, verbose=True)