"""
Example on how to implement a random function by reference to the interface.
"""
#1. import
from ifaces.algorithms_iface import PathPlanning
from ifaces.data_iface import Pose, YawAngle

#2. define function with matching parameters and return type
def plan_path(current_pose: Pose, final_pose: Pose) -> list[Pose]: #parameter names have to also match
    path = list()
    path.append(current_pose)
    # 4. implement function logic
    path.append(final_pose)
    return path

#3. Type-checking step: verifies that plan_path matches the PathPlanning interface class.
type_checking_variable: PathPlanning = plan_path

#example function call, could also be in main.py
cp = Pose((0, 0), YawAngle(0.0))
fp = Pose((300, 300), YawAngle(359.4))

plan_path(current_pose=cp, final_pose=fp)