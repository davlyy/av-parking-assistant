from ifaces.data_iface import *
import numpy as np
import rsplan as rp

def hybrid_A_star(gridmap, start: Node, end: Node):
    """
    Implementation of hybrid_A_star based on the work of Zhang, P., Zhou, S., Hu, J., Zhao, W., Zheng, J., Zhang, Z., & Gao, C. (2025). Automatic parking trajectory planning in narrow spaces based on Hybrid A* and NMPC. Scientific Reports, 15(1), 1384.
    :param gridmap: A gridmap of all obstacles.
    :param start: The starting position.
    :param end: The target position.
    :return:
    """

    closed = {}
    final = None
    remaining = {start.idx: start}
    turn_radius = WHEELBASE/np.tan(MAX_STEERING_ANGLE)
    runway_length = 0.0
    step_size = 0.2
    resolution = 0.2
    while remaining:
        current_idx, current = min(remaining.items(), key=lambda item: item[1].f_cost)
        closed[current_idx] = current
        del remaining[current_idx]
        rs_curve = rp.path(start_pose=(current.x, current.y, current.theta),
                           end_pose=(end.x, end.y, end.theta),
                           turn_radius=turn_radius,
                           runway_length=runway_length,
                           step_size=step_size)

        if is_collision_free(gridmap=gridmap, path=rs_curve, resolution=resolution): #early stop if rs_curve is collision free
            final = reconstruct_path(current) + [rs_curve]
            break
        new = node_expansion(current)
        for node in new:
            if not is_collision_free(gridmap=gridmap, path=node, resolution=resolution):
                continue
            if not is_inside_grid(gridmap=gridmap, node=node):
                continue
            if node.idx in closed:
                continue
            elif node.idx in remaining:
                old = remaining[node.idx]
                if node.g_cost < old.g_cost:
                    remaining[node.idx] = node
            else:
                remaining[node.idx] = node
    return final

def reconstruct_path(node):
    path = []

    while node is not None:
        path.append(node)
        node = node.parent

    path.reverse()
    return path


def is_collision_free(gridmap, path, resolution) -> bool:
    ...

def is_inside_grid(gridmap, node) -> bool:
    ...

def node_expansion(current) -> list[Node]:
    ...