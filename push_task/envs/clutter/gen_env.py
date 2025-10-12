from envs.grocery_details import DIMS
from envs import __file__ as base_path
import numpy as np
from cv2 import pointPolygonTest
import os.path

FRONTAL_GRASP = 1
PARTIAL_OCCLUSION = 2
FULL_OCCLUSION = 3
SINGLE_OBJ = 0

class _BboxGeom:
    """
    Class takes in points that form a convex polygon and has a method for querying
    whether an object placement won't collide with the bounding box
    """

    def __init__(self, arrangement_type, *args):
        points = []
        if arrangement_type == FRONTAL_GRASP:
            top_left = args[0]
            bottom_right = args[1]
            points = [
                top_left,
                [top_left[0], bottom_right[1]],
                bottom_right,
                [bottom_right[0], top_left[1]],
            ]
        elif arrangement_type == PARTIAL_OCCLUSION:
            pos = args[0][-1]

            for i in [-0.15, 0.15]:
                for j in [-0.5, 0.1]:
                    points.append([pos[0] + i, pos[1] + j])
            points = points[:2] + points[2:][::-1]
        elif arrangement_type == FULL_OCCLUSION:
            pos = args[0][-1]

            for i in [-0.2, 0.2]:
                for j in [-0.5, 0.1]:
                    points.append([pos[0] + i, pos[1] + j])
            points = points[:2] + points[2:][::-1]
        self.polygons = [np.array(points, dtype=np.float32)]

    def check_collision(self, point, object):
        for obj in self.polygons:
            if (pointPolygonTest(obj, np.array(point)[:2], True)) > -np.sqrt(
                DIMS[object][0] ** 2 + DIMS[object][1] ** 2
            ) / 2:
                return True
        self.polygons.append(
            np.array(
                [
                    [
                        point[0] + DIMS[object][0] * i,
                        point[1] + j * DIMS[object][1],
                    ]
                    for i in [-1, 1]
                    for j in [-1, 1]
                ],
                dtype=np.float32,
            )
        )
        return False


def gen_pos(table_pos, table_size, arrangement_type, num_new_items, positions=None):
    bbox = None  # dimensions of main bbox
    target = None  # which item is the target
    final_pos = None  # positions of the main items
    items = list(DIMS.keys())
    np.random.shuffle(items)
    if arrangement_type == FRONTAL_GRASP:
        bbox_centre = table_pos + np.random.uniform(
            -table_size / 2 + 0.2, table_size / 2 - 0.2
        )
        xpos_middle = bbox_centre[0]
        xdist_right = np.random.uniform(
            0.005, min(0.05, xpos_middle - table_pos[0] + table_size[0] / 2)
        )
        xdist_left = np.random.uniform(
            0.005, max(0.05, xpos_middle - table_pos[0] - table_size[0] / 2)
        )
        xpositions = (
            xpos_middle
            - xdist_left
            - DIMS[items[1]][0] / 2
            - DIMS[items[0]][0] / 2,
            xpos_middle,
            xpos_middle
            + DIMS[items[1]][0] / 2
            + xdist_right
            + DIMS[items[2]][0] / 2,
        )
        final_pos = tuple(
            map(
                lambda x: np.array([x, np.random.normal(bbox_centre[1], 0.01)]),
                xpositions,
            )
        )
        y_max = max(
            map(
                lambda x: x[1][1] + DIMS[items[x[0]]][1] / 2,
                enumerate(final_pos),
            )
        )

        top_left = [xpositions[0] - DIMS[items[0]][0] / 2, y_max]
        bottom_right = [
            xpositions[2] + DIMS[items[2]][0] / 2,
            table_pos[1] - table_size[1] / 2,
        ]

        target = np.random.randint(0, 3)
        bbox = _BboxGeom(FRONTAL_GRASP, top_left, bottom_right)
    elif arrangement_type == PARTIAL_OCCLUSION:
        bbox_place = table_pos + np.random.uniform(
            - table_size/4, table_size / 4
        )
        bbox_place[1] = table_pos[1] - np.random.uniform(0.05, table_size[1]/4)
        xdist = -DIMS[items[0]][0] / 2 + np.random.uniform(
                 -0.1, -.05
            ) # left
        if bbox_place[0] > table_pos[0]: # right

            xdist = DIMS[items[0]][0] / 2 +  np.random.uniform(
                0.05, 0.1
            )
        ydist = np.random.uniform(0.05, 0.1)
        final_pos = (
            bbox_place,
            (
                bbox_place[0] + xdist,
                bbox_place[1]
                + ydist
                + DIMS[items[1]][1] / 2
                + DIMS[items[0]][1] / 2,
            ),
        )
        ## do bbox stuff
        bbox = _BboxGeom(PARTIAL_OCCLUSION, final_pos)
        target = 1
    elif arrangement_type == FULL_OCCLUSION:
        bbox_place = table_pos + np.random.uniform(
            -table_size / 2 + 0.1, table_size / 2 - 0.1
        )
        xpos = bbox_place[0]
        xdist = np.random.uniform(0, 0.03)
        ydist = np.random.uniform(0.03, 0.1)
        xpos_back = np.random.uniform(
            xpos, xpos + xdist + DIMS[items[1]][0] / 2
        )
        final_pos = (
            bbox_place,
            (
                bbox_place[0]
                + xdist
                + DIMS[items[0]][0] / 2
                + DIMS[items[1]][0] / 2,
                bbox_place[1],
            ),
            (
                xpos_back,
                bbox_place[1]
                + max(DIMS[items[0]][1], DIMS[items[1]][1]) / 2
                + DIMS[items[2]][1] / 2,
            ),
        )
        bbox = _BboxGeom(FULL_OCCLUSION, final_pos)
        target = 2
    elif arrangement_type == SINGLE_OBJ:
        if positions is None:
            bbox_place = table_pos + np.random.uniform(
                -table_size / 2 + 0.15, table_size / 2 - 0.15
            )
        else:
            assert len(positions) >= 2
            bbox_place = positions[:2]
        final_pos = (bbox_place[:2],)
        target = 0
        bbox = _BboxGeom(FRONTAL_GRASP, bbox_place[:2] - np.array([0.1, -0.1]), bbox_place[:2] + np.array([0.1, -0.1]))
        
    rand_poss = []
    for i in range(2 * num_new_items):
        if len(rand_poss) == num_new_items:
            break
        rand_item = items[(3 + i) % len(items)]
        rand_pos = table_pos + np.random.uniform(
            (-table_size + DIMS[rand_item]) / 2 + 5e-2,
            (table_size - DIMS[rand_item]) / 2 - 5e-2,
        )
        cnt = 0
        while bbox.check_collision(rand_pos, rand_item):
            if cnt > 10:
                break  # give up
            rand_pos = table_pos + np.random.uniform(
                (-table_size + DIMS[rand_item]) / 2 + 5e-2,
                (table_size - DIMS[rand_item]) / 2 - 5e-2,
            )
            cnt += 1
        if cnt > 10:
            continue  # didn't find a good position
        rand_pos[2] = table_size[2] + DIMS[rand_item][2] / 2
        rand_poss.append(rand_pos)
    l_f = list(final_pos) + rand_poss
    if positions is not None:
        l_f[:len(positions)] = positions[:min(len(l_f), len(positions))]
    return (
        l_f,
        list(items)[: len(final_pos) + len(rand_poss)],
        target,
    )


def gen_table_xml(pos, dim, num_shelves=1):
    """
    Generates an mjcf xml description of a table.

    Args:
    pos: specifies the position of the table in global coordinates
    dim: specifies the length, width and height of the table
    num_shelves: number of shelves to add. 
    """
    # generate frame
    length, width, height = map(lambda x: x / 2, dim)
    THICKNESS = 0.01  # wall thickness
    desc = f"""

    <body name="Table" pos="{pos[0]} {pos[1]} {height}">
        <body name="left_leg_front" pos="{-length} {-width} 0">
            <geom type="box" size="{THICKNESS : .3f} {THICKNESS : .3f} {height : .3f}" material="wall"/>
        </body>
        <body name="right_leg_front" pos="{length} {-width} 0">
          <geom type="box" size="{THICKNESS : .3f} {THICKNESS : .3f} {height : .3f}" material="wall"/>
        </body>
        <body name="right_leg_back" pos="{length} {width} 0">
          <geom type="box" size="{THICKNESS : .3f} {THICKNESS : .3f} {height : .3f}" material="wall"/>
        </body>
        <body name="left_leg_back" pos="{-length} {width} 0">
          <geom type="box" size="{THICKNESS : .3f} {THICKNESS : .3f} {height : .3f}" material="wall"/>
        </body>
        <body name="shelf_0" pos="{0} {0} {height}">
          <geom type="box" size="{length + THICKNESS : .3f} {width + THICKNESS : .3f} {THICKNESS : .3f}" material="shelf" friction="2 0.005 0.0001"/>
        </body>
    """
    for i in range(num_shelves - 1):
        desc+= f"""
            <body name="shelf_{i+1}" pos="{0} {0} {height - (i+1)*height/(num_shelves)}">
          <geom type="box" size="{length + THICKNESS : .3f} {width + THICKNESS : .3f} {THICKNESS : .3f}" material="shelf" friction="2 0.005 0.0001"/>
        </body>
        """
    desc += f"""</body>"""
    return desc


def gen_arrangement(
    table_pos=[0, 0.6, 0],
    table_pos_low: float = -0.1,
    table_pos_high: float = 0.1,
    table_size_low=[0.35, 0.3, 0.3],
    table_size_high=[0.6, 0.5, 0.5],
    num_shelves=1,
    add_wall=False,
    add_table=False, 
    add_objects=False,
    add_mocap=False
) -> str:
    """
    Generates xml for an arrangement and places all the grocery items out of view.

    Arguments:
    arrangement_type: FRONTAL_GRASP = 0, PARTIAL_OCCLUSION = 1, FULL_OCCLUSION = 2
    """
    # seed = int(np.random.rand() * (2**32 - 1))

    # np.random.seed(seed)
    table_pos = np.array(table_pos)
    table_pos[0] = np.random.uniform(table_pos_low, table_pos_high)
    table_size = np.random.uniform(table_size_low, table_size_high)

    location = os.path.dirname(os.path.realpath(base_path))
    base_scene = os.path.join(location, "scenes", "base_scene.xml")
    with open(base_scene, "r") as f:
        xml = "\n".join(f.readlines())
    if add_mocap:
        xml += """
         <body name="target" pos="10 10 10" quat="1 0 0 0" mocap="true">
            <geom name="target" type="sphere" size="0.01" class="visual"/>
        </body>
        """
    if add_table:
        xml += gen_table_xml( table_pos, table_size)
    if add_wall:
        xml += f"""<geom name="wall" size="2 0.1 2" pos="0 1 0" type="box"/>"""
        for i in range(np.random.randint(0, 5)):
            xml += f"""<geom name="wall{i+1}" size="{np.random.uniform(0.2, 0.6)} 0.1 {np.random.uniform(0.2, 0.6)}" pos="{np.random.uniform(-0.15, 0.15)} {np.random.uniform(1.2,1.5)} {np.random.uniform(0.4, 0.7)}" type="box"/>"""
    if add_objects:
        items = list(DIMS.keys())
        cnt = 0
        for cnt, item in enumerate(items):
            item_name = item.split("_")[0]
            xml += f"""<body name="{item}" pos="{100 + cnt} {0} {0}" quat="1 1 0 0">
                        <freejoint name="{item}"/>
                            <geom material="{item_name}" mesh="{item_name}" class="visual" shellinertia="true" mass="{0.5}"/>
                            <geom name="{item}" mesh="{item_name}" class="collision"/>
                        </body>"""
    xml += f"""
    </worldbody>
    </mujoco>
    """
    a = "_".join(map(lambda x: str(int(round(x, 2) * 1000)), list(table_size)))
    fname = os.path.join(location, f"scenes/scene_{a}_{np.random.randint(2**32)}.xml")
    with open(fname, "w") as f:
        f.write(xml)
    return fname, {"table_pos": table_pos, "table_size": table_size}