import typing
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Space
import numpy as np
from numpy.typing import NDArray
from envs.clutter.gen_env import gen_pos, gen_arrangement
import mujoco
import open3d as o3d
from envs.grocery_details import DIMS
from scipy.spatial.transform import Rotation
import os
from envs.stretch import *
import time 
import mujoco.viewer


class StretchMultiObjectEnv(BaseStretchEnv):

    def __init__(
        self,
        max_episode_length=200,
        frame_skip=20,
        render_mode="rgb_array",
        camera_name="render_cam",
        seed=None,
        timestep=0.005,
        remove_model=True,
        difficulty=3, # top difficulty of the scene
        reached_threshold= 0.06,
        action_weight= 0.1, # penalize base movements
        d2goal_weight= 2, # penalize being far from the gar
        grasp_weight= 0.5, # close gripper after reached
        target_height_weight= 5, # when grasped, give reward for lifting object
        reached_goal_rew= 25, # reward when I have reached goal vicinity
        grasped_rew= 50, # reward when tips of gripper touch object
        lifted_rew= 100, # reward when the object is lifted (ep done)
        collision_penalty=10, # penalty for colliding with table
        item_move_weight=4, # penalty when object move
        target_height=0.05, # lift height threshold for ending episode
        end_ep_penalty=10, # penalty for episode terminating (goal toppled)
        depth_rendering=False # include depth rendering
    ):
        """
        Initializes a Mujoco environment with a single fixed object on a table.

        Action Space:
            Num | Action           | Min | Max | Conversion
            0   | Base Velocity    | -1  | 1   | Scaled by 0.3
            1   | Base Angular Vel | ^   | ^   | ^
            2   | Lift pos         | ^   | ^   | Scaled by 0.01 and added to current position
            3   | Arm ext          | ^   | ^   | ^
            4   | Wrist yaw        | ^   | ^   | ^
            5   | Gripper          | ^   | ^   | ^

        Observation Space: Dict
            Key           | Obs
            jnt_states    | joint_lift, joint_wrist_yaw, joint_gripper_finger_left_open, sum of arm l0-l4
            goal_pos      | delta from gripper centre to target position
            gripper_table | delta from gripper centre to table centre
        """
        self.max_objects = len(DIMS) * 8
        joint_state_shape = 5 # joint_lift, joint_yaw, joint_grip, base_link orientation, wrist extension
        self.observation_space = Dict(
            {   
                "gripper_obstacles": Box(low=-np.inf, high=np.inf, shape=(self.max_objects + 4,3)), #+4 for table
                "jnt_states": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        joint_state_shape,
                    ),  
                ),
                "goal_pos": Box(low=-np.inf, high=np.inf, shape=(3,)),
            }
        )
        self.depth_rendering = depth_rendering
        if depth_rendering:
            self.observation_space["pc"] = Box(low=-np.inf, high=np.inf, shape=(1200, 3)) # shape=(1, 58, 102)
            self.observation_space["depth_image"] = Box(low=-np.inf, high=np.inf, shape=(1, 58, 102))
        #init base env
        super().__init__(
            lambda : gen_arrangement(table_pos= [0, 0.7, 0], table_pos_low=-0.01, table_pos_high=0.01,table_size_low=[0.58,0.28,0.58], table_size_high=[0.62,0.32,0.63], add_table=True, add_objects=True),
            self.observation_space,
            frame_skip,
            camera_name,
            render_mode,
            max_episode_length,
            seed=seed,
            timestep=timestep,
        )
        self.difficulty = difficulty
        # set reward params
        self.action_weight = action_weight
        self.collision_penalty = collision_penalty
        self.reached_threshold = reached_threshold
        self.d2goal_weight = d2goal_weight
        self.grasp_weight = grasp_weight
        self.target_height_weight = target_height_weight
        self.reached_goal_rew = reached_goal_rew
        self.grasped_rew = grasped_rew
        self.lifted_rew = lifted_rew
        self.item_move_weight = item_move_weight
        self.target_height_threshold = target_height
        self.end_ep_penalty = end_ep_penalty
        self._table_poss = []
        a = np.zeros(3)
        a[2] = self.table_size[2] # only care about top surface
        for x in (-0.5, 0.5):
            a[0] = x*self.table_size[0]
            for y in (-0.5, 0.5):
                a[1] = y*self.table_size[1]
                self._table_poss.append(self.table_pos + a)
        self.table_pos[-1] = 0 # reset z to 0
        if remove_model:
            os.remove(self.model_path)

    def reset_model(self, positions=[0,0.6], stretch_pos = None):
        self.num_steps = 0
        scene_type = self.difficulty - 1#np.random.randint(0, self.difficulty)
        num_obj = 0 #np.random.randint(0, 4)
        rand_poss, items, target = gen_pos(
            self.table_pos, self.table_size, scene_type, num_obj, positions=[np.array([-0.05881808,  0.64825957,  0.65248145]), np.array([-0.15811677,  0.80584912,  0.65207418]), np.array([0.20544913, 0.69268848, 0.6520776 ])]
        )
        for pos, item in zip(rand_poss, items):
            self.data.joint(item).qpos[:2] = pos[:2]
            self.data.joint(item).qpos[2:] = np.array(
                [
                    self.table_size[2] + DIMS[item][2] / 2 + 1e-2,
                    1,
                    1,
                    0,
                    0,
                ]
            )
        
        self.target = items[target]
        self.target_id = self.data.body(self.target).id
        self._all_items = items[:]
        self.target_init_pos = self.data.joint(self.target).qpos[:3].copy()
        super().reset_model(stretch_pos)

        
        
        
        self._init_pos = []
        for i in self._all_items:
            self._init_pos.append(self.data.joint(i).qpos[:3].copy())

        self.total_dist = 0
        #reset reward terms
        self._grasp = False
        self._reached_goal = False
        self._grasped = False
        # self.depth_renderer.update_scene(self.data, camera="d435i_camera_depth")
        return self._get_obs()

    def _initialize_simulation(self):

        super()._initialize_simulation()
        self._table_id = self.model.body("Table").id
        self._robot_id = self.model.body("base_link").id
        self._left_grip_geom_ids = [
            self.model.geom("gripper_left_0").id,
            self.model.geom("gripper_left_1").id,
            self.model.geom("rubber_tip_left").id,
        ]
        self._right_grip_geom_ids = [
            self.model.geom("gripper_right_0").id,
            self.model.geom("gripper_right_1").id,
            self.model.geom("rubber_tip_right").id,
        ]
        if self.depth_rendering:
            self.img_height = 58
            self.img_width = 102
            self.depth_renderer = mujoco.Renderer(self.model, height=self.img_height, width=self.img_width)
            self.depth_renderer.enable_depth_rendering()
            
            self.f = (
                0.5
                * self.img_height
                / np.tan(self.model.cam("d435i_camera_depth").fovy * np.pi / 360)
            )
            
            self.cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.img_width,
                self.img_height,
                self.f,
                self.f,
                self.img_width // 2,
                self.img_height // 2,
            )
            self.crop_box = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array([-1, -0.4, -0.4]), max_bound=np.array([1, 0.4, 0.2])
            )
            self._cam_final_rot = np.array([[1,0,0],[0,0,-1],[0,1,0]])
            self._cam_static_rot = np.array(
                [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
            )  # mujoco camera orientation
        return self.model, self.data

    def close(self):
        if self.depth_rendering:
            self.depth_renderer.close()
        super().close()

    def _get_info(self):
        return {"reached":self._reached_goal, "grasped":self._grasped, "dist":self.total_dist}
    
    def _get_obs(self):
        base_link = self.data.joint("base_link").qpos
        w, x, y, z = base_link[3:]

        qu = self.data.body("link_gripper_finger_left").xquat.tolist()
        self._gripper_frame = np.array([[0.,-1.,0.,],[0.,0.,-1.],[1.,0.,0.]]) @ Rotation.from_quat(qu[1:] + qu[:1]).as_matrix() 
        target_quat = self.data.body(self.target).xquat.tolist()
        self._target_rot = Rotation.from_quat(target_quat[1:] + target_quat[:1]).as_euler("xzy")
        self.gripper_pos = (
            self.data.body("rubber_tip_left").xpos
            + self.data.body("rubber_tip_right").xpos
        ) / 2
        self.target_pos = self.data.body(self.target).xpos
        target_noise = np.random.uniform(-0.01, 0.01, (3,))
        self.table_pos_gripper_frame = self._gripper_frame @ (self.table_pos - self.gripper_pos)

        joint_states = []
        for i in JNT_NAMES:
            joint_states.append(self.data.joint(i).qpos[:2])
        joint_states.append(
            np.sum(
                [self.data.joint(f"joint_arm_l{i}").qpos[:2] for i in range(4)],
                keepdims=True,
            )[0]
        )       
        # yaw_z = np.arctan2(
        #     2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        # )  # only care about yaw (z-rot)
        yaw_z = 0
        joint_states.append(np.array([yaw_z], dtype=np.float32))
        object_deltas = []
        for i in self._all_items:
            a = np.zeros(3)
            size = DIMS[i]
            for x in (-0.5,0.5):
                a[0] = x*size[0]
                for y in (-0.5,0.5):
                    a[1] = y*size[1]
                    for z in (-0.5,0.5):
                        a[2] = z*size[2]
                        object_deltas.append(self.data.body(i).xpos + a)
            # object_deltas.append(self.data.body(i).xpos)

        d = np.concatenate((np.resize(object_deltas, (self.max_objects, 3)), self._table_poss)) #resize to maximum number of points
        obs = {
            "gripper_obstacles": np.float32((d - self.gripper_pos)),
            "jnt_states": np.float32(np.concatenate(joint_states)),
            "goal_pos": np.float32((self.target_init_pos + target_noise) - self.gripper_pos), # maybe subtract object width from depth to align with real
        }
        if self.depth_rendering:

            self.depth_renderer.update_scene(self.data, camera="d435i_camera_depth")
            depth_image = self.depth_renderer.render()# [::-1, ::-1]
            depth_img = o3d.geometry.Image(depth_image)

            k = np.asarray(depth_image)
            obs["depth_image"] = depth_image[np.newaxis, ...]

            min_bound = np.array([obs["goal_pos"][0] - 0.6, obs["goal_pos"][1] - 0.4, obs["goal_pos"][2] - 0.4])
            max_bound = np.array([obs["goal_pos"][0] + 0.6, obs["goal_pos"][1] + 0.4, obs["goal_pos"][2] + 0.2])
            self.crop_box = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound, max_bound=max_bound
            )
            pc = o3d.geometry.PointCloud.create_from_depth_image(
                depth_img, self.cam_intrinsic, depth_scale=1.0
            )

            t = self.data.cam("d435i_camera_depth").xpos
            q = self.data.cam("d435i_camera_depth").xmat
            r = q.reshape(3, 3)
            h_transform = np.eye(4)
            h_transform[:3, :3] = self._cam_final_rot@r@ self._cam_static_rot
            h_transform[:3, 3] = t[:3] - self.gripper_pos
            pc = pc.transform(h_transform)
            pc = pc.crop(self.crop_box)

            pc_points = np.asarray(pc.points).astype(np.float32)
            if len(pc_points) == 0:
                points_noise = np.zeros((1200, 3)).astype(np.float32)
            else:
                points_noise = pc_points[np.random.randint(len(pc_points), size=1200), :] + np.random.uniform(-0.015, 0.015, size=(1200,3))
            # o3d.io.write_point_cloud("point_cloud_goalcrop.ply", pc)
            # 4/0
            
            obs["pc"] = points_noise.astype(np.float32)
        return obs
    
    def _item_dist_func(self, dist):
        if dist < 1e-2:
            return 0.0
        return ITEM_COL_MAX / (1 + np.exp(MID_VEL - VEL_WEIGHT * dist))

    def _item_move_penalty(self):
        """
        Move reward negatively penalizes the xyz velocity of the object
        according to _item_dist_func
        """

        rew = 0
        prev_pos_l = []
        
        for init_pos, i in zip(self._init_pos, self._all_items):
            d = np.linalg.norm(self.data.joint(i).qpos[:3] - init_pos)
            self.total_dist += d
            rew += self._item_dist_func(
                d / (self.model.opt.timestep * self.frame_skip)
            )  # penalty for hitting items
            prev_pos_l.append(self.data.joint(i).qpos[:3].copy())
        self._init_pos = prev_pos_l
        return rew

    def _check_both_gripper_col(self, bodies, geoms, target, left, right):
        for body in bodies:
            for geom in geoms:
                if body == target and geom in left:
                    self._left_col = True
                    return
                if body == target and geom in right:
                    self._right_col = True
                    return
        return
     
    def _contact_checking(self):
        self._grasp = self._left_col = self._right_col = False
        self._table_col = False
        for i, j in zip(self.data.contact.geom1, self.data.contact.geom2):
            body1 = self.model.body_rootid[self.model.geom_bodyid[i]]
            body2 = self.model.body_rootid[self.model.geom_bodyid[j]]
            if not self._table_col and (
                (body1 == self._table_id and body2 == self._robot_id)
                or (body2 == self._table_id and body1 == self._robot_id)
            ):
                self._table_col = True

            if not (self._left_col and self._right_col):
                self._check_both_gripper_col(
                    [body1, body2],
                    [i, j],
                    self.target_id,
                    self._left_grip_geom_ids,
                    self._right_grip_geom_ids,
                )

    def _get_success_ended(self):
        return self._lifted, self._end_ep
    
    def _get_reward(self, obs, act_info):
        """
        Rew = Actuator penalty + Table collision penalty + Item move penalty - d2goal + reach goal rew + 
        (gripper open penalty if reached goal) + grasp reward + (target height penalty if grasped)
         + lift reward
        """
        rew = 0

        # Penalties
        rew -= self.item_move_weight * self._item_move_penalty()
        
        act_norm = act_info["act_norm"]
        rew -= self.action_weight * act_norm  # actuator penalties

        d2goal = np.linalg.norm(obs["goal_pos"]) # roughly between 0 to 0.5
        rew -= self.d2goal_weight * d2goal # d2goal penalty if not reached

        grip_actuator = self.data.ctrl[5] # between -0.005 and 0.04
        rew -= self.grasp_weight * (grip_actuator + self.low[5]) * (self._reached_goal)
        
        target_height = max(self.target_pos[2] - self.target_init_pos[2], 0) # between 0 and 0.1
        rew += self.target_height_weight * target_height * self._grasped
        # do contact checking
        self._contact_checking()
        # rew -= self.collision_penalty * self._table_col

        target_moved_from_init_pos = np.linalg.norm(self.target_pos - self.target_init_pos) > 0.6
        target_toppled = self.target_pos[2] - self.target_init_pos[2] < -0.005# TODO:check this
        self._end_ep = (target_moved_from_init_pos or target_toppled or self._table_col)
        rew -= self.end_ep_penalty * self._end_ep

        # Rewards
        self._grasp = (self._left_col and self._right_col)
        reached = d2goal < self.reached_threshold
        grasped = self._grasp
        self._lifted = self.target_pos[2] - self.target_init_pos[2] >= self.target_height_threshold
        final_rew = rew + self.reached_goal_rew*reached*(not self._reached_goal)
        final_rew += self.grasped_rew * grasped * (not self._grasped)
        final_rew += self.lifted_rew*self._lifted

        self._reached_goal |= reached
        self._grasped |= grasped
        return final_rew
    
    def _get_sparse_reward_and_done(self, obs, act_info):

        """
        Rew = Actuator penalty + Table collision penalty + Reached goal rew + Grasped reward + Lifted reward
        """

        rew = 0

        # Penalties
        # rew += self._item_move_penalty()
        
        act_norm = act_info["act_norm"]
        rew += (
            - self.action_weight * act_norm
        )  # actuator penalties

        # do contact checking
        self._contact_checking()
        rew -= self.collision_penalty * self._table_col
        target_moved_from_init_pos = np.linalg.norm(self.target_pos - self.target_init_pos) > 0.6
        target_toppled = self.target_pos[2] - self.target_init_pos[2] < -0.005# TODO:check this
        self._end_ep = (target_moved_from_init_pos or target_toppled)
        rew -= self.end_ep_penalty * self._end_ep 
        # Rewards
        d2goal = np.linalg.norm(obs["goal_pos"])
        self._grasp = self._grasp or (self._left_col and self._right_col)

        reached = d2goal < 0.08
        grasped = self._grasp
        self._lifted = self.target_pos[2] - self.target_init_pos[2] > 0.1
        final_rew = rew + 10*reached*(not self._reached_goal) + 100 * grasped * (not self._grasped) + 1000*self._lifted
        self._reached_goal |= reached
        self._grasped |= grasped
        return final_rew
    
    def passive_vis(self, model = None) -> None:
        import torch
        m = self.model
        d = self.data
        self._table_id = self.model.body("Table").id
        self._robot_id = self.model.body("base_link").id
        pcs = []
        print(d.geom("gripper_left_1"))
        with mujoco.viewer.launch_passive(m, d) as viewer:
            obs, _ = self.reset()
            start = time.time()
            cnt = 0
            while viewer.is_running() and time.time() - start < 300:
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                
                # print(self._init_pos)
                if cnt == 0:
                    obs = self._get_obs()
                    for i in obs:
                        obs[i] = torch.tensor(obs[i][np.newaxis, ...], device="cuda")
                    print(obs["goal_pos"])
                    print(obs["jnt_states"])
                    with torch.no_grad():
                        act = model(obs)[0]
                    print(act)
                    a, action_info = self.from_action_space(act.cpu().numpy())
                    d.ctrl = a
                cnt = (cnt + 1) % 20
                mujoco.mj_step(m,d)
                
                # input()
                # if time.time() - a > 5:
                #     # self.reset_scene()
                #     a = time.time()
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)