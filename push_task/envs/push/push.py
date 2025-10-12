from pickletools import read_decimalnl_short
import typing
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Space
import numpy as np
import sys
from numpy.typing import NDArray
import mujoco
import open3d as o3d
from scipy.spatial.transform import Rotation
import os
from envs.stretch import *
import time 
import mujoco.viewer
from envs import __file__ as base_path
from envs import stretch

class StretchPush(BaseStretchEnv):

    def __init__(
        self,
        max_episode_length=200,
        frame_skip=20,
        render_mode="rgb_array",
        camera_name="render_cam",
        seed=None,
        timestep=0.005,
        remove_model=False,
        depth_rendering=False, # include depth rendering
        student_obs=False,
    ):
        """
        Initializes a Mujoco environment with a single fixed object on a table.

        Action Space:
            Num | Action           | Min | Max | Conversion
            0   | Base Velocity    | -1  | 1   | Scaled by 0.3
            --- 1   | Base Angular Vel | ^   | ^   | ^ ---
            --- 2   | Lift pos         | ^   | ^   | Scaled by 0.01 and added to current position ---
            3   | Arm ext          | ^   | ^   | ^
            4   | Wrist yaw        | ^   | ^   | ^
            5   | Gripper          | ^   | ^   | ^

        Observation Space: Dict
            Key           | Obs
            jnt_states    | joint_lift, joint_wrist_yaw, joint_gripper_finger_left_open, sum of arm l0-l4
            goal_pos      | delta from gripper centre to target position
            gripper_table | delta from gripper centre to table centre
        """
        
        
        self.joints = JNT_NAMES.copy()[1:2]
        joint_state_shape = len(self.joints)+1 #  joint_yaw, joint_grip, wrist extension
        self.observation_space = Dict(
            {
                "jnt_states": Box(low=-np.inf, high=np.inf, shape=(joint_state_shape,)),
                "red_box": Box(low=-np.inf, high=np.inf, shape=(2,)),
                "green_box": Box(low=-np.inf, high=np.inf, shape=(2,)),
                "target_0": Box(low=-np.inf, high=np.inf, shape=(2,)),
                "target_1": Box(low=-np.inf, high=np.inf, shape=(2,)),
                "red_target": Box(low=-np.inf, high=np.inf, shape=(2,)),
            }
        )

        self.student_obs = student_obs
        self.student_info = {}
        if self.student_obs:
            pass

        #init base env
        location = os.path.dirname(os.path.realpath(base_path))
        fname = os.path.join(location, f"scenes/stretch_push.xml")
        action_mask = np.ones(10, dtype=np.int32)
        action_mask[1] = action_mask[2] = action_mask[5] = action_mask[6] = action_mask[7] = action_mask[8] = action_mask[9] = 0
        self.depth_rendering = depth_rendering
        super().__init__(
            fname, # put params here if wanted
            self.observation_space,
            frame_skip,
            camera_name,
            render_mode,
            max_episode_length,
            seed=seed,
            timestep=timestep,
            action_dim=5,
            action_mask=action_mask,
        )
        
        # if remove_model:
        #     os.remove(self.model_path)
    
    def reset_model(self, data = None):
        self.num_steps = 0
        self.red_target = np.random.randint(0, 2)
        self.box_fallen = False
        self.red_on_target = False
        self.reached_red = False
        self.reached_green = False
        super().reset_model([0.64, 0.1, 0., 0., -np.pi / 2, -0.65, 0, 0, 0, 0.19, 0, 0, 0, 0, 1])
        self.model.geom("target_0").rgba[:] = [int(not self.red_target), self.red_target, 0, 1]
        self.model.geom("target_1").rgba[:] = [self.red_target, int(not self.red_target), 0, 1]
        on_target = np.random.randint(0,2)

        if on_target:
            self.data.joint("green_box").qpos[1] = self.data.joint("red_box").qpos[1] = 0.95
            if self.red_target == 0:
                self.data.joint("red_box").qpos[0] *= -1
                self.data.joint("green_box").qpos[0] *= -1
                
            
        return self._get_obs()

    def _get_joint_states(self):
        joint_states = []
        for i in self.joints:
            joint_states.append(self.data.joint(i).qpos[:2])
        joint_states.append(
            np.sum(
                [self.data.joint(f"joint_arm_l{i}").qpos[:2] for i in range(4)],
                keepdims=True,
            )[0]
        )
        return np.float32(np.concatenate(joint_states))

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
        return {"red on target":self.red_on_target}

    def _get_obs(self):

        qu = self.data.body("link_gripper_finger_left").xquat.tolist()
        self._gripper_frame = np.array([[0.,-1.,0.,],[0.,0.,-1.],[1.,0.,0.]]) @ Rotation.from_quat(qu[1:] + qu[:1]).as_matrix() 
        self.gripper_pos = np.float32(
            self.data.body("link_grasp_center").xpos
        ) 
        # red_target = np.float32(self.data.geom(f"target_{self.red_target}").xpos.copy())
        # green_target = np.float32(self.data.geom(f"target_{int(not self.red_target)}").xpos.copy())
        # rb = np.float32(self.data.body("red_box").xpos.copy())
        # gb = np.float32(self.data.body("green_box").xpos.copy())
        # print(f"{red_target=}")
        # print(f"{green_target=}")
        # print(f"{rb=}")
        # print(f"{gb=}")
        # print(f"{self.gripper_pos=}")
        target_0 = np.float32(self.data.geom(f"target_0").xpos.copy()) - self.gripper_pos
        target_1 = np.float32(self.data.geom(f"target_1").xpos.copy()) - self.gripper_pos
        self.red_target_pos = np.float32(self.data.geom(f"target_{self.red_target}").xpos.copy()) - self.gripper_pos
        self.green_target_pos = np.float32(self.data.geom(f"target_{int(not self.red_target)}").xpos.copy()) - self.gripper_pos
        self.red_box = np.float32(self.data.body("red_box").xpos.copy()) - self.gripper_pos
        self.green_box = np.float32(self.data.body("green_box").xpos.copy()) - self.gripper_pos
        one_hot = np.array([1, 0], dtype=np.float32)
        obs = {
            "jnt_states": self._get_joint_states(),
            "target_0" : target_0[:2],
            "target_1" : target_1[:2],
            "red_box" : self.red_box[:2],
            "green_box" : self.green_box[:2],
            "red_target": one_hot if self.red_target == 0 else np.logical_not(one_hot).astype(np.float32),
            # "green_target": one_hot if self.red_target != 0 else np.logical_not(one_hot).astype(np.float32)
        }

        if self.student_obs:
            pass

        return obs

    def _check_both_gripper_col(self, bodies, geoms, target, left, right):
        left_col, right_col = False, False
        for body in bodies:
            for geom in geoms:
                if body == target and geom in left:
                    left_col = True
                if body == target and geom in right:
                    right_col = True
        return left_col, right_col

    def _get_success_ended(self):
        
        self.both_correct =  (self.red_dist < 0.06) and (self.green_dist < 0.06)

        return self.both_correct , self.box_fallen
   
    def _get_reward(self, obs, act_info):
        """
        Rew = Actuator penalty + Table collision penalty + Item move penalty - d2goal + reach goal rew + 
        (gripper open penalty if reached goal) + grasp reward + (target height penalty if grasped)
         + lift reward
        """
        self.red_dist  = np.linalg.norm((self.red_box - self.red_target_pos)[:2] )
        self.green_dist = np.linalg.norm((self.green_box - self.green_target_pos)[:2])
        
        # Move red to target position
        self.box_fallen = np.abs(self.red_box[-1]) > 0.2 or np.abs(self.green_box[-1]) > 0.2
        if self.box_fallen: return -150 # box fell off
        rew = 0

        rew += ((-0.2 if not self.red_on_target else 0) * np.linalg.norm(self.red_box) - max(self.red_dist - 0.06, 0))/10 #- self.red_dist
        red_on_target = self.red_dist < 0.06
        # if self.red_on_target and not red_on_target: return -100 # removed red box from target

        rew += 50*(not self.red_on_target)*(red_on_target)
        self.red_on_target |= red_on_target
        green_on_target = self.green_dist < 0.06
        if green_on_target and red_on_target: return 100
        if self.red_on_target:
            rew += (-0.2*np.linalg.norm(self.green_box) - 1*self.green_dist)/10
        return rew
    
    def passive_vis(self, model = None) -> None:
        import torch
        m = self.model
        d = self.data
        self._robot_id = self.model.body("base_link").id
        pcs = []
        # (d.geom("gripper_left_1"))
        with mujoco.viewer.launch_passive(m, d) as viewer:
            obs, _ = self.reset()
            start = time.time()
            cnt = 0
            while viewer.is_running() and time.time() - start < 300:
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                
                # print(self._init_pos)
                obs = self._get_obs()
                print(self._get_reward(obs, None))
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

