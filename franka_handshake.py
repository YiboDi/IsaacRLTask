# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
from omniisaacgymenvs.robots.articulations.views.frankaL_view import FrankaLView
from omniisaacgymenvs.robots.articulations.views.frankaR_view import FrankaRView

# from omniisaacgymenvs.robots.articulations.views.cabinet_view import CabinetView

from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

from omni.isaac.cloner import Cloner

import numpy as np
import torch
import math

from pxr import Usd, UsdGeom


class FrankaHandshakeTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]  # episode:from start state to terminal

        self.action_scale = self._task_cfg["env"][
            "actionScale"
        ]  # action scale:magnitude of an action(e.g. angle or torque)
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]  # introducing randomness
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        # self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]  # velocity of the degrees of freedom (DoFs)
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        # self._num_observations = 45 #obs_buf
        self._num_observations = 39  # obs_buf
        self._num_actions = 18  # 2*9

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_frankaL()
        self.get_frankaR()

        # self.add_cube()

        super().set_up_scene(scene)

        self._frankasL = FrankaLView(prim_paths_expr="/World/envs/.*/frankaL", name="frankaL_view")
        self._frankasR = FrankaRView(prim_paths_expr="/World/envs/.*/frankaR", name="frankaR_view")

        # self._cube = RigidPrimView(prim_paths_expr="/World/envs/.*/Cube/cube", name="cube_view", reset_xform_properties=False)

        scene.add(self._frankasL)
        scene.add(self._frankasL._handsL)
        scene.add(self._frankasL._lfingersL)
        scene.add(self._frankasL._rfingersL)

        scene.add(self._frankasR)
        scene.add(self._frankasR._handsR)
        scene.add(self._frankasR._lfingersR)
        scene.add(self._frankasR._rfingersR)

        # scene.add(self._cube)
        self.init_data()
        return

    def get_frankaL(self):
        frankaL = Franka(
            prim_path=self.default_zero_env_path + "/frankaL",
            name="frankaL",
            translation=torch.tensor([0.8, -0.5, 0.0]),
            orientation=torch.tensor([0, 0, 0, 1]),
        )
        self._sim_config.apply_articulation_settings(
            "frankaL", get_prim_at_path(frankaL.prim_path), self._sim_config.parse_actor_config("franka")
        )

    # def get_frankaR(self):
    #     frankaR = Franka(prim_path=self.default_zero_env_path + "/frankaR", name="frankaR", translation=torch.tensor([0.8, 0.5, 0.0]))
    #     self._sim_config.apply_articulation_settings("frankaR", get_prim_at_path(frankaR.prim_path), self._sim_config.parse_actor_config("franka"))
    def get_frankaR(self):
        frankaR = Franka(
            prim_path=self.default_zero_env_path + "/frankaR",
            name="frankaR",
            translation=torch.tensor([-0.2, -0.5, 0.0]),
            orientation=torch.tensor([1, 1, 1, 1]),
        )
        self._sim_config.apply_articulation_settings(
            "frankaR", get_prim_at_path(frankaR.prim_path), self._sim_config.parse_actor_config("franka")
        )

    # def add_cube(self):
    #     cube = DynamicCuboid(
    #         prim_path=self.default_zero_env_path + "/Cube/cube", name="cube_0", translation=[0.4, 0.4, 0], mass=1, scale=[0.07, 0.07, 0.07], color=torch.tensor([0.9, 0.6, 0.2])
    #     )

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        ### hand and finger pose in env-local coordinates
        stage = get_current_stage()
        handL_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/frankaL/panda_link7")),
            self._device,
        )
        lfingerL_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/frankaL/panda_leftfinger")),
            self._device,
        )
        rfingerL_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/frankaL/panda_rightfinger")),
            self._device,
        )

        fingerL_pose = torch.zeros(7, device=self._device)
        fingerL_pose[0:3] = (lfingerL_pose[0:3] + rfingerL_pose[0:3]) / 2.0
        fingerL_pose[3:7] = lfingerL_pose[3:7]
        handL_pose_inv_rot, handL_pose_inv_pos = tf_inverse(handL_pose[3:7], handL_pose[0:3])

        handR_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/frankaR/panda_link7")),
            self._device,
        )
        lfingerR_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/frankaR/panda_leftfinger")),
            self._device,
        )
        rfingerR_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/frankaR/panda_rightfinger")),
            self._device,
        )

        fingerR_pose = torch.zeros(7, device=self._device)
        fingerR_pose[0:3] = (lfingerR_pose[0:3] + rfingerR_pose[0:3]) / 2.0
        fingerR_pose[3:7] = lfingerR_pose[3:7]
        handR_pose_inv_rot, handR_pose_inv_pos = tf_inverse(handR_pose[3:7], handR_pose[0:3])
        ###

        # grasp_pose_axis = 1
        frankaL_local_grasp_pose_rot, frankaL_local_pose_pos = tf_combine(
            handL_pose_inv_rot, handL_pose_inv_pos, fingerL_pose[3:7], fingerL_pose[0:3]
        )  ###
        frankaR_local_grasp_pose_rot, frankaR_local_pose_pos = tf_combine(
            handR_pose_inv_rot, handR_pose_inv_pos, fingerR_pose[3:7], fingerR_pose[0:3]
        )

        # frankaL_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.frankaL_local_grasp_pos = frankaL_local_pose_pos.repeat((self._num_envs, 1))
        self.frankaL_local_grasp_rot = frankaL_local_grasp_pose_rot.repeat((self._num_envs, 1))

        # frankaR_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.frankaR_local_grasp_pos = frankaR_local_pose_pos.repeat((self._num_envs, 1))
        self.frankaR_local_grasp_rot = frankaR_local_grasp_pose_rot.repeat((self._num_envs, 1))

        ##cube, the grasp pos may need to be change:
        # cube_local_grasp_pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device)  ##why?reason:coordinate of cube in the cube system

        # self.cube_local_grasp_pos = cube_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        # self.cube_local_grasp_rot = cube_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        # self.cube_inward_axis = torch.tensor([0, 0, -1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1)) #
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        # self.gripper_right_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat(
        #     (self._num_envs, 1)
        # )
        # self.cube_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:
        handL_pos, handL_rot = self._frankasL._handsL.get_world_poses(clone=False)
        handR_pos, handR_rot = self._frankasR._handsR.get_world_poses(clone=False)

        # cube_pos, cube_rot = self._cube.get_world_poses(clone=False)

        frankaL_dof_pos = self._frankasL.get_joint_positions(clone=False)
        frankaL_dof_vel = self._frankasL.get_joint_velocities(clone=False)

        frankaR_dof_pos = self._frankasR.get_joint_positions(clone=False)
        frankaR_dof_vel = self._frankasR.get_joint_velocities(clone=False)

        ##may not needed

        # self.cube_pos, self.cube_rot = self._cube.get_world_poses(clone=False)
        # self.cube_vel = self._cube.get_linear_velocities()

        self.frankaL_dof_pos = frankaL_dof_pos
        self.frankaR_dof_pos = frankaR_dof_pos

        # self.cube_grasp_pos = torch.tensor(cube_pos)

        (
            self.frankaL_grasp_rot,
            self.frankaL_grasp_pos,
            self.frankaR_grasp_rot,
            self.frankaR_grasp_pos,
        ) = self.compute_grasp_transforms(
            handL_rot,
            handL_pos,
            handR_rot,
            handR_pos,
            self.frankaL_local_grasp_rot,
            self.frankaL_local_grasp_pos,
            self.frankaR_local_grasp_rot,
            self.frankaR_local_grasp_pos,
        )

        self.frankaL_lfinger_pos, self.frankaL_lfinger_rot = self._frankasL._lfingersL.get_world_poses(clone=False)
        self.frankaL_rfinger_pos, self.frankaL_rfinger_rot = self._frankasL._lfingersL.get_world_poses(clone=False)

        self.frankaR_lfinger_pos, self.frankaR_lfinger_rot = self._frankasR._lfingersR.get_world_poses(clone=False)
        self.frankaR_rfinger_pos, self.frankaR_rfinger_rot = self._frankasR._lfingersR.get_world_poses(clone=False)
        # Jul.5
        Ldof_pos_scaled = (
            2.0
            * (frankaL_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        Rdof_pos_scaled = (
            2.0
            * (frankaR_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )

        # Lto_target = self.cube_grasp_pos - self.frankaL_grasp_pos
        # Rto_target = self.cube_grasp_pos - self.frankaR_grasp_pos
        to_target = self.frankaR_grasp_pos - self.frankaL_grasp_pos

        self.obs_buf = torch.cat(
            (
                Ldof_pos_scaled,  # 9
                Rdof_pos_scaled,  # 9
                frankaL_dof_vel * self.dof_vel_scale,  # 9
                frankaR_dof_vel * self.dof_vel_scale,  # 9
                to_target,  # 3
                # Lto_target, #3
                # Rto_target, #3
                # cube_pos, #3
            ),
            dim=-1,
        )

        observations = {"Franka": {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        Ltargets = (
            self.frankaL_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions[:, :9] * self.action_scale
        )
        self.frankaL_dof_targets[:] = tensor_clamp(Ltargets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        Rtargets = (
            self.frankaR_dof_targets
            + self.franka_dof_speed_scales * self.dt * self.actions[:, 9:18] * self.action_scale
        )
        self.frankaR_dof_targets[:] = tensor_clamp(Rtargets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        env_ids_int32 = torch.arange(self._frankasL.count, dtype=torch.int32, device=self._device)

        self._frankasL.set_joint_position_targets(self.frankaL_dof_targets, indices=env_ids_int32)
        self._frankasR.set_joint_position_targets(self.frankaR_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)
        # indices = env_ids.type(torch.int32)
        num_indices = len(indices)

        # cube_pos = self.initial_cube_pos.clone()
        # cube_rot = self.initial_cube_rot.clone()
        # cube_velocities = self.initial_cube_velocities.clone()

        # reset franka
        Lpos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        Ldof_pos = torch.zeros((num_indices, self._frankasL.num_dof), device=self._device)
        Ldof_vel = torch.zeros((num_indices, self._frankasL.num_dof), device=self._device)
        Ldof_pos[:, :] = Lpos
        self.frankaL_dof_targets[env_ids, :] = Lpos
        self.frankaL_dof_pos[env_ids, :] = Lpos

        # reset frankaR
        Rpos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        Rdof_pos = torch.zeros((num_indices, self._frankasR.num_dof), device=self._device)
        Rdof_vel = torch.zeros((num_indices, self._frankasR.num_dof), device=self._device)
        Rdof_pos[:, :] = Rpos
        self.frankaR_dof_targets[env_ids, :] = Rpos
        self.frankaR_dof_pos[env_ids, :] = Rpos

        # or set local pose, necessary thing:
        # self._cube.set_world_poses(torch.zeros_like(self._cube.get_world_poses()[env_ids_64]), indices=env_ids_32)
        # self._cube.set_linear_velocities(torch.zeros_like(self._cube.get_linear_velocities()[env_ids_64]), indices=env_ids_32)

        # self._cube.set_world_poses(cube_pos[env_ids_64], cube_rot[env_ids_64], indices=env_ids_32)
        # self._cube.set_velocities(cube_velocities[env_ids_64], indices=env_ids_32)

        self._frankasL.set_joint_position_targets(self.frankaL_dof_targets[env_ids], indices=indices)
        self._frankasL.set_joint_positions(Ldof_pos, indices=indices)
        self._frankasL.set_joint_velocities(Ldof_vel, indices=indices)

        self._frankasR.set_joint_position_targets(self.frankaR_dof_targets[env_ids], indices=indices)
        self._frankasR.set_joint_positions(Rdof_pos, indices=indices)
        self._frankasR.set_joint_velocities(Rdof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_franka_dofs = self._frankasL.num_dof
        self.frankaL_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        self.frankaR_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankasL.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankasL.gripper_indices] = 0.1
        self.frankaL_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )
        self.frankaR_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )
        # self.initial_cube_pos, self.initial_cube_rot = self._cube.get_world_poses()
        # self.initial_cube_velocities = self._cube.get_velocities()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_franka_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.frankaL_grasp_pos,
            self.frankaR_grasp_pos,
            # self.cube_grasp_pos,
            self.frankaL_grasp_rot,
            self.frankaR_grasp_rot,
            # self.cube_grasp_rot,
            # self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.gripper_forward_axis,
            # self.cube_inward_axis, 
            self.gripper_up_axis, 
            # self.cube_up_axis,
            self._num_envs,
            self.dist_reward_scale,
            # self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            # self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self._max_episode_length, self.franka_dof_pos,
            # self.finger_close_reward_scale,
        )

    def is_done(self) -> None:
        # reset if cube is reached by both or max length reached

        # need to modify the following condition:
        # d1 = torch.norm(self.frankaL_grasp_pos - self.cube_grasp_pos, p=2, dim=-1)
        # d2 = torch.norm(self.frankaR_grasp_pos - self.cube_grasp_pos, p=2, dim=-1)

        d = torch.norm(self.frankaR_grasp_pos - self.frankaL_grasp_pos, p=2, dim=-1)

        # self.reset_buf = torch.where((d1 <= 0.01) & (d2 <= 0.01), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(d <= 0.02, torch.ones_like(self.reset_buf), self.reset_buf)

        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )

    def compute_grasp_transforms(
        self,
        handL_rot,
        handL_pos,
        handR_rot,
        handR_pos,
        frankaL_local_grasp_rot,
        frankaL_local_grasp_pos,
        frankaR_local_grasp_rot,
        frankaR_local_grasp_pos,
    ):
        global_frankaL_rot, global_frankaL_pos = tf_combine(
            handL_rot, handL_pos, frankaL_local_grasp_rot, frankaL_local_grasp_pos
        )
        global_frankaR_rot, global_frankaR_pos = tf_combine(
            handR_rot, handR_pos, frankaR_local_grasp_rot, frankaR_local_grasp_pos
        )
        # global_cube_rot, global_cube_pos = tf_combine(
        #     cube_rot, cube_pos, cube_local_grasp_rot, cube_local_grasp_pos
        # )

        return global_frankaL_rot, global_frankaL_pos, global_frankaR_rot, global_frankaR_pos

    def compute_franka_reward(
        self,
        reset_buf,
        progress_buf,
        actions,
        frankaL_grasp_pos,
        frankaR_grasp_pos,
        # cube_grasp_pos,
        frankaL_grasp_rot,
        frankaR_grasp_rot,
        # cube_grasp_rot,
        # franka_lfinger_pos, franka_rfinger_pos,
        gripper_forward_axis,
        # cube_inward_axis, 
        gripper_up_axis, 
        # cube_up_axis,
        num_envs,
        dist_reward_scale,  # rot_reward_scale, around_handle_reward_scale,
        # open_reward_scale,
        # finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length, joint_positions, finger_close_reward_scale
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor]

        # distance from hand to the cube
        # d = torch.norm(frankaL_grasp_pos - cube_grasp_pos, p=2, dim=-1)
        # # dist_reward = 1.0 / (1.0 + d ** 2)
        # dist_reward = 10 / (1.0 + d ** 2)
        # dist_reward *= dist_reward
        # dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward) #shape is numenvs*1

        # dR = torch.norm(frankaR_grasp_pos - cube_grasp_pos, p=2, dim=-1)
        # dist_rewardR = 1.0 / (1.0 + dR ** 2)
        # dist_rewardR *= dist_rewardR
        # dist_rewardR = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        d = torch.norm(frankaL_grasp_pos - frankaR_grasp_pos, p=2, dim=-1)
        # dist_reward = 1.0 / (1.0 + d ** 2)
        dist_reward = 1 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.03, dist_reward * 2, dist_reward)  # shape is numenvs*1

        # grasp_rewardL = torch.where(d <= 0.01, 200, 0)
        # grasp_rewardR = torch.where(dR <= 0.01, 100, 0)
        # grasp_reward = torch.where(d <= 0.01, 100, 0)

        # axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        # axis2 = tf_vector(cube_grasp_rot, cube_inward_axis)
        # axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        # axis4 = tf_vector(cube_grasp_rot, cube_up_axis)

        axis1 = tf_vector(frankaL_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(frankaR_grasp_rot, gripper_forward_axis)
        axis3 = tf_vector(frankaL_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(frankaR_grasp_rot, gripper_up_axis)

        # dot = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1) # alignment of forward axis for gripper
        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )
        # reward for matching the orientation of two grippes
        # rot_reward = 0.5 * (torch.sign(dot) * dot ** 2)
        # rot_reward = 0.5 * (-1 * dot ** 2)
        rot_reward = (-torch.sign(dot1) * dot1**2) - dot2**2

        # dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        # dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the cube (fingers wrapped)
        # rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

        # bonus if left finger is above the cube handle and right below
        # around_handle_reward = torch.zeros_like(rot_reward)
        # around_handle_reward = torch.where(franka_lfinger_pos[:, 2] > cube_grasp_pos[:, 2],
        #    torch.where(franka_rfinger_pos[:, 2] < cube_grasp_pos[:, 2],
        #                around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
        # reward for distance of each finger from the cube
        # finger_dist_reward = torch.zeros_like(rot_reward)
        # lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - cube_grasp_pos[:, 2])
        # rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - cube_grasp_pos[:, 2])
        # finger_dist_reward = torch.where(franka_lfinger_pos[:, 2] > cube_grasp_pos[:, 2],
        #                                  torch.where(franka_rfinger_pos[:, 2] < cube_grasp_pos[:, 2],
        #                                              (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)

        # finger_close_reward = torch.zeros_like(rot_reward)
        # finger_close_reward = torch.where(d <=0.03, (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]), finger_close_reward)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        # open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # cube_top_joint

        # rewards = dist_reward_scale * (dist_reward + dist_rewardR)+ grasp_rewardL + grasp_rewardR - action_penalty

        # rewards = dist_reward_scale * (dist_reward + dist_rewardR) + grasp_rewardL + grasp_rewardR - action_penalty

        rewards = dist_reward_scale * dist_reward - action_penalty * 0.01 + rot_reward
        # bonus for opening cube properly
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

        # # prevent bad style in opening cube
        # rewards = torch.where(franka_lfinger_pos[:, 0] < cube_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)
        # rewards = torch.where(franka_rfinger_pos[:, 0] < cube_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)

        return rewards
