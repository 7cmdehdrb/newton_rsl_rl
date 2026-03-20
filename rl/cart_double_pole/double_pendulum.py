import argparse
import torch
import warp as wp
import newton
import newton.examples
from newton import JointTargetMode

from rl_template import PhysicsBackend, GenericRslRlEnv, make_train_cfg
from rsl_rl.runners import OnPolicyRunner
from typing import Any
from dataclasses import dataclass


@dataclass
class CartDoublePoleCfg:
    num_envs: int = 4096
    dt: float = 0.02
    sim_substeps: int = 10
    max_steps: int = 1000

    # 🔥 개선: 에이전트가 순간적으로 폭발적인 힘을 내어 보상할 수 있도록 한계치 대폭 상향
    action_scale: float = 100.0  # (기존 10.0)
    max_cart_vel: float = 100.0  # (기존 20.0)
    max_cart_acc: float = 5000.0  # (기존 50.0) 거의 무제한에 가깝게 허용

    # 트랙 물리적 한계
    track_limit: float = 10.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CartDoublePoleBackend(PhysicsBackend):
    def __init__(self, cfg: CartDoublePoleCfg, viewer=None):
        self.cfg = cfg
        self._device = torch.device(cfg.device)

        if self._device.type == "cuda":
            print(f"Using CUDA device: {torch.cuda.get_device_name(self._device)}")
        else:
            raise RuntimeError(
                "CUDA device not available. Please check your PyTorch installation and GPU setup."
            )

        self._num_envs = cfg.num_envs
        self.viewer = viewer
        self.sim_time = 0.0

        self._action_dim = 1
        # 🔥 개선: 관측 차원 증가 (자세한 상태, 팁 위치, 트랙 여유분 등 추가)
        self._obs_dim = 11

        self.step_count = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )
        self._reward_buf = torch.zeros(self._num_envs, device=self._device)
        self._terminated = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        self._truncated = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )

        # ------------------------------------------------------------------
        # Newton 모델 빌드
        # ------------------------------------------------------------------
        base = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(base)

        cart_hx, cart_hy, cart_hz = 0.2, 0.1, 0.1
        self.pole_hz = 0.5  # 길이 계산을 위해 저장
        pole_hx, pole_hy, pole_hz = 0.05, 0.05, self.pole_hz

        link_cart = base.add_link()
        base.add_shape_box(link_cart, hx=cart_hx, hy=cart_hy, hz=cart_hz)

        link_pole1 = base.add_link()
        base.add_shape_box(link_pole1, hx=pole_hx, hy=pole_hy, hz=pole_hz)

        link_pole2 = base.add_link()
        base.add_shape_box(link_pole2, hx=pole_hx, hy=pole_hy, hz=pole_hz)

        j_cart = base.add_joint_prismatic(
            parent=-1,
            child=link_cart,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            limit_lower=-10.5,
            limit_upper=10.5,
        )

        j_pole1 = base.add_joint_revolute(
            parent=link_cart,
            child=link_pole1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, cart_hz), q=wp.quat_identity()
            ),
            child_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, -pole_hz), q=wp.quat_identity()
            ),
        )

        j_pole2 = base.add_joint_revolute(
            parent=link_pole1,
            child=link_pole2,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, pole_hz), q=wp.quat_identity()
            ),
            child_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, -pole_hz), q=wp.quat_identity()
            ),
        )

        base.add_articulation([j_cart, j_pole1, j_pole2], label="cart_double_pole")

        base.joint_target_mode[j_cart] = int(JointTargetMode.POSITION)
        base.joint_target_ke[j_cart] = 10000.0
        base.joint_target_kd[j_cart] = 500.0

        base.joint_target_mode[j_pole1] = 0
        base.joint_target_ke[j_pole1] = 0.0
        base.joint_target_kd[j_pole1] = 0.0

        base.joint_target_mode[j_pole2] = 0
        base.joint_target_ke[j_pole2] = 0.0
        base.joint_target_kd[j_pole2] = 0.0

        builder = newton.ModelBuilder()
        builder.replicate(base, self._num_envs, spacing=(25.0, 4.0, 0.0))
        builder.add_ground_plane()

        self.model = builder.finalize()

        if self.viewer is not None:
            self.viewer.set_model(self.model)

        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.articulation_view = newton.selection.ArticulationView(
            self.model, "*", exclude_joint_types=[newton.JointType.FREE]
        )

        self.ctrl_target = torch.zeros((self._num_envs, 1, 3), device=self._device)
        self.ctrl_wp = self.articulation_view.get_attribute(
            "joint_target_pos", self.control
        )

        self.step_count = torch.randint(
            0,
            self.cfg.max_steps,
            (self._num_envs,),
            dtype=torch.long,
            device=self._device,
        )

        self._has_been_upright = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )

        self.cmd_pos = torch.zeros(self._num_envs, device=self._device)
        self.cmd_vel = torch.zeros(self._num_envs, device=self._device)

        env_ids = torch.arange(self._num_envs, device=self._device)
        self.reset(env_ids)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def apply_actions(self, actions: torch.Tensor) -> None:
        actions = torch.clamp(actions, -1.0, 1.0)
        dt = self.cfg.dt

        target_vel = actions.squeeze(-1) * self.cfg.action_scale

        max_dv = self.cfg.max_cart_acc * dt
        dv = torch.clamp(target_vel - self.cmd_vel, -max_dv, max_dv)
        self.cmd_vel = self.cmd_vel + dv

        self.cmd_vel = torch.clamp(
            self.cmd_vel, -self.cfg.max_cart_vel, self.cfg.max_cart_vel
        )
        self.cmd_pos = self.cmd_pos + self.cmd_vel * dt

        self.cmd_pos = torch.clamp(
            self.cmd_pos, -self.cfg.track_limit, self.cfg.track_limit
        )

        self.ctrl_target[:, 0, 0] = self.cmd_pos
        self.ctrl_wp = wp.from_torch(self.ctrl_target)
        self.articulation_view.set_attribute(
            "joint_target_pos", self.control, self.ctrl_wp
        )

    def step(self) -> None:
        wind_prob = 0.02
        wind_mask = torch.rand(self._num_envs, device=self._device) < wind_prob

        if wind_mask.any():
            qd_wp = self.articulation_view.get_attribute("joint_qd", self.state_0)
            qd_pt = wp.to_torch(qd_wp)

            wind_strength = 2.0
            rand_vel_pole1 = (
                torch.rand(self._num_envs, device=self._device) * 2.0 - 1.0
            ) * wind_strength
            rand_vel_pole2 = (
                torch.rand(self._num_envs, device=self._device) * 2.0 - 1.0
            ) * wind_strength

            qd_pt[wind_mask, 0, 1] += rand_vel_pole1[wind_mask]
            qd_pt[wind_mask, 0, 2] += rand_vel_pole2[wind_mask]

            self.articulation_view.set_attribute(
                "joint_qd", self.state_0, wp.from_torch(qd_pt)
            )

        sim_dt = self.cfg.dt / self.cfg.sim_substeps

        for _ in range(self.cfg.sim_substeps):
            self.state_0.clear_forces()

            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.step_count += 1
        self.sim_time += self.cfg.dt

        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

        self._compute_reward_and_done()

    def reset(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        if n == 0:
            return

        q_wp = self.articulation_view.get_attribute("joint_q", self.state_0)
        qd_wp = self.articulation_view.get_attribute("joint_qd", self.state_0)
        q_pt = wp.to_torch(q_wp)
        qd_pt = wp.to_torch(qd_wp)

        noise_range = 0.1

        q_pt[env_ids, 0, 0] = (
            torch.rand(n, device=self._device) * 2.0 - 1.0
        ) * noise_range
        q_pt[env_ids, 0, 1] = (
            torch.pi + (torch.rand(n, device=self._device) * 2.0 - 1.0) * noise_range
        )
        q_pt[env_ids, 0, 2] = (
            torch.rand(n, device=self._device) * 2.0 - 1.0
        ) * noise_range

        qd_pt[env_ids, 0, :] = 0.0

        self.cmd_pos[env_ids] = q_pt[env_ids, 0, 0].clone()
        self.cmd_vel[env_ids] = 0.0

        self.articulation_view.set_attribute(
            "joint_q", self.state_0, wp.from_torch(q_pt)
        )
        self.articulation_view.set_attribute(
            "joint_qd", self.state_0, wp.from_torch(qd_pt)
        )

        self.step_count[env_ids] = 0
        self._reward_buf[env_ids] = 0.0
        self._terminated[env_ids] = False
        self._truncated[env_ids] = False
        self._has_been_upright[env_ids] = False

    def _get_tip_position(self, q_pole1, q_pole2):
        """순운동학(FK)을 통해 Base(카트 연결부) 대비 팁의 (x, z) 벡터를 구함"""
        L1, L2 = self.pole_hz * 2.0, self.pole_hz * 2.0  # 각 폴의 길이는 1.0

        # 절대 각도 계산
        q_abs1 = q_pole1
        q_abs2 = q_pole1 + q_pole2

        tip_x = L1 * torch.sin(q_abs1) + L2 * torch.sin(q_abs2)
        tip_z = L1 * torch.cos(q_abs1) + L2 * torch.cos(q_abs2)

        return tip_x, tip_z, L1 + L2

    def _compute_reward_and_done(self):
        q_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_q", self.state_0)
        ).squeeze(1)
        qd_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_qd", self.state_0)
        ).squeeze(1)

        q_cart, q_pole1, q_pole2 = q_pt[:, 0], q_pt[:, 1], q_pt[:, 2]
        qd_cart, qd_pole1, qd_pole2 = qd_pt[:, 0], qd_pt[:, 1], qd_pt[:, 2]

        # ----------------------------------------------------
        # 1. Base -> Tip 벡터 수직 보상 (가장 핵심)
        # ----------------------------------------------------
        tip_x, tip_z, max_len = self._get_tip_position(q_pole1, q_pole2)

        uprightness = tip_z / max_len  # [-1, 1]
        reward_upright = torch.exp(3.0 * (uprightness - 1.0))

        # ----------------------------------------------------
        # 2. 유지 보너스
        # ----------------------------------------------------
        is_upright = tip_z > (max_len * 0.8)
        self._has_been_upright = self._has_been_upright | is_upright

        balance_bonus = (
            is_upright.float() * 2.0 * torch.exp(-0.2 * (qd_pole1**2 + qd_pole2**2))
        )

        # ----------------------------------------------------
        # 3. 가장자리 이탈 방지 및 🔥 중앙 복귀 유도 (수정됨)
        # ----------------------------------------------------
        dist_to_edge = self.cfg.track_limit - torch.abs(q_cart)
        edge_penalty = 0.5 * torch.exp(-dist_to_edge)

        # 🔥 개선: 누운 채로 중앙에 머무는 오판(Reward Hacking) 방지
        # 폴이 80% 이상 서있을 때(is_upright)만 중앙 위치에 비례하여 보너스를 줍니다.
        # q_cart가 0.0에 가까울수록 exp() 값이 1에 수렴하여 최대 0.5의 보너스를 받습니다.
        center_bonus = is_upright.float() * 0.5 * torch.exp(-1.0 * torch.abs(q_cart))

        # ----------------------------------------------------
        # 4. 모터/카트 제어 페널티
        # ----------------------------------------------------
        action_penalty = 0.0005 * (self.cmd_vel / self.cfg.action_scale) ** 2

        # 🔥 기존 center_penalty 대신 center_bonus 합산
        self._reward_buf = (
            reward_upright
            + balance_bonus
            + center_bonus
            - edge_penalty
            - action_penalty
        )

        # ----------------------------------------------------
        # 5. 터미네이션 (종료) 조건
        # ----------------------------------------------------
        is_out_of_bounds = torch.abs(q_cart) >= self.cfg.track_limit
        is_overspeed = (
            (torch.abs(qd_cart) > 500.0)
            | (torch.abs(qd_pole1) > 200.0)
            | (torch.abs(qd_pole2) > 200.0)
        )

        is_fallen = tip_z < (max_len * 0.2)
        has_fallen_after_upright = self._has_been_upright & is_fallen

        self._terminated = is_out_of_bounds | is_overspeed | has_fallen_after_upright
        self._truncated = self.step_count >= self.cfg.max_steps

    def get_observations(self) -> torch.Tensor:
        q_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_q", self.state_0)
        ).squeeze(1)
        qd_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_qd", self.state_0)
        ).squeeze(1)

        q_cart, q_pole1, q_pole2 = q_pt[:, 0], q_pt[:, 1], q_pt[:, 2]
        qd_cart, qd_pole1, qd_pole2 = qd_pt[:, 0], qd_pt[:, 1], qd_pt[:, 2]

        tip_x, tip_z, max_len = self._get_tip_position(q_pole1, q_pole2)

        # 🔥 관측 11차원
        return torch.stack(
            [
                q_cart
                / self.cfg.track_limit,  # 1. 트랙 대비 현재 위치 (가장자리 인식용)
                qd_cart / 10.0,  # 2. 카트 속도
                torch.sin(q_pole1),  # 3. 폴1 각도 (sin)
                torch.cos(q_pole1),  # 4. 폴1 각도 (cos)
                torch.sin(q_pole2),  # 5. 폴2 각도 (sin)
                torch.cos(q_pole2),  # 6. 폴2 각도 (cos)
                qd_pole1 / 10.0,  # 7. 폴1 각속도
                qd_pole2 / 10.0,  # 8. 폴2 각속도
                tip_x / max_len,  # 9. 팁의 X축 오프셋
                tip_z / max_len,  # 10. 팁의 Z축 높이 (가장 중요한 목표 정보)
                self.cmd_vel / self.cfg.action_scale,  # 11. 현재 목표로 하는 속도명령
            ],
            dim=-1,
        )

    def get_rewards(self) -> torch.Tensor:
        return self._reward_buf

    def get_terminated(self) -> torch.Tensor:
        return self._terminated

    def get_truncated(self) -> torch.Tensor:
        return self._truncated

    def episode_log(self) -> dict[str, Any]:
        return {
            "/reward_mean": self._reward_buf.mean(),
        }


def main():
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    wp.init()

    backend_cfg = CartDoublePoleCfg()

    if viewer is not None:
        print("⚠️ Viewer is ENABLED. Rendering might be slow if num_envs is large.")

    backend = CartDoublePoleBackend(backend_cfg, viewer=viewer)

    train_cfg = make_train_cfg()
    train_cfg["actor"]["hidden_dims"] = [128, 128]
    train_cfg["critic"]["hidden_dims"] = [128, 128]
    train_cfg["save_interval"] = 50

    env = GenericRslRlEnv(backend=backend, cfg=train_cfg)

    import datetime

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=f"./logs/cart_double_pole/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        device=str(backend.device),
    )

    print(f"🚀 Cart-Double-Pole Training Started! (Headless: {args.headless})")
    runner.learn(num_learning_iterations=300000)


if __name__ == "__main__":
    main()
