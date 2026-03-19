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

    # 🔥 수정됨: 이제 action_scale은 '목표 속도(m/s)'의 최대값을 의미합니다.
    action_scale: float = 10.0

    # 트랙 물리적 한계 (이탈 방지용)
    track_limit: float = 10.0

    max_cart_vel: float = 20.0
    max_cart_acc: float = 50.0

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
        self._obs_dim = 6

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
        pole_hx, pole_hy, pole_hz = 0.05, 0.05, 0.5

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

        self._truncated = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )
        self._has_been_upright = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )

        self._upright_time = torch.zeros(self._num_envs, device=self._device)

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

        # 🔥 수정됨: 절대 위치가 아닌 '목표 속도'로 액션을 해석하여 미세 제어 해상도 확보
        target_vel = actions.squeeze(-1) * self.cfg.action_scale

        # 허용된 가속도로 속도 변화량 제한
        max_dv = self.cfg.max_cart_acc * dt
        dv = torch.clamp(target_vel - self.cmd_vel, -max_dv, max_dv)
        self.cmd_vel = self.cmd_vel + dv

        # 허용된 최고 속도로 제한
        self.cmd_vel = torch.clamp(
            self.cmd_vel, -self.cfg.max_cart_vel, self.cfg.max_cart_vel
        )

        # 물리적으로 연속적인 목표 위치 산출
        self.cmd_pos = self.cmd_pos + self.cmd_vel * dt

        # 스케일(트랙 길이) 바깥으로 타겟이 넘어가지 않도록 클램핑
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

        # 🔥 수정됨: 상태 플래그 초기화
        self._has_been_upright[env_ids] = False
        self._upright_time[env_ids] = 0.0

    def _compute_reward_and_done(self):
        q_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_q", self.state_0)
        ).squeeze(1)
        qd_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_qd", self.state_0)
        ).squeeze(1)

        q_cart, q_pole1, q_pole2 = q_pt[:, 0], q_pt[:, 1], q_pt[:, 2]
        qd_cart, qd_pole1, qd_pole2 = qd_pt[:, 0], qd_pt[:, 1], qd_pt[:, 2]

        q_pole1_norm = (q_pole1 + torch.pi) % (2 * torch.pi) - torch.pi
        q_pole2_abs = (q_pole1 + q_pole2 + torch.pi) % (2 * torch.pi) - torch.pi

        abs_q1 = torch.abs(q_pole1_norm)
        abs_q2_abs = torch.abs(q_pole2_abs)

        # ----------------------------------------------------
        # 1. 스윙업 유도 리워드 (기본)
        # ----------------------------------------------------
        reward_swing = torch.cos(q_pole1_norm) + torch.cos(q_pole2_abs)

        # ----------------------------------------------------
        # 2. 수직 상태 판별 및 업데이트
        # ----------------------------------------------------
        is_upright = (abs_q1 < 0.4) & (abs_q2_abs < 0.4)

        # 한 번이라도 세운 적이 있는지 기록
        self._has_been_upright = self._has_been_upright | is_upright

        # ----------------------------------------------------
        # 3. 유지 보너스 (🔥 수정됨: 지수적 폭발 제거, 안정적 지급)
        # ----------------------------------------------------
        balance_accuracy = torch.exp(
            -2.0 * (abs_q1**2 + abs_q2_abs**2) - 0.1 * (qd_pole1**2 + qd_pole2**2)
        )

        # 서 있을 때만 상수 기반의 보너스 지급 (예측 가능하고 안정적인 보상)
        balance_bonus = 10.0 * balance_accuracy
        reward = reward_swing + (is_upright.float() * balance_bonus)

        # ----------------------------------------------------
        # 4. 모터/카트 제어 페널티
        # ----------------------------------------------------
        cart_pos_penalty = 0.05 * (q_cart**2)
        cart_vel_penalty = 0.005 * (qd_cart**2)

        # 위치가 아니라 현재 속도명령 크기에 대한 페널티로 변경하여 과격한 제어 방지
        action_penalty = 0.01 * (self.cmd_vel**2)

        self._reward_buf = reward - cart_pos_penalty - cart_vel_penalty - action_penalty

        # ----------------------------------------------------
        # 5. 터미네이션 (종료) 조건
        # ----------------------------------------------------
        is_out_of_bounds = torch.abs(q_cart) >= self.cfg.track_limit
        is_overspeed = (
            (torch.abs(qd_cart) > 300.0)
            | (torch.abs(qd_pole1) > 100.0)
            | (torch.abs(qd_pole2) > 100.0)
        )

        # 🔥 수정됨: 쓰러짐 판별 및 조기 종료 로직 추가
        is_fallen = (abs_q1 > 1.0) | (abs_q2_abs > 1.0)

        # 성공적으로 세웠다가 다시 떨어뜨렸다면, 질질 끌지 않고 가차없이 에피소드 종료
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

        q_pole1_norm = (q_pole1 + torch.pi) % (2 * torch.pi) - torch.pi
        q_pole2_norm = (q_pole2 + torch.pi) % (2 * torch.pi) - torch.pi

        return torch.stack(
            [q_cart, q_pole1_norm, q_pole2_norm, qd_cart, qd_pole1, qd_pole2], dim=-1
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
