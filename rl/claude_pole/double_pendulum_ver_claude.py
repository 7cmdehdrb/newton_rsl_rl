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

    # 🔥 공격적인 제어: 최대 목표 속도 3배 상향
    action_scale: float = 30.0

    track_limit: float = 10.0
    joint_limit: float = 10.5  # prismatic joint 하드 한계 (obs 계산용)

    # 🔥 속도·가속도 제한 사실상 제거
    max_cart_vel: float = 100.0  # 기존 20.0
    max_cart_acc: float = 1000.0  # 기존 50.0

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
        self._obs_dim = 15  # 🔥 6 → 15

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

        target_vel = actions.squeeze(-1) * self.cfg.action_scale

        # 🔥 max_cart_acc=1000 → max_dv=20 m/s/step: 사실상 즉각 응답
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
        abs_q2 = torch.abs(q_pole2_abs)

        # ----------------------------------------------------------------
        # 🔥 팁 벡터 계산 (prismatic base → pole2 tip)
        #
        #   pole1 단위벡터: (sin θ1,  cos θ1)  (length = 1.0)
        #   pole2 단위벡터: (sin θ2a, cos θ2a) (length = 1.0, 절대각 기준)
        #   tip_vec = pole1_vec + pole2_vec
        #
        #   tip_vz ∈ [-2, +2]
        #     +2 : 두 폴 모두 완전 수직(위)
        #     -2 : 두 폴 모두 완전 역방향(아래)
        #
        #   핵심: pole1이 살짝 기울고 pole2가 반대로 기울어도
        #         합산 tip 벡터가 위를 향하면 양수 보상 → 유연한 스윙업
        # ----------------------------------------------------------------
        tip_vx = torch.sin(q_pole1_norm) + torch.sin(q_pole2_abs)
        tip_vz = torch.cos(q_pole1_norm) + torch.cos(q_pole2_abs)  # [-2, 2]

        tip_mag = torch.sqrt(tip_vx**2 + tip_vz**2 + 1e-6)
        tip_cos_angle = tip_vz / tip_mag  # cos(tip 벡터와 수직축 사이 각) ∈ [-1, 1]

        # ----------------------------------------------------------------
        # 1. 스윙업 리워드: 팁 벡터 z 성분 기반
        #    joint 각도를 개별로 보지 않고 tip 방향 하나로 통합 평가
        # ----------------------------------------------------------------
        reward_swing = tip_vz / 2.0  # [-1, 1]

        # ----------------------------------------------------------------
        # 2. 수직 상태 판별
        # ----------------------------------------------------------------
        is_upright = (abs_q1 < 0.4) & (abs_q2 < 0.4)
        self._has_been_upright = self._has_been_upright | is_upright

        # ----------------------------------------------------------------
        # 3. 밸런싱 보너스: 팁 벡터가 수직에 가까울수록 최대
        #    - tip_cos_angle → 1 일수록 balance_accuracy → 1
        #    - 하늘 방향일 때만 smooth gate (upright_gate) 로 활성화
        #    - 각속도가 크면 불안정으로 간주, 지수적으로 감점
        # ----------------------------------------------------------------
        balance_accuracy = torch.exp(
            -5.0 * (1.0 - tip_cos_angle).clamp(min=0)  # 수직 정렬 정밀도
            - 0.05 * (qd_pole1**2 + qd_pole2**2)  # 각속도 안정화
        )
        upright_gate = torch.clamp(tip_cos_angle, 0.0, 1.0) ** 2  # smooth, 위 방향만
        balance_bonus = 10.0 * balance_accuracy * upright_gate

        reward = reward_swing + balance_bonus

        # ----------------------------------------------------------------
        # 4. 페널티 (공격적 제어 허용에 맞게 scaled)
        # ----------------------------------------------------------------
        cart_pos_penalty = 0.05 * (q_cart / self.cfg.track_limit) ** 2
        cart_vel_penalty = 0.002 * (qd_cart / 30.0) ** 2
        action_penalty = 0.003 * (self.cmd_vel / self.cfg.action_scale) ** 2

        # 🔥 경계 접근 시 비선형 페널티: 2m 이내 접근하면 급격히 증가
        boundary_margin = (self.cfg.track_limit - torch.abs(q_cart)).clamp(min=0)
        boundary_penalty = 0.5 * torch.exp(-boundary_margin / 2.0)

        self._reward_buf = (
            reward
            - cart_pos_penalty
            - cart_vel_penalty
            - action_penalty
            - boundary_penalty
        )

        # ----------------------------------------------------------------
        # 5. 종료 조건 (overspeed 임계값 완화)
        # ----------------------------------------------------------------
        is_out_of_bounds = torch.abs(q_cart) >= self.cfg.track_limit
        is_overspeed = (
            (torch.abs(qd_cart) > 500.0)  # 기존 300 → 500
            | (torch.abs(qd_pole1) > 150.0)  # 기존 100 → 150
            | (torch.abs(qd_pole2) > 150.0)
        )
        is_fallen = (abs_q1 > 1.0) | (abs_q2 > 1.0)
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

        # 각도 정규화
        q_pole1_norm = (q_pole1 + torch.pi) % (2 * torch.pi) - torch.pi
        q_pole2_rel_norm = (q_pole2 + torch.pi) % (2 * torch.pi) - torch.pi  # 🔥 상대각
        q_pole2_abs = (q_pole1 + q_pole2 + torch.pi) % (2 * torch.pi) - torch.pi

        # 🔥 팁 벡터 (normalized, ∈ [-1, 1])
        tip_vx = (torch.sin(q_pole1_norm) + torch.sin(q_pole2_abs)) / 2.0
        tip_vz = (torch.cos(q_pole1_norm) + torch.cos(q_pole2_abs)) / 2.0

        # 🔥 joint 한계까지 거리 (normalized to [0, 1])
        JLIM = self.cfg.joint_limit
        dist_left = (q_cart + JLIM) / (2.0 * JLIM)
        dist_right = (JLIM - q_cart) / (2.0 * JLIM)

        return torch.stack(
            [
                q_cart / self.cfg.track_limit,  # [0]  카트 위치 (normalized)
                dist_left,  # [1]  🔥 왼쪽 joint 한계까지 거리
                dist_right,  # [2]  🔥 오른쪽 joint 한계까지 거리
                torch.cos(q_pole1_norm),  # [3]  pole1 cos
                torch.sin(q_pole1_norm),  # [4]  pole1 sin
                torch.cos(q_pole2_abs),  # [5]  pole2 절대각 cos
                torch.sin(q_pole2_abs),  # [6]  pole2 절대각 sin
                torch.cos(q_pole2_rel_norm),  # [7]  🔥 pole2 상대각 cos
                torch.sin(q_pole2_rel_norm),  # [8]  🔥 pole2 상대각 sin
                qd_cart / 30.0,  # [9]  카트 속도 (scaled)
                qd_pole1 / 10.0,  # [10] pole1 각속도
                qd_pole2 / 10.0,  # [11] pole2 각속도
                self.cmd_vel / self.cfg.action_scale,  # [12] 🔥 명령 속도 (normalized)
                tip_vx,  # [13] 🔥 팁 벡터 x (normalized)
                tip_vz,  # [14] 🔥 팁 벡터 z (normalized)
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
        q_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_q", self.state_0)
        ).squeeze(1)
        q_pole1, q_pole2 = q_pt[:, 1], q_pt[:, 2]
        q_pole1_norm = (q_pole1 + torch.pi) % (2 * torch.pi) - torch.pi
        q_pole2_abs = (q_pole1 + q_pole2 + torch.pi) % (2 * torch.pi) - torch.pi
        tip_vz = torch.cos(q_pole1_norm) + torch.cos(q_pole2_abs)

        return {
            "/reward_mean": self._reward_buf.mean(),
            "/tip_vz_mean": (tip_vz / 2.0).mean(),  # 🔥 팁 방향 모니터링
            "/upright_ratio": self._has_been_upright.float().mean(),  # 🔥 세운 비율
            "/terminated_ratio": self._terminated.float().mean(),  # 🔥 비정상 종료 비율
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
    # 🔥 obs 6→15로 확장되었으므로 네트워크 크기 증가
    train_cfg["actor"]["hidden_dims"] = [256, 256, 128]
    train_cfg["critic"]["hidden_dims"] = [256, 256, 128]
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
