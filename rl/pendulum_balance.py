import argparse
import torch
import warp as wp
import newton
import newton.examples  # 뷰어 및 파서 사용을 위해 임포트
from newton import JointTargetMode

# rl_template.py에서 제공해주신 기본 구조들 임포트
from rl_template import PhysicsBackend, GenericRslRlEnv, make_train_cfg
from rsl_rl.runners import OnPolicyRunner
from typing import Any
from dataclasses import dataclass


@dataclass
class NewtonPendulumCfg:
    num_envs: int = 1024
    dt: float = 0.02
    sim_substeps: int = 10
    max_steps: int = 1000
    action_scale: float = 3.14159
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NewtonPendulumBackend(PhysicsBackend):
    def __init__(self, cfg: NewtonPendulumCfg, viewer=None):
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
        self._obs_dim = 4

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

        # 🔥 추가: 에이전트가 "이번 생에 한 번이라도 곤두선 적이 있는지" 기억하는 변수
        self._has_reached_top = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )

        # 1. Newton 모델 빌드 및 Vectorization
        base = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(base)

        hx, hy, hz = 1.0, 0.1, 0.1

        link_0 = base.add_link()
        base.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)
        link_1 = base.add_link()
        base.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

        rot_down = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi * 0.5)

        j0 = base.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=rot_down),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        j1 = base.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        base.add_articulation([j0, j1], label="pendulum")

        base.joint_target_mode[j0] = int(JointTargetMode.POSITION)
        base.joint_target_ke[j0] = 10000.0
        base.joint_target_kd[j0] = 500.0

        base.joint_target_mode[j1] = 0
        base.joint_target_ke[j1] = 0.0
        base.joint_target_kd[j1] = 0.0

        builder = newton.ModelBuilder()
        builder.replicate(base, self._num_envs, spacing=(3.0, 3.0, 0.0))
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

        self.ctrl_target = torch.zeros((self._num_envs, 1, 2), device=self._device)
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

    def reset(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        if n == 0:
            return

        q_wp = self.articulation_view.get_attribute("joint_q", self.state_0)
        qd_wp = self.articulation_view.get_attribute("joint_qd", self.state_0)
        q_pt = wp.to_torch(q_wp)
        qd_pt = wp.to_torch(qd_wp)

        noise_range = 0.5  # 약 30도 노이즈

        q_pt[env_ids, 0, 0] = (
            torch.rand(n, device=self._device) * 2.0 - 1.0
        ) * noise_range
        q_pt[env_ids, 0, 1] = (
            torch.rand(n, device=self._device) * 2.0 - 1.0
        ) * noise_range

        qd_pt[env_ids, 0, 0] = 0.0
        qd_pt[env_ids, 0, 1] = 0.0

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

        # 🔥 환경이 리셋될 때 "곤두섰던 기억"도 초기화
        self._has_reached_top[env_ids] = False

    def apply_actions(self, actions: torch.Tensor) -> None:
        actions = torch.clamp(actions, -1.0, 1.0)
        self.ctrl_target[:, 0, 0] = (
            torch.pi + actions.squeeze(-1) * self.cfg.action_scale
        )

        self.ctrl_wp = wp.from_torch(self.ctrl_target)
        self.articulation_view.set_attribute(
            "joint_target_pos", self.control, self.ctrl_wp
        )

    def step(self) -> None:
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

        q_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_q", self.state_0)
        ).squeeze(1)
        qd_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_qd", self.state_0)
        ).squeeze(1)

        q0, q1 = q_pt[:, 0], q_pt[:, 1]
        qd0, qd1 = qd_pt[:, 0], qd_pt[:, 1]

        target_q0 = torch.pi
        target_q1 = 0.0

        q0_diff = (q0 - target_q0 + torch.pi) % (2 * torch.pi) - torch.pi
        q1_diff = (q1 - target_q1 + torch.pi) % (2 * torch.pi) - torch.pi

        # ----------------------------------------------------
        # 🔥 요구사항 1: "Tip이 곤두 선 상태" 판별 및 기억
        # ----------------------------------------------------
        # 두 관절 모두 목표에서 0.5 라디안(약 28도) 이내로 들어오면 곤두섰다고 인정
        is_upright = (torch.abs(q0_diff) < 0.5) & (torch.abs(q1_diff) < 0.5)

        # 이번 스텝에 섰거나, 과거에 한 번이라도 선 적이 있다면 계속 True 유지
        self._has_reached_top = self._has_reached_top | is_upright

        # ----------------------------------------------------
        # 🔥 요구사항 2: 리워드 구조 변경
        # ----------------------------------------------------
        # [상태 A] 아직 한 번도 서본 적 없는 상태 (학습 가이드용 아주 작은 마이너스 보상)
        shaping_reward = -1.0 * (q0_diff**2)

        # [상태 B] 한 번이라도 선 적이 있는 챔피언 상태 (엄청난 유지 보상 + 정밀도 페널티)
        # 기본 10점 만점에서, 완벽하게 수직으로 가만히 있을수록 높은 리워드 획득
        jackpot_reward = (
            10.0 - 5.0 * (q0_diff**2 + q1_diff**2) - 0.1 * (qd0**2 + qd1**2)
        )

        # has_reached_top 여부에 따라 리워드 갈림길
        actual_reward = torch.where(
            self._has_reached_top, jackpot_reward, shaping_reward
        )

        action_penalty = 0.05 * (self.ctrl_target[:, 0, 0] - torch.pi) ** 2

        self._reward_buf = actual_reward - action_penalty

        # ----------------------------------------------------
        # 🔥 요구사항 3: 무너질 경우 터미네이션
        # ----------------------------------------------------
        # 1.0 라디안(약 57도) 이상 벗어나면 무너진 것으로 간주
        is_fallen = torch.abs(q0_diff) > 1.0

        # "한 번이라도 섰는데(has_reached_top) -> 무너졌다(is_fallen)" 일 때만 즉사!
        termination_condition = self._has_reached_top & is_fallen

        # 모터를 비정상적으로 팽이처럼 돌리는 꼼수 즉사 조건은 유지
        is_overspeed = torch.abs(qd0) > 30.0

        self._terminated = termination_condition | is_overspeed
        self._truncated = self.step_count >= self.cfg.max_steps

    def get_observations(self) -> torch.Tensor:
        q_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_q", self.state_0)
        ).squeeze(1)
        qd_pt = wp.to_torch(
            self.articulation_view.get_attribute("joint_qd", self.state_0)
        ).squeeze(1)

        q0, q1 = q_pt[:, 0], q_pt[:, 1]
        qd0, qd1 = qd_pt[:, 0], qd_pt[:, 1]

        target_q0 = torch.pi
        target_q1 = 0.0

        q0_diff = (q0 - target_q0 + torch.pi) % (2 * torch.pi) - torch.pi
        q1_diff = (q1 - target_q1 + torch.pi) % (2 * torch.pi) - torch.pi

        # 신경망에게 "너 지금 섰던 상태야!"라는 정보를 관측값으로 추가로 주면 더 빨리 깨닫습니다.
        has_reached_top_float = self._has_reached_top.float().unsqueeze(1)

        obs = torch.stack([q0_diff, q1_diff, qd0, qd1], dim=-1)

        # 관측값(4개)에 상태 flag(1개)를 붙여서 5차원 관측값 생성
        return torch.cat([obs, has_reached_top_float], dim=-1)

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

    backend_cfg = NewtonPendulumCfg()

    if viewer is not None:
        print("⚠️ Viewer is ENABLED. Rendering might be slow if num_envs is large.")

    backend = NewtonPendulumBackend(backend_cfg, viewer=viewer)

    train_cfg = make_train_cfg()
    train_cfg["actor"]["hidden_dims"] = [64, 64]
    train_cfg["critic"]["hidden_dims"] = [64, 64]
    train_cfg["save_interval"] = 50

    env = GenericRslRlEnv(backend=backend, cfg=train_cfg)

    import datetime

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=f"./logs/newton_pendulum/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        device=str(backend.device),
    )

    print(f"🚀 Newton Physics + RSL-RL Training Started! (Headless: {args.headless})")
    runner.learn(num_learning_iterations=10000)


if __name__ == "__main__":
    main()
