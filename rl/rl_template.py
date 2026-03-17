from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner


# ============================================================
# 1) Physics backend 추상 인터페이스
#    - 실제 Newton / MuJoCo / Bullet / custom sim 으로 교체할 부분
# ============================================================


class PhysicsBackend(ABC):
    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_envs(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_actions(self, actions: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        """Return shape: (num_envs, obs_dim)"""
        raise NotImplementedError

    @abstractmethod
    def get_rewards(self) -> torch.Tensor:
        """Return shape: (num_envs,)"""
        raise NotImplementedError

    @abstractmethod
    def get_terminated(self) -> torch.Tensor:
        """True terminal states, shape: (num_envs,)"""
        raise NotImplementedError

    @abstractmethod
    def get_truncated(self) -> torch.Tensor:
        """True timeout states, shape: (num_envs,)"""
        raise NotImplementedError

    @abstractmethod
    def episode_log(self) -> dict[str, torch.Tensor | float]:
        raise NotImplementedError


# ============================================================
# 2) 예시용 mock physics backend
#    - 2D point mass가 원점으로 이동하는 task
#    - 실제 사용 시 이 클래스만 바꾸면 된다
# ============================================================


@dataclass
class MockBackendCfg:
    num_envs: int = 1024
    dt: float = 0.02
    max_steps: int = 300
    action_scale: float = 1.0
    target_radius: float = 0.05
    reset_radius: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MockPointMassBackend(PhysicsBackend):
    """
    state:
      pos: (x, y)
      vel: (vx, vy)

    action:
      acc_cmd: (ax, ay)

    obs:
      [pos_x, pos_y, vel_x, vel_y]
    """

    def __init__(self, cfg: MockBackendCfg):
        self.cfg = cfg
        self._device = torch.device(cfg.device)

        self._num_envs = cfg.num_envs
        self._action_dim = 2
        self._obs_dim = 4

        self.pos = torch.zeros(self._num_envs, 2, device=self._device)
        self.vel = torch.zeros(self._num_envs, 2, device=self._device)
        self.actions = torch.zeros(self._num_envs, 2, device=self._device)
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
        self.pos[env_ids] = (
            torch.rand(n, 2, device=self._device) * 2.0 - 1.0
        ) * self.cfg.reset_radius
        self.vel[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.step_count[env_ids] = 0
        self._reward_buf[env_ids] = 0.0
        self._terminated[env_ids] = False
        self._truncated[env_ids] = False

    def apply_actions(self, actions: torch.Tensor) -> None:
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions.copy_(actions * self.cfg.action_scale)

    def step(self) -> None:
        dt = self.cfg.dt

        # simple dynamics
        self.vel = self.vel + self.actions * dt
        self.pos = self.pos + self.vel * dt
        self.step_count += 1

        # reward: get close to origin, avoid high speed/action
        dist = torch.norm(self.pos, dim=-1)
        speed = torch.norm(self.vel, dim=-1)
        act_mag = torch.norm(self.actions, dim=-1)

        self._reward_buf = 1.0 * torch.exp(-2.0 * dist) - 0.01 * speed - 0.01 * act_mag

        # terminal: success or fly too far away
        success = dist < self.cfg.target_radius
        failure = dist > (self.cfg.reset_radius * 3.0)
        self._terminated = success | failure

        # timeout
        self._truncated = self.step_count >= self.cfg.max_steps

    def get_observations(self) -> torch.Tensor:
        return torch.cat([self.pos, self.vel], dim=-1)

    def get_rewards(self) -> torch.Tensor:
        return self._reward_buf

    def get_terminated(self) -> torch.Tensor:
        return self._terminated

    def get_truncated(self) -> torch.Tensor:
        return self._truncated

    def episode_log(self) -> dict[str, torch.Tensor | float]:
        dist = torch.norm(self.pos, dim=-1)
        return {
            "/dist_mean": dist.mean(),
            "/speed_mean": torch.norm(self.vel, dim=-1).mean(),
        }


# ============================================================
# 3) RSL-RL용 VecEnv 래퍼
#    - backend 종류와 무관하게 이 인터페이스만 맞추면 된다
# ============================================================


class GenericRslRlEnv(VecEnv):
    def __init__(self, backend: PhysicsBackend, cfg: dict[str, Any]):
        self.backend = backend
        self.cfg = cfg

        self.num_envs = backend.num_envs
        self.num_actions = backend.action_dim
        self.device = backend.device

        # RSL-RL 문서상 필요한 필드
        self.max_episode_length = torch.full(
            (self.num_envs,),
            int(cfg["env"]["max_episode_length"]),
            device=self.device,
            dtype=torch.long,
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # optional bookkeeping
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._episode_length = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

    def _make_obs(self) -> TensorDict:
        obs = self.backend.get_observations()
        return TensorDict(
            {
                # 이름은 자유롭게 정할 수 있음.
                # train_cfg["obs_groups"]에서 actor/critic에 연결해준다.
                "policy": obs,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._make_obs()

    def step(self, actions: torch.Tensor):
        self.backend.apply_actions(actions)
        self.backend.step()

        rewards = self.backend.get_rewards()
        terminated = self.backend.get_terminated()
        truncated = self.backend.get_truncated()
        dones = terminated | truncated

        self.episode_length_buf += 1
        self._episode_return += rewards
        self._episode_length += 1

        # 로그는 reset 전에 수집
        extras: dict[str, Any] = {
            "time_outs": truncated,
            "log": self.backend.episode_log(),
        }

        done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            extras["log"]["/episode_return"] = self._episode_return[done_ids].mean()
            extras["log"]["/episode_length"] = (
                self._episode_length[done_ids].float().mean()
            )

            self.backend.reset(done_ids)
            self.episode_length_buf[done_ids] = 0
            self._episode_return[done_ids] = 0.0
            self._episode_length[done_ids] = 0

        obs = self._make_obs()
        return obs, rewards, dones, extras


# ============================================================
# 4) RSL-RL 설정
#    - 현재 문서 구조(v5 계열)에 맞춘 최소 PPO 설정
# ============================================================


def make_train_cfg() -> dict[str, Any]:
    return {
        "runner_class_name": "OnPolicyRunner",
        "run_name": "generic_backend_pointmass",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "logger": "tensorboard",
        # env가 반환한 observation group 이름과
        # actor/critic가 실제로 쓸 observation set을 연결
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 1e-3,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128, 128],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128, 128],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": None,
        },
        # env 자체 설정도 같이 넘겨두면 logger에서 참고 가능
        "env": {
            "max_episode_length": 300,
        },
    }


# ============================================================
# 5) main
# ============================================================


def main():
    backend_cfg = MockBackendCfg(
        num_envs=2048,
        max_steps=300,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    backend = MockPointMassBackend(backend_cfg)

    train_cfg = make_train_cfg()
    env = GenericRslRlEnv(backend=backend, cfg=train_cfg)

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir="./logs/rsl_rl_generic_example",
        device=str(backend.device),
    )

    runner.learn(num_learning_iterations=500)

    # 추론용 policy 추출
    policy = runner.get_inference_policy(device=str(backend.device))
    obs = env.get_observations()
    with torch.inference_mode():
        action = policy.act(obs)
    print("sample action shape:", tuple(action.shape))


if __name__ == "__main__":
    main()
