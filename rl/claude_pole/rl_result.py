import argparse
import torch
import warp as wp
import newton
import newton.examples
import time  # 상단에 추가!

from double_pendulum_ver_claude import (
    CartDoublePoleCfg,
    CartDoublePoleBackend,
)  # 기존 훈련 스크립트에서 환경 설정과 백엔드 클래스를 임포트
from rl_template import GenericRslRlEnv, make_train_cfg
from rsl_rl.runners import OnPolicyRunner


def main():
    # 1. Newton 기본 파서 생성 및 커스텀 인자 추가
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the trained model .pt file (e.g., logs/newton_pendulum/model_500.pt)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of environments to render (default: 16)",
    )

    # 파서 초기화 (뷰어 세팅 포함)
    viewer, args = newton.examples.init(parser)

    # 2. Warp 초기화 및 평가용 백엔드 설정
    wp.init()

    backend_cfg = CartDoublePoleCfg()
    # 훈련 때는 2048개였지만, 눈으로 볼 때는 지정한 개수(기본 16개)만 생성
    backend_cfg.num_envs = args.num_envs

    backend = CartDoublePoleBackend(backend_cfg, viewer=viewer)

    # 3. RSL-RL 환경 설정 (훈련 때와 네트워크 크기가 완벽히 동일해야 함)
    train_cfg = make_train_cfg()
    # train_cfg["actor"]["hidden_dims"] = [128, 128]
    # train_cfg["critic"]["hidden_dims"] = [128, 128]

    train_cfg["actor"]["hidden_dims"] = [256, 256, 128]
    train_cfg["critic"]["hidden_dims"] = [256, 256, 128]

    env = GenericRslRlEnv(backend=backend, cfg=train_cfg)

    # 4. 러너 생성 (평가 모드이므로 로그 디렉토리는 임시로 지정)
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir="./logs/newton_pendulum_play",
        device=str(backend.device),
    )

    # 5. 지정한 체크포인트(.pt) 파일 로드
    print(f"Loading checkpoint from {args.ckpt}...")
    runner.load(args.ckpt)

    # 신경망을 추론(Inference) 모드로 추출 (탐험 노이즈 제거)
    policy = runner.get_inference_policy(device=str(backend.device))

    print("🚀 Playing the trained policy! Press Ctrl+C to stop.")

    # 6. 무한 렌더링 루프 (1배속 재생)
    obs = env.get_observations()
    while True:
        step_start_time = time.time()  # 시작 시간 기록

        # 신경망의 순전파(Forward pass) 연산
        with torch.inference_mode():
            actions = policy(obs)

        # 행동을 환경에 적용하고 다음 상태 받아오기
        obs, rewards, dones, infos = env.step(actions)

        # 쓰러지거나 300스텝(6초)이 지나 리셋된 경우 터미널에 알림
        if dones.any():
            print("🔄 Environment reset (fallen or timeout)!")

        # --- ⏱️ 1배속 재생을 위한 타이머 ---
        # 1스텝이 cfg.dt(0.02초)이므로, 연산이 너무 빨리 끝났다면 남은 시간만큼 대기합니다.
        process_time = time.time() - step_start_time
        sleep_time = backend.cfg.dt - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
