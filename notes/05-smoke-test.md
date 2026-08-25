# notes/04-smoke-test.md

## 실행 환경
- conda env: lerobot / python 3.11 / torch 2.11.0+cu130 / cuda True (GPU 사용 가능)
- gym_pusht: ok (import 성공, "pusht ok" 출력 확인)
- OS: WSL2 (Ubuntu) on Windows

## 실제로 통과한 명령
(복붙 가능한 형태로 — D20 README의 Setup 섹션 재료)

# 0단계 — 환경 확인
conda activate lerobot
python -c "import gym_pusht; print('pusht ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
which lerobot-train lerobot-eval

# 1단계 — 학습 스모크
# 주의: 계획서의 --eval_freq 는 최신 lerobot(0.5.x)에서 --env_eval_freq 로 이름이 바뀜
lerobot-train --policy.type=act --dataset.repo_id=lerobot/pusht \
  --env.type=pusht --steps=100 --batch_size=2 \
  --env_eval_freq=100 --eval.n_episodes=1 --save_freq=100 \
  --wandb.enable=false --policy.push_to_hub=false \
  --output_dir=outputs/smoke

# 2단계 — 체크포인트 경로 확인
ls -R outputs/smoke/checkpoints/

# 3단계 — eval로 mp4 생성
lerobot-eval --policy.path=outputs/smoke/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.n_episodes=2 --eval.batch_size=1

# 4단계 — mp4 확인 (주의: 옵션은 -newermt, -newmt 아님)
find outputs -name "*.mp4" -newermt "-1 hour"

## 실제 체크포인트 경로
outputs/smoke/checkpoints/last/pretrained_model
# 000100/ 과 last/ 두 폴더가 함께 생성됨 (steps=100이라 사실상 동일한 마지막 체크포인트)
# pretrained_model/ 안에 config.json + model.safetensors 존재 확인 (eval 실행 조건 충족)

## 막힌 것과 해결
1. 에러: lerobot-train: error: unrecognized arguments: --eval_freq=100
   원인: 최신 lerobot(0.5.x)에서 train config의 eval_freq 가 env_eval_freq 로 이름 변경됨
   해결: 플래그를 --env_eval_freq=100 으로 교체 → 정상 실행
   참고: unrecognized arguments 에러는 `lerobot-train --help | grep eval` 로 현재 버전의 실제 플래그명 확인 가능


## 산출물
- mp4 경로:
  - outputs/eval/2026-08-25/19-50-24_pusht_act/videos/pusht_0/eval_episode_0.mp4
  - outputs/eval/2026-08-25/19-50-24_pusht_act/videos/pusht_0/eval_episode_1.mp4
  - outputs/smoke/eval/videos_step_000100/pusht_0/eval_episode_0.mp4  (학습 중 env_eval_freq=100 시점에 자동 생성)
- eval 지표: pc_success 0.0 / avg_max_reward ≈ 0.008 (steps=100이라 정상, 오늘은 무의미)
- 소요 시간: 학습 약 1분 (첫 실행은 lerobot/pusht 데이터셋 다운로드 포함) / eval 약  1분
