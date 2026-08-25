# notes/05-training-run.md

## 설정 근거
- total_frames: 25,650 (학습 로그 dataset.num_frames=25650 으로 재확인)
- batch_size: 8 / steps_per_epoch: 3,206 / epochs: 7.8 → steps: 25,000
- 계획된 50k 대신 25,000 을 쓴 이유:
  Compute HW Guide의 "로보틱스 모방학습은 수십만 스텝이 아니라 데이터셋 기준
  5~10 에폭에서 수렴한다"는 기준으로 역산했다.
  steps_per_epoch = 25,650 / 8 = 3,206 이므로
    - 계획된 50,000 steps = 15.6 epoch (권고 범위의 2배 초과)
    - 25,000 steps       =  7.8 epoch (5~10 범위의 중앙)
  작은 쪽을 택해 시간을 절약하고 여유를 남겼다.
  wandb train/epochs 지표로 검증 완료 (10k step 시점 3.1 epoch, 손계산과 일치).

## 실행 명령 (복붙 가능)
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/pusht \
  --env.type=pusht \
  --policy.device=cuda \
  --steps=25000 \
  --batch_size=8 \
  --env_eval_freq=5000 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --save_freq=5000 \
  --wandb.enable=true \
  --job_name=act_pusht_v1 \
  --output_dir=outputs/train/act_pusht_v1 \
  --policy.repo_id=hersis0219/act_pusht_v1

# 주의 1: --eval_freq 는 이 버전에서 --env_eval_freq 로 이름이 바뀜 
# 주의 2: --policy.scheduler_decay_steps 는 ACT에 없는 필드라 제거함 
# 주의 3: 재실행 시 output_dir 이 이미 있으면 FileExistsError → rm -rf 후 실행
# 실행 환경: tmux 세션 'train' 안에서 conda activate lerobot 후 실행

## 환경
- GPU: NVIDIA GeForce RTX 4060 Ti / VRAM 8GB (8188MiB)
  - 학습 중 실사용 2,482~2,522MiB (약 31%), GPU-Util 69~79%, 온도 69~72C, 94W/160W
  - → batch_size 8 은 과도하게 보수적이었음. 32~64 도 가능했을 것
- 소요시간: 21분 11초 (25,000 steps, 평균 19.67 step/s, 0.046 s/step)
  - 문서 참고사례(50k steps / 2.5h)보다 훨씬 빠름.
    pusht 이미지가 96x96 으로 작아서 (참고사례는 640x480 기준)
- 성능 진단: dataloading_s(~0.0005) << update_s(0.044) → 데이터로더 병목 없음

## wandb
- run 링크: https://wandb.ai/hersis0219-seoul-national-university-ofscience-and-techn/lerobot/runs/bb4ko018
- l1_loss: 시작 0.62 → 끝 0.212
  (6k: 0.37 / 17k: 0.228 / 22k: 0.212 — 발산·정체 없이 꾸준히 하강, 정상)
- kld_loss: 시작 0.45 → 끝 0.001 (collapse 여부: 근접함 — 사실상 0으로 수렴)
  (6k: 0.10 / 17k: 0.002 / 22k: 0.001)
- 기타: grad_norm 150 → 11.9 (안정) / lr 1.0e-05 고정 (scheduler=None)

## eval 결과 (n_episodes=10, env_eval_freq=5000)
| step        | avg_sum_reward | avg_max_reward | pc_success |
|-------------|----------------|----------------|------------|
| 5,000       | ~15.5          | -              | 0.0        |
| 10,000      | ~16.2          | -              | 0.0        |
| 25,000 (최종)| 42.45          | 0.452          | 0.0        |
(참고: D15 스모크 테스트 steps=100 시점의 avg_max_reward 는 0.008)

## 산출물
- HF Hub: https://huggingface.co/hersis0219/act_pusht_v1 (학습 종료 시 자동 push, 207MB)
- 체크포인트: outputs/train/act_pusht_v1/checkpoints/ (005000/010000/015000/020000/025000/last)
- 최종 eval 영상 4개: outputs/train/act_pusht_v1/eval/videos_step_025000/pusht_0/eval_episode_0~3.mp4

## 막힌 것과 해결
1. 에러: `The fields 'scheduler_decay_steps' are not valid for ACTConfig`
   원인: 문서의 lr 스케줄 조언(--policy.scheduler_decay_steps≈--steps)은
        diffusion/smolvla 계열 기준이며 ACT에는 해당 필드가 없음
   해결: 해당 플래그 제거. lr 1e-5 고정으로 학습 (scheduler=None)
2. 에러: `FileExistsError: Output directory ... already exists and resume is False`
   원인: 앞선 실패 run 이 output 폴더를 이미 생성해둠 (덮어쓰기 방지 장치)
   해결: rm -rf outputs/train/act_pusht_v1 후 재실행
3. 에러: `wandb.errors.errors.CommError: user is not logged in` (무한 반복)
   원인: ~/.netrc 에 API 키가 3번 중복 이어붙어 저장됨. 붙여넣기 시 화면에
        표시되지 않아 여러 번 붙여넣은 결과. `wandb login` 은 키를 파일에
        쓰기만 하고 유효성 검사를 하지 않아 "성공"으로 보였음
   해결: rm ~/.netrc → https://wandb.ai/authorize 에서 키 재복사 →
        딱 한 번만 입력. echo 로 길이 확인 후 재실행
4. 주의: tmux 세션은 생성 시 conda 환경이 base 로 초기화됨.
   세션 진입 직후 항상 conda activate lerobot 필요

## 관찰
- pc_success 는 5k~25k 전 구간 0.0 으로 고정. 그러나 avg_sum_reward 는
  16.2 → 42.45 (2.6배), avg_max_reward 는 0.008(D15) → 0.452 로 크게 개선.
  → 성공 판정 임계치는 넘지 못했으나 T블록을 목표 위치의 절반 가까이
    정렬시키는 수준까지 학습됨.
  → 성공률 단일 지표만 보면 "학습 실패"로 오판할 수 있는 사례. l1_loss 하강과
    reward 상승을 함께 봐야 학습 진행을 정확히 판단할 수 있다.

- 개선폭이 후반 구간에 집중됨 (5k→10k 는 +0.7, 10k→25k 는 +26.2).
  5~7 epoch 에서 수렴한다는 문서 권고와 달리 마지막까지 개선이 이어졌으므로
  25,000 steps 완주는 유효한 판단이었음.

- kld_loss 가 0.001 까지 감소 → posterior collapse 에 근접한 상태로 판단.
  CVAE 의 style variable z 가 사실상 사용되지 않는 것으로 추정.
  단 l1_loss 는 정상 하강했고 reward 도 상승했으므로 태스크 학습 자체는 진행됨.
  β(kl_weight) 가 상대적으로 큰 것이 원인일 가능성. (다음에 조정)

- VRAM 을 31% 만 사용하고 GPU-Util 도 69~79% 에 머물렀으므로
  batch_size 를 크게 올려 학습 시간을 더 줄일 수 있었음.

## 향후 변형 실험 후보 (하나만 선택)
1. kl_weight(β) 낮추기 → kld collapse 완화 및 z 활용 여부 검증
2. batch_size 8 → 32 (VRAM 여유 충분, 학습 시간/안정성 비교)
3. chunk_size 100 → 20 (계획서 원안)
