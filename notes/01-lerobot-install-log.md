# 2026-08-18 LeRobot 설치 로그

## 최종 환경 정보
# packages in environment at /home/calvin219/miniconda3/envs/lerobot:
ffmpeg                                  8.0.1            gpl_h43fde53_912       conda-forge
gym-pusht                               0.1.6            pypi_0                 pypi
gymnasium                               1.3.0            pypi_0                 pypi
lerobot                                 0.6.2            pypi_0                 pypi
libopenvino-pytorch-frontend            2025.4.1         hecca717_2             conda-forge
torch                                   2.11.0           pypi_0                 pypi
torchcodec                              0.11.1           pypi_0                 pypi
torchvision                             0.26.0           pypi_0                 pypi

## 검증 결과 (a/b/c)
- (a) import: 통과. `0.6.2 2.11.0+cu130 True` → **CUDA 사용 가능 = True (로컬 GPU 학습 가능)**
- (b) 환경: 통과. PushT reset() 관측 shape `(5,)`
- (c) 데이터셋: 통과. `lerobot/pusht` 로드 성공, len=650
  - key 목록: observation.image, observation.state, action, episode_index,
    frame_index, timestamp, next.reward, next.done, next.success,
    index, task_index, task

## 특이사항
- `pkg_resources is deprecated` 경고 떴으나 무해(에러 아님)
- 환경: WSL, conda env `lerobot`, Python 3.12

## 실패한 것
lerobot-eval --policy.path=lerobot/diffusion_pusht ... 실행 시:
ImportError: 'diffusers' is required but not installed.
원인: diffusion extra 미설치. ACT 사용 예정이라 스킵.
