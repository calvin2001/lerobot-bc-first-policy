# D12 — pusht 데이터셋 분석

## 자문 4문장

1. observation은 무엇으로 이루어져 있나?
   → observation.image (c, h, w) 와 observation.state (x, y).
     각각의 의미: 이미지(채널,높이,넓이)와 상태(2차원 평면 위에서의 위치좌표)

2. action은 위치 명령인가 속도 명령인가? 근거는?
   → 위치 명령이다. 근거: action 값이 0~512 픽셀 좌표 범위로 state와 같은 스케일이고, 
   action[t]가 state[t]보다 state[t+1]에 더 가깝다. 즉 목표 위치를 지시. /
     그림에서 파란 점이 이동하기 때문.

3. 에피소드 하나는 평균 몇 프레임 / 몇 초인가?
   → 25650 / 206 = 약 125 프레임 = 약 12.5 초 (fps=10)

4. ACT의 chunk_size=100은 이 데이터셋에서 몇 초에 해당하나?
   → 100 / 10 = 10 초. 평균 에피소드의 약 80 %.

## 데이터 구조 표 (README용)

| key | shape | dtype | 역할 |
|-----|-------|-------|------|
| observation.image | [3,96,96] | float32 | 정책 입력 |
| observation.state | [2] | float32 | 정책 입력 |
| action | [2] | float32 | 정책 정답 |

## 포맷 / 환경 메모
- LeRobotDataset v2.0, meta/data/videos 3분할
- av1 코덱: 정상 (이미지 디코딩 OK)
- 환경: WSL2 + conda lerobot, Python 3.12
