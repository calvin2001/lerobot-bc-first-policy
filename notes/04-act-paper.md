# ACT (Zhao et al., 2023, arXiv:2304.13705)

*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*

## 1. 문제

모방학습(BC)의 핵심 문제는 **compounding error(오차 누적)** 이다. 정책이 액션을
한 스텝씩 예측하면, 매 스텝의 작은 예측 오차가 로봇을 데모에서 본 적 없는 상태로
밀어내고, 그 낯선 상태에서 오차가 더 커지는 식으로 눈덩이처럼 누적된다. 특히 밀리미터
단위 정밀함이 필요한 fine manipulation에서는 작은 오차도 바로 실패로 이어지기 때문에
이 문제가 두드러진다.

## 2. 핵심 아이디어

두 개의 기둥으로 compounding error에 대응한다.

- **Action Chunking**: 액션을 하나씩이 아니라 k개 청크로 한 번에 예측·실행한다
  (`πθ(a_{t:t+k}|s_t)`). 모델을 부르는 횟수가 T→T/k로 줄어, effective horizon이
  k배 짧아지고 오차가 누적될 지점도 k배 줄어든다. (ablation: k=1일 때 1% → k=100일 때 44%)
- **Temporal Ensemble**: 청킹만 하면 청크 경계에서 액션이 튀어(jerky motion) 동작이
  끊긴다. 이를 막기 위해 매 스텝 재예측해 청크들을 겹치게 하고, 같은 시점에 대한 여러
  예측을 지수 가중치 `wᵢ=exp(−m·i)`로 가중평균한다. 일반 스무딩과 달리 *같은 시점*의
  예측만 모으므로 bias가 생기지 않는다. 학습 비용 0, 추론 연산만 추가. (효과는 +3.3%로
  chunking보다 작음 = 마무리 다듬기)
- **CVAE**: 사람 데모는 같은 상황에서 매번 다르게 움직여(multimodality) 평균을 내면
  오답이 된다. 정책을 CVAE로 학습해 style 변수 z가 데모의 변동을 흡수하게 한다.
  z는 학습 때만 인코더가 만들고, 추론 때는 0(prior의 평균)으로 고정해 결정론적으로 동작.
  (ablation: 사람 데이터에서 CVAE 빼면 35.3%→2% 급락)

## 3. 구조

CVAE 형태. **인코더는 학습 때만 쓰고 추론 때는 버린다.**

- **CVAE 인코더 (학습 전용)**: 입력 = [CLS] + 관절위치 + 정답 액션 시퀀스(이미지는 안 봄).
  BERT류 트랜스포머를 거쳐 [CLS] 출력으로 z의 평균·분산을 예측.
- **CVAE 디코더 = 정책**:
  - **Vision backbone**: ResNet-18이 각 카메라 이미지를 (15×20×512) 특징맵으로 변환,
    펼쳐서 토큰화 + 2D sinusoidal PosEmb(공간 위치 복원).
  - **Transformer 인코더**: 카메라 특징 + 관절위치 + z를 종합. 모든 입력은 linear로
    512차원에 맞춤.
  - **Transformer 디코더**: k개 고정 위치임베딩(빈 슬롯)을 query로, 인코더 출력을
    key·value로 cross-attention → (k×512) 출력 → MLP로 (k×액션차원)으로 down-projection.
- **손실**: `L = L_recons(재구성, L1) + β·L_reg(KL, z를 N(0,I)로 정규화)`.
  = 데모 액션청크의 log-likelihood 최대화(=NLL 최소화)를 표준 VAE 목적함수로 푼 것.
- 참고: 재구성에 L2 대신 **L1**을 쓰고, 액션은 delta가 아니라 **target(절대 관절위치)**.
  둘 다 실험으로 더 나은 선택임을 확인.

## 4. 한계

ACT는 사람 데모 기반 모방학습이라 데모의 품질과 양에 크게 의존한다. 논문 저자도 사탕
포장 벗기기와 지퍼백 열기에서 실패를 보고했는데, 공통 원인은 이미지에서 물체를 정확히
지각하기 어렵다는 점과 데이터 부족이었다. 또한 각 태스크마다 처음부터(from scratch)
학습해야 하고, 데모와 다른 상황엔 일반화가 약하다. 개선 방향으로는 사전학습, 더 많은
데이터, 더 나은 지각이 제시되었다.

## 5. 내 프로젝트에 적용할 것

논문 기본값(Table III)을 우선 그대로 쓰고, 실제 내 값은 D12(데이터 확인)·D16(학습) 후 채운다.

| 논문 값 | 내 설정 (D16 후 확정) | 이유 |
|---|---|---|
| chunk_size 100 | TBD (데이터 길이 보고) | 논문 최적이 100 근처. LeRobot ACT 기본값으로 시작 후 D18에서 100↔20 대조 |
| kl_weight (β) 10 | 기본값 유지 | β가 z 정보량 조절. 우선 논문값. 바꾸면 posterior collapse 위험 커 후순위 |
| lr 1e-5 | 기본값 유지 | ACT 기본 학습률. 튜닝 우선순위 낮음("기본값으로 시작" 권고) |
| batch 8 | GPU 메모리 따라 조정 | 논문·LeRobot 문서 모두 8부터 시작 권장. OOM이면 낮추기 |

**논문 설정 vs 내 PushT (D12에서 shape 확인해 채울 것):**

| 항목 | 논문 (ALOHA) | 내 PushT |
|---|---|---|
| 카메라 | 4대 (손목×2 + 전면 + 상단) | TBD (보통 1대) |
| state 차원 | 14 (양팔 관절각) | TBD (보통 2, 좌표) |
| action 차원 | 14 (target 관절각) | TBD (보통 2) |
| 데모 수 | 태스크당 50개 | TBD (`len(ds.episodes)`) |
| 성공률 기준선 | 80~90% (하드웨어 튜닝됨) | `lerobot/act_pusht_keypoints` 체크포인트 |

> 주의: 논문의 80~90%는 내 기준선이 아님. 양팔 실제 로봇 + 카메라 4대 + 풀 튜닝 조건.
> 나는 단일 시뮬(PushT)이라 비교 대상은 사전학습 pusht 체크포인트.

## 6. 안 이해된 것 (남겨둘 것)

- **Transformer / Attention**: 인코더·디코더 양쪽에 쓰였는데 개략적 개념만 알고 구체적
  작동은 설명 못 한다. 향후 self-attention·cross-attention·multi-head를 깊이 파 볼 것.
  (지금은 "Q로 질문 → K로 대조 → V 가져옴" 정도로 블랙박스 처리)
- **CVAE 세부**: reparameterization trick(샘플링하면서 역전파되는 원리), ELBO 유도,
  KL의 닫힌 형태 공식. D20 이후 여유 있을 때.
- **PosEmb sinusoidal**: 사인/코사인으로 위치벡터를 만드는 구체적 수식. 지금은
  "위치를 구분시키는 고정 벡터"로만.

---

### D14 확인 완료 체크
- [x] IV. Method (chunking, temporal ensemble, CVAE, 구조, 손실)
- [x] VI. Ablations (chunk size / TE / CVAE 경향)
- [x] VII + Appendix F (한계)
- 건너뜀: II. Related Work, III. ALOHA 하드웨어, V 실험 세부(baseline 깊이)
