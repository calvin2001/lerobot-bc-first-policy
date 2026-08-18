# 02 — what breaks training

MNIST/MLP로 6가지 조건을 의도적으로 망가뜨려 "증상 → 원인" 매핑을 만들었다.
이 노트의 핵심 산출물은 개별 곡선이 아니라 **맨 위의 판별 순서**다.
3주차에 BC가 안 될 때 위에서부터 훑어 내려간다.

환경: MLP(256-256), AdamW, CrossEntropyLoss, 5 epoch, seed 0, MNIST test 10,000장

---

## 증상 → 원인 판별 순서 (곡선을 보면 위에서부터 체크)

1. **loss 절대값이 비정상인가?** (y축에 1e6 같은 지수 표기)
   → 초기화 / 입력 스케일 문제. 로짓이 거대 → 출력이 거대.
2. **loss가 특정 값에 평평한가?** (ln(클래스수)=2.30, 또는 "평균만 뱉을 때 값")
   → 상수 출력 붕괴. 어떤 자유도가 죽음(뉴런 사망 등). 재시작 필요.
3. **train은 정상인데 test만 나쁜가?**
   → 데이터 파이프라인 문제 (셔플 / 분할 / 누수). 최적화는 멀쩡.
4. **train부터 어중간한 값에서 정체 + 요동인가?**
   → 최적화 노이즈 (배치 과소, lr 부적합). 배치 키우기.

핵심 구분:
- **평평(요동 없음)** = 기울기 0으로 죽음 → 재시작
- **어중간+요동** = 기울기 과잉 진동 → 배치/lr 조정
- 같은 "loss 안 내려감"이라도 원인·처방이 정반대다.

---

## 실험 결과표

| 조건 | 증상 | 진단 근거 | 원인 |
|---|---|---|---|
| baseline (lr 1e-3, bs 64) | test acc 98.0%, loss 0.070 | — | 정상 |
| lr ×100 (→1e-1) | loss 2.30 수평, acc ≈10% | 2.30 = ln(10), logit std≈0 | 과대 스텝 → ReLU 사망 → 상수 출력 붕괴 |
| Normalize 제거 (ToTensor 유지) | 거의 영향 없음, 초기 loss 0.33 vs 0.25, 수렴 1에폭 지연 | 입력이 여전히 [0,1] | 스케일 이미 O(1), mean 0.13 오프셋만 남음 |
| shuffle=False | **train 정상**(98.8%), test만 저하(96.9%), test loss e3부터 반등 0.089→0.114 | train/test 격차만 확대 | 배치 구성 고정 → SGD 정칙화 상실 (MNIST 미정렬로 피해 제한) |
| 정렬 + shuffle off | **train 92% / test 16%** (격차 76%p), 예측 [본인 bincount 결과] 쏠림 | 극단적 train↑ / test↓ | 배치=단일클래스 → 이미지 대신 "배치 순서"를 학습 |
| 초기화 대분산 (std=[본인 확인, 예 10]) | **모양은 정상인데 loss 2.3e6→2.5e5**, acc 81% 정체 | y축 지수, median≪mean | 포화 로짓의 자신만만한 오답. CE 기울기 유계 + AdamW로 발산은 면함 |
| ×255 (raw 스케일) | 예상=붕괴, **실제=정상 학습 96.9%** | 입력 O(100)인데 안 터짐 | **AdamW가 스케일 왜곡을 가림** (SGD였다면 붕괴 예상) |
| batch_size=1 | **train부터 0.213 정체**, test loss 요동 0.234→0.297 | train/test 동시 저하 | gradient 노이즈 √B배 증가 + AdamW 2차 모멘트 팽창 → 유효 lr 감소 |

---

## ⚠ 이번에 발견한 핵심 교훈: AdamW가 병리를 가린다

`×255`, `정렬+shuffle off의 톱니 부재`, `대분산 초기화 회복` — 세 경우 모두
AdamW가 표면 증상을 완화했다. AdamW는 기울기를 2차 모멘트(√v)로 나눠
갱신 폭을 정규화하므로, 스케일이 왜곡돼도·기울기가 급변해도·거대해도 굴러간다.

**"곡선이 그럭저럭 굴러간다 ≠ 문제가 없다."**
SGD였다면 셋 다 붕괴/톱니로 즉시 드러났을 것. (예비일에 SGD로 재현 예정)

---

## 부록: 붕괴 확인용 측정 코드

추측을 사실로 바꾸는 3종. loss 곡선만 보지 말고 숫자로 확정한다.

```python
# 1) 상수 출력 붕괴 확인 (ln 10 케이스)
with torch.no_grad():
    logits = model(x_batch)
    print("logit std:", logits.std(dim=1).mean().item())   # ≈0이면 붕괴
    print("예측 분포:", torch.bincount(logits.argmax(1), minlength=10))  # 한두 칸 쏠림

# 2) 죽은 뉴런 비율 (hook 필요)
#    각 ReLU 출력에 대해:
#    (out == 0).all(dim=0).float().mean()  # 배치 전체에서 항상 0인 뉴런 비율

# 3) 큰 loss가 이상치 지배인지 (대분산 초기화 케이스)
crit = nn.CrossEntropyLoss(reduction='none')
losses = torch.cat([crit(model(x), y) for x, y in test_loader])
print("mean:", losses.mean().item(), "median:", losses.median().item())
# median ≪ mean 이면 소수 이상치가 평균을 지배
```

측정 결과 기록:
- logit std (lr=1e-1): [본인 확인]
- 죽은 뉴런 비율 (lr=1e-1): [본인 확인]
- 예측 분포 (정렬+shuffle off): [본인 bincount]
- ×255 입력 실제 범위: [본인 확인, 0~255 & 평균 33이면 적용 확정]

---

## 곡선 읽기 습관 (3주차 wandb에 그대로 적용)

- **곡선 모양보다 y축부터 본다.** matplotlib/wandb는 축을 자동 확대 →
  노이즈(acc 0.101~0.114)를 극적으로, 큰 값(loss 1e6)을 평평하게 왜곡한다.
- **step 단위 로깅.** epoch 평균은 첫 수십 step의 사건(lr=1e-1 붕괴)을 뭉갠다.
  LeRobot/wandb는 기본이 step 단위.
- **작은 차이(0.1%=10장)는 노이즈.** 시드 하나로 우열 판단 불가.
  → D18 비교표에서 차이가 작으면 "판단 불가"로 쓴다.
- **성분을 따로 본다.** ACT는 l1_loss/kld_loss 분리 관찰 (posterior collapse 감지).

---

## 3주차로 넘어가는 연결

- **평평한 loss의 높이로 원인 역추적**: ln(10) 자리에 ACT는 "action 평균 절대편차"
  (= 입력 무시하고 평균만 뱉을 때의 L1). D12에서 이 기준선을 미리 계산해둘 것.
- **정규화 필수**: 분류(softmax+CE)가 공짜로 갖던 안정성을 회귀(ACT L1)는 정규화로
  손수 확보해야 함. PushT state/action은 [0,512] = O(100).
- **작은 배치 = posterior collapse 위험**: batch=1 노이즈 교훈의 ACT 버전.
  줄여야 하면 gradient accumulation으로 유효 배치 확보.
- **AdamW가 가리는 것 주의**: 3주차도 AdamW → "l1_loss는 내려가는데 롤아웃이 이상"이면
  Adam이 가린 스케일/데이터 문제를 의심.

---

## 미결 / 예비일 할 일
- [ ] bincount, 죽은 뉴런 비율, ×255 입력 범위 실측 채우기
- [ ] train_template.py: `Lambda(x*255)` 줄을 조건부 스위치(--raw255)로 분리
      + lambda → def 함수로 (Windows pickle 대비)
- [ ] (여유 시) 동일 사보타주를 SGD로 재현 → "AdamW가 가린다" 증명
- [ ] Windows num_workers>0 + lambda = pickle 실패. 현재 num_workers=0 회피.
      3주차 본학습 전 WSL2/클라우드 준비.
