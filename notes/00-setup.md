# 00-setup.md — D2 (2026-08-12 수) Linux 셸 + Git

출처: MIT The Missing Semester of Your CS Education, Lecture 1 / 2 / 5 / 6

---

## 0. 내 환경

```bash
uname -a          # OS / 커널
echo $SHELL       # 현재 셸
nvidia-smi        # GPU, 드라이버, CUDA 버전
which python; python -V
tmux -V; git --version
```

| 항목 | 값 |
|---|---|
| 호스트명 | `ravenclaw` |
| OS | **WSL2** (Linux 6.18.33.2-microsoft-standard-WSL2, x86_64) |
| 셸 | `/bin/bash` |
| GPU | NVIDIA GeForce RTX 4060 Ti — **8188MiB (8GB)** |
| 드라이버 | 591.86 (nvidia-smi 590.57) |
| CUDA (드라이버 지원 최대) | 13.1 |
| Python | 3.11.15 — `~/miniconda3/envs/dl-summer/bin/python` |
| conda 환경 | `dl-summer` |
| tmux | 3.6 |
| git | 2.53.0 |

**이 환경에서 기억해야 할 것**

- **VRAM 8GB 중 약 1.1GB를 WSLg(`/Xwayland`)가 이미 쓰고 있음.** 실제 학습에 쓸 수 있는 건 약 7GB. 배치 크기 잡을 때 8GB가 아니라 7GB 기준으로 계산.
- 8GB는 실전에서 빠듯한 용량. 처음부터 습관화할 것: **mixed precision(`torch.amp`, bf16)**, `gradient_accumulation_steps`, `batch_size`를 먼저 줄여보는 순서.
- RTX 4060 Ti는 Ada Lovelace(compute capability 8.9) → **bf16 지원.** fp16보다 bf16이 수치적으로 안정적이라 기본으로 bf16을 쓴다.
- 드라이버 버전 591.86은 **Windows 쪽 드라이버**. WSL2는 Windows 드라이버를 그대로 통과시켜 쓰기 때문에, GPU 드라이버 업데이트는 리눅스가 아니라 **Windows에서** 한다. WSL 안에서 `apt install nvidia-driver-*` 하면 오히려 깨진다.
- `CUDA Version: 13.1`은 드라이버가 지원하는 **상한선**이고, 실제 설치된 CUDA 툴킷 버전이 아니다. 실제 확인은 `nvcc -V` 또는 `python -c "import torch; print(torch.version.cuda)"`.
- `which python`이 `envs/dl-summer/bin/python`을 가리키는 걸로 conda 환경이 활성화된 게 확인됨. 프롬프트 앞의 `(dl-summer)`와 함께 항상 이 두 개로 확인하는 습관.
- git 2.53이라 `git switch` / `git restore`(2.23+ 도입) 문제없이 사용 가능.

### WSL2 특성 (일반 리눅스와 다른 점)

```bash
# Windows 드라이브는 /mnt/c 로 마운트됨
ls /mnt/c/Users/

# WSL 안에서 Windows 탐색기 열기
explorer.exe .
```

- **작업 파일은 반드시 `~/` (리눅스 파일시스템) 안에 둔다.** `/mnt/c/...` 경로는 파일 I/O가 수십 배 느려서, 데이터셋을 여기 두면 DataLoader가 병목이 된다. 이게 WSL2에서 가장 흔한 성능 함정.
- `nvidia-smi`의 Processes 섹션에서 **GPU Memory Usage가 `N/A`로 나오는 건 WSL2의 알려진 제약.** 프로세스별 VRAM 사용량 추적이 안 되므로, 어떤 프로세스가 메모리를 잡고 있는지는 `htop`/`pgrep`으로 리눅스 쪽에서 찾아야 한다.
- WSL에 할당되는 RAM/CPU는 Windows의 `C:\Users\<user>\.wslconfig` 에서 조정. 기본값이 호스트 메모리의 절반이라, 학습 중 OOM(시스템 메모리)이 나면 여기를 먼저 본다.
- WSL 재시작은 PowerShell에서 `wsl --shutdown`. GPU가 인식 안 되는 등 이상 증상의 1차 대응.

---

## 1. 셸 기초 (L1)

### 탐색 · 파이프 · 리다이렉션

```bash
pwd                         # 현재 위치
cd -                        # 직전 디렉터리로 토글 (은근 자주 씀)
ls -lah                     # 권한 + 숨김파일 + 사람이 읽는 용량
du -sh *                    # 하위 항목별 용량 (데이터셋 관리에 필수)
df -h                       # 디스크 남은 공간
tree -L 2                   # 2단계까지 구조 보기

# 리다이렉션
python train.py > out.log            # stdout만 파일로
python train.py 2> err.log           # stderr만 파일로
python train.py > out.log 2>&1       # 둘 다 한 파일로
python train.py >> out.log 2>&1 &    # 이어쓰기 + 백그라운드
tail -f out.log                      # 실시간으로 로그 따라가기

# 파이프
cat out.log | grep -i error | wc -l          # 에러 줄 개수
ls -l | sort -k5 -n | tail -5                # 큰 파일 5개
history | awk '{print $2}' | sort | uniq -c | sort -rn | head
```

**핵심 정리**

- `>` 는 덮어쓰기, `>>` 는 이어쓰기. 학습 로그는 항상 `>>`.
- 파일 디스크립터: `0` stdin, `1` stdout, `2` stderr. `2>&1` = "stderr를 stdout이 가는 곳으로 보내라".
- 순서 주의: `2>&1 > file` 은 의도대로 안 됨. **`> file 2>&1`** 이 맞는 순서.
- 파이프는 stdout만 넘긴다. stderr까지 넘기려면 `2>&1 |` 또는 `|&`.

### 권한

`-rwxr-xr-x` 읽는 법: 첫 글자 = 타입(`-` 파일 / `d` 디렉터리 / `l` 심볼릭 링크), 이후 3칸씩 **owner / group / other**. `r=4 w=2 x=1`.

| 값 | 의미 | 쓰는 곳 |
|---|---|---|
| 644 | owner rw / 나머지 r | 일반 파일, 설정 파일 |
| 755 | owner rwx / 나머지 rx | 실행 스크립트, 디렉터리 |
| 600 | owner rw only | **SSH 개인키, .env, 토큰 파일** |
| 700 | owner rwx only | `~/.ssh` 디렉터리 |

```bash
chmod +x train.sh                    # 실행 권한 추가 (상대 방식)
chmod 755 train.sh                   # 절대 방식, 위와 결과 같음
chmod 600 ~/.ssh/id_ed25519          # 이거 안 하면 ssh가 키를 거부함
chmod 700 ~/.ssh
chmod -R 755 scripts/                # 재귀 적용
chown -R $USER:$USER data/           # 소유자 변경 (sudo 필요할 수 있음)
```

**핵심 정리**

- 디렉터리의 `x`는 "실행"이 아니라 **"들어갈 수 있음(cd)"**. `r`만 있으면 목록은 보이지만 접근이 안 됨.
- SSH가 `UNPROTECTED PRIVATE KEY FILE` 에러를 내면 거의 항상 권한 문제 → `chmod 600`.
- `sudo`는 습관적으로 붙이지 않는다. 내 홈 디렉터리 안에서 sudo가 필요하다면 그건 이미 뭔가 잘못됐다는 신호.
- WSL2 주의: `/mnt/c` 아래 Windows 파일은 리눅스 권한 체계가 제대로 적용되지 않아 전부 `777`로 보이는 경우가 많다. 권한 실습은 반드시 `~/` 안에서.

### grep

```bash
grep -i "error" out.log              # 대소문자 무시
grep -n "batch_size" config.yaml     # 줄번호 표시
grep -rn "batch_size" src/           # 디렉터리 재귀 검색 (가장 많이 쓰는 조합)
grep -v "^#" config.yaml             # 주석 줄 제외 (반전)
grep -c "epoch" out.log              # 매칭 줄 개수만
grep -A 5 -B 2 "Traceback" err.log   # 매칭 전후 문맥까지 (에러 추적에 최고)
grep -E "loss|acc" out.log           # 확장 정규식, OR 조건
grep -rl "TODO" src/                 # 파일명만 나열
grep --include="*.py" -rn "seed" .   # 특정 확장자만
```

**핵심 정리**

- 실전에서 가장 자주 쓰는 셋: `-rn`(재귀+줄번호), `-i`(대소문자 무시), `-A/-B`(문맥).
- 검색어에 공백이나 특수문자가 있으면 반드시 따옴표로 감싼다.
- `ripgrep(rg)`이 깔려 있으면 `rg "batch_size" src/` 로 끝. 기본적으로 재귀 + `.gitignore` 존중 + 빠름.

---

## 2. 셸 툴 & 스크립팅 (L2)

```bash
# 특수 변수
$?          # 직전 명령의 종료 코드 (0 = 성공)
$_          # 직전 명령의 마지막 인자
$$          # 현재 셸의 PID
$0          # 스크립트 이름
$1 $2       # 스크립트 인자
$@          # 전체 인자
!!          # 직전 명령 전체 → sudo !! 가 대표 용례

# 제어 연산자
cmd1 ; cmd2       # 순차 실행 (결과 무관)
cmd1 && cmd2      # cmd1 성공했을 때만 cmd2
cmd1 || cmd2      # cmd1 실패했을 때만 cmd2
python train.py && echo "done" || echo "failed"

# 치환
echo "지금: $(date)"        # 명령 치환 — 결과를 문자열로
diff <(ls dirA) <(ls dirB)  # 프로세스 치환 — 출력을 파일처럼

# 확장
mkdir -p exp/{01,02,03}/{ckpt,logs}   # 중괄호 확장, 한 번에 6개 디렉터리
mv log.txt{,.bak}                     # log.txt → log.txt.bak

# 찾기 & 넘기기
find . -name "*.pth" -size +100M              # 100MB 넘는 체크포인트 찾기
find . -name "*.pyc" -delete
find . -name "*.py" | xargs grep -l "torch"   # 찾은 파일들을 grep에 넘김
find . -name "*.log" -mtime +7 -delete        # 7일보다 오래된 로그 삭제

# 도움말
man grep       # 공식 매뉴얼 (길다)
tldr tar       # 예시 위주 요약 (실전에서 훨씬 유용)
grep --help
```

**스크립트 작성 시 첫 줄 관용구**

```bash
#!/usr/bin/env bash
set -euo pipefail
# -e  명령 실패하면 즉시 중단
# -u  선언 안 된 변수 쓰면 에러
# -o pipefail  파이프 중간이 실패해도 잡아냄
```

이거 없으면 스크립트가 중간에 실패해도 계속 굴러가서 조용히 잘못된 결과를 만든다. `shellcheck script.sh` 로 문법 검사하는 습관도 같이.

**막혔던 점 / 헷갈리는 것**

- `$()` 와 `<()` 구분: `$()`는 결과를 **문자열**로, `<()`는 결과를 **파일처럼** 만든다. `diff` 같이 파일 인자를 받는 명령엔 후자.
- 변수는 `"$var"` 처럼 항상 큰따옴표로 감싼다. 경로에 공백 있으면 안 감싼 쪽이 터진다.
- `find`의 `-delete`는 되돌릴 수 없다. 먼저 `-delete` 없이 실행해서 목록을 눈으로 확인한 뒤 붙인다.

---

## 3. 커맨드라인 환경 (L5)

### 환경변수

```bash
echo $PATH                       # 콜론으로 구분된 실행파일 검색 경로
env | sort                       # 전체 환경변수
printenv HOME
which python                     # 지금 어떤 python이 잡히는지
type -a python                   # 후보 전부

export CUDA_VISIBLE_DEVICES=0    # 현재 셸에만 적용
CUDA_VISIBLE_DEVICES=0 python train.py   # 이 명령 한 번만 적용 (권장)

# PATH에 추가
export PATH="$HOME/.local/bin:$PATH"
```

**영구 설정** — `~/.bashrc` 에 추가 후 `source ~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$HOME/.cache/huggingface"   # 모델 캐시 위치 (홈 용량 관리)
export PYTHONDONTWRITEBYTECODE=1            # __pycache__ 안 만들기
export PYTORCH_CUDA_ALLOC_CONF=expand_segments:True   # 8GB 환경 단편화 완화

alias gs='git status -sb'
alias gl='git log --oneline --graph --all'
alias nvs='watch -n 1 nvidia-smi'
alias ll='ls -lah'
```

**핵심 정리**

- `export` 없이 `VAR=1` 만 하면 자식 프로세스에 전달되지 않는다.
- `.bashrc` vs `.bash_profile`: 대화형 셸은 `.bashrc`, 로그인 셸은 `.bash_profile`. 원격 서버에 ssh로 붙으면 로그인 셸이라 `.bash_profile`이 읽히는데, 보통 여기서 `.bashrc`를 source하도록 해둔다. WSL은 터미널에서 바로 들어오면 `.bashrc`가 읽힌다.
- `CUDA_VISIBLE_DEVICES`는 `export`로 박아두면 나중에 "GPU가 하나만 보인다"고 삽질하게 된다. **명령 앞에 붙이는 방식이 안전.** (지금은 GPU가 1장이라 사실상 `0` 고정이지만, 습관을 미리 잡아둔다.)
- 비밀값(API 키 등)은 `.bashrc`에 넣지 말고 `.env` 파일 + `.gitignore` 조합으로.
- conda 초기화 블록은 이미 `.bashrc` 끝에 들어가 있다. 그 블록은 건드리지 말고 내 설정은 그 **아래**에 추가.

### 잡 컨트롤

```bash
Ctrl-C            # SIGINT — 종료 요청
Ctrl-Z            # SIGTSTP — 일시정지
jobs              # 현재 셸의 잡 목록
bg %1             # 1번 잡을 백그라운드에서 재개
fg %1             # 포그라운드로 가져오기
kill -TERM <PID>  # 정상 종료 요청 (먼저 이걸 시도)
kill -9 <PID>     # SIGKILL, 최후의 수단
nohup python train.py > out.log 2>&1 &   # 로그아웃해도 살아남음
ps aux | grep python
pgrep -af train.py
```

`nohup`도 되지만 **장시간 학습은 tmux를 쓴다.** 나중에 다시 붙어서 진행 상황을 눈으로 볼 수 있어서.

학습을 `Ctrl-C`로 죽였는데 `nvidia-smi`에 VRAM이 그대로 잡혀 있으면, 좀비 프로세스가 남은 것. WSL2에선 nvidia-smi가 PID를 안 보여주므로 `pgrep -af python` 으로 찾아서 `kill -9`.

### SSH

```bash
ssh-keygen -t ed25519 -C "ravenclaw-wsl"   # 키 생성 (~/.ssh/id_ed25519)
ssh-copy-id user@1.2.3.4                   # 공개키를 서버에 등록
ssh user@1.2.3.4
ssh -p 2222 user@host                      # 포트 지정

# 파일 전송
scp local.py user@host:~/work/
scp -r user@host:~/work/results ./         # 디렉터리
rsync -avzP local_dir/ user@host:~/dst/    # 중단 후 이어받기 가능, 대용량은 이걸로

# 포트 포워딩 — 서버의 TensorBoard/Jupyter를 로컬 브라우저에서 열기
ssh -L 6006:localhost:6006 user@host       # 로컬 6006 → 서버 6006
ssh -L 8888:localhost:8888 user@host
```

`~/.ssh/config` 에 별칭 등록:

```
Host gpu1
    HostName 1.2.3.4
    User myname
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    LocalForward 6006 localhost:6006
```

이러면 `ssh gpu1` 한 줄로 접속 + 포트포워딩까지 끝. `ServerAliveInterval`은 유휴 상태에서 연결이 끊기는 걸 막아준다.

GitHub 인증도 같은 키로. `ssh -T git@github.com` 으로 연결 테스트.

### tmux

**오늘 배운 것 중 실전에서 가장 자주 쓸 도구.** WSL 터미널 창을 실수로 닫아도 학습이 살아남게 해주는 게 결국 이거 하나다. (단, `wsl --shutdown`이나 Windows 재부팅에는 tmux도 같이 죽는다.)

```bash
tmux new -s train        # 'train' 세션 생성
tmux ls                  # 세션 목록
tmux a -t train          # 재접속 (attach)
tmux kill-session -t train
tmux new -s x -d 'python train.py'   # 세션 만들면서 바로 실행
```

prefix 기본값은 `Ctrl-b`. prefix 누르고 손 떼고 다음 키.

| 키 | 동작 |
|---|---|
| `Ctrl-b` `d` | **detach** — 세션은 계속 살아있음 |
| `Ctrl-b` `%` | 세로 분할 (좌우) |
| `Ctrl-b` `"` | 가로 분할 (상하) |
| `Ctrl-b` 방향키 | pane 이동 |
| `Ctrl-b` `z` | pane 전체화면 토글 |
| `Ctrl-b` `x` | pane 닫기 |
| `Ctrl-b` `c` | 새 window |
| `Ctrl-b` `n` / `p` | 다음 / 이전 window |
| `Ctrl-b` `0`~`9` | 번호로 window 이동 |
| `Ctrl-b` `,` | window 이름 변경 |
| `Ctrl-b` `[` | 스크롤 모드 진입 (`q`로 나감) |
| `Ctrl-b` `?` | 전체 키 목록 |

**계층 구조**: session > window(탭) > pane(분할). 세션 하나에 프로젝트 하나 정도로 잡으면 정리가 된다.

**detach 실습 결과**: 세션 생성 → `sleep 300` 실행 → `Ctrl-b d` → 터미널 창 완전히 종료 → 다시 접속해서 `tmux a -t train` → 프로세스가 그대로 살아있음. 터미널 세션과 프로세스의 수명이 분리된다는 게 핵심.

`~/.tmux.conf` 초안 (다음에 적용):

```
set -g mouse on          # 마우스 스크롤 / pane 크기 조절
set -g history-limit 50000
```

---

## 4. GPU · 프로세스 모니터링

```bash
nvidia-smi                                # 스냅샷
watch -n 1 nvidia-smi                     # 1초마다 갱신
nvidia-smi -l 1                           # 자체 반복 옵션
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv                   # 스크립트로 파싱하기 좋은 형태
```

**내 nvidia-smi 출력 읽는 법 (실측값 기준)**

| 필드 | 내 값 | 의미 |
|---|---|---|
| `Driver Version` | 591.86 | Windows 쪽 드라이버. 업데이트는 Windows에서. |
| `CUDA Version` | 13.1 | 드라이버가 지원하는 **최대** 버전. 설치된 툴킷 버전 아님. |
| `Perf` | P8 | 성능 상태. P8 = 유휴, 학습 중엔 P0/P2로 내려감. |
| `Pwr:Usage/Cap` | 7W / 160W | 유휴 상태. 학습 중 150W 근처면 GPU가 제대로 일하는 것. |
| `Memory-Usage` | 1094MiB / 8188MiB | 1.1GB는 WSLg(`/Xwayland`)가 점유. 가용 약 7GB. |
| `GPU-Util` | 0% | 유휴. **학습 중에 이 값이 낮으면 데이터 로딩 병목.** |
| `Temp` | 42C | 유휴 온도. 학습 중 70~80C는 정상 범위. |

- PyTorch는 캐싱 할당자를 쓰므로 `Memory-Usage`가 실제 텐서 사용량보다 크게 잡혀 있는 게 정상.
- `GPU-Util`이 낮은데 학습이 느리면 `num_workers`, `pin_memory=True`, 데이터셋 위치(`/mnt/c` 금지)를 순서대로 확인.
- **`Processes` 섹션의 GPU Memory Usage가 `N/A`인 건 WSL2 제약.** 프로세스별 VRAM 추적이 안 되니 `htop` / `pgrep`으로 리눅스 쪽에서 찾는다.

```bash
htop
```

| 키 | 동작 |
|---|---|
| `F6` | 정렬 기준 변경 (MEM% / CPU%) |
| `F4` | 필터 (예: `python`) |
| `F5` | 트리 뷰 — 부모/자식 프로세스 관계 |
| `F9` | 프로세스 종료 (시그널 선택) |
| `u` | 특정 사용자만 보기 |
| `q` | 종료 |

```bash
nproc          # 코어 수 → DataLoader num_workers 정할 근거
free -h        # 시스템 RAM (WSL에 할당된 양)
```

`num_workers`는 `nproc` 값의 절반 정도에서 시작해 실측으로 조정. WSL2는 코어/메모리가 `.wslconfig`로 제한되므로 Windows 전체 스펙이 아니라 `nproc` 값을 기준으로 삼는다.

**내 기본 tmux 모니터링 레이아웃**

```
+----------------------+----------------------+
|                      |  watch -n 1 nvidia-smi
|  python train.py     +----------------------+
|  (로그 흐르는 곳)     |  htop                |
+----------------------+----------------------+
```

만드는 순서: `tmux new -s train` → `Ctrl-b %` (좌우 분할) → 오른쪽으로 이동 → `Ctrl-b "` (상하 분할) → 각 pane에서 명령 실행.

---

## 5. Git (L6)

### 기본 흐름

```bash
git init
git status -sb                    # 짧게 보기
git add notes/00-setup.md         # 파일 단위로 의도적으로 추가
git add -p                        # 변경 덩어리 단위로 골라 담기 (커밋 쪼갤 때 유용)
git commit
git log --oneline --graph --all --decorate
git diff                          # 스테이징 전 변경
git diff --staged                 # 스테이징된 변경
git show <hash>
```

`git add .` 를 습관적으로 쓰지 않는다. 의도하지 않은 파일이 섞여 들어가는 사고의 대부분이 여기서 시작된다.

초기 설정 (아직 안 했으면):

```bash
git config --global user.name "..."
git config --global user.email "..."
git config --global init.defaultBranch main
git config --global core.autocrlf input   # WSL: Windows 줄바꿈(CRLF) 섞임 방지
```

WSL에서 Windows 에디터로 파일을 만지면 줄바꿈이 CRLF로 바뀌어 diff가 전체 파일로 잡히는 일이 있다. `core.autocrlf input`으로 예방.

### 브랜치

```bash
git branch                        # 목록
git switch -c feat/data-loader    # 생성 + 이동 (구: checkout -b)
git switch main                   # 이동 (구: checkout)
git merge feat/data-loader        # main에서 실행
git branch -d feat/data-loader    # 병합 완료된 브랜치 삭제
git branch -m old new             # 이름 변경
```

**브랜치 네이밍 규칙 (내 기준)**

```
feat/xxx     새 기능
fix/xxx      버그 수정
exp/xxx      실험 (하이퍼파라미터, 구조 변경)
docs/xxx     문서
```

**핵심 정리**

- 브랜치는 커밋을 가리키는 포인터일 뿐. 만들고 버리는 비용이 거의 0이라 부담 없이 만든다.
- `merge`는 히스토리를 보존(머지 커밋 생김), `rebase`는 선형으로 정리(히스토리 재작성).
- **이미 push한 커밋은 rebase하지 않는다.** 혼자 쓰는 로컬 브랜치에서만.

### 되돌리기

```bash
git restore <file>                # 작업 디렉터리 변경 취소
git restore --staged <file>       # 스테이징만 취소 (변경은 유지)
git commit --amend                # 직전 커밋 메시지/내용 수정
git reset --soft HEAD~1           # 커밋만 취소, 변경은 스테이징에 남김
git reset --hard HEAD~1           # 커밋 + 변경 전부 폐기 (주의)
git stash / git stash pop         # 작업 중인 변경 잠시 치워두기
git reflog                        # 사고 쳤을 때 구조선. HEAD 이동 기록 전부
```

`--hard` 로 날린 것도 커밋된 상태였다면 `reflog`로 대부분 복구된다. 커밋을 자주 하는 게 최고의 안전장치.

### .gitignore

지금 습관을 안 잡으면 **체크포인트 몇 GB를 커밋하는 사고**가 반드시 난다. 한 번 커밋되면 히스토리에 영구히 남아서 지우는 게 훨씬 어렵다.

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.ipynb_checkpoints/

# 비밀값
.env
*.pem
*.key
secrets.yaml

# 데이터 / 산출물
data/
datasets/
*.csv
*.parquet
*.pth
*.ckpt
*.safetensors
*.h5

# 실험 로그
runs/
wandb/
outputs/
lightning_logs/
mlruns/
*.log

# OS / 에디터
.DS_Store
Thumbs.db
desktop.ini
.vscode/
.idea/
```

```bash
git check-ignore -v <file>   # 이 파일이 왜 무시되는지 추적
git rm --cached <file>       # 이미 추적 중인 파일을 추적 해제
```

이미 추적된 파일은 `.gitignore`에 추가해도 계속 추적된다. `git rm --cached` 로 한 번 빼줘야 함. 템플릿은 gitignore.io 에서 스택별로 생성 가능.

### 커밋 메시지 규칙 (내 기준)

- **제목 50자 이내**, 마침표 없음, 명령형 현재시제 ("added" 아니라 "add")
- 제목과 본문 사이 빈 줄 하나
- 본문은 **무엇을 바꿨는지가 아니라 왜 바꿨는지**. 무엇은 diff가 이미 말해준다.
- 접두사(Conventional Commits): `feat` / `fix` / `docs` / `refactor` / `chore` / `test` / `exp`
- 한 커밋 = 한 가지 논리적 변경. 커밋이 커진다 싶으면 `git add -p` 로 쪼갠다.

```
feat: add CIFAR-10 dataloader with disk caching

매 epoch마다 원본 이미지를 디스크에서 다시 읽어
GPU-Util이 30%대에 머물렀음. 첫 epoch에서 텐서로
캐싱하도록 변경, 이후 epoch은 메모리에서 읽음.

- 학습 시간 epoch당 4분 20초 → 1분 05초
- 캐시 메모리 사용량 약 1.2GB
```

```
fix: correct lr scheduler step order

scheduler.step()이 optimizer.step()보다 먼저 호출돼
첫 배치의 lr이 건너뛰어지고 있었음.
```

오늘 쓴 커밋 메시지:

```
docs: add D2 shell/git command notes

Missing Semester 1/2/5/6강 학습 내용 정리.
tmux detach-attach, 권한 표기법, .gitignore 항목 기준을
나중에 다시 찾지 않도록 근거까지 함께 남김.
WSL2 + RTX 4060 Ti(8GB) 환경 실측값도 기록.
```

---

## 오늘 배운 것 중 내일부터 실제로 쓸 것 3개

1. **tmux 세션에서 학습 돌리기** — `tmux new -s train` 으로 시작해서 pane 3분할(학습 / nvidia-smi / htop). 터미널 창을 닫아도 학습이 살아남는 게 유일하게 중요한 부분.
2. **`grep -rn` 으로 코드베이스 탐색** — 파일 열어보며 찾지 않는다. `grep -rn "batch_size" src/` 로 정의 지점을 바로 찾고, 에러 추적은 `grep -A 5 -B 2 "Traceback"`.
3. **작업 시작 전 브랜치 생성 + 작은 단위 커밋** — `git switch -c exp/xxx` 로 시작하고, 실험이 망해도 main은 깨끗하게 유지. 커밋 메시지에 "왜"를 남기는 습관.

## 아직 모르는 것 / 다음에 볼 것

- 실제 설치된 CUDA 툴킷 버전 확인 (`nvcc -V`, `torch.version.cuda`)과 PyTorch가 GPU를 잡는지 검증 (`torch.cuda.is_available()`). **D3에서 가장 먼저 할 것.**
- 8GB VRAM에서 OOM 대응 순서 정리 — batch size, mixed precision(bf16), gradient accumulation, gradient checkpointing.
- `nproc` / `free -h` 실측 후 `.wslconfig` 조정할지 판단, `num_workers` 기준값 정하기.
- `rebase -i` 로 커밋 정리하기 — 개념만 알고 실제로 해본 적 없음. 로컬 브랜치에서 연습 필요.
- 셸 스크립트로 실험 자동화 (하이퍼파라미터 스윕 루프). L2 exercise 다시 볼 것.
- `~/.tmux.conf` 작성 (mouse on, history-limit), dotfiles를 별도 리포로 관리 + 심볼릭 링크 (L5 후반부).
- `git submodule`, Git LFS — 데이터셋/체크포인트 버전 관리가 필요해지면.
