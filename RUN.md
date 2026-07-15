# 실행 방법 (원격/다른 머신)

CPU 전용입니다. GPU/CUDA 없이도 그대로 돌아갑니다.

## 1. 코드 받기
```bash
git clone https://github.com/starplatina0228/CnIE.git   # 처음이면
cd CnIE
git pull                                                # 이미 clone 했으면
```

## 2. 환경 만들기 (둘 중 하나)

### (A) venv + pip — 가장 간단
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### (B) conda
```bash
conda create -n cnie python=3.11 -y
conda activate cnie
pip install -r requirements.txt
```

> torch 설치가 느리거나 실패하면 CPU 휠을 명시:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## 3. 실행

### 스모크런 (동작 검증, ~30초)
```bash
python exp_baselines.py --smoke
```

### 확장 베이스라인 실험 — 논문용 전체 (약 10~15분)
RL(D3QN) + Full-Buffer / Partial-k3 / Nearest-Robot / **Cheapest-Insertion** / **Random**
6개 정책을 (M, λ) 격자에서 동일 평가 seed(paired)로 비교.
```bash
python exp_baselines.py \
    --robots 3 4 5 6 7 \
    --lambdas 300 400 500 600 700 \
    --episodes 1000 --n-eval 20
```
결과: `logs/exp_baselines.csv` (에피소드별 원자료, paired) + 콘솔 비교표.

### 비교 테이블 생성 (위 CSV → 논문용 표 + LaTeX)
```bash
python analyze_baselines.py --csv logs/exp_baselines.csv
# → 콘솔 3개 표 + logs/table_main.tex + logs/summary_by_policy.csv
```

### 버스트 수요 + holding-cost 보상 실험 (RL 우수성 핵심)
비정상(버스트) 수요 + 미래보상(시간평균 WIP) 하에서 RL_flow vs RL_detour(보상
ablation) vs Cheapest/Nearest/Random/Full-Buffer 를 동일 수요·paired seed 로 비교.
```bash
python exp_future_reward.py --smoke                       # 동작 검증(~1분)
python exp_future_reward.py --M 5 --lam 400 \
        --episodes 1200 --n-eval 20                       # 논문용 전체
# 운영점/보상 가중치 스윕 예:
python exp_future_reward.py --M 5 --lam 550 --c-hold 1.0 --c-dist 0.1 --episodes 1500
```
> RL 우수성은 **부하가 충분히 높아 버스트가 실제 saturation 을 일으키는 구간**
> (예: λ̄=500~550/hr, ρ_mean≈0.85~0.92, ρ_peak>1)에서 두드러진다. 저부하에선
> 어떤 정책이든 큐가 안 쌓여 격차가 안 난다.

### 실험 스윕 (권장 — 논문 표 T1~T6을 exp/ 에 기록)
λ·보상가중치 등을 여러 개 돌려 `exp/summary_master.csv` 한 파일로 모읍니다.
```bash
python sweep.py --smoke              # 동작 검증(수 분)
python sweep.py                      # 전체(원격/장시간) → exp/ 에 기록
python sweep.py --only burst_main    # 특정 블록만
python sweep.py --force              # 이미 된 블록도 재실행
```
- 하이퍼파라미터를 바꾸려면 `sweep.py` 의 `build_sweep()` grid(λ 목록, `flow_cfg`
  의 c_hold/c_dist, burst `high/duty`, M)를 편집하세요.
- 산출: `exp/<block>/raw.csv`(paired 원자료), `exp/<block>/config.json`,
  `exp/summary_master.csv`(정책·λ별 mean±std — 여기서 표 피벗).
- 상세표는 개별 블록에: `python analyze_baselines.py --csv exp/burst_main/raw.csv`
- `exp/` 는 gitignore(생성물). 결과 공유는 `git add -f exp/summary_master.csv`.

### 기존 실험 (Phase 1 격자 + Phase 2 γ 민감도)
```bash
python run.py
```

### 시각화
```bash
python viz.py map                  # layout_map.png
python viz.py anim --train 150     # sim_animation.gif
```

## 참고
- 로그는 `logs/*.csv` 로 저장됩니다. `*_smoke.csv` 는 `.gitignore` 처리되어 커밋되지 않습니다.
- 전체 실행 결과(`logs/exp_baselines.csv`)를 다시 커밋해 두면 분석 머신과 공유하기 편합니다.
- 재현성: 모든 실험은 고정 seed(기본 42) 기반이며, 평가는 seed 30000~ 대역을 사용합니다
  (Phase 1 의 10000 대역과 겹치지 않음).
