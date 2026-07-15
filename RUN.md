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
