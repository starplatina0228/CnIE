"""
sweep.py — 실험 스윕 러너 (결과를 exp/ 아래에 체계적으로 기록)

논문 experiments 섹션의 표(T1~T6)를 한 번에 재현 가능하게 돌린다.

설계
────
· '블록(block)' = (학습 컨텍스트 1개) + (평가점 여러 개).
    학습은 블록당 reward_mode 마다 1회만(중복 학습 제거).
    평가는 eval_lams 의 각 λ 에서 모든 정책을 '동일 seed(paired)'로 실행.
· 산출물
    exp/<block>/config.json   — 그 블록의 모든 설정
    exp/<block>/raw.csv       — 에피소드별 원자료(paired; 검정에 바로 투입)
    exp/summary_master.csv    — (block, policy, λ) 별 metric mean±std 한 줄씩
                                ← 여기서 T1~T6 어떤 표든 피벗해서 뽑는다
· 재개: 블록의 raw.csv 가 이미 있으면 건너뛴다(--force 로 강제 재실행).
  master 는 매 실행 끝에 exp/*/raw.csv 전체에서 새로 빌드(중복 없음).

표 ↔ 블록 매핑
──────────────
  T1 방법론 비교(메인) · T2 부하 민감도 · T3 보상 ablation  → 블록 "burst_main"
  T4 수요 레짐(stationary 대조)                            → 블록 "stationary_ref"
  T5 보상 가중치 민감도                                     → 블록 "rw_*"
  T6 M 확장성(선택)                                         → 블록 "Mscale_*"

사용법
──────
  python sweep.py --smoke                 # 축소(동작 검증, 몇 분)
  python sweep.py                         # 전체(논문용; 원격/장시간)
  python sweep.py --only burst_main       # 특정 블록만
  python sweep.py --force                 # 이미 된 블록도 재실행
  # 하이퍼파라미터를 바꿔보려면 아래 build_sweep() 의 grid 를 편집.

하이퍼파라미터 방침
──────────────────
  "그냥 여러 개 돌려보고 좋은 걸 넣는다"가 맞다. λ 와 c_hold/c_dist 는 반드시
  여러 값을 스윕(T2, T5)하고, summary_master.csv 에서 최적 조합을 고르면 된다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from typing import Dict, List, Optional

from environment import WarehouseLayout, SIM_TIME
from agents import DQNAgent
from run import (
    RLSimulator, BaselineSimulator, PartialSimulator, NearestRobotSimulator,
    train, _TRAIN_LAM,
)
from baselines_extra import CheapestInsertionSimulator, RandomSimulator
from metrics import MetricsStore, print_comparison
from analyze_baselines import METRICS   # 지표 정의 재사용(일관성)

EXP_DIR   = "exp"
SEED      = 42
EVAL_SEED0 = 40000                       # 평가 seed 시작(모든 블록 공통 → 블록 간도 paired)
BASELINES = ["BL5_Cheapest", "BL4_Nearest", "BL6_Random",
             "BL1_FullBuffer", "BL2_Partial_k3"]
UPDATES_PER_EP = 20                      # 에피소드당 그래디언트 업데이트
                                         # (기본 train()은 1 → 심한 under-training. 여기서 상향)


# ─────────────────────────────────────────────────────────────────────────────
# 시뮬레이터 팩토리
# ─────────────────────────────────────────────────────────────────────────────

def _make_baseline(label, layout, lam, seed, M, K, arrival, burst_cfg):
    common = dict(sim_time=SIM_TIME, num_robots=M, robot_capacity=K,
                  arrival=arrival, burst_cfg=burst_cfg)
    if label == "BL5_Cheapest":
        return CheapestInsertionSimulator(layout, lam=lam, seed=seed, **common)
    if label == "BL4_Nearest":
        return NearestRobotSimulator(layout, lam=lam, seed=seed, **common)
    if label == "BL6_Random":
        return RandomSimulator(layout, lam=lam, seed=seed, **common)
    if label == "BL1_FullBuffer":
        return BaselineSimulator(layout, lam=lam, seed=seed, **common)
    if label == "BL2_Partial_k3":
        return PartialSimulator(layout, lam=lam, seed=seed, partial_k=3, **common)
    raise ValueError(label)


# ─────────────────────────────────────────────────────────────────────────────
# 블록 실행: 학습 → 평가 → raw.csv/config.json 저장
# ─────────────────────────────────────────────────────────────────────────────

def run_block(block: dict, layout: WarehouseLayout, force: bool) -> Optional[MetricsStore]:
    name    = block["name"]
    bdir    = os.path.join(EXP_DIR, name)
    raw_path = os.path.join(bdir, "raw.csv")
    if os.path.exists(raw_path) and not force:
        print(f"[skip] {name} (raw.csv 존재 — --force 로 재실행)")
        return None

    os.makedirs(bdir, exist_ok=True)
    M, K       = block["M"], block["K"]
    arrival    = block["arrival"]
    burst_cfg  = block.get("burst_cfg")
    flow_cfg   = block.get("flow_cfg", {"c_hold": 1.0, "c_dist": 0.2})
    lam_train  = block["lam_train_hr"] / 3600.0
    episodes   = block["episodes"]
    n_eval     = block["n_eval"]
    eval_lams  = block["eval_lams_hr"]
    reward_modes = block["reward_modes"]     # 학습할 RL 에이전트(예: ["flow","detour"])
    baselines  = block.get("baselines", BASELINES)
    updates    = block.get("updates_per_episode", UPDATES_PER_EP)

    t0 = time.time()
    print(f"\n{'='*66}\n[block] {name}  M={M} arrival={arrival} "
          f"λ_train={block['lam_train_hr']}/hr ep={episodes} "
          f"reward={reward_modes} flow_cfg={flow_cfg}\n{'='*66}")

    # ── 학습 (reward_mode 당 1회) ────────────────────────────────────────────
    agents: Dict[str, DQNAgent] = {}
    for rm in reward_modes:
        print(f"  [train] RL_{rm} …")
        ag = DQNAgent(num_robots=M, robot_capacity=K, seed=SEED, sim_time=SIM_TIME)
        train(layout, ag, MetricsStore(), lam=lam_train, seed=SEED,
              total_episodes=episodes, num_robots=M, robot_capacity=K,
              label=f"RL_{rm}_train", verbose_every=max(1, episodes // 3),
              arrival=arrival, burst_cfg=burst_cfg,
              reward_mode=rm, flow_cfg=flow_cfg,
              updates_per_episode=updates)
        agents[rm] = ag

    # ── 평가 (eval_lams × 모든 정책 × paired seeds) ──────────────────────────
    seeds = list(range(EVAL_SEED0, EVAL_SEED0 + n_eval))
    store = MetricsStore()
    for lam_hr in eval_lams:
        lam = lam_hr / 3600.0
        for i, s in enumerate(seeds):
            for rm, ag in agents.items():
                sim = RLSimulator(layout, ag, lam=lam, seed=s, sim_time=SIM_TIME,
                                  num_robots=M, robot_capacity=K,
                                  arrival=arrival, burst_cfg=burst_cfg,
                                  reward_mode=rm, eval_mode=True)
                sim.reset(s); sim.run()
                store.add(sim.build_result(f"RL_{rm}", i, lam,
                                           reward=sim.total_reward))
            for lbl in baselines:
                sim = _make_baseline(lbl, layout, lam, s, M, K, arrival, burst_cfg)
                sim.reset(s); sim.run()
                store.add(sim.build_result(lbl, i, lam))

    store.save_csv(raw_path)
    cfg = dict(block)
    cfg["updates_per_episode"] = updates
    cfg["elapsed_sec"] = round(time.time() - t0, 1)
    cfg["timestamp"]   = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(bdir, "config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"  ✔ {raw_path}  ({len(store.records)} rows, {cfg['elapsed_sec']}s)")
    return store


# ─────────────────────────────────────────────────────────────────────────────
# master 요약 빌드: exp/*/raw.csv 전체 → exp/summary_master.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_master() -> str:
    rows_out: List[dict] = []
    for name in sorted(os.listdir(EXP_DIR)):
        bdir = os.path.join(EXP_DIR, name)
        raw  = os.path.join(bdir, "raw.csv")
        cfgp = os.path.join(bdir, "config.json")
        if not (os.path.isdir(bdir) and os.path.exists(raw)):
            continue
        cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
        fc  = cfg.get("flow_cfg", {}) or {}
        bc  = cfg.get("burst_cfg", {}) or {}
        # raw 로드
        rows = list(csv.DictReader(open(raw)))
        for r in rows:
            for k, v in r.items():
                if k != "label":
                    try: r[k] = float(v)
                    except (ValueError, TypeError): r[k] = float("nan")
        # (label, λ_eval_hr) 별 집계
        groups: Dict[tuple, List[dict]] = {}
        for r in rows:
            key = (r["label"], round(r["lam"] * 3600))
            groups.setdefault(key, []).append(r)
        for (label, lam_hr), rs in sorted(groups.items()):
            row = {
                "block":        name,
                "arrival":      cfg.get("arrival", ""),
                "M":            cfg.get("M", ""),
                "K":            cfg.get("K", ""),
                "lam_train_hr": cfg.get("lam_train_hr", ""),
                "lam_eval_hr":  lam_hr,
                "c_hold":       fc.get("c_hold", ""),
                "c_dist":       fc.get("c_dist", ""),
                "burst_high":   bc.get("high", ""),
                "burst_duty":   bc.get("duty", ""),
                "episodes":     cfg.get("episodes", ""),
                "n_eval":       len(rs),
                "policy":       label,
            }
            for m, *_ in METRICS:
                vals = [r[m] for r in rs if r[m] == r[m]]
                row[f"{m}_mean"] = round(sum(vals) / len(vals), 4) if vals else ""
                row[f"{m}_std"]  = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
            rows_out.append(row)

    if not rows_out:
        print("[master] 집계할 raw.csv 없음")
        return ""
    fields = list(rows_out[0].keys())
    path = os.path.join(EXP_DIR, "summary_master.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n✔ master 요약 → {path}  ({len(rows_out)} rows)  "
          f"[여기서 T1~T6 표를 피벗]")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 스윕 정의 (여기를 편집해 하이퍼파라미터를 바꾼다)
# ─────────────────────────────────────────────────────────────────────────────

_BURST = {"period": 1800.0, "duty": 0.25, "high": 2.5, "low": 0.5}


def build_sweep(smoke: bool) -> List[dict]:
    if smoke:
        # 동작 검증용 축소 — 논문 수치 아님. 'smoke_' 접두로 real 블록 dir 과 분리한다
        # (smoke 산출물이 exp/burst_main 등 real 결과를 덮거나 skip 시키지 않도록).
        return [
            dict(name="smoke_burst_main", M=5, K=5, arrival="burst", burst_cfg=_BURST,
                 lam_train_hr=550, reward_modes=["flow", "detour"],
                 flow_cfg={"c_hold": 1.0, "c_dist": 0.1},
                 eval_lams_hr=[500, 600], episodes=60, n_eval=3,
                 baselines=["BL5_Cheapest", "BL4_Nearest", "BL6_Random"]),
            dict(name="smoke_stationary_ref", M=5, K=5, arrival="stationary", burst_cfg=None,
                 lam_train_hr=500, reward_modes=["flow", "detour"],
                 flow_cfg={"c_hold": 1.0, "c_dist": 0.1},
                 eval_lams_hr=[500], episodes=60, n_eval=3,
                 baselines=["BL5_Cheapest", "BL4_Nearest", "BL6_Random"]),
        ]

    EP, NE = 2000, 20
    blocks: List[dict] = []
    # T1/T2/T3 — 버스트 메인(방법론 비교 + 부하 민감도 + 보상 ablation)
    blocks.append(dict(
        name="burst_main", M=5, K=5, arrival="burst", burst_cfg=_BURST,
        lam_train_hr=550, reward_modes=["flow", "detour"],
        flow_cfg={"c_hold": 1.0, "c_dist": 0.1},
        eval_lams_hr=[400, 500, 550, 600, 650], episodes=EP, n_eval=NE))
    # T4 — 정상 수요 대조
    blocks.append(dict(
        name="stationary_ref", M=5, K=5, arrival="stationary", burst_cfg=None,
        lam_train_hr=500, reward_modes=["flow", "detour"],
        flow_cfg={"c_hold": 1.0, "c_dist": 0.1},
        eval_lams_hr=[400, 500, 600], episodes=EP, n_eval=NE))
    # T5 — 보상 가중치 민감도 (c_dist 스윕; c_hold 고정)
    for cd in (0.0, 0.05, 0.2, 0.5):
        blocks.append(dict(
            name=f"rw_cd{str(cd).replace('.', '')}", M=5, K=5, arrival="burst",
            burst_cfg=_BURST, lam_train_hr=550, reward_modes=["flow"],
            flow_cfg={"c_hold": 1.0, "c_dist": cd},
            eval_lams_hr=[550], episodes=EP, n_eval=NE))
    # T6 — M 확장성(선택)
    for M in (3, 7):
        blocks.append(dict(
            name=f"Mscale_M{M}", M=M, K=5, arrival="burst", burst_cfg=_BURST,
            lam_train_hr=int(_TRAIN_LAM.get(M, 400 / 3600) * 3600 * 1.3),
            reward_modes=["flow", "detour"],
            flow_cfg={"c_hold": 1.0, "c_dist": 0.1},
            eval_lams_hr=[int(M * 120 * 0.9), int(M * 120 * 1.1)],  # ρ≈0.9,1.1
            episodes=EP, n_eval=NE))
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="실험 스윕 러너 (exp/ 기록)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--only",  nargs="+", default=None, help="특정 블록만 실행")
    p.add_argument("--force", action="store_true", help="이미 된 블록도 재실행")
    args = p.parse_args()

    os.makedirs(EXP_DIR, exist_ok=True)
    layout = WarehouseLayout(seed=SEED)
    blocks = build_sweep(args.smoke)
    if args.only:
        blocks = [b for b in blocks if b["name"] in args.only]
        if not blocks:
            raise SystemExit(f"--only {args.only} 에 해당하는 블록 없음")

    # smoke 는 검증용이므로 항상 새로 실행(스킵 없음). real 만 resume(스킵) 적용.
    force = args.force or args.smoke
    if args.smoke:
        print(">> SMOKE SWEEP — 축소 설정 (동작 검증용, 논문 수치 아님; 매번 재실행)")
    print(f"실행할 블록: {[b['name'] for b in blocks]}")

    primary = None
    for b in blocks:
        st = run_block(b, layout, force=force)
        if primary is None and st is not None:
            primary = (b, st)

    # 메인(첫 블록) 비교표를 콘솔에 (대표 λ = eval_lams 중 최대)
    if primary is not None:
        b, st = primary
        lam_hr = max(b["eval_lams_hr"])
        labels = [f"RL_{rm}" for rm in b["reward_modes"]] + b.get("baselines", BASELINES)
        print_comparison(st, labels, lam=lam_hr / 3600.0, num_robots=b["M"],
                         last_n=b["n_eval"],
                         title=f"[{b['name']}] 방법론 비교  M={b['M']}  λ={lam_hr}/hr")

    build_master()
    print("\n다음: `exp/summary_master.csv` 를 피벗해 T1~T6 표 작성 "
          "(또는 python analyze_baselines.py --csv exp/<block>/raw.csv 로 상세표).")


if __name__ == "__main__":
    main()
