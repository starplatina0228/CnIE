"""
exp_baselines.py — 확장 베이스라인 비교 실험

Phase 1(run.py) 의 비교군에 두 개의 강한 OR 베이스라인을 추가해 재실행한다.

  기존 : RL(D3QN) · Full-Buffer · Partial-k3 · Nearest-Robot
  추가 : Cheapest-Insertion (한계 detour Δd 최소 그리디; 근시안 휴리스틱)
         Random           (하한 기준선)

핵심 논지
─────────
RL 보상은  exp(-α·Δd/D_max) − γ·큐압력 − β·idle고갈  이다. 즉 '거리 항' Δd 가
지배적 신호다. Cheapest-Insertion 은 바로 그 Δd 를 매 결정마다 근시안적으로
그리디 최소화하는 정책이다(큐·미래 도착·시간적 lookahead 무시). 최적이 아니라
휴리스틱임에 유의(NN 라우팅, 1-step). 따라서 RL 이 이 베이스라인을 이기면,
그 이득이 '다른 목적함수'가 아니라 큐-인지 + 비근시안 lookahead 에서 온다는
것을 분리해 정량 입증할 수 있다.

주의: 이 실험군의 목적은 모두 '로봇 이동/배정' 최적화다. SKU→저장위치 배치
(slotting/storage assignment) 는 layout 초기화 시 무작위 고정이며 최적화 대상이
아니다(현 논문 스코프 밖).

모든 정책은 (M, λ) 별로 **동일한 평가 seed 집합**에서 평가된다(paired).
따라서 산출 CSV 를 그대로 paired 검정(Wilcoxon 등)에 넣을 수 있다.

산출물
──────
  logs/exp_baselines.csv    — (label, episode, lam, num_robots, …) 에피소드별 원자료
  콘솔                       — (M, λ) 별 정책 비교표 + RL vs Cheapest 개선율 요약

사용법
──────
  # 스모크런(작게, 동작 검증용):
  python exp_baselines.py --smoke

  # 논문용 전체 실행:
  python exp_baselines.py --robots 3 4 5 6 7 \
                          --lambdas 300 400 500 600 700 \
                          --episodes 1000 --n-eval 20
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

from environment import WarehouseLayout, SIM_TIME
from agents import DQNAgent
from run import (
    RLSimulator, BaselineSimulator, PartialSimulator, NearestRobotSimulator,
    train, _TRAIN_LAM,
)
from baselines_extra import CheapestInsertionSimulator, RandomSimulator
from metrics import MetricsStore, print_comparison, compare_labels


# 비교 라벨(표시 순서). print_comparison 은 첫 컬럼부터 이 순서로 그린다.
LABELS = [
    "RL",
    "BL5_Cheapest",
    "BL4_Nearest",
    "BL2_Partial_k3",
    "BL1_FullBuffer",
    "BL6_Random",
]


# ─────────────────────────────────────────────────────────────────────────────
# 확장 평가: 6개 정책을 동일 seed 집합에서 평가
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_extended(
    layout:         WarehouseLayout,
    agent:          DQNAgent,
    store:          MetricsStore,
    lam:            float,
    num_robots:     int,
    robot_capacity: int,
    eval_seeds:     List[int],
    partial_k:      int = 3,
) -> None:
    for i, s in enumerate(eval_seeds):
        # ── RL (D3QN) ────────────────────────────────────────────────────────
        sim = RLSimulator(layout, agent, lam=lam, seed=s,
                          num_robots=num_robots, robot_capacity=robot_capacity,
                          eval_mode=True)
        sim.reset(s); sim.run()
        store.add(sim.build_result("RL", i, lam, reward=sim.total_reward))

        # ── BL1: Full-Buffer ────────────────────────────────────────────────
        sim = BaselineSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                num_robots=num_robots, robot_capacity=robot_capacity)
        sim.reset(s); sim.run()
        store.add(sim.build_result("BL1_FullBuffer", i, lam))

        # ── BL2: Partial-Buffer (k) ─────────────────────────────────────────
        sim = PartialSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                               num_robots=num_robots, robot_capacity=robot_capacity,
                               partial_k=partial_k)
        sim.reset(s); sim.run()
        store.add(sim.build_result("BL2_Partial_k3", i, lam))

        # ── BL4: Nearest-Robot (현재거리) ───────────────────────────────────
        sim = NearestRobotSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                    num_robots=num_robots, robot_capacity=robot_capacity)
        sim.reset(s); sim.run()
        store.add(sim.build_result("BL4_Nearest", i, lam))

        # ── BL5: Cheapest-Insertion (Δd 최소 = myopic-optimal) ──────────────
        sim = CheapestInsertionSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                         num_robots=num_robots,
                                         robot_capacity=robot_capacity)
        sim.reset(s); sim.run()
        store.add(sim.build_result("BL5_Cheapest", i, lam))

        # ── BL6: Random (하한) ──────────────────────────────────────────────
        sim = RandomSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                              num_robots=num_robots, robot_capacity=robot_capacity)
        sim.reset(s); sim.run()
        store.add(sim.build_result("BL6_Random", i, lam))


# ─────────────────────────────────────────────────────────────────────────────
# RL vs Cheapest-Insertion 개선율 요약 (핵심 논지 한 줄 요약)
# ─────────────────────────────────────────────────────────────────────────────

def _print_headline(store: MetricsStore, lam: float, num_robots: int,
                    n_eval: int) -> None:
    data = compare_labels(store, ["RL", "BL5_Cheapest"], lam=lam,
                          num_robots=num_robots, last_n=n_eval * 2)
    rl_flow = data["avg_flow_time"]["RL"]
    ci_flow = data["avg_flow_time"]["BL5_Cheapest"]
    rl_dist = data["dist_per_order"]["RL"]
    ci_dist = data["dist_per_order"]["BL5_Cheapest"]
    if ci_flow and ci_flow == ci_flow:  # not NaN
        d_flow = (ci_flow - rl_flow) / ci_flow * 100.0
        d_dist = (ci_dist - rl_dist) / ci_dist * 100.0 if ci_dist else float("nan")
        print(f"  ▶ RL vs Cheapest-Insertion(Δd 그리디, 근시안): "
              f"flow {d_flow:+.1f}%  ·  dist/order {d_dist:+.1f}%   "
              f"(음수=RL이 더 짧음)")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    robot_counts:   List[int],
    lambdas_per_hr: List[int],
    total_episodes: int,
    n_eval:         int,
    robot_capacity: int = 5,
    seed:           int = 42,
    out_dir:        str = "logs",
    out_name:       str = "exp_baselines.csv",
) -> MetricsStore:
    os.makedirs(out_dir, exist_ok=True)
    layout      = WarehouseLayout(seed=seed)
    lambdas     = [lp / 3600.0 for lp in lambdas_per_hr]
    eval_seeds  = list(range(30000, 30000 + n_eval))   # phase1(10000) 와 겹치지 않게
    store       = MetricsStore()

    for M in robot_counts:
        train_lam = _TRAIN_LAM.get(M, 400 / 3600)
        print(f"\n{'='*64}")
        print(f"M = {M}  (K={robot_capacity}  λ_train={int(train_lam*3600)}/hr  "
              f"episodes={total_episodes})")
        print(f"{'='*64}")

        agent    = DQNAgent(num_robots=M, robot_capacity=robot_capacity,
                            seed=seed, sim_time=SIM_TIME)
        tr_store = MetricsStore()
        train(layout, agent, tr_store,
              lam=train_lam, seed=seed, total_episodes=total_episodes,
              num_robots=M, robot_capacity=robot_capacity,
              label=f"RL_M{M}_train",
              verbose_every=max(1, total_episodes // 4))

        for lam in lambdas:
            rho = lam / (M * (120.0 / 3600))   # 로봇 처리량 ≈ 120/hr 기준 이용률
            print(f"\n  ── λ={int(lam*3600)}/hr  (ρ≈{rho:.2f}) ──")
            ev_store = MetricsStore()
            evaluate_all_extended(layout, agent, ev_store, lam=lam,
                                  num_robots=M, robot_capacity=robot_capacity,
                                  eval_seeds=eval_seeds)
            for r in ev_store.records:
                store.add(r)
            print_comparison(ev_store, LABELS, lam=lam, num_robots=M,
                             last_n=n_eval, title=f"M={M}  λ={int(lam*3600)}/hr")
            _print_headline(ev_store, lam=lam, num_robots=M, n_eval=n_eval)

    out_path = os.path.join(out_dir, out_name)
    store.save_csv(out_path)
    print(f"\n✔ 원자료 저장 → {out_path}   ({len(store.records)} rows)")
    return store


def main():
    p = argparse.ArgumentParser(description="확장 베이스라인 비교 실험")
    p.add_argument("--robots",   type=int, nargs="+", default=[3, 4, 5, 6, 7],
                   help="로봇 수 M 목록")
    p.add_argument("--lambdas",  type=int, nargs="+", default=[300, 400, 500, 600, 700],
                   help="주문 도착률 목록 (orders/hr)")
    p.add_argument("--episodes", type=int, default=1000, help="에이전트당 학습 에피소드")
    p.add_argument("--n-eval",   type=int, default=20,   help="평가 seed 수(paired)")
    p.add_argument("--capacity", type=int, default=5,    help="로봇 용량 K")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--out",      default="exp_baselines.csv")
    p.add_argument("--smoke",    action="store_true",
                   help="축소 설정으로 동작만 검증 (M=[4,5], λ=[400,500], ep=60, n_eval=5)")
    args = p.parse_args()

    if args.smoke:
        print(">> SMOKE RUN — 축소 설정 (동작 검증용, 논문 수치 아님)")
        run_experiment(robot_counts=[4, 5], lambdas_per_hr=[400, 500],
                       total_episodes=60, n_eval=5,
                       robot_capacity=args.capacity, seed=args.seed,
                       out_name="exp_baselines_smoke.csv")
    else:
        run_experiment(robot_counts=args.robots, lambdas_per_hr=args.lambdas,
                       total_episodes=args.episodes, n_eval=args.n_eval,
                       robot_capacity=args.capacity, seed=args.seed,
                       out_name=args.out)


if __name__ == "__main__":
    main()
