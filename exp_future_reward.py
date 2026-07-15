"""
exp_future_reward.py — 비정상(버스트) 수요 + holding-cost 보상에서 RL 우수성 검증

설계 근거 (why RL should win here)
──────────────────────────────────
정상 Poisson + 동질 로봇 + 거리(Δd) 목적에서는 근시안 그리디(Cheapest-Insertion)가
이미 near-optimal 이라, RL 이 이길 구조적 여지가 없다(phase1 데이터로 확인).
RL 이 미래를 보고 그리디의 국소최적을 깨려면 두 가지가 동시에 필요하다:

  ① 보상을 '미래에 드러나는 비용'으로 바꾼다 (reward_mode="flow")
     R = -( c_hold·(구간 평균 in-system N)/(M·K)  +  c_dist·Δd/D_max )
     즉 결정 사이 구간의 시간평균 혼잡(WIP)을 용량으로 정규화한 비용이다
     (average-cost 대기행렬 제어의 표준형; N 을 낮게 유지 = flow time 감소).
     한 결정이 N(τ)에 주는 영향은 미래 구간에 나타나므로 γ·N-step return 을 통한
     multi-step credit assignment 이 된다. 그리디는 Δd 만 보므로 '지금 몰아넣기
     →나중에 큐 폭발'이라는 국소최적에 빠진다.

  ② 그리디가 구조적으로 불리한 수요를 준다 (arrival="burst")
     평균 부하는 동일(λ̄)하되 순간 2.5λ 의 버스트. 그리디는 매 주문 로봇을
     근시안적으로 소진 → 버스트 때 대량 대기. RL 은 버스트 전에 예비 로봇을
     남기는 anticipation 을 학습할 여지가 생긴다.

비교군
──────
  RL_flow        : ① + ② (제안)
  RL_detour      : 같은 버스트 환경, 기존 Δd 보상  ← 보상 ablation
  BL5_Cheapest   : 한계 detour Δd 최소 그리디
  BL4_Nearest    : 현재거리 최소 그리디
  BL6_Random     : 무작위 (하한)
  BL1_FullBuffer : K 슬롯 채워 출발 (mid-trip 없음)

모두 동일 버스트 수요·동일 평가 seed(paired)에서 평가. 1차 지표는 flow time
(정책 무관한 진짜 목적). 산출 CSV 는 그대로 paired 검정에 넣을 수 있다.

사용법
──────
  python exp_future_reward.py --smoke            # 축소(동작 검증)
  python exp_future_reward.py --episodes 1200 --n-eval 20 --M 5 --lam 400
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

from environment import WarehouseLayout, SIM_TIME
from agents import DQNAgent
from run import (
    RLSimulator, BaselineSimulator, NearestRobotSimulator, train,
)
from baselines_extra import CheapestInsertionSimulator, RandomSimulator
from metrics import MetricsStore, print_comparison, compare_labels

LABELS = ["RL_flow", "RL_detour", "BL5_Cheapest", "BL4_Nearest",
          "BL6_Random", "BL1_FullBuffer"]

# 버스트 프로파일: 평균 부하 보존(0.25·2.5 + 0.75·0.5 = 1.0), 순간 최대 2.5λ.
BURST_CFG = {"period": 1800.0, "duty": 0.25, "high": 2.5, "low": 0.5}


def _eval_one(sim, store, label, i, lam):
    sim.reset(sim.seed)
    sim.run()
    store.add(sim.build_result(label, i, lam,
                               reward=getattr(sim, "total_reward", 0.0)))


def evaluate_burst(layout, agent_flow, agent_detour, store, lam,
                   M, K, seeds, burst_cfg) -> None:
    common = dict(sim_time=SIM_TIME, num_robots=M, robot_capacity=K,
                  arrival="burst", burst_cfg=burst_cfg)
    for i, s in enumerate(seeds):
        _eval_one(RLSimulator(layout, agent_flow, lam=lam, seed=s,
                              reward_mode="flow", eval_mode=True, **common),
                  store, "RL_flow", i, lam)
        _eval_one(RLSimulator(layout, agent_detour, lam=lam, seed=s,
                              reward_mode="detour", eval_mode=True, **common),
                  store, "RL_detour", i, lam)
        _eval_one(CheapestInsertionSimulator(layout, lam=lam, seed=s, **common),
                  store, "BL5_Cheapest", i, lam)
        _eval_one(NearestRobotSimulator(layout, lam=lam, seed=s, **common),
                  store, "BL4_Nearest", i, lam)
        _eval_one(RandomSimulator(layout, lam=lam, seed=s, **common),
                  store, "BL6_Random", i, lam)
        _eval_one(BaselineSimulator(layout, lam=lam, seed=s, **common),
                  store, "BL1_FullBuffer", i, lam)


def _print_headline(store, lam, M, n_eval):
    d = compare_labels(store, ["RL_flow", "RL_detour", "BL5_Cheapest"],
                       lam=lam, num_robots=M, last_n=n_eval * len(LABELS))
    f = lambda lbl: d["avg_flow_time"][lbl]
    base = f("BL5_Cheapest")
    if base and base == base:
        imp_flow   = (base - f("RL_flow"))   / base * 100.0
        imp_detour = (base - f("RL_detour")) / base * 100.0
        print(f"\n  ▶ flow time (낮을수록 좋음, burst 수요):")
        print(f"      RL_flow   {f('RL_flow'):.0f}s  → Cheapest 대비 {imp_flow:+.1f}%")
        print(f"      RL_detour {f('RL_detour'):.0f}s → Cheapest 대비 {imp_detour:+.1f}%")
        print(f"      Cheapest  {base:.0f}s")
        print(f"    (RL_flow > RL_detour 이면 '보상 재설계'가, RL_flow > Cheapest 이면 "
              f"'미래보상+버스트'가 그리디 국소최적을 깬 것)")


def run_experiment(M, lam_per_hr, total_episodes, n_eval, K=5, seed=42,
                   out_dir="logs", out_name="exp_future_reward.csv",
                   flow_cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    layout = WarehouseLayout(seed=seed)
    lam    = lam_per_hr / 3600.0
    seeds  = list(range(40000, 40000 + n_eval))   # 다른 실험과 겹치지 않게

    print(f"{'='*64}\nBurst 수요 + holding-cost 보상  (M={M}, λ̄={lam_per_hr}/hr, "
          f"peak={int(lam_per_hr*BURST_CFG['high'])}/hr)\n{'='*64}")

    # ── ① RL_flow: 미래보상(holding cost) ────────────────────────────────────
    print("\n[학습] RL_flow (reward_mode=flow, arrival=burst) …")
    ag_flow = DQNAgent(num_robots=M, robot_capacity=K, seed=seed, sim_time=SIM_TIME)
    train(layout, ag_flow, MetricsStore(), lam=lam, seed=seed,
          total_episodes=total_episodes, num_robots=M, robot_capacity=K,
          label="RL_flow_train", verbose_every=max(1, total_episodes // 4),
          arrival="burst", burst_cfg=BURST_CFG, reward_mode="flow", flow_cfg=flow_cfg)

    # ── RL_detour: 같은 환경, 기존 Δd 보상 (ablation) ────────────────────────
    print("\n[학습] RL_detour (reward_mode=detour, arrival=burst) …")
    ag_detour = DQNAgent(num_robots=M, robot_capacity=K, seed=seed, sim_time=SIM_TIME)
    train(layout, ag_detour, MetricsStore(), lam=lam, seed=seed,
          total_episodes=total_episodes, num_robots=M, robot_capacity=K,
          label="RL_detour_train", verbose_every=max(1, total_episodes // 4),
          arrival="burst", burst_cfg=BURST_CFG, reward_mode="detour")

    # ── 평가 (동일 burst 수요, paired seeds) ─────────────────────────────────
    print("\n[평가] 6개 정책, 동일 burst 수요 …")
    store = MetricsStore()
    evaluate_burst(layout, ag_flow, ag_detour, store, lam, M, K, seeds, BURST_CFG)
    print_comparison(store, LABELS, lam=lam, num_robots=M, last_n=n_eval,
                     title=f"Burst 수요  M={M}  λ̄={lam_per_hr}/hr")
    _print_headline(store, lam, M, n_eval)

    path = os.path.join(out_dir, out_name)
    store.save_csv(path)
    print(f"\n✔ 원자료 저장 → {path}  ({len(store.records)} rows)")
    return store


def main():
    p = argparse.ArgumentParser(description="버스트 수요 + holding-cost 보상 실험")
    p.add_argument("--M",        type=int, default=5)
    p.add_argument("--lam",      type=int, default=400, help="평균 도착률 λ̄ (orders/hr)")
    p.add_argument("--episodes", type=int, default=1200)
    p.add_argument("--n-eval",   type=int, default=20)
    p.add_argument("--capacity", type=int, default=5)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--c-hold",   type=float, default=1.0, help="holding cost 가중치")
    p.add_argument("--c-dist",   type=float, default=0.2, help="거리항 가중치")
    p.add_argument("--out",      default="exp_future_reward.csv")
    p.add_argument("--smoke",    action="store_true",
                   help="축소(M=5, λ̄=400, ep=80, n_eval=5)")
    args = p.parse_args()

    flow_cfg = {"c_hold": args.c_hold, "c_dist": args.c_dist}

    if args.smoke:
        print(">> SMOKE RUN — 축소 설정 (동작 검증용, 논문 수치 아님)")
        run_experiment(M=5, lam_per_hr=400, total_episodes=80, n_eval=5,
                       K=args.capacity, seed=args.seed,
                       out_name="exp_future_reward_smoke.csv", flow_cfg=flow_cfg)
    else:
        run_experiment(M=args.M, lam_per_hr=args.lam, total_episodes=args.episodes,
                       n_eval=args.n_eval, K=args.capacity, seed=args.seed,
                       out_name=args.out, flow_cfg=flow_cfg)


if __name__ == "__main__":
    main()
