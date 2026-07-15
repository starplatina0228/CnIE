"""
baselines_extra.py — 추가 OR 베이스라인 (실험 파트 심사 강화용)

두 개의 배정 휴리스틱을 추가한다. 둘 다 mid-trip 삽입을 허용하며,
NearestRobotSimulator 와 동일한 이벤트 루프를 공유하되 '어느 로봇에
배정할지'의 선택 규칙만 다르다.

  • BL5_Cheapest : 한계 삽입거리 Δd 를 최소화하는 로봇에 배정 (cheapest-insertion).
        "지금 이 주문을 넣었을 때 그 로봇의 남은 왕복 이동거리가 가장 조금
        늘어나는" 로봇을 고르는 그리디 휴리스틱이다.  ★ 최적(optimal)이 아님 ★:
          (1) 로봇 내부 라우팅이 최적 TSP 가 아니라 nearest-neighbor 근사이고,
          (2) 미래 주문·대기열을 무시한 1-step(근시안) 결정이며,
          (3) 목적이 '로봇 이동거리'이지 flow time 이 아니다.
        핵심: 이 Δd 는 RLSimulator 보상 exp(-α·Δd/D_max) 의 지수부와 같은 양이다.
        즉 BL5 는 RL 보상의 '거리 항'만 근시안적으로 그리디 최소화하는 정책이다
        (큐 압력 γ·항, idle 고갈 β·항, 시간적 lookahead 는 전부 무시).
        → RL 이 이 베이스라인을 이기면, 그 이득이 '다른 목적함수'가 아니라
          큐-인지 + 비근시안(N-step) lookahead 에서 온다는 것을 분리 입증한다.

  • BL6_Random   : 가용 로봇 중 무작위 배정. 정책 학습이 없을 때의 하한(lower
        bound). 다른 모든 정책이 얼마나 랜덤 대비 개선인지를 재는 기준선.

이로써 비교군이
    Full-Buffer / Partial-k / Nearest-Robot(현재거리) / Cheapest(Δd) / Random
로 넓어져, "왜 하필 RL 인가"에 대한 반례(강한 휴리스틱)를 모두 커버한다.

이 파일은 run.py 의 SimulatorBase 를 그대로 상속하므로 지표 계산
(flow time, utilization, dist/order, mid-trip, saving/insertion 등)은
기존 파이프라인과 100% 동일하게 집계된다.
"""
from __future__ import annotations

import random
from typing import List

from manager import (
    Event, Order,
    EVENT_ORDER_ARRIVAL, EVENT_NODE_ARRIVAL, EVENT_TRIP_COMPLETE,
)
from run import SimulatorBase


# ─────────────────────────────────────────────────────────────────────────────
# 공통 골격: mid-trip 삽입을 허용하는 배정 휴리스틱
#   NearestRobotSimulator 와 동일한 이벤트 루프. '_select_robot' 만 다르다.
# ─────────────────────────────────────────────────────────────────────────────

class _InsertionHeuristicSimulator(SimulatorBase):
    """가용 로봇 중 하나를 규칙으로 골라 즉시 배정(필요 시 mid-trip 삽입).

    서브클래스는 `_select_robot(eligible, order)` 만 구현하면 된다.
    나머지(대기열 enqueue, 삽입 회계, trip 완료 시 대기열 dequeue)는 공통.
    """

    def reset(self, seed: int):
        self._reset_common(seed)

    def _select_robot(self, eligible: List, order: Order):
        raise NotImplementedError

    # ── 이벤트 루프 (NearestRobotSimulator 와 동일 구조) ──────────────────────
    def run(self):
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._handle_arrival(ev)
                nxt = self.order_mgr.generate_next_arrival(self.now)
                if nxt <= self.sim_time:
                    self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL))
            elif ev.etype == EVENT_NODE_ARRIVAL:
                self._handle_node_arrival(ev)
            elif ev.etype == EVENT_TRIP_COMPLETE:
                robot = self.robots[ev.data["robot_id"]]
                self._complete_trip(robot)
                orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
                if orders:
                    self._dispatch(robot, orders)

    def _handle_arrival(self, ev: Event):
        order    = self.order_mgr.create_order(self.now)
        eligible = [r for r in self.robots if r.idle_slots > 0]
        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return

        robot   = self._select_robot(eligible, order)
        delta_d = robot.delta_distance_insert(order.loc, self.layout)
        self._extra_travel += delta_d

        if robot.is_idle:
            robot.route.append(order.loc)
            robot.idle_slots = max(0, robot.idle_slots - 1)
            self._robot_orders[robot.id].append(order)
            self._record_service_start(order, self.now)
            robot.is_idle = False
            robot._start_move()
            self._schedule_next(robot)
        else:
            robot.insert_and_reorder(order.loc, self.layout)
            self._robot_orders[robot.id].append(order)
            self._record_service_start(order, self.now)
            self._mid_trip_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# BL-5: Cheapest-Insertion (myopic-optimal)
# ─────────────────────────────────────────────────────────────────────────────

class CheapestInsertionSimulator(_InsertionHeuristicSimulator):
    """주문 삽입 시 그 로봇의 남은 왕복거리 증가분 Δd 가 최소인 로봇에 배정.

    목적식(선택 규칙):  robot* = argmin_r  Δd_r,
        Δd_r = route_total_distance(신규 NN 경로) − route_total_distance(기존 경로)
    즉 '로봇 이동거리'의 한계 증가를 최소화한다. flow/wait time 이나 SKU 는
    목적에 들어가지 않는다.

    RL 과의 대비: RL 은 '지금의 Δd 가 크더라도 전역 flow time 이 낮은' 배정을
    학습으로 노린다. 이 휴리스틱은 매 결정에서 Δd 만 그리디하게 최소화한다.
    (미래 도착·큐 상태를 무시한 1-step 그리디 — 근시안 휴리스틱, 최적 아님)
    """

    def _select_robot(self, eligible, order):
        return min(
            eligible,
            key=lambda r: r.delta_distance_insert(order.loc, self.layout),
        )


# ─────────────────────────────────────────────────────────────────────────────
# BL-6: Random assignment (lower bound)
# ─────────────────────────────────────────────────────────────────────────────

class RandomSimulator(_InsertionHeuristicSimulator):
    """가용 로봇 중 무작위 배정. 정책 없음의 하한 기준선.

    난수는 평가 seed 에 종속시켜 재현 가능하게 한다(같은 seed → 같은 결과).
    """

    def reset(self, seed: int):
        self._reset_common(seed)
        self._rng = random.Random(seed ^ 0x5AB6)   # 주문 스트림 RNG 와 분리

    def _select_robot(self, eligible, order):
        return self._rng.choice(eligible)
