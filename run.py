import csv
import random
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from environment import (
    WarehouseLayout, nearest_neighbor_route,
    SIM_TIME, LAMBDA_DEFAULT, ROBOT_SPEED,
)
from agents import Robot, DQNAgent
from manager import (
    OrderManager, EventQueue, Event, Order,
    EVENT_ORDER_ARRIVAL, EVENT_NODE_ARRIVAL, EVENT_TRIP_COMPLETE,
)
from metrics import MetricsStore, EpisodeResult, print_comparison, _SOLO_RT_M


@dataclass
class TrainingPoint:
    label:         str
    episode:       int
    reward:        float
    loss:          float
    epsilon:       float
    n_completed:   int
    avg_flow_time: float


class TrainingCurveStore:
    def __init__(self):
        self.records: List[TrainingPoint] = []

    def add(self, p: TrainingPoint):
        self.records.append(p)

    def save_csv(self, path: str):
        if not self.records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fields = ["label", "episode", "reward", "loss", "epsilon",
                  "n_completed", "avg_flow_time"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.records:
                w.writerow({k: getattr(r, k) for k in fields})


# ─────────────────────────────────────────────────────────────────────────────
# 보상 함수
# ─────────────────────────────────────────────────────────────────────────────

# 삽입 detour 정규화 상수 ≈ 2 × (aisle-graph 최대 leg 73.7 m).
# 통로 그래프 거리모델 기준으로 재산정(이전 맨해튼 기준 150.9 와 거의 동일).
_D_MAX = 147.4


def compute_reward(
    robot:          Robot,
    order_loc:      Tuple[float, float],
    layout:         WarehouseLayout,
    robots:         Optional[List[Robot]] = None,
    n_wait:         int   = 0,
    num_robots:     int   = 4,
    robot_capacity: int   = 5,
    gamma_q:        float = 0.3,
    alpha_mid:      float = 2.0,
    alpha_idle:     float = 1.0,
    beta_depletion: float = 0.2,
) -> float:

    delta_d = robot.delta_distance_insert(order_loc, layout)
    alpha   = alpha_idle if robot.is_idle else alpha_mid

    reward  = float(np.exp(-alpha * delta_d / _D_MAX))

    if gamma_q > 0.0:
        queue_pressure = min(n_wait / max(1, num_robots * robot_capacity), 1.0)
        reward        -= gamma_q * queue_pressure

    if beta_depletion > 0.0 and robot.is_idle and robots is not None:
        free_idle_left = sum(
            1 for r in robots
            if r.id != robot.id and r.is_idle and r.idle_slots == robot_capacity
        )
        if free_idle_left == 0:
            reward -= beta_depletion

    return reward


# ─────────────────────────────────────────────────────────────────────────────
# 기본 시뮬레이터
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorBase:
    def __init__(self, layout: WarehouseLayout, lam: float, seed: int,
                 sim_time: float, num_robots: int, robot_capacity: int,
                 arrival: str = "stationary", burst_cfg: Optional[Dict] = None):
        self.layout         = layout
        self.lam            = lam
        self.seed           = seed
        self.sim_time       = sim_time
        self.num_robots     = num_robots
        self.robot_capacity = robot_capacity
        self.arrival        = arrival
        self.burst_cfg      = burst_cfg

        self.robots    = [Robot(i, layout.workstation, layout, robot_capacity)
                          for i in range(num_robots)]
        self.order_mgr = OrderManager(layout, lam=lam, seed=seed,
                                      arrival=arrival, burst_cfg=burst_cfg)
        self.eq        = EventQueue()
        self.now       = 0.0

        self._robot_orders:          Dict[int, List[Order]] = {i: [] for i in range(num_robots)}
        self._order_service_start_t: Dict[int, float]       = {}
        self._robot_trip_distance:   Dict[int, float]       = {i: 0.0 for i in range(num_robots)}
        self._total_sim_distance:    float                  = 0.0
        self._total_partial_trips:   int                    = 0
        self._total_trips:           int                    = 0
        self._mid_trip_count:        int                    = 0
        self._extra_travel:          float                  = 0.0

    def _reset_common(self, seed: int):
        self.now = 0.0
        self.order_mgr.reset(seed)
        self.eq  = EventQueue()
        self._robot_orders          = {i: [] for i in range(self.num_robots)}
        self._order_service_start_t = {}
        self._robot_trip_distance   = {i: 0.0 for i in range(self.num_robots)}
        self._total_sim_distance    = 0.0
        self._total_partial_trips   = 0
        self._total_trips           = 0
        self._mid_trip_count        = 0
        self._extra_travel          = 0.0
        for r in self.robots:
            r.reset()
        t0 = self.order_mgr.generate_next_arrival(0.0)
        self.eq.push(Event(t0, EVENT_ORDER_ARRIVAL))

    def _record_service_start(self, order: Order, t: float):
        self._order_service_start_t[id(order)] = t

    def _handle_node_arrival(self, ev: Event):
        robot = self.robots[ev.data["robot_id"]]
        robot.end_travel(self.now)
        dest  = ev.data["dest"]
        if dest == self.layout.workstation:
            robot.pos          = self.layout.workstation
            robot.current_dest = None
            self.eq.push(Event(self.now, EVENT_TRIP_COMPLETE, {"robot_id": robot.id}))
            return
        still_going = robot.arrive_at_node()
        if still_going:
            self._schedule_next(robot)
        else:
            dist_home = self.layout.aisle_distance(robot.pos, self.layout.workstation)
            t_home    = self.now + dist_home / ROBOT_SPEED
            self._robot_trip_distance[robot.id] += dist_home
            self._total_sim_distance            += dist_home
            self.eq.push(Event(t_home, EVENT_NODE_ARRIVAL,
                               {"robot_id": robot.id, "dest": self.layout.workstation}))

    def _complete_trip(self, robot: Robot):
        robot.pos          = self.layout.workstation
        robot.is_idle      = True
        robot.idle_slots   = self.robot_capacity
        robot.route        = []
        robot.current_dest = None
        robot.progress     = 0.0
        completed = list(self._robot_orders.pop(robot.id, []))
        for o in completed:
            self.order_mgr.record_completion(o, self.now)
            self._order_service_start_t.pop(id(o), None)
        if len(completed) < self.robot_capacity:
            self._total_partial_trips += 1
        self._total_trips            += 1
        self._robot_orders[robot.id]  = []

    def _schedule_next(self, robot: Robot):
        if robot.current_dest is None:
            return
        seg = max(0.0, self.layout.aisle_distance(robot.pos, robot.current_dest) - robot.progress)
        self._robot_trip_distance[robot.id] += seg
        self._total_sim_distance            += seg
        t = self.now + seg / ROBOT_SPEED
        robot.start_travel(self.now)
        self.eq.push(Event(t, EVENT_NODE_ARRIVAL,
                           {"robot_id": robot.id, "dest": robot.current_dest}))

    def _dispatch(self, robot: Robot, orders: List[Order]):
        robot.assign_trip([o.loc for o in orders], self.layout)
        self._robot_orders[robot.id] = list(orders)
        for o in orders:
            self._record_service_start(o, self.now)
        self._schedule_next(robot)

    # ── 공통 지표 ─────────────────────────────────────────────────────────────
    def robot_utilization(self) -> float:
        if self.sim_time <= 0:
            return 0.0
        return (sum(r.total_travel_time for r in self.robots)
                / (self.num_robots * self.sim_time))

    def distance_per_completed_order(self) -> float:
        n = self.order_mgr.completed_count()
        return self._total_sim_distance / n if n > 0 else 0.0

    def partial_trip_rate(self) -> float:
        return self._total_partial_trips / max(1, self._total_trips)

    def saving_per_insertion(self) -> float:
        """삽입 1건당 절감 이동거리 = solo_rt − extra_per_insertion."""
        if self._mid_trip_count == 0:
            return 0.0
        extra_per = self._extra_travel / self._mid_trip_count
        return _SOLO_RT_M - extra_per

    def build_result(self, label: str, episode: int, lam: float,
                     reward: float = 0.0, epsilon: float = 0.0) -> EpisodeResult:
        return EpisodeResult(
            label                = label,
            episode              = episode,
            lam                  = lam,
            num_robots           = self.num_robots,
            n_completed          = self.order_mgr.completed_count(),
            n_arrived            = len(self.order_mgr.all_orders),
            avg_flow_time        = self.order_mgr.avg_flow_time(),
            avg_wait_time        = self.order_mgr.avg_wait_time(),
            robot_utilization    = self.robot_utilization(),
            dist_per_order       = self.distance_per_completed_order(),
            partial_trip_rate    = self.partial_trip_rate(),
            mid_trip_count       = self._mid_trip_count,
            mid_trip_rate        = self._mid_trip_count / max(1, self.order_mgr.completed_count()),
            extra_travel_m       = self._extra_travel,
            saving_per_insertion = self.saving_per_insertion(),
            reward               = reward,
            epsilon              = epsilon,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RL 시뮬레이터
# ─────────────────────────────────────────────────────────────────────────────

class RLSimulator(SimulatorBase):
    def __init__(self, layout: WarehouseLayout, agent: DQNAgent,
                 lam: float, seed: int, sim_time: float = SIM_TIME,
                 num_robots: int = 4, robot_capacity: int = 5,
                 reward_cfg: Optional[Dict] = None,
                 eval_mode: bool = False,
                 arrival: str = "stationary", burst_cfg: Optional[Dict] = None,
                 reward_mode: str = "detour",
                 flow_cfg: Optional[Dict] = None):
        super().__init__(layout, lam, seed, sim_time, num_robots, robot_capacity,
                         arrival=arrival, burst_cfg=burst_cfg)
        self.agent        = agent
        self.reward_cfg   = reward_cfg or {}
        self.total_reward = 0.0
        self.eval_mode    = eval_mode
        self._pending: Optional[tuple] = None   # (state, action, reward) — 다음 결정 시점에 완성

        # 보상 모드: "detour"=기존 exp(-α·Δd) / "flow"=혼잡비용(시간평균 WIP)+거리
        self.reward_mode = reward_mode
        self.flow_cfg    = {"c_hold": 1.0, "c_dist": 0.2}
        if flow_cfg:
            self.flow_cfg.update(flow_cfg)
        self._wip_area        = 0.0    # ∫ N(τ) dτ  (order-seconds 누적)
        self._wip_last_t      = 0.0
        self._wip_consumed    = 0.0    # 직전 결정까지 소비한 적분값
        self._last_decision_t = 0.0    # 직전 결정 시각(구간 평균 WIP 계산용)

    def reset(self, seed: int):
        self._reset_common(seed)
        self.total_reward     = 0.0
        self._pending         = None
        self._wip_area        = 0.0
        self._wip_last_t      = 0.0
        self._wip_consumed    = 0.0
        self._last_decision_t = 0.0

    def _wip_advance(self):
        """[_wip_last_t, now] 구간의 WIP 적분을 누적. N 은 그 구간 동안 상수."""
        n = self.order_mgr.in_system()
        self._wip_area  += n * (self.now - self._wip_last_t)
        self._wip_last_t = self.now

    def run(self) -> List[tuple]:
        transitions = []
        saved_epsilon = self.agent.epsilon
        if self.eval_mode:
            self.agent.epsilon = 0.0

        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            self._wip_advance()   # 이벤트 처리 전 상태의 N 으로 적분(구간별 정확)
            if ev.etype == EVENT_ORDER_ARRIVAL:
                t = self._handle_order_arrival(ev)
                if t:
                    transitions.append(t)
                nxt = self.order_mgr.generate_next_arrival(self.now)
                if nxt <= self.sim_time:
                    self.eq.push(Event(nxt, EVENT_ORDER_ARRIVAL))
            elif ev.etype == EVENT_NODE_ARRIVAL:
                self._handle_node_arrival(ev)
            elif ev.etype == EVENT_TRIP_COMPLETE:
                self._handle_trip_complete(ev)

        # 에피소드 종료: 마지막 결정은 다음 결정 상태가 없으므로 done=True로 마감
        # (done=1이면 학습 타겟에서 next_q가 무시되므로 ns 자리는 사용되지 않는다)
        if not self.eval_mode and self._pending is not None:
            s, a, r = self._pending
            transitions.append((s, a, r, s, True))
            self._pending = None

        if self.eval_mode:
            self.agent.epsilon = saved_epsilon
        return transitions

    def _handle_order_arrival(self, ev: Event) -> Optional[tuple]:
        order    = self.order_mgr.create_order(self.now)
        eligible = [r for r in self.robots if r.idle_slots > 0]

        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return None

        state  = self.agent.encode_state(order.loc, self.robots,
                                         self.order_mgr.queue_size(),
                                         now=self.now)
        mask   = self.agent.build_action_mask(self.robots, moving_only=False)
        action = self.agent.select_action(state, mask)
        if action < 0:
            action = min(eligible,
                         key=lambda r: r.delta_distance_insert(order.loc, self.layout)).id

        robot   = self.robots[action]
        delta_d = robot.delta_distance_insert(order.loc, self.layout)
        self._extra_travel += delta_d

        if self.reward_mode == "flow":
            # 직전 결정 이후 구간의 '평균 in-system 수'(혼잡)를 용량으로 정규화한 비용
            # + 거리항. 부호는 비용의 음수. 한 결정이 N(τ)에 주는 영향은 미래 구간에
            # 나타나므로 γ·N-step return 을 통한 multi-step credit assignment 이 된다.
            holding = self._wip_area - self._wip_consumed          # ∫N dt (구간)
            dt      = max(self.now - self._last_decision_t, 1e-6)
            avg_n   = holding / dt                                 # 구간 평균 in-system
            self._wip_consumed    = self._wip_area
            self._last_decision_t = self.now
            fc     = self.flow_cfg
            cap    = max(1, self.num_robots * self.robot_capacity)
            reward = -(fc["c_hold"] * avg_n / cap
                       + fc["c_dist"] * delta_d / _D_MAX)
        else:
            reward = compute_reward(
                robot, order.loc, self.layout,
                robots=self.robots,
                n_wait=self.order_mgr.queue_size(),
                num_robots=self.num_robots,
                robot_capacity=self.robot_capacity,
                **self.reward_cfg,
            )
        self.total_reward += reward

        self.agent.apply_action(action, order.loc, self.robots, self.layout)
        self._robot_orders[robot.id].append(order)
        self._record_service_start(order, self.now)

        if robot.is_idle:
            robot.is_idle = False
            robot._start_move()
            self._schedule_next(robot)
        else:
            self._mid_trip_count += 1

        if self.eval_mode:
            return None

        # 이전 결정의 transition을 지금(다음 결정 시점) 상태로 완성하고,
        # 이번 결정은 pending으로 보류한다.
        completed = None
        if self._pending is not None:
            s_prev, a_prev, r_prev = self._pending
            completed = (s_prev, a_prev, r_prev, state, False)
        self._pending = (state, action, reward)
        return completed

    def _handle_trip_complete(self, ev: Event):
        robot = self.robots[ev.data["robot_id"]]
        self._complete_trip(robot)
        orders = self.order_mgr.dequeue_up_to(self.robot_capacity)
        if orders:
            self._dispatch(robot, orders)


# ─────────────────────────────────────────────────────────────────────────────
# BL-1: Full-Buffer
# ─────────────────────────────────────────────────────────────────────────────

class BaselineSimulator(SimulatorBase):
    """K개 슬롯을 모두 채운 후 출발. mid-trip 삽입 없음."""

    def reset(self, seed: int):
        self._reset_common(seed)

    def run(self):
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._handle_bl(ev)
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
                    self._robot_orders[robot.id] = list(orders)
                    robot.idle_slots = self.robot_capacity - len(orders)
                    if robot.idle_slots == 0:
                        robot.assign_trip([o.loc for o in orders], self.layout)
                        robot.is_idle = False
                        self._schedule_next(robot)

    def _handle_bl(self, ev: Event):
        order = self.order_mgr.create_order(self.now)
        idle  = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        if idle:
            robot = max(idle, key=lambda r: r.idle_slots)
            self._robot_orders[robot.id].append(order)
            self._record_service_start(order, self.now)
            robot.idle_slots -= 1
            if robot.idle_slots == 0:
                robot.assign_trip([o.loc for o in self._robot_orders[robot.id]],
                                  self.layout)
                robot.is_idle = False
                self._schedule_next(robot)
        else:
            self.order_mgr.enqueue(order, self.now)


# ─────────────────────────────────────────────────────────────────────────────
# BL-2: Partial-Buffer
# ─────────────────────────────────────────────────────────────────────────────

class PartialSimulator(SimulatorBase):
    """k개 슬롯만 채워도 즉시 출발. mid-trip 삽입 없음."""

    def __init__(self, *args, partial_k: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_k = partial_k

    def reset(self, seed: int):
        self._reset_common(seed)

    def run(self):
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._handle_partial(ev)
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
                    self._robot_orders[robot.id] = list(orders)
                    robot.idle_slots = self.robot_capacity - len(orders)
                    if len(orders) >= self.partial_k:
                        robot.assign_trip([o.loc for o in orders], self.layout)
                        robot.is_idle = False
                        self._schedule_next(robot)

    def _handle_partial(self, ev: Event):
        order = self.order_mgr.create_order(self.now)
        idle  = [r for r in self.robots if r.is_idle and r.idle_slots > 0]
        if idle:
            robot = max(idle, key=lambda r: r.idle_slots)
            self._robot_orders[robot.id].append(order)
            self._record_service_start(order, self.now)
            robot.idle_slots -= 1
            if self.robot_capacity - robot.idle_slots >= self.partial_k:
                robot.assign_trip([o.loc for o in self._robot_orders[robot.id]],
                                  self.layout)
                robot.is_idle = False
                self._schedule_next(robot)
        else:
            self.order_mgr.enqueue(order, self.now)


# ─────────────────────────────────────────────────────────────────────────────
# BL-4: Greedy Nearest-Robot
# ─────────────────────────────────────────────────────────────────────────────

class NearestRobotSimulator(SimulatorBase):
    """주문 위치까지 현재 맨해튼 거리가 가장 짧은 로봇에 배정. mid-trip 삽입 허용.
    
    RL과의 비교 의도: RL은 현재 거리가 멀더라도 전체 flow time이 더 낮은 로봇을
    고르는 능력을 학습하는 반면, 이 휴리스틱은 순간적으로 가장 가까운 로봇에만
    배정하므로 전역 최적화 없이 국소 판단만 한다.
    """

    def reset(self, seed: int):
        self._reset_common(seed)

    def run(self):
        while self.eq:
            ev = self.eq.pop()
            if ev.time > self.sim_time:
                break
            self.now = ev.time
            if ev.etype == EVENT_ORDER_ARRIVAL:
                self._handle_nearest(ev)
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

    def _handle_nearest(self, ev: Event):
        order    = self.order_mgr.create_order(self.now)
        eligible = [r for r in self.robots if r.idle_slots > 0]
        if not eligible:
            self.order_mgr.enqueue(order, self.now)
            return
        robot = min(eligible,
                    key=lambda r: self.layout.aisle_distance(r.pos, order.loc))
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
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

# M별 최적 훈련 λ (ρ 임계점 분석 기반)
# M별 최적 훈련 λ (워크스테이션 중앙 이동 후 ρ≈0.75 기준, 로봇 처리량≈120/hr)
_TRAIN_LAM: Dict[int, float] = {
    3: 300 / 3600,
    4: 400 / 3600,
    5: 400 / 3600,
    6: 500 / 3600,
    7: 600 / 3600,
}

_CURRICULUM_LAM_RANGE: Dict[int, Tuple[float, float]] = {
    3: (200 / 3600, 400 / 3600),
    4: (200 / 3600, 500 / 3600),
    5: (300 / 3600, 600 / 3600),
    6: (400 / 3600, 700 / 3600),
    7: (400 / 3600, 700 / 3600),
}


def train(layout:         WarehouseLayout,
          agent:          DQNAgent,
          store:          MetricsStore,
          lam:            float = LAMBDA_DEFAULT,
          seed:           int   = 42,
          total_episodes: int   = 1000,
          num_robots:     int   = 4,
          robot_capacity: int   = 5,
          reward_cfg:     Dict  = None,
          label:          str   = "RL",
          verbose_every:  int   = 50,
          curve_store:    Optional["TrainingCurveStore"] = None,
          arrival:        str   = "stationary",
          burst_cfg:      Dict  = None,
          reward_mode:    str   = "detour",
          flow_cfg:       Dict  = None) -> None:

    reward_cfg = reward_cfg or {}
    lam_lo, lam_hi = _CURRICULUM_LAM_RANGE.get(num_robots, (lam * 0.6, lam))

    rl_sim = RLSimulator(layout, agent, lam=lam, seed=seed,
                         num_robots=num_robots, robot_capacity=robot_capacity,
                         reward_cfg=reward_cfg,
                         arrival=arrival, burst_cfg=burst_cfg,
                         reward_mode=reward_mode, flow_cfg=flow_cfg)
    rng = random.Random(seed)

    for ep in range(1, total_episodes + 1):
        ep_seed = rng.randint(0, 2**31 - 1)
        ep_lam  = (random.uniform(lam_lo, lam_hi)
                   if ep <= total_episodes // 2 else lam)

        rl_sim.lam             = ep_lam
        rl_sim.order_mgr.lam   = ep_lam
        rl_sim.reset(ep_seed)
        transitions = rl_sim.run()

        for trans in transitions:
            agent.push(*trans)
        loss = agent.train_step()
        agent.decay_epsilon(total_episodes)

        result = rl_sim.build_result(label, ep, ep_lam,
                                     reward=rl_sim.total_reward,
                                     epsilon=agent.epsilon)
        store.add(result)

        if curve_store is not None:
            curve_store.add(TrainingPoint(
                label=label,
                episode=ep,
                reward=rl_sim.total_reward,
                loss=loss if loss is not None else float("nan"),
                epsilon=agent.epsilon,
                n_completed=result.n_completed,
                avg_flow_time=result.avg_flow_time,
            ))

        if ep % verbose_every == 0:
            loss_s = f"{loss:.4f}" if loss is not None else "warmup"
            print(f"  [Ep {ep:4d}] {label}  M={num_robots}  "
                  f"λ={int(ep_lam*3600):3d}/hr  "
                  f"completed={result.n_completed:5d}  "
                  f"flow={result.avg_flow_time:.0f}s  "
                  f"util={result.robot_utilization:.2%}  "
                  f"save/ins={result.saving_per_insertion:.1f}m  "
                  f"ε={agent.epsilon:.3f}  loss={loss_s}")


# ─────────────────────────────────────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(layout:         WarehouseLayout,
                 agent:          DQNAgent,
                 store:          MetricsStore,
                 lam:            float,
                 num_robots:     int = 4,
                 robot_capacity: int = 5,
                 eval_seeds:     List[int] = None,
                 n_eval:         int = 20,
                 partial_k:      int = 3) -> None:

    if eval_seeds is None:
        eval_seeds = list(range(10000, 10000 + n_eval))

    for i, s in enumerate(eval_seeds):

        sim = RLSimulator(layout, agent, lam=lam, seed=s,
                          num_robots=num_robots, robot_capacity=robot_capacity,
                          eval_mode=True)
        sim.reset(s)
        sim.run()
        store.add(sim.build_result("RL", i, lam,
                                   reward=sim.total_reward,
                                   epsilon=0.0))

        sim = BaselineSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                num_robots=num_robots, robot_capacity=robot_capacity)
        sim.reset(s)
        sim.run()
        store.add(sim.build_result("BL1_FullBuffer", i, lam))

        sim = PartialSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                               num_robots=num_robots, robot_capacity=robot_capacity,
                               partial_k=partial_k)
        sim.reset(s)
        sim.run()
        store.add(sim.build_result(f"BL2_Partial_k{partial_k}", i, lam))

        sim = NearestRobotSimulator(layout, lam=lam, seed=s, sim_time=SIM_TIME,
                                    num_robots=num_robots, robot_capacity=robot_capacity)
        sim.reset(s)
        sim.run()
        store.add(sim.build_result("BL4_Nearest", i, lam))


# ─────────────────────────────────────────────────────────────────────────────
# 실험 1: M × λ 격자
# ─────────────────────────────────────────────────────────────────────────────

def experiment_robot_counts(
    layout:         WarehouseLayout,
    store:          MetricsStore,
    robot_counts:   List[int]   = [3, 4, 5, 6, 7],
    lambdas:        List[float] = None,
    seed:           int         = 42,
    total_episodes: int         = 1000,
    n_eval:         int         = 20,
    robot_capacity: int         = 5,
    curve_store:    Optional["TrainingCurveStore"] = None,
) -> Dict[int, DQNAgent]:
    if lambdas is None:
        lambdas = [300/3600, 400/3600, 500/3600, 600/3600, 700/3600]

    labels      = ["RL", "BL1_FullBuffer", "BL2_Partial_k3", "BL4_Nearest"]
    eval_seeds  = list(range(10000, 10000 + n_eval))
    agents: Dict[int, DQNAgent] = {}

    for M in robot_counts:
        print(f"\n{'='*60}")
        print(f"M = {M}  (K={robot_capacity}  λ_train={int(_TRAIN_LAM[M]*3600)}/hr)")
        print(f"{'='*60}")

        agent = DQNAgent(num_robots=M, robot_capacity=robot_capacity,
                         seed=seed, sim_time=SIM_TIME)
        tr_store = MetricsStore()
        train(layout, agent, tr_store,
              lam=_TRAIN_LAM[M], seed=seed,
              total_episodes=total_episodes,
              num_robots=M, robot_capacity=robot_capacity,
              label=f"RL_M{M}_train",
              curve_store=curve_store)
        agents[M] = agent

        for lam in lambdas:
            rho = lam / (M * (120.0 / 3600))
            print(f"\n  λ={int(lam*3600)}/hr  ρ≈{rho:.2f}")
            ev_store = MetricsStore()
            evaluate_all(layout, agent, ev_store, lam=lam,
                         num_robots=M, robot_capacity=robot_capacity,
                         eval_seeds=eval_seeds, n_eval=n_eval)
            for r in ev_store.records:
                store.add(r)
            print_comparison(ev_store, labels, lam=lam,
                             num_robots=M, last_n=n_eval,
                             title=f"M={M}  λ={int(lam*3600)}/hr")

    return agents


# ─────────────────────────────────────────────────────────────────────────────
# 실험 2: γ 민감도 분석 (어블레이션 재설계)
#
# 이전 어블레이션 결론:
#   - T_detour: 유일한 핵심 항 → 고정
#   - T_wait / B_cluster: 제거 (기여 없음)
#   - queue_pressure(γ): γ=1.0이 과다 패널티 → 민감도 탐색
#
# 새 어블레이션: γ ∈ {0.0, 0.1, 0.3, 0.5} 에 대해 동일 에이전트 구조로 훈련·평가
# ─────────────────────────────────────────────────────────────────────────────

def experiment_gamma_sensitivity(
    layout:         WarehouseLayout,
    store:          MetricsStore,
    lam:            float,
    num_robots:     int   = 3,
    robot_capacity: int   = 5,
    seed:           int   = 42,
    n_eval:         int   = 20,
    total_episodes: int   = 500,
    curve_store:    Optional["TrainingCurveStore"] = None,
) -> None:
    gamma_vals = [0.0, 0.1, 0.3, 0.5]
    eval_seeds = list(range(20000, 20000 + n_eval))

    for gq in gamma_vals:
        label = f"RL_gq{str(gq).replace('.','')}"
        print(f"\n── γ_q = {gq} ──")
        reward_cfg = dict(gamma_q=gq)
        ag = DQNAgent(num_robots=num_robots, robot_capacity=robot_capacity,
                      seed=seed, sim_time=SIM_TIME)
        tr_store = MetricsStore()
        train(layout, ag, tr_store, lam=lam, seed=seed,
              total_episodes=total_episodes,
              num_robots=num_robots, robot_capacity=robot_capacity,
              reward_cfg=reward_cfg, label=label,
              curve_store=curve_store)

        for i, s in enumerate(eval_seeds):
            sim = RLSimulator(layout, ag, lam=lam, seed=s,
                              num_robots=num_robots, robot_capacity=robot_capacity,
                              reward_cfg=reward_cfg, eval_mode=True)
            sim.reset(s)
            sim.run()
            store.add(sim.build_result(f"{label}_eval", i, lam,
                                       reward=sim.total_reward,
                                       epsilon=0.0))

    eval_labels = [f"RL_gq{str(g).replace('.','')}_eval" for g in gamma_vals]
    print_comparison(store, eval_labels, lam=lam, num_robots=num_robots,
                     last_n=n_eval,
                     title=f"γ sensitivity (M={num_robots}  λ={int(lam*3600)}/hr)")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SEED           = 42
    ROBOT_CAPACITY = 5
    ROBOT_COUNTS   = [3, 4, 5, 6, 7]
    LAM_GRID       = [300/3600, 400/3600, 500/3600, 600/3600, 700/3600]
    TOTAL_EPISODES = 1000
    N_EVAL         = 20
    OUT_DIR        = "logs"
    os.makedirs(OUT_DIR, exist_ok=True)

    layout = WarehouseLayout(seed=SEED)

    print("=" * 60)
    print("Phase 1: M × λ grid experiment")
    print("=" * 60)
    store1      = MetricsStore()
    curve1      = TrainingCurveStore()
    agents = experiment_robot_counts(
        layout, store1,
        robot_counts=ROBOT_COUNTS,
        lambdas=LAM_GRID,
        seed=SEED,
        total_episodes=TOTAL_EPISODES,
        n_eval=N_EVAL,
        robot_capacity=ROBOT_CAPACITY,
        curve_store=curve1,
    )
    store1.save_csv(os.path.join(OUT_DIR, "phase1_M_lambda_grid.csv"))
    curve1.save_csv(os.path.join(OUT_DIR, "phase1_training_curves.csv"))

    print("\n" + "=" * 60)
    print("Phase 2: Queue penalty γ sensitivity (M=5, λ=500/hr)")
    print("=" * 60)
    store2 = MetricsStore()
    curve2 = TrainingCurveStore()
    experiment_gamma_sensitivity(
        layout, store2,
        lam=500 / 3600,
        num_robots=5,
        robot_capacity=ROBOT_CAPACITY,
        seed=SEED,
        n_eval=N_EVAL,
        total_episodes=500,
        curve_store=curve2,
    )
    store2.save_csv(os.path.join(OUT_DIR, "phase2_gamma_sensitivity.csv"))
    curve2.save_csv(os.path.join(OUT_DIR, "phase2_training_curves.csv"))

    print(f"\nDone. CSVs → {OUT_DIR}/")


if __name__ == "__main__":
    main()