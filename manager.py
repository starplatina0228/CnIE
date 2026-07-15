import heapq
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Order:
    order_id:        int
    sku:             int
    loc_idx:         int
    loc:             tuple
    arrival_time:    float
    completion_time: Optional[float] = None


EVENT_ORDER_ARRIVAL = "ORDER_ARRIVAL"
EVENT_NODE_ARRIVAL  = "ROBOT_NODE_ARRIVAL"
EVENT_TRIP_COMPLETE = "ROBOT_TRIP_COMPLETE"


@dataclass(order=True)
class Event:
    time:  float
    etype: str  = field(compare=False)
    data:  dict = field(compare=False, default_factory=dict)


class EventQueue:
    def __init__(self):
        self._q: list = []

    def push(self, ev: Event):
        heapq.heappush(self._q, ev)

    def pop(self) -> Event:
        return heapq.heappop(self._q)

    def __len__(self):
        return len(self._q)


class OrderManager:
    """주문 도착 프로세스 관리.

    arrival 프로파일
    ────────────────
      "stationary" : 상수 λ 의 정상 Poisson (기존 동작, 기본값).
      "sine"       : λ(t) = λ·(1 + amp·sin(2π t/period)),   평균 부하 = λ.
      "burst"      : on/off 사각파. 주기 period 안에서 duty 비율만 λ·high,
                     나머지는 λ·low. 기본값(duty=.25, high=2.5, low=.5)은
                     평균 부하를 λ 로 보존하되 순간 최대 2.5λ 의 버스트를 만든다.

    비정상 프로파일은 thinning(accept-reject, 상한 λ_max)으로 정확히 샘플링한다.
    """

    def __init__(self, layout, lam: float, seed: int = 42,
                 arrival: str = "stationary", burst_cfg: Optional[dict] = None):
        self.layout   = layout
        self.lam      = lam
        self.arrival  = arrival
        self.burst_cfg = {
            "period": 1800.0,   # s — 버스트 주기(기본 30분)
            "amp":    0.8,      # sine 진폭
            "duty":   0.25,     # burst on 비율
            "high":   2.5,      # on 배율
            "low":    0.5,      # off 배율
        }
        if burst_cfg:
            self.burst_cfg.update(burst_cfg)
        self._rng    = random.Random(seed)
        self._cnt    = 0
        self._n_completed_run = 0     # 완료 카운터(WIP 적분용, O(1) 조회)
        self.wait_queue:         List[Order] = []
        self.all_orders:         List[Order] = []
        self._queue_entry_times: List[float] = []

    def reset(self, seed: int):
        self._rng = random.Random(seed)
        self._cnt = 0
        self._n_completed_run = 0
        self.wait_queue.clear()
        self.all_orders.clear()
        self._queue_entry_times.clear()

    # ── 시각 t 의 순시 도착률 λ(t) ────────────────────────────────────────────
    def lambda_at(self, t: float) -> float:
        c = self.burst_cfg
        if self.arrival == "sine":
            return self.lam * (1.0 + c["amp"] * math.sin(2.0 * math.pi * t / c["period"]))
        if self.arrival == "burst":
            phase = (t % c["period"]) / c["period"]
            return self.lam * (c["high"] if phase < c["duty"] else c["low"])
        return self.lam   # stationary

    def _lambda_max(self) -> float:
        c = self.burst_cfg
        if self.arrival == "sine":
            return self.lam * (1.0 + c["amp"])
        if self.arrival == "burst":
            return self.lam * c["high"]
        return self.lam

    def generate_next_arrival(self, current_time: float) -> float:
        if self.arrival == "stationary":
            return current_time + self._rng.expovariate(self.lam)
        # 비정상: thinning (Lewis-Shedler)
        lam_max = self._lambda_max()
        t = current_time
        while True:
            t += self._rng.expovariate(lam_max)
            if self._rng.random() <= self.lambda_at(t) / lam_max:
                return t

    def create_order(self, arrival_time: float) -> Order:
        sku             = self.layout.sample_order_sku(self._rng)
        loc_idx, loc    = self.layout.find_tote_location(sku, self._rng)
        order           = Order(self._cnt, sku, loc_idx, loc, arrival_time)
        self._cnt      += 1
        self.all_orders.append(order)
        return order

    def enqueue(self, order: Order, enqueue_time: float):
        self.wait_queue.append(order)
        self._queue_entry_times.append(enqueue_time)

    def dequeue_up_to(self, n: int) -> List[Order]:
        k      = min(n, len(self.wait_queue))
        orders = self.wait_queue[:k]
        self.last_batch_entry_times = list(self._queue_entry_times[:k])
        self.wait_queue             = self.wait_queue[k:]
        self._queue_entry_times     = self._queue_entry_times[k:]
        return orders

    def queue_size(self) -> int:
        return len(self.wait_queue)

    def record_completion(self, order: Order, time: float):
        order.completion_time = time
        self._n_completed_run += 1

    def in_system(self) -> int:
        """현재 시스템 내 미완료 주문 수 (도착−완료). WIP 적분에 사용. O(1)."""
        return len(self.all_orders) - self._n_completed_run

    def avg_wait_time(self) -> float:
        waits = [
            o.completion_time - o.arrival_time
            for o in self.all_orders if o.completion_time is not None
        ]
        return sum(waits) / len(waits) if waits else 0.0

    def avg_flow_time(self) -> float:
        return self.avg_wait_time()   # same definition (arrival→completion)

    def completed_count(self) -> int:
        return sum(1 for o in self.all_orders if o.completion_time is not None)
