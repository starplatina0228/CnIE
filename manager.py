import heapq
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
    def __init__(self, layout, lam: float, seed: int = 42):
        self.layout  = layout
        self.lam     = lam
        self._rng    = random.Random(seed)
        self._cnt    = 0
        self.wait_queue:         List[Order] = []
        self.all_orders:         List[Order] = []
        self._queue_entry_times: List[float] = []

    def reset(self, seed: int):
        self._rng = random.Random(seed)
        self._cnt = 0
        self.wait_queue.clear()
        self.all_orders.clear()
        self._queue_entry_times.clear()

    def generate_next_arrival(self, current_time: float) -> float:
        return current_time + self._rng.expovariate(self.lam)

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
