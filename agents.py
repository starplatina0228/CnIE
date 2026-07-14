import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple, Optional

from environment import (
    WarehouseLayout, nearest_neighbor_route, route_total_distance,
    ROBOT_SPEED, SIM_TIME,
)

_COORD_X_MAX = 40.0
_COORD_Y_MAX = 36.0

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────────
GAMMA               = 0.99
LR                  = 3e-4
BATCH_SIZE          = 128
REPLAY_BUFFER_SIZE  = 50_000
TARGET_UPDATE_STEPS = 200
EPS_START           = 1.0
EPS_MIN             = 0.02
EPS_DECAY           = 0.997
GRAD_CLIP           = 10.0
N_STEP              = 3


# ─────────────────────────────────────────────────────────────────────────────
# Robot
# ─────────────────────────────────────────────────────────────────────────────

class Robot:
    def __init__(self, robot_id: int, workstation: Tuple[float, float],
                 layout: WarehouseLayout, capacity: int):
        self.id          = robot_id
        self.workstation = workstation
        self.layout      = layout
        self.capacity    = capacity
        self.reset()

    def reset(self):
        self.pos               = self.workstation
        self.route             = []
        self.idle_slots        = self.capacity
        self.is_idle           = True
        self.current_dest      = None
        self.progress          = 0.0
        self.total_travel_time = 0.0
        self._travel_start     = None

    def start_travel(self, now: float):
        if self._travel_start is None:
            self._travel_start = now

    def end_travel(self, now: float):
        if self._travel_start is not None:
            self.total_travel_time += max(0.0, now - self._travel_start)
            self._travel_start = None

    def time_to_complete_current(self) -> float:
        return self.remaining_distance / ROBOT_SPEED

    @property
    def remaining_distance(self) -> float:
        d = 0.0
        if self.current_dest:
            d += max(0.0, self._md(self.pos, self.current_dest) - self.progress)
        if self.route:
            prev = self.current_dest if self.current_dest else self.pos
            for nxt in self.route:
                d += self._md(prev, nxt)
                prev = nxt
            d += self._md(prev, self.workstation)
        elif self.current_dest:
            d += self._md(self.current_dest, self.workstation)
        return d

    def arrive_at_node(self) -> bool:
        self.pos      = self.current_dest
        self.progress = 0.0
        if self.route:
            self.current_dest = self.route.pop(0)
            return True
        self.current_dest = None
        return False

    def assign_trip(self, destinations: List[Tuple[float, float]],
                    layout: WarehouseLayout):
        self.route      = nearest_neighbor_route(self.workstation, destinations,
                                                 layout.aisle_distance)
        self.idle_slots = self.capacity - len(destinations)
        self.is_idle    = False
        self._start_move()

    def insert_and_reorder(self, loc: Tuple[float, float],
                            layout: WarehouseLayout):
        pending    = list(self.route) + [loc]
        start      = self.current_dest if self.current_dest else self.pos
        self.route = nearest_neighbor_route(start, pending, layout.aisle_distance)
        self.idle_slots = max(0, self.idle_slots - 1)

    def delta_distance_insert(self, loc: Tuple[float, float],
                               layout: WarehouseLayout) -> float:
        before    = self._remaining_route_dist(layout)
        pending   = list(self.route) + [loc]
        start     = self.current_dest if self.current_dest else self.pos
        new_route = nearest_neighbor_route(start, pending, layout.aisle_distance)
        after     = route_total_distance(start, new_route, self.workstation,
                                         layout.aisle_distance)
        return after - before

    def state_vector(self, max_cap: int, max_time: float = 300.0) -> np.ndarray:
        rem_t_norm = min(self.time_to_complete_current() / max_time, 1.0)
        vec = [
            self.pos[0] / _COORD_X_MAX,
            self.pos[1] / _COORD_Y_MAX,
            float(self.idle_slots) / max_cap,
            float(self.is_idle),
            rem_t_norm,
        ]
        route_flat = []
        for p in self.route[:max_cap]:
            route_flat.extend([p[0] / _COORD_X_MAX, p[1] / _COORD_Y_MAX])
        while len(route_flat) < max_cap * 2:
            route_flat.extend([0.0, 0.0])
        vec.extend(route_flat)
        return np.array(vec, dtype=np.float32)

    def _remaining_route_dist(self, layout: WarehouseLayout) -> float:
        start = self.current_dest if self.current_dest else self.pos
        return route_total_distance(start, self.route, self.workstation,
                                    layout.aisle_distance)

    def _start_move(self):
        if self.route:
            self.current_dest = self.route.pop(0)

    def _md(self, p1, p2):
        return self.layout.aisle_distance(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# 상태 차원
# ─────────────────────────────────────────────────────────────────────────────

def build_state_dim(num_robots: int, robot_capacity: int) -> int:
    """
    order_loc     : 2
    per_robot     : 5 + robot_capacity*2  (pos, idle_norm, is_idle, rem_t, route)
    global_feats  : 2  (queue_len_norm, elapsed_norm)
    """
    return 2 + num_robots * (5 + robot_capacity * 2) + 2


# ─────────────────────────────────────────────────────────────────────────────
# D3QN (Dueling + Double DQN)
# ─────────────────────────────────────────────────────────────────────────────

class D3QNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),                          nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        return V + (A - A.mean(dim=-1, keepdim=True))


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Experience Replay
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """α=0.6, β=0.4→1.0 annealing."""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, alpha: float = 0.6):
        self.capacity      = capacity
        self.alpha         = alpha
        self.buf           = [None] * capacity
        self.priorities    = np.zeros(capacity, dtype=np.float32)
        self.pos           = 0
        self.size          = 0
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.buf[self.pos]        = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self._max_priority
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        prios   = self.priorities[:self.size] ** self.alpha
        probs   = prios / prios.sum()
        idxs    = np.random.choice(self.size, batch_size, replace=False, p=probs)
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()
        batch   = [self.buf[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
            idxs,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        prios = np.abs(td_errors) + 1e-6
        self.priorities[idxs] = prios
        self._max_priority = max(self._max_priority, prios.max())

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# N-step return buffer
# ─────────────────────────────────────────────────────────────────────────────

class NStepBuffer:
    def __init__(self, n: int = N_STEP, gamma: float = GAMMA):
        self.n     = n
        self.gamma = gamma
        self.buf   = deque()

    def push(self, transition):
        self.buf.append(transition)

    def ready(self) -> bool:
        return len(self.buf) >= self.n

    def get(self):
        s, a, _, _, _ = self.buf[0]
        _, _, _, ns, d = self.buf[-1]
        G = 0.0
        for i, (_, _, r_i, _, d_i) in enumerate(self.buf):
            G += (self.gamma ** i) * r_i
            if d_i:
                break
        self.buf.popleft()
        return (s, a, G, ns, d)

    def flush(self):
        results = []
        while self.buf:
            s, a, _, _, d = self.buf[0]
            ns = self.buf[-1][3]
            G  = sum((self.gamma ** i) * t[2] for i, t in enumerate(self.buf))
            self.buf.popleft()
            results.append((s, a, G, ns, d))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# DQNAgent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, num_robots: int, robot_capacity: int,
                 seed: int = 42, sim_time: float = SIM_TIME):
        self.num_robots     = num_robots
        self.robot_capacity = robot_capacity
        self.state_dim      = build_state_dim(num_robots, robot_capacity)
        self.n_actions      = num_robots
        self.sim_time       = sim_time

        self.device     = torch.device("cpu")
        self.policy_net = D3QNNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_net = D3QNNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer    = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)
        self.nstep_buf = NStepBuffer(n=N_STEP, gamma=GAMMA)

        self.epsilon  = EPS_START
        self.beta     = 0.4
        self.beta_end = 1.0
        self.steps    = 0
        self.rng      = random.Random(seed)

    # ── 상태 인코딩 ───────────────────────────────────────────────────────────
    def encode_state(self, order_loc: Tuple[float, float],
                     robots: List[Robot], n_wait: int,
                     now: float = 0.0) -> np.ndarray:
        robot_feat_dim = 5 + self.robot_capacity * 2
        vec = [order_loc[0] / _COORD_X_MAX, order_loc[1] / _COORD_Y_MAX]
        for r in robots:
            vec.extend(r.state_vector(self.robot_capacity).tolist())
        for _ in range(self.num_robots - len(robots)):
            vec.extend([0.0] * robot_feat_dim)
        queue_len_norm = min(n_wait / 20.0, 1.0)
        elapsed_norm   = min(now / self.sim_time, 1.0)
        vec.extend([queue_len_norm, elapsed_norm])
        assert len(vec) == self.state_dim, f"state dim mismatch: {len(vec)} != {self.state_dim}"
        return np.array(vec, dtype=np.float32)

    # ── 액션 마스크 ───────────────────────────────────────────────────────────
    def build_action_mask(self, robots: List[Robot],
                          moving_only: bool = False) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        for r in robots:
            if r.idle_slots > 0:
                if moving_only and r.is_idle:
                    continue
                mask[r.id] = True
        return mask

    # ── 액션 선택 ─────────────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return -1
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))
        with torch.no_grad():
            q    = self.policy_net(torch.tensor(state).unsqueeze(0)).squeeze(0)
            q_np = q.numpy().copy()
            q_np[~mask] = -np.inf
            return int(np.argmax(q_np))

    def apply_action(self, action: int, order_loc: Tuple[float, float],
                     robots: List[Robot], layout: WarehouseLayout) -> bool:
        if action < 0 or action >= len(robots):
            return False
        robot = robots[action]
        if robot.idle_slots <= 0:
            return False
        if robot.is_idle:
            robot.route.append(order_loc)
            robot.idle_slots = max(0, robot.idle_slots - 1)
        else:
            robot.insert_and_reorder(order_loc, layout)
        return True

    # ── 학습 ──────────────────────────────────────────────────────────────────
    def push(self, state, action, reward, next_state, done):
        self.nstep_buf.push((state, action, reward, next_state, done))
        if self.nstep_buf.ready():
            self.buffer.push(*self.nstep_buf.get())
        if done:
            for t in self.nstep_buf.flush():
                self.buffer.push(*t)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < BATCH_SIZE:
            return None

        s, a, r, ns, d, idxs, weights = self.buffer.sample(BATCH_SIZE, beta=self.beta)
        s_t = torch.tensor(s).to(self.device)
        a_t = torch.tensor(a).unsqueeze(1).to(self.device)
        r_t = torch.tensor(r).to(self.device)
        ns_t = torch.tensor(ns).to(self.device)
        d_t  = torch.tensor(d).to(self.device)
        w_t  = torch.tensor(weights).to(self.device)

        q_vals = self.policy_net(s_t).gather(1, a_t).squeeze(1)
        with torch.no_grad():
            best_a  = self.policy_net(ns_t).argmax(dim=1, keepdim=True)
            next_q  = self.target_net(ns_t).gather(1, best_a).squeeze(1)
            gamma_n = GAMMA ** N_STEP
            target  = r_t + gamma_n * next_q * (1 - d_t)

        td_errors = (q_vals - target).detach().cpu().numpy()
        self.buffer.update_priorities(idxs, td_errors)

        loss = (w_t * (q_vals - target).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_STEPS == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def decay_epsilon(self, total_episodes: int = 1000):
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)
        self.beta    = min(self.beta_end,
                           self.beta + (self.beta_end - 0.4) / total_episodes)