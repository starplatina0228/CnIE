"""
viz.py — 창고 강화학습 구조 시각화

두 가지를 제공한다.
  1. 창고 레이아웃 정적 맵           →  plot_layout_map(...)
  2. 시뮬레이션 애니메이션 (로봇 이동) →  animate_simulation(...)

애니메이션은 이벤트 기반 시뮬레이터를 그대로 돌리면서 로봇의
NODE_ARRIVAL 이벤트만 가로채 "이동 구간(segment)"을 복원한다.
  · 모든 이동은 NODE_ARRIVAL 이벤트로 도착하고, 이벤트는 취소되지 않는다.
  · robot.progress 는 0 으로만 리셋되므로 부분 구간이 없다.
  · 따라서  t_start = t_arrival − manhattan(prev, dest)/speed  가 정확하다.
    (거리로부터 출발시각을 역산하므로 워크스테이션 대기 구간도 자동 처리)

모델과의 정합:
  · environment.aisle_distance 가 통로 그래프 거리(통로 전용 이동 + tote 앞
    노드 픽업)를 계산한다. 따라서 이 애니메이션의 통로 라우팅은 근사가 아니라
    모델과 일치하는 표현이다.
  · 도착 시각(t0,t1)은 시뮬레이터의 실제 이벤트 시각에 고정된다
    (t0 = t1 − aisle_distance(prev,dest)/speed).

CLI:
    python viz.py map                 # 레이아웃 맵을 layout_map.png 로 저장
    python viz.py anim                # 애니메이션을 sim_animation.gif 로 저장
    python viz.py anim --train 150    # 150 에피소드 quick-train 후 애니메이션
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 파일 저장 백엔드 (창 없이도 동작)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation, PillowWriter

from environment import (
    WarehouseLayout,
    BN, AN, SN, Wa, Wc, Wt, Dt,
    ROBOT_SPEED, SIM_TIME,
)
from agents import DQNAgent
from run import RLSimulator, NearestRobotSimulator, SimulatorBase
from metrics import MetricsStore

# ── 색상 팔레트 ──────────────────────────────────────────────────────────────
_C_BG      = "#f4f4f2"
_C_BLOCK   = "#e6ecef"
_C_RACK    = "#9fb0b8"
_C_RACK_ED = "#ffffff"
_C_AISLE   = "#ffffff"
_C_WS      = "#e63946"
_C_ORDER   = "#f4a261"
_C_ORDER_E = "#c26a2c"
_C_LANE    = "#eef2f4"
_ROBOT_CMAP = plt.get_cmap("tab10")


# ── 통로 지오메트리 (시각화용 라우팅) ────────────────────────────────────────
_AISLE_STEP = Wa + 2 * Dt                       # 4.0 m (통로 피치)
_AISLE_CENTERS = [a * _AISLE_STEP + Dt + Wa / 2 for a in range(AN)]  # {2,6,...,38}
_BLOCK_H = Wc + SN * Wt                          # 16.0
# 교차통로(가로 이동 레인) y: 하단, 블록 사이 중간
_LANES = [Wc / 2, _BLOCK_H + Wc / 2]             # {2.0, 20.0}


def _aisle_center(x: float) -> float:
    """x 에 가장 가까운 통로 중심선."""
    cx = round((x - (Dt + Wa / 2)) / _AISLE_STEP) * _AISLE_STEP + (Dt + Wa / 2)
    return min(max(cx, _AISLE_CENTERS[0]), _AISLE_CENTERS[-1])


# ═════════════════════════════════════════════════════════════════════════════
# 1. 정적 레이아웃 맵
# ═════════════════════════════════════════════════════════════════════════════

def _rack_rectangles(layout: WarehouseLayout) -> List[Rectangle]:
    """각 저장 위치를 tote 셀(폭 Dt × 높이 Wt)로."""
    return [Rectangle((cx - Dt / 2, cy - Wt / 2), Dt, Wt)
            for (cx, cy) in layout.storage_coords]


def draw_warehouse(layout: WarehouseLayout, ax: plt.Axes,
                   sku_color: bool = False, show_lanes: bool = False) -> None:
    """정적 창고 구조를 축(ax)에 그린다. 배경으로 재사용 가능."""
    ax.set_facecolor(_C_BG)
    W = layout.total_width

    # 교차통로(가로 레인) + 세로 통로 코리도를 옅은 색으로 (grid 대비)
    for b in range(BN):
        y0 = b * (_BLOCK_H + Wc) + Wc
        # 하부/블록사이 교차통로
        ax.add_patch(Rectangle((0, y0 - Wc), W, Wc, facecolor=_C_LANE,
                               edgecolor="none", zorder=0))
    # 세로 통로(aisle corridor) — 흰색으로 명확히
    for cx in _AISLE_CENTERS:
        ax.add_patch(Rectangle((cx - Wa / 2, 0), Wa, layout.total_height,
                               facecolor=_C_AISLE, edgecolor="none", zorder=0))

    # 저장 tote 셀 — 흰 테두리로 grid 가 보이게
    rects = _rack_rectangles(layout)
    if sku_color:
        skus = np.array([layout.loc_to_sku[i] for i in range(len(rects))])
        pc = PatchCollection(rects, cmap="turbo", zorder=1)
        pc.set_array(skus.astype(float))
        pc.set_edgecolor("white")
        pc.set_linewidth(0.25)
    else:
        pc = PatchCollection(rects, facecolor=_C_RACK, edgecolor=_C_RACK_ED,
                             linewidth=0.35, zorder=1)
    ax.add_collection(pc)

    # 통로 라우팅 레인(선택) — 애니메이션 이해용
    if show_lanes:
        for ly in _LANES:
            ax.plot([0, W], [ly, ly], color="#b9c6cc", lw=0.8,
                    ls=(0, (4, 3)), zorder=2)
        for cx in _AISLE_CENTERS:
            ax.plot([cx, cx], [0, layout.total_height], color="#cdd8dd",
                    lw=0.6, ls=(0, (4, 3)), zorder=2)

    # 워크스테이션 (우측 하단)
    wx, wy = layout.workstation
    ax.add_patch(FancyBboxPatch((wx - 2.2, wy - 0.9), 2.2, 1.8,
                                boxstyle="round,pad=0.1,rounding_size=0.3",
                                facecolor=_C_WS, edgecolor="#7a1420",
                                linewidth=1.2, zorder=4))
    ax.text(wx - 1.1, wy, "WS", ha="center", va="center",
            color="white", fontsize=8, fontweight="bold", zorder=5)

    ax.set_xlim(-1, W + 1)
    ax.set_ylim(-2.5, layout.total_height + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# ── 통로 라우팅 (시각화 전용) ────────────────────────────────────────────────

def _corridor_waypoints(p0: Tuple[float, float], p1: Tuple[float, float],
                        ws: Tuple[float, float]) -> List[Tuple[float, float]]:
    """p0→p1 을 통로(세로 aisle) + 교차통로(가로 lane)를 따르는 폴리라인으로.

    tote 는 통로 중심선으로 빠져나온 뒤 이동하고, 목적 tote 면으로 진입한다.
    거리 모델(맨해튼)과 별개인 '보기용' 경로다.
    """
    def is_tote(p):
        return p != ws and abs(p[0] - _aisle_center(p[0])) > 0.5

    def cx_of(p):
        return p[0] if p == ws else _aisle_center(p[0])

    cx0, cx1 = cx_of(p0), cx_of(p1)
    pts: List[Tuple[float, float]] = [p0]

    if is_tote(p0):                       # tote → 통로 면으로 빠져나옴
        pts.append((cx0, p0[1]))

    if cx0 != cx1 or p0 == ws or p1 == ws:
        # 같은 통로가 아니면 교차통로 경유
        lane = min(_LANES, key=lambda l: abs(pts[-1][1] - l) + abs(p1[1] - l))
        pts.append((cx0, lane))
        pts.append((cx1, lane))

    pts.append((cx1, p1[1]))              # 목적 통로에서 목표 y 로
    if is_tote(p1):                       # 통로 → tote 면으로 진입
        pts.append((p1[0], p1[1]))

    # 연속 중복점 제거
    out = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - out[-1][0]) > 1e-9 or abs(q[1] - out[-1][1]) > 1e-9:
            out.append(q)
    return out


def _polyline_len(pts: List[Tuple[float, float]]) -> Tuple[float, List[float]]:
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + abs(b[0] - a[0]) + abs(b[1] - a[1]))
    return cum[-1], cum


def _interp_polyline(pts, cum, total, f):
    """폴리라인을 총길이 비율 f(0~1) 지점에서 보간."""
    if total <= 1e-9 or len(pts) == 1:
        return pts[-1]
    target = f * total
    for i in range(1, len(pts)):
        if cum[i] >= target - 1e-12:
            seg = cum[i] - cum[i - 1]
            g = 0.0 if seg <= 1e-12 else (target - cum[i - 1]) / seg
            a, b = pts[i - 1], pts[i]
            return (a[0] + (b[0] - a[0]) * g, a[1] + (b[1] - a[1]) * g)
    return pts[-1]


def plot_layout_map(layout: Optional[WarehouseLayout] = None,
                    save_path: str = "layout_map.png",
                    show: bool = False) -> str:
    """구조 맵 + SKU 분포 맵 2패널을 저장."""
    layout = layout or WarehouseLayout(seed=42)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    draw_warehouse(layout, axes[0], sku_color=False)
    axes[0].set_title(
        f"Warehouse Layout  ·  {BN} blocks × {AN} aisles × {SN}×2 shelves"
        f"  =  {layout.num_locations} locations",
        fontsize=12, fontweight="bold")

    draw_warehouse(layout, axes[1], sku_color=True)
    axes[1].set_title(
        f"SKU Distribution  ·  {layout.num_skus} SKUs (color = SKU id)",
        fontsize=12, fontweight="bold")

    fig.suptitle("Robotic Warehouse — Environment Structure",
                 fontsize=15, fontweight="bold")
    fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# 2. 이동 구간 기록 (Mixin)
# ═════════════════════════════════════════════════════════════════════════════

class RecordingMixin:
    """SimulatorBase 계열에 얹어 로봇 이동 구간을 기록한다.

    사용:  class RecRL(RecordingMixin, RLSimulator): pass
    NODE_ARRIVAL 을 가로채 (robot_id, t0, t1, p0, p1) 구간을 남긴다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segments: List[Tuple[int, float, float,
                                  Tuple[float, float], Tuple[float, float]]] = []
        self._last_pos: Dict[int, Tuple[float, float]] = {
            i: self.layout.workstation for i in range(self.num_robots)
        }

    def _handle_node_arrival(self, ev):
        r_id = ev.data["robot_id"]
        dest = ev.data["dest"]
        prev = self._last_pos[r_id]          # super() 가 robot.pos 를 바꾸기 전에 읽는다
        dist = self.layout.aisle_distance(prev, dest)
        if dist > 1e-9:
            t1 = self.now
            t0 = t1 - dist / ROBOT_SPEED
            self.segments.append((r_id, t0, t1, prev, dest))
        self._last_pos[r_id] = dest
        super()._handle_node_arrival(ev)


class RecordingRL(RecordingMixin, RLSimulator):
    pass


class RecordingNearest(RecordingMixin, NearestRobotSimulator):
    pass


# ── 프레임 위치 계산 (통로 라우팅) ──────────────────────────────────────────

def _segments_by_robot(segments, num_robots, ws):
    """로봇별 구간 목록. 각 구간에 통로 폴리라인을 미리 계산해 캐싱."""
    d: Dict[int, list] = {i: [] for i in range(num_robots)}
    for (r_id, t0, t1, p0, p1) in segments:
        pts        = _corridor_waypoints(p0, p1, ws)
        total, cum = _polyline_len(pts)
        d[r_id].append({"t0": t0, "t1": t1, "p1": p1,
                        "pts": pts, "cum": cum, "total": total})
    for i in d:
        d[i].sort(key=lambda s: s["t0"])
    return d


def _pos_at(robot_segs, t, workstation):
    """시각 t 에서 로봇 위치를 통로 폴리라인 위에서 보간. (위치, 이동중여부, 남은경로)."""
    active = None
    for seg in robot_segs:
        if seg["t0"] <= t <= seg["t1"]:
            active = seg
            break
    if active is None:                        # 이동 중 아님 → 직전 도착점(없으면 WS)
        last = workstation
        for seg in robot_segs:
            if seg["t1"] <= t:
                last = seg["p1"]
            else:
                break
        return last, False, None

    t0, t1 = active["t0"], active["t1"]
    f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    pos = _interp_polyline(active["pts"], active["cum"], active["total"], f)

    # 남은 경로(현재 위치 → 이후 waypoint들) — 목표선 그리기용
    target = f * active["total"]
    remain = [pos]
    for i in range(1, len(active["pts"])):
        if active["cum"][i] > target + 1e-9:
            remain.append(active["pts"][i])
    return pos, True, remain


# ═════════════════════════════════════════════════════════════════════════════
# 3. 시뮬레이션 실행 + 애니메이션
# ═════════════════════════════════════════════════════════════════════════════

def quick_train(agent: DQNAgent, layout: WarehouseLayout,
                num_robots: int, robot_capacity: int,
                lam: float, episodes: int = 150, seed: int = 0,
                ep_horizon: float = 1500.0) -> None:
    """짧은 학습(정책이 랜덤에서 조금 벗어나도록).

    데모용이므로 에피소드당 시뮬레이션 길이를 ep_horizon(기본 25분)으로
    짧게 잡는다. 실제 학습은 run.py 의 train() 을 쓸 것.
    """
    import random
    from run import RLSimulator as _RL
    rng = random.Random(seed)
    sim = _RL(layout, agent, lam=lam, seed=seed, sim_time=ep_horizon,
              num_robots=num_robots, robot_capacity=robot_capacity)
    for ep in range(episodes):
        sim.reset(rng.randint(0, 2**31 - 1))
        for tr in sim.run():
            agent.push(*tr)
        agent.train_step()
        agent.decay_epsilon(episodes)


def run_and_record(layout: WarehouseLayout,
                   policy: str = "rl",
                   agent: Optional[DQNAgent] = None,
                   num_robots: int = 5,
                   robot_capacity: int = 5,
                   lam: float = 400 / 3600,
                   horizon: float = 500.0,
                   seed: int = 123):
    """시뮬레이터를 horizon 초까지 돌리고 기록 데이터를 반환."""
    if policy == "rl":
        if agent is None:
            agent = DQNAgent(num_robots=num_robots, robot_capacity=robot_capacity,
                             seed=seed, sim_time=SIM_TIME)
        sim = RecordingRL(layout, agent, lam=lam, seed=seed, sim_time=horizon,
                          num_robots=num_robots, robot_capacity=robot_capacity,
                          eval_mode=True)
    elif policy == "nearest":
        sim = RecordingNearest(layout, lam=lam, seed=seed, sim_time=horizon,
                               num_robots=num_robots, robot_capacity=robot_capacity)
    else:
        raise ValueError(f"unknown policy: {policy}")

    sim.reset(seed)
    sim.run()

    orders = [
        (o.arrival_time, o.completion_time, o.loc)
        for o in sim.order_mgr.all_orders
        if o.arrival_time <= horizon
    ]
    return {
        "segments":    sim.segments,
        "orders":      orders,
        "num_robots":  num_robots,
        "capacity":    robot_capacity,
        "horizon":     horizon,
        "lam":         lam,
        "policy":      policy,
        "workstation": layout.workstation,
    }


def animate_simulation(layout: Optional[WarehouseLayout] = None,
                       record: Optional[dict] = None,
                       save_path: str = "sim_animation.gif",
                       fps: int = 20,
                       playback_speed: float = 12.0,
                       show: bool = False,
                       **run_kwargs) -> str:
    """시뮬레이션 애니메이션을 GIF 로 저장.

    playback_speed : 실시간 1초당 진행할 시뮬레이션 초 (배속)
    """
    layout = layout or WarehouseLayout(seed=42)
    if record is None:
        record = run_and_record(layout, **run_kwargs)

    segs        = record["segments"]
    orders      = record["orders"]
    num_robots  = record["num_robots"]
    capacity    = record["capacity"]
    horizon     = record["horizon"]
    ws          = record["workstation"]
    by_robot    = _segments_by_robot(segs, num_robots, ws)

    n_frames = max(1, int(horizon / playback_speed * fps))
    times    = np.linspace(0.0, horizon, n_frames)

    # ── 배경(정적 레이아웃)은 한 번만 그린다 ──
    fig, ax = plt.subplots(figsize=(13, 12))
    draw_warehouse(layout, ax, sku_color=False, show_lanes=True)
    ax.set_title("Warehouse simulation — aisle-graph routing, pick at tote-front node "
                 "(model cost = aisle_distance)",
                 fontsize=12, fontweight="bold")

    # 동적 아티스트 (프레임마다 갱신)
    order_scatter = ax.scatter([], [], s=42, marker="o",
                               facecolor=_C_ORDER, edgecolor=_C_ORDER_E,
                               linewidth=0.6, zorder=3, label="active order")

    robot_dots, robot_lines, robot_texts = [], [], []
    for i in range(num_robots):
        color = _ROBOT_CMAP(i % 10)
        (line,) = ax.plot([], [], color=color, lw=1.3, alpha=0.55, zorder=5)
        dot = ax.scatter([], [], s=140, marker="s", facecolor=color,
                         edgecolor="black", linewidth=0.8, zorder=6)
        txt = ax.text(0, 0, "", ha="center", va="center", fontsize=7,
                      color="white", fontweight="bold", zorder=7)
        robot_lines.append(line)
        robot_dots.append(dot)
        robot_texts.append(txt)

    clock_txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, ha="left",
                        va="top", fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round", fc="white", ec="#888",
                                  alpha=0.85), zorder=10)

    policy_name = {"rl": "D3QN policy", "nearest": "Nearest-robot (baseline)"}
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    order_arr = np.array([o[0] for o in orders]) if orders else np.zeros(0)
    order_loc = np.array([o[2] for o in orders]) if orders else np.zeros((0, 2))
    order_end = np.array([o[1] if o[1] is not None else np.inf for o in orders]) \
        if orders else np.zeros(0)

    def update(frame):
        t = times[frame]
        # 활성 주문: 도착 ≤ t < 완료
        if len(order_arr):
            active = (order_arr <= t) & (t < order_end)
            order_scatter.set_offsets(order_loc[active] if active.any()
                                      else np.empty((0, 2)))
        n_active = int(active.sum()) if len(order_arr) else 0

        moving = 0
        for i in range(num_robots):
            (x, y), is_moving, remain = _pos_at(by_robot[i], t, ws)
            robot_dots[i].set_offsets([[x, y]])
            robot_texts[i].set_position((x, y))
            robot_texts[i].set_text(str(i))
            moving += int(is_moving)
            # 남은 이동 경로를 통로 폴리라인으로 표시
            if remain and len(remain) > 1:
                rx = [p[0] for p in remain]
                ry = [p[1] for p in remain]
                robot_lines[i].set_data(rx, ry)
            else:
                robot_lines[i].set_data([], [])

        clock_txt.set_text(
            f"{policy_name.get(record['policy'], record['policy'])}\n"
            f"t = {t:6.1f} s / {horizon:.0f} s   ·   ×{playback_speed:.0f}\n"
            f"robots moving: {moving}/{num_robots}   ·   "
            f"active orders: {n_active}\n"
            f"M={num_robots}  K={capacity}  λ={int(record['lam']*3600)}/hr")

        return (order_scatter, clock_txt, *robot_dots, *robot_lines, *robot_texts)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps,
                         blit=False)

    if save_path.endswith(".gif"):
        anim.save(save_path, writer=PillowWriter(fps=fps))
    else:  # mp4 등 — ffmpeg 필요
        anim.save(save_path, fps=fps)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="창고 RL 구조 시각화")
    p.add_argument("mode", choices=["map", "anim"], help="map=정적맵, anim=애니메이션")
    p.add_argument("--policy", default="rl", choices=["rl", "nearest"])
    p.add_argument("--robots", type=int, default=5)
    p.add_argument("--capacity", type=int, default=5)
    p.add_argument("--lam", type=float, default=400 / 3600, help="주문 도착률(orders/s)")
    p.add_argument("--horizon", type=float, default=500.0, help="시뮬레이션 길이(초)")
    p.add_argument("--train", type=int, default=0, help="애니 전 quick-train 에피소드 수")
    p.add_argument("--speed", type=float, default=12.0, help="배속(sim초/실초)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    layout = WarehouseLayout(seed=42)

    if args.mode == "map":
        out = args.out or "layout_map.png"
        path = plot_layout_map(layout, save_path=out)
        print(f"레이아웃 맵 저장 → {path}")
        return

    # anim
    agent = None
    if args.policy == "rl":
        agent = DQNAgent(num_robots=args.robots, robot_capacity=args.capacity,
                         seed=args.seed, sim_time=SIM_TIME)
        if args.train > 0:
            print(f"quick-train {args.train} 에피소드 …")
            quick_train(agent, layout, args.robots, args.capacity,
                        args.lam, episodes=args.train, seed=args.seed)
            print(f"  학습 후 ε = {agent.epsilon:.3f}")
        else:
            print("경고: 학습되지 않은 정책(ε=1.0, 랜덤 배정). "
                  "--train N 으로 quick-train 하면 구조가 드러납니다.")

    record = run_and_record(layout, policy=args.policy, agent=agent,
                            num_robots=args.robots, robot_capacity=args.capacity,
                            lam=args.lam, horizon=args.horizon, seed=args.seed)
    print(f"기록: {len(record['segments'])} 이동구간, {len(record['orders'])} 주문")

    out = args.out or "sim_animation.gif"
    path = animate_simulation(layout, record=record, save_path=out,
                              playback_speed=args.speed)
    print(f"애니메이션 저장 → {path}")


if __name__ == "__main__":
    main()
