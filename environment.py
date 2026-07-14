import random
from typing import List, Tuple, Dict, Callable

BN            = 2       # number of blocks
AN            = 10      # aisles per block
SN            = 20      # shelves on ONE side of each aisle (rows)

Wa            = 3.0     # m  — aisle (corridor) width
Wc            = 4.0     # m  — cross-aisle width
Wt            = 0.6     # m  — tote width  (row pitch along y)
Dt            = 0.5     # m  — tote depth  (rack depth along x), single-deep

NUM_AISLES    = BN * AN             # 20 total aisles
STORAGE_LOCS  = BN * AN * 2 * SN   # 800 storage locations

NUM_SKUS      = 200     # scaled from original (300 locs → 50 SKUs  ≈ 1 SKU / 6 locs)

NUM_ROBOTS      = 10
ROBOT_CAPACITY  = 5
ROBOT_SPEED     = 1.0    # m/s

SIM_TIME        = 10 * 3600          # 10-hour shift
LAMBDA_DEFAULT  = 600 / 3600         # orders/s


class WarehouseLayout:
    def __init__(self, seed: int = 42,
                 num_robots: int = NUM_ROBOTS,
                 robot_capacity: int = ROBOT_CAPACITY):
        self.num_robots     = num_robots
        self.robot_capacity = robot_capacity
        self.rng            = random.Random(seed)

        # Derived geometry
        self.aisle_block_w  = Wa + 2 * Dt
        self.block_h        = Wc + SN * Wt
        self.total_width    = AN * self.aisle_block_w            # 40.0 m
        self.total_height   = BN * self.block_h + (BN - 1) * Wc  # 36.0 m

        self.workstation    = (self.total_width, 0.0)   # 우측 하단

        # Cross-aisle bands (통로를 바꿀 때 가로로 건너갈 수 있는 열린 복도)
        #   bottom: [0, Wc]   ·   between blocks: [block top, next-block shelf start]
        self.cross_bands = [(0.0, Wc)]
        for b in range(BN - 1):
            top_b            = b * (self.block_h + Wc) + self.block_h
            shelf_start_next = (b + 1) * (self.block_h + Wc) + Wc
            self.cross_bands.append((top_b, shelf_start_next))

        # Build coordinates and SKU assignment
        #   storage_coords : 랙 면(rack face) 좌표 — 그리기/식별용
        #   access_coords  : tote 앞 통로 접근 노드 (cx, y) — 이동/픽업의 실제 지점
        self.storage_coords, self.access_coords = self._build_storage_coords()
        self.sku_ids        = list(range(NUM_SKUS))
        self.loc_to_sku, self.sku_to_locs = self._assign_skus()

    # Coordinate builder
    def _build_storage_coords(self):
        coords, access = [], []
        for b in range(BN):
            block_y0 = b * (self.block_h + Wc)
            for a in range(AN):
                cx = a * self.aisle_block_w + Dt + Wa / 2
                x_left  = cx - Wa / 2 - Dt / 2
                x_right = cx + Wa / 2 + Dt / 2
                for side_x in (x_left, x_right):
                    for row in range(SN):
                        y = block_y0 + Wc + (row + 0.5) * Wt
                        coords.append((side_x, y))
                        access.append((cx, y))       # 통로 접근 노드

        assert len(coords) == STORAGE_LOCS, (
            f"Coordinate count mismatch: expected {STORAGE_LOCS}, got {len(coords)}"
        )
        return coords, access

    # SKU assignment
    def _assign_skus(self) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        indices = list(range(len(self.storage_coords)))
        self.rng.shuffle(indices)
        loc_to_sku: Dict[int, int]       = {}
        sku_to_locs: Dict[int, List[int]] = {s: [] for s in self.sku_ids}

        for idx in indices:
            sku = self.rng.choice(self.sku_ids)
            loc_to_sku[idx] = sku
            sku_to_locs[sku].append(idx)

        for sku in self.sku_ids:
            if not sku_to_locs[sku]:
                fallback = self.rng.choice(indices)
                sku_to_locs[sku].append(fallback)

        return loc_to_sku, sku_to_locs

    # Distance & routing
    def manhattan(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Manhattan (L1) distance — 참고용(랙 통과 허용). 이동 비용엔 쓰지 않는다."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def _band_vert(ya: float, yb: float, lo: float, hi: float) -> float:
        """교차통로 밴드 [lo,hi]를 경유해 ya↔yb 로 이동할 때의 최소 세로 우회거리.

        min over y_c in [lo,hi] of |ya−y_c| + |yb−y_c|.
        """
        lo_ab, hi_ab = (ya, yb) if ya <= yb else (yb, ya)
        if hi < lo_ab:            # 밴드가 두 점보다 아래
            yc = hi
        elif lo > hi_ab:          # 밴드가 두 점보다 위
            yc = lo
        else:                     # 밴드가 y-구간과 겹침 → 우회 없음
            return abs(ya - yb)
        return abs(ya - yc) + abs(yb - yc)

    def aisle_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """통로 그래프 최단거리 (통로 전용 이동 + tote 앞 노드 픽업).

        같은 세로 통로(x 동일) → |Δy|.
        다른 통로 → 최근접 교차통로 밴드를 경유:  밴드 세로우회 + |Δx|.
        점들은 접근 노드 (cx, y) 또는 워크스테이션 (x, y) 로 가정한다.
        """
        x1, y1 = p1
        x2, y2 = p2
        if abs(x1 - x2) < 1e-9:
            return abs(y1 - y2)
        dx   = abs(x1 - x2)
        best = float("inf")
        for lo, hi in self.cross_bands:
            best = min(best, self._band_vert(y1, y2, lo, hi) + dx)
        return best

    def sample_order_sku(self, rng: random.Random) -> int:
        """Return a uniformly random SKU id."""
        return rng.choice(self.sku_ids)

    def find_tote_location(
        self, sku: int, rng: random.Random
    ) -> Tuple[int, Tuple[float, float]]:
        """Return (location_index, access_node (cx, y)) for a random slot holding *sku*.

        loc 은 tote 앞 통로 접근 노드다. 랙 면 좌표는 storage_coords[idx] 로 별도 조회.
        """
        locs = self.sku_to_locs[sku]
        idx  = rng.choice(locs)
        return idx, self.access_coords[idx]

    @property
    def num_locations(self) -> int:
        return len(self.storage_coords)

    @property
    def num_skus(self) -> int:
        return len(self.sku_ids)


# Routing utilities

def nearest_neighbor_route(
    start:        Tuple[float, float],
    destinations: List[Tuple[float, float]],
    dist_func:    Callable[[Tuple[float, float], Tuple[float, float]], float],
) -> List[Tuple[float, float]]:
    """Greedy nearest-neighbour tour starting from *start*."""
    remaining = list(destinations)
    route, current = [], start
    while remaining:
        nearest = min(remaining, key=lambda d: dist_func(current, d))
        route.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return route


def route_total_distance(
    start:     Tuple[float, float],
    route:     List[Tuple[float, float]],
    end:       Tuple[float, float],
    dist_func: Callable[[Tuple[float, float], Tuple[float, float]], float],
) -> float:
    if not route:
        return dist_func(start, end)
    total, current = 0.0, start
    for point in route:
        total   += dist_func(current, point)
        current  = point
    total += dist_func(current, end)
    return total