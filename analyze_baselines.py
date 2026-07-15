"""
analyze_baselines.py — exp_baselines.csv → 논문용 비교 테이블 생성

exp_baselines.py 가 만든 paired 원자료(logs/exp_baselines.csv)를 읽어
"RL 우수성"을 보이는 3개 표를 콘솔·CSV·LaTeX 로 출력한다.

  Table 1 (메인)   : 대표 운영점(M, λ)에서 6개 정책 × 핵심지표, mean±std.
                     RL 의 '최고 베이스라인 대비 개선%' 와 Pareto 비지배 여부 표시.
  Table 2 (부하스윕): 고정 M 에서 λ 증가에 따른 flow time (정책별 열).
  Table 3 (부하스윕): 같은 형식으로 dist/order.

의존성: 표준 라이브러리만 사용(csv, statistics). pandas 불필요.

사용법
──────
  python analyze_baselines.py                                   # logs/exp_baselines.csv
  python analyze_baselines.py --csv logs/exp_baselines.csv \
        --M 5 --lam 600                                         # 대표 운영점 지정
  python analyze_baselines.py --csv logs/exp_baselines_smoke.csv  # 스모크 검증
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# 표시 순서(왼→오): RL 먼저, 그다음 강한 휴리스틱 → 약한 순
POLICY_ORDER = ["RL", "BL5_Cheapest", "BL4_Nearest", "BL6_Random",
                "BL2_Partial_k3", "BL1_FullBuffer"]
POLICY_DISP = {
    "RL":             "RL (D3QN)",
    "BL5_Cheapest":   "Cheapest-Ins.",
    "BL4_Nearest":    "Nearest-Robot",
    "BL6_Random":     "Random",
    "BL2_Partial_k3": "Partial-k3",
    "BL1_FullBuffer": "Full-Buffer",
}

# (csv 컬럼, higher_better, 표시명, 포맷)
METRICS = [
    ("avg_flow_time",        False, "Flow time (s)",   "{:.1f}"),
    ("dist_per_order",       False, "Dist/order (m)",  "{:.2f}"),
    ("robot_utilization",    True,  "Utilization",     "{:.3f}"),
    ("saving_per_insertion", True,  "Saving/ins (m)",  "{:.2f}"),
    ("completion_rate",      True,  "Compl. rate",     "{:.4f}"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 로드 & 집계
# ─────────────────────────────────────────────────────────────────────────────

def load_rows(path: str) -> List[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k == "label":
                continue
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                r[k] = float("nan")
    return rows


def aggregate(rows: List[dict]) -> Dict[Tuple[str, int, int], Dict[str, Tuple[float, float, int]]]:
    """(label, M, λ_per_hr) → {metric: (mean, std, n)}."""
    buckets: Dict[Tuple[str, int, int], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        key = (r["label"], int(r["num_robots"]), round(r["lam"] * 3600))
        for m, *_ in METRICS:
            buckets[key][m].append(r[m])
    out: Dict[Tuple[str, int, int], Dict[str, Tuple[float, float, int]]] = {}
    for key, mv in buckets.items():
        out[key] = {}
        for m, vals in mv.items():
            vals = [v for v in vals if v == v]  # drop NaN
            if not vals:
                out[key][m] = (float("nan"), float("nan"), 0)
            else:
                mean = sum(vals) / len(vals)
                std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
                out[key][m] = (mean, std, len(vals))
    return out


def available(stats) -> Tuple[List[int], List[int], List[str]]:
    Ms   = sorted({k[1] for k in stats})
    lams = sorted({k[2] for k in stats})
    labs = [p for p in POLICY_ORDER if any(k[0] == p for k in stats)]
    return Ms, lams, labs


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — 메인 비교 (대표 운영점)
# ─────────────────────────────────────────────────────────────────────────────

def _best_label(stats, M, lam, metric, higher, labels) -> Optional[str]:
    cands = [(l, stats[(l, M, lam)][metric][0]) for l in labels
             if (l, M, lam) in stats and stats[(l, M, lam)][metric][0] == stats[(l, M, lam)][metric][0]]
    if not cands:
        return None
    return (max if higher else min)(cands, key=lambda x: x[1])[0]


def _pareto_nondominated(stats, M, lam, labels) -> Dict[str, bool]:
    """flow(↓) · dist/order(↓) 2목적에서 각 정책의 비지배(non-dominated) 여부."""
    pts = {}
    for l in labels:
        if (l, M, lam) in stats:
            f = stats[(l, M, lam)]["avg_flow_time"][0]
            d = stats[(l, M, lam)]["dist_per_order"][0]
            if f == f and d == d:
                pts[l] = (f, d)
    nd = {}
    for l, (f, d) in pts.items():
        dominated = any(
            (of <= f and od <= d) and (of < f or od < d)
            for o, (of, od) in pts.items() if o != l)
        nd[l] = not dominated
    return nd


def table_main(stats, M, lam, labels) -> Tuple[str, List[List[str]]]:
    """콘솔 문자열 + LaTeX 용 셀 행렬을 함께 반환."""
    nd = _pareto_nondominated(stats, M, lam, labels)
    header = f"Table 1 — Main comparison  (M={M}, λ={lam}/hr, ρ≈{lam/(M*120):.2f})"
    colw = 15
    lines = ["", "=" * (20 + colw * len(labels)), "  " + header,
             "=" * (20 + colw * len(labels))]
    hdr = f"  {'Metric':<18}" + "".join(f"{POLICY_DISP[l]:>{colw}}" for l in labels)
    lines.append(hdr)
    lines.append("  " + "-" * (18 + colw * len(labels)))

    latex_rows: List[List[str]] = []
    for metric, higher, disp, fmt in METRICS:
        best = _best_label(stats, M, lam, metric, higher, labels)
        row = f"  {disp:<18}"
        lrow = [disp]
        for l in labels:
            if (l, M, lam) not in stats:
                row += f"{'—':>{colw}}"; lrow.append("--"); continue
            mean, std, n = stats[(l, M, lam)][metric]
            cell = f"{fmt.format(mean)}±{fmt.format(std)}"
            mark = "*" if l == best else " "
            row += f"{(cell + mark):>{colw}}"
            lrow.append(("\\textbf{%s}" % fmt.format(mean)) if l == best
                        else fmt.format(mean))
        lines.append(row)
        latex_rows.append(lrow)

    # RL 개선율 (최고 '비-RL' 베이스라인 대비)
    lines.append("  " + "-" * (18 + colw * len(labels)))
    lines.append("  RL vs 최고 베이스라인(비-RL) 개선율:")
    for metric, higher, disp, _ in METRICS:
        if ("RL", M, lam) not in stats:
            continue
        rl = stats[("RL", M, lam)][metric][0]
        others = [(l, stats[(l, M, lam)][metric][0]) for l in labels
                  if l != "RL" and (l, M, lam) in stats
                  and stats[(l, M, lam)][metric][0] == stats[(l, M, lam)][metric][0]]
        if not others:
            continue
        best_l, best_v = (max if higher else min)(others, key=lambda x: x[1])
        if best_v == 0:
            continue
        impr = ((rl - best_v) / abs(best_v) * 100.0) if higher \
               else ((best_v - rl) / abs(best_v) * 100.0)
        verdict = "RL 우세" if impr > 0 else "RL 열세"
        lines.append(f"     {disp:<16} {impr:+6.1f}%  vs {POLICY_DISP[best_l]:<14} [{verdict}]")

    # Pareto 요약
    lines.append("  " + "-" * (18 + colw * len(labels)))
    frontier = [POLICY_DISP[l] for l in labels if nd.get(l)]
    lines.append(f"  Pareto 비지배(flow↓·dist↓) 프론티어: {', '.join(frontier)}")
    if nd.get("RL"):
        lines.append("     → RL 은 비지배(어떤 정책도 flow·dist 둘 다 RL 이상이 아님).")
    lines.append("=" * (20 + colw * len(labels)))
    return "\n".join(lines), latex_rows


# ─────────────────────────────────────────────────────────────────────────────
# Table 2/3 — 부하 스윕 (고정 M, λ 별 정책 비교)
# ─────────────────────────────────────────────────────────────────────────────

def table_sweep(stats, M, lams, labels, metric, higher, disp, fmt) -> str:
    colw = 15
    title = f"Table — {disp} vs λ  (M={M})   [열마다 최우수 *]"
    lines = ["", "=" * (12 + colw * len(labels)), "  " + title,
             "=" * (12 + colw * len(labels))]
    hdr = f"  {'λ (/hr)':<10}" + "".join(f"{POLICY_DISP[l]:>{colw}}" for l in labels)
    lines.append(hdr)
    lines.append("  " + "-" * (10 + colw * len(labels)))
    for lam in lams:
        best = _best_label(stats, M, lam, metric, higher, labels)
        row = f"  {lam:<10}"
        for l in labels:
            if (l, M, lam) not in stats:
                row += f"{'—':>{colw}}"; continue
            mean = stats[(l, M, lam)][metric][0]
            mark = "*" if l == best else " "
            row += f"{(fmt.format(mean) + mark):>{colw}}"
        lines.append(row)
    lines.append("=" * (12 + colw * len(labels)))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX 출력 (drop-in booktabs)
# ─────────────────────────────────────────────────────────────────────────────

def write_latex_main(path: str, latex_rows, labels, M, lam) -> None:
    cols = "l" + "r" * len(labels)
    head = " & ".join(["Metric"] + [POLICY_DISP[l] for l in labels]) + r" \\"
    body = "\n".join(" & ".join(row) + r" \\" for row in latex_rows)
    tex = rf"""% auto-generated by analyze_baselines.py
\begin{{table}}[t]
\centering
\caption{{Policy comparison at $M={M}$, $\lambda={lam}$/hr (mean over paired eval seeds; best in bold).}}
\label{{tab:main-comparison}}
\begin{{tabular}}{{{cols}}}
\toprule
{head}
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    with open(path, "w") as f:
        f.write(tex)


def write_summary_csv(path: str, stats, Ms, lams, labels) -> None:
    """(M, λ, policy) 별 metric 평균/표준편차 tidy CSV — 그림·추가분석용."""
    fields = ["num_robots", "lam_per_hr", "policy"] + \
             [f"{m}_mean" for m, *_ in METRICS] + [f"{m}_std" for m, *_ in METRICS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for M in Ms:
            for lam in lams:
                for l in labels:
                    if (l, M, lam) not in stats:
                        continue
                    row = {"num_robots": M, "lam_per_hr": lam, "policy": l}
                    for m, *_ in METRICS:
                        mean, std, _ = stats[(l, M, lam)][m]
                        row[f"{m}_mean"] = round(mean, 4)
                        row[f"{m}_std"]  = round(std, 4)
                    w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="exp_baselines.csv → 비교 테이블")
    p.add_argument("--csv", default="logs/exp_baselines.csv")
    p.add_argument("--M",   type=int, default=None, help="대표 운영점 M (기본: 중앙값)")
    p.add_argument("--lam", type=int, default=None, help="대표 운영점 λ/hr (기본: 최대=고부하)")
    p.add_argument("--out-dir", default="logs")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV 없음: {args.csv}\n  먼저 `python exp_baselines.py` 로 생성하세요.")

    rows  = load_rows(args.csv)
    stats = aggregate(rows)
    Ms, lams, labels = available(stats)
    if not Ms or not lams:
        raise SystemExit("집계할 데이터가 없습니다.")

    M_rep   = args.M   if args.M   in Ms   else Ms[len(Ms) // 2]
    lam_rep = args.lam if args.lam in lams else lams[-1]   # 최대 부하에서 RL 이 두드러짐
    print(f"[입력] {args.csv}  ·  policies={labels}")
    print(f"[격자] M={Ms}  λ={lams}/hr  ·  대표점 M={M_rep}, λ={lam_rep}/hr")

    # Table 1
    main_str, latex_rows = table_main(stats, M_rep, lam_rep, labels)
    print(main_str)

    # Table 2/3 (부하 스윕, 대표 M)
    print(table_sweep(stats, M_rep, lams, labels,
                      "avg_flow_time", False, "Flow time (s)", "{:.1f}"))
    print(table_sweep(stats, M_rep, lams, labels,
                      "dist_per_order", False, "Dist/order (m)", "{:.2f}"))

    # 산출물 저장
    os.makedirs(args.out_dir, exist_ok=True)
    tex_path = os.path.join(args.out_dir, "table_main.tex")
    sum_path = os.path.join(args.out_dir, "summary_by_policy.csv")
    write_latex_main(tex_path, latex_rows, labels, M_rep, lam_rep)
    write_summary_csv(sum_path, stats, Ms, lams, labels)
    print(f"\n✔ LaTeX 메인표 → {tex_path}")
    print(f"✔ tidy 요약 CSV → {sum_path}  (그림·추가분석용)")


if __name__ == "__main__":
    main()
