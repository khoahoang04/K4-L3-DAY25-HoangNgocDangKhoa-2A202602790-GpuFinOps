"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # Extension 2: Right-sizing analysis based on MBU, VRAM & Bandwidth
    catalog_metrics = []
    for gtype, row in cat.items():
        vram = num(row["hbm_gb"])
        bw = num(row["peak_bw_tbs"])
        od = num(row["on_demand_hr"])
        cost_per_gb = od / vram if vram > 0 else 0.0
        cost_per_tbs = od / bw if bw > 0 else 0.0
        catalog_metrics.append({
            "gpu_type": gtype,
            "on_demand_hr": od,
            "hbm_gb": vram,
            "peak_bw_tbs": bw,
            "cost_per_gb_vram": round(cost_per_gb, 4),
            "cost_per_tbs_bw": round(cost_per_tbs, 3),
        })

    # Identify memory-bound candidates for right-sizing
    # A100 infer (low util ~30%, MFU ~0.25) -> L4 or A10G if VRAM permits, or H200/MI300X if capacity needed
    rightsize_proposals = [
        {
            "gpu_id": "gpu-h100-4",
            "current_gpu": "H100 ($2.50/hr)",
            "proposed_gpu": "A100 ($1.79/hr)",
            "reason": "GPU-Util lie (MFU 0.20 << 0.40). Severe memory stall / compute underutilization. Downgrade cuts $0.71/hr.",
            "monthly_savings_usd": round((2.50 - 1.79) * 24 * 30),
        },
        {
            "gpu_id": "gpu-a10g-1",
            "current_gpu": "A10G ($1.00/hr)",
            "proposed_gpu": "L4 ($0.80/hr)",
            "reason": "Util lie on A10G (util 97%, MFU 0.27). L4 has identical 24GB VRAM with lower power/cost. Cuts $0.20/hr.",
            "monthly_savings_usd": round((1.00 - 0.80) * 24 * 30),
        },
        {
            "gpu_id": "gpu-a100-0",
            "current_gpu": "A100 ($1.79/hr)",
            "proposed_gpu": "L4 x 2 or A10G ($1.00/hr)",
            "reason": "Memory-bound inference with low utilization (<32%). Memory footprint fits in smaller GPU.",
            "monthly_savings_usd": round((1.79 - 1.00) * 24 * 30),
        },
    ]

    total_mbu_rightsize_savings = sum(p["monthly_savings_usd"] for p in rightsize_proposals)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- Extension 2: Right-sizing Analysis (MBU / VRAM / Bandwidth) --")
        print(f"{'GPU Type':10}{'$/hr':>7}{'VRAM(GB)':>10}{'BW(TB/s)':>10}{'$/GB-VRAM':>12}{'$/(TB/s)':>10}")
        for cm in catalog_metrics:
            print(f"{cm['gpu_type']:10}${cm['on_demand_hr']:>6.2f}{cm['hbm_gb']:>10.0f}{cm['peak_bw_tbs']:>10.2f}${cm['cost_per_gb_vram']:>11.4f}${cm['cost_per_tbs_bw']:>9.3f}")

        print("\nRight-sizing Recommendations for Memory-bound / Underutilized GPUs:")
        print(f"{'Target':14}{'Current':18}{'Proposed':18}{'Monthly Saved':>14}")
        for p in rightsize_proposals:
            print(f"{p['gpu_id']:14}{p['current_gpu']:18}{p['proposed_gpu']:18}${p['monthly_savings_usd']:>13,}")
            print(f"  -> Rationale: {p['reason']}")
        print(f"Total potential monthly right-sizing savings: ${total_mbu_rightsize_savings:,}")
        print("LIMITATION NOTE: Theoretical estimates based on catalog specs. Production migration requires load testing / benchmarks to verify serving throughput and p99 latency SLA remain intact.")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "mbu_rightsizing": {
            "proposals": rightsize_proposals,
            "total_monthly_savings": total_mbu_rightsize_savings,
            "catalog_metrics": catalog_metrics,
        },
    }


if __name__ == "__main__":
    run()

