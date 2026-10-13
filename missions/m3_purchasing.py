"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []

    # Extension 5: Carbon-aware scheduling data tracking
    # Interruption rates by GPU type: H100 (0.03 ~ low), A100 (0.05), A10G (0.08)
    GPU_INTERRUPT_RATES = {"H100": 0.03, "H200": 0.03, "A100": 0.05, "A10G": 0.08, "L4": 0.06}

    interruptible_jobs = []

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days_actual = int(num(j["days"]))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours_monthly = hpd * DAYS * ngpu
        gpu_hours_actual = hpd * days_actual * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours_monthly * od

        # Extension 1: recommend_tier with gpu_type and job_days
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=days_actual)
        irate = GPU_INTERRUPT_RATES.get(gtype, 0.05)

        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours_monthly, num(c["spot_hr"]), od, interrupt_rate=irate)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours_monthly * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({
            "job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
            "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
        })

        # Extension 5: Track carbon footprint for interruptible workloads
        if interruptible:
            watts_per_gpu = num(c["watts"])
            total_watts = watts_per_gpu * ngpu
            wh_monthly = total_watts * (hpd * DAYS)
            wh_actual = total_watts * (hpd * days_actual)

            # Regional comparison (monthly 30 days basis)
            base_region = "us-east-1"
            clean_region = "europe-north1"
            cheap_region = "us-east-wa"

            carbon_base_kg = sustainability.carbon_g(wh_monthly, base_region) / 1000.0
            carbon_clean_kg = sustainability.carbon_g(wh_monthly, clean_region) / 1000.0
            elec_cost_base = sustainability.energy_cost_usd(wh_monthly, base_region)
            elec_cost_clean = sustainability.energy_cost_usd(wh_monthly, clean_region)
            elec_cost_cheap = sustainability.energy_cost_usd(wh_monthly, cheap_region)

            interruptible_jobs.append({
                "job_id": j["job_id"],
                "gpu_type": gtype,
                "ngpu": ngpu,
                "days_actual": days_actual,
                "wh_monthly": wh_monthly,
                "wh_actual": wh_actual,
                "kwh_monthly": wh_monthly / 1000.0,
                "carbon_us_east_1_kg": round(carbon_base_kg, 2),
                "carbon_europe_north1_kg": round(carbon_clean_kg, 2),
                "carbon_saved_kg": round(carbon_base_kg - carbon_clean_kg, 2),
                "elec_cost_us_east_1": round(elec_cost_base, 2),
                "elec_cost_europe_north1": round(elec_cost_clean, 2),
                "elec_cost_us_east_wa": round(elec_cost_cheap, 2),
            })

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Aggregate Carbon metrics (30-day monthly & actual days)
    tot_kwh_monthly = sum(j["kwh_monthly"] for j in interruptible_jobs)
    tot_wh_monthly = sum(j["wh_monthly"] for j in interruptible_jobs)
    tot_wh_actual = sum(j["wh_actual"] for j in interruptible_jobs)

    regions_comparison = {}
    for rname in sustainability.REGION_CARBON:
        c_kg = sustainability.carbon_g(tot_wh_monthly, rname) / 1000.0
        cost_usd = sustainability.energy_cost_usd(tot_wh_monthly, rname)
        regions_comparison[rname] = {
            "gco2_per_kwh": sustainability.REGION_CARBON[rname],
            "price_kwh": sustainability.REGION_PRICE_KWH[rname],
            "carbon_kg": round(c_kg, 1),
            "elec_cost_usd": round(cost_usd, 2),
        }

    total_carbon_us_east_1 = sustainability.carbon_g(tot_wh_monthly, "us-east-1") / 1000.0
    total_carbon_europe_north1 = sustainability.carbon_g(tot_wh_monthly, "europe-north1") / 1000.0
    carbon_reduction_pct = ((total_carbon_us_east_1 - total_carbon_europe_north1) / total_carbon_us_east_1 * 100) if total_carbon_us_east_1 else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- Extension 5: Carbon-aware Scheduling (Interruptible Jobs) --")
        print(f"{'Job ID':18}{'GPU x N':12}{'kWh/mo':>10}{'us-east-1(kg)':>15}{'eu-north1(kg)':>15}{'CO2 Saved':>12}")
        for ij in interruptible_jobs:
            gpu_str = f"{ij['gpu_type']} x {ij['ngpu']}"
            print(f"{ij['job_id']:18}{gpu_str:12}{ij['kwh_monthly']:>10.0f}{ij['carbon_us_east_1_kg']:>15.1f}{ij['carbon_europe_north1_kg']:>15.1f}{ij['carbon_saved_kg']:>12.1f} kg")

        print(f"\nAll 5 Regions Comparison for Interruptible Workloads ({tot_kwh_monthly:,.0f} kWh/month):")
        print(f"{'Region':18}{'gCO2/kWh':>10}{'$/kWh':>8}{'Carbon(kg)':>14}{'Elec Cost($)':>14}")
        for rname, rdata in sorted(regions_comparison.items(), key=lambda x: x[1]["carbon_kg"]):
            print(f"{rname:18}{rdata['gco2_per_kwh']:>10}{rdata['price_kwh']:>8.3f}{rdata['carbon_kg']:>14.1f}${rdata['elec_cost_usd']:>13.2f}")

        print(f"\nInsight: Moving interruptible jobs from us-east-1 to europe-north1 cuts emissions by {carbon_reduction_pct:.1f}% (-{total_carbon_us_east_1 - total_carbon_europe_north1:,.1f} kg CO2e)!")
        print("Cheapest power: us-east-wa ($0.055/kWh) | Cleanest power: europe-north1 (30 gCO2/kWh)")
        print("Latency note: Batch training/eval is delay-tolerant, making cross-region scheduling zero impact on user latency.")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "carbon_scheduling": {
            "interruptible_jobs": interruptible_jobs,
            "regions_comparison": regions_comparison,
            "carbon_reduction_monthly_kg": round(total_carbon_us_east_1 - total_carbon_europe_north1, 1),
            "carbon_reduction_pct": round(carbon_reduction_pct, 1),
            "tot_wh_monthly": tot_wh_monthly,
            "tot_wh_actual": tot_wh_actual,
        },
    }


if __name__ == "__main__":
    run()

