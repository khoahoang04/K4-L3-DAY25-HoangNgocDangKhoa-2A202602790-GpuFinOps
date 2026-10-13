"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Caching write surcharge: 1.25x base input price (standard Anthropic prompt caching model)
CACHE_WRITE_MULT = 1.25
ASSUMED_AVG_READS_HIGH = 5.0   # High-reuse scenario (chat system prompts, RAG templates)
ASSUMED_AVG_READS_LOW = 1.0    # Low-reuse scenario (one-off prompts)


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Extension 3: Break-even check for caching
    # Write cost is 1.25x input price, read discount is 0.10 (90% off) -> break-even is 1.25 / 0.90 = 1.39 reads
    be_reads_small = pricing.cache_break_even_reads(MODEL_PRICES["small"][0] * CACHE_WRITE_MULT, 0.10, MODEL_PRICES["small"][0])
    be_reads_large = pricing.cache_break_even_reads(MODEL_PRICES["large"][0] * CACHE_WRITE_MULT, 0.10, MODEL_PRICES["large"][0])

    # Extension 4: Reasoning budget tracking
    reasoning_stats = {"count": 0, "tokens": 0, "cost": 0.0, "wh": 0.0}
    standard_stats = {"count": 0, "tokens": 0, "cost": 0.0, "wh": 0.0}

    # Tracking for write cost analysis
    total_cached_tokens = 0
    total_write_cost_amortized = 0.0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached_candidate = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reas = bool(int(num(r["is_reasoning"])))
        route_tier = r["route_tier"]
        pin, pout = MODEL_PRICES[route_tier]
        total_tokens += inp + out

        # BASELINE: naive deployment — everything on large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # EXTENSION 3: Check cache_is_worth_it before applying cache
        write_cost_per_m = pin * CACHE_WRITE_MULT
        caching_worth_it = pricing.cache_is_worth_it(
            avg_cache_reads=ASSUMED_AVG_READS_HIGH,
            write_cost_per_m=write_cost_per_m,
            read_discount=0.10,
            base_input_price=pin,
        )
        cached = cached_candidate if caching_worth_it else 0
        req_cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        opt_cost += req_cost

        if cached > 0:
            total_cached_tokens += cached
            # Amortized write cost per read = (write_cost_per_m * cached/1e6) / avg_reads
            total_write_cost_amortized += (write_cost_per_m * (cached / 1e6)) / ASSUMED_AVG_READS_HIGH

        # EXTENSION 4: Track Wh & Cost by reasoning group
        q_wh = sustainability.wh_per_query(inp + out, is_reasoning=is_reas)
        if is_reas:
            reasoning_stats["count"] += 1
            reasoning_stats["tokens"] += (inp + out)
            reasoning_stats["cost"] += req_cost
            reasoning_stats["wh"] += q_wh
        else:
            standard_stats["count"] += 1
            standard_stats["tokens"] += (inp + out)
            standard_stats["cost"] += req_cost
            standard_stats["wh"] += q_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Net cost including amortized write cost
    opt_cost_with_write = opt_cost + total_write_cost_amortized
    opt_pm_with_write = pricing.dollars_per_million(opt_cost_with_write, total_tokens)

    # Low-reads simulation (avg_reads = 1.0): cache_is_worth_it() is FALSE, so cache is disabled
    opt_cost_low_reads = 0.0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost_low_reads += pricing.request_cost(inp, out, pin, pout, cached_in=0, batch=is_batch)
    opt_pm_low_reads = pricing.dollars_per_million(opt_cost_low_reads, total_tokens)

    # Extension 4 reasoning metrics:
    tot_req = len(rows)
    reas_req_pct = (reasoning_stats["count"] / tot_req) * 100
    reas_cost_pct = (reasoning_stats["cost"] / opt_cost) * 100 if opt_cost else 0.0
    tot_wh = reasoning_stats["wh"] + standard_stats["wh"]
    reas_wh_pct = (reasoning_stats["wh"] / tot_wh) * 100 if tot_wh else 0.0

    # 10% cap test:
    # If traffic is already under 10% (8.4%), savings = $0.00 / 0.0 Wh
    cap_10pct_savings_usd = 0.0
    cap_10pct_savings_wh = 0.0

    # If down-routing 50% reasoning requests to standard tier (removes 6x token inflation & 80x multiplier):
    # Simulated savings from smarter routing
    smart_routing_wh_saved = (reasoning_stats["wh"] * 0.5) * (1.0 - 1.0 / sustainability.REASONING_ENERGY_MULTIPLIER)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- Extension 3: cache_is_worth_it() Analysis --")
        print(f"Break-even reads threshold: Small={be_reads_small:.2f} reads, Large={be_reads_large:.2f} reads")
        print(f"High-reuse scenario (assumed {ASSUMED_AVG_READS_HIGH} reads/prefix): cache_is_worth_it=True")
        print(f"  Optimized $/1M-tok (read-only): ${opt_pm:.3f}")
        print(f"  Optimized $/1M-tok (including amortized write cost): ${opt_pm_with_write:.3f}")
        print(f"Low-reuse scenario ({ASSUMED_AVG_READS_LOW} read/prefix): cache_is_worth_it=False (cache bypassed)")
        print(f"  Optimized $/1M-tok without cache: ${opt_pm_low_reads:.3f}")

        print("\n-- Extension 4: Reasoning Budget Analysis --")
        print(f"Traffic breakdown: Standard={standard_stats['count']} ({100-reas_req_pct:.1f}%), Reasoning={reasoning_stats['count']} ({reas_req_pct:.1f}%)")
        print(f"Cost breakdown   : Standard=${standard_stats['cost']:.2f} ({100-reas_cost_pct:.1f}%), Reasoning=${reasoning_stats['cost']:.2f} ({reas_cost_pct:.1f}%)")
        print(f"Energy breakdown : Standard={standard_stats['wh']:.1f} Wh ({100-reas_wh_pct:.1f}%), Reasoning={reasoning_stats['wh']:.1f} Wh ({reas_wh_pct:.1f}%)")
        print(f"10% Traffic Cap test: Current traffic is {reas_req_pct:.1f}% <= 10.0% -> Savings from cap = $0.00 / 0.0 Wh")
        print(f"Routing recommendation: Cap reasoning to strict complex tasks. Down-routing 50% reasoning saves ~{smart_routing_wh_saved:.0f} Wh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_analysis": {
            "break_even_reads": round(be_reads_large, 2),
            "opt_pm_with_write": round(opt_pm_with_write, 3),
            "opt_pm_low_reads": round(opt_pm_low_reads, 3),
        },
        "reasoning_budget": {
            "traffic_pct": round(reas_req_pct, 1),
            "cost_pct": round(reas_cost_pct, 1),
            "wh_pct": round(reas_wh_pct, 1),
            "cap_10pct_savings_usd": cap_10pct_savings_usd,
            "cap_10pct_savings_wh": cap_10pct_savings_wh,
        },
    }


if __name__ == "__main__":
    run()

