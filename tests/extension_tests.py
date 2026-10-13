"""Unit tests for Lab 25 FinOps Extensions (Your Turn)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from finops import pricing, sustainability


def test_cache_break_even_reads():
    # Base input price $3.00, write cost 1.25x = $3.75, read discount = 0.10 (-90%)
    # Savings per read = $3.00 * (1 - 0.10) = $2.70
    # Break-even = 3.75 / 2.70 = 1.3888...
    be = pricing.cache_break_even_reads(write_cost_per_m=3.75, read_discount=0.10, base_input_price=3.00)
    assert abs(be - (3.75 / 2.70)) < 1e-6


def test_cache_is_worth_it_logic():
    # If read reuse is high (5 reads >= 1.39 threshold) -> True
    assert pricing.cache_is_worth_it(avg_cache_reads=5.0, write_cost_per_m=3.75, read_discount=0.10, base_input_price=3.00) is True

    # If read reuse is low (1 read < 1.39 threshold) -> False
    assert pricing.cache_is_worth_it(avg_cache_reads=1.0, write_cost_per_m=3.75, read_discount=0.10, base_input_price=3.00) is False

    # Zero or negative savings per read -> False
    assert pricing.cache_is_worth_it(avg_cache_reads=10.0, write_cost_per_m=1.0, read_discount=1.0, base_input_price=1.0) is False


def test_enhanced_recommend_tier():
    # Existing default behavior unchanged
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"

    # GPU-aware calls
    assert pricing.recommend_tier(20, True, gpu_type="H100", job_days=14) == "spot"
    assert pricing.recommend_tier(24, False, gpu_type="A100", job_days=30) == "reserved"
    assert pricing.recommend_tier(6, False, gpu_type="A10G", job_days=5) == "on_demand"
