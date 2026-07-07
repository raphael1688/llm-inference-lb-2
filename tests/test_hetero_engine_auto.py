"""
Unit tests for heterogeneous engine auto-detection (engine_type: auto)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models import (
    Pool, PoolMember, EngineType,
    initialize_engine_metrics_candidates,
    clear_all_pools_metrics_key_cache,
    get_effective_engine_type,
    compute_member_detection_status,
    DETECTION_STATUS_OK,
    DETECTION_STATUS_PARTIAL,
    DETECTION_STATUS_FAILED,
    add_or_update_pool,
    POOLS,
)
from core.metrics_collector import MetricsCollector
from config.config_loader import ConfigLoader


VLLM_METRICS = """
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="xxx"} 12.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="xxx"} 0.35
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="xxx"} 5.0
""".strip()

SGLANG_METRICS = """
# HELP sglang:num_queue_reqs The number of requests in the waiting queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="llama"} 3.0
# HELP sglang:token_usage The token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="llama"} 0.55
# HELP sglang:num_running_reqs The number of requests currently running
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="llama"} 2.0
""".strip()

VLLM_MINDIE_METRICS = """
num_requests_waiting{model="x"} 8.0
npu_cache_usage_perc{model="x"} 0.42
num_requests_running{model="x"} 3.0
""".strip()

UNKNOWN_METRICS = """
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.
process_cpu_seconds_total 12.5
""".strip()

VLLM_PARTIAL_METRICS = """
vllm:num_requests_waiting{model_name="xxx"} 12.0
vllm:num_requests_running{model_name="xxx"} 5.0
""".strip()


def _setup_candidates():
    initialize_engine_metrics_candidates({
        "vllm_mindie": {
            "waiting_queue": "num_requests_waiting",
            "cache_usage": "npu_cache_usage_perc",
            "running_req": "num_requests_running",
        },
        "vllm_v0_8": {
            "waiting_queue": "vllm:pending_requests",
            "cache_usage": "vllm:kv_cache_usage_perc",
            "running_req": "vllm:active_requests",
        },
        "sglang_v2": {
            "waiting_queue": "sglang:pending_req",
            "cache_usage": "sglang:token_usage_v2",
            "running_req": "sglang:running_req_v2",
        },
    })


def _auto_pool(members):
    return Pool("mixed-pool", "Common", EngineType.AUTO, members)


def _collector():
    return MetricsCollector(timeout=3)


def test_config_loader_accepts_auto():
    loader = ConfigLoader()
    assert "auto" in loader.SUPPORTED_ENGINE_TYPES


def test_detect_vllm_in_auto_pool():
    _setup_candidates()
    pool = _auto_pool([PoolMember("10.0.0.1", 8001, "Common")])
    member = pool.members[0]
    collector = _collector()

    metrics = collector._parse_prometheus_metrics(VLLM_METRICS, pool, member)

    assert member.detected_engine_type == EngineType.VLLM
    assert metrics["waiting_queue"] == 12.0
    assert metrics["cache_usage"] == 0.35
    assert metrics["running_req"] == 5.0
    assert member.detected_variant == "vllm"


def test_detect_sglang_in_auto_pool():
    _setup_candidates()
    pool = _auto_pool([PoolMember("10.0.0.2", 8010, "Common")])
    member = pool.members[0]
    collector = _collector()

    metrics = collector._parse_prometheus_metrics(SGLANG_METRICS, pool, member)

    assert member.detected_engine_type == EngineType.SGLANG
    assert metrics["waiting_queue"] == 3.0
    assert metrics["cache_usage"] == 0.55
    assert metrics["running_req"] == 2.0
    assert member.detected_variant == "sglang"


def test_mixed_pool_members_independent_detection():
    _setup_candidates()
    vllm_member = PoolMember("10.0.0.1", 8001, "Common")
    sglang_member = PoolMember("10.0.0.2", 8010, "Common")
    pool = _auto_pool([vllm_member, sglang_member])
    collector = _collector()

    vllm_metrics = collector._parse_prometheus_metrics(VLLM_METRICS, pool, vllm_member)
    sglang_metrics = collector._parse_prometheus_metrics(SGLANG_METRICS, pool, sglang_member)

    assert vllm_member.detected_engine_type == EngineType.VLLM
    assert sglang_member.detected_engine_type == EngineType.SGLANG
    assert vllm_metrics and sglang_metrics
    assert vllm_member.detected_engine_type != sglang_member.detected_engine_type


def test_vllm_mindie_variant_via_user_config():
    _setup_candidates()
    pool = _auto_pool([PoolMember("10.0.0.3", 8003, "Common")])
    member = pool.members[0]
    collector = _collector()

    metrics = collector._parse_prometheus_metrics(VLLM_MINDIE_METRICS, pool, member)

    assert member.detected_engine_type == EngineType.VLLM
    assert member.detected_variant == "vllm_mindie"
    assert metrics["waiting_queue"] == 8.0
    assert metrics["cache_usage"] == 0.42


def test_unknown_metrics_detection_failed():
    _setup_candidates()
    pool = _auto_pool([PoolMember("10.0.0.9", 8099, "Common")])
    member = pool.members[0]
    collector = _collector()

    metrics = collector._parse_prometheus_metrics(UNKNOWN_METRICS, pool, member)

    assert metrics == {}
    assert member.detected_engine_type is None


def test_homogeneous_vllm_pool_unchanged():
    _setup_candidates()
    pool = Pool("vllm-pool", "Common", EngineType.VLLM, [PoolMember("10.0.0.1", 8001, "Common")])
    member = pool.members[0]
    collector = _collector()

    metrics = collector._parse_prometheus_metrics(VLLM_METRICS, pool, member)

    assert member.detected_engine_type is None
    assert metrics["waiting_queue"] == 12.0
    assert member.detected_variant == "vllm"


def test_cross_engine_fallback_misconfigured_pool():
    _setup_candidates()
    pool = Pool("wrong-type", "Common", EngineType.VLLM, [PoolMember("10.0.0.2", 8010, "Common")])
    member = pool.members[0]
    collector = _collector()

    metrics = collector._parse_prometheus_metrics(SGLANG_METRICS, pool, member)

    assert metrics["waiting_queue"] == 3.0
    assert member.detected_engine_type == EngineType.SGLANG


def test_engine_family_sticky_with_key_cache():
    _setup_candidates()
    pool = _auto_pool([PoolMember("10.0.0.1", 8001, "Common")])
    member = pool.members[0]
    collector = _collector()

    collector._parse_prometheus_metrics(VLLM_METRICS, pool, member)
    assert member.detected_engine_type == EngineType.VLLM
    cached_keys = dict(member.metrics_key_cache)

    # Signatures absent but cached keys still parse
    minimal_text = "\n".join(
        f'{key}{{}} {1.0}' for key in cached_keys.values()
    )
    member.metrics_key_cache = cached_keys
    engine = collector._resolve_prometheus_engine_type(minimal_text, member, pool)
    assert engine == EngineType.VLLM


def test_member_cache_preserved_on_pool_update():
    _setup_candidates()
    pool = _auto_pool([PoolMember("10.0.0.1", 8001, "Common")])
    member = pool.members[0]
    collector = _collector()
    collector._parse_prometheus_metrics(VLLM_METRICS, pool, member)

    new_member = PoolMember("10.0.0.1", 8001, "Common")
    pool.update_members_smartly([new_member])

    assert pool.members[0].detected_engine_type == EngineType.VLLM
    assert pool.members[0].metrics_key_cache == member.metrics_key_cache
    assert pool.members[0].detected_variant == "vllm"


def test_detection_status_values():
    _setup_candidates()
    pool = _auto_pool([PoolMember("10.0.0.1", 8001, "Common")])
    member = pool.members[0]
    collector = _collector()

    collector._parse_prometheus_metrics(VLLM_METRICS, pool, member)
    member.metrics = collector._parse_prometheus_metrics(VLLM_METRICS, pool, member)
    member.detection_status = compute_member_detection_status(member, pool)
    assert member.detection_status == DETECTION_STATUS_OK

    member.metrics = {"waiting_queue": 1.0}
    assert compute_member_detection_status(member, pool) == DETECTION_STATUS_PARTIAL

    member.metrics = {}
    member.detected_engine_type = None
    assert compute_member_detection_status(member, pool) == DETECTION_STATUS_FAILED


def test_get_effective_engine_type():
    _setup_candidates()
    member = PoolMember("10.0.0.1", 8001, "Common")
    member.detected_engine_type = EngineType.SGLANG

    auto_pool = _auto_pool([member])
    vllm_pool = Pool("p", "Common", EngineType.VLLM, [member])

    assert get_effective_engine_type(member, auto_pool) == EngineType.SGLANG
    assert get_effective_engine_type(member, vllm_pool) == EngineType.VLLM


def test_scheduler_status_includes_detection_fields():
    _setup_candidates()
    POOLS.clear()
    vllm_member = PoolMember("10.0.0.1", 8001, "Common")
    sglang_member = PoolMember("10.0.0.2", 8010, "Common")
    pool = _auto_pool([vllm_member, sglang_member])
    collector = _collector()
    vllm_member.metrics = collector._parse_prometheus_metrics(VLLM_METRICS, pool, vllm_member)
    sglang_member.metrics = collector._parse_prometheus_metrics(SGLANG_METRICS, pool, sglang_member)
    vllm_member.detection_status = compute_member_detection_status(vllm_member, pool)
    sglang_member.detection_status = compute_member_detection_status(sglang_member, pool)
    add_or_update_pool(pool)

    from core.scheduler import Scheduler
    status = Scheduler().get_pool_status("mixed-pool", "Common")

    assert status is not None
    by_ip = {m["ip"]: m for m in status["members"]}
    assert by_ip["10.0.0.1"]["detected_engine_type"] == "vllm"
    assert by_ip["10.0.0.2"]["detected_engine_type"] == "sglang"
    assert by_ip["10.0.0.1"]["detection_status"] == DETECTION_STATUS_OK
    assert by_ip["10.0.0.2"]["detection_status"] == DETECTION_STATUS_OK

    POOLS.clear()


def test_partial_metrics_warning_path():
    _setup_candidates()
    pool = _auto_pool([PoolMember("10.0.0.1", 8001, "Common")])
    member = pool.members[0]
    collector = _collector()

    metrics = collector._parse_prometheus_metrics(VLLM_PARTIAL_METRICS, pool, member)

    assert member.detected_engine_type == EngineType.VLLM
    assert "waiting_queue" in metrics
    assert "cache_usage" not in metrics
    member.metrics = metrics
    assert compute_member_detection_status(member, pool) == DETECTION_STATUS_PARTIAL


def test_clear_cache_on_config_reload():
    _setup_candidates()
    pool = _auto_pool([PoolMember("10.0.0.1", 8001, "Common")])
    member = pool.members[0]
    collector = _collector()
    collector._parse_prometheus_metrics(VLLM_METRICS, pool, member)
    add_or_update_pool(pool)

    assert member.detected_engine_type == EngineType.VLLM
    clear_all_pools_metrics_key_cache()
    assert member.detected_engine_type is None
    assert member.metrics_key_cache == {}

    POOLS.clear()


def run_all_tests():
    tests = [
        test_config_loader_accepts_auto,
        test_detect_vllm_in_auto_pool,
        test_detect_sglang_in_auto_pool,
        test_mixed_pool_members_independent_detection,
        test_vllm_mindie_variant_via_user_config,
        test_unknown_metrics_detection_failed,
        test_homogeneous_vllm_pool_unchanged,
        test_cross_engine_fallback_misconfigured_pool,
        test_engine_family_sticky_with_key_cache,
        test_member_cache_preserved_on_pool_update,
        test_detection_status_values,
        test_get_effective_engine_type,
        test_scheduler_status_includes_detection_fields,
        test_partial_metrics_warning_path,
        test_clear_cache_on_config_reload,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {test_fn.__name__}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
