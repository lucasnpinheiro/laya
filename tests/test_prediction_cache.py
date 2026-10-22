"""Fast tier: the prediction memo used by the test client (no server, no model)."""

from __future__ import annotations

from prediction_cache import PredictionCache, make_key


def test_key_ignores_dict_ordering():
    a = make_key("laya_predict", {"state": "x", "questions": {"a": 1, "b": 2}}, salt="s")
    b = make_key("laya_predict", {"questions": {"b": 2, "a": 1}, "state": "x"}, salt="s")
    assert a == b


def test_key_changes_with_input_tool_and_salt():
    base = make_key("laya_predict", {"state": "x"}, salt="s")
    assert base != make_key("laya_predict", {"state": "y"}, salt="s")
    assert base != make_key("laya_classify", {"state": "x"}, salt="s")
    assert base != make_key("laya_predict", {"state": "x"}, salt="other-checkpoint")


def test_memory_cache_computes_once_per_key():
    cache = PredictionCache()
    calls = []

    def compute():
        calls.append(1)
        return {"v": len(calls)}

    first = cache.get_or_compute("k", compute)
    second = cache.get_or_compute("k", compute)
    assert first == second == {"v": 1}
    assert len(calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)


def test_cached_value_is_isolated_from_caller_mutation():
    cache = PredictionCache()
    got = cache.get_or_compute("k", lambda: {"answers": {"q": 1}})
    got["answers"]["q"] = 99
    assert cache.get_or_compute("k", lambda: {"answers": {"q": 2}}) == {"answers": {"q": 1}}


def test_disk_cache_is_opt_in_and_survives_a_new_instance(tmp_path):
    first = PredictionCache(disk_dir=tmp_path)
    first.get_or_compute("k", lambda: {"v": 1})

    second = PredictionCache(disk_dir=tmp_path)
    assert second.get_or_compute("k", lambda: {"v": 2}) == {"v": 1}
    assert second.hits == 1

    no_disk = PredictionCache()
    assert no_disk.get_or_compute("k", lambda: {"v": 3}) == {"v": 3}


def test_corrupt_disk_entry_is_recomputed(tmp_path):
    cache = PredictionCache(disk_dir=tmp_path)
    cache.get_or_compute("k", lambda: {"v": 1})
    for f in tmp_path.iterdir():
        f.write_text("{not json")
    fresh = PredictionCache(disk_dir=tmp_path)
    assert fresh.get_or_compute("k", lambda: {"v": 2}) == {"v": 2}
