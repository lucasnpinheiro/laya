"""Seleção dos tiers (fast / model / bench / slow) pelo conftest, sem tocar o servidor.

Roda o conftest real num projeto temporário (pytester) com testes de mentira, um por tier.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

HERE = Path(__file__).parent

SAMPLE = '''
import pytest

def test_plain(): pass

@pytest.mark.model
def test_model(): pass

@pytest.mark.bench
def test_bench(): pass

@pytest.mark.slow
def test_slow(): pass
'''


@pytest.fixture
def tier_run(pytester):
    shutil.copy(HERE / "conftest.py", pytester.path / "conftest.py")
    shutil.copy(HERE / "prediction_cache.py", pytester.path / "prediction_cache.py")
    pytester.makepyfile(test_sample=SAMPLE)

    def run(*flags):
        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-v", *flags)
        ran = {line.split("::")[1].split()[0] for line in result.outlines
               if "::" in line and " PASSED" in line}
        return ran

    return run


def test_default_runs_only_fast(tier_run):
    assert tier_run() == {"test_plain"}


def test_model_flag_excludes_bench(tier_run):
    assert tier_run("--model") == {"test_plain", "test_model"}


def test_bench_flag_adds_bench_only(tier_run):
    assert tier_run("--bench") == {"test_plain", "test_bench"}


def test_all_is_fast_model_bench_without_slow(tier_run):
    assert tier_run("--all") == {"test_plain", "test_model", "test_bench"}


def test_run_slow_does_not_pull_bench(tier_run):
    assert "test_bench" not in tier_run("--run-slow")
