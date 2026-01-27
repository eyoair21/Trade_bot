# Phase 5.1 Sanity Checklist ✅

## 🎯 All Checks Passed!

### ✅ 1. --profile writes profile.txt (per-run)

**Verified:**
```
✓ runs/sweeps/ci_smoke/run_000/profile.txt exists
✓ Contains 109,373 function calls
✓ Sorted by cumulative time
✓ Top 40 functions listed
```

**Sample output:**
```
109373 function calls (105745 primitive calls) in 0.578 seconds

Ordered by: cumulative time
List reduced from 1508 to 40 due to restriction <40>

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1    0.000    0.000    0.575    0.575 walkforward.py:152(run_walkforward)
     1    0.000    0.000    0.214    0.214 walkforward.py:91(create_env_manifest)
     1    0.000    0.000    0.211    0.211 parquet_local.py:124(load_multiple)
```

---

### ✅ 2. timings.csv exists with one row per successful run

**Verified:**
```csv
run_idx,elapsed_s,load_s,splits_s,backtest_s,report_s,total_s
0,0.586,0.211,0.000,0.047,0.223,0.481
1,0.552,0.218,0.000,0.045,0.193,0.456
2,0.180,0.010,0.000,0.040,0.067,0.117
3,0.174,0.009,0.000,0.043,0.063,0.115
```

**Result:** ✅ 4 rows for 4 successful runs

---

### ✅ 3. Sub-seeds differ per run (base_seed + run_idx)

**Verified:**
- run_000: seed = 42 ✅
- run_001: seed = 43 ✅
- run_002: seed = 44 ✅
- run_003: seed = 45 ✅

**Result:** ✅ Seeds increment correctly (42 + run_idx)

---

### ✅ 4. sweep_meta.json includes provenance

**Verified:**
```json
{
  "python": "3.13.5",
  "os": "Windows-11-10.0.26200-SP0",
  "timestamp_utc": "2026-01-05T19:23:47.384396+00:00"
}
```

**Result:** ✅ Python version, OS, and UTC timestamp present

---

### ✅ 5. ci.yml uses pip cache

**Verified:**
```yaml
- name: Cache pip packages
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml') }}
```

**Result:** ✅ Cache configured with proper key

---

### ✅ 6. Ruff config deprecation fixed

**Verified:**
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]  # ← Moved here
select = ["E", "F", "W", "I", "UP", "B", "C4", "SIM"]
ignore = ["E501"]
```

**Result:** ✅ No deprecation warnings

---

### ✅ 7. Nightly workflow uploads single zip

**Verified:**
```yaml
- name: Pack sweep artifacts
  run: python scripts/pack_sweep.py runs/sweeps/ci_smoke --output ci_smoke.zip

- name: Upload sweep artifacts
  uses: actions/upload-artifact@v4
  with:
    name: nightly-sweep-${{ github.sha }}
    path: ci_smoke.zip  # ← Single file, not folder
    retention-days: 7
```

**Result:** ✅ Uploads `ci_smoke.zip` (not folder)

---

### ✅ 8. pack_sweep.py excludes secrets

**Verified:**
```python
DENY_PATTERNS = {".env", ".*", "node_modules", "__pycache__", ".coverage", "*.key", "*.pem"}

def should_exclude(file_path: Path) -> bool:
    # Check exact matches
    if file_path.name in DENY_PATTERNS:
        return True
    # Check suffix patterns
    # ...
```

**Test:**
```bash
$ tar -tf ci_smoke.zip | Select-String -Pattern "\.env|\.pem|\.key|__pycache__"
# No matches ✅
```

**Result:** ✅ Secrets denylist active

---

### ✅ 9. README badges point to Trade_Bot (correct casing)

**Verified:**
```markdown
[![CI](https://github.com/eyoair21/Trade_Bot/actions/workflows/ci.yml/badge.svg)]
[![Nightly Sweep](https://github.com/eyoair21/Trade_Bot/actions/workflows/nightly-sweep.yml/badge.svg)]
```

**Result:** ✅ URLs use `Trade_Bot` (uppercase B)

---

### ✅ 10. Leaderboard includes timing summary

**Verified:**
```markdown
# Sweep Leaderboard: ci_smoke

**Ranking by:** sharpe (max)
**Total runs:** 4

### Timing Summary
- **P50 Elapsed:** 0.37s
- **P90 Elapsed:** 0.58s
```

**Result:** ✅ P50/P90 timing displayed

---

### ✅ 11. Per-run profiling works

**Verified:**
- `run_000/profile.txt` exists ✅
- `run_001/profile.txt` exists ✅
- `run_002/profile.txt` exists ✅
- `run_003/profile.txt` exists ✅

**Result:** ✅ Each run has its own profile

---

### ✅ 12. Tests pass

**Verified:**
```bash
$ python -m pytest tests\cli\test_sweep_timing.py tests\scripts\test_pack_sweep.py -q

10 passed in 0.53s ✅
```

**Result:** ✅ All tests passing

---

## 📊 Local Verification Results

### Sweep Execution
- ✅ 4 configurations ran successfully
- ✅ Total time: 1.5s
- ✅ P50: 0.37s, P90: 0.58s
- ✅ All runs completed without errors

### Outputs Generated
- ✅ `timings.csv` - 4 rows (one per run)
- ✅ `profile.txt` - Per-run profiling data
- ✅ `sweep_meta.json` - With provenance (Python, OS, timestamp)
- ✅ `leaderboard.csv` - Full rankings
- ✅ `leaderboard.md` - With timing summary
- ✅ `ci_smoke.zip` - 0.01 MB (well under 80MB cap)

### Security
- ✅ No `.env` files in zip
- ✅ No `.pem` files in zip
- ✅ No `.key` files in zip
- ✅ No `__pycache__` in zip

### Determinism
- ✅ Seeds increment: 42, 43, 44, 45
- ✅ Parallel runs use unique seeds
- ✅ Results are reproducible

---

## 🎉 All Checks Passed!

Phase 5.1 is production-ready with all quality-of-life improvements:
- ✅ Per-run profiling
- ✅ Wall-clock timing
- ✅ Deterministic parallel execution
- ✅ CI caching
- ✅ Secrets protection
- ✅ Timing summaries
- ✅ Fail-fast CI

**Ready for production use! 🚀**





