"""`scripts/diagnose_compile.py` (P4-J2 follow-up): the per-mixer, per-row drift report runs end to
end. CPU with the eager backend, so every row is exact; the H100 run is scripts/slurm/diag_compile.sh."""


def test_diagnose_compile_reports_every_case(script, capsys):
    mod = script("diagnose_compile")
    variants = ["full", "mamba3", "mlstm", "attention-sdpa"]  # no FlexAttention compile on CPU
    rc = mod.main(
        ["--device", "cpu", "--backend", "eager", "--seq-len", "32", "--dim", "64", "--variants", *variants]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "DIAGNOSE_COMPILE DONE" in out
    lines = [ln for ln in out.splitlines() if ln.startswith("CASE ")]
    assert len(lines) == len(variants) * 3
    assert all(ln.endswith(" ok") for ln in lines), lines
