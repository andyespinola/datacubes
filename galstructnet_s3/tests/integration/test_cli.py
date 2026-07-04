"""Smoke del CLI (specs/60): train Etapa 2 y Etapa 1 sobre la fixture."""
import json
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("yaml")

from galstructnet_s3 import cli


@pytest.fixture()
def project(synthetic_root, tmp_path):
    """Directorio con entries + splits + config minima."""
    root = tmp_path / "entries"
    root.mkdir()
    name = "dataset_entry_SYNTH-0000_v0.h5"
    shutil.copy(synthetic_root / name, root / name)
    sp = root / "splits"
    sp.mkdir()
    (sp / "train.txt").write_text("SYNTH-0000\n")
    (sp / "val.txt").write_text("SYNTH-0000\n")

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
data:
  root: {root}
  num_workers: 0
model:
  spatial: trivial
  physical: std
  fusion: concat
  decoder: trivial
  head: evidence
  spectral: {{backbone: trivial}}
loss:
  kappa: 0.25
  n_eff_cap: none
  weights: {{seg: 1.0, dice: 0.5, phys: 0.1}}
training:
  stage: 2
  epochs: 1
  batch_size: 1
  lr: 1.0e-3
  precision: fp32
  seed: 0
""")
    return cfg, tmp_path


def test_cli_train_stage2(project, capsys):
    cfg, tmp = project
    cli.main(["train", "--config", str(cfg),
              "--ckpt-dir", str(tmp / "ck")])
    out = json.loads(capsys.readouterr().out)
    assert "val/total" in out and "val/rho_S_neff" in out
    assert (tmp / "ck" / "last.pt").exists()


def test_cli_train_stage1(project, capsys):
    cfg, tmp = project
    text = cfg.read_text().replace("backbone: trivial", "backbone: conv1d")
    cfg.write_text(text)
    cli.main(["train", "--config", str(cfg), "--stage", "1",
              "--ckpt-dir", str(tmp / "ck1")])
    out = json.loads(capsys.readouterr().out)
    assert "val/mae" in out
    assert (tmp / "ck1" / "stage1_last.pt").exists()


def test_cli_ablate_ladder(project, capsys, tmp_path):
    """`ablate --ladder epn` corre una mini-escalera con un comando."""
    cfg, tmp = project
    ladder = tmp_path / "ladder"
    ladder.mkdir()
    (ladder / "L0_std.yaml").write_text(
        f"_base_: {cfg}\nmodel.head: std\n")
    (ladder / "L1_evidence.yaml").write_text(
        f"_base_: {cfg}\nmodel.head: evidence\nmodel.physical: normconv\n")
    cli.main(["ablate", "--configs-dir", str(ladder),
              "--ckpt-dir", str(tmp / "abl")])
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"L0_std", "L1_evidence"}
    for m in out.values():
        assert "val/rho_S_neff" in m
    assert (tmp / "abl" / "ladder_results.json").exists()
