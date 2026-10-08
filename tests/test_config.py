"""Tests for the per-user configuration system (settings + constants overlay)."""
import json
import os
import subprocess
import sys

from midas_gui import settings


# ── settings unit tests (no Qt) ──────────────────────────────────────────────
def test_user_config_path_per_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg_home_test")
    p = settings.user_config_path()
    # active profile's file, under <config dir>/profiles/
    assert p.name == "Default.json" and p.parent.name == "profiles"
    assert str(p).startswith("/tmp/xdg_home_test")


def test_bundled_beamline_profiles_seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    names = settings.list_profiles()
    assert set(names) == {"Default", "20-ID-D", "20-ID-E", "1-ID-E", "17-BM", "S3IDC"}
    assert settings.active_profile() == "Default"  # unseeded fresh install stays on Default

    d_names = [d["name"] for d in settings.read_json(settings.profile_path("20-ID-D"))["devices"]]
    assert d_names == ["20iddNF", "s20idPil", "pg4", "20iddTomo", "20iddFF", "Sim Detector"]

    e_devices = {d["name"]: d["prefix"]
                 for d in settings.read_json(settings.profile_path("20-ID-E"))["devices"]}
    assert e_devices == {
        "pimega": "PITEC:D:RAD1_5Mh:", "spl1": "20idsp1:", "s20varex2": "20idVarex2:",
        "pg6": "20idPG6s:", "gh2": "20idGH2S:", "Sim Detector": "midasSim:",
    }

    one_devices = {d["name"]: d["prefix"]
                   for d in settings.read_json(settings.profile_path("1-ID-E"))["devices"]}
    assert one_devices == {
        "ge1": "GE1:", "ge2": "GE2:", "ge3": "GE3:", "ge4": "GE4:", "ge5": "GE5:",
        "pixirad": "s1_pixirad2:", "gh1": "1idGH1:", "pg1": "1idPG1:", "pg5": "1idSP5:",
        "s1varex1": "1idVarex1:", "Sim Detector": "midasSim:",
    }

    s3idc_devices = {d["name"]: d["prefix"]
                     for d in settings.read_json(settings.profile_path("S3IDC"))["devices"]}
    assert s3idc_devices == {"eiger2": "dp_eiger_sn:", "Sim Detector": "midasSim:"}

    # A bundled profile a user deletes must not be silently resurrected.
    settings.delete_profile("20-ID-E")
    assert "20-ID-E" not in settings.list_profiles()


def test_save_reload_reset_roundtrip(tmp_path, monkeypatch):
    cfgfile = tmp_path / "midas_gui" / "config.json"
    monkeypatch.setattr(settings, "user_config_path", lambda: cfgfile)
    settings.save_user_config({"geometry": {"wavelength_A": 0.271}})
    assert cfgfile.is_file()
    assert settings.reload()["geometry"]["wavelength_A"] == 0.271
    # reset writes an empty overlay rather than deleting the profile's file, so
    # the profile itself stays discoverable via list_profiles().
    settings.reset_user_config()
    assert cfgfile.is_file()
    assert settings.reload() == {}


# ── overlay in a fresh interpreter (constants applies overlay at import) ──────
def _env_with_config_home(tmp_path):
    """Env that redirects the per-user config dir to a temp location on all OSes."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["HOME"] = str(tmp_path)              # macOS: ~/Library/Application Support
    env["XDG_CONFIG_HOME"] = str(tmp_path / ".config")   # Linux
    env["APPDATA"] = str(tmp_path / "AppData")           # Windows
    return env


def _write_user_config(env, cfg):
    """Compute the user path under the temp env and write the config there."""
    snippet = ("import json,sys;from midas_gui import settings as s;"
               "p=s.user_config_path();p.parent.mkdir(parents=True,exist_ok=True);"
               "p.write_text(sys.stdin.read());print(p)")
    r = subprocess.run([sys.executable, "-c", snippet], input=json.dumps(cfg),
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_overlay_applies_and_replaces(tmp_path):
    env = _env_with_config_home(tmp_path)
    _write_user_config(env, {
        "geometry": {"wavelength_A": 0.222},
        "materials": {"OnlyPhase": {"a": 3.6, "b": 3.6, "c": 3.6,
                                    "alpha": 90, "beta": 90, "gamma": 90, "sg": 225}},
        "ui": {"integration_kernel": "hard", "calibration_pipeline": "four_stage"},
    })
    check = (
        "import midas_gui.constants as c;"
        "assert abs(c.DEFAULT_WAVELENGTH-0.222)<1e-9, c.DEFAULT_WAVELENGTH;"
        # Lattice materials are REPLACED, not merged: none of the shipped
        # crystalline entries survive a config that lists only OnlyPhase.
        "assert [n for n,m in c.MATERIALS.items() if m.get('kind')!='dspacing']"
        "       ==['OnlyPhase'], list(c.MATERIALS);"
        # ...but a shipped "dspacing" material is restored rather than masked.
        # The Preferences table cannot show or delete one, so its absence from
        # a saved config only ever means the config predates it.
        "assert 'AgBH (silver behenate)' in c.MATERIALS, list(c.MATERIALS);"
        "assert c.DEFAULT_KERNEL=='hard';"
        "assert c.DEFAULT_PIPELINE=='four_stage';"
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_malformed_config_does_not_break(tmp_path):
    env = _env_with_config_home(tmp_path)
    # write invalid JSON directly at the computed path
    snippet = ("from midas_gui import settings as s;"
               "p=s.user_config_path();p.parent.mkdir(parents=True,exist_ok=True);"
               "p.write_text('{ not valid json ,,,');print(p)")
    r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    check = ("import midas_gui.constants as c;"
             "assert c.DEFAULT_WAVELENGTH==0.39, c.DEFAULT_WAVELENGTH;print('ok')")
    r = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
