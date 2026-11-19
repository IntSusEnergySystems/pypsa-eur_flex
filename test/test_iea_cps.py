# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""IEA WEO 2019 Current Policies helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest
import yaml

from scripts.build_iea_cps_capacity_limits import build_limits
from scripts.co2_budget import national_co2_fraction
from scripts.nuclear_fleet import apply_nuclear_capacity, nuclear_capacity_by_country

ROOT = Path(__file__).resolve().parents[1]


def _powerplants_csv(path) -> str:
    plants = pd.DataFrame(
        {
            "Fueltype": ["Nuclear", "Nuclear", "Nuclear", "Nuclear", "Hard Coal"],
            "Country": ["BE", "CZ", "CZ", "FR", "CZ"],
            "Capacity": [1090.0, 2000.0, 2164.0, 1000.0, 400.0],
            "DateIn": [1985, 1985, 2002, 1980, 2000],
            "DateRetrofit": [np.nan, np.nan, np.nan, 2005, np.nan],
            "DateOut": [np.nan, 2037, np.nan, np.nan, 2040],
            "Name": ["Doel 4", "Dukovany", "Temelin", "retrofit", "coal"],
        }
    )
    plants.to_csv(path, index=False)
    return path.as_posix()


def test_nuclear_capacity_follows_decommissioning_date(tmp_path):
    """Published and filled retirement dates decide which units stay online."""
    path = _powerplants_csv(tmp_path / "powerplants.csv")
    online_2030 = nuclear_capacity_by_country(path, 2030)
    online_2040 = nuclear_capacity_by_country(path, 2040)
    online_2050 = nuclear_capacity_by_country(path, 2050)
    assert online_2030["BE"] == 1090
    assert "BE" not in online_2040.index
    assert "BE" not in online_2050.index
    assert online_2030["CZ"] == 4164
    assert online_2040["CZ"] == 2164
    assert "CZ" not in online_2050.index
    assert "FR" not in online_2030.index


def test_apply_nuclear_capacity_carries_new_build(tmp_path):
    """The surviving fleet is a lower bound and earlier new build is kept."""
    path = _powerplants_csv(tmp_path / "powerplants.csv")
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=1, freq="h"))
    n.add("Bus", ["BE", "CZ"], country=["BE", "CZ"])
    n.add(
        "Generator",
        ["BE nuclear", "CZ nuclear", "CZ nuclear-2037"],
        bus=["BE", "CZ", "CZ"],
        carrier="nuclear",
        p_nom=[4096, 2164, 2000],
        p_nom_min=[4096, 2164, 2000],
        p_nom_extendable=False,
        capital_cost=100,
    )

    apply_nuclear_capacity(n, path, 2030)
    nuclear = n.generators.query("carrier == 'nuclear'")
    assert set(nuclear.index) == {"BE nuclear", "CZ nuclear"}
    assert nuclear.loc["BE nuclear", "p_nom_min"] == 1090
    assert bool(nuclear.loc["BE nuclear", "p_nom_extendable"])
    assert np.isinf(nuclear.loc["BE nuclear", "p_nom_max"])

    previous = n.copy()
    previous.generators["p_nom_opt"] = previous.generators["p_nom"]
    previous.generators.loc["BE nuclear", "p_nom_opt"] = 2500
    apply_nuclear_capacity(n, path, 2040, n_previous=previous)
    assert n.generators.loc["BE nuclear", "p_nom_min"] == 1410
    assert n.generators.loc["CZ nuclear", "p_nom_min"] == 2164
    assert "CZ nuclear-2037" not in n.generators.index


def test_apply_nuclear_capacity_stays_on_the_cps_share(tmp_path):
    """An EU Current Policies share replaces the surviving fleet and new build."""
    path = _powerplants_csv(tmp_path / "powerplants.csv")
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2040-01-01", periods=1, freq="h"))
    n.add("Bus", ["BE", "CZ", "CH"], country=["BE", "CZ", "CH"])
    n.add(
        "Generator",
        ["BE nuclear", "CZ nuclear", "CH nuclear"],
        bus=["BE", "CZ", "CH"],
        carrier="nuclear",
        p_nom=[1090, 2164, 3000],
        p_nom_min=[1090, 2164, 3000],
        p_nom_extendable=False,
        capital_cost=100,
    )
    previous = n.copy()
    previous.generators["p_nom_opt"] = previous.generators["p_nom"] + 500
    apply_nuclear_capacity(
        n,
        path,
        2040,
        n_previous=previous,
        cps_max=pd.Series({"BE": 500.0, "CZ": 2000.0}),
    )
    # Belgium has no unit left in 2040, so the share is new capacity.
    assert n.generators.loc["BE nuclear", "p_nom"] == 0
    assert n.generators.loc["BE nuclear", "p_nom_min"] == 500
    assert n.generators.loc["BE nuclear", "p_nom_max"] == 500
    # Czech units still online exceed the share, so the share is already built.
    assert n.generators.loc["CZ nuclear", "p_nom"] == 2000
    assert n.generators.loc["CZ nuclear", "p_nom_min"] == 2000
    assert n.generators.loc["CZ nuclear", "p_nom_max"] == 2000
    # Switzerland is outside the EU table, so earlier new build is kept and
    # further new build stays open. No Swiss unit is in the plant file.
    assert n.generators.loc["CH nuclear", "p_nom_min"] == 500
    assert np.isinf(n.generators.loc["CH nuclear", "p_nom_max"])


def test_national_co2_fraction_uses_the_2019_cps_index():
    """2030 and 2040 are the published EU totals. Later years keep that ratio."""
    iea = {2018: 3083, 2030: 2661, 2035: 2488.4, 2040: 2327, 2045: 2176.1, 2050: 2034.9}
    assert national_co2_fraction(iea, 2030, 2018) == 2661 / 3083
    assert national_co2_fraction(iea, 2040, 2018) == 2327 / 3083
    ratio = 2327 / 2661
    assert national_co2_fraction(iea, 2050, 2018) == pytest.approx(2034.9 / 3083)
    assert abs((2661 * ratio**0.5) - 2488.4) < 0.1
    assert abs((2327 * ratio) - 2034.9) < 0.1


def test_capacity_file_matches_eu_cps_totals():
    """Country shares add back to the published EU electrical capacity."""
    table = pd.read_csv(
        ROOT / "data" / "agg_p_nom_minmax.csv", index_col=[0, 1], header=[0, 1]
    )
    totals = table.xs("max", level=1, axis=1).groupby(level=1).sum() / 1000
    assert abs(totals.loc["onwind", "2030"] + totals.loc["offwind-all", "2030"] - 286) < 0.05
    assert abs(totals.loc["onwind", "2040"] + totals.loc["offwind-all", "2040"] - 321) < 0.05
    assert abs(totals.loc["solar-all", "2030"] - 213) < 0.05
    assert abs(totals.loc["solar-all", "2040"] - 240) < 0.05
    assert abs(totals.loc["gas", "2030"] - 304) < 0.05
    assert abs(totals.loc["gas", "2040"] - 345) < 0.05
    assert abs(totals.loc["coal-all", "2030"] - 112) < 0.05
    assert abs(totals.loc["coal-all", "2040"] - 66) < 0.05
    assert abs(totals.loc["oil", "2030"] - 20) < 0.05
    assert abs(totals.loc["oil", "2040"] - 13) < 0.05
    assert abs(totals.loc["nuclear", "2030"] - 97) < 0.05
    assert abs(totals.loc["nuclear", "2040"] - 92) < 0.05
    # Coal and oil are caps only. Wind, solar, gas and nuclear are fixed.
    assert pd.isna(table.loc[("DE", "coal-all"), ("2030", "min")])
    assert table.loc[("DE", "onwind"), ("2030", "min")] == table.loc[
        ("DE", "onwind"), ("2030", "max")
    ]
    assert table.loc[("FR", "nuclear"), ("2030", "min")] == table.loc[
        ("FR", "nuclear"), ("2030", "max")
    ]
    # German units in the database are already closed, so the 2030 share is zero.
    assert table.loc[("DE", "nuclear"), ("2030", "max")] == 0


def test_build_limits_splits_a_total_by_country_share():
    plants = pd.DataFrame(
        {
            "Fueltype": ["Wind", "Wind", "Solar", "Natural Gas", "Hard Coal", "Oil"],
            "Technology": ["Onshore", "Offshore", "PV", "CCGT", "Steam Turbine", "Steam Turbine"],
            "Country": ["Germany"] * 6,
            "Capacity": [10, 10, 10, 10, 10, 10],
        }
    )
    frame = build_limits(plants)
    # One country holds the whole EU total when it is the only one with capacity,
    # and the other EU countries are capped at zero.
    assert frame.loc[("DE", "onwind"), (2030, "max")] == frame.xs(
        "onwind", level=1
    )[(2030, "max")].sum()


def test_eu_co2_cap_is_one_total(tmp_path):
    """The EU setting writes one cap and leaves countries outside the EU out."""
    pytest.importorskip("pandera")
    from scripts.solve_network import add_national_co2_constraints

    reference = pd.DataFrame(
        {"electricity": [10.0, 20.0, 7.0]},
        index=pd.Index(["DE", "FR", "NO"], name="country"),
    )
    path = tmp_path / "co2_totals_reference.csv"
    reference.to_csv(path)
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "co2 atmosphere", carrier="co2", country="")
    n.add("Bus", ["DE", "FR", "NO"], carrier="AC", country=["DE", "FR", "NO"])
    n.add(
        "Link",
        ["DE emit", "FR emit", "NO emit"],
        bus0=["DE", "FR", "NO"],
        bus1=["DE", "FR", "NO"],
        bus2="co2 atmosphere",
        carrier="gas",
        p_nom_extendable=True,
        efficiency=0.5,
        efficiency2=1.0,
    )
    n.config = {
        "countries": ["DE", "FR", "NO"],
        "sector": {
            "transport": False,
            "heating": False,
            "industry": False,
            "agriculture": False,
        },
        "co2_budget_national": {
            "aggregate": "eu",
            "countries": ["DE", "FR"],
            "reference_year": 2018,
            "iea_mt": {2018: 3083, 2030: 2661},
        },
    }

    class _Inputs:
        input = {"co2_totals_reference": path.as_posix()}

    n.optimize.create_model()
    add_national_co2_constraints(n, "2030", _Inputs())
    assert "co2_limit_eu" in n.model.constraints
    assert "co2_limit_country_DE" not in n.model.constraints
    expected = (10.0 + 20.0) * 1e6 * (2661 / 3083) * (2 / 8760)
    rhs = float(n.model.constraints["co2_limit_eu"].rhs)
    assert rhs == pytest.approx(expected)
    used = set(n.model.constraints["co2_limit_eu"].data.vars.values)
    labels = n.model.variables["Link-p"].labels.to_series()
    names = set(labels[labels.isin(used)].index.get_level_values("Link"))
    assert names == {"DE emit", "FR emit"}


def test_ccl_horizon_block_accepts_string_and_int_years():
    pytest.importorskip("pandera")
    from scripts.solve_network import _ccl_horizon_block

    table = pd.DataFrame(
        {("2030", "min"): [1.0], ("2030", "max"): [2.0]},
        index=pd.MultiIndex.from_tuples([("DE", "gas")]),
    )
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    assert _ccl_horizon_block(table, "2030")["max"].iloc[0] == 2.0
    numeric = table.copy()
    numeric.columns = pd.MultiIndex.from_tuples([(2030, "min"), (2030, "max")])
    assert _ccl_horizon_block(numeric, "2030")["max"].iloc[0] == 2.0


def test_ccl_existing_links_are_counted_in_electrical_mw():
    pytest.importorskip("pandera")
    from scripts.solve_network import _ccl_existing_mw

    n = pypsa.Network()
    n.add("Bus", "DE", country="DE")
    n.add(
        "Link",
        "DE coal",
        bus0="DE",
        bus1="DE",
        carrier="coal",
        p_nom=100,
        efficiency=0.4,
        p_nom_extendable=False,
        build_year=2010,
        lifetime=40,
    )
    existing = _ccl_existing_mw(n, "Link", {"coal": "coal-all"}, 2030)
    assert existing.loc[("DE", "coal-all")] == 40


def test_cps_config_disables_the_europe_wide_co2_cap():
    with open(ROOT / "config" / "config.yaml") as file:
        config = yaml.safe_load(file)
    assert config["foresight"] == "myopic"
    assert config["planning_horizons"] == [2030, 2035, 2040, 2045, 2050]
    assert all(value is None for value in config["co2_budget"]["upper"].values())
    assert config["co2_budget_national"]["enable"] is True
    assert config["co2_budget_national"]["aggregate"] == "eu"
    assert config["co2_budget_national"]["iea_mt"][2030] == 2661
    assert config["electricity"]["nuclear_fleet"] is True
    assert config["solving"]["constraints"]["CCL"] is True
    assert config["sector"]["co2_sequestration_potential"][2050] == 0
    assert config["sector"]["conventional_generation"]["coal"] == "coal"
