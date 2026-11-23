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
    # Belgium has no unit left in 2040. The 500 MW is new, so it stays above
    # p_nom_min and keeps the generator capital cost.
    assert n.generators.loc["BE nuclear", "p_nom"] == 0
    assert n.generators.loc["BE nuclear", "p_nom_min"] == 0
    assert n.generators.loc["BE nuclear", "p_nom_max"] == 500
    assert n.generators.loc["BE nuclear", "capital_cost"] == 100
    # Czech units still online exceed the share, so no new MW is added.
    assert n.generators.loc["CZ nuclear", "p_nom"] == 2000
    assert n.generators.loc["CZ nuclear", "p_nom_min"] == 2000
    assert n.generators.loc["CZ nuclear", "p_nom_max"] == 2000
    assert n.generators.loc["CZ nuclear", "capital_cost"] == 100
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
    assert abs(totals.loc["hydro", "2030"] - 163) < 0.05
    assert abs(totals.loc["hydro", "2040"] - 168) < 0.05
    assert abs(totals.loc["biomass", "2030"] - 56) < 0.05
    assert abs(totals.loc["biomass", "2040"] - 59) < 0.05
    assert abs(totals.loc["onwind", "2018"] + totals.loc["offwind-all", "2018"] - 180) < 0.05
    assert abs(totals.loc["solar-all", "2018"] - 117) < 0.05
    assert abs(totals.loc["gas", "2018"] - 217) < 0.05
    assert abs(totals.loc["coal-all", "2018"] - 164) < 0.05
    assert abs(totals.loc["oil", "2018"] - 45) < 0.05
    assert abs(totals.loc["nuclear", "2018"] - 125) < 0.05
    assert abs(totals.loc["hydro", "2018"] - 155) < 0.05
    assert abs(totals.loc["biomass", "2018"] - 48) < 0.05
    assert abs(totals.loc["geothermal", "2018"] - 1) < 0.05
    # Every technology is fixed at the CPS total, including coal and oil.
    assert table.loc[("DE", "coal-all"), ("2030", "min")] == table.loc[
        ("DE", "coal-all"), ("2030", "max")
    ]
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


def test_iea_generation_fixes_annual_electricity():
    """A Current Policies generation total is an equality on electrical output."""
    pytest.importorskip("pandera")
    from scripts.solve_network import add_iea_generation_constraints

    n = pypsa.Network()
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    n.set_snapshots(snapshots)
    n.add("Bus", ["coal bus", "DE"], carrier=["coal", "AC"])
    n.add(
        "Link",
        "DE coal",
        bus0="coal bus",
        bus1="DE",
        carrier="coal",
        p_nom_extendable=True,
        efficiency=0.4,
    )
    n.config = {"iea_generation": {"enable": True, "coal": {"2030": 479, "2040": 293}}}
    n.optimize.create_model()
    add_iea_generation_constraints(n, "2030")
    constraint = n.model.constraints["iea_generation_coal"]
    nyears = 2 / 8760
    assert float(constraint.rhs) == pytest.approx(479 * 1e6 * nyears)
    assert float(constraint.data.coeffs[0]) == pytest.approx(0.4)


def test_biomass_electricity_skips_biogas_supply_generators():
    """A generator on a biogas bus is fuel supply, not electrical biomass."""
    pytest.importorskip("pandera")
    from scripts.solve_network import add_iea_generation_constraints

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", ["DE", "biogas"], carrier=["AC", "biogas"])
    n.add(
        "Generator",
        "DE biomass",
        bus="DE",
        carrier="biomass",
        p_nom=10,
        marginal_cost=1,
    )
    n.add(
        "Generator",
        "biogas supply",
        bus="biogas",
        carrier="biogas",
        p_nom=1.0e6,
        marginal_cost=1,
    )
    n.config = {
        "iea_generation": {"enable": True, "biomass": {"2030": 278, "2040": 303}}
    }
    n.optimize.create_model()
    add_iea_generation_constraints(n, "2030")
    labels = n.model.variables["Generator-p"].labels.to_series()
    used = set(n.model.constraints["iea_generation_biomass"].data.vars.values)
    names = set(labels[labels.isin(used)].index.get_level_values("name"))
    assert names == {"DE biomass"}


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
    assert config["planning_horizons"] == [2030, 2035, 2040]
    assert config["run"]["name"] == "iea-cps"
    with open(ROOT / "config" / "config.historic.yaml") as historic_file:
        historic = yaml.safe_load(historic_file)
    assert historic["run"]["name"] == "iea-historic"
    assert historic["planning_horizons"] == [2018]
    assert historic["run"]["shared_resources"]["policy"] == "iea-cps"
    assert all(value is None for value in config["co2_budget"]["upper"].values())
    assert config["co2_budget_national"]["enable"] is False
    assert config["iea_co2"]["enable"] is True
    assert config["iea_co2"]["coal"][2030] == 637
    assert config["iea_co2"]["transport_oil"][2030] == 831
    assert config["iea_final"]["buildings"]["heat"][2030] == 34
    assert config["iea_must_run"]["csp"]["gw"][2030] == 3
    assert config["iea_primary_check"]["oil_mbd"][2030] == 9.6
    assert config["sector"]["exclude_international_bunkers"] is True
    assert config["costs"]["emission_prices"]["co2"][2030] == pytest.approx(27 / 1.1810, abs=0.02)
    assert config["iea_generation"]["enable"] is True
    assert config["iea_generation"]["coal"][2030] == 479
    assert config["iea_generation"]["geothermal"][2030] == 9
    assert config["iea_generation"]["nuclear"][2018] == 829
    assert config["iea_co2"]["coal"][2018] == 874
    assert config["costs"]["emission_prices"]["co2"][2018] == pytest.approx(20 / 1.1810, abs=0.02)
    assert config["electricity"]["nuclear_fleet"] is True
    assert set(config["countries"]) == {
        "AT", "BE", "BG", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB",
        "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "NL", "PL", "PT",
        "RO", "SE", "SI", "SK",
    }
    assert config["solving"]["constraints"]["CCL"] is True
    assert config["sector"]["co2_sequestration_potential"][2050] == 0
    assert config["sector"]["conventional_generation"]["coal"] == "coal"


def test_2018_follows_the_weo_2019_historical_column():
    """2018 is the published historical column, not a backcast from 2030."""
    from scripts.iea_cps_constraints import industry_fuel_share_mt, published_value

    with open(ROOT / "config" / "config.yaml") as file:
        config = yaml.safe_load(file)
    generation = {
        "coal": 666,
        "oil": 58,
        "gas": 629,
        "nuclear": 829,
        "hydro": 339,
        "biomass": 225,
        "wind": 392,
        "solar": 129,
        "geothermal": 7,
    }
    for name, twh in generation.items():
        assert published_value(config["iea_generation"][name], 2018) == twh
    # Solar in 2018 is 124 TWh of PV plus 5 TWh of CSP.
    assert published_value(config["iea_must_run"]["csp"]["twh"], 2018) == 5
    assert published_value(config["iea_must_run"]["csp"]["gw"], 2018) == 2
    assert published_value(config["iea_must_run"]["marine"]["gw"], 2018) == 0
    assert published_value(config["iea_must_run"]["marine"]["twh"], 2018) == 1
    co2 = {
        "coal": 874,
        "oil": 1332,
        "gas": 878,
        "power_coal": 694,
        "power_oil": 49,
        "power_gas": 280,
        "final_coal": 147,
        "final_oil": 1197,
        "final_gas": 559,
        "transport_oil": 919,
    }
    for name, mt in co2.items():
        assert published_value(config["iea_co2"][name], 2018) == mt
    assert 694 + 49 + 280 == 1023
    final = {
        ("industry", "coal"): 21,
        ("industry", "oil"): 27,
        ("industry", "gas"): 88,
        ("industry", "electricity"): 89,
        ("transport", "oil"): 302,
        ("transport", "electricity"): 6,
        ("transport", "biofuels"): 17,
        ("buildings", "coal"): 10,
        ("buildings", "oil"): 50,
        ("buildings", "gas"): 149,
        ("buildings", "electricity"): 142,
    }
    for (sector, fuel), mtoe in final.items():
        assert published_value(config["iea_final"][sector][fuel], 2018) == mtoe
    assert config["iea_primary_check"]["oil_mbd"][2018] == 11.1
    assert config["iea_primary_check"]["gas_bcm"][2018] == 480
    assert config["iea_primary_check"]["coal_mtce"][2018] == 319
    # 2035 stays the geometric midpoint of 2030 and 2040.
    coal = config["iea_co2"]["coal"]
    assert published_value(coal, 2035) == pytest.approx(637 * (424 / 637) ** 0.5)
    assert published_value(coal, 2018) != pytest.approx(637 * (424 / 637) ** (-12 / 10))
    shares = industry_fuel_share_mt(config, 2018)
    assert shares["coal"] == pytest.approx(21 / (21 + 10) * 147)
    assert shares["oil"] == pytest.approx(27 / (27 + 50) * (1197 - 919))
    assert shares["gas"] == pytest.approx(88 / (88 + 149) * 559)
    electric = 6 * 53.19
    combustion = (302 + 17 + 3) * 16.0712
    share = electric / (electric + combustion)
    assert config["sector"]["land_transport_electric_share"][2018] == pytest.approx(share, abs=5e-5)
    assert config["sector"]["land_transport_ice_share"][2018] == pytest.approx(1 - share, abs=5e-5)
    prices = pd.read_csv(ROOT / "data" / "iea_cps_fuel_prices.csv")
    prices = prices[prices.planning_horizon.eq(2018)].set_index("technology").value
    usd_per_eur = 1.1810
    assert prices["gas"] == pytest.approx(7.6 / usd_per_eur / 0.293071, abs=0.01)
    assert prices["coal"] == pytest.approx(92 / usd_per_eur / (6000 * 4.1868 / 3600), abs=0.01)
    assert prices["oil"] == pytest.approx(68 / usd_per_eur / ((41.868 / 3.6) / 7.33), abs=0.01)


def test_marine_2018_installs_the_energy_when_capacity_prints_as_zero(tmp_path):
    from scripts.iea_cps_constraints import add_csp_and_marine

    path = _limit_file(
        tmp_path / "limits.csv",
        [("DE", "solar-all", 1000.0, 1000.0)],
        year="2018",
    )
    n = pypsa.Network()
    n.add("Bus", "DE", carrier="AC", country="DE")
    add_csp_and_marine(
        n,
        {
            "iea_must_run": {
                "enable": True,
                "marine": {"gw": {2018: 0, 2030: 1}, "twh": {2018: 1, 2030: 2}},
            },
            "solving": {"agg_p_nom_limits": {"file": path.as_posix()}},
        },
        "2018",
    )
    assert n.generators.p_nom.sum() == pytest.approx(1000.0 / 8.76)
    assert float(n.generators.p_min_pu.iloc[0]) == 1.0


def _limit_file(path, rows, year="2030"):
    index = pd.MultiIndex.from_tuples(
        [(country, carrier) for country, carrier, _, _ in rows],
        names=["country", "carrier"],
    )
    columns = pd.MultiIndex.from_tuples([(str(year), "min"), (str(year), "max")])
    frame = pd.DataFrame(
        [[minimum, maximum] for _, _, minimum, maximum in rows],
        index=index,
        columns=columns,
    )
    frame.to_csv(path)
    return path


def test_derate_brings_existing_coal_down_to_the_share(tmp_path):
    from scripts.iea_cps_constraints import derate_existing_to_cps

    path = _limit_file(tmp_path / "limits.csv", [("DE", "coal-all", 100.0, 100.0)])
    n = pypsa.Network()
    n.add("Bus", "DE", carrier="AC", country="DE")
    n.add(
        "Link",
        "DE coal",
        bus0="DE",
        bus1="DE",
        carrier="coal",
        p_nom=1000,
        p_nom_min=1000,
        p_nom_extendable=False,
        efficiency=0.4,
    )
    derate_existing_to_cps(
        n,
        {"solving": {"agg_p_nom_limits": {"file": path.as_posix(), "agg_offwind": True, "agg_solar": True}}},
        "2030",
    )
    assert n.links.loc["DE coal", "p_nom"] == pytest.approx(250.0)
    assert n.links.loc["DE coal", "p_nom"] * 0.4 == pytest.approx(100.0)


def test_derate_leaves_fuel_supply_generators_alone(tmp_path):
    from scripts.iea_cps_constraints import derate_existing_to_cps

    path = _limit_file(tmp_path / "limits.csv", [("FR", "biomass", 1000.0, 1000.0)])
    n = pypsa.Network()
    n.add("Bus", "FR biogas", carrier="biogas", country="FR")
    n.add(
        "Generator",
        "FR biogas",
        bus="FR biogas",
        carrier="biogas",
        p_nom=2.0e7,
        p_nom_extendable=False,
    )
    derate_existing_to_cps(
        n,
        {
            "solving": {
                "agg_p_nom_limits": {
                    "file": path.as_posix(),
                    "agg_offwind": True,
                    "agg_solar": True,
                }
            }
        },
        "2030",
    )
    assert n.generators.loc["FR biogas", "p_nom"] == pytest.approx(2.0e7)


def test_wind_capacity_rises_when_the_weather_year_is_short(tmp_path):
    from scripts.iea_cps_constraints import lift_weather_limited_generation

    path = _limit_file(tmp_path / "limits.csv", [("DE", "onwind", 1000.0, 1000.0)])
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.snapshot_weightings.loc[:, :] = 1.0
    n.add("Bus", "DE", carrier="AC", country="DE")
    n.add(
        "Generator",
        "DE wind",
        bus="DE",
        carrier="onwind",
        p_nom=1000,
        p_nom_extendable=True,
        p_nom_max=1000,
        p_max_pu=0.5,
        capital_cost=1,
        marginal_cost=1,
    )
    # 1000 MW at 50% over 2 hours is 1000 MWh. Ask for 2000 MWh.
    lift_weather_limited_generation(
        n,
        {
            "iea_generation": {
                "enable": True,
                "wind": {2030: 8.76, 2040: 8.76},
            },
            "solving": {"agg_p_nom_limits": {"file": path.as_posix()}},
        },
        "2030",
    )
    assert n._iea_capacity_override[("DE", "onwind")][0] == pytest.approx(1000.0)
    assert n._iea_capacity_override[("DE", "onwind")][1] == pytest.approx(2000.0, rel=1e-4)


def test_reservoir_inflow_rises_to_the_hydro_total():
    from scripts.iea_cps_constraints import lift_hydro_inflow_to_published_energy

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.snapshot_weightings.loc[:, :] = 1.0
    n.add("Bus", "DE", carrier="AC", country="DE")
    n.add("Generator", "DE ror", bus="DE", carrier="ror", p_nom=100, p_max_pu=0.5)
    n.add("StorageUnit", "DE hydro", bus="DE", carrier="hydro", p_nom=1000, max_hours=10)
    n.storage_units_t.inflow["DE hydro"] = 10.0
    # Run-of-river supplies 100 MWh. Inflow supplies 20 MWh. Ask for 200 MWh.
    lift_hydro_inflow_to_published_energy(
        n,
        {"iea_generation": {"enable": True, "hydro": {2030: 0.876, 2040: 0.876}}},
        "2030",
    )
    assert float(n.storage_units_t.inflow["DE hydro"].sum()) == pytest.approx(100.0, rel=1e-4)


def test_oil_co2_follows_the_fuel_the_plants_actually_burn():
    from scripts.iea_cps_constraints import reconcile_co2_with_fuel_use

    n = pypsa.Network()
    n.add("Bus", ["EU oil", "DE", "DE transport", "co2 atmosphere"], carrier=["oil", "AC", "land transport oil", "co2"])
    n.add(
        "Link",
        "DE oil power",
        bus0="EU oil",
        bus1="DE",
        bus2="co2 atmosphere",
        carrier="oil",
        efficiency=0.35,
        efficiency2=0.2571,
        p_nom=100,
    )
    n.add(
        "Link",
        "DE transport",
        bus0="EU oil",
        bus1="DE transport",
        bus2="co2 atmosphere",
        carrier="land transport oil",
        efficiency=1.0,
        efficiency2=0.2571,
        p_nom=100,
    )
    n.config = {
        "iea_co2": {
            "coal": {2030: 637, 2040: 424},
            "oil": {2030: 1132, 2040: 975},
            "gas": {2030: 893, 2040: 929},
            "power_coal": {2030: 483, 2040: 292},
            "power_oil": {2030: 19, 2040: 13},
            "power_gas": {2030: 294, 2040: 327},
            "final_coal": {2030: 123, 2040: 105},
            "final_oil": {2030: 1047, 2040: 900},
            "final_gas": {2030: 567, 2040: 571},
            "transport_oil": {2030: 831, 2040: 736},
        },
        "iea_generation": {"oil": {2030: 21, 2040: 14}},
        "iea_final": {
            "transport": {"oil": {2030: 274, 2040: 242}, "biofuels": {2030: 25, 2040: 27}},
            "buildings": {"oil": {2030: 0, 2040: 0}, "coal": {2030: 0, 2040: 0}, "gas": {2030: 0, 2040: 0}},
            "industry": {"oil": {2030: 0, 2040: 0}, "coal": {2030: 0, 2040: 0}, "gas": {2030: 0, 2040: 0}},
        },
    }
    targets = reconcile_co2_with_fuel_use(n, 2030)
    assert targets["power_oil"] == pytest.approx(21 * 0.2571 / 0.35, rel=1e-3)
    assert targets["transport_oil"] == pytest.approx(299 * (41.868e6 / 3.6) * 0.2571 / 1e6, rel=1e-3)


def test_industry_fuel_plus_process_emissions_meet_the_share():
    from scripts.iea_cps_constraints import (
        MWH_PER_MTOE,
        _industry_fuel_co2_mt,
        _process_co2_mt,
        industry_co2_target_mt,
        match_industry_co2_including_process_emissions,
        process_in_industry_mt,
    )

    config = {
        "iea_co2": {
            "enable": True,
            "final_coal": {2030: 123, 2040: 105},
            "final_oil": {2030: 1047, 2040: 900},
            "final_gas": {2030: 567, 2040: 571},
            "transport_oil": {2030: 831, 2040: 736},
        },
        "iea_final": {
            "enable": True,
            "industry": {
                "coal": {2030: 20, 2040: 18},
                "oil": {2030: 26, 2040: 24},
                "gas": {2030: 87, 2040: 84},
            },
            "buildings": {
                "coal": {2030: 6, 2040: 3},
                "oil": {2030: 32, 2040: 18},
                "gas": {2030: 154, 2040: 158},
            },
        },
    }
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.snapshot_weightings.loc[:, :] = 4380.0
    n.add("Bus", ["EU oil", "EU coal", "EU gas", "industry", "process", "co2 atmosphere"], carrier=["oil", "coal", "gas", "industry", "process emissions", "co2"])
    n.add("Link", "coal", bus0="EU coal", bus1="industry", bus2="co2 atmosphere", carrier="coal for industry", efficiency=1.0, efficiency2=0.3361)
    n.add("Link", "gas", bus0="EU gas", bus1="industry", bus2="co2 atmosphere", carrier="gas for industry", efficiency=1.0, efficiency2=0.198)
    n.add("Link", "naphtha", bus0="EU oil", bus1="industry", bus2="process", bus3="co2 atmosphere", carrier="naphtha for industry", efficiency=1.0, efficiency2=0.02, efficiency3=0.2365)
    n.add("Link", "process", bus0="process", bus1="co2 atmosphere", carrier="process emissions", efficiency=1.0, p_nom=50.0, p_nom_extendable=True)
    hours = float(n.snapshot_weightings.generators.sum())
    n.add("Load", "naphtha for industry", bus="industry", carrier="naphtha for industry", p_set=26 * MWH_PER_MTOE / hours)
    n.add("Load", "process emissions", bus="process", carrier="process emissions", p_set=-80e6 / hours)
    match_industry_co2_including_process_emissions(n, config, "2030")
    process = _process_co2_mt(n)
    assert process > 1.0
    assert float(n.loads.at["process emissions", "p_set"]) < 0.0
    assert float(n.links.at["naphtha", "efficiency"]) == 1.0
    assert float(n.links.at["naphtha", "efficiency2"]) == pytest.approx(0.02)
    assert float(n.links.at["process", "p_nom"]) == 50.0
    assert bool(n.links.at["process", "p_nom_extendable"])
    assert _industry_fuel_co2_mt(n, config, 2030) + process == pytest.approx(
        industry_co2_target_mt(config, 2030), abs=0.05
    )
    n.config = config
    assert sum(process_in_industry_mt(n, 2030).values()) == pytest.approx(process, abs=0.05)
    assert float(n.links.at["coal", "efficiency"]) == 1.0


def test_csp_is_must_run_without_cost(tmp_path):
    from scripts.iea_cps_constraints import add_csp_and_marine

    path = _limit_file(
        tmp_path / "limits.csv",
        [("DE", "solar-all", 3000.0, 3000.0), ("FR", "solar-all", 1000.0, 1000.0)],
    )
    n = pypsa.Network()
    n.add("Bus", ["DE", "FR"], carrier="AC", country=["DE", "FR"])
    add_csp_and_marine(
        n,
        {
            "iea_must_run": {
                "enable": True,
                "csp": {"gw": {2030: 3, 2040: 6}, "twh": {2030: 8, 2040: 16}},
            },
            "solving": {"agg_p_nom_limits": {"file": path.as_posix()}},
        },
        "2030",
    )
    assert n.generators.p_nom.sum() == pytest.approx(3000.0)
    assert n.generators.capital_cost.sum() == 0
    assert float(n.generators.p_min_pu.iloc[0]) == pytest.approx(8 / (3 * 8.76))


def test_combustion_co2_uses_published_tonnes_and_skips_process():
    pytest.importorskip("pandera")
    from scripts.iea_cps_constraints import add_iea_combustion_co2_constraints

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add("Bus", ["DE", "heat", "EU oil", "process"], carrier=["AC", "heat", "oil", "process emissions"])
    n.add(
        "Link",
        "DE gas",
        bus0="DE",
        bus1="DE",
        bus2="co2 atmosphere",
        carrier="OCGT",
        p_nom_extendable=True,
        marginal_cost=1.0,
        efficiency=0.5,
        efficiency2=1.0,
    )
    n.add(
        "Link",
        "DE boiler",
        bus0="DE",
        bus1="heat",
        bus2="co2 atmosphere",
        carrier="residential rural gas boiler",
        p_nom_extendable=True,
        efficiency=0.9,
        efficiency2=1.0,
    )
    n.add(
        "Link",
        "EU oil refining",
        bus0="EU oil",
        bus1="EU oil",
        bus2="co2 atmosphere",
        carrier="oil refining",
        p_nom_extendable=True,
        efficiency=1.0,
        efficiency2=1.0,
    )
    n.add(
        "Link",
        "process",
        bus0="process",
        bus1="co2 atmosphere",
        carrier="process emissions",
        p_nom_extendable=True,
        efficiency=1.0,
    )
    n.config = {
        "iea_co2": {
            "enable": True,
            "gas": {2030: 893, 2040: 929},
            "power_gas": {2030: 294, 2040: 327},
            "oil": {2030: 1132, 2040: 975},
        }
    }
    n.optimize.create_model()
    add_iea_combustion_co2_constraints(n, "2030")
    nyears = 2 / 8760
    assert float(n.model.constraints["iea_co2_gas"].rhs) == pytest.approx(893 * 1e6 * nyears)
    assert float(n.model.constraints["iea_co2_oil"].rhs) == pytest.approx(1132 * 1e6 * nyears)
    gas_vars = set(n.model.constraints["iea_co2_gas"].data.vars.values)
    power_vars = set(n.model.constraints["iea_co2_power_gas"].data.vars.values)
    labels = n.model.variables["Link-p"].labels.to_series()
    gas_names = set(labels[labels.isin(gas_vars)].index.get_level_values("Link"))
    power_names = set(labels[labels.isin(power_vars)].index.get_level_values("Link"))
    assert gas_names == {"DE gas", "DE boiler"}
    assert power_names == {"DE gas"}
    oil_vars = set(n.model.constraints["iea_co2_oil"].data.vars.values)
    oil_names = set(labels[labels.isin(oil_vars)].index.get_level_values("Link"))
    assert oil_names == {"EU oil refining"}
    assert "iea_co2_process" not in n.model.constraints


def test_process_link_is_counted_in_the_fuel_co2_rows():
    pytest.importorskip("pandera")
    from scripts.iea_cps_constraints import add_iea_combustion_co2_constraints

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add(
        "Bus",
        ["oil", "naphtha", "co2", "process", "coal", "industry"],
        carrier=["oil", "naphtha for industry", "co2", "process emissions", "coal", "industry"],
    )
    n.add("Link", "naphtha", bus0="oil", bus1="naphtha", bus2="process", bus3="co2", carrier="naphtha for industry", efficiency=1.0, efficiency2=0.02, efficiency3=0.2, p_nom_extendable=True, marginal_cost=1.0)
    n.add("Link", "coal", bus0="coal", bus1="industry", bus2="co2", carrier="coal for industry", efficiency=1.0, efficiency2=0.3, p_nom_extendable=True, marginal_cost=1.0)
    n.add("Link", "process", bus0="process", bus1="co2", carrier="process emissions", efficiency=1.0, p_nom_extendable=True, marginal_cost=1.0)
    n.add("Generator", "oil supply", bus="oil", carrier="oil", p_nom=1.0e6, marginal_cost=1.0)
    n.add("Generator", "coal supply", bus="coal", carrier="coal", p_nom=1.0e6, marginal_cost=1.0)
    n.add("Load", "naphtha", bus="naphtha", carrier="naphtha for industry", p_set=10.0)
    n.add("Load", "coal load", bus="industry", carrier="coal for industry", p_set=10.0)
    n.add("Load", "process emissions", bus="process", carrier="process emissions", p_set=-5.0)
    pair = {2030: 100, 2040: 80}
    n.config = {
        "iea_co2": {
            "enable": True,
            **{key: dict(pair) for key in (
                "coal", "oil", "gas", "power_coal", "power_oil", "power_gas",
                "final_coal", "final_oil", "final_gas",
            )},
            "transport_oil": {2030: 40, 2040: 30},
        },
        "iea_final": {
            "enable": True,
            "industry": {"coal": {2030: 20, 2040: 18}, "oil": {2030: 26, 2040: 24}, "gas": {2030: 10, 2040: 8}},
            "buildings": {"coal": {2030: 6, 2040: 3}, "oil": {2030: 10, 2040: 8}, "gas": {2030: 10, 2040: 8}},
            "transport": {"oil": {2030: 10, 2040: 8}},
        },
    }
    n.optimize.create_model()
    add_iea_combustion_co2_constraints(n, "2030")
    labels = n.model.variables["Link-p"].labels.to_series()

    def linked(constraint):
        variables = set(n.model.constraints[constraint].data.vars.values)
        chosen = labels[labels.isin(variables)]
        return set(chosen.index.get_level_values(-1))

    assert "process" in linked("iea_co2_oil")
    assert "process" in linked("iea_co2_final_oil")
    assert "process" in linked("iea_co2_coal")
    assert "process" in linked("iea_co2_final_coal")


def test_final_electricity_load_is_scaled_to_the_cps_mtoe():
    from scripts.iea_cps_constraints import (
        MWH_PER_MTOE,
        _load_energy_mwh,
        scale_iea_final_loads,
    )

    n = pypsa.Network()
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    n.set_snapshots(snapshots)
    n.add("Bus", "DE", carrier="AC")
    n.add("Load", "DE industry electricity", bus="DE", carrier="industry electricity", p_set=1.0)
    skipped = scale_iea_final_loads(
        n,
        {"iea_final": {"enable": True, "industry": {"electricity": {2030: 90, 2040: 89}}}},
        "2030",
    )
    nyears = 2 / 8760
    annual = _load_energy_mwh(n, n.loads.index)
    assert annual == pytest.approx(90 * MWH_PER_MTOE * nyears)
    assert "industry electricity" not in skipped


def test_building_gas_final_energy_is_an_equality():
    pytest.importorskip("pandera")
    from scripts.iea_cps_constraints import MWH_PER_MTOE, add_iea_final_fuel_constraints

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", ["gas", "heat", "coal", "co2"], carrier=["gas", "heat", "coal", "co2"])
    n.add(
        "Link",
        "DE gas boiler",
        bus0="gas",
        bus1="heat",
        carrier="residential rural gas boiler",
        p_nom_extendable=True,
        efficiency=0.9,
    )
    n.add(
        "Link",
        "DE coal boiler",
        bus0="coal",
        bus1="heat",
        bus2="co2",
        carrier="residential rural coal boiler",
        p_nom_extendable=True,
        efficiency=0.9,
        efficiency2=0.3,
    )
    n.config = {
        "iea_final": {
            "enable": True,
            "buildings": {
                "gas": {2030: 154, 2040: 158},
                "coal": {2030: 6, 2040: 3},
            },
        }
    }
    n.optimize.create_model()
    add_iea_final_fuel_constraints(n, "2030")
    nyears = 2 / 8760
    gas = float(n.model.constraints["iea_final_buildings_gas"].rhs)
    coal = float(n.model.constraints["iea_final_buildings_coal"].rhs)
    assert gas == pytest.approx(154 * MWH_PER_MTOE * nyears)
    assert coal == pytest.approx(6 * MWH_PER_MTOE * nyears)


def test_coal_boilers_and_other_energy_sector_are_added():
    pytest.importorskip("pandera")
    from scripts.iea_cps_constraints import (
        add_coal_boilers,
        add_iea_combustion_co2_constraints,
        add_other_energy_sector,
    )

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add(
        "Bus",
        ["DE", "heat", "EU coal", "EU oil primary", "EU gas", "co2 atmosphere"],
        carrier=["AC", "heat", "coal", "oil primary", "gas", "co2"],
    )
    n.add(
        "Link",
        "DE coal",
        bus0="EU coal",
        bus1="DE",
        bus2="co2 atmosphere",
        carrier="coal",
        p_nom_extendable=True,
        efficiency=0.4,
        efficiency2=0.3,
    )
    n.add(
        "Link",
        "DE oil",
        bus0="EU oil primary",
        bus1="DE",
        bus2="co2 atmosphere",
        carrier="oil",
        p_nom_extendable=True,
        efficiency=0.4,
        efficiency2=0.25,
    )
    n.add(
        "Link",
        "DE OCGT",
        bus0="EU gas",
        bus1="DE",
        bus2="co2 atmosphere",
        carrier="OCGT",
        p_nom_extendable=True,
        efficiency=0.4,
        efficiency2=0.2,
    )
    n.add(
        "Link",
        "DE residential rural oil boiler",
        bus0="EU oil primary",
        bus1="heat",
        bus2="co2 atmosphere",
        carrier="residential rural oil boiler",
        p_nom_extendable=True,
        efficiency=0.85,
        capital_cost=10.0,
        lifetime=20.0,
    )
    config = {
        "iea_final": {"enable": True, "buildings": {"coal": {2030: 6, 2040: 3}}},
        "iea_co2": {
            "enable": True,
            "coal": {2030: 637, 2040: 424},
            "power_coal": {2030: 483, 2040: 292},
            "final_coal": {2030: 123, 2040: 105},
            "oil": {2030: 1132, 2040: 975},
            "power_oil": {2030: 19, 2040: 13},
            "final_oil": {2030: 1047, 2040: 900},
            "gas": {2030: 893, 2040: 929},
            "power_gas": {2030: 294, 2040: 327},
            "final_gas": {2030: 567, 2040: 571},
        },
    }
    add_coal_boilers(n, config)
    boiler = n.links.loc["DE residential rural coal boiler"]
    assert boiler.carrier == "residential rural coal boiler"
    assert boiler.bus0 == "EU coal"
    assert boiler.bus1 == "heat"
    assert float(boiler.efficiency) == pytest.approx(0.85)
    assert float(boiler.efficiency2) == pytest.approx(0.3)
    assert float(boiler.capital_cost) == pytest.approx(10.0)

    add_other_energy_sector(n, config, "2030")
    assert float(n.links.at["EU other energy coal", "efficiency"]) == pytest.approx(0.3)
    assert n.links.at["EU other energy oil", "bus0"] == "EU oil primary"
    assert n.links.at["EU other energy gas", "bus0"] == "EU gas"

    n.config = config
    n.optimize.create_model()
    add_iea_combustion_co2_constraints(n, "2030")
    labels = n.model.variables["Link-p"].labels.to_series()

    def names(constraint: str) -> set[str]:
        variables = set(n.model.constraints[constraint].data.vars.values)
        chosen = labels[labels.isin(variables)].index.get_level_values("Link")
        return set(chosen)

    assert "EU other energy coal" in names("iea_co2_coal")
    assert "EU other energy coal" not in names("iea_co2_power_coal")
    assert "EU other energy coal" not in names("iea_co2_final_coal")
    assert "DE residential rural coal boiler" in names("iea_co2_final_coal")


def test_eu_comparison_matches_a_fixed_dispatch(tmp_path):
    from scripts.iea_cps_constraints import MWH_PER_MTOE
    from postprocessing.postprocess_iea_cps import compare_network, plot_category

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.snapshot_weightings.loc[:, :] = 4380.0
    n.add("Bus", "DE", carrier="AC")
    n.add("Generator", "DE nuclear", bus="DE", carrier="nuclear", p_nom=1000)
    n.generators_t.p["DE nuclear"] = 1000.0
    n.add("Load", "DE industry electricity", bus="DE", carrier="industry electricity", p_set=MWH_PER_MTOE / 8760)
    config = {
        "iea_generation": {"enable": True, "nuclear": {2030: 8.76, 2040: 8.76}},
        "iea_final": {"enable": True, "industry": {"electricity": {2030: 1, 2040: 1}}},
    }
    report = compare_network(n, config, 2030)
    nuclear = report.loc[report["name"].eq("nuclear")].iloc[0]
    electricity = report.loc[report["name"].eq("industry electricity")].iloc[0]
    assert nuclear.status == "pass"
    assert nuclear.model == pytest.approx(8.76)
    assert electricity.status == "pass"
    assert electricity.model == pytest.approx(1.0)
    plot_category(report, "electricity", tmp_path / "electricity.png")
    assert (tmp_path / "electricity.png").stat().st_size > 0


def test_naphtha_is_scaled_to_industry_oil():
    from scripts.iea_cps_constraints import (
        MWH_PER_MTOE,
        _load_energy_mwh,
        scale_iea_final_loads,
    )

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "naphtha", carrier="oil")
    n.add("Load", "DE naphtha for industry", bus="naphtha", carrier="naphtha for industry", p_set=1.0)
    scale_iea_final_loads(
        n,
        {"iea_final": {"enable": True, "industry": {"oil": {2030: 26, 2040: 24}}}},
        "2030",
    )
    annual = _load_energy_mwh(n, n.loads.index)
    assert annual == pytest.approx(26 * MWH_PER_MTOE * (2 / 8760))


def test_biofuels_are_added_to_internal_combustion_oil():
    from scripts.iea_cps_constraints import (
        MWH_PER_MTOE,
        _load_energy_mwh,
        scale_iea_final_loads,
    )

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "oil", carrier="oil")
    n.add("Load", "DE ice", bus="oil", carrier="land transport oil", p_set=1.0)
    n.add("Load", "DE jet", bus="oil", carrier="kerosene for aviation", p_set=1.0)
    scale_iea_final_loads(
        n,
        {
            "iea_final": {
                "enable": True,
                "transport": {
                    "oil": {2030: 10, 2040: 10},
                    "biofuels": {2030: 5, 2040: 5},
                },
            }
        },
        "2030",
    )
    nyears = 2 / 8760
    ice = _load_energy_mwh(n, pd.Index(["DE ice"]))
    jet = _load_energy_mwh(n, pd.Index(["DE jet"]))
    assert ice == pytest.approx(10 * MWH_PER_MTOE * nyears)
    assert jet == pytest.approx(5 * MWH_PER_MTOE * nyears)


def test_geothermal_capacity_is_raised_when_energy_exceeds_the_gigawatts(tmp_path):
    from scripts.iea_cps_constraints import lift_geothermal_if_energy_exceeds_capacity

    path = _limit_file(
        tmp_path / "limits.csv",
        [("DE", "geothermal", 600.0, 600.0), ("FR", "geothermal", 400.0, 400.0)],
    )
    n = pypsa.Network()
    n.add("Bus", ["DE", "FR"], carrier="AC", country=["DE", "FR"])
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Generator", "DE geo", bus="DE", carrier="geothermal", p_nom=600, p_max_pu=0.5, p_min_pu=1)
    n.add("Generator", "FR geo", bus="FR", carrier="geothermal", p_nom=400, p_max_pu=0.5, p_min_pu=1)
    n.generators_t.p_min_pu["DE geo"] = 1.0
    n.generators_t.p_max_pu["FR geo"] = 0.5
    config = {
        "iea_generation": {"enable": True, "geothermal": {2030: 9, 2040: 12}},
        "solving": {"agg_p_nom_limits": {"file": path.as_posix()}},
    }
    lift_geothermal_if_energy_exceeds_capacity(n, config, "2030")
    required = 9 / 8.76 * 1000
    assert n.generators.p_nom.sum() == pytest.approx(required, rel=1e-5)
    assert n.generators.p_nom.sum() > required
    assert n.generators.p_nom_max.sum() == pytest.approx(n.generators.p_nom.sum())
    assert float(n.generators.p_max_pu.min()) == 1
    assert float(n.generators.p_min_pu.max()) == 0
    assert "DE geo" not in n.generators_t.p_min_pu.columns
    assert "FR geo" not in n.generators_t.p_max_pu.columns
    assert sum(n._iea_capacity_override.values()) == pytest.approx(required, rel=1e-5)
    # 2 GW can produce 12 TWh, so 2040 is not raised.
    n2040 = pypsa.Network()
    n2040.add("Bus", "DE", carrier="AC", country="DE")
    n2040.add("Generator", "DE geo", bus="DE", carrier="geothermal", p_nom=2000)
    wide = _limit_file(
        tmp_path / "wide.csv",
        [("DE", "geothermal", 2000.0, 2000.0)],
        year="2040",
    )
    config["solving"]["agg_p_nom_limits"]["file"] = wide.as_posix()
    lift_geothermal_if_energy_exceeds_capacity(n2040, config, "2040")
    assert n2040.generators.at["DE geo", "p_nom"] == 2000
    assert not getattr(n2040, "_iea_capacity_override", None)
    # The file allows 2 GW, but the plant is only 0.9 GW and cannot make 12 TWh.
    short = pypsa.Network()
    short.add("Bus", "DE", carrier="AC", country="DE")
    short.add(
        "Generator",
        "DE geo",
        bus="DE",
        carrier="geothermal",
        p_nom=900,
        p_nom_max=900,
    )
    lift_geothermal_if_energy_exceeds_capacity(short, config, "2040")
    required_2040 = 12 / 8.76 * 1000
    assert short.generators.at["DE geo", "p_nom"] == pytest.approx(
        required_2040, rel=1e-5
    )
    assert short.generators.at["DE geo", "p_nom"] > required_2040


def test_agriculture_oil_stays_inside_the_other_energy_residual():
    from scripts.iea_cps_constraints import (
        _co2_weights,
        keep_agriculture_oil_within_other_energy,
    )

    config = {
        "iea_co2": {
            "oil": {2030: 1132, 2040: 975},
            "power_oil": {2030: 19, 2040: 13},
            "final_oil": {2030: 1047, 2040: 900},
        }
    }
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.snapshot_weightings.loc[:, :] = 4380.0
    hours = float(n.snapshot_weightings.generators.sum())
    n.add(
        "Bus",
        ["EU oil", "DE agriculture machinery oil", "co2 atmosphere"],
        carrier=["oil", "agriculture machinery oil", "co2"],
    )
    n.add(
        "Link",
        "DE agriculture machinery oil",
        bus0="EU oil",
        bus1="DE agriculture machinery oil",
        bus2="co2 atmosphere",
        carrier="agriculture machinery oil",
        efficiency=1.0,
        efficiency2=0.25,
        p_nom_extendable=True,
    )
    n.add(
        "Load",
        "DE agriculture oil",
        bus="DE agriculture machinery oil",
        carrier="agriculture machinery oil",
        p_set=10e6 / hours,
    )
    keep_agriculture_oil_within_other_energy(n, config, "2030")
    assert n.loads.at["DE agriculture oil", "p_set"] == pytest.approx(10e6 / hours)
    weights = _co2_weights(n)
    assert weights.at["DE agriculture machinery oil", "other"] == 1.0
    assert weights.at["DE agriculture machinery oil", "final"] == 0.0

    n.links.at["DE agriculture machinery oil", "efficiency2"] = 10.0
    keep_agriculture_oil_within_other_energy(n, config, "2030")
    # 10 TWh at 10 t/MWh is 100 Mt. The residual is 1132 - 19 - 1047 = 66 Mt.
    assert n.loads.at["DE agriculture oil", "p_set"] == pytest.approx(10e6 / hours * 0.66)


def test_refinery_emissions_are_outside_the_eu_total_and_agriculture_stays():
    from scripts.iea_cps_constraints import (
        _co2_weights,
        drop_refinery_emissions,
        keep_agriculture_oil_within_other_energy,
    )

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.snapshot_weightings.loc[:, :] = 4380.0
    hours = float(n.snapshot_weightings.generators.sum())
    n.add(
        "Bus",
        ["primary", "EU oil", "agri", "co2 atmosphere"],
        carrier=["oil primary", "oil", "agriculture machinery oil", "co2"],
    )
    n.add(
        "Link",
        "EU oil refining",
        bus0="primary",
        bus1="EU oil",
        bus2="co2 atmosphere",
        carrier="oil refining",
        efficiency=0.95,
        efficiency2=0.02,
        p_nom=1.0e6,
    )
    n.add(
        "Link",
        "agri",
        bus0="EU oil",
        bus1="agri",
        bus2="co2 atmosphere",
        carrier="agriculture machinery oil",
        efficiency=1.0,
        efficiency2=0.25,
        p_nom_extendable=True,
    )
    n.add("Load", "agri", bus="agri", carrier="agriculture machinery oil", p_set=20e6 / hours)
    config = {
        "iea_co2": {
            "oil": {2030: 1132, 2040: 975},
            "power_oil": {2030: 19, 2040: 13},
            "final_oil": {2030: 1047, 2040: 900},
        }
    }
    drop_refinery_emissions(n)
    keep_agriculture_oil_within_other_energy(n, config, "2030")
    assert n.links.at["EU oil refining", "efficiency"] == 1.0
    assert n.links.at["EU oil refining", "efficiency2"] == 0.0
    assert n.loads.at["agri", "p_set"] == pytest.approx(20e6 / hours)
    assert _co2_weights(n).empty or "EU oil refining" not in _co2_weights(n).index


def test_solid_biomass_ceiling_covers_electricity_fuel():
    from scripts.iea_cps_constraints import raise_solid_biomass_ceiling

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "biomass", carrier="solid biomass")
    n.add("Bus", "AC", carrier="AC")
    n.add(
        "Generator",
        "supply",
        bus="biomass",
        carrier="solid biomass",
        p_nom=100,
        e_sum_max=100,
    )
    n.add(
        "GlobalConstraint",
        "biomass limit",
        type="operational_limit",
        carrier_attribute="solid biomass",
        sense="<=",
        constant=100,
    )
    n.add(
        "GlobalConstraint",
        "unsustainable biomass limit",
        type="operational_limit",
        carrier_attribute="unsustainable solid biomass",
        sense="==",
        constant=50,
    )
    n.add(
        "Link",
        "chp",
        bus0="biomass",
        bus1="AC",
        carrier="urban central solid biomass CHP",
        efficiency=0.5,
        p_nom=10,
    )
    raise_solid_biomass_ceiling(
        n,
        {
            "iea_generation": {"enable": True, "biomass": {2030: 8.76, 2040: 8.76}},
            "iea_final": {"enable": True},
        },
        "2030",
    )
    # 8.76 TWh over the toy year fraction, divided by efficiency 0.5, minus 50 MWh unsustainable.
    nyears = 2 / 8760
    required = 8.76e6 * nyears / 0.5
    expected = required * (1 + 1e-4) - 50
    assert n.global_constraints.at["biomass limit", "constant"] == pytest.approx(expected)
    assert n.generators.at["supply", "e_sum_max"] == pytest.approx(expected)
    assert n.generators.at["supply", "p_nom"] == pytest.approx(expected)
    assert n.global_constraints.at["unsustainable biomass limit", "constant"] == 50


def test_international_aviation_and_shipping_are_omitted():
    from types import SimpleNamespace

    from scripts.iea_cps_constraints import _load_energy_mwh
    from scripts.prepare_sector_network import add_aviation, add_shipping

    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    totals = pd.DataFrame(
        {
            "total domestic aviation": [1.0, 0.5],
            "total international aviation": [4.0, 3.0],
            "total domestic navigation": [2.0, 1.0],
        },
        index=["DE", "FR"],
    )
    costs = pd.DataFrame({"CO2 intensity": [0.27]}, index=["oil"])
    spatial = SimpleNamespace(
        oil=SimpleNamespace(
            kerosene=pd.Index(["DE kerosene for aviation", "FR kerosene for aviation"]),
            shipping=pd.Index(["DE shipping oil", "FR shipping oil"]),
            demand_locations=pd.Index(["DE", "FR"]),
            nodes=["EU oil"],
        )
    )
    options = {
        "aviation_demand_factor": 1,
        "regional_oil_demand": True,
        "exclude_international_bunkers": True,
        "methanol": {"methanol_to_kerosene": False},
        "shipping_oil_share": 1,
        "shipping_hydrogen_share": 0,
        "shipping_methanol_share": 0,
    }
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "EU oil", carrier="oil")
    n.add("Bus", "co2 atmosphere", carrier="co2")
    add_aviation(n, costs, totals, totals, options, spatial)
    assert _load_energy_mwh(n, n.loads.index) == pytest.approx(1.5e6)

    shipping = pypsa.Network()
    shipping.set_snapshots(snapshots)
    shipping.add("Bus", "EU oil", carrier="oil")
    shipping.add("Bus", "co2 atmosphere", carrier="co2")
    add_shipping(
        shipping,
        costs,
        "unused.csv",
        totals,
        totals,
        options,
        spatial,
        2030,
    )
    assert _load_energy_mwh(shipping, shipping.loads.index) == pytest.approx(3.0e6)


def test_primary_oil_check_uses_barrels_per_toe():
    from scripts.iea_cps_constraints import MBD_TO_MWH, MWH_PER_MTOE

    mtoe = 9.6 * 365.25 / 7.33
    assert MBD_TO_MWH * 9.6 / MWH_PER_MTOE == pytest.approx(mtoe)


def test_baseline_abatement_ratio_matches_the_cost_and_co2_difference(tmp_path, caplog):
    """cost 130 against 100 and CO2 40 against 50 is 3 EUR/t. 2018 is not drawn."""
    import logging

    from postprocessing.baseline_abatement import (
        BASELINE_YEAR,
        build_project_table,
        eur_per_tonne,
        group_carriers,
        group_columns,
        plot_baseline_abatement,
    )

    assert eur_per_tonne(130, 100, 50, 40) == pytest.approx(3)

    baseline = group_carriers(pd.Series({"onwind": 40.0, "OCGT": 60.0}))
    year = group_carriers(pd.Series({"onwind": 100.0, "coal": 30.0}))
    assert baseline.sum() == pytest.approx(100)
    assert year.sum() == pytest.approx(130)

    caplog.set_level(logging.WARNING)
    table = build_project_table(
        {
            2018: baseline,
            2030: year,
            2035: group_carriers(pd.Series({"onwind": 10.0})),
        },
        {2018: 50.0, 2030: 40.0, 2035: 50.0},
    )
    assert table["year"].tolist() == [2030]
    assert BASELINE_YEAR not in set(table["year"])
    assert table.loc[0, "eur_per_t"] == pytest.approx(3)
    assert table.loc[0, "emission_change"] == pytest.approx(40 - 50)
    assert table.loc[0, "cost_change"] == pytest.approx(30)
    changes = table.loc[0, group_columns(table)]
    assert changes.sum() == pytest.approx(table.loc[0, "cost_change"])
    assert changes.sum() == pytest.approx(130 - 100)
    assert "2035 has the same CO2 as 2018" in caplog.text

    injected = pd.concat(
        [table, table.assign(year=BASELINE_YEAR)],
        ignore_index=True,
    )
    plotted = plot_baseline_abatement(injected, tmp_path / "baseline_abatement.png")
    assert plotted == [2030]
    assert BASELINE_YEAR not in plotted
    assert (tmp_path / "baseline_abatement.png").stat().st_size > 0
