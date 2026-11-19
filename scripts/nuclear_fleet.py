# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Existing nuclear fleet for myopic planning.

Published closure or licence years overwrite the powerplant database.
Missing dates are retrofit plus 20 years or commissioning plus 40, and
each country keeps one generator. Without a Current Policies share, the
capacity still online is a lower bound and capacity built in an earlier
myopic horizon is carried forward. With a share, that country's capacity
is fixed at the share.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def _generator_countries(n: pypsa.Network, generators: pd.DataFrame) -> pd.Series:
    if "country" in n.buses.columns:
        country = generators.bus.map(n.buses.country)
    else:
        country = pd.Series(pd.NA, index=generators.index)
    return country.fillna(generators.bus.astype(str).str[:2])


# Missing nuclear DateOut: retrofit year + 20, otherwise commissioning year + 40.
_NUCLEAR_LIFETIME = 40
_NUCLEAR_RETROFIT_EXTENSION = 20
# Belgian units with no published DateOut are retired in 2035, the licensed
# end year of Doel 4 and Tihange 3. Recorded dates are not overwritten.
_BE_NUCLEAR_DATE_OUT = 2035

# Published closure, licence, or operator end year (checked October 2026).
# France and Sweden have no unit licence, so those entries are commissioning
# year + 60, the current planning life. This replaces dates from the
# powerplant database whenever the powerplant file is rebuilt.
_PUBLISHED_NUCLEAR_DATEOUT = {
    "Doel 1": 2025,
    "Doel 2": 2025,
    "Tihange 1": 2025,
    "Doel 4": 2035,
    "Tihange 3": 2035,
    "Kozloduy 5": 2049,
    "Kozloduy 6": 2051,
    "Beznau 1": 2033,
    "Beznau 2": 2032,
    "Goesgen": 2039,
    "Leibstadt": 2045,
    "Dukovany 1": 2065,
    "Dukovany 2": 2066,
    "Dukovany 3": 2066,
    "Dukovany 4": 2067,
    "Temelin 1": 2062,
    "Temelin 2": 2063,
    "Almaraz 1": 2030,
    "Almaraz 2": 2030,
    "Asco 1": 2030,
    "Asco 2": 2032,
    "Cofrentes": 2030,
    "Vandellos 2": 2035,
    "Trillo 1": 2035,
    "Loviisa 1": 2050,
    "Loviisa 2": 2050,
    "Olkiluoto 1": 2038,
    "Olkiluoto 2": 2038,
    "Olkiluoto 3": 2083,
    "Belleville 1": 2048,
    "Belleville 2": 2049,
    "Blayais 1": 2041,
    "Blayais 2": 2043,
    "Blayais 3": 2043,
    "Blayais 4": 2043,
    "Bugey 2": 2039,
    "Bugey 3": 2039,
    "Bugey 4": 2039,
    "Bugey 5": 2040,
    "Cattenom 1": 2047,
    "Cattenom 2": 2048,
    "Cattenom 3": 2051,
    "Cattenom 4": 2052,
    "Chinon B 1": 2044,
    "Chinon B 2": 2044,
    "Chinon B 3": 2047,
    "Chinon B 4": 2048,
    "Chooz B 1": 2060,
    "Chooz B 2": 2060,
    "Civaux 1": 2062,
    "Civaux 2": 2062,
    "Cruas 1": 2044,
    "Cruas 2": 2045,
    "Cruas 3": 2044,
    "Cruas 4": 2045,
    "Dampierre 1": 2040,
    "Dampierre 2": 2041,
    "Dampierre 3": 2041,
    "Dampierre 4": 2041,
    "Flamanville 1": 2046,
    "Flamanville 2": 2047,
    "Flamanville 3": 2084,
    "Golfech 1": 2051,
    "Golfech 2": 2054,
    "Gravelines 1": 2040,
    "Gravelines 2": 2040,
    "Gravelines 3": 2041,
    "Gravelines 4": 2041,
    "Gravelines 5": 2045,
    "Gravelines 6": 2045,
    "Nogent 1": 2048,
    "Nogent 2": 2049,
    "Paluel 1": 2045,
    "Paluel 2": 2045,
    "Paluel 3": 2046,
    "Paluel 4": 2046,
    "Penly 1": 2050,
    "Penly 2": 2052,
    "Saint Laurent B 1": 2043,
    "Saint Laurent B 2": 2043,
    "St Alban 1": 2046,
    "St Alban 2": 2047,
    "Tricastin 1": 2040,
    "Tricastin 2": 2040,
    "Tricastin 3": 2041,
    "Tricastin 4": 2041,
    "Hartlepool A1": 2030,
    "Heysham A1": 2030,
    "Heysham B1": 2030,
    "Torness 1": 2030,
    "Torness 2": 2030,
    "Sizewell B": 2055,
    "Paks 1": 2032,
    "Paks 2": 2034,
    "Paks 3": 2036,
    "Paks 4": 2037,
    "Borssele 1": 2033,
    "Cernavoda 1": 2060,
    "Cernavoda 2": 2037,
    "Forsmark 1": 2040,
    "Forsmark 2": 2041,
    "Forsmark 3": 2045,
    "Oskarshamn 3": 2045,
    "Ringhals 3": 2041,
    "Ringhals 4": 2043,
    "Krsko 1": 2043,
    "Bohunice 3": 2045,
    "Bohunice 4": 2045,
    "Mochovce 1": 2058,
    "Mochovce 2": 2060,
    "Mochovce 3": 2083,
    "Mochovce 4": 2085,
}
_PUBLISHED_NUCLEAR_RETROFIT = {
    "Doel 4": 2025,
    "Tihange 3": 2025,
    "Cernavoda 1": 2029,
}


def apply_published_nuclear_dates(plants: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrite nuclear retirement years with the published-date table.

    A matching plant name always gets that ``DateOut``, including when the
    powerplant database already has a different year. Plants that are not in
    the table are left unchanged.
    """
    out = plants.copy()
    if "Name" not in out.columns or "Fueltype" not in out.columns:
        return out
    nuclear = out["Fueltype"].astype(str).str.lower().eq("nuclear")
    names = out["Name"].astype(str)
    for name, year in _PUBLISHED_NUCLEAR_DATEOUT.items():
        out.loc[nuclear & names.eq(name), "DateOut"] = year
    if "DateRetrofit" in out.columns:
        for name, year in _PUBLISHED_NUCLEAR_RETROFIT.items():
            out.loc[nuclear & names.eq(name), "DateRetrofit"] = year
        # A retrofit year equal to the build year is not a life extension.
        date_in = (
            pd.to_numeric(out["DateIn"], errors="coerce")
            if "DateIn" in out.columns
            else pd.Series(pd.NA, index=out.index)
        )
        retrofit = pd.to_numeric(out["DateRetrofit"], errors="coerce")
        bogus = nuclear & retrofit.notna() & (retrofit == date_in)
        bogus &= ~names.isin(_PUBLISHED_NUCLEAR_RETROFIT)
        out.loc[bogus, "DateRetrofit"] = pd.NA
    return out


def nuclear_online_capacity(plants: pd.DataFrame, year: int) -> pd.Series:
    """Nuclear MW still online in ``year``, summed by country."""
    if "Fueltype" not in plants.columns:
        return pd.Series(dtype=float)
    fuel = plants["Fueltype"].astype(str).str.lower()
    nuclear = plants.loc[fuel.eq("nuclear"), ["Country", "Capacity", "DateOut"]].copy()
    date_out = pd.to_numeric(nuclear["DateOut"], errors="coerce")
    online = date_out.notna() & (date_out >= int(year))
    nuclear = nuclear.loc[online]
    if nuclear.empty:
        return pd.Series(dtype=float)
    return nuclear.groupby(nuclear["Country"].astype(str))["Capacity"].sum()


def fill_missing_nuclear_dateout(plants: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing nuclear ``DateOut`` from retrofit or commissioning year.

    A recorded ``DateOut`` is left unchanged. If ``DateRetrofit`` is present,
    ``DateOut`` is that year plus 20. Otherwise it is ``DateIn`` plus 40.
    Belgian units with no date are set to 2035.
    """
    out = plants.copy()
    if "Fueltype" not in out.columns or "DateOut" not in out.columns:
        return out
    nuclear = out["Fueltype"].astype(str).str.lower().eq("nuclear")
    missing = nuclear & pd.to_numeric(out["DateOut"], errors="coerce").isna()
    if "Country" in out.columns:
        be = missing & out["Country"].astype(str).eq("BE")
        out.loc[be, "DateOut"] = _BE_NUCLEAR_DATE_OUT
        missing = missing & ~be
    if not missing.any():
        return out
    date_in = (
        pd.to_numeric(out["DateIn"], errors="coerce")
        if "DateIn" in out.columns
        else pd.Series(pd.NA, index=out.index)
    )
    if "DateRetrofit" in out.columns:
        retrofit = pd.to_numeric(out["DateRetrofit"], errors="coerce")
    else:
        retrofit = pd.Series(pd.NA, index=out.index)
    filled = (date_in + _NUCLEAR_LIFETIME).astype("float64")
    has_retrofit = retrofit.notna()
    filled.loc[has_retrofit] = (retrofit + _NUCLEAR_RETROFIT_EXTENSION).loc[has_retrofit]
    out.loc[missing, "DateOut"] = filled.loc[missing]
    return out


def nuclear_capacity_by_country(powerplants_file: str, year: int) -> pd.Series:
    """
    Nuclear MW still online in ``year``, summed by country.

    Missing retirement dates are filled first (retrofit + 20 years, otherwise
    commissioning + 40; Belgian units with no date use 2035). A unit is then
    dropped when ``DateOut`` is before ``year``.
    """
    plants = pd.read_csv(powerplants_file)
    plants = apply_published_nuclear_dates(plants)
    plants = fill_missing_nuclear_dateout(plants)
    fuel = plants["Fueltype"].astype(str).str.lower()
    nuclear = plants.loc[fuel.eq("nuclear"), ["Country", "Capacity", "DateOut"]].copy()
    date_out = pd.to_numeric(nuclear["DateOut"], errors="coerce")
    online = date_out.notna() & (date_out >= int(year))
    nuclear = nuclear.loc[online]
    if nuclear.empty:
        return pd.Series(dtype=float)
    return nuclear.groupby(nuclear["Country"].astype(str))["Capacity"].sum()


def apply_nuclear_capacity(
    n: pypsa.Network,
    powerplants_file: str,
    year: int,
    n_previous: pypsa.Network | None = None,
    cps_max: pd.Series | None = None,
) -> None:
    """
    Set existing nuclear from ``powerplants.csv`` for this planning horizon.

    Units whose decommissioning date is before ``year`` are removed. All
    nuclear units in a country are collapsed to one generator. Without
    ``cps_max``, ``p_nom_min`` is the capacity still online plus new build
    from the previous myopic horizon, and further new build is allowed.
    With ``cps_max``, an EU country is fixed at that country's share of the
    Current Policies total. ``p_nom`` and ``p_nom_min`` stay at the capacity
    already in service, and ``capital_cost`` is left unchanged, so MW above
    that fleet is new and carries its annualised capital cost.
    """
    available = nuclear_capacity_by_country(powerplants_file, year)

    carried = pd.Series(dtype=float)
    if n_previous is not None:
        prev = n_previous.generators[n_previous.generators.carrier == "nuclear"]
        if not prev.empty and "p_nom_opt" in prev.columns:
            new_mw = (prev["p_nom_opt"] - prev["p_nom_min"]).clip(lower=0)
            carried = new_mw.groupby(_generator_countries(n_previous, prev)).sum()

    if cps_max is not None:
        cps_max = cps_max.copy()
        cps_max.index = cps_max.index.astype(str)

    gens = n.generators[n.generators.carrier == "nuclear"]
    if gens.empty:
        logger.info("No nuclear generators present for %s", year)
        return

    countries = _generator_countries(n, gens)
    removed = []
    for country, idx in gens.groupby(countries).groups.items():
        idx = pd.Index(idx)
        keep = n.generators.loc[idx, "p_nom"].idxmax()
        drop = idx.difference([keep])
        existing = float(available.get(country, 0.0))
        code = str(country)
        if cps_max is not None and code in cps_max.index and pd.notna(cps_max.loc[code]):
            total = max(float(cps_max.loc[code]), 0.0)
            # Only the fleet already in service is sunk. The share can be
            # higher: that extra MW stays above p_nom_min, so the annualised
            # capital cost on the generator applies to it. The country
            # capacity limit then holds the generator at the share.
            sunk = min(existing, total)
            n.generators.loc[keep, "p_nom"] = sunk
            n.generators.loc[keep, "p_nom_min"] = sunk
            n.generators.loc[keep, "p_nom_max"] = total
            n.generators.loc[keep, "p_nom_extendable"] = total > 0
            logger.info(
                "Nuclear %s in %s: %.3f GW still online, fixed at %.3f GW "
                "from the EU Current Policies share (%.3f GW of that is new).",
                code,
                year,
                existing / 1e3,
                total / 1e3,
                max(total - existing, 0.0) / 1e3,
            )
        else:
            extra = (
                float(carried.get(country, 0.0)) if country in carried.index else 0.0
            )
            total = existing + extra
            n.generators.loc[keep, "p_nom"] = total
            n.generators.loc[keep, "p_nom_min"] = total
            n.generators.loc[keep, "p_nom_extendable"] = True
            n.generators.loc[keep, "p_nom_max"] = np.inf
            logger.info(
                "Nuclear %s in %s: %.3f GW still online, %.3f GW carried from "
                "previous builds, extendable",
                code,
                year,
                existing / 1e3,
                extra / 1e3,
            )
        if len(drop):
            removed.extend(drop)

    if cps_max is not None:
        present = set(countries.astype(str))
        missing = cps_max.index[(cps_max.fillna(0) > 0) & ~cps_max.index.isin(present)]
        if len(missing):
            detail = ", ".join(f"{code} {float(cps_max[code]) / 1e3:.2f} GW" for code in missing)
            logger.warning(
                "Current Policies nuclear share has no generator in %s: %s",
                year,
                detail,
            )

    if removed:
        n.remove("Generator", removed)
