import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Saka Haphong Summit Cryosphere Detector", layout="wide")

LAT = 21.7865
LON = 92.6101
SUMMIT_ELEV_M = 1063.0
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# ---------------------------------------------------------------- constants
SIGMA = 5.670374e-8        # Stefan-Boltzmann, W m-2 K-4
LF = 3.34e5                # latent heat of fusion, J/kg
LV = 2.50e6                # latent heat of condensation (wet snow surface), J/kg
CP_AIR = 1005.0            # J/kg/K
CW = 4186.0                # specific heat of water, J/kg/K
RHO_W = 1000.0             # kg/m3
R_AIR = 287.05             # J/kg/K
SNOW_EMISS = 0.98
WET_SNOW_ALBEDO = 0.60
MIN_WIND_MS = 0.5          # floor so turbulent fluxes never vanish in calm air
SNOW_FALL_SPEED = 1.0      # m/s, wet snowflakes
WET_WINDOW = (-0.5, 1.5)   # wet-bulb range (°C) where snow is wet and sticky
PACK_DENSITY = 250.0       # kg/m3, settled wet snowpack
DEFAULT_LAPSE = 6.0        # °C/km


# ------------------------------------------------------------------ physics
def sat_vp_water(t_c):
    """Saturation vapour pressure over water, hPa (Magnus form of Clausius-Clapeyron)."""
    t = np.asarray(t_c, dtype=float)
    return 6.112 * np.exp(17.62 * t / (243.12 + t))


def station_pressure_hpa(z_m):
    """Standard-atmosphere pressure at elevation z, hPa."""
    return 1013.25 * (1.0 - 2.25577e-5 * z_m) ** 5.25588


def dew_point(t_c, rh_pct):
    t = np.asarray(t_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    g = np.log(rh / 100.0) + 17.62 * t / (243.12 + t)
    return 243.12 * g / (17.62 - g)


def wet_bulb(t_c, rh_pct, p_hpa):
    """Thermodynamic wet-bulb temperature from the psychrometric equation,
    e = es(Tw) - A*p*(T - Tw), solved by bisection at the station pressure."""
    t = np.asarray(t_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    e = rh / 100.0 * sat_vp_water(t)
    lo, hi = t - 60.0, t.copy()
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f = sat_vp_water(mid) - 6.6e-4 * (1.0 + 1.15e-3 * mid) * p_hpa * (t - mid) - e
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    return 0.5 * (lo + hi)


def snow_probability(tw_c, t50, scale):
    """Probability a falling hydrometeor is snow: logistic in wet-bulb temperature."""
    return 1.0 / (1.0 + np.exp((np.asarray(tw_c, dtype=float) - t50) / scale))


def snow_density(t_c):
    """Fresh-snow density (kg/m3), Hedstrom & Pomeroy (1998)."""
    t = np.clip(np.asarray(t_c, dtype=float), -15.0, 2.0)
    return 67.92 + 51.25 * np.exp(t / 2.59)


def get_verdict(row):
    t_crest = row["Crest_Air_Temp_C"]
    tw = row["Crest_Wet_Bulb_C"]
    t_leaf = row["Leaf_Surface_Temp_C"]
    precip = row["Precip_mm"]
    rh = row["RH_pct"]
    ps = row["Snow_Prob"]

    if precip > 0:
        if ps >= 0.5:
            if tw < WET_WINDOW[0]:
                return "DRY SNOW (cold, non-sticky)"
            return f"STICKY WET SNOW (SLR {row['SLR']:.0f}:1, accretion on bamboo/wires)"
        elif ps >= 0.2:
            return "SLEET / WET SNOW-RAIN MIX"
        elif tw <= 2.5:
            return "NEAR-FREEZING COLD RAIN / GRAUPEL RISK"
        return "LIQUID WINTER RAIN (snow level above summit)"
    if t_leaf <= 0.0 and rh >= 75:
        return "RIME ICE & HOARFROST ACCRETION ON BAMBOO/ROCK"
    if t_leaf <= 0.0:
        return "DRY RADIATIVE FROST ON GRASS & ROCK"
    if t_crest <= 4.0:
        return "SEVERE COLD-WAVE CHILL (Near-Frost Threshold)"
    return "DRY WINTER RIDGE"


def classify_wind(deg):
    if deg >= 315 or deg <= 45:
        return "N/NW Himalayan Cold Surge"
    elif 135 <= deg <= 225:
        return "S/SE Bay of Bengal Moisture"
    return "Transitional Ridge Flow"


def compute_physics(df, cliff, leaf_drop, t50, scale, lapse, diam_m, ch):
    """Add all wet-snow physics columns to the hourly dataframe."""
    df = df.copy()
    p_hpa = station_pressure_hpa(SUMMIT_ELEV_M)

    t = df["Air_Temp_1063m_C"].to_numpy(float)
    rh = df["RH_pct"].to_numpy(float)
    wind_kmh = df["Wind_kmh"].to_numpy(float)
    u = np.maximum(wind_kmh / 3.6, MIN_WIND_MS)
    cloud = df["Cloud_pct"].to_numpy(float) / 100.0
    precip = df["Precip_mm"].to_numpy(float)
    sw = df["Shortwave_Wm2"].to_numpy(float)

    df["Hour"] = df["Timestamp"].dt.hour
    chill = np.round(wind_kmh / 10.0 * cliff, 2)
    df["Adiabatic_Chill_C"] = chill
    t_crest = t - chill
    df["Crest_Air_Temp_C"] = np.round(t_crest, 2)

    # --- moisture thermodynamics
    e_hpa = np.clip(rh, 1, 100) / 100.0 * sat_vp_water(t)
    df["Dew_Point_C"] = np.round(dew_point(t, rh), 2)
    tw_raw = wet_bulb(t, rh, p_hpa)
    df["Wet_Bulb_C"] = np.round(tw_raw, 2)
    tw = tw_raw - chill
    df["Crest_Wet_Bulb_C"] = np.round(tw, 2)

    # --- precipitation phase, density, snowfall
    p_snow = snow_probability(tw, t50, scale)
    slr = np.clip(RHO_W / snow_density(t_crest), 3.0, 30.0)
    df["Snow_Prob"] = np.round(p_snow, 3)
    df["SLR"] = np.round(slr, 1)
    df["Snowfall_Derived_cm"] = np.round(precip * p_snow * slr / 10.0, 2)

    # --- snow level and freezing level (lapse rate, km -> m)
    df["Snow_Level_m"] = np.round(SUMMIT_ELEV_M + (tw - t50) / lapse * 1000.0, 0)
    df["Freezing_Level_m"] = np.round(SUMMIT_ELEV_M + t_crest / lapse * 1000.0, 0)
    df["Summit_Elev_m"] = SUMMIT_ELEV_M
    # depth of above-freezing air that falling flakes cross while partly melting
    df["Melt_Layer_Depth_m"] = np.round(df["Freezing_Level_m"] - df["Snow_Level_m"], 0)

    # --- surface energy balance of a melting (0 °C) snow surface, W/m2
    t_k = t_crest + 273.15
    rho_air = p_hpa * 100.0 / (R_AIR * t_k)
    ch_coef = ch * 1e-3
    q_air = 0.622 * e_hpa / (p_hpa - 0.378 * e_hpa)
    e0 = sat_vp_water(0.0)
    q_sfc = 0.622 * e0 / (p_hpa - 0.378 * e0)
    h_sens = rho_air * CP_AIR * ch_coef * u * t_crest
    h_lat = rho_air * LV * ch_coef * u * (q_air - q_sfc)
    eps_clear = 1.24 * (e_hpa / t_k) ** (1.0 / 7.0)                  # Brutsaert (1975)
    eps_sky = np.minimum(1.0, eps_clear * (1.0 + 0.22 * cloud ** 2))
    lw_net = SNOW_EMISS * (eps_sky * SIGMA * t_k ** 4 - SIGMA * 273.15 ** 4)
    sw_abs = (1.0 - WET_SNOW_ALBEDO) * sw
    rain_flux = precip * (1.0 - p_snow) / 3600.0                      # kg m-2 s-1
    q_rain = rain_flux * CW * np.maximum(tw, 0.0)
    q_net = h_sens + h_lat + lw_net + sw_abs + q_rain
    df["Sensible_W"] = np.round(h_sens, 1)
    df["Latent_W"] = np.round(h_lat, 1)
    df["LW_Net_W"] = np.round(lw_net, 1)
    df["SW_Abs_W"] = np.round(sw_abs, 1)
    df["Rain_Heat_W"] = np.round(q_rain, 1)
    df["Q_Net_W"] = np.round(q_net, 1)
    melt = np.maximum(q_net, 0.0) * 3600.0 / LF
    df["Melt_mm_h"] = np.round(melt, 3)
    df["Refreeze_Capacity_mm_h"] = np.round(np.maximum(-q_net, 0.0) * 3600.0 / LF, 3)

    # --- snowpack mass balance over the window: SWE(t) = max(SWE(t-1) + snow - melt, 0)
    snow_we = precip * p_snow                                          # mm w.e. per hour
    swe = np.zeros(len(df))
    pack = 0.0
    for i in range(len(df)):
        pack = max(pack + snow_we[i] - melt[i], 0.0)
        swe[i] = pack
    df["Snowpack_SWE_mm"] = np.round(swe, 2)
    df["Snowpack_Depth_cm"] = np.round(swe / PACK_DENSITY * 100.0, 2)

    # --- wet-snow accretion on a cylinder (ISO 12494-style)
    wet_window = (tw >= WET_WINDOW[0]) & (tw <= WET_WINDOW[1])
    stick = np.where(wet_window, 1.0, np.where(tw < WET_WINDOW[0], 0.1, 0.0))
    alpha2 = np.minimum(1.0, 1.0 / u)
    conc = (precip * p_snow / 3600.0) / SNOW_FALL_SPEED               # kg m-3
    v_tot = np.sqrt(u ** 2 + SNOW_FALL_SPEED ** 2)
    df["Accretion_kg_m_h"] = np.round(3600.0 * stick * alpha2 * conc * v_tot * diam_m, 4)

    # --- surface (leaf/grass) temperature with the original radiative knob
    is_night = (df["Hour"] <= 6) | (df["Hour"] >= 19)
    clear_factor = np.where(is_night, (1.0 - cloud), 0.0)
    df["Leaf_Surface_Temp_C"] = np.round(t - chill - clear_factor * leaf_drop, 2)

    df["Airmass_Origin"] = df["Wind_Deg"].apply(classify_wind)
    df["Summit_Verdict"] = df.apply(get_verdict, axis=1)
    return df


# --------------------------------------------------------------------- data
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_live(start_date, end_date):
    """Cached API call. Raises on failure so errors are never cached."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "elevation": SUMMIT_ELEV_M,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,snowfall,"
                  "wind_speed_10m,wind_direction_10m,cloud_cover,shortwave_radiation",
        "timezone": "Asia/Dhaka",
    }
    last_err = ""
    for _ in range(3):
        try:
            response = requests.get(API_URL, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "hourly" in data:
                    return data
            last_err = f"HTTP {response.status_code}: {response.text[:150]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.5)
    raise RuntimeError(last_err)


def fallback_data():
    return {
        "hourly": {
            "time": [
                "2018-01-08T03:00", "2018-01-08T04:00", "2018-01-08T05:00", "2018-01-08T06:00",
                "2018-01-09T02:00", "2018-01-09T03:00", "2018-01-09T04:00", "2018-01-09T05:00", "2018-01-09T06:00",
                "2018-01-10T03:00", "2018-01-10T04:00", "2018-01-10T05:00", "2018-01-10T06:00",
            ],
            "temperature_2m": [5.8, 5.4, 5.1, 5.2, 3.6, 3.3, 3.1, 2.9, 3.0, 4.1, 3.8, 3.5, 3.7],
            "relative_humidity_2m": [62, 65, 68, 67, 90, 92, 94, 95, 91, 55, 58, 60, 57],
            "precipitation": [0.0] * 13,
            "snowfall": [0.0] * 13,
            "wind_speed_10m": [14.2, 15.0, 15.5, 14.8, 16.5, 17.2, 18.0, 17.5, 16.0, 12.5, 13.0, 13.4, 12.8],
            "wind_direction_10m": [350, 355, 0, 5, 160, 165, 170, 170, 165, 340, 345, 350, 355],
            "cloud_cover": [10, 15, 10, 10, 85, 90, 95, 95, 80, 5, 5, 0, 5],
            "shortwave_radiation": [0.0] * 13,
        }
    }


def load_data(start_date, end_date):
    try:
        return fetch_live(start_date, end_date), "LIVE SYNC: Open-Meteo ERA5 reanalysis (1,063 m summit)"
    except Exception as e:
        return fallback_data(), (
            f"OFFLINE BACKUP ACTIVE ({e}): showing a small ILLUSTRATIVE sample dataset, "
            "not real observations."
        )


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def fetch_daily_history(first_year, last_year):
    """Daily ERA5 record (Dec of first_year-1 through Feb of last_year). Raises on failure."""
    end = (pd.Timestamp(last_year, 3, 1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "latitude": LAT,
        "longitude": LON,
        "elevation": SUMMIT_ELEV_M,
        "start_date": f"{first_year - 1}-12-01",
        "end_date": end,
        "daily": "temperature_2m_mean,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Dhaka",
    }
    last_err = ""
    for _ in range(3):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if "daily" in data:
                    return data["daily"]
            last_err = f"HTTP {response.status_code}: {response.text[:150]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.5)
    raise RuntimeError(last_err)


def find_extremes(daily, window=7):
    """Find the rainiest Dec-Feb winter and the coldest `window`-day spell in the record."""
    d = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "tmean": pd.to_numeric(pd.Series(daily["temperature_2m_mean"]), errors="coerce"),
        "precip": pd.to_numeric(pd.Series(daily["precipitation_sum"]), errors="coerce"),
    }).set_index("date").sort_index()

    in_winter = d.index.month.isin([12, 1, 2])
    d["wyear"] = d.index.year + (d.index.month == 12).astype(int)
    w = d[in_winter]

    # rainiest complete winter (skip partial winters at the ends of the record)
    totals = w.groupby("wyear")["precip"].sum(min_count=1)
    counts = w.groupby("wyear")["precip"].count()
    totals = totals[counts >= 85].dropna()
    rain_year = int(totals.idxmax())

    # coldest rolling window fully inside a winter
    roll = d["tmean"].rolling(window).mean()
    starts = roll.index - pd.Timedelta(days=window - 1)
    mask = in_winter & starts.month.isin([12, 1, 2]) & roll.notna().to_numpy()
    cold_end = roll[mask].idxmin()
    cold_start = cold_end - pd.Timedelta(days=window - 1)

    return {
        "rain_year": rain_year,
        "rain_total_mm": float(totals.loc[rain_year]),
        "rain_start": f"{rain_year - 1}-12-01",
        "rain_end": (pd.Timestamp(rain_year, 3, 1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "cold_start": cold_start.strftime("%Y-%m-%d"),
        "cold_end": cold_end.strftime("%Y-%m-%d"),
        "cold_mean_c": float(roll.loc[cold_end]),
        "n_winters": int(len(totals)),
    }


def find_wet_snow_days(daily, top_n=10, min_precip=1.0, cold_pct=25, lapse=DEFAULT_LAPSE):
    """Rank winter days where it rains, it is very cold, and the freezing level
    sits closest to the summit.

    Filters: Dec-Feb, precipitation >= min_precip mm, daily-mean temperature in the
    coldest `cold_pct` percent of winter days. Rank: |freezing level - summit| ascending
    (freezing level from daily minimum temperature and the lapse rate), wetter first on ties.
    """
    d = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "tmin": pd.to_numeric(pd.Series(daily["temperature_2m_min"]), errors="coerce"),
        "tmean": pd.to_numeric(pd.Series(daily["temperature_2m_mean"]), errors="coerce"),
        "precip": pd.to_numeric(pd.Series(daily["precipitation_sum"]), errors="coerce"),
    }).set_index("date").sort_index()
    w = d[d.index.month.isin([12, 1, 2])].dropna()
    thr = w["tmean"].quantile(cold_pct / 100.0)
    cand = w[(w["precip"] >= min_precip) & (w["tmean"] <= thr)].copy()
    cand["Above_Summit_m"] = cand["tmin"] / lapse * 1000.0
    cand["Freezing_Level_m"] = SUMMIT_ELEV_M + cand["Above_Summit_m"]
    cand["abs_gap"] = cand["Above_Summit_m"].abs()
    cand = cand.sort_values(["abs_gap", "precip"], ascending=[True, False]).head(top_n)
    out = cand.reset_index()
    out["Date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.rename(columns={"tmin": "Tmin_C", "tmean": "Tmean_C", "precip": "Precip_mm"})
    return out[["Date", "Tmin_C", "Tmean_C", "Precip_mm", "Freezing_Level_m", "Above_Summit_m"]].round(1)


def last_complete_winter_year():
    today = pd.Timestamp.today()
    return today.year if (today.month, today.day) >= (3, 10) else today.year - 1


def build_dataframe(hourly):
    n = len(hourly["time"])

    def col(name, default=None):
        vals = hourly.get(name, [default] * n)
        return pd.to_numeric(pd.Series(vals), errors="coerce")

    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(hourly["time"]),
        "Air_Temp_1063m_C": col("temperature_2m"),
        "RH_pct": col("relative_humidity_2m"),
        "Precip_mm": col("precipitation"),
        "Snowfall_Model_cm": col("snowfall"),
        "Wind_kmh": col("wind_speed_10m"),
        "Wind_Deg": col("wind_direction_10m"),
        "Cloud_pct": col("cloud_cover"),
        "Shortwave_Wm2": col("shortwave_radiation", 0.0),
    })
    df = df.dropna(subset=["Air_Temp_1063m_C", "RH_pct"]).reset_index(drop=True)
    for c, v in [("Precip_mm", 0.0), ("Snowfall_Model_cm", 0.0), ("Wind_kmh", 0.0),
                 ("Wind_Deg", 180.0), ("Cloud_pct", 50.0), ("Shortwave_Wm2", 0.0)]:
        df[c] = df[c].fillna(v)
    return df


# ----------------------------------------------------------------------- UI
st.title("Saka Haphong (1,063m) True Summit Cryosphere & Wet-Snow Detector")
st.markdown("**Hindu Kush Himalaya Outpost (`21.78°N, 92.61°E`)** — ERA5 summit microphysics & frost/wet-snow analyzer.")
st.divider()

st.sidebar.header("1. Winter Season Selection")
AUTO_RAIN = "Auto: Rainiest Winter on Record (ERA5 1991-present)"
AUTO_COLD = "Auto: Coldest 7-Day Cold Wave on Record (ERA5 1991-present)"
AUTO_WET = "Auto: Wet-Snow Setup Days (freezing layer at summit + rain + very cold)"
season_map = {
    AUTO_WET: ("auto", "auto"),
    AUTO_RAIN: ("auto", "auto"),
    AUTO_COLD: ("auto", "auto"),
    "Winter 2017-2018 (Record 2.9°C Cold Wave)": ("2017-12-01", "2018-02-28"),
    "Winter 2012-2013 (Historic Jan 2013 Surge)": ("2012-12-01", "2013-02-28"),
    "Winter 2010-2011 (Coldest Drizzle Event)": ("2010-12-01", "2011-02-28"),
    "Winter 2023-2024 (Jan 24 Cold Rainstorm)": ("2023-12-01", "2024-02-29"),
    "Winter 2025-2026 (Recent Winter Season)": ("2025-12-01", "2026-02-28"),
    "Custom Date Range": ("custom", "custom"),
}

selected_season = st.sidebar.selectbox("Choose Winter Season:", list(season_map.keys()))
wet_table = None

if selected_season == "Custom Date Range":
    start_d = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01").date())
    end_d = st.sidebar.date_input("End Date", pd.to_datetime("2018-01-31").date())
    if start_d > end_d:
        st.error("Start date must be on or before the end date.")
        st.stop()
    start_d, end_d = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
elif selected_season == AUTO_WET:
    with st.spinner("Scanning the ERA5 winter record for wet-snow setups..."):
        try:
            wet_table = find_wet_snow_days(fetch_daily_history(1991, last_complete_winter_year()))
        except Exception as e:
            st.error(f"Could not scan the historical record ({e}). Pick another season or try again.")
            st.stop()
    if wet_table.empty:
        st.error("No rainy, very cold winter days were found in the record.")
        st.stop()
    rank = st.sidebar.selectbox("Candidate day (1 = closest freezing layer):", list(range(1, len(wet_table) + 1)))
    pick = wet_table.iloc[rank - 1]
    day = pd.Timestamp(pick["Date"])
    start_d = (day - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end_d = (day + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    st.sidebar.caption(
        f"Centered on {pick['Date']}: Tmin {pick['Tmin_C']:.1f} °C, {pick['Precip_mm']:.1f} mm rain, "
        f"freezing level ≈ {pick['Freezing_Level_m']:.0f} m ({pick['Above_Summit_m']:+.0f} m vs summit). "
        "Window shows 3 days either side."
    )
elif selected_season in (AUTO_RAIN, AUTO_COLD):
    with st.spinner("Scanning the ERA5 winter record for extremes..."):
        try:
            ext = find_extremes(fetch_daily_history(1991, last_complete_winter_year()))
        except Exception as e:
            st.error(f"Could not scan the historical record ({e}). Pick another season or try again.")
            st.stop()
    if selected_season == AUTO_RAIN:
        start_d, end_d = ext["rain_start"], ext["rain_end"]
        st.sidebar.caption(
            f"Rainiest of {ext['n_winters']} winters: {ext['rain_year'] - 1}-{str(ext['rain_year'])[2:]} "
            f"({ext['rain_total_mm']:.0f} mm, ERA5 at 1,063 m)"
        )
    else:
        start_d, end_d = ext["cold_start"], ext["cold_end"]
        st.sidebar.caption(
            f"Coldest 7-day spell: {start_d} to {end_d} "
            f"(mean {ext['cold_mean_c']:.1f} °C, ERA5 at 1,063 m)"
        )
else:
    start_d, end_d = season_map[selected_season]

st.sidebar.divider()
st.sidebar.header("2. Summit Microclimate Physics")
cliff_uplift_factor = st.sidebar.slider("Border Cliff Adiabatic Uplift (°C per 10 km/h wind)", 0.0, 1.0, 0.35, 0.05)
leaf_radiative_drop = st.sidebar.slider("Clear-Sky Leaf/Grass Radiative Cooling (°C)", 0.0, 3.5, 2.0, 0.25)
predawn_only = st.sidebar.checkbox("Filter Pre-Dawn Window Only (02:00 – 06:00 BST)", value=False)
rain_or_cold_only = st.sidebar.checkbox("Show Only Cold (<6°C) or Rainy Hours", value=True)

st.sidebar.divider()
st.sidebar.header("3. Wet-Snow Physics Parameters")
t50 = st.sidebar.slider("Rain/Snow 50% Wet-Bulb Threshold (°C)", 0.0, 2.5, 1.0, 0.1)
phase_scale = st.sidebar.slider("Phase Transition Width (°C)", 0.2, 1.5, 0.5, 0.1)
lapse = st.sidebar.slider("Lapse Rate for Snow Level (°C/km)", 4.0, 9.8, 6.0, 0.1)
bamboo_cm = st.sidebar.slider("Bamboo / Branch Diameter (cm)", 1.0, 15.0, 3.0, 0.5)
ch_1e3 = st.sidebar.slider("Bulk Transfer Coefficient C_H = C_E (×10⁻³)", 0.8, 3.0, 1.5, 0.1)

with st.spinner("Syncing 1,063m summit telemetry..."):
    raw, status_msg = load_data(start_d, end_d)

st.caption(status_msg)

if wet_table is not None:
    st.subheader("Top Wet-Snow Setup Days on Record")
    st.caption(
        "Winter days with at least 1 mm of rain, daily-mean temperature in the coldest 25% of winter days, "
        f"ranked by how close the freezing level (daily minimum temperature, {DEFAULT_LAPSE:.0f} °C/km) "
        "sits to the 1,063 m summit."
    )
    st.dataframe(wet_table, use_container_width=True)

df = build_dataframe(raw["hourly"])
if df.empty:
    st.warning("No data returned for the selected period.")
    st.stop()

df = compute_physics(df, cliff_uplift_factor, leaf_radiative_drop, t50, phase_scale,
                     lapse, bamboo_cm / 100.0, ch_1e3)

snow_hours = (df["Precip_mm"] > 0) & (df["Snow_Prob"] >= 0.5)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Lowest 2m Air Temp (1,063m)", f"{df['Air_Temp_1063m_C'].min():.1f} °C")
col2.metric("Lowest Crest Wet-Bulb", f"{df['Crest_Wet_Bulb_C'].min():.1f} °C")
col3.metric("Lowest Bamboo/Rock Surface", f"{df['Leaf_Surface_Temp_C'].min():.1f} °C")
col4.metric("Sub-Zero Frost/Rime Hours", f"{int((df['Leaf_Surface_Temp_C'] <= 0.0).sum())} hrs")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Snow-Capable Precip Hours", f"{int(snow_hours.sum())} hrs")
col6.metric("Derived Snowfall Total", f"{df['Snowfall_Derived_cm'].sum():.1f} cm")
col7.metric("Lowest Snow Level", f"{df['Snow_Level_m'].min():.0f} m")
col8.metric("Peak Wet-Snow Accretion", f"{df['Accretion_kg_m_h'].max():.3f} kg/m/h")

col9, col10, col11, col12 = st.columns(4)
col9.metric("Peak Snowpack Depth", f"{df['Snowpack_Depth_cm'].max():.1f} cm")
col10.metric("Deepest Melt Layer", f"{df['Melt_Layer_Depth_m'].max():.0f} m")
col11.metric("Peak Melt Rate", f"{df['Melt_mm_h'].max():.2f} mm/h")
col12.metric("Lowest Freezing Level", f"{df['Freezing_Level_m'].min():.0f} m")

if not snow_hours.any():
    st.info(
        "No hour in this window combined precipitation with a wet-bulb temperature low enough "
        "for snow at the summit. That is the physically expected result for most seasons here."
    )

view_df = df.copy()
if predawn_only:
    view_df = view_df[(view_df["Hour"] >= 2) & (view_df["Hour"] <= 6)]
if rain_or_cold_only:
    view_df = view_df[(view_df["Air_Temp_1063m_C"] <= 6.0) | (view_df["Precip_mm"] > 0)]

st.subheader(f"Hourly Summit Physics Ledger ({len(view_df)} Matching Hours)")
display_cols = [
    "Timestamp", "Air_Temp_1063m_C", "Crest_Wet_Bulb_C", "Leaf_Surface_Temp_C",
    "RH_pct", "Precip_mm", "Snow_Prob", "SLR", "Snowfall_Derived_cm", "Snow_Level_m",
    "Freezing_Level_m", "Melt_Layer_Depth_m", "Melt_mm_h", "Snowpack_Depth_cm",
    "Accretion_kg_m_h", "Wind_kmh", "Airmass_Origin", "Summit_Verdict",
]
st.dataframe(view_df[display_cols], use_container_width=True)

st.subheader("Summit Temperature vs. Leaf Surface Freeze Line")
st.line_chart(df.set_index("Timestamp")[["Air_Temp_1063m_C", "Crest_Wet_Bulb_C", "Leaf_Surface_Temp_C"]])

st.subheader("Snow Level vs. Summit Elevation (m)")
st.line_chart(df.set_index("Timestamp")[["Snow_Level_m", "Freezing_Level_m", "Summit_Elev_m"]])

st.subheader("Snow-Surface Energy Balance (W/m²)")
st.line_chart(df.set_index("Timestamp")[["Sensible_W", "Latent_W", "LW_Net_W", "SW_Abs_W", "Rain_Heat_W", "Q_Net_W"]])

with st.expander("Physics reference: laws used for wet snow"):
    st.markdown(
        """
**1. Saturation vapour pressure (Clausius–Clapeyron, Magnus form)**
`es(T) = 6.112 · exp(17.62 T / (243.12 + T))` hPa, with `e = RH · es(T)` and dew point from the inverse.

**2. Psychrometric equation (wet-bulb)**
`e = es(Tw) − A · p · (T − Tw)`, `A = 6.6×10⁻⁴ (1 + 1.15×10⁻³ Tw)`, solved numerically at the summit pressure
(about 900 hPa at 1,063 m). Evaporative cooling of a falling flake is what lets snow survive above 0 °C air temperature.

**3. Rain/snow partition**
`P(snow) = 1 / (1 + exp((Tw − T50) / s))`. An empirical logistic fit; T50 and s are adjustable in the sidebar.

**4. Fresh-snow density (Hedstrom & Pomeroy 1998)**
`ρ = 67.92 + 51.25 · exp(T / 2.59)` kg/m³. Snow-to-liquid ratio `SLR = 1000 / ρ`, snowfall depth `= precip · P(snow) · SLR / 10` cm.

**5. Snow level and freezing level (lapse rate)**
`z_snow = z + (Tw − T50) / Γ`, `z_0°C = z + T / Γ`. Temperature falls with height at Γ, so the Tw threshold sits higher or lower than the summit.

**6. Surface energy balance of melting snow (0 °C surface)**
`Q = H + LE + LW_net + SW_abs + Q_rain`
- Sensible: `H = ρ_air · c_p · C_H · u · (T − 0)`
- Latent: `LE = ρ_air · L_v · C_E · u · (q_air − q_sat(0 °C))` (condensation on wet snow releases heat)
- Longwave: `LW_net = ε_s (ε_sky σ T⁴ − σ 273.15⁴)`, with clear-sky `ε = 1.24 (e/T)^(1/7)` (Brutsaert) and cloud factor `(1 + 0.22 N²)`
- Shortwave: `SW_abs = (1 − α) · SW↓`, wet-snow albedo α ≈ 0.6
- Rain heat: `Q_rain = ṁ_rain · c_w · Tw`

**7. Melt rate**
`M = Q / (ρ_w · L_f)`, i.e. `mm w.e./h = Q · 3600 / 3.34×10⁵` for `Q > 0`.

**8. Wet-snow accretion on a cylinder (ISO 12494 form)**
`dM/dt = α₁ · α₂ · w · V · D`, with `w = P / v_fall`, `V = √(u² + v_fall²)`, `α₂ = min(1, 1/u)`.
Stickiness is full when Tw is between −0.5 and +1.5 °C (liquid-water film bonds flakes), and near zero for rain.

**9. Melting-layer depth**
`D = z_0°C − z_snow = (T − Tw + T50) / Γ`. The thickness of above-freezing air a falling flake crosses while partly melting.
A thin layer means snow reaches the summit wet; a thick one means it arrives as rain.

**10. Snowpack mass balance and refreeze**
`SWE(t) = max(SWE(t−1) + P·P(snow) − M, 0)`, depth `= SWE / ρ_pack` (ρ_pack ≈ 250 kg/m³ for wet settled snow).
When `Q < 0` the surface has refreeze capacity `−Q · 3600 / L_f` mm/h, which is how a wet pack ripens and then glazes over.

**11. Wet-snow setup day finder**
Winter days with ≥ 1 mm rain and daily-mean temperature in the coldest 25%, ranked by `|z_0°C − z_summit|`, with `z_0°C` from the daily minimum temperature and the lapse rate.

*Coefficients (T50, density fit, C_H, albedo, accretion window) are empirical approximations, not exact constants.*
        """
    )

csv_bytes = view_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Summit Physics CSV",
    data=csv_bytes,
    file_name=f"saka_haphong_1063m_{start_d}_to_{end_d}.csv",
    mime="text/csv",
)
