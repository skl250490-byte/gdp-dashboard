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


def wet_bulb_stull(temp_c, rh_pct):
    """Wet-bulb temperature (°C) from air temp and RH using Stull (2011)."""
    t = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


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
                  "wind_speed_10m,wind_direction_10m,cloud_cover",
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


st.title("Saka Haphong (1,063m) True Summit Cryosphere & Wet-Snow Detector")
st.markdown("**Hindu Kush Himalaya Outpost (`21.78°N, 92.61°E`)** — ERA5 summit microphysics & frost/wet-snow analyzer.")
st.divider()

st.sidebar.header("1. Winter Season Selection")
season_map = {
    "Winter 2017-2018 (Record 2.9°C Cold Wave)": ("2017-12-01", "2018-02-28"),
    "Winter 2012-2013 (Historic Jan 2013 Surge)": ("2012-12-01", "2013-02-28"),
    "Winter 2010-2011 (Coldest Drizzle Event)": ("2010-12-01", "2011-02-28"),
    "Winter 2023-2024 (Jan 24 Cold Rainstorm)": ("2023-12-01", "2024-02-29"),
    "Winter 2025-2026 (Recent Winter Season)": ("2025-12-01", "2026-02-28"),
    "Custom Date Range": ("custom", "custom"),
}

selected_season = st.sidebar.selectbox("Choose Winter Season:", list(season_map.keys()))

if selected_season == "Custom Date Range":
    start_d = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01").date())
    end_d = st.sidebar.date_input("End Date", pd.to_datetime("2018-01-31").date())
    if start_d > end_d:
        st.error("Start date must be on or before the end date.")
        st.stop()
    start_d, end_d = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
else:
    start_d, end_d = season_map[selected_season]

st.sidebar.divider()
st.sidebar.header("2. Summit Microclimate Physics")
cliff_uplift_factor = st.sidebar.slider("Border Cliff Adiabatic Uplift (°C per 10 km/h wind)", 0.0, 1.0, 0.35, 0.05)
leaf_radiative_drop = st.sidebar.slider("Clear-Sky Leaf/Grass Radiative Cooling (°C)", 0.0, 3.5, 2.0, 0.25)
predawn_only = st.sidebar.checkbox("Filter Pre-Dawn Window Only (02:00 – 06:00 BST)", value=False)
rain_or_cold_only = st.sidebar.checkbox("Show Only Cold (<6°C) or Rainy Hours", value=True)

with st.spinner("Syncing 1,063m summit telemetry..."):
    raw, status_msg = load_data(start_d, end_d)

st.caption(status_msg)

hourly = raw["hourly"]
df = pd.DataFrame({
    "Timestamp": pd.to_datetime(hourly["time"]),
    "Air_Temp_1063m_C": pd.to_numeric(pd.Series(hourly["temperature_2m"]), errors="coerce"),
    "RH_pct": pd.to_numeric(pd.Series(hourly["relative_humidity_2m"]), errors="coerce"),
    "Precip_mm": pd.to_numeric(pd.Series(hourly["precipitation"]), errors="coerce"),
    "Snowfall_cm": pd.to_numeric(pd.Series(hourly["snowfall"]), errors="coerce"),
    "Wind_kmh": pd.to_numeric(pd.Series(hourly["wind_speed_10m"]), errors="coerce"),
    "Wind_Deg": pd.to_numeric(pd.Series(hourly["wind_direction_10m"]), errors="coerce"),
    "Cloud_pct": pd.to_numeric(pd.Series(hourly["cloud_cover"]), errors="coerce"),
})

# Drop hours with missing core values, fill remaining gaps with neutral defaults
df = df.dropna(subset=["Air_Temp_1063m_C", "RH_pct"]).reset_index(drop=True)
df["Precip_mm"] = df["Precip_mm"].fillna(0.0)
df["Snowfall_cm"] = df["Snowfall_cm"].fillna(0.0)
df["Wind_kmh"] = df["Wind_kmh"].fillna(0.0)
df["Wind_Deg"] = df["Wind_Deg"].fillna(180.0)
df["Cloud_pct"] = df["Cloud_pct"].fillna(50.0)

if df.empty:
    st.warning("No data returned for the selected period.")
    st.stop()

df["Wet_Bulb_C"] = np.round(wet_bulb_stull(df["Air_Temp_1063m_C"], df["RH_pct"]), 2)
df["Hour"] = df["Timestamp"].dt.hour
df["Adiabatic_Chill_C"] = np.round((df["Wind_kmh"] / 10.0) * cliff_uplift_factor, 2)

is_night = (df["Hour"] <= 6) | (df["Hour"] >= 19)
clear_factor = np.where(is_night, (100.0 - df["Cloud_pct"]) / 100.0, 0.0)
df["Leaf_Surface_Temp_C"] = np.round(
    df["Air_Temp_1063m_C"] - df["Adiabatic_Chill_C"] - (clear_factor * leaf_radiative_drop), 2
)
df["Crest_Wet_Bulb_C"] = np.round(df["Wet_Bulb_C"] - df["Adiabatic_Chill_C"], 2)


def classify_wind(deg):
    if deg >= 315 or deg <= 45:
        return "N/NW Himalayan Cold Surge"
    elif 135 <= deg <= 225:
        return "S/SE Bay of Bengal Moisture"
    return "Transitional Ridge Flow"


df["Airmass_Origin"] = df["Wind_Deg"].apply(classify_wind)


def get_verdict(row):
    t_crest = row["Air_Temp_1063m_C"] - row["Adiabatic_Chill_C"]
    tw = row["Crest_Wet_Bulb_C"]
    t_leaf = row["Leaf_Surface_Temp_C"]
    precip = row["Precip_mm"]
    rh = row["RH_pct"]

    if precip > 0:
        if tw <= 0.5 and t_crest <= 1.5:
            return "STICKY WET SNOW / SLEET UNDERCUT (5:1 Water Fuel)"
        elif tw <= 2.5:
            return "NEAR-FREEZING COLD RAIN / GRAUPEL RISK"
        else:
            return "LIQUID WINTER RAIN (Warm Layer > 200m)"
    else:
        if t_leaf <= 0.0 and rh >= 75:
            return "RIME ICE & HOARFROST ACCRETION ON BAMBOO/ROCK"
        elif t_leaf <= 0.0:
            return "DRY RADIATIVE FROST ON GRASS & ROCK"
        elif t_crest <= 4.0:
            return "SEVERE COLD-WAVE CHILL (Near-Frost Threshold)"
        return "DRY WINTER RIDGE"


df["Summit_Verdict"] = df.apply(get_verdict, axis=1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Lowest 2m Air Temp (1,063m)", f"{df['Air_Temp_1063m_C'].min():.1f} °C")
col2.metric("Lowest Crest Wet-Bulb", f"{df['Crest_Wet_Bulb_C'].min():.1f} °C")
col3.metric("Lowest Bamboo/Rock Surface", f"{df['Leaf_Surface_Temp_C'].min():.1f} °C")
col4.metric("Sub-Zero Frost/Rime Hours", f"{int((df['Leaf_Surface_Temp_C'] <= 0.0).sum())} hrs")

view_df = df.copy()
if predawn_only:
    view_df = view_df[(view_df["Hour"] >= 2) & (view_df["Hour"] <= 6)]
if rain_or_cold_only:
    view_df = view_df[(view_df["Air_Temp_1063m_C"] <= 6.0) | (view_df["Precip_mm"] > 0)]

st.subheader(f"Hourly Summit Physics Ledger ({len(view_df)} Matching Hours)")
display_cols = [
    "Timestamp", "Air_Temp_1063m_C", "Crest_Wet_Bulb_C", "Leaf_Surface_Temp_C",
    "RH_pct", "Precip_mm", "Wind_kmh", "Airmass_Origin", "Summit_Verdict",
]
st.dataframe(view_df[display_cols], use_container_width=True)

st.subheader("Summit Temperature vs. Leaf Surface Freeze Line")
st.line_chart(df.set_index("Timestamp")[["Air_Temp_1063m_C", "Crest_Wet_Bulb_C", "Leaf_Surface_Temp_C"]])

csv_bytes = view_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Summit Physics CSV",
    data=csv_bytes,
    file_name=f"saka_haphong_1063m_{start_d}_to_{end_d}.csv",
    mime="text/csv",
)
