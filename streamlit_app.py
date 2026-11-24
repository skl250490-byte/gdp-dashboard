import streamlit as st
import pandas as pd
import numpy as np
import requests
import math

st.set_page_config(page_title="Saka Haphong Summit Cryosphere Detector", layout="wide")

# --- CONSTANTS FOR SAKA HAPHONG (HINDU KUSH HIMALAYA OUTPOST) ---
LAT = 21.7865
LON = 92.6101
SUMMIT_ELEV_M = 1063.0
RIDGE_AREA_M2 = 50000.0  # ~5 hectares upper crest above 1,000m

def calc_wet_bulb(temp_c, rh_pct):
    """Stull (2011) thermodynamic wet-bulb approximation (accurate within 0.3C)."""
    tw = (
        temp_c * math.atan(0.151977 * math.sqrt(rh_pct + 8.313659))
        + math.atan(temp_c + rh_pct)
        - math.atan(rh_pct - 1.676331)
        + 0.00391838 * (rh_pct ** 1.5) * math.atan(0.023101 * rh_pct)
        - 4.686035
    )
    return round(tw, 2)

@st.cache_data(ttl=86400)
def fetch_true_summit_winter_data(start_date, end_date):
    """Fetches hourly ERA5 reanalysis data downscaled to 1,063m summit elevation."""
    url = "https://open-meteo.com"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "elevation": SUMMIT_ELEV_M,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wet_bulb_temperature_2m",
            "precipitation",
            "rain",
            "snowfall",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
            "weather_code"
        ],
        "timezone": "Asia/Dhaka"
    }
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    return response.json()

# --- UI HEADER ---
st.title("Saka Haphong (1,063m) True Summit Cryosphere & Wet-Snow Detector")
st.markdown("""
**Official Hindu Kush Himalaya Outpost (`21.78°N, 92.61°E`)** — Live ERA5 atmospheric reanalysis engine scanning 
winter cold waves (`Dec–Feb`), pre-dawn undercut windows (`03:00–06:00 BST`), adiabatic wind-ramp chill, 
and seasonal meltwater storage.
""")

st.divider()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Winter Season Selection")
season_map = {
    "Winter 2017-2018 (Record 2.9°C Cold Wave)": ("2017-12-01", "2018-02-28"),
    "Winter 2012-2013 (Historic Jan 2013 Surge)": ("2012-12-01", "2013-02-28"),
    "Winter 2010-2011 (Coldest Drizzle Event)": ("2010-12-01", "2011-02-28"),
    "Winter 2023-2024 (Jan 24 Cold Rainstorm)": ("2023-12-01", "2024-02-29"),
    "Winter 2025-2026 (Recent Winter Season)": ("2025-12-01", "2026-02-28"),
    "Custom Date Range": ("custom", "custom")
}

selected_season = st.sidebar.selectbox("Choose Winter Season:", list(season_map.keys()))

if selected_season == "Custom Date Range":
    start_d = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01")).strftime("%Y-%m-%d")
    end_d = st.sidebar.date_input("End Date", pd.to_datetime("2018-01-31")).strftime("%Y-%m-%d")
else:
    start_d, end_d = season_map[selected_season]

st.sidebar.divider()
st.sidebar.header("2. Summit Microclimate Physics")
cliff_uplift_factor = st.sidebar.slider(
    "Border Cliff Adiabatic Uplift Cooling (°C per 10 km/h wind)",
    min_value=0.0, max_value=1.0, value=0.35, step=0.05,
    help="Accounts for forced vertical air cooling as wind hits the steep Myanmar border wall."
)
leaf_radiative_drop = st.sidebar.slider(
    "Clear-Sky Leaf/Grass Radiative Cooling (°C)",
    min_value=0.0, max_value=3.5, value=2.0, step=0.25,
    help="Bamboo leaves & sungrass cool 1.5°C-3.0°C below 2m air on clear pre-dawn nights."
)
predawn_only = st.sidebar.checkbox("Filter Pre-Dawn Window Only (02:00 – 06:00 BST)", value=False)
rain_or_cold_only = st.sidebar.checkbox("Show Only Cold (<6°C) or Rainy Hours", value=True)

# --- FETCH & PROCESS DATA ---
try:
    with st.spinner("Syncing live 1,063m summit data from Open-Meteo ERA5 archives..."):
        raw = fetch_true_summit_winter_data(start_d, end_d)

    hourly = raw["hourly"]
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(hourly["time"]),
        "Air_Temp_1063m_C": hourly["temperature_2m"],
        "Wet_Bulb_C": hourly["wet_bulb_temperature_2m"],
        "RH_pct": hourly["relative_humidity_2m"],
        "Precip_mm": hourly["precipitation"],
        "Snowfall_cm": hourly["snowfall"],
        "Wind_kmh": hourly["wind_speed_10m"],
        "Wind_Deg": hourly["wind_direction_10m"],
        "Cloud_pct": hourly["cloud_cover"],
        "WMO_Code": hourly["weather_code"]
    })

    df["Hour"] = df["Timestamp"].dt.hour

    # --- SUMMIT MICROPHYSICS CALCULATIONS ---
    # 1. Adiabatic wind-ramp cooling on the knife-edge border ridge
    df["Adiabatic_Chill_C"] = np.round((df["Wind_kmh"] / 10.0) * cliff_uplift_factor, 2)

    # 2. Radiative cooling on bamboo/sungrass (strongest at night when cloud cover < 40%)
    is_night = (df["Hour"] <= 6) | (df["Hour"] >= 19)
    clear_factor = np.where(is_night, (100.0 - df["Cloud_pct"]) / 100.0, 0.0)
    df["Leaf_Surface_Temp_C"] = np.round(
        df["Air_Temp_1063m_C"] - df["Adiabatic_Chill_C"] - (clear_factor * leaf_radiative_drop), 2
    )

    # 3. True Crest Wet-Bulb Temperature (with cliff uplift included)
    df["Crest_Wet_Bulb_C"] = np.round(df["Wet_Bulb_C"] - df["Adiabatic_Chill_C"], 2)

    # 4. Wind Origin Classification
    def classify_wind(deg):
        if deg >= 315 or deg <= 45:
            return "N/NW Himalayan Cold Surge"
        elif 135 <= deg <= 225:
            return "S/SE Bay of Bengal Moisture"
        return "Transitional Ridge Flow"

    df["Airmass_Origin"] = df["Wind_Deg"].apply(classify_wind)

    # 5. Physical Precipitation & Ice Regime Verdict
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

    # --- SUMMARY METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lowest 2m Air Temp (1,063m)", f"{df['Air_Temp_1063m_C'].min():.1f} °C")
    col2.metric("Lowest Crest Wet-Bulb", f"{df['Crest_Wet_Bulb_C'].min():.1f} °C")
    col3.metric("Lowest Bamboo/Rock Surface", f"{df['Leaf_Surface_Temp_C'].min():.1f} °C")
    frost_hours = (df["Leaf_Surface_Temp_C"] <= 0.0).sum()
    col4.metric("Sub-Zero Frost/Rime Hours", f"{frost_hours} hrs")

    # --- FILTERING FOR DISPLAY ---
    view_df = df.copy()
    if predawn_only:
        view_df = view_df[(view_df["Hour"] >= 2) & (view_df["Hour"] <= 6)]
    if rain_or_cold_only:
        view_df = view_df[(view_df["Air_Temp_1063m_C"] <= 6.0) | (view_df["Precip_mm"] > 0)]

    st.subheader(f"Hourly Summit Physics Ledger ({len(view_df)} Matching Hours)")
    display_cols = [
        "Timestamp", "Air_Temp_1063m_C", "Crest_Wet_Bulb_C", "Leaf_Surface_Temp_C",
        "RH_pct", "Precip_mm", "Wind_kmh", "Airmass_Origin", "Summit_Verdict"
    ]
    st.dataframe(view_df[display_cols], use_container_width=True)

    # --- TEMPERATURE & PRECIPITATION CHART ---
    st.subheader("Summit Temperature vs. Leaf Surface Freeze Line")
    chart_df = df.set_index("Timestamp")[["Air_Temp_1063m_C", "Crest_Wet_Bulb_C", "Leaf_Surface_Temp_C"]]
    st.line_chart(chart_df)

    # --- CSV EXPORT ---
    csv_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Summit Physics CSV",
        data=csv_bytes,
        file_name=f"saka_haphong_1063m_{start_d}_to_{end_d}.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"API Connection Error: {e}")
