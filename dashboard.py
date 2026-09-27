import sqlite3
import pandas as pd
import streamlit as st
from PIL import Image
import config

st.set_page_config(page_title="Traffic Violations Dashboard", layout="wide")
st.title("Traffic Violations Dashboard")

conn = sqlite3.connect(config.LOG_DB_PATH)
df = pd.read_sql_query("SELECT * FROM violations ORDER BY timestamp DESC", conn)

if df.empty:
    st.info("No violations logged yet. Run main.py on a video first.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total violations", len(df))
    col2.metric("Avg. speed (km/h)", f"{df['speed_kmh'].mean():.1f}")
    col3.metric("Top speed (km/h)", f"{df['speed_kmh'].max():.1f}")

    st.dataframe(
        df[["track_id", "plate_text", "speed_kmh", "timestamp"]],
        use_container_width=True,
    )

    st.subheader("Snapshots")
    for _, row in df.iterrows():
        try:
            img = Image.open(row["snapshot_path"])
            st.image(
                img,
                caption=f"ID {row['track_id']} - {row['plate_text']} - {row['speed_kmh']:.0f} km/h",
                width=250,
            )
        except FileNotFoundError:
            continue
