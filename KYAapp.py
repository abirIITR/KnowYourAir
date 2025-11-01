import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import calendar
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Know Your Air", layout="wide")
st.title("Know Your Air 🌬️")
st.markdown("Your personal dashboard to track and predict the Air Quality Index (AQI) in your city.")

# --- Color Palette ---
PLOTLY_COLORS = ['royalblue', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


# --- Helper Functions ---
def get_aqi_category(aqi):
    """Returns AQI category and color for st.metric."""
    if aqi <= 50:
        return "Good", "normal"
    elif aqi <= 100:
        return "Moderate", "off"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "inverse"
    elif aqi <= 200:
        return "Unhealthy", "inverse"
    elif aqi <= 300:
        return "Very Unhealthy", "inverse"
    else:
        return "Hazardous", "inverse"


def health_advisory(aqi):
    """Returns a simple health advisory message."""
    if aqi <= 50:
        return "Safe to go outside 🌞"
    elif aqi <= 100:
        return "Sensitive people should take care"
    elif aqi <= 150:
        return "Limit prolonged outdoor exposure"
    elif aqi <= 200:
        return "Reduce outdoor activities ❌"
    elif aqi <= 300:
        return "Avoid outdoor exposure ❌"
    else:
        return "Stay indoors ❌"


# --- Load Models and Cities ---
models_dir = "models"
if os.path.exists(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    cities = sorted([f.replace("_prophet_model.pkl", "") for f in model_files])
else:
    st.error("Model directory not found. Please run the training script first.")
    st.stop()


# --- Load Historical Data ---
@st.cache_data
def load_city_data(city):
    """Loads and caches historical data for a given city (real AQI)."""
    df = pd.read_csv("combined_aqi_data.csv", parse_dates=["Date"], dayfirst=True)
    city_df = df[df["City"] == city][["Date", "AQI"]].copy()
    city_df = city_df.rename(columns={"Date": "ds", "AQI": "y"})
    city_df['ds'] = pd.to_datetime(city_df['ds'])
    city_df = city_df.sort_values("ds").reset_index(drop=True)
    return city_df


# --- Sidebar Navigation ---
st.sidebar.title("Controls")
view_mode = st.sidebar.radio(
    "Select Mode",
    ("🏠 Home", "📈 View Past Data", "🔮 Forecast Future AQI"),
    help="Choose whether to explore historical data or get a future forecast."
)

selected_cities = []  # Initialize

# --- Conditional Sidebar Controls ---
if view_mode == '📈 View Past Data':
    selected_cities = st.sidebar.multiselect("Select Cities", cities)

    # rolling average toggle
    show_rolling_avg = st.sidebar.checkbox("Show Rolling Average", value=True)
    if show_rolling_avg:
        rolling_window = st.sidebar.slider(
            "Rolling Average Window (days)",
            min_value=3, max_value=60, value=20, step=1,
            help="Smooths data by averaging over the selected number of days"
        )
    else:
        rolling_window = 20

    if selected_cities:
        # Use first city to populate year/month selectors
        first_city_df = load_city_data(selected_cities[0])
        years = ['All Years'] + sorted(first_city_df['ds'].dt.year.unique(), reverse=True)
        selected_year = st.sidebar.selectbox("Select Year", years)

        # Correct month_map (skip blank at index 0)
        month_map = {calendar.month_name[m]: m for m in range(1, 13)}
        if selected_year == 'All Years':
            selected_month = 'All Months'
        else:
            year_df = first_city_df[first_city_df['ds'].dt.year == selected_year]
            available_months = ['All Months'] + [calendar.month_name[m] for m in sorted(year_df['ds'].dt.month.unique())]
            selected_month = st.sidebar.selectbox("Select Month", available_months, key=f"month_{selected_year}")

elif view_mode == '🔮 Forecast Future AQI':
    selected_cities = st.sidebar.multiselect("Select Cities", cities)

    period_map = {"Next 1 Month": 30, "Next 2 Months": 60, "Next 3 Months": 90, "Next 6 Months": 180,
                  "Next 1 Year": 365, "Next 5 Years": 1825}
    period_option = st.sidebar.selectbox("Forecast Period", list(period_map.keys()))
    days = period_map[period_option]


# --- Main Panel Logic ---
if view_mode == "🏠 Home" or not selected_cities:
    # Welcome Page
    st.subheader("Welcome to Your Air Quality Dashboard")
    st.markdown(
        """
        Air pollution is one of the most significant environmental challenges today.
        This tool is designed to help you understand and anticipate air quality in your city.

        **Select a mode and a city (or multiple cities) from the sidebar on the left to get started.**
        """
    )

    st.subheader("What is the Air Quality Index (AQI)?")
    st.markdown(
        """
        The Air Quality Index (AQI) is your guide to air quality. Think of it like a thermometer that runs from 0 to 500, but for air pollution.
        It's a simple, color-coded scale that helps you understand what the air quality around you means for your health.

        * **0-50 (Good - Green):** Air quality is excellent.
        * **51-100 (Moderate - Yellow):** Air quality is acceptable, but may be a concern for some.
        * **101-150 (Unhealthy for Sensitive Groups - Orange):** People with lung disease, older adults, and children are at greater risk.
        * **151-200 (Unhealthy - Red):** Everyone may begin to experience health effects.
        * **201-300 (Very Unhealthy - Purple):** Health alert. Everyone may experience more serious health effects.
        * **301+ (Hazardous - Maroon):** Health warning of emergency conditions.
        """
    )

    st.stop()


# --- View Past Data ---
if view_mode == '📈 View Past Data':
    st.header("📜 Historical Air Quality Data")
    st.subheader("Average Historical AQI")

    # Create metric columns but guard against zero
    metric_cols = st.columns(len(selected_cities)) if selected_cities else []
    fig = go.Figure()
    all_dfs = []

    for i, city in enumerate(selected_cities):
        city_df = load_city_data(city)
        display_df = city_df.copy()

        # Apply year/month filtering if present
        if 'selected_year' in locals() and selected_year != 'All Years':
            display_df = display_df[display_df['ds'].dt.year == selected_year]
            if 'selected_month' in locals() and selected_month != 'All Months' and selected_month in month_map:
                display_df = display_df[display_df['ds'].dt.month == month_map[selected_month]]

        if not display_df.empty:
            avg_aqi = display_df['y'].mean()
            category, color = get_aqi_category(avg_aqi)
            if metric_cols:
                with metric_cols[i]:
                    st.metric(label=city, value=f"{avg_aqi:.2f}", delta=category, delta_color=color)

            line_color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
            display_df = display_df.sort_values('ds')

            # Dynamic style
            if show_rolling_avg:
                actual_line_width = 1
                actual_opacity = 0.3
            else:
                actual_line_width = 3
                actual_opacity = 1.0

            fig.add_trace(go.Scatter(
                x=display_df['ds'], y=display_df['y'], mode='lines', name=f"{city} (Actual)",
                line=dict(color=line_color, width=actual_line_width),
                opacity=actual_opacity,
                showlegend=True,
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>AQI: %{y:.2f}<extra></extra>'
            ))

            if show_rolling_avg:
                display_df['rolling_avg'] = display_df['y'].rolling(window=rolling_window, min_periods=1, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=display_df['ds'], y=display_df['rolling_avg'], mode='lines',
                    name=f"{city} ({rolling_window}-day Avg)",
                    line=dict(color=line_color, width=3), opacity=1.0, showlegend=True,
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Rolling Avg: %{y:.2f}<extra></extra>'
                ))
            else:
                display_df['rolling_avg'] = pd.NA

            display_df['City'] = city
            all_dfs.append(display_df)

    chart_title = f"Historical AQI with {rolling_window}-Day Rolling Average" if show_rolling_avg else "Historical AQI (Actual)"
    fig.update_layout(
        title=chart_title,
        xaxis_title="Date", yaxis_title="AQI", hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparison Insights (historical)
    if len(selected_cities) > 1 and all_dfs:
        st.subheader("🔎 Comparison Insights")
        avg_list = [(city, df['y'].mean()) for city, df in zip(selected_cities, all_dfs) if not df.empty]
        if avg_list:
            avg_list.sort(key=lambda x: x[1], reverse=True)
            highest_city, highest_val = avg_list[0]
            lowest_city, lowest_val = avg_list[-1]
            st.markdown(f"Highest Average AQI: *{highest_city}* at {highest_val:.2f}")
            st.markdown(f"Lowest Average AQI: *{lowest_city}* at {lowest_val:.2f}")

    if all_dfs:
        combined_df = pd.concat(all_dfs)
        if show_rolling_avg:
            cols_to_show = ['City', 'ds', 'y', 'rolling_avg']
            rename_cols = {'ds': 'Date', 'y': 'AQI', 'rolling_avg': f'{rolling_window}-Day Rolling Average'}
        else:
            cols_to_show = ['City', 'ds', 'y']
            rename_cols = {'ds': 'Date', 'y': 'AQI'}

        csv_data = combined_df[cols_to_show].rename(columns=rename_cols).to_csv(index=False)
        st.download_button("📥 Download Raw Historical Data", csv_data, "historical_data.csv", "text/csv")

        with st.expander("View Raw Historical Data"):
            st.dataframe(combined_df[cols_to_show].rename(columns=rename_cols))

    st.stop()


# --- Forecast Future AQI ---
elif view_mode == '🔮 Forecast Future AQI':
    st.header("🔭 Predict Future Air Quality")
    st.subheader("Predicted Average AQI")
    metric_cols = st.columns(len(selected_cities)) if selected_cities else []
    fig = go.Figure()
    all_forecasts = []

    for i, city in enumerate(selected_cities):
        with st.spinner(f"Forecasting for {city}..."):
            model_path = f"models/{city}_prophet_model.pkl"
            loaded = joblib.load(model_path)

            # Support both new (model,floor,cap) and old (model only) formats
            if isinstance(loaded, tuple):
                model, floor, cap = loaded
            else:
                model = loaded
                # defaults MUST match training if model was trained with floor/cap
                floor = np.log1p(25)
                cap = np.log1p(500)

            city_df = load_city_data(city)
            last_date = city_df['ds'].max()

            # Create future dataframe with required bounds for logistic growth
            future_df = model.make_future_dataframe(periods=days)
            future_df["floor"] = floor
            future_df["cap"] = cap

            forecast = model.predict(future_df)
            forecast['ds'] = pd.to_datetime(forecast['ds'])

            # Convert from log space back to real AQI (clip then expm1)
            forecast['yhat'] = np.expm1(np.clip(forecast['yhat'], floor, cap))
            # also convert uncertainty bounds if present
            if 'yhat_lower' in forecast.columns:
                forecast['yhat_lower'] = np.expm1(np.clip(forecast['yhat_lower'], floor, cap))
            if 'yhat_upper' in forecast.columns:
                forecast['yhat_upper'] = np.expm1(np.clip(forecast['yhat_upper'], floor, cap))

            # Select forecast window (only future days after last observed)
            forecast_future = forecast[forecast['ds'] > last_date].copy()
            # If user wanted a fixed anchor (like a specific pres_start) you'd filter differently.
            forecast_future = forecast_future.sort_values('ds').reset_index(drop=True)

        # Metrics & advisory
        if not forecast_future.empty:
            avg_aqi = forecast_future['yhat'].mean()
            cat, color = get_aqi_category(avg_aqi)
            if metric_cols:
                with metric_cols[i]:
                    st.metric(label=city, value=f"{avg_aqi:.2f}", delta=cat, delta_color=color)
                    st.markdown(f"*Advisory:* {health_advisory(avg_aqi)}")
        else:
            if metric_cols:
                with metric_cols[i]:
                    st.metric(label=city, value="N/A", delta="Out of range")

        # Plot predicted AQI (real scale)
        line_color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'], y=forecast_future['yhat'], mode='lines',
            name=f"{city} (Predicted)", line=dict(color=line_color, dash='solid', width=3),
            opacity=1,
            showlegend=True,
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicted: %{y:.2f}<extra></extra>'
        ))

        # store forecast (converted to real AQI) for insights/downloads
        forecast_future['City'] = city
        all_forecasts.append(forecast_future)

    fig.update_layout(
        title="AQI Forecast",
        xaxis_title="Date", yaxis_title="AQI", hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- MOVED INSIGHTS SECTION UP (converted to real AQI) ---
    if all_forecasts:
        st.subheader("🔎 Forecast Insights")
        for city, forecast_df in zip(selected_cities, all_forecasts):
            if not forecast_df.empty:
                hist_df = load_city_data(city)

                # forecast_df already has yhat in real AQI
                future_avg = forecast_df['yhat'].mean()
                forecast_start_date = forecast_df['ds'].min()
                forecast_end_date = forecast_df['ds'].max()

                comparison_start_date = forecast_start_date - pd.DateOffset(years=1)
                comparison_end_date = forecast_end_date - pd.DateOffset(years=1)

                # historical data is in real AQI already under column 'y'
                comparison_period_df = hist_df[
                    (hist_df['ds'] >= comparison_start_date) &
                    (hist_df['ds'] <= comparison_end_date)
                ]

                if not comparison_period_df.empty:
                    historical_comparison_avg = comparison_period_df['y'].mean()
                    change = ((future_avg - historical_comparison_avg) / historical_comparison_avg) * 100 if historical_comparison_avg else 0
                    st.markdown(
                        f"**{city}**: The forecasted AQI ({future_avg:.2f}) is *{change:+.1f}%* compared to the same period last year ({historical_comparison_avg:.2f}).")
                    if change > 5:
                        st.markdown(f"↳ 📈 Air quality is expected to worsen for {city}.")
                    elif change < -5:
                        st.markdown(f"↳ ✅ Air quality is expected to improve for {city}.")
                    else:
                        st.markdown(f"↳ ⚖ No significant change in air quality expected for {city}.")
                else:
                    st.markdown(
                        f"**{city}**: Cannot provide a year-over-year comparison as no historical data exists for {comparison_start_date.strftime('%b %Y')} to {comparison_end_date.strftime('%b %Y')}."
                    )

    # --- DOWNLOAD AND EXPANDER SECTION ---
    if all_forecasts:
        combined_forecast = pd.concat(all_forecasts, ignore_index=True)
        # Ensure expected columns exist
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            if col not in combined_forecast.columns:
                combined_forecast[col] = pd.NA

        forecast_csv = combined_forecast[['City', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={'ds': 'Date', 'yhat': 'Predicted AQI', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
        ).to_csv(index=False)

        st.download_button("📥 Download Forecast Data", forecast_csv, "forecast_data.csv", "text/csv")

        with st.expander("View Detailed Forecast Data"):
            st.dataframe(
                combined_forecast[['City', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                    columns={'ds': 'Date', 'yhat': 'Predicted AQI', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
                )
            )
