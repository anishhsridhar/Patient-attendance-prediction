import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #ffffff; text-align: center; padding: 10px 0 5px 0;
    }
    .sub-title {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 20px;
    }
    .risk-low {
        background-color: #adedbc; border-left: 6px solid #28a745;
        padding: 15px; border-radius: 8px;
        font-size: 1.2rem; font-weight: bold; color: #155724;text-align: center;
    }
    .risk-medium {
        background-color: #fcebb1; border-left: 6px solid #ffc107;
        padding: 15px; border-radius: 8px;
        font-size: 1.2rem; font-weight: bold; color: #856404;text-align: center;
    }
    .risk-high {
        background-color: #fcb1b7; border-left: 6px solid #dc3545;
        padding: 15px; border-radius: 8px;
        font-size: 1.2rem; font-weight: bold; color: #721c24;text-align: center;
    }
    .insight-box {
        background-color: #262f40; border-left: 5px solid #1a73e8;
        padding: 12px 15px; border-radius: 6px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODELS & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf = joblib.load("classifier.joblib")
    fct = joblib.load("forecaster.joblib")
    return clf, fct

@st.cache_data
def load_data():
    df = pd.read_csv("Clean_total5.csv")
    df['appointment_date_continuous'] = pd.to_datetime(
        df['appointment_date_continuous'])
    return df

try:
    classifier, forecaster = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.info("Place classifier.joblib, forecaster.joblib and Clean_total5.csv in the same folder as app.py")
    st.stop()


# ─────────────────────────────────────────────
#  ENCODING MAPS
# ─────────────────────────────────────────────
SPECIALTY_MAP = {'Assist': 0, 'Enf': 1, 'Occupational Therapy': 2,'Pedagogo': 3, 'Physiotherapy': 4, 'Psychotherapy': 5,'Sem Especialidade': 6, 'Speech Therapy': 7, 'Others': 8}
DISABILITY_MAP = {'Intellectual': 0, 'Motor': 1, 'None': 2}
GENDER_MAP     = {'Female ': 0, 'Male': 2, 'Others': 1}
SHIFT_MAP      = {'Afternoon': 0, 'Morning': 1}
RAIN_MAP       = {'No Rain': 0, 'Weak Rain': 1, 'Moderate Rain': 2, 'Heavy Rain': 3}
HEAT_MAP       = {'Cold': 0, 'Mild': 1, 'Warm': 2, 'Heavy Warm': 3, 'Heavy Cold': 4}
AGE_GROUP_MAP  = {'Baby (0-2)': 0, 'Kids (2-12)': 1, 'Teens (12-18)': 2,'Young Adults (18-30)': 3, 'Adults (30-45)': 4,'Seniors (45-60)': 5, 'Elderly (60+)': 6}
DAY_MAP = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,'Friday': 4, 'Saturday': 5, 'Sunday': 6}

place_risk  = df.groupby('place')['no_show'].apply(lambda x: (x == 'yes').mean()).round(4).to_dict()
global_mean = round((df['no_show'] == 'yes').mean(), 4)
place_list  = sorted(df['place'].dropna().unique().tolist())


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dataset Info")
    st.metric("Total Appointments", f"{len(df)}")
    st.metric("No-show Rate",       f"{round((df['no_show']=='yes').mean()*100,1)}%")
    st.metric("Specialties",        df['specialty'].nunique())
    st.metric("No. of cities",      df['place'].nunique())
    st.markdown("---")
    st.markdown("### Models Used")
    st.markdown("")
    st.markdown("No-Show Predictor")
    if st.button("Random Forest Classifier"):
        st.write("F1 Score  - 0.72 ")
        st.write("ROC-AUC   - 0.89 ")
        st.write("Precision - 0.71 ")
        st.write("Recall    - 0.73 ")
    st.markdown("Demand Forecaster")
    if st.button("XGBoost Regressor"):
        st.write("MAE   - 48.58 ")
        st.write("MAPE  - 19.15 % ")
        st.write("R2    - 0.92 ")

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">Medical Appointment Analytics Dashboard</div>',
            unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  No-Show Predictor  ",
    "  Demand Forecasting  ",
    "  Business Insights  "
])


# ═══════════════════════════════════════════
#  TAB 1 — NO-SHOW PREDICTOR
# ═══════════════════════════════════════════
with tab1:
    st.markdown("### Patient No-Show Risk Predictor")
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### Patient Info")
        age             = st.slider("Age", 0, 110, 25)
        gender          = st.selectbox("Gender", list(GENDER_MAP.keys()))
        disability      = st.selectbox("Disability", list(DISABILITY_MAP.keys()))
        scholarship     = st.selectbox("Has Scholarship?", ["No", "Yes"])
        needs_companion = st.selectbox("Needs A Companion?", ["Yes", "No"])
    
    with col2:
        st.markdown("#### Health")
        hipertension = st.checkbox("Hypertension")
        diabetes     = st.checkbox("Diabetes")
        alcoholism   = st.checkbox("Alcoholism")
        handcap      = st.checkbox("Handicap")
        st.markdown("")
        specialty    = st.selectbox("Specialty", list(SPECIALTY_MAP.keys()))
        shift        = st.selectbox("Which Shift?", list(SHIFT_MAP.keys()))
        appt_time    = st.slider("Appointment Hour (24 hr)", 7, 18, 10)
        sms_received = st.selectbox("SMS Reminder Sent?", ["Yes", "No"])
    
    with col3:
        st.markdown("#### Location")
        place          = st.selectbox("City", place_list)
        day            = st.selectbox("Day", list(DAY_MAP.keys()))
        month          = st.slider("Month", 1, 12, 4)
        rain_intensity = st.selectbox("Intensity of Rain", list(RAIN_MAP.keys()))
        heat_intensity = st.selectbox("Intensity of the Heat", list(HEAT_MAP.keys()))
    with col4:
        st.markdown("#### Weather Report")
        avg_temp       = st.number_input("Average temperature",  5.0, 45.0, 22.0, 0.5)
        avg_rain       = st.number_input("Average mm of rain",  0.0, 50.0,  0.5, 0.1)
        max_temp       = st.number_input("Maximum temperature",  5.0, 50.0, 27.0, 0.5)
        max_rain       = st.number_input("Maximum mm of rain",  0.0,100.0,  1.0, 0.5)
        rainy_before   = st.selectbox("Rainy Day Before?", ["Yes", "No"])
        storm_before   = st.selectbox("Storm Day Before?", ["Yes", "No"])

    st.markdown("---")
    predict_btn = st.button("Predict Risk",type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Analysing ..."):
            under_12     = 1 if age < 12  else 0
            over_60      = 1 if age > 60  else 0
            health_count = sum([hipertension, diabetes, alcoholism, handcap])
            less_health  = 1 if health_count <= 1 else 0
            more_health  = 1 if health_count >= 2 else 0
            is_weekend   = 1 if day in ['Saturday', 'Sunday'] else 0
            is_extreme   = 1 if heat_intensity in ['Heavy Warm', 'Heavy Cold'] else 0

            if   age < 2:  ag = 0
            elif age < 12: ag = 1
            elif age < 18: ag = 2
            elif age < 30: ag = 3
            elif age < 45: ag = 4
            elif age < 60: ag = 5
            else:          ag = 6

            input_df = pd.DataFrame([{
                'appointment_time':        appt_time,
                'age':                     age,
                'under_12_years_old':      under_12,
                'over_60_years_old':       over_60,
                'patient_needs_companion': 1 if needs_companion == 'Yes' else 0,
                'average_temp_day':        avg_temp,
                'average_rain_day':        avg_rain,
                'max_temp_day':            max_temp,
                'max_rain_day':            max_rain,
                'rainy_day_before':        1 if rainy_before == 'Yes' else 0,
                'storm_day_before':        1 if storm_before == 'Yes' else 0,
                'Hipertension':            1 if hipertension else 0,
                'Diabetes':                1 if diabetes else 0,
                'Alcoholism':              1 if alcoholism else 0,
                'Handcap':                 1 if handcap else 0,
                'Scholarship':             1 if scholarship == 'Yes' else 0,
                'SMS_received':            1 if sms_received == 'Yes' else 0,
                'month':                   month,
                'weekend':                 is_weekend,
                'health_issues_count':     health_count,
                'less_health_issues':      less_health,
                'more_health_issues':      more_health,
                'extreme_weather':         is_extreme,
                'rain_intensity_enc':      RAIN_MAP[rain_intensity],
                'heat_intensity_enc':      HEAT_MAP[heat_intensity],
                'age_group_enc':           ag,
                'day_enc':                 DAY_MAP[day],
                'shift_enc':               SHIFT_MAP[shift],
                'specialty_le':            SPECIALTY_MAP[specialty],
                'disability_le':           DISABILITY_MAP[disability],
                'gender_le':               GENDER_MAP[gender],
                'place_enc':               place_risk.get(place, global_mean),
            }])

            risk_prob  = classifier.predict_proba(input_df)[0][1]
            prediction = classifier.predict(input_df)[0]
            risk_pct   = round(risk_prob * 100, 1)

        st.markdown("### Result")
        _, res_col, _ = st.columns([1, 2, 1])
        with res_col:
            if risk_pct < 30:
                st.markdown(f'<div class="risk-low"> Low Risk</div>',unsafe_allow_html=True)
            elif risk_pct < 55:
                st.markdown(f'<div class="risk-medium"> Medium Risk</div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-high"> High Risk </div>',unsafe_allow_html=True)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("No-show Risk",f"{risk_pct}%")
        m2.metric("Prediction", "Not Show" if prediction == 1 else "Will Attend")
        m3.metric("Confidence", f"{max(risk_prob, 1-risk_prob)*100:.1f}%")


# ═══════════════════════════════════════════
#  TAB 2 — DEMAND FORECASTING
# ═══════════════════════════════════════════
with tab2:
    st.markdown("### Demand Forecasting")
    st.markdown("")

    # Build daily series
    daily = df.groupby('appointment_date_continuous').size().reset_index()
    daily.columns = ['date', 'appointment_count']
    daily = daily.sort_values('date').reset_index(drop=True)
    daily = daily[daily['appointment_count'] >= 50].reset_index(drop=True)

    daily['lag_1']         = daily['appointment_count'].shift(1)
    daily['lag_7']         = daily['appointment_count'].shift(7)
    daily['lag_14']        = daily['appointment_count'].shift(14)
    daily['rolling_7']     = daily['appointment_count'].rolling(7).mean()
    daily['rolling_14']    = daily['appointment_count'].rolling(14).mean()
    daily['rolling_std_7'] = daily['appointment_count'].rolling(7).std()
    daily['ema_7']         = daily['appointment_count'].ewm(span=7).mean()
    daily['diff_1']        = daily['appointment_count'].diff(1)
    daily['month']         = daily['date'].dt.month
    daily['is_weekend']    = daily['date'].dt.dayofweek.isin([5,6]).astype(int)
    for d in range(7):
        daily[f'day_{d}'] = (daily['date'].dt.dayofweek == d).astype(int)

    daily.dropna(inplace=True)
    daily.reset_index(drop=True, inplace=True)

    feature_cols = ['lag_1','lag_7','lag_14','rolling_7','rolling_14',
                    'rolling_std_7','ema_7','diff_1','month','is_weekend',
                    'day_0','day_1','day_2','day_3','day_4','day_5','day_6']

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        forecast_days = st.number_input("Total Days to Predict", 1, 30, 7)
    with fc2:
        show_history  = st.number_input("Total Days to Learn", 7, 60, 30)
    with fc3:
        specialty_filter = st.selectbox(
            "Specialty ",
            ["All Specialties"] + list(SPECIALTY_MAP.keys()))
    st.markdown("")
    st.markdown("")

    forecast_btn = st.button(" Forecast",
                              type="primary", use_container_width=True)

    if forecast_btn:
        with st.spinner("Generating forecast..."):
            last_known  = daily.copy()
            predictions = []
            future_dates = []

            for _ in range(forecast_days):
                next_date = last_known['date'].iloc[-1] + pd.Timedelta(days=1)
                nf = {
                    'lag_1':         last_known['appointment_count'].iloc[-1],
                    'lag_7':         last_known['appointment_count'].iloc[-7]  if len(last_known) >= 7  else last_known['appointment_count'].mean(),
                    'lag_14':        last_known['appointment_count'].iloc[-14] if len(last_known) >= 14 else last_known['appointment_count'].mean(),
                    'rolling_7':     last_known['appointment_count'].tail(7).mean(),
                    'rolling_14':    last_known['appointment_count'].tail(14).mean(),
                    'rolling_std_7': last_known['appointment_count'].tail(7).std(),
                    'ema_7':         last_known['appointment_count'].ewm(span=7).mean().iloc[-1],
                    'diff_1':        last_known['appointment_count'].iloc[-1] - last_known['appointment_count'].iloc[-2],
                    'month':         next_date.month,
                    'is_weekend':    1 if next_date.dayofweek in [5,6] else 0,
                }
                for d in range(7):
                    nf[f'day_{d}'] = 1 if next_date.dayofweek == d else 0

                pred = max(0, round(forecaster.predict(pd.DataFrame([nf])[feature_cols])[0]))
                predictions.append(pred)
                future_dates.append(next_date)

                new_row = pd.DataFrame([{'date': next_date, 'appointment_count': pred, **nf}])
                last_known = pd.concat([last_known, new_row], ignore_index=True)

        forecast_df = pd.DataFrame({
            'Date of Appointments':            [d.strftime('%Y-%m-%d') for d in future_dates],
            'Day of the Week':             [d.strftime('%A') for d in future_dates],
            'Predicted Count of Patients':  predictions
        })

        st.markdown("#### Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)
    
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Predicted", f"{sum(predictions):,}")
        m2.metric("Daily Average",   f"{int(np.mean(predictions)):,}")
        m3.metric("Peak Day",        forecast_df.loc[forecast_df['Predicted Count of Patients'].idxmax(), 'Day of the Week'])
        m4.metric("Peak Volume",     f"{max(predictions):,}")



# ═══════════════════════════════════════════
#  TAB 3 — EDA & INSIGHTS
# ═══════════════════════════════════════════
with tab3:
    st.markdown("### Business Insights")
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records",  f"{len(df):,}")
    c2.metric("No-show Rate",   f"{round((df['no_show']=='yes').mean()*100,1)}%")
    c3.metric("Total Specialties",    df['specialty'].nunique())

    st.markdown("---")
 # Business insights
    #st.markdown("### Data Analysis Conclusion")
    ins1, ins2 = st.columns(2)

    with ins1:
        st.markdown("#### 1. Top No-show Risk Factors")
        for txt in [
            "<b>Young Adults (18-30)</b> has the highest no-show rate of 48.4% ",
            "<b>Sem Especialidade</b> has 52.8% no-show because of unconfirmed bookings",
            "<b>Heavy Cold Weather</b> has the risk of 52.9% no-show",
            "<b>10am Morning Slot</b> is the peak hour of no show "
        ]:
            st.markdown(f'<div class="insight-box">{txt}</div>', unsafe_allow_html=True)
            

        st.markdown("#### 2. Low Risk Profiles")
        for txt in [
            " <b>Seniors and Elderly </b> are the reliable patients with risk factor ",
            " <b>Pedagogo</b> has the best attendence percentage",
            " Patients with <b>2+ Health Conditions</b> show the least no show risk",
            " <b>SMS Sent</b> reduces no-show risk significantly"
        ]:
            st.markdown(f'<div class="insight-box">{txt}</div>', unsafe_allow_html=True)

            
    with ins2:    
        st.markdown("#### 3. Demand Patterns")
        for txt in [
            "<b>Thursday & Friday</b> are the peak demanding days",
            "<b>Morning shift</b> has more appointments but less no show rate",
            "<b>April</b> is the highest volume month",
            "<b>November to December </b> has a drastic volume drop"
        ]:
            st.markdown(f'<div class="insight-box">{txt}</div>', unsafe_allow_html=True)


        st.markdown("#### 4. Recommendations")
        for txt in [
            " <b>Target SMS reminders</b> for Ages 18 to 30",
            " <b>Phone and confirm</b> Sem Especialidade appointments",
            " <b>Overbooking buffer</b> on extreme weather forecast days",
            " <b>Extra staffing</b> on Thursday and Friday mornings",
            " <b>Flag Unknown city</b> patients for manual confirmation"
        ]:
            st.markdown(f'<div class="insight-box">{txt}</div>', unsafe_allow_html=True)
    st.markdown("---")
    # Charts row 1
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("##### No-show Rate by Specialty")
        spec_ns = df.groupby('specialty')['no_show'].apply(
            lambda x: (x=='yes').mean()*100).sort_values(ascending=True).round(2)
        fig, ax = plt.subplots(figsize=(7,5))
        ax.barh(spec_ns.index, spec_ns.values,
                color=['#3498db'],
                edgecolor='white')
        ax.axvline(31.8, color='black', linestyle='--', linewidth=1.2, label='Average no show%')
        ax.set_xlabel('No-show Rate (%)')
        ax.set_title('No-show by Specialty', fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with ch2:
        st.markdown("##### No-show Rate by Age Group")
        age_ns = df.groupby('age_group')['no_show'].apply(
            lambda x: (x=='yes').mean()*100).round(2)
        age_order = ['Baby','Kids','Teens','Young Adults','Adults','Seniors','Elderly']
        age_ns = age_ns.reindex([a for a in age_order if a in age_ns.index])
        fig, ax = plt.subplots(figsize=(7,5))
        bars = ax.bar(age_ns.index, age_ns.values,
                      color=['#2ecc71'],
                      edgecolor='white')
        ax.axhline(31.8, color='black', linestyle='--', linewidth=1.2, label='Average no show%')
        ax.set_ylabel('No-show Rate (%)')
        ax.set_title('No-show by Age Group', fontweight='bold')
        ax.legend(fontsize=9)
        plt.xticks(rotation=30, ha='right')
        for bar, val in zip(bars, age_ns.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val}%', ha='center', fontsize=8, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    st.markdown("")
    # Charts row 2
    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown("##### No-show by Appointment Shift")
        shift_ns = df.groupby('appointment_shift')['no_show'].apply(
            lambda x: (x=='yes').mean()*100).round(2)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(shift_ns.index, shift_ns.values,
               color=['#f39c12'], edgecolor='white', width=0.5)
        ax.axhline(31.8, color='black', linestyle='--', linewidth=1.2)
        ax.set_ylabel('No-show Rate (%)')
        ax.set_title('No-show by Shift', fontweight='bold')
        for i, v in enumerate(shift_ns.values):
            ax.text(i, v+0.3, f'{v}%', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with ch4:
        st.markdown("##### No-show by Day of Week")
        day_ns = df.groupby('day')['no_show'].apply(
            lambda x: (x=='yes').mean()*100).round(2)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        day_ns = day_ns.reindex([d for d in day_order if d in day_ns.index])
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(day_ns.index, day_ns.values, color='#9b59b6',
               edgecolor='white', alpha=0.85)
        ax.axhline(31.8, color='black', linestyle='--', linewidth=1.2)
        ax.set_ylabel('No-show Rate (%)')
        ax.set_title('No-show by Day of Week', fontweight='bold')
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Daily volume
    st.markdown("---")
    st.markdown("##### Daily Appointment Volume")
    daily_vol = df.groupby('appointment_date_continuous').size().reset_index()
    daily_vol.columns = ['date','count']
    daily_vol['rolling_7'] = daily_vol['count'].rolling(7, center=True).mean()
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(daily_vol['date'], daily_vol['count'],
            color='steelblue', linewidth=0.7, alpha=0.5, label='Daily')
    ax.plot(daily_vol['date'], daily_vol['rolling_7'],
            color='red', linewidth=2, label='7 day rolling avg')
    ax.set_title('Daily Appointment Volume', fontweight='bold', fontsize=13)
    ax.set_ylabel('Appointments')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()