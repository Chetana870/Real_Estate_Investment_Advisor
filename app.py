import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")


@st.cache_resource
def load_models():
    return (joblib.load("models/investment_classifier.joblib"),
            joblib.load("models/price_regressor.joblib"))


@st.cache_data
def load_data():
    cols = ["State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt_Rs", "Good_Investment", "Availability_Status"]
    return pd.read_csv("data/processed/housing_clean.csv", usecols=cols)


clf, reg = load_models()
df = load_data()
state_cities = df.groupby("State")["City"].unique().apply(sorted).to_dict()

st.title("Real Estate Investment Advisor")
tab_predict, tab_explore, tab_model = st.tabs(["Predict", "Explore market", "Model insights"])

# ---------------- Predict ----------------
with tab_predict:
    st.subheader("Enter property details")
    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", sorted(state_cities))
        city = st.selectbox("City", state_cities[state])
        ptype = st.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
        bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
        size = st.number_input("Size (sq ft)", 500, 5000, 2500, step=50)
        price = st.number_input("Current price (lakhs)", 10.0, 500.0, 250.0, step=5.0)
    with c2:
        year_built = st.number_input("Year built", 1990, 2023, 2010)
        total_floors = st.number_input("Total floors", 1, 30, 15)
        floor_no = st.number_input("Floor number", 0, int(total_floors), min(5, int(total_floors)))
        schools = st.slider("Nearby schools", 1, 10, 5)
        hospitals = st.slider("Nearby hospitals", 1, 10, 5)
        transport = st.selectbox("Public transport accessibility", ["Low", "Medium", "High"], index=1)
    with c3:
        furnished = st.selectbox("Furnished status", ["Unfurnished", "Semi-furnished", "Furnished"])
        facing = st.selectbox("Facing", ["North", "South", "East", "West"])
        owner = st.selectbox("Owner type", ["Owner", "Builder", "Broker"])
        availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
        parking = st.checkbox("Parking space")
        security = st.checkbox("Security")
        amenities = st.multiselect("Amenities", ["Playground", "Gym", "Garden", "Pool", "Clubhouse"])

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price,
            "Price_per_SqFt_Rs": price * 100000 / size,
            "Age_of_Property": 2025 - year_built,
            "Floor_No": floor_no, "Total_Floors": total_floors,
            "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Parking_Space": int(parking), "Security": int(security),
            "Amenity_Count": max(len(amenities), 1),
            "State": state, "City": city, "Property_Type": ptype,
            "Furnished_Status": furnished, "Public_Transport_Accessibility": transport,
            "Facing": facing, "Owner_Type": owner, "Availability_Status": availability,
        }])
        prob = clf.predict_proba(row)[0, 1]
        future = reg.predict(row)[0]
        growth = (future / price) ** (1 / 5) - 1
        is_good = prob >= 0.5

        m1, m2, m3 = st.columns(3)
        m1.metric("Good investment?", "Yes" if is_good else "No")
        m2.metric("Confidence", f"{(prob if is_good else 1 - prob):.0%}")
        m3.metric("Estimated price in 5 years", f"{future:,.0f} lakhs",
                  f"{growth:.1%} per year")
        st.progress(float(prob), text=f"Probability of being a good investment: {prob:.0%}")
        st.caption("Future price uses an assumed growth model (base 6% plus infrastructure, "
                   "readiness and valuation adjustments), not real market forecasts.")

# ---------------- Explore ----------------
with tab_explore:
    f1, f2, f3 = st.columns(3)
    sel_cities = f1.multiselect("City (empty = all)", sorted(df["City"].unique()))
    price_range = f2.slider("Price range (lakhs)", 10, 500, (10, 500))
    sel_bhk = f3.multiselect("BHK (empty = all)", [1, 2, 3, 4, 5])

    view = df[df["Price_in_Lakhs"].between(*price_range)]
    if sel_cities:
        view = view[view["City"].isin(sel_cities)]
    if sel_bhk:
        view = view[view["BHK"].isin(sel_bhk)]

    st.write(f"{len(view):,} properties match")
    if len(view):
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(view, x="Price_in_Lakhs", nbins=40,
                                    title="Price distribution (lakhs)"))
        top = (view.groupby("City")["Price_per_SqFt_Rs"].mean().nlargest(10)
               .reset_index())
        b.plotly_chart(px.bar(top, x="City", y="Price_per_SqFt_Rs",
                              title="Top 10 cities by avg price per sq ft (Rs)"))
        rate = view.groupby("Property_Type")["Good_Investment"].mean().reset_index()
        st.plotly_chart(px.bar(rate, x="Property_Type", y="Good_Investment",
                               title="Share of good investments by property type"))

# ---------------- Model insights ----------------
with tab_model:
    names = [n.split("__", 1)[1] for n in clf.named_steps["prep"].get_feature_names_out()]
    imp = (pd.Series(clf.named_steps["model"].feature_importances_, index=names)
           .nlargest(12).sort_values())
    st.plotly_chart(px.bar(imp, orientation="h",
                           labels={"value": "Importance", "index": "Feature"},
                           title="Top 12 features (XGBoost classifier)"))
    st.write("The model mostly relies on the inputs of the investment rule: price per sq ft, "
             "availability status and BHK. Locality, floors and age contribute almost nothing.")
