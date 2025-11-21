import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Income Predictor",
    page_icon="💰",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        metadata = joblib.load('model_metadata.pkl')
        feature_importance = joblib.load('feature_importance.pkl')
        return model, metadata, feature_importance
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, metadata, feature_importance = load_models()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💰 Adult Income Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict whether an individual's income exceeds $50,000/year")
st.markdown("---")

# Sidebar - Model Info
st.sidebar.header("📊 Model Information")

if metadata:
    st.sidebar.success(f"**Model:** {metadata['model_name']}")
    st.sidebar.markdown("**Performance Metrics:**")
    metrics = metadata['metrics']
    st.sidebar.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    st.sidebar.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
    st.sidebar.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

if feature_importance is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 Top 5 Important Features:**")
    
    top_5 = feature_importance.head(5)
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        st.sidebar.write(f"{i}. **{row['feature']}**: {row['importance']:.4f}")
    
    # Feature importance chart
    fig = go.Figure(go.Bar(
        x=top_5['importance'].values,
        y=top_5['feature'].values,
        orientation='h',
        marker=dict(color='teal')
    ))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.sidebar.plotly_chart(fig, use_container_width=True)

# Main form
st.header("📝 Enter Person's Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Details")
    age = st.number_input("Age", min_value=17, max_value=90, value=39)
    sex = st.selectbox("Sex", ["Male", "Female"])
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    relationship = st.selectbox("Relationship", ["Husband", "Wife", "Not-in-family", "Own-child", "Unmarried", "Other-relative"])
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])

with col2:
    st.subheader("Work Information")
    workclass = st.selectbox("Work Class", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
    occupation = st.selectbox("Occupation", ["Prof-specialty", "Exec-managerial", "Craft-repair", "Sales", "Adm-clerical", "Tech-support", "Other-service", "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", "Farming-fishing", "Protective-serv", "Priv-house-serv", "Armed-Forces"])
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

with col3:
    st.subheader("Education & Financial")
    education = st.selectbox("Education Level", ["Bachelors", "HS-grad", "Some-college", "Masters", "Assoc-voc", "Assoc-acdm", "11th", "10th", "Prof-school", "Doctorate", "7th-8th", "9th", "12th", "5th-6th", "1st-4th", "Preschool"])
    education_num = st.number_input("Years of Education", min_value=1, max_value=16, value=13)
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)

col4, col5 = st.columns(2)
with col4:
    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Italy", "Other"], index=0)
with col5:
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1500000, value=77516, help="Census weighting - leave default if unsure")

st.markdown("---")

# Predict button
if st.button("🔮 Predict Income", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not loaded!")
    else:
        try:
            # Prepare input
            input_dict = {
                'age': age,
                'workclass': workclass,
                'fnlwgt': fnlwgt,
                'education': education,
                'education.num': education_num,
                'marital.status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'sex': sex,
                'capital.gain': capital_gain,
                'capital.loss': capital_loss,
                'hours.per.week': hours_per_week,
                'native.country': native_country
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Predict
            with st.spinner("Making prediction..."):
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
            
            prediction_label = ">50K" if prediction == 1 else "<=50K"
            confidence = "High" if (probability > 0.7 or probability < 0.3) else "Medium"
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Predicted Income", prediction_label, "High Income" if prediction == 1 else "Low Income")
            with col_b:
                st.metric("Probability (>50K)", f"{probability:.2%}")
            with col_c:
                st.metric("Confidence", confidence)
            
            # Gauge chart
            st.markdown("### Prediction Confidence")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probability of Income >50K"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("### 📊 Interpretation")
            if prediction == 1:
                st.info(f"✨ This person is predicted to earn **more than $50,000/year** with {probability:.1%} confidence.")
            else:
                st.info(f"✨ This person is predicted to earn **$50,000 or less/year** with {1-probability:.1%} confidence.")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><p>Built with Streamlit | Model: TPOT AutoML</p></div>", unsafe_allow_html=True)
