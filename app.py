import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure utils can be imported
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from utils.topsis import Topsis

st.set_page_config(layout="wide", page_title="Life Partner Selector", page_icon="💍")

# Load Custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Helper function to load default data
# Helper function to load default data
# Removed cache to ensure updates to CSV are reflected immediately
def load_default_data(gender):
    if gender == 'Male':
        return pd.read_csv('data/potential_grooms.csv')
    else:
        return pd.read_csv('data/potential_brides.csv')

# --- Sidebar ---
st.sidebar.title("Your Preferences")
user_name = st.sidebar.text_input("Your Name", "User")
looking_for = st.sidebar.radio("I am looking for a:", ("Groom", "Bride"))

# Data Source Logic
st.sidebar.markdown("---")
st.sidebar.subheader("Data Management")
data_source = st.sidebar.radio("Source:", ("Default List", "Upload CSV"))

# --- Main Area ---
st.title(f"💍 Whom to Marry? - {user_name}'s Edition")
if looking_for == "Groom":
    st.markdown("### Finding your perfect Prince Charming")
    target_gender = "Male"
else:
    st.markdown("### Finding your perfect Partner")
    target_gender = "Female"

# Load Data
if 'data' not in st.session_state:
    st.session_state.data = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload candidates file (CSV)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
    else:
        st.info("Please upload a CSV file to proceed.")
        st.stop()
else:
    # Use default data, but allow editing in session state
    # Force reload if data is None OR if 'Name' column is missing (legacy state) OR Reset button clicked
    should_reload = (st.session_state.data is None) or \
                    ('Name' not in st.session_state.data.columns) or \
                    st.sidebar.button("Reset Data")
                    
    if should_reload:
        st.session_state.data = load_default_data(target_gender)
        # Clear previous results if data reloaded to avoid mismatch
        if 'last_results' in st.session_state:
            del st.session_state['last_results']

# Data Editor
st.write("### Candidates List (Editable)")
edited_df = st.data_editor(st.session_state.data, num_rows="dynamic", use_container_width=True)
st.session_state.data = edited_df

if edited_df.empty:
    st.warning("No candidates available.")
    st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["Recommendation Engine", "Test your decisions"])

# Pre-processing for Ranking (moved before tabs to be accessible in sidebar)
ranking_df = edited_df.copy()

# 1. Handle Categorical mappings if they exist in valid columns
if "Siblings" in ranking_df.columns:
    if ranking_df["Siblings"].dtype == 'object':
        ranking_df["Siblings"] = ranking_df["Siblings"].replace(["Yes", "No"], [0, 1])

if "Complexion" in ranking_df.columns:
     if ranking_df["Complexion"].dtype == 'object':
        ranking_df["Complexion"] = ranking_df["Complexion"].replace(["Fair", "Wheatish", "Dark"], [3, 2, 1])

# 2. Handle BMI if Height & Weight exist
if "Height" in ranking_df.columns and "Weight" in ranking_df.columns:
    ranking_df["BMI"] = 10000 * ranking_df["Weight"] / (ranking_df["Height"] ** 2)
    ranking_df["dist_ideal_BMI"] = abs(ranking_df["BMI"] - 22)

# Determine available criteria for sliders
available_criteria_cols = ranking_df.select_dtypes(include=[np.number]).columns.tolist()
if 'ID' in available_criteria_cols: available_criteria_cols.remove('ID')
if 'Name' in available_criteria_cols: available_criteria_cols.remove('Name')
if 'Weight' in available_criteria_cols: available_criteria_cols.remove('Weight')
if 'BMI' in available_criteria_cols: available_criteria_cols.remove('BMI')

# --- SIDEBAR: Criteria & Weights ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Criteria Weights")
st.sidebar.markdown("*Set importance (0-10)*")

# Generate Sliders in Sidebar
selected_criteria = []
final_weights = []
final_is_beneficial = []

# Use session state to persist slider values across tabs
if 'slider_weights' not in st.session_state:
    st.session_state.slider_weights = {}

for col in available_criteria_cols:
    val = st.sidebar.slider(f"{col}", 0, 10, 5, key=f"slider_{col}")
    st.session_state.slider_weights[col] = val
    
    if val > 0:
        selected_criteria.append(col)
        final_weights.append(val)
        
        is_ben = True
        if col == 'dist_ideal_BMI':
            is_ben = False
        elif col == 'Siblings':
             is_ben = True 
        
        final_is_beneficial.append(is_ben)

with tab1:
    st.write("### 🎯 Recommendation Results")
    st.markdown("Adjust the criteria weights in the sidebar, then click **Calculate Rankings** below.")
    
    # --- Calculate ---
    if st.button("Calculate Rankings", type="primary"):
        if not selected_criteria:
            st.error("Please select at least one criteria with weight > 0 in the sidebar")
        else:
            # Validate data
            if len(selected_criteria) != len(final_weights) or len(selected_criteria) != len(final_is_beneficial):
                st.error(f"Data mismatch: {len(selected_criteria)} criteria, {len(final_weights)} weights, {len(final_is_beneficial)} flags")
                st.stop()
            
            # Prepare Matrix
            eval_matrix = ranking_df[selected_criteria].to_numpy(dtype="float")
            
            # Check for NaN or Inf in input data
            if np.any(np.isnan(eval_matrix)) or np.any(np.isinf(eval_matrix)):
                st.error("Input data contains invalid values (NaN or Inf). Please check your data.")
                st.write("Problematic columns:", selected_criteria)
                st.write("Data preview:", ranking_df[selected_criteria].head())
                st.stop()
            
            t = Topsis(eval_matrix, final_weights, final_is_beneficial)
            t.calc()
            
            # Check if TOPSIS produced valid results
            if np.any(np.isnan(t.worst_similarity)) or np.any(np.isinf(t.worst_similarity)):
                st.warning("TOPSIS calculation produced some invalid scores. Using fallback values.")
            
            result_df = edited_df.copy()
            result_df['Score'] = t.worst_similarity
            result_df = result_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
            result_df.index += 1  # Start rank from 1 instead of 0
            
            # Store in session state
            st.session_state['last_results'] = result_df
            
            st.balloons()
            st.write("## 🏆 Top Recommendations")
            
            # Display Top 3 Cards
            top_n = min(3, len(result_df))
            top_cols = st.columns(top_n)
            
            for i in range(top_n):
                candidate = result_df.iloc[i]
                with top_cols[i]:
                    st.markdown(f"""
                    <div class="css-1r6slb0">
                        <h3 style="text-align: center;">#{i+1}</h3>
                        <h2 style="text-align: center;">{candidate.get('Name', 'N/A')}</h2>
                        <p style="text-align: center; color: #999;">ID: {candidate.get('ID', 'N/A')}</p>
                        <hr>
                        <p><b>Match Score:</b> {candidate['Score']:.4f}</p>
                        <p><b>Salary:</b> {candidate.get('Salary', 'N/A')}</p>
                        <p><b>Height:</b> {candidate.get('Height', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.write("### Detailed Rankings")
            st.dataframe(result_df)
    
    # Show previous results if available
    elif 'last_results' in st.session_state:
        st.info("Showing previous results. Adjust weights and click 'Calculate Rankings' to update.")
        result_df = st.session_state['last_results']
        
        st.write("## 🏆 Top Recommendations")
        top_n = min(3, len(result_df))
        top_cols = st.columns(top_n)
        
        for i in range(top_n):
            candidate = result_df.iloc[i]
            with top_cols[i]:
                st.markdown(f"""
                <div class="css-1r6slb0">
                    <h3 style="text-align: center;">#{i+1}</h3>
                    <h2 style="text-align: center;">{candidate.get('Name', 'N/A')}</h2>
                    <p style="text-align: center; color: #999;">ID: {candidate.get('ID', 'N/A')}</p>
                    <hr>
                    <p><b>Match Score:</b> {candidate['Score']:.4f}</p>
                    <p><b>Salary:</b> {candidate.get('Salary', 'N/A')}</p>
                    <p><b>Height:</b> {candidate.get('Height', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.write("### Detailed Rankings")
        st.dataframe(result_df)



with tab2:
    st.write("### Lets test your decisions")
    
    # Check if we have results to test against
    if 'last_results' not in st.session_state:
        st.warning("⚠️ Please go to the **Recommendation Engine** tab and click **Calculate Rankings** first!")
        st.stop()
    
    result_df = st.session_state['last_results']
    
    # Show all candidates (shuffled and without scores to prevent bias)
    st.write("Below is the list of candidates. Select one you prefer based on their attributes, and see if your choice matches the algorithm's recommendation!")
    

    
    # Create a shuffled view for display
    # Drop 'Score' so user is not biased
    shuffled_df = result_df.sample(frac=1, random_state=None).reset_index(drop=True)
    shuffled_df.index += 1 # Force 1-based index for display (1, 2, 3...)
    
    display_cols = ['ID', 'Name', 'Salary', 'Height', 'Weight', 'Complexion'] 
    # specific columns that exist
    actual_cols = [c for c in display_cols if c in shuffled_df.columns]
    
    st.dataframe(shuffled_df[actual_cols], use_container_width=True, hide_index=True)
    
    # Form for selection
    with st.form("test_decision_form"):
        # Create a display list with Names
        if 'Name' in result_df.columns:
            # Sort options alphabetically by Name for easier lookup
            sorted_df = result_df.sort_values(by='Name')
            
            options_display = []
            options_names = []
            for idx, row in sorted_df.iterrows():
                display = f"{row['Name']} (ID: {row.get('ID', 'N/A')}) - Salary: {row.get('Salary', 'N/A')}, Height: {row.get('Height', 'N/A')}"
                options_display.append(display)
                options_names.append(str(row['Name']))
        else:
            # Fallback for ID only
            sorted_df = result_df.sort_values(by='ID')
            options_display = [f"Candidate {i}" for i in sorted_df['ID']]
            options_names = [str(i) for i in sorted_df.index]
            
        selected_display = st.selectbox("Whom do you want to marry?", options_display)
        submitted = st.form_submit_button("Submit", type="primary")
        
        if submitted:
            # Get the selected Name
            selected_idx = options_display.index(selected_display)
            selected_name = options_names[selected_idx]
            
            # Find in the full results
            if 'Name' in result_df.columns:
                match = result_df[result_df['Name'] == selected_name]
            else:
                match = result_df.iloc[[int(selected_idx)]] # Fallback
            
            if not match.empty:
                # Calculate Rank based on position in the sorted results (Score High -> Low)
                # This guarantees 1-based ranking regardless of the dataframe index
                try:
                    # Get the list of names in order of rank (0 is Rank 1)
                    ranked_names = result_df['Name'].tolist()
                    rank = ranked_names.index(selected_name) + 1
                except:
                    # Fallback if name not found or logic fails
                    rank = result_df.index.get_loc(match.index[0]) + 1
                
                score = match.iloc[0]['Score']
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Your Choice Rank", f"#{rank}")
                with col2:
                    st.metric("Match Score", f"{score:.4f}")
                    
                if rank == 1:
                    st.success("🎉 Excellent choice! This is the #1 recommended match!")
                elif rank <= 3:
                    st.info("👍 Good choice! This candidate is in the top 3.")
                elif rank <= 5:
                    st.info("✅ Decent choice! This candidate is in the top 5.")
                else:
                    st.warning("🤔 This candidate is ranked lower based on your current priorities. You might want to review what you value most!")
            else:
                st.error("Candidate not found in ranking list.")


# --- Disclaimer ---
st.markdown(
    """
    <div class="fixed-footer">
        <p style="color: rgba(255, 255, 255, 0.5); font-size: 11px; margin: 0;">
            <span style="font-weight: bold; color: #FF6B6B;">⚠️ DISCLAIMER: FOR ENTERTAINMENT PURPOSES ONLY ⚠️</span><br>
            This app is a simulation for fun. Scores are fictional. Do not make life decisions based on this.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
