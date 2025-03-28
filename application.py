from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg
from pycaret.classification import (
    setup as setup_clf,
    compare_models as compare_models_clf,
)
from pycaret.regression import pull as pull_reg, save_model as save_model_reg
from pycaret.classification import pull as pull_clf, save_model as save_model_clf
import sweetviz as sv
import pandas as pd
import os
from src.agents.data_analyst_agent import generate_response, Hannah_Baker_chatbot_ui
from src.analysis.analysis import descriptive_analysis_main
import src.agents.ml_agent
from src.agents.ml_agent import determine_model_type_with_llm
import pickle
from src.analysis.post_analysis import post_analysis_main, post_analysis_chatbot

# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report

# Ensure required directories exist
def ensure_directories():
    directories = ['assets', 'reports', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Check if asset exists
def check_asset(asset_path):
    if not os.path.exists(asset_path):
        st.error(f"Asset file not found: {asset_path}")
        return False
    return True

# Initialize directories
ensure_directories()

st.set_page_config(page_title="AutoML: Powered with Gen-AI", page_icon="⚙️")

# Initialize session state for the dataset and target column if not already initialized
if "df" not in st.session_state:
    st.session_state.df = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
# Initialize session state for storing chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for post analysis chatbot
if "future_chat_history" not in st.session_state:
    st.session_state.future_chat_history = []


# Sidebar Navigation
with st.sidebar:
    logo_path = "assets/creative_logo_for_ml.jpeg"
    if check_asset(logo_path):
        st.image(logo_path)
    st.title("AutoML: Powered with Gen-AI")
    choice = st.radio(
        "Navigation",
        [
            "Data ingestion Zone",
            "Data Analysis Zone",
            "Data Modelling Zone",
            "Future Recommendation Zone",
            "Download Model",
        ],
    )
    st.info(
        "From CSV input to comprehensive data analysis and the best-trained model in one streamlined process."
    )

# Data Ingestion Section
if choice == "Data ingestion Zone":
    st.title("Data ingestion Zone")
    # Add an image below the title with custom dimensions
    # data_white_path = "assets/download (2).png"
    # if check_asset(data_white_path):
        # st.image(data_white_path, caption="Mr. Data White", width=300)
    description_data_ingestion_zone = "Welcome to the Data Ingestion Zone \n\n Upload your CSV file now to start training your data!"

    # File uploader
    file = st.file_uploader(description_data_ingestion_zone)

    # Store the uploaded file into session_state if a new file is uploaded
    if file:
        st.session_state.df = pd.read_csv(
            file, index_col=None
        )  # Store the CSV in session state (RAM)

    # Display the stored dataframe if it's available
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df)  # Display the stored dataframe

        # Select the target column, preselecting if already stored in session_state
        st.session_state.target_column = st.selectbox(
            "Choose the Target Column",
            st.session_state.df.columns,
            index=(
                st.session_state.df.columns.get_loc(st.session_state.target_column)
                if st.session_state.target_column in st.session_state.df.columns
                else 0
            ),
        )


# Data Analysis Section
if choice == "Data Analysis Zone":
    st.title("Data Analysis Zone")
    hannah_path = "assets/download (3).png"
    if check_asset(hannah_path):
        st.image(hannah_path, caption="Hannah Baker", width=300)
    
    st.subheader("Exploratory Data Analysis")

    if "df" in st.session_state:  # Ensure dataframe is available
        if "reports_generated" not in st.session_state:
            st.session_state.reports_generated = False

        if st.button("Generate Both Reports"):
            try:
                # Generate EDA report
                eda_report = sv.analyze(st.session_state.df)
                report_path = "reports/SWEETVIZ_REPORT.html"
                eda_report.show_html(filepath=report_path)

                # Generate PDF report and store TEXT_DATA in session state
                st.session_state.TEXT_DATA = descriptive_analysis_main(
                    st.session_state.df, st.session_state.target_column
                )

                # Save report generation history in session state
                st.session_state.reports_generated = True
                st.session_state.eda_report_file = report_path
                st.session_state.pdf_report_file = "reports/report.pdf"
                
                st.success("Reports generated successfully!")
            except Exception as e:
                st.error(f"Error generating reports: {str(e)}")

        # Show download buttons if both reports are already generated
        if st.session_state.reports_generated:
            col1, col2 = st.columns(2)
            with col1:
                with open(st.session_state.eda_report_file, "rb") as f:
                    st.download_button(
                        "Download EDA Report", f, file_name="reports/SWEETVIZ_REPORT.html"
                    )
            with col2:
                with open(st.session_state.pdf_report_file, "rb") as f:
                    st.download_button(
                        "Descriptive & Diagnostic Analysis Report",
                        f,
                        file_name="reports/report.pdf",
                    )

    else:
        st.error("No dataset found. Please upload data for analysis.")

    # Hannah Baker chatbot section
    st.subheader("Hannah Baker: Personal Data Analyst")
    st.write("Ask Hannah any questions about your data!")
    
    try:
        if "TEXT_DATA" not in st.session_state or st.session_state.TEXT_DATA is None:
            raise NameError("TEXT_DATA is not defined")
        Hannah_Baker_chatbot_ui(st.session_state.TEXT_DATA)
    except NameError as e:
        st.error("Please generate both reports first to interact with the chatbot.")
        st.write("Click the 'Generate Both Reports' button to create the necessary reports.")


if choice == "Data Modelling Zone":
    st.title("Model Training")

    # Check if history is already in session_state; if not, initialize it
    if "model_type" not in st.session_state:
        st.session_state.model_type = None
    if "setup_df" not in st.session_state:
        st.session_state.setup_df = None
    if "compare_df" not in st.session_state:
        st.session_state.compare_df = None
    if "best_model" not in st.session_state:
        st.session_state.best_model = None

    # Display stored history if available
    if st.session_state.model_type:
        st.write(f"{st.session_state.model_type}")
    if st.session_state.setup_df is not None:
        st.subheader("Previous Setup Dataframe:")
        st.dataframe(st.session_state.setup_df)
    if st.session_state.compare_df is not None:
        st.subheader("Previous Comparison Dataframe:")
        st.dataframe(st.session_state.compare_df)
    if st.session_state.best_model:
        st.success("A model has already been trained and saved.")

    # Automatically determine the model type using the LLM and pandas
    model_type = determine_model_type_with_llm(
        st.session_state.df, st.session_state.target_column
    )
    st.session_state.model_type = model_type  # Store the detected model type
    print(model_type)

    if st.button("Run Modelling"):
        if "regression" in model_type.lower():
            try:
                setup_reg(st.session_state.df, target=st.session_state.target_column)
                setup_df = pull_reg()
                st.session_state.setup_df = setup_df
                st.dataframe(setup_df)

                algorithms = src.agents.ml_agent.get_regression_model_list(
                    st.session_state.TEXT_DATA
                )
                best_model = compare_models_reg(include=algorithms)
                compare_df = pull_reg()
                st.session_state.compare_df = compare_df
                st.dataframe(compare_df)
                st.session_state.best_model = best_model
                
                # Save model with error handling
                try:
                    save_model_reg(best_model, "models/best_model")
                    st.success("Model saved successfully!")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred during regression modeling: {str(e)}")
                # Handle the fallback scenario
                setup_reg(
                    data=st.session_state.df, 
                    target=st.session_state.target_column, 
                    session_id=123, 
                    normalize=True,
                    verbose=True
                )

                # Pull the setup DataFrame
                setup_df = pull_reg()
                st.session_state.setup_df = setup_df
                st.dataframe(setup_df)

                # Compare all available models
                best_model = compare_models_reg()
                compare_df = pull_reg()  # Pull the comparison DataFrame
                st.session_state.compare_df = compare_df
                st.dataframe(compare_df)

                # Save the best model and pipeline
                st.session_state.best_model = best_model
                st.session_state.best_pipeline = save_model_reg(best_model, "models/best_model")  # Save the pipeline reference

        else:
            try:
                setup_clf(st.session_state.df, target=st.session_state.target_column)
                setup_df = pull_clf()
                st.session_state.setup_df = setup_df
                st.dataframe(setup_df)

                algorithms = src.agents.ml_agent.get_classification_models_lst(
                    st.session_state.TEXT_DATA
                )
                best_model = compare_models_clf(include=algorithms)
                compare_df = pull_clf()
                st.session_state.compare_df = compare_df
                st.dataframe(compare_df)
                st.session_state.best_model = best_model
                
                # Save model with error handling
                try:
                    save_model_clf(best_model, "models/best_model")
                    st.success("Model saved successfully!")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred during classification modeling: {str(e)}")
                setup_clf(st.session_state.df, target=st.session_state.target_column)
                setup_df = pull_clf()
                st.session_state.setup_df = setup_df
                st.dataframe(setup_df)

                best_model = compare_models_clf()
                compare_df = pull_clf()
                st.session_state.compare_df = compare_df
                st.dataframe(compare_df)
                st.session_state.best_model = best_model
                save_model_clf(best_model, "models/best_model")

        st.success("Model Training Completed")



if choice == "Future Recommendation Zone":
    st.title("Future Recommendation Zone")
    
    if st.session_state.best_model is None:
        st.error("Please train a model first in the Data Modelling Zone")
    else:
        if "post_analysis_generated" not in st.session_state:
            st.session_state.post_analysis_generated = False
            
        if "post_analysis_text" not in st.session_state:
            st.session_state.post_analysis_text = None
            
        st.subheader("Predictive & Prescriptive Analysis")
        
        if st.button("Generate Analysis"):
            with st.spinner("Generating comprehensive analysis..."):
                TEXT_DATA, report_path = post_analysis_main(
                    st.session_state.df,
                    st.session_state.best_model,
                    st.session_state.target_column,
                    st.session_state.model_type,
                    st.session_state.compare_df
                )
                st.session_state.post_analysis_text = TEXT_DATA
                st.session_state.post_analysis_generated = True
                st.session_state.report_path = report_path
                
        if st.session_state.post_analysis_generated:
            with open(st.session_state.report_path, "rb") as f:
                st.download_button(
                    "Download Full Analysis Report",
                    f,
                    file_name="post_analysis_report.pdf"
                )
            
            st.subheader("AI Assistant for Future Recommendations")
            st.write("Ask questions about the predictive and prescriptive analysis!")
            
            try:
                post_analysis_chatbot(st.session_state.post_analysis_text)
            except Exception as e:
                st.error(f"Error initializing chatbot: {str(e)}")


if choice == "Download Model":
    st.title("Downloads")
    if st.session_state.best_model:

        # Save st.session_state.best_model to a file
        # with open("best_model.pkl", "wb") as f:
        #     pickle.dump(st.session_state.best_model, f)
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download Model", f, file_name="best_model.pkl")
        with open("Run.ipynb", "rb") as f:
            st.download_button("Download Load Model File", f, file_name="Run.ipynb")
        with open("api_code.py", "rb") as f:
            st.download_button("Download API Deployment Code", f, file_name="api_code.py")
        with open("gui.ipynb", "rb") as f:
            st.download_button("Download GUI Code", f, file_name="gui.ipynb")
        with open("Guide.txt", "rb") as f:
            st.download_button("Download Guide.txt", f, file_name="Guide.txt")
    
    else:
        st.error("No model has been trained yet.")
