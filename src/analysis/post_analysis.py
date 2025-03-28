import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.metrics import accuracy_score, mean_squared_error
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

def make_headings_bold(text):
    bold_pattern = r"\*\*(.*?)\*\*"  # Regular expression to find **heading**
    formatted_text = re.sub(bold_pattern, r"<b>\1</b>", text)
    return formatted_text

def create_pdf_report(result, insights, file_name="post_analysis_report.pdf"):
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    subheading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        spaceAfter=10,
        leading=16,
    )
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=12,
        spaceAfter=10,
        leading=14,
    )

    insights = make_headings_bold(insights)

    elements.append(Paragraph("Predictive and Prescriptive Analytics", subheading_style))
    elements.append(Paragraph(insights.replace("\n", "<br/>"), normal_style))
    elements.append(Paragraph("<br/><br/>", normal_style))

    tabular_subheading = Paragraph("Model Performance Metrics", subheading_style)
    elements.append(tabular_subheading)

    for column, stats in result.items():
        elements.append(Paragraph(f"<b>{column}</b>", styles["Heading2"]))
        table_data = [["Metric", "Value"]] + [[stat, str(value)] for stat, value in stats.items()]
        table = Table(table_data, colWidths=[200, 400])
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("PAD", (0, 0), (-1, -1), 6),
            ])
        )
        elements.append(table)
        elements.append(Paragraph("<br/><br/>", normal_style))

    doc.build(elements)


def analyze_model_performance(compare_df):
    result = compare_df.to_string()    
    return result

def generate_insights_desc(result, target):
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    
    prompt_template_desc = """
    You are given descriptive statistics about a dataset. Generate a detailed and insightful description of the data, highlighting key metrics and their implications. The statistics are as follows:
    
    {result}
    Don't Draw table

    Target variable for analysis: {target}
    
    Provide a comprehensive analysis.
    """
    
    template_desc = PromptTemplate(
        input_variables=["result", "target"], template=prompt_template_desc
    )
    
    llm_chain_desc = LLMChain(llm=llm, prompt=template_desc)
    insights = llm_chain_desc.run(result=result, target=target)  # No need to convert result to key-value pairs
    return insights

def generate_model_insights(model_results, target, model_type):
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    
    prompt_template = """
    You are an advanced data science strategist with extensive expertise in machine learning, statistical analysis, and business strategy. Your task is to generate a **holistic, nuanced, and forward-thinking analysis** of the model performance results provided below, combining advanced analytical reasoning with actionable business insights.

    ### **Model Details:**
    - **Model Type:** {model_type}  
    - **Target Variable:** {target}  
    - **Performance Metrics:**  
    {model_results}  

    ---

    ### **Your Analysis Should Include:**

    1. **Deep Technical Analysis:**
    - Evaluate key metrics (accuracy, precision, recall, F1-score, AUC-ROC, R², RMSE, etc.) in detail.
    - Compare results with baselines and industry standards.
    - Detect overfitting, underfitting, and bias using methods like cross-validation and learning curves.
    - Analyze feature importance, multicollinearity, and the impact of data preprocessing.

    2. **Advanced Error Analysis:**
    - Conduct granular error analysis to identify failure modes and edge cases.
    - Examine residual distributions, diagnostic plots, and confusion matrices/ROC curves.
    - Assess the effects of data imbalances, outliers, or leakage.

    3. **Business Impact & Strategic Insights:**
    - Translate technical metrics into business implications (e.g., revenue, customer engagement, efficiency).
    - Evaluate risks from mispredictions and propose mitigation strategies.
    - Provide actionable insights aligning model performance with business objectives.

    4. **Decision-Making Support & Actionable Insights:**
    - Outline clear recommendations to guide decision-making.
    - Detail implications of model performance on both strategic and operational decisions.
    - Highlight trade-offs, risk factors, and prioritize actions for optimal outcomes.

    5. **Future-Proofing & Prescriptive Analysis:**
    - Forecast model performance considering evolving data and market trends.
    - Recommend continuous monitoring, periodic retraining, and adaptive strategies.
    - Discuss emerging technologies and offer prescriptive suggestions for a future-proof AI/ML strategy.

    **Formatting Guidelines:**
    - Use clear sections and bullet points.
    - Balance technical depth with strategic insights.
    - Maintain clarity and structure; avoid tables.

    Generate a comprehensive, advanced, and strategic analysis of the given model results.

    - **Maintain clarity and structure in your narrative while addressing both current performance and future enhancements.**

    Now, generate a **comprehensive, advanced, and strategic analysis** of the given model results.
    """



    template = PromptTemplate(
        input_variables=["model_results", "target", "model_type"],
        template=prompt_template
    )
    
    llm_chain = LLMChain(llm=llm, prompt=template)
    insights = llm_chain.run(model_results=model_results, target=target, model_type=model_type)
    return insights

def initialize_vectordb(TEXT_DATA):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(TEXT_DATA)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    return vector_store.as_retriever()

def generate_response(user_input, chat_history, retriever):
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    
    relevant_docs = retriever.get_relevant_documents(user_input)
    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    template = """
    Role: You are an AI expert in predictive and prescriptive analytics, focusing on model performance and business insights.
    
    Chat History:
    {chat_history}

    Context:
    {retrieved_context}
    
    User Question: {user_input}
    
    Provide a clear, informative response focusing on model insights and recommendations.
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input", "retrieved_context"],
        template=template,
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    response = llm_chain.run({
        "chat_history": chat_history,
        "user_input": user_input,
        "retrieved_context": retrieved_context,
    })
    
    return response

def post_analysis_chatbot(TEXT_DATA):
    # Using unique session state names for post analysis chatbot
    if "future_chat_history" not in st.session_state:
        st.session_state.future_chat_history = []
    
    if "future_retriever" not in st.session_state:
        st.session_state.future_retriever = initialize_vectordb(TEXT_DATA)
    
    for message in st.session_state.future_chat_history:
        role, content = message.split(":", 1)
        with st.chat_message(role.strip().lower()):
            st.markdown(content.strip())
    
    if user_input := st.chat_input("Ask about the model analysis:"):
        st.chat_message("user").markdown(user_input)
        st.session_state.future_chat_history.append(f"User: {user_input}")
        
        response = generate_response(
            user_input,
            "\n".join(st.session_state.future_chat_history),
            st.session_state.future_retriever
        )
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.future_chat_history.append(f"AI: {response}")

def generate_future_recommendations(df, model, target_column, model_type, compare_df):
    """Generate future recommendations based on model analysis and data insights."""
    recommendations = []
    
    # Analyze feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': df.drop(columns=[target_column]).columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(3)
        recommendations.append(f"Key factors affecting {target_column}:")
        for _, row in top_features.iterrows():
            recommendations.append(f"- {row['feature']} (importance: {row['importance']:.3f})")
    
    # Analyze data distribution and trends
    for col in df.columns:
        if col != target_column:
            if df[col].dtype in ['int64', 'float64']:
                # Check for outliers
                z_scores = stats.zscore(df[col])
                outliers = np.abs(z_scores) > 3
                if outliers.any():
                    recommendations.append(f"\nPotential outliers detected in {col}:")
                    recommendations.append(f"- Consider investigating extreme values in {col}")
                
                # Check for trends
                if len(df) > 10:  # Only check trends if we have enough data points
                    x = np.arange(len(df))
                    slope, _, r_value, _, _ = stats.linregress(x, df[col])
                    if abs(slope) > 0.1 and r_value**2 > 0.5:
                        trend = "increasing" if slope > 0 else "decreasing"
                        recommendations.append(f"\nTrend detected in {col}:")
                        recommendations.append(f"- {col} shows a {trend} trend over time")
    
    # Model-specific recommendations
    if "regression" in model_type.lower():
        # Analyze prediction errors
        y_pred = model.predict(df.drop(columns=[target_column]))
        errors = df[target_column] - y_pred
        mse = mean_squared_error(df[target_column], y_pred)
        r2 = r2_score(df[target_column], y_pred)
        
        recommendations.append("\nModel Performance Insights:")
        recommendations.append(f"- Mean Squared Error: {mse:.2f}")
        recommendations.append(f"- R² Score: {r2:.2f}")
        
        # Analyze error patterns
        if abs(errors.mean()) > 0.1:
            recommendations.append("\nPrediction Bias:")
            recommendations.append(f"- Model tends to {'overestimate' if errors.mean() < 0 else 'underestimate'} {target_column}")
    else:
        # Classification-specific insights
        y_pred = model.predict(df.drop(columns=[target_column]))
        accuracy = accuracy_score(df[target_column], y_pred)
        report = classification_report(df[target_column], y_pred, output_dict=True)
        
        recommendations.append("\nClassification Model Insights:")
        recommendations.append(f"- Overall Accuracy: {accuracy:.2f}")
        
        # Analyze class balance
        class_counts = df[target_column].value_counts()
        if len(class_counts) > 1:
            imbalance = class_counts.max() / class_counts.min()
            if imbalance > 2:
                recommendations.append("\nClass Imbalance Warning:")
                recommendations.append(f"- Significant class imbalance detected (ratio: {imbalance:.2f})")
                recommendations.append("- Consider using techniques like SMOTE or class weights")
    
    # Future predictions and scenarios
    recommendations.append("\nFuture Scenarios:")
    
    # Generate scenarios based on feature ranges
    for col in df.columns:
        if col != target_column and df[col].dtype in ['int64', 'float64']:
            current_mean = df[col].mean()
            current_std = df[col].std()
            
            # Best case scenario
            best_case = current_mean + current_std
            # Worst case scenario
            worst_case = current_mean - current_std
            
            recommendations.append(f"\n{col} Scenarios:")
            recommendations.append(f"- Best case: {best_case:.2f}")
            recommendations.append(f"- Worst case: {worst_case:.2f}")
    
    return "\n".join(recommendations)

def post_analysis_main(df, model, target_column, model_type, compare_df):
    """Main function for post-analysis and future recommendations."""
    # Create PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Post-Analysis Report", ln=True, align="C")
    
    # Add timestamp
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    # Generate recommendations
    recommendations = generate_future_recommendations(df, model, target_column, model_type, compare_df)
    
    # Add recommendations to PDF
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Future Recommendations and Insights", ln=True)
    pdf.set_font("Arial", "", 10)
    
    # Split recommendations into lines and add to PDF
    for line in recommendations.split('\n'):
        if line.strip():
            pdf.multi_cell(0, 10, line)
    
    # Save the PDF
    report_path = "reports/post_analysis_report.pdf"
    pdf.output(report_path)
    
    return recommendations, report_path

