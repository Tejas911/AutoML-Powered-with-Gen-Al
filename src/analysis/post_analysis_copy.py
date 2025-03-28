import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from langchain_groq import ChatGroq
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

load_dotenv()

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY"),
)

def generate_predictive_insights(model_results, model_type):
    prompt_template = """
    You are analyzing the performance of a machine learning model. Generate detailed insights about the model's predictive capabilities based on the following metrics:

    {results}

    Model Type: {model_type}

    Provide a comprehensive analysis of:
    1. Model Performance
    2. Prediction Accuracy
    3. Areas of Strength
    4. Areas for Improvement
    5. Reliability of Predictions
    """

    template = PromptTemplate(
        input_variables=["results", "model_type"],
        template=prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=template)
    results_str = "\n".join([f"{key}: {value}" for key, value in model_results.items()])
    insights = llm_chain.run(results=results_str, model_type=model_type)
    return insights

def generate_prescriptive_insights(data_info, model_results, target):
    prompt_template = """
    Based on the model analysis and data characteristics, provide actionable recommendations:

    Data Information: {data_info}
    Model Results: {model_results}
    Target Variable: {target}

    Please provide:
    1. Specific Actions to Improve Outcomes
    2. Strategic Recommendations
    3. Risk Mitigation Strategies
    4. Implementation Guidelines
    5. Expected Impact of Recommendations
    """

    template = PromptTemplate(
        input_variables=["data_info", "model_results", "target"],
        template=prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=template)
    insights = llm_chain.run(
        data_info=str(data_info),
        model_results=str(model_results),
        target=target
    )
    return insights

def analyze_model_performance(model, df, target, model_type):
    results = {}
    
    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Get predictions
    y_pred = model.predict(X)
    
    if "regression" in model_type.lower():
        results['mse'] = mean_squared_error(y, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['r2'] = r2_score(y, y_pred)
        
        # Additional regression metrics
        results['mae'] = np.mean(np.abs(y - y_pred))
        results['explained_variance'] = np.var(y_pred) / np.var(y)
        
    else:  # Classification
        results['accuracy'] = accuracy_score(y, y_pred)
        results['classification_report'] = classification_report(y, y_pred)
        results['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
        
    return results

def create_post_analysis_report(predictive_insights, prescriptive_insights, model_results, file_name="post_analysis_report.pdf"):
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "Heading1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.blue,
        spaceAfter=12
    )
    
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=12,
        spaceAfter=10
    )
    
    # Add Predictive Analysis Section
    elements.append(Paragraph("Predictive Analysis", heading_style))
    elements.append(Paragraph(predictive_insights.replace("\n", "<br/>"), normal_style))
    elements.append(Paragraph("<br/>", normal_style))
    
    # Add Prescriptive Analysis Section
    elements.append(Paragraph("Prescriptive Analysis", heading_style))
    elements.append(Paragraph(prescriptive_insights.replace("\n", "<br/>"), normal_style))
    elements.append(Paragraph("<br/>", normal_style))
    
    # Add Model Metrics Section
    elements.append(Paragraph("Model Performance Metrics", heading_style))
    
    # Create table for metrics
    table_data = [["Metric", "Value"]]
    for metric, value in model_results.items():
        if isinstance(value, (int, float)):
            table_data.append([metric, f"{value:.4f}"])
        else:
            table_data.append([metric, str(value)])
    
    table = Table(table_data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    doc.build(elements)
    return file_name

def post_analysis_main(df, model, target, model_type):
    # Get model performance metrics
    model_results = analyze_model_performance(model, df, target, model_type)
    
    # Generate insights
    predictive_insights = generate_predictive_insights(model_results, model_type)
    prescriptive_insights = generate_prescriptive_insights(
        df.describe().to_dict(),
        model_results,
        target
    )
    
    # Create PDF report
    report_path = create_post_analysis_report(
        predictive_insights,
        prescriptive_insights,
        model_results
    )
    
    # Combine all text data for the chatbot
    TEXT_DATA = f"""
    Model Type: {model_type}
    Target Variable: {target}
    
    Predictive Analysis:
    {predictive_insights}
    
    Prescriptive Analysis:
    {prescriptive_insights}
    
    Model Results:
    {str(model_results)}
    """
    
    return TEXT_DATA, report_path 