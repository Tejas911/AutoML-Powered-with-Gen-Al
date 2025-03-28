import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from langchain_groq import ChatGroq
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json
import re

# work on report for num and char
# improve the quality of data

load_dotenv()

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY"),
    # max_tokens=None,
    # timeout=None,
    # max_retries=2,
    # other params...
)


def generate_insights_desc(result, target):
    # Define a prompt template to guide the model in generating insights
    prompt_template_desc = """
    You are given descriptive statistics about a dataset. Generate a detailed and insightful description of the data, highlighting key metrics and their implications. The statistics are as follows:

    {result}

    Target variable for analysis: {target}

    Provide a comprehensive analysis.
    """

    # Create a LangChain prompt template
    template_desc = PromptTemplate(
        input_variables=["result", "target"], template=prompt_template_desc
    )

    # Create an LLMChain with the language model and prompt template
    llm_chain_desc = LLMChain(llm=llm, prompt=template_desc)
    # Convert the result dictionary to a string
    result_str = "\n".join([f"{key}: {value}" for key, value in result.items()])

    # Generate the insights
    insights = llm_chain_desc.run(result=result_str, target=target)
    # print(insights)
    return insights


def generate_diagnostic_analysis(result, target):
    """
    Generate a diagnostic analysis of the data based on descriptive statistics.

    Args:
    - result (dict): A dictionary containing descriptive statistics of the data.
    - target (str): The target variable for analysis.

    Returns:
    - str: A detailed diagnostic analysis of the data.
    """

    # Define a prompt template for diagnostic analysis
    diagnostic_prompt_template = """
    You are given descriptive statistics about a dataset. Perform a diagnostic analysis to identify potential issues, anomalies, or areas that require further investigation. The statistics are as follows:

    {result}

    Target variable for analysis: {target}

    Provide a detailed diagnostic analysis, highlighting any unusual patterns or possible concerns.
    """

    # Create a LangChain prompt template for diagnostic analysis
    diagnostic_template = PromptTemplate(
        input_variables=["result", "target"], template=diagnostic_prompt_template
    )

    # Create an LLMChain with the language model and diagnostic prompt template
    diagnostic_llm_chain = LLMChain(llm=llm, prompt=diagnostic_template)

    # Convert the result dictionary to a string
    result_str = "\n".join([f"{key}: {value}" for key, value in result.items()])

    # Generate the diagnostic analysis
    diagnostic_analysis = diagnostic_llm_chain.run(result=result_str, target=target)

    # print(diagnostic_analysis)
    return diagnostic_analysis


def descriptive_analysis(df, target):
    result = {}

    for column in df.columns:
        col_data = df[column]
        col_result = {}

        # Check if the column is numeric and not boolean
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(
            col_data
        ):
            # Basic statistics for numerical data
            col_result["mean"] = col_data.mean()
            col_result["median"] = col_data.median()
            col_result["mode"] = (
                col_data.mode()[0] if not col_data.mode().empty else None
            )
            col_result["min"] = col_data.min()
            col_result["max"] = col_data.max()

            # Measures of dispersion
            col_result["range"] = col_data.max() - col_data.min()
            col_result["variance"] = col_data.var()
            col_result["std_dev"] = col_data.std()
            col_result["iqr"] = col_data.quantile(0.75) - col_data.quantile(0.25)

            # Percentiles and quartiles
            col_result["25th_percentile"] = col_data.quantile(0.25)
            col_result["50th_percentile"] = col_data.median()  # Same as the median
            col_result["75th_percentile"] = col_data.quantile(0.75)

            # Skewness and Kurtosis
            col_result["skewness"] = col_data.skew()
            col_result["kurtosis"] = col_data.kurtosis()

            # Detect outliers using the IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

            # Calculate the percentage of outliers
            outlier_percentage = (len(outliers) / len(col_data)) * 100
            col_result["outlier_percentage"] = outlier_percentage

            # Calculate correlations with target
            if pd.api.types.is_numeric_dtype(df[target]):
                pearson_corr = col_data.corr(df[target])
                spearman_corr = col_data.corr(df[target], method="spearman")
                col_result["correlation_pearson"] = pearson_corr
                col_result["correlation_spearman"] = spearman_corr

        elif pd.api.types.is_bool_dtype(col_data):
            # Special handling for boolean data
            col_result["mode"] = (
                col_data.mode()[0] if not col_data.mode().empty else None
            )
            col_result["true_count"] = col_data.sum()
            col_result["false_count"] = len(col_data) - col_data.sum()

            # Calculate correlations with target
            if pd.api.types.is_numeric_dtype(df[target]):
                col_data_numeric = col_data.astype(int)
                pearson_corr = col_data_numeric.corr(df[target])
                spearman_corr = col_data_numeric.corr(df[target], method="spearman")
                col_result["correlation_pearson"] = pearson_corr
                col_result["correlation_spearman"] = spearman_corr

        else:
            # Statistics for categorical data
            col_result["mode"] = (
                col_data.mode()[0] if not col_data.mode().empty else None
            )
            col_result["unique_values"] = col_data.unique().tolist()
            col_result["frequency_distribution"] = col_data.value_counts().to_dict()

        # Store the result for the current column
        result[column] = col_result

    return result


# Function to make headings bold by replacing **heading** with <b>heading</b>
def make_headings_bold(text):
    bold_pattern = r"\*\*(.*?)\*\*"  # Regular expression to find **heading**
    formatted_text = re.sub(bold_pattern, r"<b>\1</b>", text)
    return formatted_text


# Function to create PDF report
def create_pdf_report(result, desc_insights, diago_insights, file_name="reports/report.pdf"):
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "Heading1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.blue,
        spaceAfter=12,
        leading=18,  # Increased line spacing for headings
    )

    subheading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        spaceAfter=10,
        leading=16,  # Increased line spacing for subheadings
    )

    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=12,
        spaceAfter=10,
        leading=14,  # Increased line spacing for normal text
    )

    # Apply the bold-heading logic to desc_insights and diago_insights
    desc_insights = make_headings_bold(desc_insights)
    diago_insights = make_headings_bold(diago_insights)

    # Add Descriptive Analysis subheading
    subheading = Paragraph("Descriptive Analysis", subheading_style)
    elements.append(subheading)

    # Add descriptive insights with proper line breaks
    desc_paragraph = Paragraph(desc_insights.replace("\n", "<br/>"), normal_style)
    elements.append(desc_paragraph)

    # Add some space between sections
    elements.append(Paragraph("<br/><br/>", normal_style))

    # Add Diagnostic Analysis subheading
    subheading = Paragraph("Diagnostic Analysis", subheading_style)
    elements.append(subheading)

    # Add diagnostic insights with proper line breaks
    diago_paragraph = Paragraph(diago_insights.replace("\n", "<br/>"), normal_style)
    elements.append(diago_paragraph)

    # Add some space between sections
    elements.append(Paragraph("<br/><br/>", normal_style))

    # Add Descriptive Analysis in Tabular Format subheading
    tabular_subheading = Paragraph(
        "Descriptive Analysis in Tabular Format", subheading_style
    )
    elements.append(tabular_subheading)

    for column, stats in result.items():
        # Add column heading
        col_heading = Paragraph(f"<b>{column}</b>", styles["Heading2"])
        elements.append(col_heading)

        # Prepare data for table
        table_data = [["Statistic", "Value"]]
        for stat_name, value in stats.items():
            if isinstance(value, (list, dict)):
                value = ", ".join(map(str, value))
            table_data.append([stat_name, value])

        # Set column widths for better table appearance
        col_widths = [200, 400]  # Adjust these values as needed

        table = Table(table_data, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("PAD", (0, 0), (-1, -1), 6),  # Add padding for better readability
                ]
            )
        )
        elements.append(table)

        # Add some space after each table
        elements.append(Paragraph("<br/><br/>", normal_style))

    doc.build(elements)


def descriptive_analysis_main(df, target):
    # Example of using the function
    TEXT_DATA = ""
    new_df = df
    TEXT_DATA += f"Features / Columns in this df are {list(df.columns)}"
    result = descriptive_analysis(new_df, target)
    result_str = str(result)
    TEXT_DATA += result_str
    desc_insights = generate_insights_desc(result, target)
    TEXT_DATA += desc_insights
    diago_insights = generate_diagnostic_analysis(result, target)
    TEXT_DATA += diago_insights
    create_pdf_report(result, desc_insights, diago_insights)
    return TEXT_DATA
