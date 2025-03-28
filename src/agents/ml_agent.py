from src.utils.llms import get_gemini, get_llama3_8b
from langchain import LLMChain
from langchain.prompts import PromptTemplate


# Create the prompt template
prompt_template = PromptTemplate(
    input_variables=["target", "unique_count", "data_type"],
    template="""
    The target variable in my dataset is named '{target}'.
    Here is some information about it:
    - Number of unique values: {unique_count}
    - Data type: {data_type}

    Based on this information, is this a regression or a classification problem?
    Respond with either "Regression" or "Classification".
    """,
)


# Function to determine model type
def determine_model_type_with_llm(df, target):
    # Get target column data info
    data_type = str(df[target].dtype)
    unique_count = df[target].nunique()
    llm = get_llama3_8b()
    # Use LLM to analyze and determine model type
    llm_chain = LLMChain(
        prompt=prompt_template, llm=llm  # The correct parameter name is 'prompt'
    )

    response = llm_chain.run(
        {"target": target, "unique_count": unique_count, "data_type": data_type}
    )

    return response.strip()  # Either "Regression" or "Classification"


regression_models = [
    "lr",  # Linear Regression
    "lasso",  # Lasso Regression
    "ridge",  # Ridge Regression
    "en",  # Elastic Net
    "lar",  # Least Angle Regression
    "llar",  # Lasso Least Angle Regression
    "omp",  # Orthogonal Matching Pursuit
    "br",  # Bayesian Ridge
    "ard",  # Automatic Relevance Determination
    "par",  # Passive Aggressive Regressor
    "ransac",  # Random Sample Consensus
    "tr",  # TheilSen Regressor
    "huber",  # Huber Regressor
    "kr",  # Kernel Ridge
    "svm",  # Support Vector Regression
    "knn",  # K Neighbors Regressor
    "dt",  # Decision Tree Regressor
    "rf",  # Random Forest Regressor
    "et",  # Extra Trees Regressor
    "ada",  # AdaBoost Regressor
    "gbr",  # Gradient Boosting Regressor
    "mlp",  # MLP Regressor
    "lightgbm",  # Light Gradient Boosting Machine
    "dummy",  # Dummy Regressor
]

classification_models = [
    "lr",  # Logistic Regression
    "knn",  # K Neighbors Classifier
    "nb",  # Naive Bayes
    "dt",  # Decision Tree Classifier
    "svm",  # SVM - Linear Kernel
    "rbfsvm",  # SVM - Radial Kernel
    "gpc",  # Gaussian Process Classifier
    "mlp",  # MLP Classifier
    "ridge",  # Ridge Classifier
    "rf",  # Random Forest Classifier
    "qda",  # Quadratic Discriminant Analysis
    "ada",  # Ada Boost Classifier
    "gbc",  # Gradient Boosting Classifier
    "lda",  # Linear Discriminant Analysis
    "et",  # Extra Trees Classifier
    "xgboost",  # Extreme Gradient Boosting
    "lightgbm",  # Light Gradient Boosting Machine
    "dummy",  # Dummy Classifier
]


def make_output_lst(output, models):
    # Convert the input string to lowercase to make the search case-insensitive
    output = output.lower()

    # Check for each model in the input string
    found_models = [model for model in models if model in output]

    return found_models


def get_classification_models_lst(data_frame_stats):
    """
    Uses LangChain to return a list of classification models based on the provided DataFrame stats.

    Parameters:
    - data_frame_stats (str): The stats of the DataFrame used in the prompt.
    - valid_models (list): A list of valid classification models to check against.

    Returns:
    - list: A list of classification models if valid, else an empty list.
    """

    # Define the prompt template
    prompt_template = """
    You are provided with the following data:
    {data_frame_stats}
    
    Based on this data, return a list of the 5 most suitable classification algorithms from the following:

    Classification Models:
    {classification_models}

    Select algorithms based on the following metrics:
    - Dataset size
    - Class imbalance
    - Feature distribution
    - Model complexity
    - Feature interaction
    - Interpretability

    Return only the list of algorithms, with no additional text.
    """

    # Create a LangChain prompt template
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["data_frame_stats", "classification_models"],
    )

    # Initialize the LLM model (you can use OpenAI or any other supported model)
    llm = get_gemini()  # Adjust the model or temperature if needed

    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    global classification_models
    # Run the chain with the provided DataFrame stats
    output = chain.run(
        data_frame_stats=data_frame_stats, classification_models=classification_models
    )

    # Split the output string into a list of algorithms
    output_list = make_output_lst(output, classification_models)
    print("Suggested classification algorithms:", output_list)
    return output_list


def get_regression_model_list(data_frame_stats):
    """
    Uses LangChain to return a list of regression models based on the provided DataFrame stats.

    Parameters:
    - data_frame_stats (str): The stats of the DataFrame used in the prompt.
    - valid_models (list): A list of valid regression models to check against.

    Returns:
    - list: A list of regression models if valid, else an empty list.
    """

    # Define the prompt template
    prompt_template = """
    You are provided with the following data:
    {data_frame_stats}
    
    Based on this data, return a list of the 5 most suitable regression algorithms from the following:

    Regression Models:
    {regression_models}

    Select algorithms based on the following metrics:
    - Dataset size
    - Presence of outliers
    - Feature distribution
    - Collinearity
    - Model complexity
    - Feature interaction

    Return only the list of algorithms, with no additional text.
    """

    # Create a LangChain prompt template
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["data_frame_stats", "regression_models"],
    )

    # Initialize the LLM model (you can use OpenAI or any other supported model)
    llm = get_gemini()  # Adjust the temperature if needed

    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    global regression_models
    # Run the chain with the provided DataFrame stats
    output = chain.run(
        data_frame_stats=data_frame_stats, regression_models=regression_models
    )

    # Split the output string into a list of algorithms
    output_list = make_output_lst(output, regression_models)
    print("Suggested algorithms:", output_list)
    return output_list
