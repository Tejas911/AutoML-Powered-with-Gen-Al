# AutoML Powered with Gen-AI

A comprehensive machine learning platform that provides predictive and prescriptive analytics with advanced visualization capabilities and interactive analysis tools. This project combines the power of AutoML with Generative AI to provide intelligent insights and recommendations.

## Features

- **Data Analysis & Visualization**
  - Interactive data exploration
  - Advanced statistical analysis
  - Customizable visualizations
  - Trend analysis and pattern detection

- **Machine Learning Capabilities**
  - Support for both classification and regression models
  - Automated model selection and comparison
  - Feature importance analysis
  - Model performance evaluation

- **Advanced Analytics**
  - Predictive modeling
  - Prescriptive analytics
  - Future scenario generation
  - Automated insights generation

- **Interactive Interface**
  - Streamlit-based web interface
  - Real-time model training and evaluation
  - Interactive data visualization
  - Chatbot interface for analysis queries

## Project Structure

```
├── src/
│   ├── agents/
│   │   ├── ml_agent.py
│   │   └── data_agent.py
│   ├── analysis/
│   │   ├── post_analysis.py
│   │   └── data_analysis.py
│   ├── utils/
│   │   ├── llms.py
│   │   └── visualization.py
│   └── application.py
├── assets/
│   └── creative_logo_for_ml.jpeg
├── reports/
│   └── post_analysis_report.pdf
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tejas911/AutoML-Powered-with-Gen-Al.git
cd AutoML-Powered-with-Gen-Al
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
GROQ_API_KEY=your_groq_api_key
```

## Usage

1. Start the application:
```bash
streamlit run src/application.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload your dataset and begin analysis:
   - Select your target variable
   - Choose analysis type (classification/regression)
   - View generated insights and visualizations
   - Interact with the chatbot for detailed analysis

## Key Components

### ML Agent
- Handles model training and evaluation
- Supports multiple ML algorithms
- Provides model comparison and selection
- Automated hyperparameter tuning

### Data Agent
- Manages data preprocessing
- Handles feature engineering
- Performs data validation
- Automated data cleaning

### Post Analysis
- Generates comprehensive reports
- Provides future recommendations
- Creates interactive visualizations
- Offers chatbot interface for analysis queries
- Automated insights generation

## Features in Detail

### Data Analysis
- Statistical analysis
- Feature importance
- Correlation analysis
- Outlier detection
- Trend analysis
- Automated data profiling

### Model Training
- Automated model selection
- Hyperparameter optimization
- Cross-validation
- Performance metrics
- Model comparison
- Automated feature selection

### Visualization
- Interactive plots
- Customizable charts
- Real-time updates
- Export capabilities
- Multiple visualization types
- Custom theme support

### Reporting
- PDF report generation
- Performance metrics
- Future recommendations
- Actionable insights
- Automated report customization
- Export in multiple formats

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Powered by Groq API for advanced language models
- Uses various open-source ML libraries
- Special thanks to all contributors and maintainers

## Contact

For any questions or feedback, please open an issue in the GitHub repository or contact the maintainers. 