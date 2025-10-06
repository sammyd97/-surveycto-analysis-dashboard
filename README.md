# SurveyCTO Data Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python Streamlit application for analyzing SurveyCTO survey data with interactive visualizations and comprehensive reporting.

## Features

- **Interactive Dashboard**: Clean, modern web interface for data exploration
- **Per-Question Analysis**: Detailed analysis for each question in your survey
- **Multiple Question Types**: Support for:
  - Single choice questions (`select_one`)
  - Multiple choice questions (`select_multiple`)
  - Numeric questions (`integer`, `decimal`)
  - Text questions (`text`)
- **Visualizations**: Automatic chart generation based on question type
- **Response Rate Tracking**: Monitor data completeness and quality
- **Filtering**: Filter analysis by question type

## Data Requirements

The application expects three CSV files in the same directory:

1. **`rawsurvey.csv`**: The main survey data export from SurveyCTO
2. **`questions.csv`**: Question metadata with columns:
   - `type`: Question type (select_one, select_multiple, integer, etc.)
   - `name`: Question name (column name in raw data)
   - `label`: Question label/description
   - `analysis note`: Optional analysis notes
3. **`choices.csv`**: Choice definitions with columns:
   - `list_name`: Choice list identifier
   - `value`: Choice value
   - `label`: Choice label
   - `image`: Optional image reference
   - `filter`: Optional filter condition

## Quick Start

### Option 1: Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sammyd97/-surveycto-analysis-dashboard.git
   cd -surveycto-analysis-dashboard
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your data files:**
   - Place your `rawsurvey.csv`, `questions.csv`, and `choices.csv` files in the project directory

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   - Navigate to `http://localhost:8501`

### Option 2: Deploy to Streamlit Cloud (Recommended for Team Sharing)

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Click "New app"**
4. **Connect your GitHub account** and select your forked repository
5. **Configure deployment**:
   - **Repository**: `yourusername/-surveycto-analysis-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
6. **Click "Deploy!"**
7. **Share the generated URL** with your team

### Live Demo
The application is deployed and available at: **[https://surveycto-analysis-dashboard.streamlit.app/](https://surveycto-analysis-dashboard.streamlit.app/)**

## Usage

The application automatically processes your SurveyCTO data and presents it in an interactive dashboard. No configuration is required - just run the app and explore your data!

### Key Features:
- **Smart Visualizations**: Automatically chooses the best chart type for each question
- **Interactive Exploration**: Hover over charts to see detailed information
- **Data Quality Insights**: Track response rates and data completeness
- **Flexible Analysis**: Handle various question types and data formats

## How It Works

1. **Data Loading**: The app loads and caches your survey data for fast performance
2. **Question Analysis**: Each question is analyzed based on its type:
   - **Single Choice**: Bar charts showing response distribution
   - **Multiple Choice**: Horizontal bar charts for option selection counts
   - **Numeric**: Histograms and descriptive statistics
   - **Text**: Sample responses and uniqueness metrics
3. **Visualization**: Interactive charts using Plotly for data exploration
4. **Reporting**: Comprehensive metrics including response rates and data quality

## Dashboard Sections

- **Sidebar**: Survey overview and filtering options
- **Main Area**: Question-by-question analysis with:
  - Response rate metrics
  - Interactive visualizations
  - Detailed statistics tables
  - Data quality indicators

## Customization

The application can be easily customized by:
- Modifying the CSS styles in the `st.markdown()` sections
- Adding new question types in the `analyze_question()` function
- Extending visualizations in the `create_visualization()` function
- Adding new filtering options in the sidebar

## Troubleshooting

- **File Not Found**: Ensure all CSV files are in the same directory as `app.py`
- **Data Loading Issues**: Check that your CSV files have the correct column names
- **Performance**: Large datasets are cached for better performance
- **Browser Issues**: Try refreshing the page or clearing browser cache

## Dependencies

- `streamlit==1.29.0`: Web application framework
- `pandas==2.1.4`: Data manipulation and analysis
- `plotly==5.17.0`: Interactive visualizations
- `numpy==1.24.3`: Numerical computing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues or questions:
1. Check that your data files are properly formatted
2. Ensure all dependencies are installed correctly
3. Verify your Python environment is compatible (Python 3.7+)
4. Open an issue on GitHub for bugs or feature requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Powered by [Plotly](https://plotly.com/) for interactive visualizations
- Data processing with [Pandas](https://pandas.pydata.org/)
