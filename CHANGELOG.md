# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-10-06

### Added
- Initial release of SurveyCTO Data Analysis Dashboard
- Interactive Streamlit web application for survey data analysis
- Support for multiple question types:
  - Single choice questions (select_one) with pie charts
  - Multiple choice questions (select_multiple) with horizontal bar charts
  - Numeric questions (integer, decimal) with grouped bar charts
  - Text questions with full response listings
- Smart data processing features:
  - Automatic job categorization with pie charts
  - Phone number matching analysis
  - Age-specific handling (individual responses)
  - Household size grouping with outlier handling
  - Travel time grouping in 15-minute chunks
- User experience improvements:
  - Auto-hide questions with no data
  - Collapsible long lists with preview functionality
  - Interactive hover tooltips showing raw counts
  - Percentage-based select_multiple visualizations
  - Proper space-separated value parsing
- Data quality features:
  - Missing value handling (treats "", "NA", "N/A", "na" as missing)
  - Response rate tracking
  - Data completeness metrics
- Visualization features:
  - Pie charts for single choice questions
  - Horizontal bar charts for multiple choice (percentage axis)
  - Grouped bar charts for numeric data
  - Smart grouping logic (max 7 groups)
  - Interactive Plotly charts

### Technical Features
- Robust error handling for binary data conversion
- Efficient data caching with Streamlit
- Clean, modern UI with custom CSS styling
- Comprehensive data validation
- Flexible question metadata processing
- Choice label mapping from SurveyCTO exports

### Data Requirements
- `rawsurvey.csv`: Main survey data export from SurveyCTO
- `questions.csv`: Question metadata (type, name, label, analysis note)
- `choices.csv`: Choice definitions (list_name, value, label)
