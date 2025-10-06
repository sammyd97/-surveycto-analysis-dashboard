import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="SurveyCTO Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(survey_type='caregiver'):
    """Load and cache the survey data files based on survey type"""
    try:
        if survey_type == 'caregiver':
            # Load the main survey data
            survey_df = pd.read_csv('rawsurvey.csv')
            questions_df = pd.read_csv('questions.csv')
            choices_df = pd.read_csv('choices.csv')
        elif survey_type == 'facility':
            survey_df = pd.read_csv('Facility observations analysis 23 Sep - Facility observations.csv')
            questions_df = pd.read_csv('facility obs questions - Sheet1 (1).csv')
            choices_df = pd.read_csv('choices.csv')
        elif survey_type == 'health':
            survey_df = pd.read_csv('NA Health worker survey export 23 sep - NA Health worker survey.csv')
            questions_df = pd.read_csv('health worker questions - Sheet1.csv')
            choices_df = pd.read_csv('choices.csv')
        
        return survey_df, questions_df, choices_df
    except FileNotFoundError as e:
        st.error(f"Error loading {survey_type} data files: {e}")
        return None, None, None

def get_question_info(question_name, questions_df):
    """Get question information from questions metadata"""
    question_info = questions_df[questions_df['name'] == question_name]
    if not question_info.empty:
        return question_info.iloc[0]
    return None

def get_choice_labels(question_name, choices_df):
    """Get choice labels for a question"""
    # Find the list name for this question from choices
    choice_mapping = {}
    
    # For select_one questions, find the corresponding list
    for _, choice in choices_df.iterrows():
        list_name = choice['list_name']
        value = choice['value']
        label = choice['label']
        
        if pd.notna(list_name) and pd.notna(value) and pd.notna(label):
            choice_mapping[value] = label
    
    return choice_mapping

def minutes_to_time_format(minutes):
    """Convert minutes from midnight back to HH:MM format"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

def categorize_jobs(job_responses):
    """Categorize job responses into meaningful groups"""
    categories = {
        'Agriculture/Farming': 0,
        'Healthcare': 0,
        'Education': 0,
        'Business/Trading': 0,
        'Government/Public Service': 0,
        'Manual Labor/Construction': 0,
        'Unemployed/Homemaker': 0,
        'Other': 0
    }
    
    for job in job_responses:
        job_lower = str(job).lower().strip()
        
        if any(word in job_lower for word in ['farm', 'agriculture', 'crop', 'plant', 'grow']):
            categories['Agriculture/Farming'] += 1
        elif any(word in job_lower for word in ['nurse', 'doctor', 'health', 'medical', 'healthcare']):
            categories['Healthcare'] += 1
        elif any(word in job_lower for word in ['teacher', 'education', 'school', 'teach']):
            categories['Education'] += 1
        elif any(word in job_lower for word in ['business', 'trade', 'sell', 'market', 'shop', 'store']):
            categories['Business/Trading'] += 1
        elif any(word in job_lower for word in ['government', 'public', 'civil', 'service', 'official']):
            categories['Government/Public Service'] += 1
        elif any(word in job_lower for word in ['labor', 'construction', 'build', 'worker', 'manual']):
            categories['Manual Labor/Construction'] += 1
        elif any(word in job_lower for word in ['unemployed', 'home', 'housewife', 'none', 'nothing', '']):
            categories['Unemployed/Homemaker'] += 1
        else:
            categories['Other'] += 1
    
    # Remove categories with 0 count
    return {k: v for k, v in categories.items() if v > 0}

def create_grouped_data(numeric_data, question_name, question_type=None, survey_type='caregiver'):
    """Create grouped data for integer questions with smart grouping logic"""
    if len(numeric_data) == 0:
        return None
    
    # Special handling for time questions - 1 hour blocks
    if question_type == 'time':
        def group_time_blocks(minutes):
            hour = int(minutes // 60)
            return f"{hour:02d}-{hour+1:02d}"
        
        grouped = numeric_data.apply(group_time_blocks)
        # Sort by hour value
        sorted_groups = grouped.value_counts().sort_index(key=lambda x: [int(block.split('-')[0]) for block in x])
        return sorted_groups
    
    # For observations and health worker surveys - show individual responses without grouping
    if survey_type in ['facility', 'health']:
        return numeric_data.value_counts().sort_index()
    
    # Special handling for age - show all individual ages (caregiver survey only)
    if 'age' in question_name.lower() and survey_type == 'caregiver':
        return numeric_data.value_counts().sort_index()
    
    # Special handling for household size (caregiver survey only)
    if ('household_size' in question_name.lower() or 'household' in question_name.lower()) and survey_type == 'caregiver':
        # Group household sizes: 1-2, 3-4, 5-6, 7-8, 9-10, 11+
        def household_group(x):
            if x <= 2:
                return "1-2 people"
            elif x <= 4:
                return "3-4 people"
            elif x <= 6:
                return "5-6 people"
            elif x <= 8:
                return "7-8 people"
            elif x <= 10:
                return "9-10 people"
            else:
                return "11+ people"
        
        grouped = numeric_data.apply(household_group)
        result = grouped.value_counts()
        # Sort by group order, not alphabetically
        order = ["1-2 people", "3-4 people", "5-6 people", "7-8 people", "9-10 people", "11+ people"]
        return result.reindex([x for x in order if x in result.index])
    
    # Special handling for travel time - use 15-minute chunks (caregiver survey only)
    if ('travel_time' in question_name.lower() or 'minutes' in question_name.lower()) and survey_type == 'caregiver':
        def travel_group(x):
            if x <= 15:
                return "0-15 min"
            elif x <= 30:
                return "16-30 min"
            elif x <= 45:
                return "31-45 min"
            elif x <= 60:
                return "46-60 min"
            elif x <= 90:
                return "61-90 min"
            elif x <= 120:
                return "91-120 min"
            else:
                return "120+ min"
        
        grouped = numeric_data.apply(travel_group)
        result = grouped.value_counts()
        # Sort by time order
        order = ["0-15 min", "16-30 min", "31-45 min", "46-60 min", "61-90 min", "91-120 min", "120+ min"]
        return result.reindex([x for x in order if x in result.index])
    
    # General integer grouping logic (caregiver survey only)
    min_val = numeric_data.min()
    max_val = numeric_data.max()
    data_range = max_val - min_val
    
    # If range is small, use individual values
    if max_val - min_val <= 6:
        return numeric_data.value_counts().sort_index()
    
    # Create 5-7 groups based on data distribution
    num_groups = min(7, max(5, int(np.ceil(data_range / 10))))
    
    # Use quantile-based grouping for better distribution
    if len(numeric_data) > num_groups:
        percentiles = np.linspace(0, 100, num_groups + 1)
        bin_edges = np.percentile(numeric_data, percentiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        # Create labels
        labels = []
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] == bin_edges[i+1]:
                continue
            if i == len(bin_edges) - 2:  # Last group
                labels.append(f"{int(bin_edges[i])}+")
            else:
                labels.append(f"{int(bin_edges[i])}-{int(bin_edges[i+1]-1)}")
        
        # Group the data
        grouped = pd.cut(numeric_data, bins=bin_edges, labels=labels, include_lowest=True)
        result = grouped.value_counts()
        # Sort by numeric order of the ranges
        return result.sort_index()
    
    return numeric_data.value_counts().sort_index()

def analyze_question(question_name, survey_df, questions_df, choices_df, survey_type='caregiver'):
    """Analyze a single question and return analysis results"""
    if question_name not in survey_df.columns:
        return None
    
    question_info = get_question_info(question_name, questions_df)
    if question_info is None:
        return None
    
    question_type = question_info['type']
    question_label = question_info.get('label', question_name)
    
    # Get the data for this question
    question_data = survey_df[question_name].dropna()
    
    # Auto-hide questions with no data
    if len(question_data) == 0:
        return None  # This will skip the question entirely
    
    analysis = {}
    choice_labels = get_choice_labels(question_name, choices_df)
    
    if 'select_one' in question_type:
        # Single choice question
        value_counts = question_data.value_counts()
        
        # Replace values with labels if available
        if choice_labels:
            value_counts.index = value_counts.index.map(
                lambda x: choice_labels.get(x, x)
            )
        
        analysis['value_counts'] = value_counts
        analysis['percentage'] = (value_counts / len(question_data) * 100).round(1)
        
    elif 'select_multiple' in question_type:
        # Multiple choice question - analyze space-separated values
        # Get the main question column (not the binary columns)
        main_question_data = survey_df[question_name].dropna()
        
        # Clean the data - treat empty, NA, N/A, na as missing
        def is_missing(value):
            if pd.isna(value):
                return True
            str_val = str(value).strip().lower()
            return str_val in ['', 'nan', 'na', 'n/a', 'none']
        
        # Filter out missing responses
        valid_responses = main_question_data[~main_question_data.apply(is_missing)]
        
        if len(valid_responses) == 0:
            return {
                'type': question_type,
                'label': question_label,
                'total_responses': 0,
                'missing_responses': len(survey_df) - len(valid_responses),
                'data': valid_responses,
                'analysis': None
            }
        
        # Count each option
        option_counts = {}
        total_respondents = len(valid_responses)
        
        for response in valid_responses:
            # Split by space and get unique options
            options = set(str(response).strip().split())
            
            for option in options:
                if option.strip():  # Skip empty strings
                    option_counts[option] = option_counts.get(option, 0) + 1
        
        # Replace with labels if available
        if choice_labels:
            labeled_counts = {}
            for option, count in option_counts.items():
                label = choice_labels.get(option, option)
                labeled_counts[label] = count
            option_counts = labeled_counts
        
        analysis['option_counts'] = option_counts
        # Calculate percentage based on actual question respondents
        analysis['percentage'] = {k: round(v / total_respondents * 100, 1) 
                                for k, v in option_counts.items()}
        analysis['total_question_respondents'] = total_respondents
    
    elif question_type in ['integer', 'decimal', 'time']:
        # Numeric question (including time questions)
        if question_type == 'time':
            # Handle time format data (HH:MM:SS)
            time_data = question_data.dropna()
            numeric_values = []
            
            for time_str in time_data:
                try:
                    # Try to parse time format HH:MM:SS or HH:MM
                    if ':' in str(time_str):
                        time_parts = str(time_str).split(':')
                        if len(time_parts) >= 2:
                            hours = int(time_parts[0])
                            minutes = int(time_parts[1])
                            # Convert to minutes from midnight (ignore seconds)
                            total_minutes = hours * 60 + minutes
                            numeric_values.append(total_minutes)
                    else:
                        # Try regular numeric conversion
                        numeric_values.append(float(time_str))
                except (ValueError, IndexError):
                    continue
            
            if numeric_values:
                numeric_data = pd.Series(numeric_values)
            else:
                numeric_data = pd.Series(dtype=float)
        else:
            # Regular numeric conversion for integer/decimal
            numeric_data = pd.to_numeric(question_data, errors='coerce').dropna()
        
        if len(numeric_data) > 0:
            analysis['mean'] = numeric_data.mean()
            analysis['median'] = numeric_data.median()
            analysis['std'] = numeric_data.std()
            analysis['min'] = numeric_data.min()
            analysis['max'] = numeric_data.max()
            analysis['histogram_data'] = numeric_data
            
            # Create grouped data for visualization
            grouped_data = create_grouped_data(numeric_data, question_name, question_type, survey_type)
            analysis['grouped_data'] = grouped_data
    
    elif question_type == 'text':
        # Text question - show all responses
        text_responses = question_data.astype(str)
        
        # Special handling for "other" questions - get actual text responses
        if 'other' in question_name.lower():
            # SurveyCTO exports duplicate columns for "other" questions
            # Pandas renames them as question_name.1, question_name.2, etc.
            # The text responses are in the .1 version
            text_column = f"{question_name}.1"
            
            if text_column in survey_df.columns:
                # Get text responses from the .1 column
                text_responses = survey_df[text_column].dropna().astype(str)
                
                # Filter out empty/null responses
                text_responses = text_responses[text_responses.str.strip() != '']
                text_responses = text_responses[text_responses.str.lower() != 'nan']
                text_responses = text_responses[text_responses.str.lower() != 'none']
            else:
                # Fallback to original logic if .1 column doesn't exist
                text_responses = text_responses[text_responses.str.strip() != '']
                text_responses = text_responses[text_responses.str.lower() != 'nan']
        
        analysis['all_responses'] = text_responses.tolist()
        analysis['unique_responses'] = text_responses.nunique()
        analysis['total_responses'] = len(text_responses)
        
        # Special handling for occupation/job questions
        if 'occupation' in question_name.lower() or 'job' in question_name.lower():
            job_categories = categorize_jobs(text_responses)
            analysis['job_categories'] = job_categories
    
    return {
        'type': question_type,
        'label': question_label,
        'total_responses': len(question_data),
        'missing_responses': len(survey_df) - len(question_data),
        'data': question_data,
        'analysis': analysis
    }

def create_visualization(question_result):
    """Create appropriate visualization for the question"""
    if question_result is None or question_result['analysis'] is None:
        return None
    
    analysis = question_result['analysis']
    question_type = question_result['type']
    
    if 'select_one' in question_type and 'value_counts' in analysis:
        # Pie chart for single choice
        fig = px.pie(
            values=analysis['value_counts'].values,
            names=analysis['value_counts'].index,
            title=f"Distribution of Responses: {question_result['label']}"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    elif 'select_multiple' in question_type and 'option_counts' in analysis:
        # Horizontal bar chart for multiple choice with percentages
        options = list(analysis['option_counts'].keys())
        counts = list(analysis['option_counts'].values())
        percentages = list(analysis['percentage'].values())
        
        # Create DataFrame for plotly
        df = pd.DataFrame({
            'Option': options,
            'Percentage': percentages,
            'Count': counts
        })
        
        fig = px.bar(
            df,
            x='Percentage',
            y='Option',
            orientation='h',
            title=f"Multiple Choice Responses: {question_result['label']}",
            labels={'Percentage': 'Percentage (%)', 'Option': 'Option'},
            color='Percentage',
            color_continuous_scale='Viridis',
            hover_data={'Count': True}
        )
        
        # Add custom hover text showing raw numbers
        total_respondents = analysis.get('total_question_respondents', 'N/A')
        fig.update_traces(
            hovertemplate=f'<b>%{{y}}</b><br>Percentage: %{{x}}%<br>Count: %{{customdata[0]}}<br>Out of {total_respondents} question respondents<extra></extra>',
            customdata=df[['Count']]
        )
        
        fig.update_layout(showlegend=False)
        return fig
    
    elif question_type in ['integer', 'decimal', 'time'] and 'grouped_data' in analysis:
        # Grouped bar chart for numeric data
        grouped_data = analysis['grouped_data']
        if grouped_data is not None and len(grouped_data) > 0:
            fig = px.bar(
                x=grouped_data.index,
                y=grouped_data.values,
                title=f"Distribution: {question_result['label']}",
                labels={'x': 'Group', 'y': 'Count'},
                color=grouped_data.values,
                color_continuous_scale='Viridis'
            )
            
            # For time questions, make sure x-axis is treated as categorical
            if question_type == 'time':
                fig.update_layout(
                    xaxis_type='category',
                    showlegend=False
                )
            else:
                fig.update_layout(showlegend=False)
            return fig
        else:
            # Fallback to histogram if grouping fails
            fig = px.histogram(
                x=analysis['histogram_data'],
                title=f"Distribution: {question_result['label']}",
                labels={'x': 'Value', 'y': 'Frequency'},
                nbins=20
            )
            return fig
    
    elif question_type == 'text' and 'job_categories' in analysis:
        # Pie chart for job categories
        job_cats = analysis['job_categories']
        if job_cats:
            fig = px.pie(
                values=list(job_cats.values()),
                names=list(job_cats.keys()),
                title=f"Job Categories: {question_result['label']}"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            return fig
    
    return None

def analyze_phone_number_matching(survey_df):
    """Analyze phone number matching between card and self-reported"""
    if 'phone_number_card' not in survey_df.columns or 'phone_number' not in survey_df.columns:
        return None
    
    # Get data where both questions were answered
    card_phones = survey_df['phone_number_card'].dropna().astype(str)
    reported_phones = survey_df['phone_number'].dropna().astype(str)
    
    # Clean phone numbers (remove spaces, dashes, etc.)
    def clean_phone(phone):
        return ''.join(filter(str.isdigit, str(phone)))
    
    card_phones_clean = card_phones.apply(clean_phone)
    reported_phones_clean = reported_phones.apply(clean_phone)
    
    # Find matching indices
    common_indices = set(card_phones.index) & set(reported_phones.index)
    
    matches = 0
    total_comparisons = 0
    
    for idx in common_indices:
        if pd.notna(card_phones.loc[idx]) and pd.notna(reported_phones.loc[idx]):
            card_clean = clean_phone(card_phones.loc[idx])
            reported_clean = clean_phone(reported_phones.loc[idx])
            
            if card_clean and reported_clean:  # Both have digits
                total_comparisons += 1
                if card_clean == reported_clean:
                    matches += 1
    
    return {
        'matches': matches,
        'total_comparisons': total_comparisons,
        'match_rate': (matches / total_comparisons * 100) if total_comparisons > 0 else 0
    }

def analyze_survey(survey_type, survey_title):
    """Analyze a specific survey"""
    # Load data for this survey
    survey_df, questions_df, choices_df = load_data(survey_type)
    
    if survey_df is None:
        st.error(f"Failed to load {survey_title} data files. Please ensure all CSV files are in the same directory as this app.")
        return
    
    # Sidebar filters
    st.sidebar.header(f"📋 {survey_title} Overview")
    
    # Basic statistics
    total_responses = len(survey_df)
    st.sidebar.metric("Total Responses", total_responses)
    
    # Filter out metadata rows and get actual questions
    # Remove rows with empty names or special types like 'start', 'end', 'begin group', etc.
    metadata_types = ['start', 'end', 'calculate', 'text audit', 'begin group', 'end group', 'note', 'geopoint']
    actual_questions = questions_df[
        questions_df['name'].notna() & 
        (~questions_df['type'].isin(metadata_types)) &
        (questions_df['name'] != '') &
        (questions_df['name'].str.strip() != '')
    ].copy()
    
    # Question name filter (instead of type filter)
    question_names = actual_questions['name'].unique()
    selected_questions = st.sidebar.multiselect(
        "Filter by Question Name",
        options=question_names,
        default=question_names
    )
    
    # Filter questions by selected names
    filtered_questions = actual_questions[
        actual_questions['name'].isin(selected_questions)
    ]
    
    st.sidebar.metric("Questions to Analyze", len(filtered_questions))
    
    # Main content
    if len(filtered_questions) == 0:
        st.warning("No questions selected. Please choose question types in the sidebar.")
        return
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Analyze each question
    for idx, (_, question_row) in enumerate(filtered_questions.iterrows()):
        question_name = question_row['name']
        
        # Update progress
        progress_bar.progress((idx + 1) / len(filtered_questions))
        
        # Skip empty question names
        if pd.isna(question_name) or question_name == '':
            continue
        
        # Analyze question
        question_result = analyze_question(question_name, survey_df, questions_df, choices_df, survey_type)
        
        # Skip questions with no data (auto-hide)
        if question_result is None:
            continue
        
        # Display question header
        st.markdown(f'<div class="question-header">Question: {question_result["label"]}</div>', 
                   unsafe_allow_html=True)
        
        # Display question metadata
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Response Rate", f"{question_result['total_responses']}/{total_responses}")
        
        with col2:
            response_rate = (question_result['total_responses'] / total_responses * 100) if total_responses > 0 else 0
            st.metric("Response Rate %", f"{response_rate:.1f}%")
        
        with col3:
            st.metric("Missing Responses", question_result['missing_responses'])
        
        with col4:
            st.metric("Question Type", question_result['type'])
        
        # Display analysis
        analysis = question_result['analysis']
        
        if analysis is None:
            st.info("No analysis available for this question.")
            continue
        
        # Create visualizations
        fig = create_visualization(question_result)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed results
        if 'select_one' in question_result['type'] and 'value_counts' in analysis:
            # Single choice results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Response Counts")
                value_df = pd.DataFrame({
                    'Response': analysis['value_counts'].index,
                    'Count': analysis['value_counts'].values,
                    'Percentage': analysis['percentage'].values
                })
                
                # Use collapsible for long lists
                if len(value_df) > 8:
                    with st.expander(f"View all {len(value_df)} responses", expanded=False):
                        st.dataframe(value_df, use_container_width=True)
                    
                    # Show summary
                    st.write("**Summary:**")
                    st.dataframe(value_df.head(5), use_container_width=True)
                    if len(value_df) > 5:
                        st.write(f"... and {len(value_df) - 5} more (click expand to see all)")
                else:
                    st.dataframe(value_df, use_container_width=True)
            
            with col2:
                st.subheader("Summary Statistics")
                st.write(f"**Most Common Response:** {analysis['value_counts'].index[0]} ({analysis['percentage'].iloc[0]}%)")
                st.write(f"**Total Unique Responses:** {len(analysis['value_counts'])}")
        
        elif 'select_multiple' in question_result['type'] and 'option_counts' in analysis:
            # Multiple choice results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Option Selection Counts")
                option_df = pd.DataFrame({
                    'Option': list(analysis['option_counts'].keys()),
                    'Count': list(analysis['option_counts'].values()),
                    'Percentage': list(analysis['percentage'].values())
                })
                
                # Use collapsible for long lists
                if len(option_df) > 8:
                    with st.expander(f"View all {len(option_df)} options", expanded=False):
                        st.dataframe(option_df, use_container_width=True)
                    
                    # Show summary
                    st.write("**Summary:**")
                    st.dataframe(option_df.head(5), use_container_width=True)
                    if len(option_df) > 5:
                        st.write(f"... and {len(option_df) - 5} more (click expand to see all)")
                else:
                    st.dataframe(option_df, use_container_width=True)
            
            with col2:
                st.subheader("Summary")
                most_selected = max(analysis['option_counts'], key=analysis['option_counts'].get)
                st.write(f"**Most Selected Option:** {most_selected}")
                st.write(f"• Selected by: {analysis['option_counts'][most_selected]} people ({analysis['percentage'][most_selected]}% of question respondents)")
                st.write(f"**Total Options Selected:** {len(analysis['option_counts'])}")
                st.write(f"**Question Respondents:** {analysis.get('total_question_respondents', 'N/A')}")
                st.write(f"**Total Survey Respondents:** {len(survey_df)}")
        
        elif question_result['type'] in ['integer', 'decimal', 'time']:
            # Numeric results - check if statistics exist
            if all(key in analysis for key in ['mean', 'median', 'std', 'min', 'max']):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Descriptive Statistics")
                    
                    # Check if this is a time question to format differently
                    if question_result['type'] == 'time':
                        stats_df = pd.DataFrame({
                            'Statistic': ['Mean Time', 'Median Time', 'Std Dev (min)', 'Earliest', 'Latest'],
                            'Value': [
                                minutes_to_time_format(analysis['mean']),
                                minutes_to_time_format(analysis['median']),
                                f"{analysis['std']:.1f} minutes",
                                minutes_to_time_format(analysis['min']),
                                minutes_to_time_format(analysis['max'])
                            ]
                        })
                    else:
                        stats_df = pd.DataFrame({
                            'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max'],
                            'Value': [
                                f"{analysis['mean']:.2f}",
                                f"{analysis['median']:.2f}",
                                f"{analysis['std']:.2f}",
                                f"{analysis['min']:.2f}",
                                f"{analysis['max']:.2f}"
                            ]
                        })
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    st.subheader("Data Summary")
                    if 'histogram_data' in analysis:
                        st.write(f"**Valid Responses:** {len(analysis['histogram_data'])}")
                    st.write(f"**Missing Values:** {question_result['missing_responses']}")
                
                # Special handling for time questions - show time_why responses
                if question_result['type'] == 'time' and 'time_preference' in question_result['label'].lower():
                    st.markdown("---")
                    st.subheader("📝 Time Preference Explanations")
                    
                    # Try to find the corresponding time_why column
                    time_why_column = None
                    for col in survey_df.columns:
                        if 'time_preference_why' in col.lower():
                            time_why_column = col
                            break
                    
                    if time_why_column and time_why_column in survey_df.columns:
                        # Get time_why responses for non-empty time preferences
                        time_responses = survey_df[question_name].dropna()
                        why_responses = survey_df.loc[time_responses.index, time_why_column].dropna()
                        
                        # Filter out empty responses
                        why_responses = why_responses[why_responses.str.strip() != '']
                        why_responses = why_responses[why_responses.str.lower() != 'nan']
                        
                        if len(why_responses) > 0:
                            st.write("**Reasons for time preferences:**")
                            if len(why_responses) > 10:
                                with st.expander(f"View all {len(why_responses)} explanations", expanded=False):
                                    for i, response in enumerate(why_responses, 1):
                                        st.write(f"**{i}.** {response}")
                                
                                # Show first few as preview
                                st.write("**Preview (first 5 explanations):**")
                                for i, response in enumerate(why_responses.head(5), 1):
                                    st.write(f"**{i}.** {response}")
                                if len(why_responses) > 5:
                                    st.write(f"... and {len(why_responses) - 5} more (click expand to see all)")
                            else:
                                for i, response in enumerate(why_responses, 1):
                                    st.write(f"**{i}.** {response}")
                        else:
                            st.info("No explanations provided for time preferences.")
                    else:
                        st.info("No corresponding 'why' question found for this time preference.")
            else:
                st.info("No numeric statistics available for this question.")
        
        elif question_result['type'] == 'text':
            # Text results
            col1, col2 = st.columns(2)
            
            # Check if this is an "other" question
            is_other_question = 'other' in question_result['label'].lower()
            
            # Filter out empty responses for "other" questions
            if is_other_question:
                filtered_responses = [resp for resp in analysis['all_responses'] 
                                    if resp.strip() and resp.lower() not in ['', 'nan', 'none', 'nothing']]
            else:
                filtered_responses = analysis['all_responses']
            
            with col1:
                if is_other_question:
                    st.subheader("Other Responses (Non-empty only)")
                else:
                    st.subheader("All Text Responses")
                
                if filtered_responses:
                    # Use collapsible for long lists
                    if len(filtered_responses) > 10:
                        with st.expander(f"View all {len(filtered_responses)} responses", expanded=False):
                            for i, response in enumerate(filtered_responses, 1):
                                st.write(f"**{i}.** {response}")
                        
                        # Show first few responses as preview
                        st.write("**Preview (first 5 responses):**")
                        for i, response in enumerate(filtered_responses[:5], 1):
                            st.write(f"**{i}.** {response}")
                        if len(filtered_responses) > 5:
                            st.write(f"... and {len(filtered_responses) - 5} more (click expand to see all)")
                    else:
                        # For short lists, show directly
                        for i, response in enumerate(filtered_responses, 1):
                            st.write(f"**{i}.** {response}")
                else:
                    st.info("No non-empty responses found.")
            
            with col2:
                st.subheader("Text Analysis")
                st.write(f"**Unique Responses:** {len(set(filtered_responses))}")
                st.write(f"**Total Responses:** {len(filtered_responses)}")
                
                if is_other_question:
                    st.write(f"**Empty Responses Filtered:** {analysis['total_responses'] - len(filtered_responses)}")
                
                # Show most common responses if there are many unique ones
                if len(set(filtered_responses)) > 10 and filtered_responses:
                    response_series = pd.Series(filtered_responses)
                    most_common = response_series.value_counts().head(5)
                    st.write("**Most Common Responses:**")
                    for response, count in most_common.items():
                        st.write(f"• {response}: {count} times")
            
            # Add phone number matching analysis for phone number questions
            if 'phone_number' in question_result['label'].lower() and 'phone' in question_result['label'].lower():
                phone_analysis = analyze_phone_number_matching(survey_df)
                if phone_analysis:
                    st.markdown("---")
                    st.subheader("📞 Phone Number Matching Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Matching Numbers", phone_analysis['matches'])
                    with col2:
                        st.metric("Total Comparisons", phone_analysis['total_comparisons'])
                    with col3:
                        st.metric("Match Rate", f"{phone_analysis['match_rate']:.1f}%")
        
        # Add separator between questions
        st.markdown("---")
    
    # Clear progress bar
    progress_bar.empty()
    
    # Footer
    st.markdown("---")
    st.markdown("**Analysis completed!** 📈")

def main():
    """Main function with page selection"""
    # Page selection in sidebar
    st.sidebar.title("📊 Survey Analysis")
    
    page = st.sidebar.selectbox(
        "Select Survey to Analyze",
        ["Caregiver Survey", "Facility Observations", "Health Worker Survey"]
    )
    
    # Main header
    st.markdown('<h1 class="main-header">📊 SurveyCTO Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Route to appropriate survey analysis
    if page == "Caregiver Survey":
        st.markdown('<h2 style="color: #2c3e50; text-align: center;">👥 Caregiver Survey Analysis</h2>', unsafe_allow_html=True)
        analyze_survey('caregiver', 'Caregiver Survey')
    elif page == "Facility Observations":
        st.markdown('<h2 style="color: #2c3e50; text-align: center;">🏥 Facility Observations Analysis</h2>', unsafe_allow_html=True)
        analyze_survey('facility', 'Facility Observations')
    elif page == "Health Worker Survey":
        st.markdown('<h2 style="color: #2c3e50; text-align: center;">👨‍⚕️ Health Worker Survey Analysis</h2>', unsafe_allow_html=True)
        analyze_survey('health', 'Health Worker Survey')

if __name__ == "__main__":
    main()
