# cord19_analysis_app.py
# Fully Functional CORD-19 Research Metadata Explorer
# Author: [Your Name]
# Date: [Current Date]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="CORD-19 Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TASK 1: DATA LOADING AND EXPLORATION WITH ERROR HANDLING
# =============================================================================

@st.cache_data
def create_sample_data():
    """Create sample CORD-19 metadata if the actual file is not available"""
    np.random.seed(42)
    
    # Sample journals and authors
    journals = [
        'Nature', 'Science', 'The Lancet', 'NEJM', 'Cell', 'PNAS', 'BMJ', 
        'JAMA', 'PLoS ONE', 'Virology', 'Journal of Virology', 'Vaccine',
        'Clinical Infectious Diseases', 'Lancet Infectious Diseases', 
        'Nature Medicine', 'Science Translational Medicine'
    ]
    
    authors_list = [
        'Smith, J.; Johnson, K.; Williams, L.',
        'Brown, M.; Davis, P.; Miller, R.',
        'Wilson, A.; Moore, T.; Taylor, C.',
        'Anderson, D.; Thomas, N.; Jackson, B.',
        'White, S.; Harris, E.; Martin, G.',
        'Thompson, F.; Garcia, H.; Martinez, I.',
        'Robinson, J.; Clark, M.; Rodriguez, N.',
        'Lewis, O.; Lee, P.; Walker, Q.'
    ]
    
    # Generate sample data
    n_papers = 2000
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    data = []
    for i in range(n_papers):
        # Generate random publish date with higher concentration in 2020-2021
        if np.random.random() < 0.6:  # 60% of papers in 2020-2021
            year = np.random.choice([2020, 2021], p=[0.6, 0.4])
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            publish_date = datetime(year, month, day)
        else:
            # Random date in other years
            days_diff = (end_date - start_date).days
            random_days = np.random.randint(0, days_diff)
            publish_date = start_date + timedelta(days=random_days)
        
        # Generate abstract with varying lengths
        abstract_words = np.random.randint(50, 300)
        abstract = " ".join(["word"] * abstract_words)
        
        # Create paper data
        paper = {
            'cord_uid': f'paper_{i:04d}',
            'title': f'COVID-19 Research Paper {i+1}: {np.random.choice(["Epidemiology", "Treatment", "Vaccine", "Diagnosis", "Public Health"])} Study',
            'authors': np.random.choice(authors_list),
            'journal': np.random.choice(journals),
            'publish_time': publish_date.strftime('%Y-%m-%d'),
            'abstract': abstract,
            'url': f'https://example.com/paper_{i}',
            'pmcid': f'PMC{np.random.randint(1000000, 9999999)}' if np.random.random() > 0.3 else '',
            'pubmed_id': f'{np.random.randint(10000000, 99999999)}' if np.random.random() > 0.2 else '',
        }
        data.append(paper)
    
    return pd.DataFrame(data)

@st.cache_data
def load_data(file_path="metadata.csv"):
    """Load CORD-19 metadata with fallback to sample data"""
    try:
        # Try to load the actual dataset
        df = pd.read_csv(file_path)
        st.success(f"Successfully loaded actual dataset: {file_path}")
        return df, "actual"
    except FileNotFoundError:
        st.warning(f"File '{file_path}' not found. Using sample data for demonstration.")
        return create_sample_data(), "sample"
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}. Using sample data.")
        return create_sample_data(), "sample"

# MAIN APPLICATION

# App Header
st.markdown('<h1 class="main-header"> CORD-19 Research Metadata Explorer</h1>', unsafe_allow_html=True)
st.markdown("Exploring COVID-19 research papers in the spirit of Ubuntu: I am because we are")

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
page_selection = st.sidebar.selectbox(
    "Choose Analysis Section:",
    ["Dataset Overview", "Data Analysis", "Interactive Filtering", "Insights & Findings"]
)

# Load data
with st.spinner("Loading dataset..."):
    df, data_source = load_data()

# Display data source information
if data_source == "sample":
    st.info("Demo Mode: Using sample data that mimics CORD-19 structure for demonstration purposes.")
else:
    st.success("Live Data: Using actual CORD-19 metadata file.")

# TASK 1 CONTINUED: DATASET OVERVIEW

if page_selection == "Dataset Overview":
    st.header("Dataset Overview")
    
    # Dataset metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>{df.shape[0]:,}</h3><p>Total Papers</p></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Data Columns</p></div>',
            unsafe_allow_html=True
        )
    
    with col3:
        unique_journals = df['journal'].nunique() if 'journal' in df.columns else 0
        st.markdown(
            f'<div class="metric-card"><h3>{unique_journals:,}</h3><p>Unique Journals</p></div>',
            unsafe_allow_html=True
        )
    
    with col4:
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.markdown(
            f'<div class="metric-card"><h3>{missing_percent:.1f}%</h3><p>Missing Data</p></div>',
            unsafe_allow_html=True
        )
    
    # Raw data preview
    st.subheader("Data Preview")
    if st.checkbox("Show raw data sample"):
        st.dataframe(df.head(10))
    
    # Dataset information
    st.subheader("Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Column Information:")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': [df[col].count() for col in df.columns],
            'Data Type': [str(df[col].dtype) for col in df.columns]
        })
        st.dataframe(info_df)
    
    with col2:
        st.write("Missing Values Analysis:")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': [df[col].isnull().sum() for col in df.columns],
            'Missing %': [df[col].isnull().sum()/len(df)*100 for col in df.columns]
        }).sort_values('Missing %', ascending=False)
        st.dataframe(missing_df)

# TASK 2: DATA CLEANING AND PREPARATION

# Clean and prepare data
@st.cache_data
def clean_data(df):
    """Clean and prepare the dataset"""
    df_clean = df.copy()
    
    # Handle missing values
    if 'abstract' in df_clean.columns:
        df_clean['abstract'] = df_clean['abstract'].fillna("")
    else:
        df_clean['abstract'] = ""
    
    if 'journal' in df_clean.columns:
        df_clean['journal'] = df_clean['journal'].fillna("Unknown Journal")
    
    if 'authors' in df_clean.columns:
        df_clean['authors'] = df_clean['authors'].fillna("Unknown Author")
    
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].fillna("Untitled")
    
    # Handle dates
    if 'publish_time' in df_clean.columns:
        df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors="coerce")
        df_clean['year'] = df_clean['publish_time'].dt.year
        df_clean['month'] = df_clean['publish_time'].dt.month
    else:
        df_clean['year'] = 2020  # Default year
        df_clean['month'] = 6    # Default month
    
    # Calculate abstract word count
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    
    # Remove rows with invalid years
    df_clean = df_clean[df_clean['year'].between(1900, 2030, inclusive='both')]
    
    return df_clean

# Clean the data
df_clean = clean_data(df)

# TASK 3: DATA ANALYSIS AND VISUALIZATIONS

if page_selection == "Data Analysis":
    st.header("Data Analysis & Visualizations")
    
    # Set up matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Publications per Year (Line Chart)
    st.subheader("Publications Timeline")
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    if df_clean['year'].notna().any():
        yearly_counts = df_clean['year'].value_counts().sort_index()
        yearly_counts.plot(kind="line", ax=ax1, marker='o', linewidth=3, markersize=8)
        ax1.set_title("COVID-19 Research Publications Over Time", fontsize=16, fontweight='bold')
        ax1.set_xlabel("Year", fontsize=12)
        ax1.set_ylabel("Number of Publications", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on points
        for x, y in zip(yearly_counts.index, yearly_counts.values):
            ax1.annotate(f'{y:,}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'No valid date data available', 
                ha='center', va='center', transform=ax1.transAxes)
    
    st.pyplot(fig1)
    
    # 2. Top Journals (Bar Chart)
    st.subheader("Top Research Journals")
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    if 'journal' in df_clean.columns and df_clean['journal'].notna().any():
        top_journals = df_clean['journal'].value_counts().head(15)
        bars = top_journals.plot(kind="barh", ax=ax2, color='skyblue', edgecolor='navy')
        ax2.set_title("Top 15 Journals Publishing COVID-19 Research", fontsize=16, fontweight='bold')
        ax2.set_xlabel("Number of Publications", fontsize=12)
        ax2.set_ylabel("Journal", fontsize=12)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(top_journals.items()):
            ax2.text(val + max(top_journals) * 0.01, i, f'{val:,}', 
                    va='center', ha='left', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No journal data available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    st.pyplot(fig2)
    
    # 3. Abstract Word Count Distribution (Histogram)
    st.subheader("Abstract Length Analysis")
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    if df_clean['abstract_word_count'].max() > 0:
        # Filter out zero word counts for better visualization
        word_counts = df_clean[df_clean['abstract_word_count'] > 0]['abstract_word_count']
        
        sns.histplot(word_counts, bins=50, ax=ax3, kde=True, alpha=0.7, color='lightcoral')
        ax3.axvline(word_counts.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {word_counts.mean():.0f} words')
        ax3.axvline(word_counts.median(), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {word_counts.median():.0f} words')
        
        ax3.set_title("Distribution of Abstract Lengths", fontsize=16, fontweight='bold')
        ax3.set_xlabel("Word Count", fontsize=12)
        ax3.set_ylabel("Frequency", fontsize=12)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No abstract data available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    st.pyplot(fig3)
    
    # 4. Scatter Plot: Year vs Abstract Length
    st.subheader("Temporal Trends in Abstract Length")
    
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    if df_clean['year'].notna().any() and df_clean['abstract_word_count'].max() > 0:
        # Sample data for better performance if dataset is large
        plot_data = df_clean.sample(min(2000, len(df_clean))) if len(df_clean) > 2000 else df_clean
        
        sns.scatterplot(x="year", y="abstract_word_count", data=plot_data, 
                       ax=ax4, alpha=0.5, s=30)
        
        # Add trend line
        if len(plot_data) > 1:
            z = np.polyfit(plot_data['year'].dropna(), 
                          plot_data['abstract_word_count'].dropna(), 1)
            p = np.poly1d(z)
            ax4.plot(plot_data['year'].sort_values(), 
                    p(plot_data['year'].sort_values()), 
                    "r--", alpha=0.8, linewidth=2, label='Trend Line')
        
        ax4.set_title("Abstract Length Trends Over Time", fontsize=16, fontweight='bold')
        ax4.set_xlabel("Publication Year", fontsize=12)
        ax4.set_ylabel("Abstract Word Count", fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for scatter plot', 
                ha='center', va='center', transform=ax4.transAxes)
    
    st.pyplot(fig4)
    
    # Additional Analysis - Monthly Distribution
    st.subheader("Monthly Publication Patterns")
    
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    
    if 'month' in df_clean.columns and df_clean['month'].notna().any():
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_counts = df_clean['month'].value_counts().sort_index()
        
        bars = ax5.bar(range(1, 13), [monthly_counts.get(i, 0) for i in range(1, 13)], 
                      color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax5.set_title("Publication Distribution by Month", fontsize=16, fontweight='bold')
        ax5.set_xlabel("Month", fontsize=12)
        ax5.set_ylabel("Number of Publications", fontsize=12)
        ax5.set_xticks(range(1, 13))
        ax5.set_xticklabels(month_names)
        ax5.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig5)

# TASK 4: INTERACTIVE FILTERING

elif page_selection == "Interactive Filtering":
    st.header("Interactive Data Filtering")
    
    # Sidebar filters
    st.sidebar.subheader("Filter Options")
    
    # Year filter
    available_years = sorted([int(y) for y in df_clean['year'].dropna().unique() if pd.notna(y)])
    if available_years:
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=min(available_years),
            max_value=max(available_years),
            value=(min(available_years), max(available_years))
        )
        
        # Filter by year range
        filtered_df = df_clean[
            (df_clean['year'] >= year_range[0]) & 
            (df_clean['year'] <= year_range[1])
        ]
    else:
        filtered_df = df_clean
        st.sidebar.write("No valid years available for filtering")
    
    # Journal filter
    if 'journal' in df_clean.columns:
        top_journals_list = df_clean['journal'].value_counts().head(20).index.tolist()
        selected_journals = st.sidebar.multiselect(
            "Select Journals",
            options=["All"] + top_journals_list,
            default=["All"]
        )
        
        if "All" not in selected_journals:
            filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
    
    # Abstract length filter
    if filtered_df['abstract_word_count'].max() > 0:
        word_count_range = st.sidebar.slider(
            "Abstract Word Count Range",
            min_value=0,
            max_value=int(filtered_df['abstract_word_count'].max()),
            value=(0, int(filtered_df['abstract_word_count'].quantile(0.95)))
        )
        
        filtered_df = filtered_df[
            (filtered_df['abstract_word_count'] >= word_count_range[0]) & 
            (filtered_df['abstract_word_count'] <= word_count_range[1])
        ]
    
    # Display filtered results
    st.subheader(f"Filtered Results: {len(filtered_df):,} papers")
    
    if len(filtered_df) > 0:
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Papers", f"{len(filtered_df):,}")
        
        with col2:
            avg_words = filtered_df['abstract_word_count'].mean()
            st.metric("Avg Abstract Length", f"{avg_words:.0f} words")
        
        with col3:
            unique_journals = filtered_df['journal'].nunique() if 'journal' in filtered_df.columns else 0
            st.metric("Unique Journals", unique_journals)
        
        # Display sample of filtered data
        st.subheader("Sample of Filtered Papers")
        display_columns = []
        for col in ['title', 'authors', 'journal', 'year', 'abstract_word_count']:
            if col in filtered_df.columns:
                display_columns.append(col)
        
        if display_columns:
            sample_size = min(20, len(filtered_df))
            st.dataframe(filtered_df[display_columns].head(sample_size))
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_cord19_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No papers match the current filter criteria. Please adjust your filters.")

# INSIGHTS AND FINDINGS

elif page_selection == "Insights & Findings":
    st.header("Key Insights & Findings")
    
    # Calculate key statistics
    total_papers = len(df_clean)
    if 'year' in df_clean.columns and df_clean['year'].notna().any():
        peak_year = df_clean['year'].value_counts().index[0] if len(df_clean['year'].value_counts()) > 0 else "N/A"
        peak_count = df_clean['year'].value_counts().iloc[0] if len(df_clean['year'].value_counts()) > 0 else 0
    else:
        peak_year, peak_count = "N/A", 0
    
    avg_abstract_length = df_clean['abstract_word_count'].mean()
    top_journal = df_clean['journal'].value_counts().index[0] if 'journal' in df_clean.columns and len(df_clean['journal'].value_counts()) > 0 else "N/A"
    
    # Display insights in organized sections
    st.markdown("""
    <div class="insight-box">
    <h3>🔍 Research Publication Patterns</h3>
    </div>
    """, unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.write("**📈 Publication Trends:**")
        if peak_year != "N/A":
            st.write(f"- Peak publication year: **{peak_year}** with **{peak_count:,}** papers")
            st.write(f"- This reflects the urgent research response to the COVID-19 pandemic")
        else:
            st.write("- Publication trend data not available")
        
        st.write(f"- Total papers analyzed: **{total_papers:,}**")
        
        if 'journal' in df_clean.columns:
            unique_journals = df_clean['journal'].nunique()
            st.write(f"- Research published across **{unique_journals:,}** different journals")
    
    with insights_col2:
        st.write("**📊 Content Analysis:**")
        st.write(f"- Average abstract length: **{avg_abstract_length:.0f}** words")
        
        if top_journal != "N/A":
            st.write(f"- Most prolific journal: **{top_journal}**")
        
        # Calculate abstract length insights
        if df_clean['abstract_word_count'].max() > 0:
            short_abstracts = (df_clean['abstract_word_count'] < 100).sum()
            medium_abstracts = ((df_clean['abstract_word_count'] >= 100) & 
                              (df_clean['abstract_word_count'] < 200)).sum()
            long_abstracts = (df_clean['abstract_word_count'] >= 200).sum()
            
            st.write("**Abstract Length Distribution:**")
            st.write(f"- Short (<100 words): **{short_abstracts:,}** papers")
            st.write(f"- Medium (100-200 words): **{medium_abstracts:,}** papers") 
            st.write(f"- Long (>200 words): **{long_abstracts:,}** papers")
    
    # Research Impact Section
    st.markdown("""
    <div class="insight-box">
    <h3>Global Research Collaboration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    **Key Observations:**
    
    - **Research Mobilization**: The dramatic increase in COVID-19 research publications demonstrates the global scientific community's rapid response to the pandemic.
    
    - **Interdisciplinary Approach**: Papers span multiple journals and disciplines, showing collaborative efforts across medical, biological, and public health fields.
    
    - **Knowledge Sharing**: The volume and diversity of research reflects the principle of Ubuntu - "I am because we are" - as researchers worldwide shared knowledge to combat a common threat.
    
    - **Publication Patterns**: Peak publication periods correlate with major pandemic milestones, showing how research responds to urgent global needs.
    """)
    
    # Technical Insights
    st.markdown("""
    <div class="insight-box">
    <h3>🔬 Technical Analysis Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Data quality metrics
    data_quality_col1, data_quality_col2 = st.columns(2)
    
    with data_quality_col1:
        st.write("**Data Quality Metrics:**")
        missing_title = df['title'].isnull().sum() if 'title' in df.columns else 0
        missing_abstract = df['abstract'].isnull().sum() if 'abstract' in df.columns else 0
        missing_journal = df['journal'].isnull().sum() if 'journal' in df.columns else 0
        
        st.write(f"- Missing titles: **{missing_title:,}** ({missing_title/len(df)*100:.1f}%)")
        st.write(f"- Missing abstracts: **{missing_abstract:,}** ({missing_abstract/len(df)*100:.1f}%)")
        st.write(f"- Missing journal info: **{missing_journal:,}** ({missing_journal/len(df)*100:.1f}%)")
    
    with data_quality_col2:
        st.write("**Dataset Completeness:**")
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.write(f"- Overall completeness: **{completeness:.1f}%**")
        st.write(f"- Suitable for comprehensive analysis: **{'Yes' if completeness > 70 else 'Partial'}**")
        
        if data_source == "sample":
            st.write("- **Note**: Analysis based on representative sample data")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>CORD-19 Research Metadata Explorer | Built with Streamlit</p>
    <p><em>In the spirit of Ubuntu: I am because we are</em></p>
</div>
""", unsafe_allow_html=True)