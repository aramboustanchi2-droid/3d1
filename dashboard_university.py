"""
University Knowledge Dashboard
تابلوی نظارتی سیستم یادگیری دانشگاهی

نمایش real-time اطلاعات، آمار و وضعیت سیستم
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Must be first Streamlit command
st.set_page_config(
    page_title="University Knowledge Dashboard",
    page_icon="🎓",
    layout="wide"
)

from cad3d.super_ai.knowledge_database import UniversityKnowledgeDB
from cad3d.super_ai.university_config import UNIVERSITIES
from cad3d.super_ai.agent_security import AgentSecuritySystem

# Initialize
@st.cache_resource
def init_systems():
    db = UniversityKnowledgeDB()
    security = AgentSecuritySystem()
    return db, security

db, security = init_systems()

# Title
st.title("🎓 University Knowledge System Dashboard")
st.markdown("### سامانه نظارت و کنترل یادگیری از دانشگاه‌های برتر دنیا")

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["Overview", "Universities", "Security", "Specializations", "Database"]
)

# ====================
# PAGE: OVERVIEW
# ====================
if page == "Overview":
    st.header("📈 System Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    summary = db.get_all_universities_summary()
    total_docs = sum(u['doc_count'] or 0 for u in summary)
    total_unis = len(summary)
    total_content = sum(u['total_content'] or 0 for u in summary)
    
    with col1:
        st.metric("Total Universities", total_unis)
    with col2:
        st.metric("Total Documents", f"{total_docs:,}")
    with col3:
        st.metric("Total Content", f"{total_content/1e6:.1f}M chars")
    with col4:
        security_summary = db.get_security_summary()
        st.metric("Security Events", security_summary['total_events'])
    
    st.markdown("---")
    
    # Universities Table
    st.subheader("🎓 Universities Status")
    
    if summary:
        df = pd.DataFrame(summary)
        df['focus_areas'] = df['focus_areas'].apply(lambda x: ', '.join(json.loads(x)[:3]) if x else '')
        
        display_df = df[['name', 'country', 'rank', 'doc_count', 'compliance_score', 'focus_areas']]
        display_df.columns = ['University', 'Country', 'Rank', 'Documents', 'Compliance %', 'Focus Areas']
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No data yet. Run agents to collect data.")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Documents by University")
        if summary and total_docs > 0:
            chart_data = pd.DataFrame({
                'University': [u['name'][:20] for u in summary],
                'Documents': [u['doc_count'] or 0 for u in summary]
            })
            fig = px.bar(chart_data, x='University', y='Documents', color='Documents')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Universities by Country")
        if summary:
            country_counts = {}
            for u in summary:
                country = u['country']
                country_counts[country] = country_counts.get(country, 0) + 1
            
            fig = px.pie(
                names=list(country_counts.keys()),
                values=list(country_counts.values()),
                title="Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

# ====================
# PAGE: UNIVERSITIES
# ====================
elif page == "Universities":
    st.header("🎓 Universities Details")
    
    # Select university
    uni_names = {info['name']: key for key, info in UNIVERSITIES.items()}
    selected_name = st.selectbox("Select University:", list(uni_names.keys()))
    selected_key = uni_names[selected_name]
    
    uni_info = UNIVERSITIES[selected_key]
    stats = db.get_university_stats(selected_key)
    
    # Info cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Country:** {uni_info['country']}")
        st.info(f"**Rank:** #{uni_info['rank']}")
    
    with col2:
        if stats.get('documents'):
            st.success(f"**Documents:** {stats['documents']['total']}")
            st.success(f"**Total Content:** {stats['documents']['total_chars']/1e6:.2f}M chars")
    
    with col3:
        if stats.get('sessions'):
            st.warning(f"**Sessions:** {stats['sessions']['sessions']}")
            st.warning(f"**Pages Scraped:** {stats['sessions']['total_pages']}")
    
    # Focus Areas
    st.subheader("🎯 Focus Areas")
    focus_areas = uni_info['focus_areas']
    cols = st.columns(min(5, len(focus_areas)))
    for i, area in enumerate(focus_areas):
        with cols[i % 5]:
            st.button(area, key=f"focus_{i}", disabled=True)
    
    # Resources
    st.subheader("🌐 Resources")
    for resource_key, resource_info in uni_info['resources'].items():
        with st.expander(f"📚 {resource_key.upper()}"):
            st.write(f"**Description:** {resource_info['description']}")
            st.write(f"**URL:** {resource_info['url']}")
            st.write(f"**Type:** {resource_info['type']}")
            st.write(f"**Formats:** {', '.join(resource_info['format'])}")

# ====================
# PAGE: SECURITY
# ====================
elif page == "Security":
    st.header("🔒 Security & Compliance")
    
    security_summary = db.get_security_summary()
    monitoring = security.get_monitoring_report()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Checks", monitoring['statistics']['total_checks'])
    with col2:
        st.metric("Compliant", monitoring['statistics']['compliant'], 
                 delta=monitoring['compliance_rate'])
    with col3:
        st.metric("Warnings", monitoring['statistics']['warnings'])
    with col4:
        st.metric("Violations", monitoring['statistics']['violations'],
                 delta=monitoring['statistics']['blocked'], delta_color="inverse")
    
    # Compliance Rate
    st.subheader("📊 Compliance Rate")
    
    stats = monitoring['statistics']
    if stats['total_checks'] > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['Compliant', 'Warnings', 'Violations', 'Blocked'],
            values=[
                stats['compliant'],
                stats['warnings'],
                stats['violations'],
                stats['blocked']
            ],
            marker=dict(colors=['#28a745', '#ffc107', '#fd7e14', '#dc3545'])
        )])
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Security Events
    st.subheader("📋 Recent Security Events")
    
    recent_logs = monitoring['recent_logs']
    if recent_logs:
        for log in recent_logs[:5]:
            severity = log.get('type', 'info')
            icon = "🔴" if 'violation' in severity else "🟡" if 'warning' in severity else "🟢"
            st.text(f"{icon} [{log['timestamp']}] {log['message']}")
    else:
        st.success("No security events recorded")
    
    # Security Configuration
    with st.expander("⚙️ Security Configuration"):
        config = monitoring['config']
        st.json(config)

# ====================
# PAGE: SPECIALIZATIONS
# ====================
elif page == "Specializations":
    st.header("📚 Specialization Coverage")
    
    coverage = db.get_specialization_coverage()
    
    if coverage:
        # Chart
        fig = px.bar(
            x=list(coverage.keys()),
            y=list(coverage.values()),
            labels={'x': 'Field', 'y': 'Documents'},
            title="Documents by Specialization Field"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.subheader("📊 Detailed Coverage")
        df = pd.DataFrame({
            'Field': list(coverage.keys()),
            'Documents': list(coverage.values())
        })
        df = df.sort_values('Documents', ascending=False)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No specialization data yet.")
    
    # Required specializations
    st.subheader("🎯 Target Specializations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Engineering:**")
        st.markdown("- Civil Engineering")
        st.markdown("- Mechanical Engineering")
        st.markdown("- Electrical Engineering")
        st.markdown("- Computer Engineering")
        st.markdown("- Architecture")
    
    with col2:
        st.markdown("**Management:**")
        st.markdown("- Business Management")
        st.markdown("- Project Management")
        st.markdown("- HR Management")
        st.markdown("- Quality Management")
    
    with col3:
        st.markdown("**Economics:**")
        st.markdown("- Microeconomics")
        st.markdown("- Macroeconomics")
        st.markdown("- Financial Economics")
        st.markdown("- Development Economics")

# ====================
# PAGE: DATABASE
# ====================
elif page == "Database":
    st.header("💾 Database Explorer")
    
    table = st.selectbox(
        "Select Table:",
        ["universities", "documents", "scraping_sessions", "security_events", "specialization_stats"]
    )
    
    # Query
    cursor = db.conn.cursor()
    cursor.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 100")
    
    rows = cursor.fetchall()
    
    if rows:
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in rows])
        
        st.write(f"**Records:** {len(df)}")
        st.dataframe(df, use_container_width=True)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{table}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info(f"No data in {table} table")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**University Knowledge System**

Version: 1.0  
Last Updated: Nov 2025

© 2025 CAD3D Project
""")
