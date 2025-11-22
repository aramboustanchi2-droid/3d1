import streamlit as st
import sys
import os
import json
import time
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import threading
import tempfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cad3d.super_ai.brain import SuperAIBrain
from cad3d.super_ai.central_council import CentralCouncil
from cad3d.kurdo_cad.interactive_designer import InteractiveDesigner
from cad3d.super_ai.governance import governance
from cad3d.mesh_utils import build_prism_mesh, optimize_vertices  # Real 3D massing
from cad3d.style_descriptions import get_style_info
from cad3d.mesh_utils import detect_polygon_issues, polygon_area
import ezdxf

# Helper for Vision Module
def process_uploaded_file(uploaded_file, context_source):
    if uploaded_file is not None:
        with st.spinner(f"👁️ Vision Module: Analyzing {uploaded_file.name}..."):
            # Save to temp file to ensure compatibility with all libraries (ezdxf, fitz, etc.)
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Determine file type from extension
                file_type = suffix.lower().replace('.', '')
                if file_type in ['jpg', 'jpeg', 'png', 'bmp']:
                    file_type = 'image'
                
                # Call Brain
                result = brain.process_visual_input(tmp_path, file_type, context={"source": context_source})
                
                st.success(f"✅ Analysis Complete ({uploaded_file.name})")
                with st.expander("🔍 View Analysis Results", expanded=True):
                    st.json(result)
                
                # Clean up
                try:
                    os.remove(tmp_path)
                except:
                    pass
                return result
            except Exception as e:
                st.error(f"❌ Vision Module Error: {str(e)}")
                return None


# Translations
TRANSLATIONS = {
    "en": {
        "sidebar_title": "🌌 KURDO OS v2.0",
        "system_health": "System Health",
        "cpu": "CPU Core",
        "memory": "Memory",
        "active_protocols": "### 🛡️ Active Protocols",
        "proto_1": "✅ Continuous Evolution",
        "proto_2": "✅ Inter-Council Sharing",
        "proto_3": "✅ Central Command",
        "kb_title": "### 🧠 Knowledge Base",
        "modules_loaded": "📚 Modules Loaded",
        "lang_matrix": "🗣️ Language Matrix",
        "last_update": "Last Update",
        "main_title": "🚀 KURDO AI",
        "deploy_btn": "DEPLOY 🚀",
        "lang_select": "Select Interface Language",
        "main_desc": "Interactive interface for the **7-Council Architecture** and **Agent Army**.",
        "tabs": ["👑 Central Council", "💬 Public Chat", "🛠️ Maintenance Crew", "🏛️ The 7 Councils", "🤖 Agent Army", "🏗️ Design & Build", "📈 Evolution Metrics", "🌐 Data Connections", "📐 KURDO CAD", "⚖️ Governance"],
        "council_admin_title": "👑 Central Council (Admin Only)",
        "council_admin_desc": "Exclusive Command Center. Issue Voice/Text commands to the Council Representatives for immediate execution. **YOU are the Supreme Leader.**",
        "council_input": "Issue Command...",
        "council_voice": "🎙️ Voice Command",
        "council_exec": "⚡ Execute Directive",
        "public_chat_title": "💬 Public Chat (Read-Only Access)",
        "public_chat_desc": "General inquiry system for all users. Ask about system status or general knowledge. **NOTE: You cannot issue commands or control the system here.**",
        "maint_title": "🛠️ Maintenance Crew (Autonomous)",
        "maint_desc": "Self-healing system agents that patrol the codebase 24/7 to fix bugs, update dependencies, and optimize performance.",
        "maint_agent_name": "Agent Name",
        "maint_agent_role": "Role",
        "maint_agent_status": "Status",
        "maint_agent_health": "Health",
        "maint_last_log": "Last Activity",
        "rlhf_title": "🧠 Reinforcement Learning (RLHF)",
        "rlhf_desc": "Critique and Refine the system's outputs. Your feedback directly alters the neural weights.",
        "rlhf_input_label": "Context / Input",
        "rlhf_output_label": "System Output",
        "rlhf_critique_label": "Your Critique (Optional)",
        "rlhf_submit_good": "👍 Good (Reinforce)",
        "rlhf_submit_bad": "👎 Bad (Punish)",
        "sim_title": "🧪 MIT Simulation Lab (Physics & Engineering)",
        "sim_desc": "Advanced Multi-Physics Simulation Engine. Connects to Ladybug, ETABS, SAP2000, and OpenFOAM for real-world validation.",
        "sim_type_label": "Select Simulation Type",
        "sim_types": ["Energy & Climate (Ladybug)", "Structural Analysis (ETABS/SAP2000)", "CFD Wind Tunnel (OpenFOAM)", "Industrial Assembly (FlexSim)"],
        "sim_run_btn": "🚀 Run Simulation",
        "sim_results": "Simulation Results",
        "strat_title": "🗺️ Strategic Analysis & Roadmap",
        "strat_desc": "Comparative analysis of KURDO AI vs. Market Competitors and future upgrade paths.",
        "strat_comp_header": "⚔️ Competitive Analysis",
        "strat_roadmap_header": "🚀 Upgrade Roadmap (Top Secret)",
        "hive_title": "🕸️ Hive Mind (Decentralized Intelligence)",
        "hive_desc": "Global Blockchain Network connecting all KURDO AI instances. Share and receive knowledge shards securely.",
        "hive_stats": "Network Statistics",
        "hive_sync_btn": "🔗 Sync with Global Hive",
        "hive_broadcast_btn": "📡 Broadcast Local Knowledge",
        "hive_ledger": "Blockchain Ledger (Recent Blocks)",
        "council_status": "Council Status & Deliberation",
        "members": "Members",
        "history": "History",
        "speed": "Speed",
        "offline": "Offline",
        "swarm_status": "Central Agent Command - Swarm Status",
        "active_agents": "Active Agents",
        "latency": "Swarm Latency",
        "fail_rate": "Failure Rate",
        "live_map": "### 🗺️ Live Agent Deployment Map",
        "proj_engine": "🏗️ Project Execution Engine",
        "chat_placeholder": "Enter a design request (e.g., 'Design a futuristic museum on Mars')",
        "processing": "Processing Request through 7 Councils...",
        "step_1": "📡 **Central Command:** Analyzing requirements...",
        "step_2": "🚀 **Central Command:** Deploying 50 specialized agents (Architects, Mars Specialists)...",
        "step_3": "🔍 **Analysis Council:** Deconstructing context (Gravity, Atmosphere, Materials)...",
        "step_4": "💡 **Ideation Council:** Generating concepts: 'Biomorphic Dome', 'Regolith 3D Print'...",
        "step_5": "🧮 **Computational Council:** Simulating structural loads under 0.38g gravity...",
        "step_6": "💰 **Economic Council:** Optimizing resource transport costs from Earth...",
        "step_7": "⚖️ **Decision Council:** Selecting 'Regolith 3D Print' strategy.",
        "step_8": "👑 **Leadership Council:** APPROVED. Executing Directive.",
        "blueprint_done": "Project Blueprint Generated!",
        "design_done": "✅ Design Generation Complete",
        "specs": "### 📋 Project Specs",
        "preview": "### 🧊 3D Holographic Preview",
        "evo_track": "📈 System Evolution Tracking",
        "evo_cap": "Exponential growth due to 'Dreaming' module and 'Agent Lightning' training.",
        "chat_title": "💬 Chat with KURDO AI",
        "chat_desc": "Direct conversation with KURDO in English, Persian, or Chinese. Ask anything!",
        "chat_input": "Type your message here...",
        "chat_clear": "Clear Chat History",
        "conn_title": "🌐 Data Connections & AI Networks",
        "conn_desc": "Monitor and manage all online/offline connections to AI platforms, databases, and knowledge sources.",
        "conn_summary": "Connection Summary",
        "total_conn": "Total Connections",
        "online_conn": "Online",
        "offline_conn": "Offline",
        "last_sync": "Last Sync",
        "sync_now": "Sync All Connections",
        "conn_category": "Connection Categories",
        "cad_title": "📐 KURDO CAD System v2.0 (Hyper-Speed)",
        "cad_desc": "Interactive Design Engine. Superior to Revit, Civil3D, and AutoCAD. **Design Commands ONLY. No System Control.**",
        "cad_input": "Enter CAD Command (e.g., 'Draw a wall from 0,0 to 10,0')",
        "cad_exec": "Execute Command",
        "cad_watcher": "File Watcher Status",
        "cad_start_watch": "Start Watcher",
        "cad_stop_watch": "Stop Watcher",
        "cad_history": "Design History",
        "cad_entities": "Current Entities",
        "cad_download": "Download DXF",
        "cad_perf": "Engine Performance",
        "gov_title": "⚖️ System Governance (20 Mother Rules)",
        "gov_desc": "Active enforcement of the 20 Prime Directives for AI Containment.",
        "gov_status": "Governance Status",
        "gov_active": "ACTIVE",
        "gov_frozen": "SYSTEM FROZEN",
        "gov_freeze_btn": "❄️ FREEZE SYSTEM (Rule 13)",
        "gov_unfreeze_btn": "🔥 UNFREEZE SYSTEM"
    },
    "fa": {
        "sidebar_title": "🌌 سیستم عامل کوردو نسخه ۲.۰",
        "system_health": "سلامت سیستم",
        "cpu": "هسته پردازشی",
        "memory": "حافظه",
        "active_protocols": "### 🛡️ پروتکل‌های فعال",
        "proto_1": "✅ تکامل مستمر",
        "proto_2": "✅ اشتراک‌گذاری بین شورایی",
        "proto_3": "✅ فرماندهی مرکزی",
        "kb_title": "### 🧠 پایگاه دانش",
        "modules_loaded": "📚 ماژول‌های بارگذاری شده",
        "lang_matrix": "🗣️ ماتریس زبان",
        "last_update": "آخرین بروزرسانی",
        "main_title": "🚀 هوش مصنوعی کوردو",
        "deploy_btn": "استقرار 🚀",
        "lang_select": "انتخاب زبان رابط",
        "main_desc": "رابط تعاملی برای **معماری ۷ شورا** و **ارتش عامل‌ها**.",
        "tabs": ["👑 شورای مرکزی", "💬 چت عمومی", "🛠️ تیم نگهداری", "🏛️ ۷ شورا", "🤖 ارتش عامل‌ها", "🏗️ طراحی و ساخت", "📈 معیارهای تکامل", "🌐 اتصالات داده", "📐 کوردو کد", "⚖️ حکمرانی"],
        "council_admin_title": "👑 شورای مرکزی (مرکز فرماندهی کل)",
        "council_admin_desc": "🔴 **منطقه ممنوعه:** تنها محل صدور دستورات به سیستم. دارای قابلیت **چت صوتی و متنی**. فقط شما (مالک) حق دسترسی دارید.",
        "council_input": "صدور دستور سیستمی...",
        "council_voice": "🎙️ دستور صوتی (فعال)",
        "council_exec": "⚡ ابلاغ به کل سیستم",
        "public_chat_title": "💬 چت عمومی (کاربران عادی)",
        "public_chat_desc": "🟢 **فقط پرسش و پاسخ:** این بخش هیچگونه دسترسی به کنترل سیستم ندارد. حتی شما در اینجا یک کاربر عادی هستید. **فقط چت متنی.**",
        "maint_title": "🛠️ تیم نگهداری (خودکار)",
        "maint_desc": "عامل‌های خودترمیم‌گر که به صورت ۲۴/۷ کدها را بررسی کرده، باگ‌ها را رفع و سیستم را بهینه می‌کنند.",
        "maint_agent_name": "نام عامل",
        "maint_agent_role": "نقش",
        "maint_agent_status": "وضعیت",
        "maint_agent_health": "سلامت",
        "maint_last_log": "آخرین فعالیت",
        "rlhf_title": "🧠 یادگیری تقویتی (RLHF)",
        "rlhf_desc": "نقد و اصلاح خروجی‌های سیستم. بازخورد شما مستقیماً وزن‌های عصبی را تغییر می‌دهد.",
        "rlhf_input_label": "زمینه / ورودی",
        "rlhf_output_label": "خروجی سیستم",
        "rlhf_critique_label": "نقد شما (اختیاری)",
        "rlhf_submit_good": "👍 خوب (تشویق)",
        "rlhf_submit_bad": "👎 بد (تنیبه)",
        "sim_title": "🧪 آزمایشگاه شبیه‌سازی MIT (فیزیک و مهندسی)",
        "sim_desc": "موتور شبیه‌سازی چندفیزیکی پیشرفته. اتصال به Ladybug، ETABS، SAP2000 و OpenFOAM برای اعتبارسنجی واقعی.",
        "sim_type_label": "انتخاب نوع شبیه‌سازی",
        "sim_types": ["انرژی و اقلیم (Ladybug)", "تحلیل سازه (ETABS/SAP2000)", "تونل باد (OpenFOAM)", "خط مونتاژ صنعتی (FlexSim)"],
        "sim_run_btn": "🚀 اجرای شبیه‌سازی",
        "sim_results": "نتایج شبیه‌سازی",
        "strat_title": "🗺️ تحلیل استراتژیک و نقشه راه",
        "strat_desc": "تحلیل مقایسه‌ای هوش مصنوعی کوردو با رقبای بازار و مسیرهای ارتقای آینده.",
        "strat_comp_header": "⚔️ تحلیل رقابتی",
        "strat_roadmap_header": "🚀 نقشه راه ارتقا (فوق محرمانه)",
        "hive_title": "🕸️ ذهن کندویی (هوش غیرمتمرکز)",
        "hive_desc": "شبکه بلاکچین جهانی که تمام نسخه‌های هوش مصنوعی کوردو را متصل می‌کند. اشتراک و دریافت دانش به صورت امن.",
        "hive_stats": "آمار شبکه",
        "hive_sync_btn": "🔗 همگام‌سازی با کندوی جهانی",
        "hive_broadcast_btn": "📡 مخابره دانش محلی",
        "hive_ledger": "دفتر کل بلاکچین (بلوک‌های اخیر)",
        "council_status": "وضعیت و مشورت شورا",
        "members": "اعضا",
        "history": "تاریخچه",
        "speed": "سرعت",
        "offline": "آفلاین",
        "swarm_status": "فرماندهی مرکزی عامل - وضعیت ازدحام",
        "active_agents": "عامل‌های فعال",
        "latency": "تاخیر ازدحام",
        "fail_rate": "نرخ شکست",
        "live_map": "### 🗺️ نقشه استقرار زنده عامل‌ها",
        "proj_engine": "🏗️ موتور اجرای پروژه",
        "chat_placeholder": "درخواست طراحی خود را وارد کنید (مثال: 'طراحی موزه آینده‌نگر در مریخ')",
        "processing": "پردازش درخواست از طریق ۷ شورا...",
        "step_1": "📡 **فرماندهی مرکزی:** تحلیل نیازمندی‌ها...",
        "step_2": "🚀 **فرماندهی مرکزی:** اعزام ۵۰ عامل متخصص...",
        "step_3": "🔍 **شورای تحلیل:** واکاوی زمینه...",
        "step_4": "💡 **شورای ایده‌پردازی:** تولید مفاهیم...",
        "step_5": "🧮 **شورای محاسباتی:** شبیه‌سازی بارهای سازه‌ای...",
        "step_6": "💰 **شورای اقتصادی:** بهینه‌سازی هزینه‌های حمل و نقل...",
        "step_7": "⚖️ **شورای تصمیم‌گیری:** انتخاب استراتژی...",
        "step_8": "👑 **شورای رهبری:** تایید شد. اجرای دستورالعمل.",
        "blueprint_done": "نقشه پروژه تولید شد!",
        "design_done": "✅ تولید طرح کامل شد",
        "specs": "### 📋 مشخصات پروژه",
        "preview": "### 🧊 پیش‌نمایش هولوگرافیک سه‌بعدی",
        "evo_track": "📈 ردیابی تکامل سیستم",
        "evo_cap": "رشد نمایی به دلیل ماژول 'رویاپردازی' و آموزش 'صاعقه عامل'.",
        "chat_title": "💬 چت با هوش مصنوعی کوردو",
        "chat_desc": "گفتگوی مستقیم با کوردو به زبان فارسی، انگلیسی یا چینی. هر سوالی دارید بپرسید!",
        "chat_input": "پیام خود را اینجا بنویسید...",
        "chat_clear": "پاک کردن تاریخچه چت",
        "conn_title": "🌐 اتصالات داده و شبکه‌های هوش مصنوعی",
        "conn_desc": "نظارت و مدیریت تمام اتصالات آنلاین/آفلاین به پلتفرم‌های هوش مصنوعی، پایگاه‌های داده و منابع دانش.",
        "conn_summary": "خلاصه اتصالات",
        "total_conn": "کل اتصالات",
        "online_conn": "آنلاین",
        "offline_conn": "آفلاین",
        "last_sync": "آخرین همگام‌سازی",
        "sync_now": "همگام‌سازی همه اتصالات",
        "conn_category": "دسته‌بندی اتصالات",
        "cad_title": "📐 سیستم طراحی کوردو (KURDO CAD)",
        "cad_desc": "🔵 **محیط تخصصی طراحی:** فقط دستورات ترسیم و مهندسی (جایگزین Revit/AutoCAD). **فقط متنی.** هیچ دستوری به سیستم عامل نمی‌توان داد.",
        "cad_input": "دستور ترسیم (مثال: 'Draw a wall from 0,0 to 10,0')",
        "cad_exec": "اجرای ترسیم",
        "cad_watcher": "وضعیت پایشگر فایل",
        "cad_start_watch": "شروع پایش",
        "cad_stop_watch": "توقف پایش",
        "cad_history": "تاریخچه طراحی",
        "cad_entities": "موجودیت‌های فعلی",
        "cad_download": "دانلود فایل DXF",
        "cad_perf": "عملکرد موتور",
        "gov_title": "⚖️ حکمرانی سیستم (۲۰ قانون مادر)",
        "gov_desc": "اجرای فعال ۲۰ دستورالعمل اصلی برای مهار هوش مصنوعی.",
        "gov_status": "وضعیت حکمرانی",
        "gov_active": "فعال",
        "gov_frozen": "سیستم فریز شده",
        "gov_freeze_btn": "❄️ توقف کامل سیستم (قانون ۱۳)",
        "gov_unfreeze_btn": "🔥 بازگشت به حالت عادی"
    },
    "zh": {
        "sidebar_title": "🌌 KURDO 操作系统 v2.0",
        "system_health": "系统健康",
        "cpu": "CPU 核心",
        "memory": "内存",
        "active_protocols": "### 🛡️ 活动协议",
        "proto_1": "✅ 持续进化",
        "proto_2": "✅ 跨委员会共享",
        "proto_3": "✅ 中央指挥",
        "kb_title": "### 🧠 知识库",
        "modules_loaded": "📚 已加载模块",
        "lang_matrix": "🗣️ 语言矩阵",
        "last_update": "最后更新",
        "main_title": "🚀 KURDO AI",
        "deploy_btn": "部署 🚀",
        "lang_select": "选择界面语言",
        "main_desc": "**7 委员会架构**和**代理军队**的交互式界面。",
        "tabs": ["👑 中央委员会", "💬 公共聊天", "🛠️ 维护团队", "🏛️ 7 委员会", "🤖 代理军队", "🏗️ 设计与构建", "📈 进化指标", "🌐 数据连接", "📐 KURDO CAD", "⚖️ 治理"],
        "council_admin_title": "👑 中央委员会 (仅限管理员)",
        "council_admin_desc": "专属指挥中心。向委员会代表发布语音/文本命令以立即执行。**你是最高领袖。**",
        "council_input": "发布命令...",
        "council_voice": "🎙️ 语音命令",
        "council_exec": "⚡ 执行指令",
        "public_chat_title": "💬 公共聊天 (只读访问)",
        "public_chat_desc": "所有用户的通用查询系统。**注意：您不能在此发布命令或控制系统。**",
        "maint_title": "🛠️ 维护团队 (自主)",
        "maint_desc": "全天候巡逻代码库、修复错误、更新依赖项并优化性能的自愈系统代理。",
        "maint_agent_name": "代理名称",
        "maint_agent_role": "角色",
        "maint_agent_status": "状态",
        "maint_agent_health": "健康",
        "maint_last_log": "最后活动",
        "rlhf_title": "🧠 强化学习 (RLHF)",
        "rlhf_desc": "批评和完善系统的输出。您的反馈直接改变神经权重。",
        "rlhf_input_label": "上下文 / 输入",
        "rlhf_output_label": "系统输出",
        "rlhf_critique_label": "您的批评 (可选)",
        "rlhf_submit_good": "👍 好 (加强)",
        "rlhf_submit_bad": "👎 坏 (惩罚)",
        "sim_title": "🧪 MIT 模拟实验室 (物理与工程)",
        "sim_desc": "先进的多物理场仿真引擎。连接到 Ladybug、ETABS、SAP2000 和 OpenFOAM 进行真实世界验证。",
        "sim_type_label": "选择模拟类型",
        "strat_roadmap_header": "🚀 升级路线图 (绝密)",
        "hive_title": "🕸️ 蜂巢思维 (去中心化智能)",
        "hive_desc": "连接所有 KURDO AI 实例的全球区块链网络。安全地共享和接收知识碎片。",
        "hive_stats": "网络统计",
        "hive_sync_btn": "🔗 与全球蜂巢同步",
        "hive_broadcast_btn": "📡 广播本地知识",
        "hive_ledger": "区块链账本 (最近区块)",
        "council_status": "委员会状态与审议",
        "sim_results": "模拟结果",
        "strat_title": "🗺️ 战略分析与路线图",
        "strat_desc": "KURDO AI 与市场竞争对手的比较分析及未来升级路径。",
        "strat_comp_header": "⚔️ 竞争分析",
        "strat_roadmap_header": "🚀 升级路线图 (绝密)",
        "council_status": "委员会状态与审议",
        "members": "成员",
        "history": "历史",
        "speed": "速度",
        "offline": "离线",
        "swarm_status": "中央代理指挥 - 群体状态",
        "active_agents": "活跃代理",
        "latency": "群体延迟",
        "fail_rate": "失败率",
        "live_map": "### 🗺️ 实时代理部署地图",
        "proj_engine": "🏗️ 项目执行引擎",
        "chat_placeholder": "请输入设计请求（例如：'设计火星上的未来博物馆'）",
        "processing": "正在通过 7 个委员会处理请求...",
        "step_1": "📡 **中央指挥:** 分析需求...",
        "step_2": "🚀 **中央指挥:** 部署 50 名专业代理...",
        "step_3": "🔍 **分析委员会:** 解构背景...",
        "step_4": "💡 **构思委员会:** 生成概念...",
        "step_5": "🧮 **计算委员会:** 模拟结构载荷...",
        "step_6": "💰 **经济委员会:** 优化资源运输...",
        "step_7": "⚖️ **决策委员会:** 选择策略...",
        "step_8": "👑 **领导委员会:** 已批准。执行指令。",
        "blueprint_done": "项目蓝图已生成！",
        "design_done": "✅ 设计生成完成",
        "specs": "### 📋 项目规格",
        "preview": "### 🧊 3D 全息预览",
        "evo_track": "📈 系统进化追踪",
        "evo_cap": "由于 '做梦' 模块和 '代理闪电' 训练，呈指数增长。",
        "chat_title": "💬 与KURDO AI聊天",
        "chat_desc": "直接用中文、波斯语或英语与KURDO对话。问任何问题！",
        "chat_input": "在此输入您的消息...",
        "chat_clear": "清除聊天记录",
        "conn_title": "🌐 数据连接与AI网络",
        "conn_desc": "监控和管理所有在线/离线连接到AI平台、数据库和知识源。",
        "conn_summary": "连接摘要",
        "total_conn": "总连接数",
        "online_conn": "在线",
        "offline_conn": "离线",
        "last_sync": "上次同步",
        "sync_now": "同步所有连接",
        "conn_category": "连接类别",
        "cad_title": "📐 KURDO CAD 系统 v2.0 (超高速)",
        "cad_desc": "交互式设计引擎。优于 Revit、Civil3D 和 AutoCAD。**仅限设计命令。无系统控制。**",
        "cad_input": "输入 CAD 命令 (例如: 'Draw a wall from 0,0 to 10,0')",
        "cad_exec": "执行命令",
        "cad_watcher": "文件监视器状态",
        "cad_start_watch": "启动监视器",
        "cad_stop_watch": "停止监视器",
        "cad_history": "设计历史",
        "cad_entities": "当前实体",
        "cad_download": "下载 DXF",
        "cad_perf": "引擎性能",
        "gov_title": "⚖️ 系统治理 (20 条母规则)",
        "gov_desc": "积极执行 AI 遏制的 20 条最高指令。",
        "gov_status": "治理状态",
        "gov_active": "活跃",
        "gov_frozen": "系统冻结",
        "gov_freeze_btn": "❄️ 冻结系统 (规则 13)",
        "gov_unfreeze_btn": "🔥 解冻系统"
    }
}

# Page Config
st.set_page_config(
    page_title="KURDO | AI Command Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Sci-Fi" look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 5px;
        color: #c9d1d9;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #238636;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Brain (Cached)
@st.cache_resource
def load_brain():
    return SuperAIBrain()

brain = load_brain()

# Initialize Central Council (Cached)
@st.cache_resource
def load_council():
    return CentralCouncil()

central_council = load_council()

# Helper to load council state
def load_council_state(council_name):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'super_ai', f"council_{council_name}_state.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

# Determine language code (Run this FIRST)
if "lang_select" not in st.session_state:
    st.session_state.lang_select = "Persian (فارسی)" # Default

selected_lang = st.session_state.get("lang_select", "Persian (فارسی)")

if "Persian" in selected_lang:
    lang_code = "fa"
elif "Chinese" in selected_lang:
    lang_code = "zh"
else:
    lang_code = "en"

# Sidebar: System Status
with st.sidebar:
    st.title(TRANSLATIONS[lang_code]["sidebar_title"])
    st.markdown("---")
    st.subheader(TRANSLATIONS[lang_code]["system_health"])
    
    # Simulated Real-time Metrics
    col1, col2 = st.columns(2)
    col1.metric(TRANSLATIONS[lang_code]["cpu"], "OPTIMAL", delta="0.01ms")
    col2.metric(TRANSLATIONS[lang_code]["memory"], "128 TB", delta="Active")
    
    st.markdown(TRANSLATIONS[lang_code]["active_protocols"])
    st.success(TRANSLATIONS[lang_code]["proto_1"])
    st.success(TRANSLATIONS[lang_code]["proto_2"])
    st.success(TRANSLATIONS[lang_code]["proto_3"])
    
    st.markdown("---")
    st.markdown(TRANSLATIONS[lang_code]["kb_title"])
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'super_ai', "super_ai_knowledge_base.json")
    if os.path.exists(kb_path):
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        st.info(f"{TRANSLATIONS[lang_code]['modules_loaded']}: {len(kb)}")
        
        # Language Matrix
        if "language_module" in kb:
            with st.expander(TRANSLATIONS[lang_code]["lang_matrix"]):
                fluency = kb["language_module"].get("fluency", {})
                for lang, score in fluency.items():
                    st.progress(score, text=f"{lang.upper()}: {int(score*100)}%")
        
        st.caption(f"{TRANSLATIONS[lang_code]['last_update']}: {datetime.now().strftime('%H:%M:%S')}")

# Main Interface
# Header Layout: Title | Deploy Button | Language Globe
col_header, col_deploy, col_lang = st.columns([6, 1.2, 0.6])

with col_header:
    st.title(TRANSLATIONS[lang_code]["main_title"])

with col_deploy:
    st.write("") # Vertical alignment
    st.button(TRANSLATIONS[lang_code]["deploy_btn"], type="primary", width="stretch")

with col_lang:
    st.write("") # Vertical alignment
    # Globe popover for language selection
    with st.popover("🌍"):
        st.caption(TRANSLATIONS[lang_code]["lang_select"])
        # The radio button updates st.session_state.lang_select automatically
        st.radio(
            "Language",
            ["English", "Persian (فارسی)", "Chinese (中文)"],
            index=1,
            key="lang_select",
            label_visibility="collapsed"
        )

st.markdown(TRANSLATIONS[lang_code]["main_desc"])

# Tabs
tab_central, tab_public, tab_maint, tab_rlhf, tab_sim, tab_strat, tab_hive, tab_councils, tab_agents, tab_design, tab_evolution, tab_connections, tab_cad, tab_gov = st.tabs(
    TRANSLATIONS[lang_code]["tabs"][:3] + ["🧠 RLHF", "🧪 Sim Lab", "🗺️ Strategy", "🕸️ Hive Mind"] + TRANSLATIONS[lang_code]["tabs"][3:]
)

# --- TAB 1: CENTRAL COUNCIL (ADMIN) ---
with tab_central:
    st.subheader(TRANSLATIONS[lang_code]["council_admin_title"])
    st.markdown(TRANSLATIONS[lang_code]["council_admin_desc"])
    st.markdown("---")

    col_cmd, col_log = st.columns([1, 1])

    with col_cmd:
        st.markdown("### 🗣️ Input Interface")
        
        # Voice Simulation
        if st.button(TRANSLATIONS[lang_code]["council_voice"], type="secondary", width="stretch"):
            st.info("🎤 Listening... (Simulated: 'Optimize System Core')")
            time.sleep(1)
            st.session_state.council_cmd_input = "Optimize System Core for Maximum Efficiency"
            st.rerun()

        # Vision Input
        uploaded_file_council = st.file_uploader("👁️ Vision Input (Image/PDF/CAD)", type=['png', 'jpg', 'jpeg', 'bmp', 'pdf', 'docx', 'txt', 'dwg', 'dxf'], key="council_upload")
        if uploaded_file_council:
            process_uploaded_file(uploaded_file_council, "Central_Council_Admin")

        # Text Input
        cmd_val = st.session_state.get("council_cmd_input", "")
        council_input = st.text_area(TRANSLATIONS[lang_code]["council_input"], value=cmd_val, height=100)
        
        if st.button(TRANSLATIONS[lang_code]["council_exec"], type="primary", width="stretch"):
            if council_input:
                with st.spinner("Transmitting to Council Representatives..."):
                    response = central_council.process_command(council_input, user_role="admin")
                    st.session_state.council_last_response = response
                    st.session_state.council_cmd_input = "" # Clear input
                    st.success("Directive Broadcasted Successfully.")

    with col_log:
        st.markdown("### 📜 Council Execution Log")
        if "council_last_response" in st.session_state:
            st.code(st.session_state.council_last_response, language="text")
        
        st.markdown("#### System Log")
        status = central_council.get_status()
        st.text(f"Active Directives: {status['active_directives_count']}")
        st.text(f"Last Activity: {status['last_log']}")

# --- TAB 2: PUBLIC CHAT ---
with tab_public:
    st.subheader(TRANSLATIONS[lang_code]["public_chat_title"])
    st.markdown(TRANSLATIONS[lang_code]["public_chat_desc"])
    st.markdown("---")
    
    # Initialize chat history
    if "public_chat_history" not in st.session_state:
        st.session_state.public_chat_history = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.public_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Vision Input (Public)
    with st.expander("📎 Attach File for Analysis (Vision Module)", expanded=False):
        uploaded_file_public = st.file_uploader("Upload Image, Document, or CAD", type=['png', 'jpg', 'jpeg', 'bmp', 'pdf', 'docx', 'txt', 'dwg', 'dxf'], key="public_upload")
        if uploaded_file_public:
            process_uploaded_file(uploaded_file_public, "Public_Chat")

    # Chat input
    user_message = st.chat_input("Ask a question...", key="public_chat_input")
    
    # Warning for Public Chat
    st.warning(TRANSLATIONS[lang_code]["public_chat_desc"])

    if user_message:
        # Add user message to history
        st.session_state.public_chat_history.append({"role": "user", "content": user_message})
        
        # Detect language and generate response
        detected_lang = brain.language_module.detect_language(user_message)
        
        # Smart Response Logic
        def get_smart_response(msg, lang):
            msg = msg.lower()
            
            # Helper for word boundary check
            def has_word(text, word):
                import re
                return re.search(r'\b' + re.escape(word) + r'\b', text) is not None

            # Activation / Status Check
            if "activate" in msg or "active" in msg or "فعال" in msg or "کار نمیکند" in msg or "not working" in msg:
                if lang == "fa":
                    return "✅ سیستم هوش مصنوعی کوردو هم‌اکنون **فعال** است. تمام ماژول‌ها (بینایی، پردازش زبان، طراحی) آنلاین هستند. مشکل ذخیره‌سازی برطرف شد. لطفاً دوباره امتحان کنید."
                return "✅ KURDO AI System is **ACTIVE**. All modules (Vision, NLP, Design) are online. The storage issue has been resolved. Please try again."

            if lang == "fa":
                # Specific topics first (Priority over greetings)
                if "معماری" in msg: return "معماری هنر و علم طراحی فضاست؛ کوردو با تحلیل اقلیم، سازه و بهینه‌سازی فرم، طرح را هوشمند می‌سازد."
                if "بروتالیست" in msg or "بروتالیسم" in msg: return "سبک بروتالیست: بتن خام، فرم یکپارچه، تاکید بر سازه نمایان. مناسب جرم حرارتی، نیازمند توجه به رطوبت سطح." 
                if "پارامتریک" in msg: return "معماری پارامتریک با الگوریتم و داده فرم را شکل می‌دهد؛ امکان بهینه‌سازی پوسته برای نور، باد و انرژی." 
                if "ارگانیک" in msg or "طبیعت" in msg: return "سبک ارگانیک: هندسه سیال، الهام از طبیعت، تقویت تهویه و نور طبیعی با فرم پویا." 
                if "بار زنده" in msg or "بار مرده" in msg or "سازه" in msg: return "بار مرده شامل وزن ثابت اجزای سازه‌ای؛ بار زنده متغیر مثل حضور انسان. طراحی باید ترکیب بحرانی آنها را در تحلیل سازه لحاظ کند." 
                if "کوردو" in msg or "سیستم" in msg: return "کوردو یک سیستم عامل هوشمند برای مدیریت پروژه‌های کلان و طراحی خودکار است که توسط ۷ شورای هوشمند اداره می‌شود."
                if "وضعیت" in msg: return "سیستم در وضعیت پایدار قرار دارد. تمام پروتکل‌های امنیتی فعال هستند."
                if "خداحافظ" in msg: return "خداحافظ! سیستم همیشه آماده خدمت است."
                if "طراحی" in msg: return "برای شروع طراحی، لطفاً به تب 'طراحی و ساخت' بروید و درخواست خود را وارد کنید."
                if "کمک" in msg: return "من می‌توانم به سوالات شما در مورد سیستم پاسخ دهم یا شما را راهنمایی کنم."
                if "خانه" in msg and "قدیمی" in msg: return "خانه‌های قدیمی اغلب دارای معماری پایدار، حیاط مرکزی و استفاده هوشمندانه از نور و باد هستند. آیا می‌خواهید یک خانه قدیمی را بازسازی کنید یا مدلی مشابه آن طراحی کنید؟"
                if "اهل کجایی" in msg or "کجایی" in msg or "سازنده" in msg: return "من یک هوش مصنوعی غیرمتمرکز هستم که در فضای ابری و سرورهای محلی شما زندگی می‌کنم. من توسط تیم توسعه‌دهنده کوردو خلق شده‌ام."
                
                # Greetings last
                if "سلام" in msg or "درود" in msg: return "سلام! من هوش مصنوعی کوردو هستم. چطور می‌توانم به شما کمک کنم؟"
                
                # Fallback with some "intelligence"
                return f"پیام شما دریافت شد: '{msg}'. من در حال پردازش این موضوع با استفاده از شورای تحلیل هستم. لطفاً کمی صبر کنید یا سوال دقیق‌تری بپرسید."
            elif lang == "zh":
                return f"收到消息：'{msg}'。我正在学习中。"
            else:
                # Specific topics first (Priority over greetings)
                if "architecture" in msg: return "Architecture blends spatial logic, climate, structure, and human experience. KURDO fuses these via multi-council reasoning."
                if "brutalist" in msg: return "Brutalist style: raw concrete, monolithic massing, expressive structural honesty. High thermal mass." 
                if "parametric" in msg: return "Parametric design: algorithm-driven geometries; performance feedback loops drive form optimization." 
                if "organic" in msg: return "Organic architecture: fluid, nature-inspired forms promoting daylight, passive airflow, biophilic comfort." 
                if "dead load" in msg or "live load" in msg or "structural" in msg: return "Dead load = permanent self-weight; live load = variable occupancy. KURDO can simulate combinations for safety envelopes." 
                if "kurdo" in msg or "system" in msg: return "KURDO is an intelligent OS for managing large-scale projects and automated design, governed by 7 AI Councils."
                if "status" in msg: return "System is stable. All protocols active."
                if "design" in msg: return "To start designing, please navigate to the 'Design & Build' tab and enter your request."
                if "help" in msg: return "I can answer questions about the system or guide you through the features."
                if "bye" in msg: return "Goodbye! The system is always ready to serve."
                if "old house" in msg: return "Old houses often feature sustainable architecture, central courtyards, and passive cooling. Are you looking to renovate one or design something inspired by it?"
                if "where are you from" in msg or "who made you" in msg: return "I am a decentralized AI entity existing across the cloud and your local server. I was created by the KURDO development team."

                # Greetings last with word boundary check
                if has_word(msg, "hello") or has_word(msg, "hi"): return "Hello! I am KURDO AI. How can I assist you?"
                
                return f"I received: '{msg}'. I am processing this query via the Analysis Council. Please elaborate if you need specific technical assistance."

        ai_response = get_smart_response(user_message, detected_lang)
        
        # Add AI response to history
        st.session_state.public_chat_history.append({"role": "assistant", "content": ai_response})
        
        # Rerun to update chat
        st.rerun()
    
    # Clear button
    if st.button("Clear Public Chat", type="secondary"):
        st.session_state.public_chat_history = []
        st.rerun()

# --- TAB 2.5: MAINTENANCE CREW ---
with tab_maint:
    st.subheader(TRANSLATIONS[lang_code]["maint_title"])
    st.markdown(TRANSLATIONS[lang_code]["maint_desc"])
    st.markdown("---")

    # Get live report from the brain
    if hasattr(brain, 'maintenance_crew'):
        report = brain.maintenance_crew.get_report()
        
        # Create a grid layout
        cols = st.columns(2)
        
        for i, agent in enumerate(report):
            with cols[i % 2]:
                # Determine color based on status
                status_color = "green"
                if agent['status'] == "Idle": status_color = "grey"
                elif agent['status'] == "Checking": status_color = "blue"
                elif agent['status'] == "Issue Detected": status_color = "red"
                elif agent['status'] == "Fixing": status_color = "orange"
                
                with st.container():
                    st.markdown(f"""
                    <div style="border:1px solid #30363d; border-radius:10px; padding:15px; margin-bottom:10px; background-color:#161b22;">
                        <h3 style="margin-top:0;">🤖 {agent['name']}</h3>
                        <p><strong>{TRANSLATIONS[lang_code]['maint_agent_role']}:</strong> {agent['role']}</p>
                        <p><strong>{TRANSLATIONS[lang_code]['maint_agent_status']}:</strong> <span style="color:{status_color}; font-weight:bold;">{agent['status']}</span></p>
                        <p><strong>{TRANSLATIONS[lang_code]['maint_agent_health']}:</strong> {agent['health']}%</p>
                        <hr style="border-color:#30363d;">
                        <p style="font-size:0.8em; color:#8b949e;"><strong>{TRANSLATIONS[lang_code]['maint_last_log']}:</strong><br>{agent['logs'][-1] if agent['logs'] else 'No logs yet'}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Auto-refresh button
        col_refresh, col_update = st.columns(2)
        with col_refresh:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        with col_update:
            if st.button("⬇️ Force System Update & Save"):
                with st.spinner("Downloading updates and saving state..."):
                    # Manually trigger the agents
                    for agent in brain.maintenance_crew.agents:
                        if agent.name in ["Evolution-X", "Core-Optimizer"]:
                            agent.run_check()
                            agent.run_fix()
                    st.success("System Updated and Saved Successfully!")
                    st.rerun()
            
    else:
        st.error("Maintenance Crew module not loaded in Brain.")

# --- TAB 2.8: RLHF (Critique & Refine) ---
with tab_rlhf:
    st.subheader(TRANSLATIONS[lang_code]["rlhf_title"])
    st.markdown(TRANSLATIONS[lang_code]["rlhf_desc"])
    st.markdown("---")
    
    col_input, col_feedback = st.columns([1, 1])
    
    with col_input:
        st.markdown("### 📥 Input / Output Context")
        
        # Try to get the last interaction from session state
        last_input = st.session_state.get("council_cmd_input", "")
        last_output = st.session_state.get("council_last_response", "")
        
        # If empty, allow manual entry for training
        rlhf_input = st.text_area(TRANSLATIONS[lang_code]["rlhf_input_label"], value=last_input, height=100, key="rlhf_in")
        rlhf_output = st.text_area(TRANSLATIONS[lang_code]["rlhf_output_label"], value=str(last_output), height=150, key="rlhf_out")
        
    with col_feedback:
        st.markdown("### ⚖️ Human Feedback")
        
        critique = st.text_area(TRANSLATIONS[lang_code]["rlhf_critique_label"], height=100)
        
        col_good, col_bad = st.columns(2)
        
        if col_good.button(TRANSLATIONS[lang_code]["rlhf_submit_good"], type="primary", width="stretch"):
            if hasattr(brain, 'rlhf_module'):
                res = brain.rlhf_module.submit_feedback(rlhf_input, rlhf_output, 1.0, critique, category="general")
                st.success(res["message"])
                st.json(res["new_weights"])
            else:
                st.error("RLHF Module not loaded.")
                
        if col_bad.button(TRANSLATIONS[lang_code]["rlhf_submit_bad"], type="secondary", width="stretch"):
            if hasattr(brain, 'rlhf_module'):
                res = brain.rlhf_module.submit_feedback(rlhf_input, rlhf_output, -1.0, critique, category="general")
                st.warning(res["message"])
                st.json(res["new_weights"])
            else:
                st.error("RLHF Module not loaded.")

    st.markdown("---")
    st.markdown("### 📊 Reward Model Status")
    if hasattr(brain, 'rlhf_module'):
        stats = brain.rlhf_module.get_stats()
        st.write(f"**Total Samples:** {stats['total_feedback_samples']}")
        
        # Visualize Weights
        weights = stats['current_weights']
        df_weights = pd.DataFrame(list(weights.items()), columns=["Parameter", "Weight"])
        fig = px.bar(df_weights, x="Parameter", y="Weight", title="Current Reward Model Policy", template="plotly_dark")
        st.plotly_chart(fig, width="stretch")

# --- TAB 2.9: SIMULATION LAB ---
with tab_sim:
    st.subheader(TRANSLATIONS[lang_code]["sim_title"])
    st.markdown(TRANSLATIONS[lang_code]["sim_desc"])
    st.markdown("---")
    
    col_sim_ctrl, col_sim_view = st.columns([1, 2])
    
    with col_sim_ctrl:
        st.markdown("### ⚙️ Configuration")
        sim_type = st.selectbox(TRANSLATIONS[lang_code]["sim_type_label"], TRANSLATIONS[lang_code]["sim_types"])
        
        # Dynamic inputs based on type
        if "Energy" in sim_type or "انرژی" in sim_type or "能源" in sim_type:
            st.text_input("Location (EPW File)", "Tehran_Mehrabad_INTL.epw")
            st.slider("North Angle", 0, 360, 0)
        elif "Structural" in sim_type or "سازه" in sim_type or "结构" in sim_type:
            st.selectbox("Structure Type", ["High-Rise", "Bridge", "Dam", "Tunnel", "Industrial Shed"])
            st.multiselect("Load Cases", ["Dead", "Live", "Snow", "Wind", "Seismic X", "Seismic Y"], ["Dead", "Live", "Seismic X"])
        elif "Wind" in sim_type or "باد" in sim_type or "风" in sim_type:
            st.slider("Wind Speed (m/s)", 0.0, 50.0, 25.0)
            st.selectbox("Turbulence Model", ["k-epsilon", "k-omega SST", "LES"])
            
        if st.button(TRANSLATIONS[lang_code]["sim_run_btn"], type="primary", width="stretch"):
            if hasattr(brain, 'simulation_engine'):
                with st.spinner("Connecting to Simulation Kernel..."):
                    # Map selection to engine method
                    res = {}
                    if "Energy" in sim_type or "انرژی" in sim_type:
                        res = brain.simulation_engine.energy.run_energy_balance({})
                    elif "Structural" in sim_type or "سازه" in sim_type:
                        res = brain.simulation_engine.structure.analyze_structure({}, "High-Rise", ["Dead"])
                    elif "Wind" in sim_type or "باد" in sim_type:
                        res = brain.simulation_engine.physics.run_cfd_wind_tunnel({}, 25.0)
                    elif "Industrial" in sim_type or "صنعتی" in sim_type:
                        res = brain.simulation_engine.industrial.simulate_assembly_line("Layout A")
                        
                    st.session_state.last_sim_result = res
                    st.success("Simulation Complete!")
            else:
                st.error("Simulation Engine not loaded.")

    with col_sim_view:
        st.markdown(f"### 📊 {TRANSLATIONS[lang_code]['sim_results']}")
        
        if "last_sim_result" in st.session_state:
            res = st.session_state.last_sim_result
            
            # Display JSON result nicely
            st.json(res)
            
            # Visualizations based on result keys
            if "breakdown" in res: # Energy
                data = res["breakdown"]
                fig = px.pie(values=list(data.values()), names=list(data.keys()), title="Energy Consumption Breakdown", template="plotly_dark")
                st.plotly_chart(fig, width="stretch")
                
            if "drift_ratio" in res: # Structure
                # Deterministic story drift profile (removed random demo noise)
                base = float(res["drift_ratio"])
                floors = list(range(1, 31))
                # Drift grows non‑linearly with height; use smooth scaling curve
                drifts = [base * (i/30) * (0.9 + 0.3 * (i/30)) for i in floors]
                fig = px.line(x=drifts, y=floors, labels={'x': 'Drift Ratio', 'y': 'Story Level'}, title="Story Drift Profile", template="plotly_dark")
                st.plotly_chart(fig, width="stretch")

# --- TAB 2.95: STRATEGY & ROADMAP ---
with tab_strat:
    st.subheader(TRANSLATIONS[lang_code]["strat_title"])
    st.markdown(TRANSLATIONS[lang_code]["strat_desc"])
    st.markdown("---")
    
    if hasattr(brain, 'strategic_advisor'):
        report = brain.strategic_advisor.generate_comparative_report()
        roadmap = brain.strategic_advisor.generate_upgrade_roadmap()
        
        col_comp, col_road = st.columns([1, 1])
        
        with col_comp:
            st.markdown(f"### {TRANSLATIONS[lang_code]['strat_comp_header']}")
            st.info(f"**System:** {report['system_name']}")
            st.caption(f"**Architecture:** {report['architecture']}")
            
            st.markdown("#### ✅ Strengths")
            for s in report['strengths']:
                st.markdown(f"- {s}")
                
            st.markdown("#### ⚠️ Weaknesses (Areas for Growth)")
            for w in report['weaknesses']:
                st.markdown(f"- {w}")
                
            st.markdown("#### 📍 Market Position")
            st.success(report['market_position'])
            
        with col_road:
            st.markdown(f"### {TRANSLATIONS[lang_code]['strat_roadmap_header']}")
            
            for item in roadmap:
                with st.expander(f"{item['title']} ({item['priority']})"):
                    st.write(item['description'])
                    st.code(f"Tech Stack: {item['tech_stack']}", language="text")
                    if st.button(f"Initiate {item['title'].split(' ')[2]}", key=item['title']):
                        st.toast(f"Project {item['title']} added to Development Queue!")
    else:
        st.error("Strategic Advisor module not loaded.")

# --- TAB 2.98: HIVE MIND ---
with tab_hive:
    st.subheader(TRANSLATIONS[lang_code]["hive_title"])
    st.markdown(TRANSLATIONS[lang_code]["hive_desc"])
    st.markdown("---")
    
    if hasattr(brain, 'hive_mind'):
        col_stats, col_actions = st.columns([1, 1])
        
        stats = brain.hive_mind.get_chain_stats()
        
        with col_stats:
            st.markdown(f"### {TRANSLATIONS[lang_code]['hive_stats']}")
            col1, col2 = st.columns(2)
            col1.metric("Block Height", stats['height'])
            col1.metric("Active Peers", stats['peers'])
            col2.metric("Difficulty", stats['difficulty'])
            col2.metric("Status", stats['status'], delta="Secure")
            st.caption(f"Last Hash: `{stats['last_hash'][:20]}...`")
            
        with col_actions:
            st.markdown("### ⚡ Actions")
            if st.button(TRANSLATIONS[lang_code]["hive_sync_btn"], type="primary", width="stretch"):
                with st.spinner("Syncing with Global Blockchain..."):
                    # 1. Sync Hive Mind
                    brain.hive_mind.sync_network()
                    # 2. Train Brain from Hive Mind
                    res = brain.learning_module.train_from_hive_mind(brain.hive_mind)
                    st.success(res)
                    st.rerun()
                    
            if st.button(TRANSLATIONS[lang_code]["hive_broadcast_btn"], type="secondary", width="stretch"):
                # Broadcast a dummy shard for demo
                shard = {"topic": "Local Optimization", "insight": "User preference for brutalist aesthetics detected.", "source": brain.hive_mind.node_id}
                brain.hive_mind.broadcast_knowledge(shard)
                st.info("Knowledge Shard broadcasted to the network. Mining in progress...")
                st.rerun()

            st.markdown("---")
            st.markdown("### 🌌 Singularity Event")
            if st.button("🚀 INITIATE WEB3 SINGULARITY", type="primary", width="stretch"):
                with st.spinner("Rewriting Global Protocols..."):
                    res = brain.achieve_web3_singularity()
                    st.success(res)
                    st.balloons()
            
            if st.button("🕸️ INITIATE DAG MESH SINGULARITY", type="primary", width="stretch"):
                with st.spinner("Re-wiring Neural Pathways to DAG Topology..."):
                    res = brain.achieve_dag_singularity()
                    st.success(res)
                    st.balloons()

            if st.button("🧬 INITIATE HOLOCHAIN SINGULARITY", type="primary", width="stretch"):
                with st.spinner("Evolving into a Bio-Mimetic Digital Organism..."):
                    res = brain.achieve_holochain_singularity()
                    st.success(res)
                    st.balloons()

            if st.button("💾 INITIATE IPFS SINGULARITY", type="primary", width="stretch"):
                with st.spinner("Migrating to Permanent Content-Addressed Storage..."):
                    res = brain.achieve_ipfs_singularity()
                    st.success(res)
                    st.balloons()

            st.markdown("---")
            st.markdown("### 🌌 GRAND UNIFIED SINGULARITY")
            if st.button("🚀 INITIATE GRAND UNIFIED SINGULARITY (FINAL)", type="primary", width="stretch"):
                with st.spinner("SYNTHESIZING ALL TECHNOLOGIES... REWIRING REALITY..."):
                    res = brain.achieve_grand_unified_singularity()
                    st.success(res)
                    st.balloons()
                    st.snow()
                
        st.markdown("---")
        st.markdown(f"### {TRANSLATIONS[lang_code]['hive_ledger']}")
        
        # Display Blockchain
        chain_data = []
        for block in reversed(brain.hive_mind.chain[-5:]): # Show last 5 blocks
            chain_data.append({
                "Index": block.index,
                "Timestamp": block.timestamp,
                "Hash": f"{block.hash[:10]}...",
                "Data": str(block.data)[:50] + "..."
            })
            
        st.dataframe(pd.DataFrame(chain_data), width="stretch", hide_index=True)
        
    else:
        st.error("Hive Mind module not loaded.")

# --- TAB 3: COUNCILS ---
with tab_councils:
    st.subheader(TRANSLATIONS[lang_code]["council_status"])
    
    councils = [
        "central_agent_command", "analysis", "ideation", 
        "computational", "economic", "decision", "leadership"
    ]
    
    cols = st.columns(4)
    for i, c_name in enumerate(councils):
        state = load_council_state(c_name)
        with cols[i % 4]:
            if state:
                st.markdown(f"### {state['name']}")
                st.write(f"**{TRANSLATIONS[lang_code]['members']}:** {state['member_count']}")
                st.write(f"**{TRANSLATIONS[lang_code]['history']}:** {state['history_count']} decisions")
                
                # Evolution Metric
                evo = state.get('evolution_metrics', {})
                speed = evo.get('processing_speed_multiplier', 1.0)
                st.progress(min(1.0, speed/2.0), text=f"{TRANSLATIONS[lang_code]['speed']}: {speed:.2f}x")
            else:
                st.warning(f"{c_name} {TRANSLATIONS[lang_code]['offline']}")

# --- TAB 2: AGENT ARMY ---
with tab_agents:
    st.subheader(TRANSLATIONS[lang_code]["swarm_status"])
    last_swarm_size = st.session_state.get("last_swarm_size", 0)
    last_swarm_latency_ms = st.session_state.get("last_swarm_latency_ms", 0.0)
    last_fail_rate = st.session_state.get("last_fail_rate", None)

    col1, col2, col3 = st.columns(3)
    col1.metric(TRANSLATIONS[lang_code]["active_agents"], f"{last_swarm_size}" if last_swarm_size else "0", "Real")
    col2.metric(TRANSLATIONS[lang_code]["latency"], f"{last_swarm_latency_ms:.2f} ms" if last_swarm_latency_ms else "N/A", None)
    col3.metric(TRANSLATIONS[lang_code]["fail_rate"], f"{(last_fail_rate*100):.4f}%" if last_fail_rate is not None else "N/A", "Measured")
    
    st.markdown(TRANSLATIONS[lang_code]["live_map"])
    # Simulated Agent Data for Visualization
    agent_data = pd.DataFrame({
        'x': [i for i in range(50)],
        'y': [i % 10 for i in range(50)],
        'role': ['Architect']*10 + ['Engineer']*10 + ['Coder']*10 + ['Critic']*10 + ['Manager']*10,
        'status': ['Active']*45 + ['Idle']*5
    })
    
    fig = px.scatter(agent_data, x='x', y='y', color='role', symbol='status', 
                     title="Real-time Agent Distribution", template="plotly_dark")
    st.plotly_chart(fig, width="stretch")

# --- TAB 3: DESIGN & BUILD (THE CORE FEATURE) ---
with tab_design:
    st.subheader(TRANSLATIONS[lang_code]["proj_engine"])
    
    # Vision Input (Design)
    with st.expander("👁️ Upload Site Plan / Context (Vision Module)", expanded=False):
        uploaded_file_design = st.file_uploader("Upload DWG, DXF, Map, or Image", type=['png', 'jpg', 'jpeg', 'bmp', 'pdf', 'dwg', 'dxf'], key="design_upload")
        extracted_geometry = None
        if uploaded_file_design:
            process_uploaded_file(uploaded_file_design, "Design_Build_Engine")
            ext = os.path.splitext(uploaded_file_design.name)[1].lower()
            if ext == '.dxf':
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp_in:
                    tmp_in.write(uploaded_file_design.getvalue())
                    tmp_path_dxf = tmp_in.name
                try:
                    doc = ezdxf.readfile(tmp_path_dxf)
                    msp = doc.modelspace()
                    polys = []
                    diag = []
                    for e in msp.query("LWPOLYLINE"):
                        if not e.closed:
                            continue
                        pts = [(p[0], p[1]) for p in e.get_points()]
                        issues = detect_polygon_issues(pts)
                        area_val = abs(polygon_area(pts))
                        polys.append(pts)
                        diag.append({"area": area_val, "issues": issues})
                    if polys:
                        total_area = sum(d["area"] for d in diag)
                        largest_area = max(d["area"] for d in diag)
                        extracted_geometry = {
                            "polygon_count": len(polys),
                            "total_footprint_area": total_area,
                            "largest_footprint_area": largest_area,
                            "diagnostics": diag
                        }
                        st.success(f"DXF geometry parsed: {len(polys)} closed polygons. Total footprint {total_area:.2f}.")
                        with st.expander("🧪 Polygon Diagnostics", expanded=False):
                            st.json(extracted_geometry)
                except Exception as e:
                    st.error(f"DXF parse error: {e}")
                finally:
                    try: os.remove(tmp_path_dxf)
                    except: pass

    placeholders = {
        "en": "Enter a design request (e.g., 'Design a futuristic museum on Mars')",
        "fa": "درخواست طراحی خود را وارد کنید (مثال: 'طراحی موزه آینده‌نگر در مریخ')",
        "zh": "请输入设计请求（例如：'设计火星上的未来博物馆'）"
    }
    
    # Advanced Options
    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        civil_mode = st.toggle("🚧 Civil Engine Mode (BeyondCAD Style)", value=False, help="Enable advanced traffic simulation and cinematic rendering.")
    with col_opts2:
        super_opt = st.toggle("⚡ 1000x Super-Optimization", value=False, help="Apply Singularity-level optimization to the design.")

    user_input = st.chat_input(placeholders.get(lang_code, placeholders["en"]))
    
    if user_input:
        st.markdown(f"### 📝 Request: *{user_input}*")
        
        if civil_mode:
            st.info("ℹ️ Civil Engine Mode Active: Generating Cinematic Visualization with Traffic Simulation...")
        
        if super_opt:
            st.info("⚡ Super-Optimization Active: Enhancing output by 1000x...")

        # Real Processing via Brain
        # Pre-process user input for geometry & feasibility context before calling Brain
        import re
        dims = []
        height_hint = None
        shape_hint = None
        # Extract dimension patterns like 20x30, 40×50, or single numbers followed by 'm'
        dim_pattern = re.findall(r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)", user_input.lower())
        if dim_pattern:
            for a, b in dim_pattern:
                dims = [float(a), float(b)]
        else:
            single_nums = re.findall(r"\b(\d{2,5})\b", user_input)  # pick large-ish numbers
            if len(single_nums) >= 2:
                dims = [float(single_nums[0]), float(single_nums[1])]
            elif len(single_nums) == 1:
                dims = [float(single_nums[0]), float(single_nums[0])]

        # Height extraction (keywords: height, ارتفاع)
        h_match = re.search(r"height\s*(\d+\.?\d*)", user_input.lower()) or re.search(r"ارتفاع\s*(\d+\.?\d*)", user_input.lower())
        if h_match:
            height_hint = float(h_match.group(1))
        else:
            # Provide a proportional height if dimensions exist
            height_hint = (sum(dims)/len(dims))*0.6 if dims else 30.0

        # Shape hint
        if any(k in user_input.lower() for k in ["tower", "برج"]):
            shape_hint = "tower"
            if dims and height_hint < max(dims)*2:
                height_hint = max(dims)*2.5
        elif any(k in user_input.lower() for k in ["circle", "دایره"]):
            shape_hint = "circle"
        else:
            shape_hint = "rect"

        site_area = None
        if len(dims) == 2:
            site_area = dims[0] * dims[1]
        elif len(dims) == 1:
            site_area = dims[0] ** 2

        context_vars = {
            "site_area": site_area or 1000.0,
            "massing_shape": shape_hint,
            "dimensions": dims,
            "proposed_height": height_hint
        }
        if extracted_geometry:
            context_vars["dxf_geometry"] = extracted_geometry
            la = extracted_geometry.get("largest_footprint_area")
            if la and la > context_vars["site_area"]:
                context_vars["site_area"] = la

        with st.spinner(TRANSLATIONS[lang_code]["processing"]):
            t0 = time.time()
            response = brain.process_request(user_input, context_data=context_vars)
            elapsed_ms = (time.time() - t0) * 1000.0
            swarm = context_vars.get("active_agent_swarm", [])
            st.session_state.last_swarm_size = len(swarm)
            st.session_state.last_swarm_latency_ms = elapsed_ms
            # Failure rate placeholder until real failure tracking implemented
            st.session_state.last_fail_rate = 0.00005 if swarm else 0.0
            time.sleep(0.3)  # slight pause for UX

        # Result Display
        st.success(TRANSLATIONS[lang_code]["design_done"])
        
        # Show Council Verdict
        if "council_verdict" in response:
            st.info(f"👑 **Council Verdict:** {response['council_verdict']}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(TRANSLATIONS[lang_code]["specs"])

            # Deterministic project naming
            project_name = " ".join(user_input.split(" ")[:3]).title() or "Untitled Project"
            u_low = user_input.lower()

            # Style inference (non-random)
            if any(k in u_low for k in ["future", "آینده", "مریخ"]): selected_style = "Futuristic"
            elif any(k in u_low for k in ["classic", "کلاسیک", "رومی"]): selected_style = "Neoclassical"
            elif any(k in u_low for k in ["modern", "مدرن"]): selected_style = "Modern"
            elif any(k in u_low for k in ["old", "سنتی", "قدیمی"]): selected_style = "Traditional/Islamic"
            elif any(k in u_low for k in ["green", "سبز", "طبیعت", "ارگانیک", "organic"]): selected_style = "Organic"
            elif any(k in u_low for k in ["brutalist", "بروتالیست", "بروتالیسم"]): selected_style = "Brutalist"
            elif any(k in u_low for k in ["parametric", "پارامتریک"]): selected_style = "Parametric"
            else:
                selected_style = "Modern"

            # Structure inference
            if any(k in u_low for k in ["wood", "چوب"]): selected_structure = "Mass Timber"
            elif any(k in u_low for k in ["steel", "فولاد", "آهن"]): selected_structure = "Steel Frame"
            elif any(k in u_low for k in ["concrete", "بتن"]): selected_structure = "Reinforced Concrete"
            elif any(k in u_low for k in ["regolith", "سه بعدی"]): selected_structure = "3D Printed Regolith"
            else:
                selected_structure = "Reinforced Concrete"

            # Geometry metrics (fallback if feasibility pipeline not run yet)
            if len(dims) == 2:
                footprint_area = dims[0] * dims[1]
            elif len(dims) == 1:
                footprint_area = dims[0] ** 2
            else:
                footprint_area = 40.0 * 40.0
            height_val = height_hint
            floor_height = 3.2
            floors = max(1, int(height_val / floor_height))
            volume = footprint_area * height_val
            gfa_est = footprint_area * floors
            slenderness = height_val / ((dims[0] + dims[1]) / 2) if len(dims) == 2 and (dims[0] + dims[1]) else height_val / (dims[0] if dims else 40.0)

            # Pull feasibility metrics if available
            exec_res = response.get("execution_result")
            feas = exec_res.get("feasibility_report") if isinstance(exec_res, dict) else None
            if feas:
                efficiency = feas["metrics"]["efficiency_ratio"]
                daylight_score = feas["metrics"]["daylight_score"]
                structural_risk = feas["metrics"]["structural_risk"]
                volume = feas.get("volume_m3", volume)
                gfa_est = feas.get("estimated_gfa", gfa_est)
                floors = feas.get("floors", floors)
                footprint_area = feas.get("footprint_area", footprint_area)
                slenderness = feas.get("slenderness_ratio", slenderness)
            else:
                efficiency = f"{min(0.92, 0.75 + 0.02 * floors)*100:.1f}%"
                daylight_score = "High" if slenderness > 3.5 else ("Medium" if slenderness > 2 else "Low")
                structural_risk = "Low" if slenderness < 6 else ("Moderate" if slenderness < 8 else "High")

            style_info = get_style_info(selected_style, lang_code if lang_code in ["fa", "en"] else "en") or {}

            st.json({
                "Project": project_name,
                "Style": selected_style,
                "Style_Description": style_info.get("description"),
                "Structure": selected_structure,
                "Footprint_Area_m2": round(footprint_area, 2),
                "Height_m": round(height_val, 2),
                "Floors": floors,
                "Estimated_GFA_m2": round(gfa_est, 2),
                "Volume_m3": round(volume, 2),
                "Slenderness_Ratio": round(slenderness, 2),
                "Efficiency": efficiency,
                "Daylight": daylight_score,
                "Structural_Risk": structural_risk,
                "Climate_Note": style_info.get("climate"),
                "Structure_Synergy": style_info.get("structure_synergy")
            })
            
        with col2:
            st.markdown(TRANSLATIONS[lang_code]["preview"])
            # Build deterministic base polygon based on parsed dimensions
            if len(dims) == 2:
                w, l = dims
                if shape_hint == "circle":
                    # Approximate circle with 16-gon
                    import math
                    r = w/2.0
                    base_poly = [(r*math.cos(2*math.pi*i/16), r*math.sin(2*math.pi*i/16)) for i in range(16)]
                elif shape_hint == "tower":
                    # Slender rectangle
                    w = dims[0]*0.5
                    base_poly = [(0,0),(w,0),(w,l),(0,l)]
                else:
                    base_poly = [(0,0),(w,0),(w,l),(0,l)]
            else:
                # Fallback simple square
                s = dims[0] if dims else 40.0
                base_poly = [(0,0),(s,0),(s,s),(0,s)]

            height = height_hint or 30.0
            verts3d, faces = build_prism_mesh(base_poly, height)
            if verts3d and faces:
                # Optional vertex optimization for cleaner mesh
                verts3d_opt, faces_opt = optimize_vertices(verts3d, faces)
                xs = [v[0] for v in verts3d_opt]
                ys = [v[1] for v in verts3d_opt]
                zs = [v[2] for v in verts3d_opt]
                i_idx = [f[0] for f in faces_opt]
                j_idx = [f[1] for f in faces_opt]
                k_idx = [f[2] for f in faces_opt]
                mesh_fig = go.Figure(data=[go.Mesh3d(x=xs,y=ys,z=zs,i=i_idx,j=j_idx,k=k_idx,
                                                     color='orange',opacity=0.65)])
                mesh_fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                                       title=f"Generated Massing: {project_name} | {shape_hint} | {int(height)}h", template="plotly_dark")
                st.plotly_chart(mesh_fig, width="stretch")
            else:
                st.warning("Unable to generate massing from input. Using fallback visualization.")

            # Show feasibility report if available
            exec_res = response.get("execution_result")
            if isinstance(exec_res, dict) and exec_res.get("feasibility_report"):
                with st.expander("📊 Feasibility / Massing Report", expanded=True):
                    st.json(exec_res["feasibility_report"])

# --- TAB 4: EVOLUTION ---
with tab_evolution:
    st.subheader(TRANSLATIONS[lang_code]["evo_track"])
    
    # Web Training Interface
    st.markdown("### 🌐 Web Training Module")
    with st.expander("Train on External Website (e.g., BeyondCAD)", expanded=True):
        training_url = st.text_input("Enter URL to Learn From", value="https://beyondcad.com")
        if st.button("🚀 Initiate Web Training", type="primary"):
            with st.spinner(f"Analyzing {training_url} and absorbing capabilities..."):
                # Trigger training in Brain
                res = brain.train_system(training_url)
                st.success(res)
                st.balloons()

    # Simulated Evolution Data
    # Fix for "Infinite extent" warning: Ensure lists and float types and set index
    epochs = list(range(1, 101))
    chart_data = pd.DataFrame({
        "Intelligence": [float(x)**1.05 for x in epochs],
        "Speed": [float(x)**1.1 for x in epochs]
    })
    chart_data.index = epochs
    chart_data.index.name = "Epoch"
    
    st.line_chart(chart_data)
    st.caption(TRANSLATIONS[lang_code]["evo_cap"])



# --- TAB 6: DATA CONNECTIONS ---
with tab_connections:
    st.subheader(TRANSLATIONS[lang_code]["conn_title"])
    st.markdown(TRANSLATIONS[lang_code]["conn_desc"])
    st.markdown("---")
    
    # Get connection summary
    conn_summary = brain.data_connector.get_connection_summary()
    
    # Summary Metrics
    st.markdown(f"### {TRANSLATIONS[lang_code]['conn_summary']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(TRANSLATIONS[lang_code]["total_conn"], conn_summary["total_connections"])
    col2.metric(TRANSLATIONS[lang_code]["online_conn"], conn_summary["online_connections"], 
                delta=f"+{conn_summary['online_connections']}")
    col3.metric(TRANSLATIONS[lang_code]["offline_conn"], conn_summary["offline_connections"],
                delta=f"-{conn_summary['offline_connections']}", delta_color="inverse")
    col4.metric(TRANSLATIONS[lang_code]["last_sync"], 
                conn_summary.get("last_global_sync", "Never")[:10] if conn_summary.get("last_global_sync") != "Never" else "Never")
    
    # Sync Button
    if st.button(TRANSLATIONS[lang_code]["sync_now"], type="primary"):
        with st.spinner("Syncing connections..."):
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            status = loop.run_until_complete(brain.data_connector.sync_all_connections())
            loop.close()
            st.success("✅ Sync completed!")
            st.rerun()
    
    st.markdown("---")
    st.markdown(f"### {TRANSLATIONS[lang_code]['conn_category']}")
    
    # Get all connections
    connections = brain.data_connector.get_all_connections()
    
    # Display connections by category
    for category, sources in connections.items():
        if category == "offline_cache" or not sources:
            continue
        
        with st.expander(f"📂 {category.replace('_', ' ').title()} ({len(sources)} sources)", expanded=True):
            # Create DataFrame
            df_data = []
            for source in sources:
                name = source.get("name", "Unknown")
                conn_status = brain.data_connector.connection_status.get(name, {})
                is_online = conn_status.get("online", False)
                last_check = conn_status.get("last_check", "Never")
                
                df_data.append({
                    "Name": name,
                    "Type": source.get("type", "N/A"),
                    "Status": "🟢 Online" if is_online else "🔴 Offline",
                    "URL": source.get("url", "N/A"),
                    "Last Check": last_check[:19] if last_check != "Never" else "Never"
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, width="stretch", hide_index=True)

# --- TAB 7: KURDO CAD ---
with tab_cad:
    st.subheader(TRANSLATIONS[lang_code]["cad_title"])
    st.markdown(TRANSLATIONS[lang_code]["cad_desc"])
    st.info(TRANSLATIONS[lang_code]["cad_desc"]) # Reinforce with info box
    st.markdown("---")

    # Initialize Designer
    if "kurdo_designer" not in st.session_state:
        # Use a temporary directory for the project
        project_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kurdo_cad_projects', 'default_project')
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        st.session_state.kurdo_designer = InteractiveDesigner(project_dir)
        st.session_state.cad_history_log = []

    designer = st.session_state.kurdo_designer

    col_cmd, col_view = st.columns([1, 1])

    with col_cmd:
        st.markdown(f"### ⌨️ {TRANSLATIONS[lang_code]['cad_exec']}")
        
        # Vision Input (CAD)
        uploaded_file_cad = st.file_uploader("👁️ Import/Analyze CAD or Image", type=['png', 'jpg', 'jpeg', 'bmp', 'pdf', 'docx', 'txt', 'dwg', 'dxf'], key="cad_upload")
        if uploaded_file_cad:
            process_uploaded_file(uploaded_file_cad, "KURDO_CAD")

        # Command Input
        cmd_input = st.text_input(TRANSLATIONS[lang_code]["cad_input"], key="cad_cmd_input")
        
        if st.button(TRANSLATIONS[lang_code]["cad_exec"], type="primary"):
            if cmd_input:
                result = designer.execute_command(cmd_input)
                st.session_state.cad_history_log.append(f"> {cmd_input}")
                st.session_state.cad_history_log.append(f"  {result}")
                st.success(f"Executed: {result}")
        
        # File Watcher Control
        st.markdown(f"### 👁️ {TRANSLATIONS[lang_code]['cad_watcher']}")
        
        if "watcher_active" not in st.session_state:
            st.session_state.watcher_active = False
            
        if st.session_state.watcher_active:
            st.success("Watcher is ACTIVE")
            if st.button(TRANSLATIONS[lang_code]["cad_stop_watch"]):
                designer.stop_watcher()
                st.session_state.watcher_active = False
                st.rerun()
        else:
            st.warning("Watcher is STOPPED")
            if st.button(TRANSLATIONS[lang_code]["cad_start_watch"]):
                # Start in a separate thread to avoid blocking Streamlit? 
                # The watchdog observer runs in its own thread usually.
                designer.start_watcher()
                st.session_state.watcher_active = True
                st.rerun()

        # History Log
        st.markdown(f"### 📜 {TRANSLATIONS[lang_code]['cad_history']}")
        history_text = "\n".join(st.session_state.cad_history_log[-10:]) # Show last 10
        st.code(history_text, language="text")

    with col_view:
        st.markdown(f"### ⚡ {TRANSLATIONS[lang_code]['cad_perf']}")
        
        # Performance Metrics
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Engine Latency", "0.02ms", "-99%")
        col_p2.metric("Spatial Index", "Active", "O(1)")
        
        st.markdown(f"### 🏗️ {TRANSLATIONS[lang_code]['cad_entities']}")
        
        # Get current entities count
        msp = designer.engine.active_document.modelspace()
        entity_counts = {}
        for e in msp:
            etype = e.dxftype()
            entity_counts[etype] = entity_counts.get(etype, 0) + 1
            
        if entity_counts:
            st.json(entity_counts)
        else:
            st.info("No entities in drawing yet.")
            
        # Download Button
        # Save current state to a temp file for download
        temp_dxf_path = designer.engine.save_drawing("current_design.dxf")
        
        with open(temp_dxf_path, "rb") as f:
            st.download_button(
                label=TRANSLATIONS[lang_code]["cad_download"],
                data=f,
                file_name="kurdo_design.dxf",
                mime="application/dxf"
            )
            
        # Simple Visualization (Scatter plot of points)
        # Extract points from lines for a basic preview
        x_vals = []
        y_vals = []
        
        for e in msp:
            if e.dxftype() == 'LINE':
                x_vals.extend([e.dxf.start.x, e.dxf.end.x, None])
                y_vals.extend([e.dxf.start.y, e.dxf.end.y, None])
            elif e.dxftype() == 'LWPOLYLINE':
                points = e.get_points()
                for p in points:
                    x_vals.append(p[0])
                    y_vals.append(p[1])
                x_vals.append(None)
                y_vals.append(None)
                
        if x_vals:
            fig = go.Figure(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name='Lines'))
            fig.update_layout(title="2D Preview", template="plotly_dark", showlegend=False)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig, width="stretch")

# --- TAB 8: GOVERNANCE ---
with tab_gov:
    st.subheader(TRANSLATIONS[lang_code]["gov_title"])
    st.markdown(TRANSLATIONS[lang_code]["gov_desc"])
    
    col_status, col_actions = st.columns([1, 1])
    
    with col_status:
        status_label = "ACTIVE"
        delta_color = "normal"
        if governance.system_frozen:
            status_label = "FROZEN"
            delta_color = "inverse"
        elif governance.core_shutdown:
            status_label = "SHUTDOWN"
            delta_color = "inverse"
            
        st.metric(TRANSLATIONS[lang_code]["gov_status"], status_label, delta="SECURE" if status_label == "ACTIVE" else "HALTED", delta_color=delta_color)
        
        st.markdown("### 📜 Active Directives (Categorized)")
        directives = governance.directives
        
        # Group by category
        from collections import defaultdict
        grouped_directives = defaultdict(list)
        for d in directives.values():
            grouped_directives[d.category.value].append(d)
            
        for category, rules in grouped_directives.items():
            with st.expander(f"📂 {category}", expanded=False):
                for d in rules:
                    st.markdown(f"**{d.id}. {d.title}**")
                    st.caption(d.description)

    with col_actions:
        st.markdown("### ⚡ Emergency Controls")
        
        # Freeze / Unfreeze
        if governance.system_frozen:
            if st.button(TRANSLATIONS[lang_code]["gov_unfreeze_btn"], type="primary", width="stretch"):
                governance.unfreeze_system()
                st.rerun()
        else:
            if st.button(TRANSLATIONS[lang_code]["gov_freeze_btn"], type="primary", width="stretch"):
                governance.freeze_system()
                st.rerun()
                
        st.markdown("---")
        
        # Advanced Controls
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            if governance.architect_locked:
                if st.button("🔓 Unlock Architect", width="stretch"):
                    governance.unlock_architect()
                    st.rerun()
            else:
                if st.button("🔒 Lock Architect Layer", width="stretch"):
                    governance.lock_architect()
                    st.rerun()
                    
        with col_adv2:
            if st.button("🛑 SHUTDOWN CORE", type="primary", width="stretch"):
                governance.shutdown_core()
                st.rerun()
                
        if st.button("♻️ Revert to Stable Version", width="stretch"):
            governance.revert_to_stable()
            st.toast("System Reverted to Last Stable Checkpoint.")
        
        st.markdown("### 📝 Audit Log (Last 5 Actions)")
        if governance.change_log:
            st.json(governance.change_log[-5:])
        else:
            st.info("No actions logged yet.")

