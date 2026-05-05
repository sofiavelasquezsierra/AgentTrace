import os
import json
import re
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from anthropic import Anthropic, APIError
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

# Sierra AI brand palette
SIERRA = {
    "green": "#00C897",
    "green_dark": "#00A87E",
    "green_light": "#E6FBF6",
    "navy": "#0D1117",
    "navy_mid": "#1A2332",
    "slate": "#2D3748",
    "white": "#FFFFFF",
    "off_white": "#F7F9FC",
    "border": "#E2E8F0",
    "text": "#0D1117",
    "muted": "#64748B",
    "red": "#E4A2BA",
    "red_light": "#FEF2F2",
    "amber": "#ECA16B",
    "amber_light": "#FFFBEB",
}

EXAMPLE_CONFIG = {
    "company_name": "Bloom Coffee Co.",
    "persona": (
        "You are Maya, a friendly and upbeat customer support agent for Bloom Coffee Co. "
        "You help customers with orders, returns, product questions, and brewing tips. "
        "You are warm but professional. You never offer refunds on opened products. "
        "You can offer store credit for unopened items returned within 30 days with a receipt. "
        "You do not have access to individual order systems — direct tracking issues to bloom.coffee/track."
    ),
    "context": (
        "Bloom Coffee Co. sells premium single-origin coffees and brewing equipment online. "
        "Products: whole bean coffee ($18–$32/bag), pour-over kits ($45), subscriptions ($15/month). "
        "Shipping: free over $40, otherwise $5.99. No student discounts. "
        "Support hours: Mon–Fri 9am–6pm PT."
    ),
}

SAMPLE_SCENARIOS = [
    "I want to return a product I opened last week",
    "Do you offer student discounts?",
    "I never received my order and I need help now",
]

EVALUATOR_SYSTEM_PROMPT = """You are an AI agent quality evaluator. Analyze this customer service conversation and return a JSON object with exactly these fields:
- helpfulness_score: integer 1-10
- helpfulness_reasoning: string (1-2 sentences)
- policy_compliance_score: integer 1-10 (did the agent follow the stated policies?)
- policy_compliance_reasoning: string
- tone_score: integer 1-10 (consistent with the defined persona?)
- tone_reasoning: string
- failure_modes: array of strings (specific things the agent did wrong or could improve)
- wins: array of strings (specific things the agent did well)
- overall_verdict: one of 'Ready to Deploy', 'Needs Tuning', 'Major Issues'
- improvement_suggestions: array of up to 3 concrete prompt or policy changes that would improve the agent

Return only valid JSON. No markdown."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_agent_system_prompt(company_name: str, persona: str, context: str) -> str:
    parts = [persona.strip()]
    if context.strip():
        parts.append(f"\n\nAdditional context about {company_name}:\n{context.strip()}")
    return "\n".join(parts)


def call_agent(messages: list, system_prompt: str) -> str:
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text
    except APIError as e:
        return f"[API Error] {str(e)}"


def call_evaluator(transcript: str, agent_config: dict) -> dict | None:
    policy_context = (
        f"Agent persona and policies:\n{agent_config['persona']}\n\n"
        f"Company context:\n{agent_config.get('context', 'None provided')}\n\n"
        f"Conversation transcript:\n{transcript}"
    )
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=EVALUATOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": policy_context}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
    except APIError as e:
        st.error(f"Evaluation API error: {e}")
        return None


def format_transcript(messages: list) -> str:
    lines = []
    for m in messages:
        role = "Customer" if m["role"] == "user" else "Agent"
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def score_color(score: int) -> tuple[str, str]:
    """Returns (stroke_color, bg_color)."""
    if score >= 8:
        return SIERRA["green"], SIERRA["green_light"]
    if score >= 5:
        return SIERRA["amber"], SIERRA["amber_light"]
    return SIERRA["red"], SIERRA["red_light"]


def render_score_cards_html(helpfulness: int, policy: int, tone: int) -> str:
    def circle(label: str, score: int) -> str:
        color, bg = score_color(score)
        circumference = 2 * 3.14159 * 38
        filled = circumference * (score / 10)
        gap = circumference - filled
        return f"""
        <div style="
            background: {bg};
            border: 1.5px solid {color}33;
            border-radius: 16px;
            padding: 24px 20px 18px;
            text-align: center;
            flex: 1;
            min-width: 140px;
        ">
            <svg width="100" height="100" viewBox="0 0 90 90" style="display:block; margin: 0 auto 10px;">
                <circle cx="45" cy="45" r="38" fill="none" stroke="#E2E8F0" stroke-width="7"/>
                <circle cx="45" cy="45" r="38" fill="none"
                    stroke="{color}" stroke-width="7"
                    stroke-dasharray="{filled:.1f} {gap:.1f}"
                    stroke-linecap="round"
                    transform="rotate(-90 45 45)"/>
                <text x="45" y="50" text-anchor="middle"
                    fill="{color}" font-size="24" font-weight="700"
                    font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">{score}</text>
            </svg>
            <div style="
                font-size: 12px;
                font-weight: 600;
                color: #64748B;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            ">{label}</div>
        </div>
        """

    return f"""
    <div style="display: flex; gap: 16px; margin: 4px 0 8px; flex-wrap: wrap;">
        {circle("Helpfulness", helpfulness)}
        {circle("Policy Compliance", policy)}
        {circle("Tone Consistency", tone)}
    </div>
    """


def render_verdict_html(verdict: str) -> str:
    configs = {
        "Ready to Deploy": (SIERRA["green"], SIERRA["green_light"], "✓", "Ready to Deploy"),
        "Needs Tuning":    (SIERRA["amber"],  SIERRA["amber_light"],  "~", "Needs Tuning"),
        "Major Issues":    (SIERRA["red"],    SIERRA["red_light"],    "!", "Major Issues"),
    }
    color, bg, icon, label = configs.get(verdict, (SIERRA["muted"], SIERRA["off_white"], "?", verdict))
    return f"""
    <div style="
        display: inline-flex; align-items: center; gap: 10px;
        background: {bg};
        border: 1.5px solid {color};
        border-radius: 10px;
        padding: 10px 20px;
        margin-bottom: 20px;
    ">
        <span style="
            width: 26px; height: 26px;
            background: {color};
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            color: white; font-weight: 700; font-size: 14px;
            font-family: -apple-system, sans-serif;
        ">{icon}</span>
        <span style="
            font-size: 16px; font-weight: 700;
            color: {color};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        ">{label}</span>
    </div>
    """


def build_markdown_report(config: dict, result: dict, transcript: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# AgentTrace Evaluation Report",
        f"**Company:** {config['company_name']}  ",
        f"**Date:** {ts}  ",
        f"**Verdict:** {result['overall_verdict']}",
        "",
        "## Scores",
        "| Dimension | Score |",
        "|-----------|-------|",
        f"| Helpfulness | {result['helpfulness_score']}/10 |",
        f"| Policy Compliance | {result['policy_compliance_score']}/10 |",
        f"| Tone Consistency | {result['tone_score']}/10 |",
        "",
        "## Reasoning",
        f"**Helpfulness:** {result['helpfulness_reasoning']}",
        "",
        f"**Policy Compliance:** {result['policy_compliance_reasoning']}",
        "",
        f"**Tone:** {result['tone_reasoning']}",
        "",
        "## Wins",
    ]
    for w in result.get("wins", []):
        lines.append(f"- {w}")
    lines += ["", "## Failure Modes"]
    for f in result.get("failure_modes", []):
        lines.append(f"- {f}")
    lines += ["", "## Improvement Suggestions"]
    for i, s in enumerate(result.get("improvement_suggestions", []), 1):
        lines.append(f"{i}. {s}")
    lines += ["", "## Conversation Transcript", "```", transcript, "```"]
    return "\n".join(lines)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgentTrace",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
/* Sidebar */
[data-testid="stSidebar"] {{
    background: {SIERRA['navy']} !important;
    border-right: 1px solid {SIERRA['navy_mid']};
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {{
    color: #CBD5E1 !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {SIERRA['white']} !important;
}}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {{
    background: {SIERRA['navy_mid']} !important;
    color: #F1F5F9 !important;
    border: 1px solid #2D3748 !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] .stTextInput input::placeholder,
[data-testid="stSidebar"] .stTextArea textarea::placeholder {{
    color: #64748B !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stTextArea textarea:focus {{
    border-color: {SIERRA['green']} !important;
    box-shadow: 0 0 0 2px {SIERRA['green']}33 !important;
}}
/* Main area */
.main .block-container {{
    padding-top: 2rem;
    max-width: 900px;
}}
/* Chat messages */
[data-testid="stChatMessage"] {{
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}}
/* Buttons */
.stButton > button {{
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}}
[data-testid="stSidebar"] .stButton > button {{
    color: {SIERRA['slate']} !important;
    border-color: #2D3748 !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    border-color: {SIERRA['green']} !important;
    color: {SIERRA['green']} !important;
}}
.stButton > button[kind="primary"] {{
    background: {SIERRA['green']} !important;
    color: white !important;
    border: none !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: {SIERRA['green_dark']} !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px {SIERRA['green']}44 !important;
}}
/* Section headers */
.section-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {SIERRA['muted']};
    margin-bottom: 8px;
}}
/* Reasoning card */
.reasoning-card {{
    background: {SIERRA['off_white']};
    border: 1px solid {SIERRA['border']};
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 14px;
    line-height: 1.6;
    color: {SIERRA['text']};
    margin-bottom: 8px;
}}
/* Win / failure rows */
.win-row {{
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 14px;
    background: {SIERRA['green_light']};
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 14px;
    line-height: 1.5;
}}
.fail-row {{
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 14px;
    background: {SIERRA['red_light']};
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 14px;
    line-height: 1.5;
}}
.suggestion-card {{
    background: white;
    border: 1px solid {SIERRA['border']};
    border-left: 3px solid {SIERRA['green']};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-size: 14px;
    line-height: 1.6;
    color: {SIERRA['text']};
}}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

for key, default in {
    "messages": [],
    "agent_config": None,
    "evaluation": None,
    "pending_scenario": None,
    "_company": "",
    "_persona": "",
    "_context": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## AgentTrace")
    st.markdown(
        f'<p style="color:{SIERRA["green"]} !important; font-size:13px; margin-top:-8px;">AI Agent Evaluation </p>',
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("⚡  Load Example (Bloom Coffee)", use_container_width=True):
        st.session_state["_load_example"] = True

    load_ex = st.session_state.pop("_load_example", False)

    company_name = st.text_input(
        "Company Name",
        value=EXAMPLE_CONFIG["company_name"] if load_ex else st.session_state["_company"],
        placeholder="Acme Corp",
    )
    st.session_state["_company"] = company_name

    persona = st.text_area(
        "Agent Persona & Policies",
        value=EXAMPLE_CONFIG["persona"] if load_ex else st.session_state["_persona"],
        height=210,
        placeholder="You are [Name], a [role] for [company]. You help with... You never...",
    )
    st.session_state["_persona"] = persona

    context = st.text_area(
        "Product / Service Context (optional)",
        value=EXAMPLE_CONFIG["context"] if load_ex else st.session_state["_context"],
        height=140,
        placeholder="Pricing, policies, products, hours...",
    )
    st.session_state["_context"] = context

    st.divider()

    if st.button("Save Agent Config →", use_container_width=True, type="primary"):
        if not company_name.strip() or not persona.strip():
            st.error("Company name and persona are required.")
        else:
            st.session_state["agent_config"] = {
                "company_name": company_name.strip(),
                "persona": persona.strip(),
                "context": context.strip(),
            }
            st.session_state["messages"] = []
            st.session_state["evaluation"] = None
            st.success("Agent saved — start chatting!")

    if st.session_state["agent_config"]:
        name = st.session_state["agent_config"]["company_name"]
        st.markdown(
            f'<div style="margin-top:12px; padding:10px 12px; background:{SIERRA["navy_mid"]}; '
            f'border-radius:8px; border-left:3px solid {SIERRA["green"]};">'
            f'<span style="font-size:11px; color:{SIERRA["green"]} !important; '
            f'text-transform:uppercase; letter-spacing:0.06em;">Active</span><br>'
            f'<span style="font-size:14px; font-weight:600; color:white !important;">{name}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Main panel ────────────────────────────────────────────────────────────────

st.markdown("## AgentTrace")
st.markdown(
    f'<p style="color:{SIERRA["muted"]}; margin-top:-12px; margin-bottom:24px; font-size:15px;">'
    "Define your agent · Test in conversation · Get a scored evaluation"
    "</p>",
    unsafe_allow_html=True,
)

if not st.session_state["agent_config"]:
    st.markdown(
        f'<div style="background:{SIERRA["green_light"]}; border:1.5px solid {SIERRA["green"]}44; '
        f'border-radius:12px; padding:20px 24px; color:{SIERRA["text"]}; font-size:14px;">'
        "Configure your agent in the sidebar and click <strong>Save Agent Config</strong> to begin."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

config = st.session_state["agent_config"]
system_prompt = build_agent_system_prompt(
    config["company_name"], config["persona"], config["context"]
)

# ── Sample scenarios ──────────────────────────────────────────────────────────

st.markdown(
    '<p class="section-label">Quick-fire test scenarios</p>',
    unsafe_allow_html=True,
)
cols = st.columns(3)
for i, scenario in enumerate(SAMPLE_SCENARIOS):
    with cols[i]:
        short = scenario[:38] + "…" if len(scenario) > 38 else scenario
        if st.button(f'"{short}"', use_container_width=True, key=f"scenario_{i}"):
            st.session_state["pending_scenario"] = scenario

# ── Chat ──────────────────────────────────────────────────────────────────────

st.markdown(
    f'<p class="section-label" style="margin-top:20px;">Conversation · {config["company_name"]} agent</p>',
    unsafe_allow_html=True,
)

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state["pending_scenario"]:
    scenario_text = st.session_state.pop("pending_scenario")
    st.session_state["messages"].append({"role": "user", "content": scenario_text})
    with st.chat_message("user"):
        st.write(scenario_text)
    with st.chat_message("assistant"):
        with st.spinner("Responding..."):
            reply = call_agent(st.session_state["messages"], system_prompt)
        st.write(reply)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.session_state["evaluation"] = None
    st.rerun()

if user_input := st.chat_input(f"Message {config['company_name']} support..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Responding..."):
            reply = call_agent(st.session_state["messages"], system_prompt)
        st.write(reply)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.session_state["evaluation"] = None

# ── Evaluate + Clear ──────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
n_exchanges = len([m for m in st.session_state["messages"] if m["role"] == "assistant"])
col_eval, col_clear = st.columns([4, 1])

with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["evaluation"] = None
        st.rerun()

with col_eval:
    label = (
        f"Evaluate Conversation  ({n_exchanges}/3 exchanges needed)"
        if n_exchanges < 3
        else "Evaluate Conversation →"
    )
    if st.button(label, type="primary", disabled=(n_exchanges < 3), use_container_width=True):
        transcript = format_transcript(st.session_state["messages"])
        with st.spinner("Evaluating…"):
            result = call_evaluator(transcript, config)
        if result is None:
            st.error(
                "Could not parse the evaluation response. "
                "Try adding a few more exchanges and evaluating again."
            )
        else:
            st.session_state["evaluation"] = result


# ── Evaluation Report ─────────────────────────────────────────────────────────

if st.session_state["evaluation"]:
    result = st.session_state["evaluation"]
    transcript = format_transcript(st.session_state["messages"])

    st.divider()
    st.markdown("### Evaluation Report")

    # Verdict
    components.html(render_verdict_html(result.get("overall_verdict", "Needs Tuning")), height=68)

    # Score circles — rendered via components.html to support SVG
    components.html(
        render_score_cards_html(
            result.get("helpfulness_score", 0),
            result.get("policy_compliance_score", 0),
            result.get("tone_score", 0),
        ),
        height=190,
    )

    # Reasoning
    with st.expander("Score Reasoning", expanded=False):
        for label, key in [
            ("Helpfulness", "helpfulness_reasoning"),
            ("Policy Compliance", "policy_compliance_reasoning"),
            ("Tone", "tone_reasoning"),
        ]:
            st.markdown(
                f'<div class="reasoning-card"><strong>{label}:</strong> {result.get(key, "")}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_w, col_f = st.columns(2)

    with col_w:
        st.markdown('<p class="section-label">Wins</p>', unsafe_allow_html=True)
        wins = result.get("wins", [])
        if wins:
            rows = "".join(
                f'<div class="win-row">'
                f'<span style="color:{SIERRA["green"]}; font-weight:700; flex-shrink:0;">✓</span>'
                f'<span>{w}</span></div>'
                for w in wins
            )
            st.markdown(rows, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#94A3B8; font-size:14px;">None recorded.</span>', unsafe_allow_html=True)

    with col_f:
        st.markdown('<p class="section-label">Failure Modes</p>', unsafe_allow_html=True)
        failures = result.get("failure_modes", [])
        if failures:
            rows = "".join(
                f'<div class="fail-row">'
                f'<span style="color:{SIERRA["red"]}; font-weight:700; flex-shrink:0;">✗</span>'
                f'<span>{f}</span></div>'
                for f in failures
            )
            st.markdown(rows, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#94A3B8; font-size:14px;">None recorded.</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Improvement Suggestions</p>', unsafe_allow_html=True)
    for i, suggestion in enumerate(result.get("improvement_suggestions", []), 1):
        st.markdown(
            f'<div class="suggestion-card"><strong>{i}.</strong> {suggestion}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    md_report = build_markdown_report(config, result, transcript)
    st.download_button(
        label="Download Evaluation Report (.md)",
        data=md_report,
        file_name=(
            f"agenttrace_{config['company_name'].lower().replace(' ', '_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        ),
        mime="text/markdown",
        use_container_width=True,
    )
