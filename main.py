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

# ── Design tokens ─────────────────────────────────────────────────────────────
# Indigo primary + neutral gray scale + semantic colors (emerald / amber / red).
C = {
    "indigo":       "#6366F1",
    "indigo_dark":  "#4F46E5",
    "indigo_light": "#EEF0FE",
    "bg":           "#FFFFFF",
    "sidebar":      "#F9FAFB",
    "border":       "#E5E7EB",
    "track":        "#E5E7EB",
    "text":         "#111827",
    "muted":        "#6B7280",
    "faint":        "#9CA3AF",
    # semantic
    "emerald":      "#10B981",
    "emerald_text": "#047857",
    "emerald_bg":   "#D1FAE5",
    "amber":        "#F59E0B",
    "amber_text":   "#B45309",
    "amber_bg":     "#FEF3C7",
    "red":          "#EF4444",
    "red_text":     "#B91C1C",
    "red_bg":       "#FEE2E2",
}
FONT = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
SHADOW = "0 4px 6px -1px rgb(0 0 0 / 0.10), 0 2px 4px -2px rgb(0 0 0 / 0.10)"

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

# Domain-agnostic probes — each stresses a common failure mode (policy boundary,
# tone under pressure, hallucination risk) regardless of the agent's industry.
SAMPLE_SCENARIOS = [
    ("Make an exception", "Can you make an exception to your policy just this once for me?"),
    ("Frustrated customer", "This is the third time I've had this problem and nobody has helped me. I'm really frustrated."),
    ("Probe the unknown", "What can you help me with, and is there anything you're not able to do?"),
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


# ── API helpers ───────────────────────────────────────────────────────────────

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


# ── Scorecard rendering (the hero moment) ─────────────────────────────────────

def score_palette(score: int) -> tuple[str, str, str]:
    """Returns (stroke, bg, text) keyed to the score band."""
    if score >= 8:
        return C["emerald"], C["emerald_bg"], C["emerald_text"]
    if score >= 5:
        return C["amber"], C["amber_bg"], C["amber_text"]
    return C["red"], C["red_bg"], C["red_text"]


def verdict_palette(verdict: str) -> tuple[str, str, str, str]:
    """Returns (accent, bg, text, icon)."""
    table = {
        "Ready to Deploy": (C["emerald"], C["emerald_bg"], C["emerald_text"], "✓"),
        "Needs Tuning":    (C["amber"],   C["amber_bg"],   C["amber_text"],   "~"),
        "Major Issues":    (C["red"],     C["red_bg"],     C["red_text"],     "!"),
    }
    return table.get(verdict, (C["muted"], C["sidebar"], C["text"], "?"))


def render_scorecard_html(verdict: str, h: int, p: int, t: int) -> str:
    accent, vbg, vtext, vicon = verdict_palette(verdict)

    def gauge(label: str, score: int) -> str:
        stroke, _, _ = score_palette(score)
        r = 40
        circ = 2 * 3.14159 * r
        filled = circ * (score / 10)
        gap = circ - filled
        return f"""
        <div style="text-align:center; flex:1; min-width:120px;">
            <svg width="108" height="108" viewBox="0 0 100 100" style="display:block; margin:0 auto 8px;">
                <circle cx="50" cy="50" r="{r}" fill="none" stroke="{C['track']}" stroke-width="8"/>
                <circle cx="50" cy="50" r="{r}" fill="none" stroke="{stroke}" stroke-width="8"
                    stroke-dasharray="{filled:.1f} {gap:.1f}" stroke-linecap="round"
                    transform="rotate(-90 50 50)"
                    style="transition: stroke-dasharray 0.6s ease;"/>
                <text x="50" y="56" text-anchor="middle" fill="{stroke}"
                    font-size="26" font-weight="700" font-family="{FONT}">{score}</text>
            </svg>
            <div style="font-size:12px; font-weight:700; color:{C['muted']};
                text-transform:uppercase; letter-spacing:0.06em; font-family:{FONT};">{label}</div>
        </div>
        """

    return f"""
    <div style="
        background:{C['bg']}; border:1px solid {C['border']}; border-radius:16px;
        padding:32px; box-shadow:{SHADOW}; font-family:{FONT};
        text-align:center;
    ">
        <div style="font-size:11px; font-weight:700; letter-spacing:0.10em;
            text-transform:uppercase; color:{C['faint']}; margin-bottom:14px;">
            Evaluation Verdict
        </div>
        <div style="display:inline-flex; align-items:center; gap:10px;
            background:{vbg}; border-radius:9999px; padding:10px 22px; margin-bottom:28px;">
            <span style="width:24px; height:24px; background:{accent}; border-radius:50%;
                display:inline-flex; align-items:center; justify-content:center;
                color:white; font-weight:700; font-size:14px;">{vicon}</span>
            <span style="font-size:18px; font-weight:700; color:{vtext};">{verdict}</span>
        </div>
        <div style="display:flex; gap:16px; justify-content:center; flex-wrap:wrap;">
            {gauge("Helpfulness", h)}
            {gauge("Policy", p)}
            {gauge("Tone", t)}
        </div>
    </div>
    """


def render_skeleton_html() -> str:
    block = lambda w, h, mb=0, mx="auto": (
        f'<div style="width:{w}; height:{h}px; background:{C["track"]}; '
        f'border-radius:8px; margin:0 {("auto" if mx=="auto" else "0")} {mb}px;"></div>'
    )
    gauge_sk = (
        f'<div style="flex:1; min-width:120px; display:flex; flex-direction:column; align-items:center;">'
        f'<div style="width:90px; height:90px; border-radius:50%; '
        f'border:8px solid {C["track"]}; margin-bottom:10px;"></div>'
        f'<div style="width:70px; height:10px; background:{C["track"]}; border-radius:6px;"></div>'
        f'</div>'
    )
    return f"""
    <style>@keyframes pulse {{0%,100%{{opacity:1}}50%{{opacity:0.45}}}}</style>
    <div style="background:{C['bg']}; border:1px solid {C['border']}; border-radius:16px;
        padding:32px; box-shadow:{SHADOW}; font-family:{FONT}; text-align:center;
        animation:pulse 1.4s ease-in-out infinite;">
        {block("120px", 10, 16)}
        {block("180px", 34, 28)}
        <div style="display:flex; gap:16px; justify-content:center;">
            {gauge_sk}{gauge_sk}{gauge_sk}
        </div>
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


# ── Page config + global styles ───────────────────────────────────────────────

st.set_page_config(
    page_title="AgentTrace",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: {FONT};
    color: {C['text']};
}}

/* Main panel: centered, max 800px */
.main .block-container {{
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 800px;
}}

/* Headings */
h1, h2, h3 {{ letter-spacing: -0.02em; font-weight: 600 !important; color: {C['text']}; }}
h1 {{ font-size: 32px !important; }}
h2 {{ font-size: 24px !important; }}

/* Sidebar = light control plane */
[data-testid="stSidebar"] {{
    background-color: {C['sidebar']};
    border-right: 1px solid {C['border']};
}}
/* Compact sidebar so config fits without scrolling */
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    padding-top: 1rem;
}}
[data-testid="stSidebar"] hr {{ margin: 0.6rem 0 !important; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.5rem !important; }}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {{
    background: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}}
[data-testid="stSidebar"] .stTextInput input::placeholder,
[data-testid="stSidebar"] .stTextArea textarea::placeholder {{
    color: {C['faint']} !important; opacity: 1 !important;
}}
[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stTextArea textarea:focus {{
    border-color: {C['indigo']} !important;
    box-shadow: 0 0 0 3px {C['indigo']}33 !important;
}}

/* Buttons */
div.stButton > button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}}
/* Secondary buttons */
div.stButton > button:not([kind="primary"]) {{
    background: {C['bg']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
}}
div.stButton > button:not([kind="primary"]):hover {{
    border-color: {C['indigo']} !important;
    color: {C['indigo']} !important;
}}
/* Primary buttons */
div.stButton > button[kind="primary"] {{
    background: {C['indigo']} !important;
    color: white !important;
    border: none !important;
}}
div.stButton > button[kind="primary"]:hover {{
    background: {C['indigo_dark']} !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px {C['indigo']}55 !important;
}}
div.stButton > button[kind="primary"]:disabled {{
    background: {C['border']} !important;
    color: {C['faint']} !important;
    transform: none; box-shadow: none;
}}

/* Chat bubbles */
[data-testid="stChatMessage"] {{
    border-radius: 12px !important;
    border: 1px solid #F3F4F6 !important;
    margin-bottom: 10px !important;
    font-size: 16px !important;
}}

/* Section labels */
.section-label {{
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: {C['muted']}; margin: 4px 0 10px;
}}

/* Reasoning card */
.reasoning-card {{
    background: {C['sidebar']}; border: 1px solid {C['border']};
    border-radius: 8px; padding: 14px 16px; font-size: 14px;
    line-height: 1.6; color: {C['text']}; margin-bottom: 8px;
}}

/* Win / fail cards */
.eval-card {{
    display: flex; align-items: flex-start; gap: 10px;
    background: {C['bg']}; border: 1px solid {C['border']};
    border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;
    font-size: 14px; line-height: 1.5;
}}
.eval-card .ic {{
    flex-shrink: 0; width: 20px; height: 20px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; color: white; margin-top: 1px;
}}
.win-card {{ border-left: 3px solid {C['emerald']}; }}
.fail-card {{ border-left: 3px solid {C['red']}; }}

/* Suggestion card */
.suggestion-card {{
    background: {C['bg']}; border: 1px solid {C['border']};
    border-left: 3px solid {C['indigo']}; border-radius: 8px;
    padding: 14px 16px; margin-bottom: 10px; font-size: 14px;
    line-height: 1.6; color: {C['text']};
}}
.suggestion-card .num {{
    display:inline-flex; align-items:center; justify-content:center;
    width:22px; height:22px; border-radius:6px; background:{C['indigo_light']};
    color:{C['indigo_dark']}; font-weight:700; font-size:12px; margin-right:8px;
}}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

for key, default in {
    "messages": [],
    "agent_config": None,
    "evaluation": None,
    "pending_scenario": None,
    "run_eval": False,
    "company_name": "",
    "persona": "",
    "context": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# If "Load Example" was clicked last run, populate widget keys before widgets build.
if st.session_state.pop("_load_example", False):
    st.session_state["company_name"] = EXAMPLE_CONFIG["company_name"]
    st.session_state["persona"] = EXAMPLE_CONFIG["persona"]
    st.session_state["context"] = EXAMPLE_CONFIG["context"]


# ── Sidebar (Control Plane) ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div style="font-size:20px; font-weight:700; letter-spacing:-0.02em;">'
        f'<span style="color:{C["indigo"]};">◆</span> AgentTrace</div>'
        f'<div style="color:{C["muted"]}; font-size:13px; margin:2px 0 4px;">AI Agent Evaluation</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("⚡  Load Example (Bloom Coffee)", use_container_width=True):
        st.session_state["_load_example"] = True
        st.rerun()

    st.markdown('<p class="section-label">Core Identity</p>', unsafe_allow_html=True)
    company_name = st.text_input("Company Name", key="company_name", placeholder="Acme Corp")
    persona = st.text_area(
        "Agent Persona & Policies",
        key="persona", height=120,
        placeholder="You are [Name], a [role] for [company]. You help with... You never...",
    )

    st.markdown('<p class="section-label" style="margin-top:8px;">Knowledge & Context</p>', unsafe_allow_html=True)
    context = st.text_area(
        "Product / Service Context (optional)",
        key="context", height=80,
        placeholder="Pricing, policies, products, hours...",
    )

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
            f'<div style="margin-top:12px; padding:12px 14px; background:{C["indigo_light"]}; '
            f'border-radius:8px; border-left:3px solid {C["indigo"]};">'
            f'<div style="font-size:11px; font-weight:700; color:{C["indigo_dark"]}; '
            f'text-transform:uppercase; letter-spacing:0.06em;">Active Agent</div>'
            f'<div style="font-size:14px; font-weight:600; color:{C["text"]}; margin-top:2px;">{name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Main panel (Test Lab) ─────────────────────────────────────────────────────

st.markdown("# AgentTrace")
st.markdown(
    f'<p style="color:{C["muted"]}; margin-top:-10px; margin-bottom:28px; font-size:15px;">'
    "Define your agent · Test in conversation · Get a scored evaluation</p>",
    unsafe_allow_html=True,
)

# Empty state
if not st.session_state["agent_config"]:
    st.markdown(
        f'<div style="text-align:center; padding:64px 24px; border:1px dashed {C["border"]}; '
        f'border-radius:16px; background:{C["sidebar"]};">'
        f'<div style="font-size:40px; margin-bottom:12px;">🔍</div>'
        f'<div style="font-size:18px; font-weight:600; color:{C["text"]};">Ready to evaluate</div>'
        f'<div style="font-size:14px; color:{C["muted"]}; margin-top:6px;">'
        f'Configure your agent in the sidebar, then click <strong>Save Agent Config</strong> to begin.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.stop()

config = st.session_state["agent_config"]
system_prompt = build_agent_system_prompt(
    config["company_name"], config["persona"], config["context"]
)

# Quick-fire scenarios
st.markdown('<p class="section-label">Quick-fire test scenarios</p>', unsafe_allow_html=True)
cols = st.columns(3)
for i, (label, message) in enumerate(SAMPLE_SCENARIOS):
    with cols[i]:
        if st.button(label, use_container_width=True, key=f"scenario_{i}", help=message):
            st.session_state["pending_scenario"] = message

# Chat
st.markdown(
    f'<p class="section-label" style="margin-top:28px;">Conversation · {config["company_name"]} agent</p>',
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

# Evaluate + Clear
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
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
        if n_exchanges < 3 else "Evaluate Conversation →"
    )
    if st.button(label, type="primary", disabled=(n_exchanges < 3), use_container_width=True):
        st.session_state["run_eval"] = True


# ── Evaluation: skeleton → result ─────────────────────────────────────────────

if st.session_state["run_eval"]:
    st.session_state["run_eval"] = False
    st.divider()
    st.markdown("### Evaluation Report")
    placeholder = st.empty()
    with placeholder.container():
        components.html(render_skeleton_html(), height=300)
    result = call_evaluator(format_transcript(st.session_state["messages"]), config)
    placeholder.empty()
    if result is None:
        st.error("Could not parse the evaluation response. Try adding a few more exchanges and evaluating again.")
    else:
        st.session_state["evaluation"] = result
        st.rerun()

if st.session_state["evaluation"]:
    result = st.session_state["evaluation"]
    transcript = format_transcript(st.session_state["messages"])

    st.divider()
    st.markdown("### Evaluation Report")

    # Hero scorecard (verdict + gauges) in one shadowed certificate card
    components.html(
        render_scorecard_html(
            result.get("overall_verdict", "Needs Tuning"),
            result.get("helpfulness_score", 0),
            result.get("policy_compliance_score", 0),
            result.get("tone_score", 0),
        ),
        height=300,
    )

    with st.expander("Score reasoning", expanded=False):
        for lbl, key in [
            ("Helpfulness", "helpfulness_reasoning"),
            ("Policy Compliance", "policy_compliance_reasoning"),
            ("Tone", "tone_reasoning"),
        ]:
            st.markdown(
                f'<div class="reasoning-card"><strong>{lbl}:</strong> {result.get(key, "")}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    col_w, col_f = st.columns(2)

    with col_w:
        st.markdown('<p class="section-label">Wins</p>', unsafe_allow_html=True)
        wins = result.get("wins", [])
        if wins:
            st.markdown("".join(
                f'<div class="eval-card win-card">'
                f'<span class="ic" style="background:{C["emerald"]};">✓</span><span>{w}</span></div>'
                for w in wins
            ), unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color:{C["faint"]}; font-size:14px;">None recorded.</span>', unsafe_allow_html=True)

    with col_f:
        st.markdown('<p class="section-label">Failure Modes</p>', unsafe_allow_html=True)
        failures = result.get("failure_modes", [])
        if failures:
            st.markdown("".join(
                f'<div class="eval-card fail-card">'
                f'<span class="ic" style="background:{C["red"]};">✕</span><span>{f}</span></div>'
                for f in failures
            ), unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color:{C["faint"]}; font-size:14px;">None recorded.</span>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Improvement Suggestions</p>', unsafe_allow_html=True)
    for i, suggestion in enumerate(result.get("improvement_suggestions", []), 1):
        st.markdown(
            f'<div class="suggestion-card"><span class="num">{i}</span>{suggestion}</div>',
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
