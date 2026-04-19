"""
(C)AIOS Control Plane — Streamlit Dashboard
─────────────────────────────────────────────
Four sections:
  📊 Overview      — live agent status + activity timeline
  ✅ Approval Queue — review drafts, approve / edit+approve / reject with reason
  📈 KPIs          — approval rate, edit rate, mistake patterns
  🧠 Train Agent   — sandbox tests, system prompt editor, correction log

Every rejection reason and edit note feeds back into the agent's few-shot
memory automatically — the agent gets better with every correction.

Run:
  cd "05 Reporting & Metrics/dashboard"
  source venv/bin/activate
  streamlit run app.py
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AIOS Control Plane",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Secrets: works locally (reads skool-agent .env) AND on Streamlit Cloud ───
# Priority: st.secrets (cloud) → local .env file → environment variable

def _secret(key: str) -> str:
    """Read a secret from Streamlit Cloud secrets, local .env, or env var — in that order."""
    # 1. Streamlit Cloud secrets (set in app settings on share.streamlit.io)
    try:
        return st.secrets[key]
    except Exception:
        pass
    # 2. Local .env from the skool-agent directory (avoids duplicating secrets)
    _agent_env = Path(__file__).parents[2] / "04 Delivery & Support" / "skool-agent" / ".env"
    load_dotenv(_agent_env if _agent_env.exists() else Path(".env"), override=False)
    return os.environ.get(key, "")

SUPABASE_URL       = _secret("SUPABASE_URL")
SUPABASE_KEY       = _secret("SUPABASE_KEY")
GEMINI_API_KEY     = _secret("GEMINI_API_KEY")
_GEMINI_BASE_URL   = "https://generativelanguage.googleapis.com/v1beta/openai/"
CARLOS_PROMPT_PATH = Path(__file__).parents[2] / "(C)Carlos-Voice-System-Prompt.md"


def _inbound_secret(key: str) -> str:
    """Read from the inbound-agent .env directly (different Supabase project)."""
    try:
        return st.secrets[f"INBOUND_{key}"]
    except Exception:
        pass
    _env = Path(__file__).parents[2] / "01 Acquisition" / "inbound-agent" / ".env"
    if _env.exists():
        for line in _env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line[len(key) + 1:].strip()
    return ""


INBOUND_SUPABASE_URL = _inbound_secret("SUPABASE_URL")
INBOUND_SUPABASE_KEY = _inbound_secret("SUPABASE_KEY")


# ── Connections ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_db() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def get_gemini() -> OpenAI:
    return OpenAI(api_key=GEMINI_API_KEY, base_url=_GEMINI_BASE_URL)


@st.cache_resource
def get_inbound_db() -> Client:
    return create_client(INBOUND_SUPABASE_URL, INBOUND_SUPABASE_KEY)


def idb() -> Client:
    return get_inbound_db()


# ── Data helpers ──────────────────────────────────────────────────────────────

def db() -> Client:
    return get_db()


def fmt_ts(ts_str: str, fmt: str = "%b %d %H:%M") -> str:
    """Parse an ISO timestamp and return a human-readable string."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime(fmt)
    except Exception:
        return ts_str[:16] if ts_str else "—"


def fetch_pending_drafts() -> list[dict]:
    result = (
        db().table("reply_drafts")
        .select("*, skool_posts(title, body, author, post_url, category)")
        .eq("status", "pending_approval")
        .order("generated_at", desc=True)
        .execute()
    )
    return result.data or []


def fetch_stats() -> dict:
    rows      = db().table("reply_drafts").select("status, edited_reply").execute().data or []
    total     = len(rows)
    approved  = sum(1 for r in rows if r["status"] in ("approved", "posted"))
    rejected  = sum(1 for r in rows if r["status"] == "rejected")
    edited    = sum(1 for r in rows if r.get("edited_reply"))
    pending   = sum(1 for r in rows if r["status"] == "pending_approval")
    week_ago  = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    posts_wk  = db().table("skool_posts").select("id", count="exact").gte("scraped_at", week_ago).execute()
    posted_wk = db().table("posted_replies").select("id", count="exact").gte("posted_at", week_ago).execute()
    return {
        "total":            total,
        "pending":          pending,
        "approved":         approved,
        "rejected":         rejected,
        "edited":           edited,
        "approval_rate":    round(approved / total * 100) if total else 0,
        "edit_rate":        round(edited / max(approved, 1) * 100),
        "posts_this_week":  posts_wk.count  or 0,
        "posted_this_week": posted_wk.count or 0,
    }


def fetch_recent_runs(limit: int = 20) -> list[dict]:
    result = (
        db().table("agent_runs")
        .select("*")
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def fetch_rejection_reasons() -> list[str]:
    result = (
        db().table("reply_drafts")
        .select("rejection_reason")
        .eq("status", "rejected")
        .execute()
    )
    return [r["rejection_reason"] for r in (result.data or []) if r.get("rejection_reason")]


def fetch_edit_notes() -> list[str]:
    result = (
        db().table("reply_drafts")
        .select("edit_note")
        .not_.is_("edit_note", "null")
        .execute()
    )
    return [r["edit_note"] for r in (result.data or []) if r.get("edit_note")]


def fetch_correction_log(limit: int = 30) -> list[dict]:
    result = (
        db().table("reply_drafts")
        .select("id, status, rejection_reason, edit_note, draft_reply, edited_reply, generated_at, skool_posts(title)")
        .in_("status", ["rejected", "approved", "posted"])
        .order("generated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [
        r for r in (result.data or [])
        if r.get("rejection_reason") or r.get("edit_note") or r.get("edited_reply")
    ]


def fetch_posted_replies(limit: int = 10) -> list[dict]:
    result = (
        db().table("posted_replies")
        .select("*, reply_drafts(draft_reply, edited_reply, skool_posts(title, post_url))")
        .order("posted_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def fetch_sandbox_tests(limit: int = 10) -> list[dict]:
    result = (
        db().table("sandbox_tests")
        .select("*")
        .order("tested_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── Inbound agent data helpers ────────────────────────────────────────────────

def ib_leads_by_state() -> dict:
    rows = idb().table("ig_leads").select("state").execute().data or []
    return dict(Counter(r["state"] for r in rows))


def ib_recent_leads(limit: int = 12) -> list:
    return (
        idb().table("ig_leads")
        .select("first_name, ig_username, state, trigger_word, current_question, "
                "follow_up_count, call_confirmed, last_message_at, created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )


def ib_runs(limit: int = 20) -> list:
    return (
        idb().table("ig_runs")
        .select("*")
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )


def ib_today_stats() -> dict:
    today = (datetime.now(timezone.utc)
             .replace(hour=0, minute=0, second=0, microsecond=0)
             .isoformat())
    runs = (
        idb().table("ig_runs")
        .select("run_type, leads_processed, messages_sent, errors")
        .gte("started_at", today)
        .execute()
        .data or []
    )
    return {
        "runs":    len(runs),
        "leads":   sum(r.get("leads_processed", 0) for r in runs),
        "msgs":    sum(r.get("messages_sent", 0) for r in runs),
        "errors":  sum(len(r.get("errors") or []) for r in runs),
    }


def ib_last_run() -> dict | None:
    rows = (
        idb().table("ig_runs")
        .select("run_type, started_at, errors")
        .order("started_at", desc=True)
        .limit(1)
        .execute()
        .data or []
    )
    return rows[0] if rows else None


def ib_intent_distribution() -> dict:
    rows = (
        idb().table("ig_messages")
        .select("intent_label")
        .eq("direction", "inbound")
        .not_.is_("intent_label", "null")
        .execute()
        .data or []
    )
    return dict(Counter(r["intent_label"] for r in rows if r.get("intent_label")))


def ib_training_counts() -> dict:
    rows = idb().table("ig_training_examples").select("outcome").execute().data or []
    counts = Counter(r["outcome"] for r in rows)
    return {"total": len(rows), **counts}


def ib_save_feedback(
    agent_name: str,
    bad_response: str = "",
    good_response: str = "",
    rule_text: str = "",
    rating: str = "",
    note: str = "",
) -> None:
    row: dict = {"agent_name": agent_name}
    if bad_response:
        row["bad_response"] = bad_response
    if good_response:
        row["good_response"] = good_response
    if rule_text:
        row["rule_text"] = rule_text
    if rating:
        row["rating"] = rating
    if note:
        row["note"] = note
    idb().table("ig_feedback").insert(row).execute()


def ib_fetch_feedback(agent_name: str = "", limit: int = 20) -> list:
    q = idb().table("ig_feedback").select("*").order("created_at", desc=True).limit(limit)
    if agent_name:
        q = q.eq("agent_name", agent_name)
    return q.execute().data or []


# ── Write actions ─────────────────────────────────────────────────────────────

def approve_draft(draft_id: int, edited_reply: str = None, edit_note: str = None) -> None:
    update = {"status": "approved", "approved_at": datetime.now(timezone.utc).isoformat()}
    if edited_reply:
        update["edited_reply"] = edited_reply
    if edit_note:
        update["edit_note"] = edit_note
    db().table("reply_drafts").update(update).eq("id", draft_id).execute()


def reject_draft(draft_id: int, reason: str = "") -> None:
    db().table("reply_drafts").update({
        "status":           "rejected",
        "rejection_reason": reason,
    }).eq("id", draft_id).execute()


def save_sandbox(post_text, category, author, reply, rating=None, note=None) -> None:
    db().table("sandbox_tests").insert({
        "post_text":       post_text,
        "category":        category,
        "author":          author,
        "generated_reply": reply,
        "rating":          rating,
        "note":            note,
    }).execute()


def generate_sandbox_reply(post_text: str, category: str, author: str) -> str:
    # Read system prompt from Supabase (same source as the live agent)
    _cfg = db().table("config").select("value").eq("key", "carlos_system_prompt").maybe_single().execute()
    system = (_cfg.data or {}).get("value") or "You are Carlos. Be direct. Push action."
    user_msg = f"""Reply to this Skool post as Carlos.

Category: {category}
Author: {author}
Post: {post_text}

Follow: Reality → Reframe → Actions (1-3 max) → Push. Short. No fluff."""
    resp = get_gemini().chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=400,
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🤖 AIOS Control Plane")
st.sidebar.caption("6FigureEngineer · AI Operations")
st.sidebar.divider()

_pending_count = len(fetch_pending_drafts())

page = st.sidebar.radio(
    "Navigate",
    [
        "🔁 Inbound Agent",
        "📊 Delivery Agent",
        f"✅ Approval Queue  ({_pending_count} pending)",
        "📈 KPIs & Mistakes",
        "🧠 Train Agent",
    ],
)

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh", use_container_width=True):
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Inbound Agent
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🔁 Inbound Agent":
    st.title("🔁 Inbound Instagram Agent")
    st.caption("Live monitoring for the IG DM qualification system · polls GHL every 2 min")

    # ── System health banner ──────────────────────────────────────────────────
    today_stats = ib_today_stats()
    last_run    = ib_last_run()

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Leads today",    today_stats["leads"])
    h2.metric("Messages sent",  today_stats["msgs"])
    h3.metric("Poll cycles",    today_stats["runs"])
    h4.metric("Errors today",   today_stats["errors"],
              delta="⚠ check logs" if today_stats["errors"] else None,
              delta_color="inverse")

    if last_run:
        last_errors = last_run.get("errors") or []
        last_ts     = fmt_ts(last_run.get("started_at", ""))
        status_icon = "🔴" if last_errors else "🟢"
        h5.metric("Last poll", last_ts, delta=status_icon)
    else:
        h5.metric("Last poll", "No runs yet")

    st.divider()

    # ── Lead pipeline funnel ──────────────────────────────────────────────────
    st.subheader("Lead Pipeline")

    states      = ib_leads_by_state()
    total_leads = sum(states.values())

    FUNNEL = [
        ("TRIGGERED",    "🟡", "Triggered"),
        ("QUALIFYING",   "🔵", "Qualifying"),
        ("CALL_OFFERED", "🟠", "Call offered"),
        ("CONFIRMED",    "🟢", "Confirmed"),
    ]
    TERMINAL = [
        ("GHOSTED",      "👻", "Ghosted"),
        ("UNQUALIFIED",  "❌", "Unqualified"),
    ]

    if total_leads == 0:
        st.info("No leads yet. Send a DM with 'FREELANCE' to the 6FE Instagram account to trigger the agent.")
    else:
        f_cols = st.columns(len(FUNNEL) + len(TERMINAL))
        for i, (state, icon, label) in enumerate(FUNNEL):
            n = states.get(state, 0)
            pct = round(n / total_leads * 100) if total_leads else 0
            f_cols[i].metric(f"{icon} {label}", n, help=f"{pct}% of all leads")

        st.caption("─" * 4 + " terminal " + "─" * 4)
        t_cols = st.columns(len(TERMINAL) + 4)
        for i, (state, icon, label) in enumerate(TERMINAL):
            n = states.get(state, 0)
            t_cols[i].metric(f"{icon} {label}", n)

        # Conversion summary
        offered   = states.get("CALL_OFFERED", 0) + states.get("CONFIRMED", 0)
        confirmed = states.get("CONFIRMED", 0)
        if offered:
            conv = round(confirmed / offered * 100)
            st.caption(f"Call → Confirmed conversion: **{conv}%** ({confirmed}/{offered})")

    st.divider()

    # ── Agent monitoring cards ────────────────────────────────────────────────
    st.subheader("Agent Status")

    runs_all  = ib_runs(limit=100)
    intents   = ib_intent_distribution()
    training  = ib_training_counts()

    inbound_runs = [r for r in runs_all if r["run_type"] == "inbound_message"]
    n_inbound    = len(inbound_runs)
    top_intent   = max(intents, key=intents.get) if intents else "—"
    n_classified = sum(intents.values())

    qualifying_leads  = states.get("QUALIFYING", 0)
    call_offered      = states.get("CALL_OFFERED", 0)
    confirmed_leads   = states.get("CONFIRMED", 0)
    unqualified_leads = states.get("UNQUALIFIED", 0)

    # Determine system-level status from last run
    def _agent_status(last_active_ts: str | None, has_error: bool = False) -> str:
        if has_error:
            return "🔴 Error"
        if not last_active_ts:
            return "⚪ No data"
        try:
            dt = datetime.fromisoformat(last_active_ts.replace("Z", "+00:00"))
            mins_ago = (datetime.now(timezone.utc) - dt).total_seconds() / 60
            return "🟢 Active" if mins_ago < 60 else "🟡 Idle"
        except Exception:
            return "⚪ Unknown"

    last_run_ts = (last_run or {}).get("started_at")
    last_run_err = bool((last_run or {}).get("errors"))
    sys_status = _agent_status(last_run_ts, last_run_err)

    AGENTS = [
        {
            "title":   "Supervisor Agent",
            "icon":    "🧠",
            "desc":    "Routes every inbound message to the correct worker node",
            "metrics": [
                ("Dispatches (all time)", n_inbound),
                ("Today",               today_stats["leads"]),
            ],
            "detail": f"Top intent routed: **{top_intent}**",
            "status": sys_status,
        },
        {
            "title":   "Intent & Context Agent",
            "icon":    "🔍",
            "desc":    "Classifies lead intent before every supervisor decision",
            "metrics": [
                ("Labels assigned", n_classified),
                ("Unique intents",  len(intents)),
            ],
            "detail": (
                "\n".join(f"- `{k}`: {v}" for k, v in
                          sorted(intents.items(), key=lambda x: -x[1])[:4])
                or "No labels yet"
            ),
            "status": sys_status,
        },
        {
            "title":   "Qualification Agent",
            "icon":    "📋",
            "desc":    "Drives Carlos's 11-question funnel one question at a time",
            "metrics": [
                ("Active (in Q flow)", qualifying_leads),
                ("Q flow completions", states.get("CALL_OFFERED", 0) + confirmed_leads),
            ],
            "detail":  "Handles price/trust objections inline — no detour nodes",
            "status":  "🟢 Active" if qualifying_leads > 0 else sys_status,
        },
        {
            "title":   "Booking & CRM Agent",
            "icon":    "📅",
            "desc":    "Sends the Calendly offer and confirms screenshot receipt",
            "metrics": [
                ("Calls offered",   call_offered + confirmed_leads),
                ("Calls confirmed", confirmed_leads),
            ],
            "detail": (
                f"Conversion: **{round(confirmed_leads / (call_offered + confirmed_leads) * 100)}%**"
                if (call_offered + confirmed_leads) > 0 else "No offers yet"
            ),
            "status": "🟢 Active" if confirmed_leads > 0 else sys_status,
        },
        {
            "title":   "Graceful Close Agent",
            "icon":    "🤝",
            "desc":    "Politely exits leads who only want clients, not coaching",
            "metrics": [
                ("Total closes", unqualified_leads),
                ("",             ""),
            ],
            "detail":  "Fires on `wants_clients_only` intent",
            "status":  "🟡 Idle" if unqualified_leads == 0 else "🟢 Active",
        },
        {
            "title":   "Training Extraction Agent",
            "icon":    "🧬",
            "desc":    "Labels terminal-state conversations and stores them as training data",
            "metrics": [
                ("Examples collected", training["total"]),
                ("Confirmed / Ghosted",
                 f"{training.get('confirmed', 0)} / {training.get('ghosted', 0)}"),
            ],
            "detail":  "Runs async when a lead reaches CONFIRMED, GHOSTED, or UNQUALIFIED",
            "status":  "🟢 Active" if training["total"] > 0 else "⚪ No data",
        },
    ]

    for i in range(0, len(AGENTS), 2):
        col_a, col_b = st.columns(2)
        for col, agent in zip((col_a, col_b), AGENTS[i:i+2]):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {agent['icon']} {agent['title']}")
                    st.caption(agent["desc"])
                    st.markdown(f"**Status:** {agent['status']}")
                    st.divider()
                    m_cols = st.columns(2)
                    for j, (label, val) in enumerate(agent["metrics"]):
                        if label:
                            m_cols[j % 2].metric(label, val)
                    if agent.get("detail"):
                        st.markdown(agent["detail"])

    st.divider()

    # ── Recent leads table ────────────────────────────────────────────────────
    st.subheader("Recent Leads")

    leads = ib_recent_leads()
    if not leads:
        st.info("No leads yet.")
    else:
        STATE_ICONS = {
            "TRIGGERED":    "🟡",
            "QUALIFYING":   "🔵",
            "CALL_OFFERED": "🟠",
            "CONFIRMED":    "🟢",
            "GHOSTED":      "👻",
            "UNQUALIFIED":  "❌",
        }
        for lead in leads:
            state      = lead.get("state", "")
            icon       = STATE_ICONS.get(state, "⚪")
            name       = lead.get("first_name") or lead.get("ig_username") or "Unknown"
            q_num      = lead.get("current_question", 1)
            trigger    = lead.get("trigger_word", "—")
            last_msg   = fmt_ts(lead.get("last_message_at") or lead.get("created_at", ""))
            confirmed  = "✅" if lead.get("call_confirmed") else ""
            fu         = lead.get("follow_up_count", 0)
            fu_str     = f" · {fu} follow-ups" if fu else ""

            state_label = state.replace("_", " ").title()
            detail = f"Q{q_num}" if state == "QUALIFYING" else state_label

            st.text(
                f"{icon}  {name:<20}  [{detail:<14}]  "
                f"trigger: {(trigger or '—')[:20]:<22}  "
                f"last msg: {last_msg}  {confirmed}{fu_str}"
            )

    st.divider()

    # ── Run history ───────────────────────────────────────────────────────────
    st.subheader("Run History")
    runs = ib_runs(limit=20)
    if not runs:
        st.info("No runs recorded yet.")
    else:
        for run in runs:
            rtype    = run.get("run_type", "")
            ts       = fmt_ts(run.get("started_at", ""))
            n_leads  = run.get("leads_processed", 0)
            n_msgs   = run.get("messages_sent", 0)
            errs     = run.get("errors") or []
            icon     = "❌" if errs else ("✅" if n_msgs else "⚪")
            err_str  = f"  ⚠ {errs[0][:60]}" if errs else ""
            st.text(
                f"{icon}  {ts}  [{rtype:<18}]  "
                f"{n_leads} lead(s) · {n_msgs} sent{err_str}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Delivery Agent Overview
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Delivery Agent":
    st.title("📊 Delivery Agent — Overview")

    stats = fetch_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts scraped (7d)",  stats["posts_this_week"])
    c2.metric("Pending approval",    stats["pending"],
              delta=f"-{stats['pending']} to review" if stats["pending"] else None,
              delta_color="inverse")
    c3.metric("Posted this week",    stats["posted_this_week"])
    c4.metric("Total drafts ever",   stats["total"])

    st.divider()
    st.subheader("Agent Activity")

    runs = fetch_recent_runs()
    if not runs:
        st.info("No agent runs recorded yet. Runs appear here after the agent executes for the first time.")
    else:
        for run in runs:
            status   = run.get("status", "")
            run_type = run.get("run_type", "")
            ts       = fmt_ts(run.get("started_at", ""))
            icon     = {"success": "✅", "error": "❌", "running": "⏳"}.get(status, "⚪")

            if run_type == "scrape":
                scraped   = run.get("posts_scraped",    0)
                generated = run.get("drafts_generated", 0)
                detail    = f"Scraped {scraped} new posts · {generated} drafts generated"
            else:
                posted = run.get("drafts_posted", 0)
                detail = f"Posted {posted} replies to Skool"

            if run.get("error_message"):
                detail += f"  ⚠ {run['error_message']}"

            st.text(f"{icon}  {ts}  [{run_type.upper():6}]  {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Approval Queue
# ═══════════════════════════════════════════════════════════════════════════════

elif page.startswith("✅ Approval Queue"):
    st.title("✅ Approval Queue")

    drafts = fetch_pending_drafts()

    if not drafts:
        st.success("🎉 All clear — no drafts waiting for review.")
        st.caption("Run the scrape job to generate new drafts.")
    else:
        st.caption(f"{len(drafts)} draft(s) waiting · Approve, edit, or reject each one")
        st.divider()

        for draft in drafts:
            post     = draft.get("skool_posts") or {}
            draft_id = draft["id"]
            kp       = f"d{draft_id}"   # key prefix — unique per draft

            with st.container(border=True):
                # ── Header ───────────────────────────────────────────────────
                h1, h2 = st.columns([3, 1])
                with h1:
                    title    = post.get("title", "Untitled")
                    post_url = post.get("post_url", "#")
                    st.markdown(f"### [{title}]({post_url})")
                    st.caption(
                        f"by **{post.get('author', 'Unknown')}** · "
                        f"{post.get('category', 'Uncategorized')} · "
                        f"Draft #{draft_id}"
                    )
                with h2:
                    st.caption(fmt_ts(draft.get("generated_at", "")))

                # ── Original post (collapsed) ─────────────────────────────────
                with st.expander("📄 View original post"):
                    st.write(post.get("body", "No content available."))

                # ── Draft reply ───────────────────────────────────────────────
                st.markdown("**Carlos draft reply:**")
                st.info(draft.get("draft_reply", ""))

                st.divider()

                # ── Action flow ───────────────────────────────────────────────
                editing   = st.session_state.get(f"{kp}_editing",   False)
                rejecting = st.session_state.get(f"{kp}_rejecting", False)

                if editing:
                    edited = st.text_area(
                        "Edit the reply:",
                        value=draft.get("draft_reply", ""),
                        key=f"{kp}_edited_text",
                        height=150,
                    )
                    note = st.text_input(
                        "What was wrong with the original? (trains the agent)",
                        key=f"{kp}_edit_note",
                        placeholder="e.g. Too long, opener was too soft, didn't call out the real problem",
                    )
                    ec1, ec2 = st.columns(2)
                    if ec1.button("✅ Confirm & Approve", key=f"{kp}_confirm_edit", type="primary"):
                        approve_draft(draft_id, edited_reply=edited, edit_note=note or None)
                        st.session_state.pop(f"{kp}_editing", None)
                        st.toast("Approved with edits ✏️")
                        st.rerun()
                    if ec2.button("Cancel", key=f"{kp}_cancel_edit"):
                        st.session_state.pop(f"{kp}_editing", None)
                        st.rerun()

                elif rejecting:
                    reason = st.text_input(
                        "Why are you rejecting this? (trains the agent to avoid this mistake)",
                        key=f"{kp}_reason",
                        placeholder="e.g. Started with 'I', sounded agreeable, missed the real problem",
                    )
                    rc1, rc2 = st.columns(2)
                    if rc1.button("❌ Confirm Rejection", key=f"{kp}_confirm_reject", type="primary"):
                        reject_draft(draft_id, reason=reason)
                        st.session_state.pop(f"{kp}_rejecting", None)
                        st.toast("Draft rejected ❌")
                        st.rerun()
                    if rc2.button("Cancel", key=f"{kp}_cancel_reject"):
                        st.session_state.pop(f"{kp}_rejecting", None)
                        st.rerun()

                else:
                    b1, b2, b3 = st.columns(3)
                    if b1.button("✅ Approve", key=f"{kp}_approve", type="primary", use_container_width=True):
                        approve_draft(draft_id)
                        st.toast("Approved ✅")
                        st.rerun()
                    if b2.button("✏️ Edit & Approve", key=f"{kp}_edit", use_container_width=True):
                        st.session_state[f"{kp}_editing"] = True
                        st.rerun()
                    if b3.button("❌ Reject", key=f"{kp}_reject", use_container_width=True):
                        st.session_state[f"{kp}_rejecting"] = True
                        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: KPIs & Mistakes
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 KPIs & Mistakes":
    st.title("📈 KPIs & Mistakes")

    stats = fetch_stats()

    # ── Top metrics ────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Approval rate",  f"{stats['approval_rate']}%",
              help="% of drafts approved (including edited). Target: >80%")
    k2.metric("Edit rate",      f"{stats['edit_rate']}%",
              help="% of approved drafts that needed editing. Lower is better.")
    k3.metric("Total rejected", stats["rejected"])
    k4.metric("Total processed", stats["total"])

    st.divider()

    col_l, col_r = st.columns(2)

    # ── Rejection reasons ──────────────────────────────────────────────────────
    with col_l:
        st.subheader("❌ Top Rejection Reasons")
        st.caption("These feed directly into the agent's 'AVOID' few-shot block")
        reasons = fetch_rejection_reasons()
        if not reasons:
            st.info("No rejections with reasons yet.\nReject a draft with an explanation to build this list.")
        else:
            counts = Counter(reasons).most_common(10)
            for reason, count in counts:
                bar = "█" * count
                st.markdown(f"**{reason}**")
                st.caption(f"{bar} {count}x")
                st.write("")

    # ── Edit patterns ──────────────────────────────────────────────────────────
    with col_r:
        st.subheader("✏️ Edit Patterns")
        st.caption("What gets changed most often when a draft is edited")
        notes = fetch_edit_notes()
        if not notes:
            st.info("No edit notes yet.\nAdd a note when editing a draft to build this list.")
        else:
            for note in notes[:10]:
                st.markdown(f"- {note}")

    st.divider()

    # ── Posted replies log ─────────────────────────────────────────────────────
    st.subheader("✅ Posted Replies")
    posted = fetch_posted_replies()
    if not posted:
        st.info("No replies posted yet.")
    else:
        for p in posted:
            draft = p.get("reply_drafts") or {}
            post  = draft.get("skool_posts") or {}
            title = post.get("title", "Unknown post")
            url   = post.get("post_url", "#")
            reply = draft.get("edited_reply") or draft.get("draft_reply", "")
            ts    = fmt_ts(p.get("posted_at", ""))
            with st.expander(f"✅ {ts} · {title}"):
                st.markdown(f"[View on Skool →]({url})")
                st.write(reply)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Train Agent
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 Train Agent":
    st.title("🧠 Train Agent")
    st.caption(
        "Everything here feeds back into the agent automatically. "
        "Good sandbox replies become positive few-shot examples. "
        "Rejections and edit notes become negative examples. "
        "The agent gets better with every correction — no retraining needed."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["🧪 Sandbox", "✍️ System Prompt", "📋 Correction Log", "🔁 Inbound Feedback"])

    # ── Tab 1: Sandbox ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Test a reply in isolation")
        st.caption("Paste any Skool post → generate a Carlos reply → rate it. No production impact.")

        with st.form("sandbox_form", clear_on_submit=False):
            fc1, fc2 = st.columns(2)
            category = fc1.text_input("Category", placeholder="e.g. Closing, Mindset, Outreach")
            author   = fc2.text_input("Author name", placeholder="e.g. John M.")
            post_text = st.text_area("Post content", height=150,
                                     placeholder="Paste the Skool post here...")
            submitted = st.form_submit_button("⚡ Generate Reply", type="primary")

        if submitted and post_text.strip():
            with st.spinner("Calling Gemini..."):
                try:
                    reply = generate_sandbox_reply(
                        post_text, category or "General", author or "Member"
                    )
                    st.session_state["sb_reply"]    = reply
                    st.session_state["sb_post"]     = post_text
                    st.session_state["sb_category"] = category
                    st.session_state["sb_author"]   = author
                    st.session_state.pop("sb_rated", None)
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        if st.session_state.get("sb_reply") and not st.session_state.get("sb_rated"):
            st.divider()
            st.markdown("**Generated reply:**")
            st.info(st.session_state["sb_reply"])

            st.markdown("**Rate this reply:**")
            r1, r2 = st.columns(2)

            if r1.button("👍 Good — use as example", use_container_width=True, type="primary"):
                save_sandbox(
                    post_text=st.session_state["sb_post"],
                    category=st.session_state["sb_category"],
                    author=st.session_state["sb_author"],
                    reply=st.session_state["sb_reply"],
                    rating="good",
                )
                st.session_state["sb_rated"] = True
                st.toast("Saved as a positive example 👍")
                st.rerun()

            if r2.button("👎 Bad — save as example to avoid", use_container_width=True):
                st.session_state["sb_show_bad_input"] = True
                st.rerun()

            if st.session_state.get("sb_show_bad_input"):
                bad_note = st.text_input(
                    "What was wrong with this reply?",
                    placeholder="e.g. Too generic, didn't call out the problem, sounded like ChatGPT",
                )
                if st.button("Save negative example"):
                    save_sandbox(
                        post_text=st.session_state["sb_post"],
                        category=st.session_state["sb_category"],
                        author=st.session_state["sb_author"],
                        reply=st.session_state["sb_reply"],
                        rating="bad",
                        note=bad_note,
                    )
                    st.session_state["sb_rated"] = True
                    st.session_state.pop("sb_show_bad_input", None)
                    st.toast("Saved as a negative example 👎")
                    st.rerun()

        # Recent sandbox tests
        st.divider()
        st.subheader("Recent sandbox tests")
        tests = fetch_sandbox_tests()
        if not tests:
            st.info("No sandbox tests yet. Generate and rate a reply above to start building the training library.")
        else:
            for t in tests:
                icon = {"good": "👍", "bad": "👎"}.get(t.get("rating"), "⚪")
                ts   = fmt_ts(t.get("tested_at", ""))
                cat  = t.get("category") or "General"
                snip = (t.get("post_text") or "")[:60]
                with st.expander(f"{icon} {ts} · {cat} · {snip}..."):
                    st.markdown("**Post:**")
                    st.write(t.get("post_text", ""))
                    st.markdown("**Reply:**")
                    st.write(t.get("generated_reply", ""))
                    if t.get("note"):
                        st.caption(f"📝 Note: {t['note']}")

    # ── Tab 2: System Prompt ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Carlos Voice System Prompt")
        st.caption(
            "Stored in Supabase — editable from anywhere (Mac, cloud, phone). "
            "The agent reads this before every reply generation. "
            "Changes take effect on the next run — no restart needed."
        )

        # Read from Supabase — works from cloud AND local
        current = db().table("config").select("value, updated_at").eq("key", "carlos_system_prompt").maybe_single().execute()

        if not current.data:
            st.warning(
                "⚠ System prompt not in Supabase yet.\n\n"
                "Run this on your Mac to upload it:\n"
                "```\ncd '04 Delivery & Support/skool-agent'\n"
                "source venv/bin/activate\n"
                "python seed_config.py\n```"
            )
        else:
            prompt_text = current.data.get("value", "")
            updated_at  = fmt_ts(current.data.get("updated_at", ""))

            new_prompt = st.text_area(
                "System prompt:",
                value=prompt_text,
                height=500,
                label_visibility="collapsed",
            )

            pc1, pc2, _ = st.columns([1, 1, 4])
            if pc1.button("💾 Save", type="primary"):
                db().table("config").upsert(
                    {"key": "carlos_system_prompt", "value": new_prompt},
                    on_conflict="key",
                ).execute()
                st.toast("Prompt saved to Supabase ✅ — takes effect on next run.")
                st.rerun()
            if pc2.button("↩ Revert"):
                st.rerun()

            st.caption(f"Last updated: {updated_at} · Stored in Supabase `config` table")

    # ── Tab 3: Correction Log ─────────────────────────────────────────────────
    with tab3:
        st.subheader("Correction History")
        st.caption(
            "Every rejection reason and edit note is stored here "
            "and injected into the agent's prompt as negative few-shot examples."
        )

        corrections = fetch_correction_log(30)

        if not corrections:
            st.info(
                "No corrections yet.\n"
                "Reject a draft with a reason, or edit a draft with a note, "
                "to start building the correction library."
            )
        else:
            for c in corrections:
                post   = c.get("skool_posts") or {}
                title  = post.get("title", "Unknown post")
                status = c.get("status", "")
                icon   = "❌" if status == "rejected" else "✏️"
                ts     = fmt_ts(c.get("generated_at", ""), fmt="%b %d")

                with st.expander(f"{icon}  {ts} · Draft #{c['id']} · {title}"):
                    if c.get("rejection_reason"):
                        st.error(f"**Rejected:** {c['rejection_reason']}")

                    if c.get("edit_note"):
                        st.warning(f"**Edit note:** {c['edit_note']}")

                    if c.get("edited_reply"):
                        o_col, e_col = st.columns(2)
                        o_col.markdown("**Original draft:**")
                        o_col.write(c.get("draft_reply", ""))
                        e_col.markdown("**Edited version:**")
                        e_col.write(c["edited_reply"])
                    elif c.get("draft_reply"):
                        st.markdown("**Draft:**")
                        st.write(c["draft_reply"])

    # ── Tab 4: Inbound Feedback ───────────────────────────────────────────────
    with tab4:
        st.subheader("Train the Inbound Agent System")
        st.caption(
            "Corrections and rules saved here are injected directly into the agent's prompt "
            "on every message — no redeploy needed. Same flywheel as the Delivery Agent."
        )

        INBOUND_AGENTS = [
            "Qualification Agent",
            "Supervisor Agent",
            "Intent & Context Agent",
            "Booking & CRM Agent",
            "Graceful Close Agent",
            "Training Extraction Agent",
        ]

        selected_agent = st.selectbox("Which agent are you training?", INBOUND_AGENTS)

        st.divider()

        ftype = st.radio(
            "Feedback type",
            ["✏️ Message correction", "📏 Behavioral rule", "⭐ Conversation rating"],
            horizontal=True,
        )

        # ── Message correction ────────────────────────────────────────────────
        if ftype == "✏️ Message correction":
            st.caption(
                "Show the agent what it said wrong and what it should have said instead. "
                "These become few-shot examples injected before every reply."
            )
            with st.form("ib_correction_form", clear_on_submit=True):
                bad  = st.text_area("❌ What the bot said (bad response)",  height=100,
                                    placeholder="e.g. Claro, entiendo tu situación...")
                good = st.text_area("✅ What it should have said instead", height=100,
                                    placeholder="e.g. Oye, ¿cuántos años llevas en tech?")
                submitted = st.form_submit_button("💾 Save correction", type="primary")

            if submitted:
                if bad.strip() and good.strip():
                    ib_save_feedback(selected_agent, bad_response=bad.strip(), good_response=good.strip())
                    st.toast(f"Correction saved for {selected_agent} ✅")
                    st.rerun()
                else:
                    st.warning("Fill in both fields before saving.")

        # ── Behavioral rule ───────────────────────────────────────────────────
        elif ftype == "📏 Behavioral rule":
            st.caption(
                "Write a rule that applies to this agent at all times. "
                "e.g. 'Never ask about income before asking about experience.'"
            )
            with st.form("ib_rule_form", clear_on_submit=True):
                rule = st.text_area("Rule", height=100,
                                    placeholder="Never ask two questions in one message.\n"
                                                "Always acknowledge the answer before asking the next question.")
                submitted = st.form_submit_button("💾 Save rule", type="primary")

            if submitted:
                if rule.strip():
                    ib_save_feedback(selected_agent, rule_text=rule.strip())
                    st.toast(f"Rule saved for {selected_agent} ✅")
                    st.rerun()
                else:
                    st.warning("Write a rule before saving.")

        # ── Conversation rating ───────────────────────────────────────────────
        else:
            st.caption(
                "Rate a full conversation moment — no specific correction needed. "
                "Helps the Training Extractor label what good vs. bad looks like."
            )
            with st.form("ib_rating_form", clear_on_submit=True):
                rating_choice = st.radio("Rating", ["👍 Good", "👎 Bad"], horizontal=True)
                note = st.text_area("Optional note", height=80,
                                    placeholder="e.g. Lead was warming up and the bot pushed too hard on the call offer")
                submitted = st.form_submit_button("💾 Save rating", type="primary")

            if submitted:
                rating_val = "thumbs_up" if "Good" in rating_choice else "thumbs_down"
                ib_save_feedback(selected_agent, rating=rating_val, note=note.strip() or None)
                st.toast("Rating saved ✅")
                st.rerun()

        st.divider()

        # ── Feedback log ──────────────────────────────────────────────────────
        st.subheader(f"Recent feedback — {selected_agent}")

        log_all = st.checkbox("Show all agents", value=False)
        feed = ib_fetch_feedback(agent_name="" if log_all else selected_agent, limit=30)

        if not feed:
            st.info("No feedback saved yet for this agent. Add a correction above to start the flywheel.")
        else:
            FTYPE_ICONS = {
                "thumbs_up": "👍", "thumbs_down": "👎",
            }
            for f in feed:
                ts    = fmt_ts(f.get("created_at", ""), fmt="%b %d %H:%M")
                agent = f.get("agent_name", "")
                label = f"[{agent}]  " if log_all else ""

                if f.get("bad_response"):
                    with st.expander(f"✏️  {ts}  {label}Message correction"):
                        c1, c2 = st.columns(2)
                        c1.markdown("**❌ Bad:**")
                        c1.write(f["bad_response"])
                        c2.markdown("**✅ Good:**")
                        c2.write(f.get("good_response", ""))

                elif f.get("rule_text"):
                    with st.expander(f"📏  {ts}  {label}Rule"):
                        st.write(f["rule_text"])

                elif f.get("rating"):
                    icon = FTYPE_ICONS.get(f["rating"], "⭐")
                    note_str = f"  — {f['note']}" if f.get("note") else ""
                    with st.expander(f"{icon}  {ts}  {label}Rating{note_str[:50]}"):
                        st.write(f.get("note") or "No note.")
