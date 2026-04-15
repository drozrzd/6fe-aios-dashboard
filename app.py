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
from groq import Groq
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
GROQ_API_KEY       = _secret("GROQ_API_KEY")
CARLOS_PROMPT_PATH = Path(__file__).parents[2] / "(C)Carlos-Voice-System-Prompt.md"


# ── Connections ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_db() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def get_groq() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


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
    system = CARLOS_PROMPT_PATH.read_text() if CARLOS_PROMPT_PATH.exists() else "You are Carlos."
    user_msg = f"""Reply to this Skool post as Carlos.

Category: {category}
Author: {author}
Post: {post_text}

Follow: Reality → Reframe → Actions (1-3 max) → Push. Short. No fluff."""
    resp = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
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
st.sidebar.caption("6FigureEngineer · Delivery Agent")
st.sidebar.divider()

_pending_count = len(fetch_pending_drafts())

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
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

if page == "📊 Overview":
    st.title("📊 Overview")

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

    tab1, tab2, tab3 = st.tabs(["🧪 Sandbox", "✍️ System Prompt", "📋 Correction Log"])

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
            with st.spinner("Calling Groq..."):
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
            "This is the core identity of the agent. Every reply generation starts here. "
            "Changes take effect on the next run — no restart needed."
        )

        if not CARLOS_PROMPT_PATH.exists():
            st.error(f"Prompt file not found: `{CARLOS_PROMPT_PATH}`")
        else:
            current = CARLOS_PROMPT_PATH.read_text()
            new_prompt = st.text_area(
                "System prompt:",
                value=current,
                height=500,
                label_visibility="collapsed",
            )

            pc1, pc2, _ = st.columns([1, 1, 4])
            if pc1.button("💾 Save", type="primary"):
                CARLOS_PROMPT_PATH.write_text(new_prompt)
                st.toast("Prompt saved ✅ — takes effect on next run.")
            if pc2.button("↩ Revert"):
                st.rerun()

            st.caption(f"File: `{CARLOS_PROMPT_PATH}`")

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
