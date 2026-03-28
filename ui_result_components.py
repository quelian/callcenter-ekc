from __future__ import annotations

import streamlit as st

from transcription import QualityEvaluation


def _score_color(score: int) -> str:
    if score >= 8:
        return "#16a34a"
    if score >= 5:
        return "#b45309"
    return "#b91c1c"


def _score_label(score: int) -> str:
    if score >= 9:
        return "Отлично"
    if score >= 8:
        return "Хорошо"
    if score >= 5:
        return "Нормально"
    return "Нужно улучшить"


def _score_badge_style(score: int) -> str:
    if score >= 8:
        return "background:#dcfce7;color:#16a34a;"
    if score >= 5:
        return "background:#fef9c3;color:#b45309;"
    return "background:#fee2e2;color:#b91c1c;"


def render_header() -> None:
    st.markdown(
        """
<div class="qa-header">
  <div class="qa-header-icon">📞</div>
  <div>
    <div class="qa-header-title">Анализ звонков</div>
    <div class="qa-header-sub">ДВФУ · Единый контактный центр</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_score_card(col, title: str, score: int, icon: str) -> None:
    color = _score_color(score)
    label = _score_label(score)
    badge_style = _score_badge_style(score)
    pct = score * 10
    with col:
        st.markdown(
            f"""
<div class="score-card">
  <div class="score-title">{icon} {title}</div>
  <div class="score-value" style="color:{color};">{score}</div>
  <div class="score-sub">/10</div>
  <div class="score-bar">
    <div class="score-fill" style="width:{pct}%; background:{color};"></div>
  </div>
  <div><span class="score-badge" style="{badge_style}">{label}</span></div>
</div>""",
            unsafe_allow_html=True,
        )


def render_summary(evaluation: QualityEvaluation) -> None:
    color = _score_color(evaluation.total_score)
    op = evaluation.operator_name or "Не определено"
    app = evaluation.applicant_name or "Не определено"
    op_badge = "" if evaluation.operator_in_staff else " ⚠️"
    st.markdown(
        f"""
<div class="summary-card">
  <div>
    <div class="summary-person-label">Оператор</div>
    <div class="summary-person-name">{op}{op_badge}</div>
  </div>
  <div>
    <div class="summary-person-label">Заявитель</div>
    <div class="summary-person-name">{app}</div>
  </div>
  <div class="summary-score-wrap">
    <div class="summary-score-label">Итоговая оценка</div>
    <div class="summary-score-val" style="color:{color};">
      {evaluation.total_score}<span class="summary-score-denom">/10</span>
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_gauge(score: int) -> None:
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"font": {"size": 52, "family": "Inter"}, "suffix": "/10"},
                gauge={
                    "axis": {
                        "range": [0, 10],
                        "tickwidth": 1,
                        "tickcolor": "#e5e7eb",
                        "tickfont": {"size": 11},
                    },
                    "bar": {"color": _score_color(score), "thickness": 0.28},
                    "bgcolor": "#f9fafb",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 5], "color": "#fee2e2"},
                        {"range": [5, 8], "color": "#fef9c3"},
                        {"range": [8, 10], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": _score_color(score), "width": 3},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            )
        )
        fig.update_layout(
            height=220,
            margin={"t": 20, "b": 10, "l": 20, "r": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except ImportError:
        st.metric("Итоговая оценка", f"{score}/10")


def requires_manual_review(
    evaluation: QualityEvaluation,
    tone_summary: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not evaluation.operator_in_staff:
        reasons.append("оператор не найден в штатном списке")
    if evaluation.total_score <= 4:
        reasons.append("низкий итоговый балл")
    if evaluation.consultation_score <= 4:
        reasons.append("слабая консультация")
    if evaluation.script_score <= 4:
        reasons.append("серьёзные нарушения скрипта")
    if "напряж" in tone_summary.lower():
        reasons.append("напряжённый тон разговора")
    return bool(reasons), reasons


def render_verdict(evaluation: QualityEvaluation, tone_summary: str) -> None:
    needs_review, reasons = requires_manual_review(evaluation, tone_summary)
    if needs_review:
        reasons_html = f'<div class="verdict-reasons">Причины: {", ".join(reasons)}</div>'
        st.markdown(
            f"""
<div class="verdict-bad">
  <span style="font-size:22px;">⚠️</span>
  <div><div>Требует ручной проверки</div>{reasons_html}</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="verdict-ok">
  <span style="font-size:22px;">✅</span>
  <div>Автопроверка пройдена — критических нарушений не выявлено</div>
</div>""",
            unsafe_allow_html=True,
        )


def render_checklist(script_details: list[str]) -> None:
    items_html = []
    for item in script_details:
        label, _, val = item.rpartition(":")
        ok = val.strip().lower().startswith("да")
        css = "ok" if ok else "fail"
        icon = "✅" if ok else "❌"
        items_html.append(
            f'<div class="checklist-item {css}">'
            f'<span class="checklist-icon">{icon}</span>{label.strip()}'
            f"</div>"
        )
    st.markdown("\n".join(items_html), unsafe_allow_html=True)


def render_feedback(positives: list[str], negatives: list[str]) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">💪 Сильные стороны</div>', unsafe_allow_html=True)
        if positives:
            for item in positives:
                st.markdown(
                    f'<div class="feedback-item pos">✦ {item}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="feedback-item pos">Нарушений не выявлено</div>',
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown('<div class="section-title">⚠️ Замечания</div>', unsafe_allow_html=True)
        if negatives:
            for item in negatives:
                st.markdown(
                    f'<div class="feedback-item neg">→ {item}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="feedback-item neg" style="background:#f0fdf4;border-color:#22c55e;color:#166534;">'
                "Замечаний нет</div>",
                unsafe_allow_html=True,
            )


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_transcript(role_text: str) -> None:
    if not role_text or role_text.strip() == "—":
        st.info("Транскрипт не сформирован.")
        return
    lines_html: list[str] = []
    for raw_line in role_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Оператор:"):
            text = line[len("Оператор:") :].strip()
            lines_html.append(
                '<div class="tx-line">'
                '<span class="tx-badge op">Оператор</span>'
                f'<span class="tx-text">{_esc(text)}</span>'
                "</div>"
            )
        elif line.startswith("Заявитель:"):
            text = line[len("Заявитель:") :].strip()
            lines_html.append(
                '<div class="tx-line">'
                '<span class="tx-badge ap">Заявитель</span>'
                f'<span class="tx-text">{_esc(text)}</span>'
                "</div>"
            )
        else:
            lines_html.append(
                '<div class="tx-line"><span class="tx-text" style="color:#9ca3af;">'
                f"{_esc(line)}</span></div>"
            )
    st.markdown(
        '<div class="transcript-wrap">' + "\n".join(lines_html) + "</div>",
        unsafe_allow_html=True,
    )
