"""Safe HTML formatting helpers for Twin-2K persona sections."""

from __future__ import annotations

import html
import re


DEMOGRAPHIC_INTRO_LINES = {
    "the following is a description of a person.",
    "the person's demographics are the following...",
    "the person's demographics are the following:",
    "persona demographics",
}

DEMOGRAPHIC_LABELS = {
    "CREGION": "Region",
    "AGE": "Age",
    "SEX": "Sex",
    "EDUCATION": "Education",
    "CITIZEN": "Citizenship",
    "MARITAL": "Marital status",
    "RELIG": "Religion",
    "RELIGATTEND": "Religious service attendance",
    "POLPARTY": "Political party",
    "INCOME": "Income",
    "POLIDEOLOGY": "Political ideology",
    "RACE": "Race",
}


def demographic_items(text: str) -> list[str]:
    """Return one clean display item per demographic field."""
    raw_parts = re.split(
        r"[\r\n]+|\s*•\s*|,\s*(?=[A-Z][A-Z0-9_]*\s*:)",
        text,
    )
    items: list[str] = []
    for part in raw_parts:
        cleaned = part.strip().lstrip("-*• ").strip()
        if not cleaned or cleaned.casefold() in DEMOGRAPHIC_INTRO_LINES:
            continue
        items.append(cleaned)
    return items or [text.strip()]


def qualitative_items(text: str) -> list[tuple[str, str]]:
    """Extract the three standard qualitative questions and their answers."""
    marker = "The person also answered three purely qualitative questions about their concept of self."
    content = text.strip()
    if content.casefold().startswith(marker.casefold()):
        content = content[len(marker) :].strip()

    segments = re.split(
        r"(?=The person was(?: then| finally)? asked(?: the question)?(?:,|\s))",
        content,
        flags=re.IGNORECASE,
    )
    items: list[tuple[str, str]] = []
    for segment in segments:
        if "They answered:" not in segment:
            continue
        question, answer = segment.split("They answered:", 1)
        question = re.sub(
            r"^The person was(?: then| finally)? asked(?: the question)?(?:,|\s)*",
            "",
            question.strip(),
            flags=re.IGNORECASE,
        )
        question = question.strip().strip('"“”¨ ').strip()
        answer = answer.strip().strip('"“” ').strip()
        question = re.sub(r"\s+", " ", question)
        answer = re.sub(r"\s+", " ", answer)
        if question and answer:
            items.append((question, answer))
    return items


def demographics_box(text: str) -> str:
    rendered_lines: list[str] = []
    for item in demographic_items(text):
        if ":" in item:
            label, value = item.split(":", 1)
            label = label.strip()
            display_label = DEMOGRAPHIC_LABELS.get(label.upper(), label)
            content = (
                f"<strong>{html.escape(display_label)}:</strong> "
                f"{html.escape(value.strip())}"
            )
        else:
            content = html.escape(item)
        rendered_lines.append(
            f'<div style="margin:0.35rem 0;">{content}</div>'
        )
    lines = "".join(rendered_lines)
    return (
        '<div style="background:#eaf4ff;border:1px solid #a9cfee;'
        'border-radius:12px;padding:0.9rem 1.1rem;color:#17324d;">'
        f"{lines}</div>"
    )


def qualitative_box(text: str) -> str:
    items = qualitative_items(text)
    if items:
        body = "".join(
            '<div style="margin-bottom:1rem;">'
            f'<div style="font-weight:700;margin-bottom:0.3rem;">{html.escape(question)}</div>'
            f'<div style="font-weight:400;line-height:1.5;">{html.escape(answer)}</div>'
            "</div>"
            for question, answer in items
        )
    else:
        body = (
            '<div style="font-weight:400;line-height:1.5;">'
            f"{html.escape(text.strip())}</div>"
        )
    return (
        '<div style="background:#fff3e5;border:1px solid #edc28d;'
        'border-radius:12px;padding:1rem 1.1rem;color:#4a2b0b;">'
        f"{body}</div>"
    )
