from __future__ import annotations

import random

import streamlit as st

from storage import load_questions, save_response


st.set_page_config(
    page_title="Persona Pilot Annotations",
    page_icon="📝",
    layout="centered",
)


def initialize_session(question_count: int) -> None:
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = ""
    if "started" not in st.session_state:
        st.session_state.started = False
    if "question_order" not in st.session_state:
        order = list(range(question_count))
        random.SystemRandom().shuffle(order)
        st.session_state.question_order = order
        st.session_state.position = 0


def restart(question_count: int) -> None:
    order = list(range(question_count))
    random.SystemRandom().shuffle(order)
    st.session_state.participant_id = ""
    st.session_state.question_order = order
    st.session_state.position = 0
    st.session_state.started = False


@st.cache_data
def get_questions() -> list[dict[str, str]]:
    return load_questions()


try:
    questions = get_questions()
except (OSError, ValueError) as exc:
    st.error(f"The question set could not be loaded: {exc}")
    st.stop()

initialize_session(len(questions))

st.title("Persona Pilot Annotations")

if not st.session_state.participant_id:
    st.subheader("Participant ID")
    st.write("Enter your assigned participant ID to continue.")
    with st.form("participant-details"):
        participant_id = st.text_input(
            "Participant ID",
            placeholder="Enter your assigned ID",
            max_chars=100,
        )
        begin = st.form_submit_button(
            "Begin", type="primary", use_container_width=True
        )

    if begin:
        cleaned_id = participant_id.strip()
        if not cleaned_id:
            st.warning("Please enter your participant ID before beginning.")
        else:
            st.session_state.participant_id = cleaned_id
            st.rerun()
    st.stop()

if not st.session_state.started:
    st.subheader("Welcome")
    st.write(
        "You will be given a persona description and a question. Answer as you "
        "think the given persona would."
    )
    if st.button("Begin", type="primary", use_container_width=True):
        st.session_state.started = True
        st.rerun()
    st.stop()

position = st.session_state.position
order = st.session_state.question_order

if position >= len(order):
    st.success("Thank you — you have completed every question.")
    st.write("Your responses have been saved.")
    if st.button("Start a new response session", use_container_width=True):
        restart(len(questions))
        st.rerun()
    st.stop()

question = questions[order[position]]
st.progress(position / len(order), text=f"Question {position + 1} of {len(order)}")
st.write("Answer the following question as the persona.")
st.markdown("**Persona:**")
st.write("[persona placeholder]")
st.markdown("**Question:**")
st.write("[question placeholder]")

with st.form(f"response-{question['id']}", clear_on_submit=True):
    scale_labels = {
        1: "Strongly disagree",
        2: "Disagree",
        3: "Neither agree nor disagree",
        4: "Agree",
        5: "Strongly agree",
    }
    answer = st.radio(
        "Your answer",
        options=list(scale_labels),
        format_func=lambda value: f"{value} — {scale_labels[value]}",
        index=None,
    )
    submitted = st.form_submit_button(
        "Submit answer", type="primary", use_container_width=True
    )

if submitted:
    if answer is None:
        st.warning("Please select an answer before submitting.")
    else:
        try:
            save_response(
                participant_id=st.session_state.participant_id,
                question=question,
                answer=str(answer),
                sequence=position + 1,
            )
        except OSError as exc:
            st.error(f"Your answer could not be saved. Please try again. ({exc})")
        else:
            st.session_state.position += 1
            st.rerun()
