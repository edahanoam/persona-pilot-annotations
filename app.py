from __future__ import annotations

import random
import uuid

import streamlit as st

from storage import load_questions, save_response


st.set_page_config(
    page_title="Persona Pilot Annotations",
    page_icon="📝",
    layout="centered",
)


def initialize_session(question_count: int) -> None:
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = str(uuid.uuid4())
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
    st.session_state.participant_id = str(uuid.uuid4())
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

if not st.session_state.started:
    st.subheader("Welcome")
    st.write(
        "You will be shown a series of prompts. Read each description, write your "
        "answer in the text box, and submit it before moving on."
    )
    st.info(
        "Your answers are saved as you submit them. Please avoid including names, "
        "contact details, or other sensitive personal information."
    )
    st.markdown(
        """
        **Instructions**

        1. Answer in your own words.
        2. Give enough detail for another person to understand your reasoning.
        3. Select **Submit answer** once you are satisfied with your response.
        """
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
st.subheader(question["question"])
st.caption(question["description"])

with st.form(f"response-{question['id']}", clear_on_submit=True):
    answer = st.text_area(
        "Your answer",
        height=180,
        placeholder="Type your response here…",
    )
    submitted = st.form_submit_button(
        "Submit answer", type="primary", use_container_width=True
    )

if submitted:
    cleaned_answer = answer.strip()
    if not cleaned_answer:
        st.warning("Please write an answer before submitting.")
    else:
        try:
            save_response(
                participant_id=st.session_state.participant_id,
                question=question,
                answer=cleaned_answer,
                sequence=position + 1,
            )
        except OSError as exc:
            st.error(f"Your answer could not be saved. Please try again. ({exc})")
        else:
            st.session_state.position += 1
            st.rerun()

