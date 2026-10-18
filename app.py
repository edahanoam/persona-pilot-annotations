from __future__ import annotations

import streamlit as st

from opinionqa import assign_opinion_questions, load_opinion_assignments
from persona_display import demographics_box, qualitative_box
from storage import (
    StorageError,
    load_completed_assignment_ids,
    load_completed_question_ids,
    save_response,
)
from twin2k import assign_persona, bfi44_questions, load_twin_personas


st.set_page_config(
    page_title="Persona Pilot Annotations",
    page_icon="📝",
    layout="centered",
)


def initialize_session() -> None:
    defaults = {
        "participant_id": "",
        "experiment": "",
        "stage": "welcome",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def restart() -> None:
    for key in ("participant_id", "experiment", "stage"):
        st.session_state.pop(key, None)
    st.rerun()


@st.cache_resource
def get_twin_personas() -> list[dict[str, str]]:
    return list(load_twin_personas())


@st.cache_data
def get_opinion_assignments() -> list[dict[str, object]]:
    return list(load_opinion_assignments())


def show_twin2k() -> None:
    try:
        personas = get_twin_personas()
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"The Twin-2K personas could not be loaded: {exc}")
        st.stop()

    persona, sampling_strategy = assign_persona(
        st.session_state.participant_id, personas  # type: ignore[arg-type]
    )
    questions = bfi44_questions()
    try:
        completed = load_completed_question_ids(
            st.session_state.participant_id,
            experiment="twin2k",
            persona_id=persona["persona_id"],
        )
    except StorageError as exc:
        st.error(f"Your saved progress could not be loaded. Please try again. ({exc})")
        st.stop()

    if completed and st.session_state.stage != "complete":
        st.session_state.stage = "bfi44"

    if st.session_state.stage == "welcome":
        st.subheader("Welcome")
        st.write(
            "You will first read a persona in two parts. You will then answer "
            "44 statements as you think that person would answer them."
        )
        if st.button("Begin", type="primary", use_container_width=True):
            st.session_state.stage = "demographics"
            st.rerun()
        return

    if st.session_state.stage in ("demographics", "qualitative"):
        st.progress(0.0, text="Persona information")
        st.subheader("Persona")
        st.markdown("**Demographics**")
        st.markdown(demographics_box(persona["demographics"]), unsafe_allow_html=True)
        st.markdown("**Qualitative responses**")
        st.markdown(qualitative_box(persona["qualitative"]), unsafe_allow_html=True)
        if st.button("Begin the BFI-44", type="primary", use_container_width=True):
            st.session_state.stage = "bfi44"
            st.rerun()
        return

    unanswered = [question for question in questions if question["id"] not in completed]
    if not unanswered:
        st.session_state.stage = "complete"
        st.success("Thank you — you have completed all 44 questions.")
        st.write("Your responses have been saved.")
        if st.button("Start a new response session", use_container_width=True):
            restart()
        return

    question = unanswered[0]
    position = len(completed) + 1
    st.progress(position / len(questions), text=f"Question {position} of {len(questions)}")
    st.write("Answer as you think the persona would answer.")
    with st.expander("Review the full persona", expanded=position == 1):
        st.markdown("**Demographics**")
        st.markdown(demographics_box(persona["demographics"]), unsafe_allow_html=True)
        st.markdown("**This person answers these 3 questions about himself**")
        st.markdown(qualitative_box(persona["qualitative"]), unsafe_allow_html=True)
    st.markdown("**I see myself as someone who…**")
    item = str(question["question"])
    st.write(item[0].lower() + item[1:])

    choices = question["choices"]
    with st.form(f"response-{question['id']}", clear_on_submit=True):
        answer = st.radio(
            "Your answer",
            options=list(choices),  # type: ignore[arg-type]
            format_func=lambda value: f"{value} — {choices[value]}",  # type: ignore[index]
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
                    question=question,  # type: ignore[arg-type]
                    answer=str(answer),
                    sequence=position,
                    experiment="twin2k",
                    persona_id=persona["persona_id"],
                    sampling_strategy=sampling_strategy,
                    choices=[choices[value] for value in choices],  # type: ignore[index]
                )
            except (OSError, StorageError) as exc:
                st.error(f"Your answer could not be saved. Please try again. ({exc})")
            else:
                st.rerun()


def show_opinionqa() -> None:
    try:
        assignments = get_opinion_assignments()
    except (OSError, ValueError) as exc:
        st.error(f"The OpinionQA sample could not be loaded: {exc}")
        st.stop()

    assigned_questions, sampling_strategy = assign_opinion_questions(
        st.session_state.participant_id,
        assignments,  # type: ignore[arg-type]
        question_count=10,
    )
    try:
        completed = load_completed_assignment_ids(
            st.session_state.participant_id,
            experiment="opinionqa",
        )
    except StorageError as exc:
        st.error(f"Your saved progress could not be loaded. Please try again. ({exc})")
        st.stop()

    unanswered = [
        assignment
        for assignment in assigned_questions
        if assignment["assignment_id"] not in completed
    ]
    if not unanswered:
        st.success("Thank you — you have completed all 10 questions.")
        st.write("Your responses have been saved.")
        if st.button("Start a new response session", use_container_width=True):
            restart()
        return

    if st.session_state.stage == "welcome":
        st.subheader("Welcome")
        st.write(
            "You will be given 10 persona and survey-question pairs. For each "
            "question, answer as you think the person described would answer."
        )
        if st.button("Begin", type="primary", use_container_width=True):
            st.session_state.stage = "opinion_question"
            st.rerun()
        return

    assignment = unanswered[0]
    position = len(assigned_questions) - len(unanswered) + 1
    st.progress(
        position / len(assigned_questions),
        text=f"Question {position} of {len(assigned_questions)}",
    )
    st.markdown("**Persona:**")
    st.write(assignment["persona"])
    st.markdown("**Question:**")
    st.write(assignment["question"])

    choices = assignment["choices"]
    with st.form(f"opinion-response-{assignment['assignment_id']}"):
        answer = st.radio(
            "Your answer",
            options=choices,
            index=None,
        )
        submitted = st.form_submit_button(
            "Submit answer", type="primary", use_container_width=True
        )

    if submitted:
        if answer is None:
            st.warning("Please select an answer before submitting.")
        else:
            question = {
                "id": assignment["question_id"],
                "question": assignment["question"],
                "description": "OpinionQA survey question",
            }
            try:
                save_response(
                    participant_id=st.session_state.participant_id,
                    question=question,
                    answer=str(answer),
                    sequence=position,
                    experiment="opinionqa",
                    persona_id=assignment["persona_id"],
                    sampling_strategy=sampling_strategy,
                    assignment_id=assignment["assignment_id"],
                    choices=list(choices),
                )
            except (OSError, StorageError) as exc:
                st.error(f"Your answer could not be saved. Please try again. ({exc})")
            else:
                st.rerun()


initialize_session()
st.title("Persona Pilot Annotations")

if not st.session_state.participant_id:
    st.subheader("Participant ID")
    st.write("Enter your assigned participant ID to continue.")
    with st.form("participant-details"):
        participant_id = st.text_input(
            "Participant ID", placeholder="Enter your assigned ID", max_chars=100
        )
        begin = st.form_submit_button("Begin", type="primary", use_container_width=True)
    if begin:
        cleaned_id = participant_id.strip()
        if not cleaned_id:
            st.warning("Please enter your participant ID before beginning.")
        elif cleaned_id.startswith("0"):
            st.session_state.participant_id = cleaned_id
            st.session_state.experiment = "twin2k"
            st.rerun()
        elif cleaned_id.startswith("1"):
            st.session_state.participant_id = cleaned_id
            st.session_state.experiment = "opinionqa"
            st.rerun()
        else:
            st.warning("The participant ID must begin with 0 or 1.")
    st.stop()

if st.session_state.experiment == "twin2k":
    show_twin2k()
elif st.session_state.experiment == "opinionqa":
    show_opinionqa()
else:
    st.error("This participant ID does not have an available experiment.")
