from __future__ import annotations

import streamlit as st

from opinionqa import assign_opinion_question, load_opinion_assignments
from storage import load_completed_question_ids, save_response
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


@st.cache_data
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
    completed = load_completed_question_ids(
        st.session_state.participant_id,
        experiment="twin2k",
        persona_id=persona["persona_id"],
    )

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

    if st.session_state.stage == "demographics":
        st.progress(0.0, text="Persona information: demographics")
        st.subheader("Part 1: Demographics")
        st.write(persona["demographics"])
        if st.button(
            "Continue to qualitative responses", type="primary", use_container_width=True
        ):
            st.session_state.stage = "qualitative"
            st.rerun()
        return

    if st.session_state.stage == "qualitative":
        st.progress(0.0, text="Persona information: qualitative responses")
        st.subheader("Part 2: Qualitative responses")
        st.write(persona["qualitative"])
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
        st.write(persona["combined"])
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
                )
            except OSError as exc:
                st.error(f"Your answer could not be saved. Please try again. ({exc})")
            else:
                st.rerun()


def show_opinionqa() -> None:
    try:
        assignments = get_opinion_assignments()
    except (OSError, ValueError) as exc:
        st.error(f"The OpinionQA sample could not be loaded: {exc}")
        st.stop()

    assignment, sampling_strategy = assign_opinion_question(
        st.session_state.participant_id, assignments  # type: ignore[arg-type]
    )
    completed = load_completed_question_ids(
        st.session_state.participant_id,
        experiment="opinionqa",
        persona_id=assignment["persona_id"],
    )

    if assignment["question_id"] in completed:
        st.success("Thank you — you have completed your question.")
        st.write("Your response has been saved.")
        if st.button("Start a new response session", use_container_width=True):
            restart()
        return

    if st.session_state.stage == "welcome":
        st.subheader("Welcome")
        st.write(
            "You will be given a persona and one survey question. Answer as you "
            "think the person described would answer."
        )
        if st.button("Begin", type="primary", use_container_width=True):
            st.session_state.stage = "opinion_question"
            st.rerun()
        return

    st.progress(0.0, text="OpinionQA question")
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
                    sequence=1,
                    experiment="opinionqa",
                    persona_id=assignment["persona_id"],
                    sampling_strategy=sampling_strategy,
                    assignment_id=assignment["assignment_id"],
                    choices=list(choices),
                )
            except OSError as exc:
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
