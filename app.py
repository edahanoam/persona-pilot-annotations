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
from twin2k import (
    PILOT_BFI44_NUMBERS,
    PersonaCondition,
    TwinPersona,
    assign_persona_rounds,
    bfi44_questions,
    load_twin_personas,
)


st.set_page_config(
    page_title="Persona Pilot Annotations",
    page_icon="📝",
    layout="wide",
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


def show_answering_instruction(*, green: bool = False) -> None:
    message = (
        "**Instructions:** Answer each question as you think the person described "
        "in the persona would answer. Do not answer as yourself."
    )
    if green:
        st.success(message)
    else:
        st.info(message)


def show_persona_definition(
    persona: TwinPersona, condition: PersonaCondition
) -> None:
    if condition in ("demographics", "combined"):
        st.markdown("**Demographics**")
        st.markdown(demographics_box(persona["demographics"]), unsafe_allow_html=True)
    if condition in ("qualitative", "combined"):
        st.markdown("**How this person described themselves**")
        st.markdown(qualitative_box(persona["qualitative"]), unsafe_allow_html=True)


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

    rounds, sampling_strategy = assign_persona_rounds(
        st.session_state.participant_id, personas  # type: ignore[arg-type]
    )
    questions = bfi44_questions(PILOT_BFI44_NUMBERS)
    required_question_ids = {str(question["id"]) for question in questions}

    if st.session_state.stage == "welcome":
        st.subheader("Welcome")
        st.write(
            "You will meet three different personas. For each one, read the "
            "information provided and answer the personality questions from that "
            "person's point of view."
        )
        if st.button("Begin", type="primary", use_container_width=True):
            st.session_state.stage = "persona_intro"
            st.rerun()
        return

    completed_by_round: list[set[str]] = []
    try:
        for round_ in rounds:
            completed_by_round.append(
                load_completed_question_ids(
                    st.session_state.participant_id,
                    experiment="twin2k",
                    persona_id=round_["persona"]["persona_id"],
                )
                & required_question_ids
            )
    except StorageError as exc:
        st.error(f"Your saved progress could not be loaded. Please try again. ({exc})")
        st.stop()

    active_round_index = next(
        (
            index
            for index, completed in enumerate(completed_by_round)
            if len(completed) < len(questions)
        ),
        None,
    )
    if active_round_index is None:
        st.session_state.stage = "complete"
        st.success("Thank you — you have completed all three persona rounds.")
        st.write("Your responses have been saved.")
        if st.button("Start a new response session", use_container_width=True):
            restart()
        return

    active_round = rounds[active_round_index]
    persona = active_round["persona"]
    condition = active_round["condition"]
    completed = completed_by_round[active_round_index]
    expected_question_stage = f"twin_questions_{active_round_index}"

    if st.session_state.stage != expected_question_stage:
        st.progress(
            active_round_index / len(rounds),
            text=f"Persona {active_round_index + 1} of {len(rounds)}",
        )
        st.subheader(f"Meet persona {active_round_index + 1}")
        st.success(
            "This is a new person. Read this persona definition carefully before "
            "continuing. Your answers should describe this person only."
        )
        show_persona_definition(persona, condition)
        if st.button(
            "I understand — begin questions",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.stage = expected_question_stage
            st.rerun()
        return

    unanswered = [question for question in questions if question["id"] not in completed]

    # Keep the persona visible beside the complete questionnaire so participants
    # can refer to it without opening an expander or moving between pages.
    with st.sidebar:
        st.header(f"Persona {active_round_index + 1} of {len(rounds)}")
        show_persona_definition(persona, condition)

    st.subheader("Questions")
    st.progress(
        len(completed) / len(questions),
        text=f"{len(completed)} of {len(questions)} answers saved",
    )
    show_answering_instruction(green=True)
    st.caption("Answer every question, then submit the complete questionnaire once.")

    answers: dict[str, object] = {}
    with st.form("twin2k-questionnaire"):
        for question in unanswered:
            display_number = int(question["display_number"])
            item = str(question["question"])
            choices = question["choices"]
            st.markdown(f"**{display_number}. I see myself as someone who…**")
            st.write(item[0].lower() + item[1:])
            answers[str(question["id"])] = st.radio(
                f"Answer for question {display_number}",
                options=list(choices),  # type: ignore[arg-type]
                format_func=lambda value, labels=choices: f"{value} — {labels[value]}",  # type: ignore[index]
                index=None,
                key=f"answer-{question['id']}",
                label_visibility="collapsed",
            )
            st.divider()

        submitted = st.form_submit_button(
            "Submit all answers", type="primary", use_container_width=True
        )

    if submitted:
        missing = [
            int(question["display_number"])
            for question in unanswered
            if answers[str(question["id"])] is None
        ]
        if missing:
            missing_text = ", ".join(str(number) for number in missing)
            st.warning(
                "Please answer every question before submitting. "
                f"Missing: {missing_text}."
            )
        else:
            try:
                for question in unanswered:
                    choices = question["choices"]
                    save_response(
                        participant_id=st.session_state.participant_id,
                        question=question,  # type: ignore[arg-type]
                        answer=str(answers[str(question["id"])]),
                        sequence=(active_round_index * len(questions))
                        + int(question["display_number"]),
                        experiment="twin2k",
                        persona_id=persona["persona_id"],
                        sampling_strategy=sampling_strategy,
                        assignment_id=(
                            f"persona-{active_round_index + 1}-{condition}"
                        ),
                        choices=[choices[value] for value in choices],  # type: ignore[index]
                    )
            except (OSError, StorageError) as exc:
                st.error(f"Your answers could not all be saved. Please try again. ({exc})")
            else:
                st.session_state.stage = "persona_intro"
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
            "You will be presented with a persona description and a question. "
            "Answer the question as you think that person would answer."
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
    show_answering_instruction(green=True)
    st.markdown("**Persona:**")
    st.markdown(demographics_box(assignment["persona"]), unsafe_allow_html=True)
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
