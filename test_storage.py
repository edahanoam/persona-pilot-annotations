import json
import tempfile
import unittest
from pathlib import Path

from storage import load_questions, save_response


class StorageTests(unittest.TestCase):
    def test_bundled_questions_are_valid(self):
        questions = load_questions()
        self.assertGreater(len(questions), 0)
        self.assertEqual(len({item["id"] for item in questions}), len(questions))

    def test_response_is_appended_as_jsonl(self):
        question = {
            "id": "example",
            "question": "Example question?",
            "description": "Example description.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            save_response(
                participant_id="participant-1",
                question=question,
                answer="First answer",
                sequence=1,
                path=path,
            )
            save_response(
                participant_id="participant-1",
                question=question,
                answer="Second answer",
                sequence=2,
                path=path,
            )

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["answer"] for record in records], ["First answer", "Second answer"])
            self.assertTrue(all(record["question_id"] == "example" for record in records))


if __name__ == "__main__":
    unittest.main()
