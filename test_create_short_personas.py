import unittest

from create_short_personas import create_short_versions


class CreateShortVersionsTests(unittest.TestCase):
    def test_extracts_demographics_and_qualitative_answers(self):
        summary = (
            "Persona Demographics\nAge: 31\n\n"
            "The person's Big 5 scores are the following:\nOpenness: 4\n\n"
            "Other psychological material.\n\n"
            "The person also answered three purely qualitative questions about "
            "their concept of self:\nAnswer one.\nAnswer two.\nAnswer three."
        )

        result = create_short_versions(summary)

        self.assertEqual(
            result["persona_demographics"],
            "Persona Demographics\nAge: 31",
        )
        self.assertEqual(
            result["persona_demographics_and_qualitative"],
            "Persona Demographics\nAge: 31\n\n"
            "The person also answered three purely qualitative questions about "
            "their concept of self:\nAnswer one.\nAnswer two.\nAnswer three.",
        )

    def test_missing_marker_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Could not find expected marker"):
            create_short_versions("Persona Demographics only")


if __name__ == "__main__":
    unittest.main()
