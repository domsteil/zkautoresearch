from __future__ import annotations

import asyncio
import unittest

from prepare import (
    FixedReferenceReward,
    LOCKED_SELECTION_METRIC_NAME,
    _bootstrap_selection_means,
    _selection_topic_metric_name,
)


class FixedReferenceRewardTests(unittest.TestCase):
    def test_reference_reward_scores_grounded_response_high(self) -> None:
        reward = FixedReferenceReward()
        score = asyncio.run(
            reward.compute_reward(
                [
                    {
                        "role": "assistant",
                        "content": (
                            "I can help with the refund and send a return label with the next steps today."
                        ),
                    }
                ],
                {
                    "reference_response": (
                        "I can help start a return or refund and share the next steps for sending it back."
                    ),
                    "required_concepts": [["refund"], ["return label", "label"], ["next steps"]],
                },
            )
        )

        self.assertGreater(score, 0.7)

    def test_reference_reward_penalizes_missing_required_concepts(self) -> None:
        reward = FixedReferenceReward()
        score = asyncio.run(
            reward.compute_reward(
                [
                    {
                        "role": "assistant",
                        "content": "Thanks for reaching out. Let me know if there is anything else.",
                    }
                ],
                {
                    "reference_response": "I can help with the refund and return instructions.",
                    "required_concepts": [["refund"], ["return"], ["instructions"]],
                },
            )
        )

        self.assertLess(score, 0.35)

    def test_locked_metric_name_is_stable(self) -> None:
        self.assertEqual(LOCKED_SELECTION_METRIC_NAME, "selection_score")

    def test_bootstrap_means_are_stable_for_constant_scores(self) -> None:
        means = _bootstrap_selection_means([0.75, 0.75, 0.75])

        self.assertTrue(means)
        self.assertTrue(all(mean == 0.75 for mean in means))

    def test_selection_topic_metric_name_is_sanitized(self) -> None:
        self.assertEqual(
            _selection_topic_metric_name("Technical Support / Android"),
            "selection_topic_technical_support_android",
        )


if __name__ == "__main__":
    unittest.main()
