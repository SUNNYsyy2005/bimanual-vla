import numpy as np

from bimanual_vla.collection.quest.control import QuestController


def test_hold_retains_closing_command_instead_of_contact_opening():
    runtime = QuestController({"left": object()}, "unused")
    runtime.last_targets["left"] = np.r_[np.zeros(6), 0.]
    measured = np.r_[np.ones(6) * .1, .42]
    hold = runtime.held_target("left", measured)
    np.testing.assert_array_equal(hold[:6], measured[:6])
    assert hold[6] == 0
    assert measured[6] == .42
