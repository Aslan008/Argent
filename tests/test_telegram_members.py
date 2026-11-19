"""The roster: who the bot has actually seen, and how honest it is about the limit.

The Bot API has no "list every member" call, so this module can only know people it has
watched write or join. The tests hold both halves: it remembers what it saw, and it never
pretends to have seen more.
"""

from src.telegram import members


def _user(uid=7, first="Иван", last="", username=""):
    return {"id": uid, "first_name": first, "last_name": last, "username": username}


def test_a_writer_is_remembered():
    roster = members.Roster()
    member = roster.note(-100, _user(username="ivan"))

    assert member is not None
    assert roster.count(-100) == 1
    assert member.name == "Иван"
    assert member.username == "ivan"
    assert member.handle == "@ivan"


def test_messages_are_counted_but_a_join_is_not():
    roster = members.Roster()
    roster.note(-100, _user())
    roster.note(-100, _user())
    roster.note(-100, _user(), count_message=False)

    assert roster.get(-100, 7).messages == 2


def test_bots_and_anonymous_admins_are_skipped():
    roster = members.Roster()
    assert roster.note(-100, {"id": 9, "is_bot": True, "first_name": "Bot"}) is None
    assert roster.note(-100, {}) is None
    assert roster.count(-100) == 0


def test_a_renamed_person_is_updated_not_duplicated():
    roster = members.Roster()
    roster.note(-100, _user(first="Иван"))
    roster.note(-100, _user(first="Иоанн"))

    assert roster.count(-100) == 1
    assert roster.get(-100, 7).name == "Иоанн"


def test_admins_are_marked():
    roster = members.Roster()
    roster.note(-100, _user())
    roster.mark_admin(-100, _user())

    assert roster.get(-100, 7).is_admin is True


def test_the_list_is_freshest_first():
    roster = members.Roster()
    roster.note(-100, _user(1, "Первый"), at=1000)
    roster.note(-100, _user(2, "Второй"), at=2000)
    roster.note(-100, _user(2, "Второй"), at=3000)

    assert [m.name for m in roster.known(-100)] == ["Второй", "Первый"]


def test_an_empty_roster_says_why_it_is_empty():
    text = members.roster_text([])

    assert "ничего" in text.lower() or "никого" in text.lower()
    assert "не отдаёт" in text


def test_the_list_states_the_limit():
    roster = members.Roster()
    roster.note(-100, _user())
    text = members.roster_text(roster.known(-100), total=25)

    assert "Иван" in text
    assert "из 25" in text
    assert "не отдаёт" in text, "the honest limit must be visible, not implied"


def test_an_unknown_person_is_admitted_not_invented():
    text = members.member_text(None)

    assert "не знаю" in text
    assert "id" not in text.lower() or "не даёт" in text
