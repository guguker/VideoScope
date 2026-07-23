from videoscope.search.query_router import QueryRouter


def test_routes_proper_name_to_textual_evidence() -> None:
    plan = QueryRouter().route("Мозгов")

    assert plan.intent == "entity"
    assert plan.modalities == frozenset({"speech", "ocr"})
    assert plan.use_lighthouse is False
    assert plan.refine_temporally is False


def test_routes_action_to_visual_temporal_search() -> None:
    plan = QueryRouter().route("игрок поднимает руку", requested_lighthouse=True)

    assert plan.intent == "action"
    assert {"visual", "objects", "lighthouse"} <= plan.modalities
    assert plan.use_lighthouse is True
    assert plan.refine_temporally is True


def test_routes_screen_text_to_ocr() -> None:
    plan = QueryRouter().route("на табло написано 87:86")

    assert plan.intent == "ocr"
    assert plan.modalities == frozenset({"ocr"})


def test_routes_object_description_without_expensive_lighthouse() -> None:
    plan = QueryRouter().route("человек с чашкой", requested_lighthouse=True)

    assert plan.intent == "object"
    assert plan.modalities == frozenset({"objects", "visual"})
    assert plan.use_lighthouse is False


def test_routes_mixed_speech_and_action_to_multiple_channels() -> None:
    plan = QueryRouter().route("игрок говорит тайм-аут и поднимает руку", requested_lighthouse=True)

    assert plan.intent == "mixed"
    assert {"speech", "visual", "objects", "lighthouse"} <= plan.modalities


def test_explicit_mode_overrides_automatic_routing() -> None:
    plan = QueryRouter().route("игрок поднимает руку", mode="speech", requested_lighthouse=True)

    assert plan.intent == "speech"
    assert plan.modalities == frozenset({"speech"})
    assert plan.use_lighthouse is False

