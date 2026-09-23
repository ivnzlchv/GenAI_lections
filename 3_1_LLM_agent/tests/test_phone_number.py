import pytest

from llm_agent.tool_phone_number import PhoneNumberTool


@pytest.fixture
def tool():
    return PhoneNumberTool(default_region="RU")


def test_normalize_russian_number_to_e164(tool):
    assert tool.normalize("8 (912) 345-67-89") == "+79123456789"


def test_validate_international_number(tool):
    assert tool.validate("+1 415 555 2671") is True


def test_invalid_number_is_rejected(tool):
    assert tool.validate("123") is False
    with pytest.raises(ValueError, match="недействителен"):
        tool.normalize("123")


def test_analyze_returns_normalized_data(tool):
    result = tool.analyze("+44 20 7946 0958")

    assert result["is_possible"] is True
    assert result["is_valid"] is True
    assert result["e164"] == "+442079460958"
    assert result["region"] == "GB"


def test_use_handles_unparseable_input(tool):
    assert tool.use("not a phone").startswith("Ошибка:")


def test_use_extracts_number_from_tool_annotation(tool):
    result = tool.use("8 912 345-67-89 (без пробелов)")

    assert "действителен" in result
    assert "+79123456789" in result


def test_constructor_rejects_unknown_region():
    with pytest.raises(ValueError, match="Неподдерживаемый код региона"):
        PhoneNumberTool(default_region="XX")


def test_normalize_rejects_unknown_format(tool):
    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        tool.normalize("+7 912 345-67-89", output_format="CUSTOM")
