from collections import UserDict

from src.utils.config import get_nested, mapping_section


def test_mapping_section_returns_mapping_values_as_plain_dict():
    section = UserDict({"enabled": True})

    result = mapping_section({"training": section}, "training")

    assert result == {"enabled": True}
    assert isinstance(result, dict)


def test_mapping_section_returns_empty_dict_for_non_mapping_value():
    assert mapping_section({"training": "invalid"}, "training") == {}


def test_mapping_section_supports_nested_sections_with_get_nested():
    config = {"loss": {"diffusion": {"noise": {"std_max": 0.2}}}}
    diffusion = get_nested(config, "loss.diffusion")

    assert mapping_section(diffusion, "noise") == {"std_max": 0.2}
