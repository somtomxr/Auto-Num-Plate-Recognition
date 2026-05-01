from plate_utils import clean_plate, format_plate, is_valid_indian_plate


def test_valid_standard_plate_formats():
    assert is_valid_indian_plate("MH12AB1234")
    assert is_valid_indian_plate("MH12A1234")
    assert is_valid_indian_plate("MH12ABC1234")


def test_valid_bh_series_formatting():
    plate = clean_plate("22BH1234AA")

    assert plate == "22BH1234AA"
    assert is_valid_indian_plate(plate)
    assert format_plate(plate) == "22 BH 1234 AA"


def test_noise_cleanup_strips_ind_and_extra_edges():
    assert clean_plate("IND MH 12 AB 1234") == "MH12AB1234"
    assert clean_plate("1MH12AB1234") == "MH12AB1234"
    assert clean_plate("MH12AB12345") == "MH12AB1234"


def test_invalid_state_code_is_rejected():
    assert not is_valid_indian_plate("ZZ12AB1234")
    assert not is_valid_indian_plate("MX12AB1234")


def test_format_standard_plate():
    assert format_plate("MH12AB1234") == "MH 12 AB 1234"
