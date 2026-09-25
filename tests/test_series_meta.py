from knee.data.series_meta import contrast_class, fat_sat_from_header, normalize_manufacturer, scanner_fingerprint


def test_contrast_class_from_tr_te():
    assert contrast_class(500, 12, None, "SE", None) == "T1"
    assert contrast_class(3000, 30, None, "SE", None) == "PD"
    assert contrast_class(4000, 80, None, "SE", None) == "T2"
    assert contrast_class(4000, 40, 150, "IR", None) == "STIR"
    assert contrast_class(20, 5, None, "GR", None) == "GRE"


def test_contrast_class_falls_back_to_description():
    # Toshiba files in the test stub carry no TR/TE.
    assert contrast_class(None, None, None, None, "SG PD FatSat") == "PD"
    assert contrast_class(None, None, None, None, "CO T2W FS") == "T2"
    assert contrast_class(None, None, None, None, "SG T1W") == "T1"
    assert contrast_class(None, None, None, None, "localizer") == "UNK"


def test_fat_sat_tokens_are_exact():
    assert fat_sat_from_header("FS\\SAT", None)
    assert not fat_sat_from_header("SAT_GEMS", None)          # GE spatial saturation, not fat-sat
    assert fat_sat_from_header(None, "t2_de3d_we_tra")         # water excitation hides between underscores
    assert fat_sat_from_header(None, "SG PD FatSat")
    assert not fat_sat_from_header(None, "pd_tse_sag_d")


def test_manufacturer_normalization_and_fingerprint():
    assert normalize_manufacturer("Siemens Healthineers") == "Siemens"
    assert normalize_manufacturer("GE MEDICAL SYSTEMS") == "GE"
    assert normalize_manufacturer("GEHC") == "GE"
    assert normalize_manufacturer("Philips Medical Systems") == "Philips"
    assert normalize_manufacturer(None) == "Unknown"
    assert scanner_fingerprint("TOSHIBA", "Vantage", None, None) == "Toshiba|Vantage|?T|?"
    assert scanner_fingerprint("SIEMENS", "Skyra", "3", "VE11") == "Siemens|Skyra|3.0T|VE11"
