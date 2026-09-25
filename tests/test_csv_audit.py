from knee.audit.csv_audit import guess_language, report_hash


def test_report_hash_ignores_case_whitespace_punctuation():
    assert report_hash("Effusion.  Normal ACL") == report_hash("effusion normal acl")
    assert report_hash("Effusion") != report_hash("No effusion")


def test_guess_language():
    assert guess_language("There is a tear of the medial meniscus with no effusion and the ACL is normal.") == "en"
    assert guess_language("Hay una lesión en el menisco de la rodilla y no hay derrame en los tejidos.") == "es"
    assert guess_language("Ρήξη του έσω μηνίσκου.") == "greek"
    assert guess_language("") == "empty"
