"""Synthetic naming fixtures; no clinical identities or reports."""
import pytest
from backend.clinic_naming import canonical_filename

@pytest.mark.parametrize("digits", ["０１-０１-１９００", "٠١-٠١-١٩٠٠"])
def test_unicode_lookalike_is_description_not_canonical_id(digits):
    name = f"ZZ_{digits}__synthetic.pdf"
    assert canonical_filename(name, "ZZ_01-01-1900") == f"ZZ_01-01-1900__{name}"

def test_ascii_identity_is_removed_from_description():
    assert canonical_filename("ZZ_01-01-1900__synthetic.pdf", "ZZ_01-01-1900") == "ZZ_01-01-1900__synthetic.pdf"
