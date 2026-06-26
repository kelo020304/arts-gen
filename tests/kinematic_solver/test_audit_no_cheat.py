from post_process.kinematic_solver.tools.audit_no_cheat import (
    audit_helper_file,
    audit_non_helper_file,
)


def test_non_helper_with_get_lower_limit_attr_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text("def x(j): return j.GetLowerLimitAttr().Get()\n")

    violations = audit_non_helper_file(path)

    assert violations
    assert any("GetLowerLimitAttr" in v for v in violations)


def test_non_helper_with_get_attribute_physics_lower_limit_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text("def x(j): return j.GetAttribute('physics:lowerLimit').Get()\n")

    violations = audit_non_helper_file(path)

    assert violations
    assert any("physics:lowerLimit" in v for v in violations)


def test_non_helper_with_getattr_concat_limit_name_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text("def x(j): return getattr(j, 'Get' + 'LowerLimitAttr')()\n")

    violations = audit_non_helper_file(path)

    assert violations
    assert any("GetLowerLimitAttr" in v for v in violations)


def test_non_helper_with_hasattr_name_alias_limit_name_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text(
        "def x(j):\n"
        "    name = 'Get' + 'UpperLimitAttr'\n"
        "    return hasattr(j, name)\n"
    )

    violations = audit_non_helper_file(path)

    assert violations
    assert any("GetUpperLimitAttr" in v for v in violations)


def test_non_helper_with_module_constant_limit_name_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text(
        "LIMIT = 'Get' + 'LowerLimitAttr'\n"
        "def x(j):\n"
        "    return getattr(j, LIMIT)()\n"
    )

    violations = audit_non_helper_file(path)

    assert violations
    assert any("GetLowerLimitAttr" in v for v in violations)


def test_non_helper_with_tuple_alias_limit_name_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text(
        "def x(j):\n"
        "    name, other = 'Get' + 'LowerLimitAttr', 'ok'\n"
        "    return getattr(j, name)()\n"
    )

    violations = audit_non_helper_file(path)

    assert violations
    assert any("GetLowerLimitAttr" in v for v in violations)


def test_non_helper_with_fstring_limit_name_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text("def x(j): return getattr(j, f\"Get{'Upper'}LimitAttr\")()\n")

    violations = audit_non_helper_file(path)

    assert violations
    assert any("GetUpperLimitAttr" in v for v in violations)


def test_non_helper_with_format_limit_name_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text("def x(j): return getattr(j, 'Get{}LimitAttr'.format('Lower'))()\n")

    violations = audit_non_helper_file(path)

    assert violations
    assert any("GetLowerLimitAttr" in v for v in violations)


def test_non_helper_with_join_limit_name_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text(
        "def x(j):\n"
        "    return getattr(j, ''.join(['Get', 'Upper', 'Limit', 'Attr']))()\n"
    )

    violations = audit_non_helper_file(path)

    assert violations
    assert any("GetUpperLimitAttr" in v for v in violations)


def test_non_helper_with_replace_limit_name_fails(tmp_path):
    path = tmp_path / "evil.py"
    path.write_text(
        "def x(j):\n"
        "    return getattr(j, 'GtLowerLimitAttr'.replace('Gt', 'Get'))()\n"
    )

    violations = audit_non_helper_file(path)

    assert violations
    assert any("GetLowerLimitAttr" in v for v in violations)


def test_non_helper_clean_file_passes(tmp_path):
    path = tmp_path / "clean.py"
    path.write_text("def x(j): return j\n")

    assert audit_non_helper_file(path) == []


def test_reader_helper_calling_set_fails(tmp_path):
    path = tmp_path / "usd_limit_reader.py"
    path.write_text("def x(j): j.GetLowerLimitAttr().Set(0.0)\n")

    violations = audit_helper_file(path, kind="reader")

    assert any("reader" in v for v in violations)


def test_writer_helper_calling_get_fails(tmp_path):
    path = tmp_path / "usd_limit_writer.py"
    path.write_text("def x(j): return j.GetLowerLimitAttr().Get()\n")

    violations = audit_helper_file(path, kind="writer")

    assert any("writer" in v for v in violations)


def test_reader_helper_only_get_passes(tmp_path):
    path = tmp_path / "usd_limit_reader.py"
    path.write_text("def x(j): return j.GetLowerLimitAttr().Get()\n")

    assert audit_helper_file(path, kind="reader") == []


def test_writer_helper_only_set_passes(tmp_path):
    path = tmp_path / "usd_limit_writer.py"
    path.write_text("def x(j): j.GetLowerLimitAttr().Set(0.0)\n")

    assert audit_helper_file(path, kind="writer") == []
