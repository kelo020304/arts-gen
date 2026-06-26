import math
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from post_process.kinematic_solver.utils.usd_limit_reader import read_limits_from_source_usd
from post_process.kinematic_solver.utils.usd_limit_writer import write_predicted_limits


def test_usd_limit_reader_revolute_converts_degrees_to_radians():
    joint = MagicMock()
    joint.GetLowerLimitAttr.return_value.Get.return_value = -90.0
    joint.GetUpperLimitAttr.return_value.Get.return_value = 90.0

    lower, upper = read_limits_from_source_usd(joint, "revolute")

    assert lower == pytest.approx(-math.pi / 2)
    assert upper == pytest.approx(math.pi / 2)


def test_usd_limit_reader_prismatic_multiplies_meters_per_unit():
    joint = MagicMock()
    joint.GetLowerLimitAttr.return_value.Get.return_value = -18.0
    joint.GetUpperLimitAttr.return_value.Get.return_value = 18.0

    lower, upper = read_limits_from_source_usd(
        joint,
        "prismatic",
        meters_per_unit=0.01,
    )

    assert lower == pytest.approx(-0.18)
    assert upper == pytest.approx(0.18)


def test_usd_limit_writer_revolute_writes_degrees():
    joint = MagicMock()

    write_predicted_limits(
        joint,
        "revolute",
        pred_lower=-math.pi / 2,
        pred_upper=math.pi / 2,
        meters_per_unit=1.0,
    )

    assert joint.GetLowerLimitAttr.return_value.Set.call_args[0][0] == pytest.approx(-90.0)
    assert joint.GetUpperLimitAttr.return_value.Set.call_args[0][0] == pytest.approx(90.0)


def test_usd_limit_writer_prismatic_divides_meters_per_unit():
    joint = MagicMock()

    write_predicted_limits(
        joint,
        "prismatic",
        pred_lower=-0.18,
        pred_upper=0.18,
        meters_per_unit=0.01,
    )

    assert joint.GetLowerLimitAttr.return_value.Set.call_args[0][0] == pytest.approx(-18.0)
    assert joint.GetUpperLimitAttr.return_value.Set.call_args[0][0] == pytest.approx(18.0)


def test_usd_limit_helpers_reject_unknown_joint_type():
    with pytest.raises(ValueError):
        read_limits_from_source_usd(MagicMock(), "floating")
    with pytest.raises(ValueError):
        write_predicted_limits(MagicMock(), "floating", 0.0, 1.0, 1.0)


def test_usd_limit_writer_import_keeps_pxr_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import post_process.kinematic_solver.utils.usd_limit_writer; "
                "print('pxr' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
