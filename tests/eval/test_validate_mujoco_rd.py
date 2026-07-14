from pathlib import Path

from scripts.eval.post.validate_mujoco_rd import _overall_ok, _static_xml_checks


def test_static_checks_cover_multi_joint_kin_agent_contract(tmp_path: Path):
    xml = tmp_path / "object.xml"
    xml.write_text(
        """<mujoco model="fixture">
  <compiler balanceinertia="true" inertiagrouprange="3 5" />
  <worldbody>
    <body name="object" quat="0.707106781 0.707106781 0 0">
      <geom name="body_visual" group="0" />
      <body name="knob">
        <joint name="knob_joint" type="hinge" axis="1 0 0" range="-0.5 0" />
        <geom name="knob_visual" group="0" />
        <geom name="knob_collision" type="box" group="3" />
      </body>
      <body name="drawer">
        <joint name="drawer_joint" type="slide" axis="0 0 1" range="0 0.3" />
        <geom name="drawer_visual" group="0" />
        <geom name="drawer_collision" type="box" group="3" />
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )

    report = _static_xml_checks(xml)

    assert report["xml_joint_count"] == 2
    assert report["slide_joint_count"] == 1
    assert report["slide_joint_axes"] == ["0 0 1"]
    assert report["moving_body_count"] == 2
    assert report["collision_group3_5_count"] == 2
    assert report["inertia_ready_body_names"] == ["knob", "drawer"]


def test_overall_ok_checks_every_slide_and_moving_mass():
    report = {
        "xml": {
            "floor_count": 0,
            "shell_count": 0,
            "missing_assets": [],
            "object_quat": "0.707106781 0.707106781 0 0",
            "compiler_balanceinertia": "true",
            "compiler_inertiagrouprange": "3 5",
            "visual_group0_count": 3,
            "collision_group3_5_count": 2,
            "moving_body_count": 2,
            "inertia_ready_body_names": ["knob", "drawer"],
            "glass_geom_count": 0,
        },
        "compile": {"available": False},
        "mujoco": {
            "nq": 2,
            "nv": 2,
            "nu": 0,
            "njnt": 2,
            "joint_body_masses": [0.02, 1.5],
            "slide_joint_checks": [{
                "local_z_ok": True,
                "positive_range_ok": True,
                "forward_negative_y_ok": True,
            }],
            "joint_motion_checks": [{"motion_ok": True}, {"motion_ok": True}],
            "has_drawer_glass": False,
            "render": {"ok": True},
        },
    }

    assert _overall_ok(report, expect_nq=2, expect_nv=2, expect_nu=0, expect_njnt=2)
    report["mujoco"]["slide_joint_checks"][0]["local_z_ok"] = False
    assert not _overall_ok(report, expect_nq=2, expect_nv=2, expect_nu=0, expect_njnt=2)
    report["mujoco"]["slide_joint_checks"][0]["local_z_ok"] = True
    report["mujoco"]["joint_motion_checks"][0]["motion_ok"] = False
    assert not _overall_ok(report, expect_nq=2, expect_nv=2, expect_nu=0, expect_njnt=2)
