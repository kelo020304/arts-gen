"""Stable SDK helpers for the single-file estimate_limit agent artifact."""

from .agent_loop import (
    AgentLoopConfig,
    AgentLoopError,
    AgentLoopResult,
    apply_action_update,
    build_action_messages,
    extract_editable_response_code,
    extract_action_response_json,
    render_estimate_limits_from_report,
    resolve_agent_loop_config,
    run_agent_loop,
)
from .candidate_file import extract_editable_code, replace_editable_code
from .compiler import CandidateCompileError, compile_candidate_report
from .context_builder import build_context_from_roots
from .axis_candidates import (
    AxisCandidate,
    infer_axis_candidates_for_joint,
    with_axis_candidate_evidence,
)
from .axis_feedback import next_axis_from_feedback
from .live_viewer import (
    LiveViewer,
    context_details,
    editable_region_sha256,
    start_live_viewer_server,
)
from .motion_search import (
    AXIS_ACTIONS,
    AxisAction,
    AxisSearchResult,
    RangeSearchResult,
    refine_positive_limit,
    search_axis_actions,
)
from .motion_validation import (
    MotionSearchValidationResult,
    validate_motion_search,
    validate_motion_search_from_roots,
    validate_motion_samples,
    validate_motion_samples_from_roots,
)
from .object_visualization import write_estimate_viewers
from .mjcf_preview import write_iteration_mjcf_preview, write_rest_mjcf_preview
from .schemas import (
    CandidateReport,
    CompileSignal,
    CompileSignalBundle,
    EstimateContext,
    LimitEstimate,
)
from .vlm_initial import load_vlm_initial_context
from .kin_agent import (
    KinematicAgentConfig,
    KinematicAgentResult,
    KinematicCandidate,
    infer_kinematics,
    load_mesh_points,
    load_obj_points,
)
from .kin_export import (
    delivery_joint_payload,
    export_decoded_mesh_obj,
    write_kinematic_bundle_mjcf,
    write_kinematic_bundle_usda,
    write_kinematic_mjcf,
    write_kinematic_usda,
)
from .motion_observation import (
    MotionObservationEstimate,
    StaticPartObservationEstimate,
    estimate_static_part_observation,
    estimate_motion_hypotheses_from_render_states,
    estimate_motion_from_render_states,
    fit_motion_trajectory,
)
from .range_prior import (
    DEFAULT_RANGE_PRIOR,
    RANGE_CALIBRATOR_FEATURES,
    RangePriorEstimate,
    calibrate_range_candidate,
    load_range_prior,
    range_calibration_features,
)
from .axis_family_model import (
    DEFAULT_AXIS_FAMILY_MODEL,
    apply_axis_family_reranker,
    axis_family_numeric_features,
    predict_axis_family,
)
from .static_axis_family_model import (
    DEFAULT_STATIC_AXIS_FAMILY_MODEL,
    apply_static_axis_family_reranker,
    predict_static_axis_family,
)
from .static_dino_features import StaticDinoPartFeature, pool_static_part_dino_feature
from .static_dino_axis_model import (
    DEFAULT_STATIC_DINO_AXIS_MODEL,
    apply_static_dino_door_axis_reranker,
)
from .thin_axis_critic import (
    PHYX_KNOB_THIN_AXIS_MAX_SCORE_DROP,
    PHYX_KNOB_THIN_AXIS_MIN_CONFIDENCE,
    apply_phyx_knob_thin_axis_critic,
    decoded_thin_axis_evidence,
)
from .door_contact_axis_critic import (
    PHYX_DOOR_CONTACT_MAX_SCORE_DROP,
    PHYX_DOOR_CONTACT_MIN_CONFIDENCE,
    PHYX_DOOR_CONTACT_QUANTILE,
    apply_phyx_door_contact_axis_critic,
    decoded_door_contact_axis_evidence,
)
from .collision_audit import (
    AUDIT_VERSION as DECODED_COLLISION_AUDIT_VERSION,
    DecodedCollisionAuditConfig,
    audit_decoded_bundle_collisions,
    audit_joint_collision,
)
from .collision_feedback import (
    FEEDBACK_VERSION as DECODED_COLLISION_FEEDBACK_VERSION,
    CollisionFeedbackConfig,
    propose_collision_clear_interval,
)

__all__ = [
    "AgentLoopConfig",
    "AgentLoopError",
    "AgentLoopResult",
    "apply_action_update",
    "build_action_messages",
    "AXIS_ACTIONS",
    "AxisAction",
    "AxisCandidate",
    "AxisSearchResult",
    "CandidateCompileError",
    "CandidateReport",
    "CompileSignal",
    "CompileSignalBundle",
    "EstimateContext",
    "LimitEstimate",
    "MotionSearchValidationResult",
    "RangeSearchResult",
    "build_context_from_roots",
    "compile_candidate_report",
    "context_details",
    "editable_region_sha256",
    "extract_action_response_json",
    "extract_editable_response_code",
    "extract_editable_code",
    "infer_axis_candidates_for_joint",
    "LiveViewer",
    "load_vlm_initial_context",
    "delivery_joint_payload",
    "next_axis_from_feedback",
    "replace_editable_code",
    "render_estimate_limits_from_report",
    "resolve_agent_loop_config",
    "refine_positive_limit",
    "run_agent_loop",
    "search_axis_actions",
    "start_live_viewer_server",
    "validate_motion_search",
    "validate_motion_search_from_roots",
    "validate_motion_samples",
    "validate_motion_samples_from_roots",
    "with_axis_candidate_evidence",
    "write_estimate_viewers",
    "write_iteration_mjcf_preview",
    "write_rest_mjcf_preview",
    "KinematicAgentConfig",
    "KinematicAgentResult",
    "KinematicCandidate",
    "infer_kinematics",
    "load_mesh_points",
    "load_obj_points",
    "export_decoded_mesh_obj",
    "write_kinematic_bundle_mjcf",
    "write_kinematic_bundle_usda",
    "write_kinematic_mjcf",
    "write_kinematic_usda",
    "MotionObservationEstimate",
    "StaticPartObservationEstimate",
    "estimate_static_part_observation",
    "estimate_motion_hypotheses_from_render_states",
    "estimate_motion_from_render_states",
    "fit_motion_trajectory",
    "RangePriorEstimate",
    "DEFAULT_RANGE_PRIOR",
    "RANGE_CALIBRATOR_FEATURES",
    "calibrate_range_candidate",
    "load_range_prior",
    "range_calibration_features",
    "DEFAULT_AXIS_FAMILY_MODEL",
    "apply_axis_family_reranker",
    "axis_family_numeric_features",
    "predict_axis_family",
    "DEFAULT_STATIC_AXIS_FAMILY_MODEL",
    "apply_static_axis_family_reranker",
    "predict_static_axis_family",
    "StaticDinoPartFeature",
    "pool_static_part_dino_feature",
    "DEFAULT_STATIC_DINO_AXIS_MODEL",
    "apply_static_dino_door_axis_reranker",
    "PHYX_KNOB_THIN_AXIS_MAX_SCORE_DROP",
    "PHYX_KNOB_THIN_AXIS_MIN_CONFIDENCE",
    "apply_phyx_knob_thin_axis_critic",
    "decoded_thin_axis_evidence",
    "PHYX_DOOR_CONTACT_MAX_SCORE_DROP",
    "PHYX_DOOR_CONTACT_MIN_CONFIDENCE",
    "PHYX_DOOR_CONTACT_QUANTILE",
    "apply_phyx_door_contact_axis_critic",
    "decoded_door_contact_axis_evidence",
    "DECODED_COLLISION_AUDIT_VERSION",
    "DecodedCollisionAuditConfig",
    "audit_decoded_bundle_collisions",
    "audit_joint_collision",
    "DECODED_COLLISION_FEEDBACK_VERSION",
    "CollisionFeedbackConfig",
    "propose_collision_clear_interval",
]
