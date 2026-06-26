"""Custom fail-loud exceptions for V1 KinematicSolver."""


class KinematicSolverError(Exception):
    """Base class for all V1 KinematicSolver errors."""


class DependencyMissingError(KinematicSolverError):
    pass


class CoacdParamsMissingError(KinematicSolverError):
    pass


class MissingModelError(KinematicSolverError):
    pass


class BakedPartsegMissingError(KinematicSolverError):
    pass


class MeshCoordFrameMismatchError(KinematicSolverError):
    pass


class JointLinkMismatchError(KinematicSolverError):
    pass


class SchemaMismatchError(KinematicSolverError):
    pass


class DegenerateAxisExtentError(KinematicSolverError):
    pass


class UnsupportedJointGraphError(KinematicSolverError):
    pass


class VhacdCacheMissingError(KinematicSolverError):
    pass


class VhacdParamsMismatchError(KinematicSolverError):
    pass


class FingerprintAlreadyWrittenError(KinematicSolverError):
    pass


class DatasetFingerprintDriftError(KinematicSolverError):
    pass


class InvalidValidationContextError(KinematicSolverError):
    pass
