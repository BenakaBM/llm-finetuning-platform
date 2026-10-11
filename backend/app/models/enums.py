import enum


class UserRole(str, enum.Enum):
    USER = "USER"
    ADMIN = "ADMIN"


class DatasetFormat(str, enum.Enum):
    INSTRUCTION = "instruction"
    CONVERSATIONAL = "conversational"


class TrainingMethod(str, enum.Enum):
    LORA = "LORA"
    QLORA = "QLORA"


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class EvaluationType(str, enum.Enum):
    BASE = "BASE"
    FINE_TUNED = "FINE_TUNED"
