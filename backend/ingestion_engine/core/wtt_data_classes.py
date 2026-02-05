#!/usr/bin/env python3
"""
Wide Table Transformer Data Classes
Dataclass definitions for WTT pipeline components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class ColumnRole(Enum):
    """Column roles for wide table classification"""
    KEY_DIMENSION = "key_dimension"
    ANALYTIC_DIMENSION = "analytic_dimension"
    TIME_DIMENSION = "time_dimension"
    MEASURE = "measure"
    METRIC_DESCRIPTOR = "metric_descriptor"
    FREE_TEXT = "free_text"
    METADATA = "metadata"
    IDENTIFIER_CODE = "identifier_code"


class TransformType(Enum):
    """Transformation types for column mappings"""
    DIRECT = "DIRECT"
    RENAME = "RENAME"
    CAST = "CAST"
    DERIVED = "DERIVED"
    EXTRACT = "EXTRACT"


class TableShape(Enum):
    """Valid wide table shapes"""
    ENTITY_SNAPSHOT_WIDE = "entity_snapshot_wide"
    EVENT_WIDE = "event_wide"
    KPI_OBSERVATION_WIDE = "kpi_observation_wide"
    TRANSACTION_WIDE = "transaction_wide"
    PERIODIC_SUMMARY_WIDE = "periodic_summary_wide"


@dataclass
class KeyCandidate:
    """A candidate column for grain key"""
    column_name: str
    score: float
    null_rate: float
    distinct_rate: float
    role_alignment: float
    name_heuristics: float
    criticality_alignment: float
    is_selected: bool = False
    reason: str = ""


@dataclass
class GrainAmbiguity:
    """Describes ambiguity in grain determination"""
    ambiguity_type: str  # e.g., "multiple_candidates", "missing_time_axis", "duplicates"
    description: str
    affected_columns: List[str]
    confidence_penalty: float
    resolution_options: List[str] = field(default_factory=list)


@dataclass
class GrainDiscoveryResult:
    """Result of grain discovery phase"""
    description: str  # e.g., "One row per customer per week"
    keys: List[str]
    time_axis: Optional[str] = None
    confidence: float = 1.0
    human_review_required: bool = False
    ambiguities: List[GrainAmbiguity] = field(default_factory=list)
    key_candidates: List[KeyCandidate] = field(default_factory=list)
    alternative_grains: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ColumnClassification:
    """Classification result for a column"""
    column_name: str
    role: ColumnRole
    rule_applied: str  # e.g., "WTT-R1"
    confidence: float = 1.0
    conflict_resolved: bool = False
    original_role: Optional[ColumnRole] = None
    resolution_reason: str = ""


@dataclass
class ColumnDefinition:
    """Definition of a column in the wide table schema"""
    name: str
    role: str  # ColumnRole value
    type: str  # SQL type
    nullable: bool
    source_columns: List[str]
    transform: str  # TransformType value
    description: str = ""
    derivation_logic: Optional[str] = None


@dataclass
class ExcludedColumn:
    """A column excluded from the wide table"""
    source_column: str
    reason: str
    role: str
    recommended_action: str = "sidecar"


@dataclass
class SidecarTable:
    """Definition of a sidecar table for excluded columns"""
    table_name: str
    purpose: str
    grain_keys: List[str]  # Keys to join back to main table
    columns: List[ColumnDefinition]


@dataclass
class ColumnMapping:
    """Mapping from source to target column"""
    source_column: str
    target_column: str
    transform: str
    type_cast: Optional[str] = None
    derivation_logic: Optional[str] = None


@dataclass
class DerivedColumn:
    """A column derived from source columns"""
    target_column: str
    source_columns: List[str]
    derivation_logic: str
    type: str


@dataclass
class LoadPlan:
    """Plan for loading data into wide table"""
    source_entity: str
    target_table: str
    column_mappings: List[ColumnMapping]
    derived_columns: List[DerivedColumn] = field(default_factory=list)
    type_casts: Dict[str, str] = field(default_factory=dict)
    null_handling_notes: List[str] = field(default_factory=list)


@dataclass
class WideTableDefinition:
    """Definition of a single wide table"""
    table_name: str
    grain: GrainDiscoveryResult
    shape: str  # TableShape value
    columns: List[ColumnDefinition]
    source_entity: str


@dataclass
class ValidationTest:
    """Result of a validation test"""
    test_id: str  # e.g., "WTT-V1"
    test_description: str
    passed: bool
    details: str = ""
    fail_action: str = "reject_plan"


@dataclass
class ValidationResult:
    """Overall validation result"""
    passed: bool
    tests: List[ValidationTest]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class WideTablePlan:
    """Complete wide table transformation plan"""
    version: str = "1.0.0"
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    source_profile_version: str = ""
    source_dataset_name: str = ""
    tables: List[WideTableDefinition] = field(default_factory=list)
    excluded_columns: List[ExcludedColumn] = field(default_factory=list)
    sidecar_tables: List[SidecarTable] = field(default_factory=list)
    load_plan: Optional[LoadPlan] = None
    validation: Optional[ValidationResult] = None
    human_review_required: bool = False
    human_review_reasons: List[str] = field(default_factory=list)


# Type mapping from pandas/profile types to SQL types
TYPE_MAPPING = {
    "int64": "BIGINT",
    "int32": "BIGINT",
    "float64": "DECIMAL(38,10)",
    "float32": "DECIMAL(38,10)",
    "object": "TEXT",
    "string": "TEXT",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "datetime64[ns]": "TIMESTAMP",
    "datetime64": "TIMESTAMP",
    "date": "DATE",
    "category": "TEXT",
}


def get_sql_type(pandas_type: str) -> str:
    """Map pandas type to SQL type"""
    pandas_type_lower = str(pandas_type).lower()

    # Check for exact match
    if pandas_type_lower in TYPE_MAPPING:
        return TYPE_MAPPING[pandas_type_lower]

    # Check for partial matches
    if "int" in pandas_type_lower:
        return "BIGINT"
    if "float" in pandas_type_lower or "decimal" in pandas_type_lower:
        return "DECIMAL(38,10)"
    if "datetime" in pandas_type_lower or "timestamp" in pandas_type_lower:
        return "TIMESTAMP"
    if "date" in pandas_type_lower:
        return "DATE"
    if "bool" in pandas_type_lower:
        return "BOOLEAN"

    # Default to TEXT
    return "TEXT"
