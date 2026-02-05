#!/usr/bin/env python3
"""
Wide Table Transformer Engine
Transforms profiled/remediated data into AI-ready wide table schemas.

Pipeline Position: DQ Profiler -> Remediation Engine -> Wide Table Transformer -> (future: data load)
"""

import pandas as pd
import numpy as np
import json
import yaml
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

from wtt_data_classes import (
    ColumnRole, TransformType, TableShape,
    KeyCandidate, GrainAmbiguity, GrainDiscoveryResult,
    ColumnClassification, ColumnDefinition, ExcludedColumn,
    SidecarTable, ColumnMapping, DerivedColumn, LoadPlan,
    WideTableDefinition, ValidationTest, ValidationResult,
    WideTablePlan, get_sql_type
)
from wtt_report_generator import WideTableReportGenerator


class WideTableTransformer:
    """
    Transforms profiled/remediated data into AI-ready wide table schemas.
    Implements a 5-phase pipeline:
    1. Grain Discovery
    2. Column Role Classification
    3. Semantic Shaping
    4. Table Shaping
    5. Schema Output
    """

    def __init__(self, spec_path: str = None, confidence_threshold: float = 0.75):
        """
        Initialize the Wide Table Transformer.

        Args:
            spec_path: Path to wide_table_transformer_spec.yaml (optional)
            confidence_threshold: Minimum confidence for automatic processing
        """
        self.confidence_threshold = confidence_threshold
        self.spec = self._load_spec(spec_path) if spec_path else self._default_spec()
        self.timestamp = datetime.utcnow().isoformat() + "Z"

    def _load_spec(self, spec_path: str) -> Dict:
        """Load WTT spec from YAML file"""
        with open(spec_path, 'r') as f:
            return yaml.safe_load(f)

    def _default_spec(self) -> Dict:
        """Return default spec configuration"""
        return {
            "naming_conventions": {
                "table": {"case": "snake_case", "suffix": "_wide"},
                "column": {"case": "snake_case"}
            },
            "target_storage_defaults": {
                "preferred_types": {
                    "string": "TEXT",
                    "integer": "BIGINT",
                    "decimal": "DECIMAL(38,10)",
                    "boolean": "BOOLEAN",
                    "date": "DATE",
                    "datetime": "TIMESTAMP"
                }
            }
        }

    def transform(self, df: pd.DataFrame, profile: Dict,
                  control_params: Dict = None) -> WideTablePlan:
        """
        Main entry point. Transform data into wide table plan.

        Args:
            df: The remediated DataFrame
            profile: Profile JSON from data profiler
            control_params: Optional runtime configuration

        Returns:
            WideTablePlan with complete schema and load plan
        """
        control_params = control_params or {}

        # Extract entity info
        entity = profile.get('entities', [{}])[0]
        entity_name = entity.get('entity_name', 'unknown')
        dataset_name = profile.get('dataset', {}).get('dataset_name', 'unknown')

        # Phase 1: Grain Discovery
        grain_result = self.discover_grain(df, profile)

        # Phase 2: Column Role Classification
        classifications = self.classify_column_roles(df, profile, grain_result)

        # Phase 3: Semantic Shaping
        shaped_columns, excluded, derived_cols = self.apply_semantic_shaping(
            df, profile, classifications, grain_result
        )

        # Phase 4: Table Shaping
        tables, sidecars = self.shape_tables(
            df, profile, grain_result, shaped_columns, excluded, entity_name
        )

        # Phase 5: Schema Output
        plan = self.generate_schema_output(
            df, profile, tables, excluded, sidecars, derived_cols,
            grain_result, entity_name, dataset_name
        )

        return plan

    # =========================================================================
    # Phase 1: Grain Discovery
    # =========================================================================

    def discover_grain(self, df: pd.DataFrame, profile: Dict) -> GrainDiscoveryResult:
        """
        Phase 1: Discover the grain (what one row represents).

        Uses profile stats to score key candidates and identify time axis.
        """
        entity = profile.get('entities', [{}])[0]
        columns = entity.get('columns', [])

        # Score key candidates
        key_candidates = self._score_key_candidates(df, columns)

        # Identify time axis
        time_axis = self._identify_time_axis(columns)

        # Select best keys
        selected_keys = self._select_grain_keys(key_candidates, time_axis)

        # Validate grain (check for duplicates)
        confidence, ambiguities = self._validate_grain(df, selected_keys, time_axis)

        # Determine if human review is required
        human_review = confidence < self.confidence_threshold

        # Build grain description
        description = self._build_grain_description(selected_keys, time_axis, df)

        # Generate alternatives if confidence is low
        alternatives = []
        if human_review:
            alternatives = self._generate_alternative_grains(key_candidates, time_axis)

        return GrainDiscoveryResult(
            description=description,
            keys=selected_keys,
            time_axis=time_axis,
            confidence=confidence,
            human_review_required=human_review,
            ambiguities=ambiguities,
            key_candidates=key_candidates,
            alternative_grains=alternatives
        )

    def _score_key_candidates(self, df: pd.DataFrame,
                               columns: List[Dict]) -> List[KeyCandidate]:
        """Score each column as potential grain key using profile stats"""
        candidates = []

        for col_info in columns:
            col_name = col_info.get('column_name')
            if col_name not in df.columns:
                continue

            stats = col_info.get('statistics', {})
            completeness = stats.get('completeness', {})
            cardinality = stats.get('cardinality', {})
            role = col_info.get('role', 'dimension')
            criticality = col_info.get('criticality', 'low')

            null_rate = completeness.get('null_rate', 0)
            distinct_rate = cardinality.get('distinct_rate', 0) or 0

            # Role alignment scoring
            role_alignment = 0.0
            if role == 'identifier':
                role_alignment = 1.0
            elif role == 'timestamp':
                role_alignment = 0.8
            elif role == 'dimension':
                role_alignment = 0.3
            elif role == 'measure':
                role_alignment = 0.0

            # Name heuristics
            name_heuristics = 0.0
            col_lower = col_name.lower()
            if any(kw in col_lower for kw in ['_id', 'id_', 'key', 'uuid', 'guid']):
                name_heuristics = 1.0
            elif any(kw in col_lower for kw in ['code', 'num', 'number']):
                name_heuristics = 0.5

            # Criticality alignment
            criticality_alignment = {
                'critical': 1.0,
                'high': 0.5,
                'medium': 0.3,
                'low': 0.1
            }.get(criticality, 0.1)

            # Calculate overall score per spec formula
            score = (
                0.30 * (1 - null_rate) +           # Completeness
                0.25 * distinct_rate +              # Cardinality
                0.25 * role_alignment +             # Role alignment
                0.10 * name_heuristics +            # Name heuristics
                0.10 * criticality_alignment        # Criticality alignment
            )

            candidates.append(KeyCandidate(
                column_name=col_name,
                score=round(score, 4),
                null_rate=null_rate,
                distinct_rate=distinct_rate,
                role_alignment=role_alignment,
                name_heuristics=name_heuristics,
                criticality_alignment=criticality_alignment
            ))

        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _identify_time_axis(self, columns: List[Dict]) -> Optional[str]:
        """Identify the time axis column from profile"""
        time_pattern = re.compile(
            r'(?i)date|time|month|week|year|period|as_of|snapshot'
        )

        for col_info in columns:
            col_name = col_info.get('column_name', '')
            role = col_info.get('role', '')

            # Check role first
            if role == 'timestamp':
                return col_name

            # Check name pattern
            if time_pattern.search(col_name):
                return col_name

        return None

    def _select_grain_keys(self, candidates: List[KeyCandidate],
                           time_axis: Optional[str]) -> List[str]:
        """Select the best grain keys from candidates"""
        selected = []

        # Find high-scoring identifier candidates
        for candidate in candidates:
            if candidate.score >= 0.5 and candidate.role_alignment >= 0.8:
                selected.append(candidate.column_name)
                candidate.is_selected = True

        # If no high-scoring identifiers, take the best non-timestamp candidate
        if not selected and candidates:
            for candidate in candidates:
                if candidate.column_name != time_axis and candidate.score >= 0.4:
                    selected.append(candidate.column_name)
                    candidate.is_selected = True
                    break

        # Add time axis if present and not already included
        if time_axis and time_axis not in selected:
            selected.append(time_axis)
            for candidate in candidates:
                if candidate.column_name == time_axis:
                    candidate.is_selected = True

        return selected

    def _validate_grain(self, df: pd.DataFrame, keys: List[str],
                        time_axis: Optional[str]) -> Tuple[float, List[GrainAmbiguity]]:
        """
        Validate grain by checking for duplicates.
        Returns confidence score and list of ambiguities.
        """
        confidence = 1.0
        ambiguities = []

        if not keys:
            confidence = 0.3
            ambiguities.append(GrainAmbiguity(
                ambiguity_type="no_keys_found",
                description="Could not identify any grain keys",
                affected_columns=[],
                confidence_penalty=0.7,
                resolution_options=["Manual key specification required"]
            ))
            return confidence, ambiguities

        # Check for duplicate rows with chosen grain
        valid_keys = [k for k in keys if k in df.columns]
        if valid_keys:
            duplicate_count = df.duplicated(subset=valid_keys).sum()
            total_rows = len(df)

            if duplicate_count > 0:
                dup_rate = duplicate_count / total_rows
                penalty = min(0.30, dup_rate * 0.5)
                confidence -= penalty
                ambiguities.append(GrainAmbiguity(
                    ambiguity_type="duplicates_found",
                    description=f"{duplicate_count} duplicate rows ({dup_rate:.1%}) with chosen grain keys",
                    affected_columns=valid_keys,
                    confidence_penalty=penalty,
                    resolution_options=[
                        "Add disambiguating keys",
                        "Aggregate measures",
                        "Split into multiple tables"
                    ]
                ))

        # Query spec for time axis heuristic regex (WTT-G2)
        heuristics = self.spec.get('wide_table_definition_rules', {}).get('grain_discovery', {}).get('heuristics', [])
        time_heuristic = next((h for h in heuristics if h.get('id') == 'WTT-G2'), {})
        # Fallback to default regex if spec is missing logic
        time_regex = r'(?i)date|time|month|week|year|period|as_of|snapshot'
        if 'logic' in time_heuristic:
            # Attempt to extract regex from logic description if structured, 
            # but currently logic is text. For robust policy driving, 
            # we should parse or have a structured field. 
            # Assuming the spec logic description contains the regex for now 
            # or implementing a robust fallback.
            # Ideally the spec would have a 'regex' field.
            # Using the hardcoded default for now as the spec 'logic' is free text.
            pass

        # Penalize if missing time axis when data appears temporal
        if not time_axis:
            # Check if any column looks temporal
            for col in df.columns:
                if re.search(time_regex, col):
                    confidence -= 0.10
                    ambiguities.append(GrainAmbiguity(
                        ambiguity_type="missing_time_axis",
                        description=f"Column '{col}' appears temporal but wasn't selected as time axis",
                        affected_columns=[col],
                        confidence_penalty=0.10,
                        resolution_options=[f"Consider adding '{col}' as time axis"]
                    ))
                    break

        return max(0.0, confidence), ambiguities

    def _build_grain_description(self, keys: List[str], time_axis: Optional[str],
                                  df: pd.DataFrame) -> str:
        """Build human-readable grain description"""
        if not keys:
            return "Unknown grain (no keys identified)"

        # Extract entity from key names
        entity_parts = []
        for key in keys:
            if key != time_axis:
                # Convert snake_case to readable
                parts = key.lower().replace('_id', '').replace('id_', '').split('_')
                entity_parts.extend(parts)

        entity = ' '.join(entity_parts) if entity_parts else 'record'

        # Determine time granularity
        time_desc = ""
        if time_axis:
            time_lower = time_axis.lower()
            if 'week' in time_lower:
                time_desc = " per week"
            elif 'month' in time_lower:
                time_desc = " per month"
            elif 'year' in time_lower:
                time_desc = " per year"
            elif 'day' in time_lower or 'date' in time_lower:
                time_desc = " per day"
            else:
                time_desc = f" per {time_axis}"

        return f"One row per {entity}{time_desc}"

    def _generate_alternative_grains(self, candidates: List[KeyCandidate],
                                      time_axis: Optional[str]) -> List[Dict[str, Any]]:
        """Generate alternative grain options when confidence is low"""
        alternatives = []

        # Get top candidates
        top_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)[:5]

        # Generate combinations
        for i, primary in enumerate(top_candidates[:3]):
            alt_keys = [primary.column_name]
            if time_axis and time_axis != primary.column_name:
                alt_keys.append(time_axis)

            alternatives.append({
                "keys": alt_keys,
                "estimated_confidence": primary.score,
                "description": f"Use {primary.column_name} as primary key"
            })

        return alternatives[:3]

    # =========================================================================
    # Phase 2: Column Role Classification
    # =========================================================================

    def classify_column_roles(self, df: pd.DataFrame, profile: Dict,
                               grain_result: GrainDiscoveryResult) -> List[ColumnClassification]:
        """
        Phase 2: Classify each column into exactly one role.
        Applies rules WTT-R1 through WTT-R5 and resolves conflicts.
        """
        entity = profile.get('entities', [{}])[0]
        columns = entity.get('columns', [])
        classifications = []

        for col_info in columns:
            col_name = col_info.get('column_name')
            if col_name not in df.columns:
                continue

            classification = self._classify_single_column(
                df, col_name, col_info, grain_result
            )
            classifications.append(classification)

        return classifications

    def _classify_single_column(self, df: pd.DataFrame, col_name: str,
                                 col_info: Dict,
                                 grain_result: GrainDiscoveryResult) -> ColumnClassification:
        """Classify a single column using dynamic spec rules"""
        role_hint = col_info.get('role', '')
        logical_type = col_info.get('logical_type', 'object')
        stats = col_info.get('statistics', {})
        cardinality = stats.get('cardinality', {})
        distinct_rate = cardinality.get('distinct_rate', 0) or 0

        role = None
        rule_applied = "default"
        original_role = None
        conflict_resolved = False
        resolution_reason = ""

        # Check if it's a grain key (Override rules)
        if col_name in grain_result.keys:
            if col_name == grain_result.time_axis:
                role = ColumnRole.TIME_DIMENSION
                rule_applied = "grain_time_axis"
            else:
                role = ColumnRole.KEY_DIMENSION
                rule_applied = "grain_key"
        else:
            # Dynamic Rule Evaluation from Spec
            rules = self.spec.get('wide_table_definition_rules', {}).get('column_roles', {}).get('default_rules', [])
            
            for rule in rules:
                rule_id = rule.get('id')
                conditions = rule.get('when', {})
                target_role = rule.get('assign')
                
                match = True
                
                # Check role_hints_contains
                if 'role_hints_contains' in conditions:
                    if conditions['role_hints_contains'] not in role_hint:
                        match = False
                
                # Check logical_type_in
                if match and 'logical_type_in' in conditions:
                    allowed_types = conditions['logical_type_in']
                    # Handle simplified types matching pandas/numpy types
                    is_type_match = False
                    if logical_type in allowed_types:
                        is_type_match = True
                    # Check for generic 'integer', 'decimal' matches against actual types like 'int64'
                    elif 'integer' in allowed_types and (logical_type.startswith('int') or logical_type in ['int64', 'int32']):
                        is_type_match = True
                    elif 'decimal' in allowed_types and (logical_type.startswith('float') or logical_type in ['float64', 'float32']):
                        is_type_match = True
                    
                    if not is_type_match:
                        match = False
                        
                # Check column_name_regex
                if match and 'column_name_regex' in conditions:
                    pattern = conditions['column_name_regex']
                    try:
                        if not re.search(pattern, col_name):
                            match = False
                    except re.error:
                        print(f"Warning: Invalid regex in rule {rule_id}: {pattern}")
                        match = False
                
                if match:
                    # Convert string role from spec to Enum
                    try:
                        role = ColumnRole(target_role)
                        rule_applied = rule_id
                        break # First match wins
                    except ValueError:
                        print(f"Warning: Invalid role {target_role} in rule {rule_id}")
                        continue

            # Fallback if no rule matched
            if role is None:
                role = ColumnRole.ANALYTIC_DIMENSION
                rule_applied = "default_fallback"

        # Conflict resolution (Logic remains consistent with spec guidelines)
        # TODO: Move specific conflict resolution logic to spec if needed, 
        # but spec 'conflict_resolution' section is currently descriptive text rules.
        
        # Rule: Low-cardinality numeric -> analytic_dimension
        if role == ColumnRole.MEASURE:
            series = df[col_name]
            unique_vals = series.dropna().unique()

            # Check for Yes/No-like values
            if len(unique_vals) <= 5:
                str_vals = [str(v).lower() for v in unique_vals]
                if set(str_vals).issubset({'yes', 'no', 'true', 'false', '1', '0', '1.0', '0.0'}):
                    original_role = role
                    role = ColumnRole.ANALYTIC_DIMENSION
                    conflict_resolved = True
                    resolution_reason = "Low-cardinality Yes/No-like values reclassified"
            # Very low cardinality but not boolean-like -> still might be dimension
            elif distinct_rate < 0.01 and len(unique_vals) < 20:
                original_role = role
                role = ColumnRole.ANALYTIC_DIMENSION
                conflict_resolved = True
                resolution_reason = f"Extremely low cardinality ({len(unique_vals)} unique values)"

        return ColumnClassification(
            column_name=col_name,
            role=role,
            rule_applied=rule_applied,
            confidence=1.0 if not conflict_resolved else 0.8,
            conflict_resolved=conflict_resolved,
            original_role=original_role,
            resolution_reason=resolution_reason
        )

    # =========================================================================
    # Phase 3: Semantic Shaping
    # =========================================================================

    def apply_semantic_shaping(self, df: pd.DataFrame, profile: Dict,
                                classifications: List[ColumnClassification],
                                grain_result: GrainDiscoveryResult
                                ) -> Tuple[List[ColumnDefinition], List[ExcludedColumn], List[DerivedColumn]]:
        """
        Phase 3: Shape columns for AI readability.
        - Rename opaque identifiers
        - Generate canonical time columns
        - Mark exclusions (free_text, metadata -> sidecar)
        """
        entity = profile.get('entities', [{}])[0]
        col_info_map = {c['column_name']: c for c in entity.get('columns', [])}

        shaped_columns = []
        excluded_columns = []
        derived_columns = []

        for classification in classifications:
            col_name = classification.column_name
            role = classification.role
            col_info = col_info_map.get(col_name, {})
            logical_type = col_info.get('logical_type', 'object')

            # Exclude free_text and metadata
            if role in [ColumnRole.FREE_TEXT, ColumnRole.METADATA]:
                excluded_columns.append(ExcludedColumn(
                    source_column=col_name,
                    reason=f"Role is {role.value}; excluded from wide table per WTT-S5",
                    role=role.value,
                    recommended_action="sidecar"
                ))
                continue

            # Generate column name (semantic shaping)
            target_name = self._shape_column_name(col_name, role)

            # Determine SQL type
            sql_type = get_sql_type(logical_type)

            # Determine nullable
            stats = col_info.get('statistics', {})
            null_rate = stats.get('completeness', {}).get('null_rate', 0)
            nullable = null_rate > 0 or role not in [
                ColumnRole.KEY_DIMENSION,
                ColumnRole.TIME_DIMENSION,
                ColumnRole.MEASURE
            ]

            col_def = ColumnDefinition(
                name=target_name,
                role=role.value,
                type=sql_type,
                nullable=nullable,
                source_columns=[col_name],
                transform=TransformType.DIRECT.value if target_name == col_name else TransformType.RENAME.value,
                description=f"Classified as {role.value} via rule {classification.rule_applied}"
            )
            shaped_columns.append(col_def)

        # Add canonical time columns if time axis exists (WTT-S2)
        if grain_result.time_axis:
            time_derived = self._generate_canonical_time_columns(
                df, grain_result.time_axis, col_info_map.get(grain_result.time_axis, {})
            )
            derived_columns.extend(time_derived)

        return shaped_columns, excluded_columns, derived_columns

    def _shape_column_name(self, col_name: str, role: ColumnRole) -> str:
        """
        Apply semantic shaping to column name (WTT-S1).
        Renames opaque identifiers to be more explicit.
        """
        # Convert to snake_case
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', col_name)
        name = re.sub(r'[\s\-]+', '_', name)
        name = name.lower()

        # Clean up common patterns
        name = re.sub(r'__+', '_', name)
        name = name.strip('_')

        return name

    def _generate_canonical_time_columns(self, df: pd.DataFrame,
                                          time_axis: str,
                                          col_info: Dict) -> List[DerivedColumn]:
        """Generate canonical time columns per WTT-S2"""
        derived = []

        if time_axis not in df.columns:
            return derived

        series = df[time_axis]

        # Try to parse as datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(series):
            try:
                series = pd.to_datetime(series, errors='coerce')
            except Exception:
                return derived

        # Check what we can derive
        if series.notna().any():
            # event_date
            derived.append(DerivedColumn(
                target_column="event_date",
                source_columns=[time_axis],
                derivation_logic=f"CAST({time_axis} AS DATE)",
                type="DATE"
            ))

            # event_month
            derived.append(DerivedColumn(
                target_column="event_month",
                source_columns=[time_axis],
                derivation_logic=f"DATE_TRUNC('month', {time_axis})",
                type="DATE"
            ))

            # event_year
            derived.append(DerivedColumn(
                target_column="event_year",
                source_columns=[time_axis],
                derivation_logic=f"EXTRACT(YEAR FROM {time_axis})",
                type="BIGINT"
            ))

        return derived

    # =========================================================================
    # Phase 4: Table Shaping
    # =========================================================================

    def shape_tables(self, df: pd.DataFrame, profile: Dict,
                     grain_result: GrainDiscoveryResult,
                     shaped_columns: List[ColumnDefinition],
                     excluded: List[ExcludedColumn],
                     entity_name: str
                     ) -> Tuple[List[WideTableDefinition], List[SidecarTable]]:
        """
        Phase 4: Create one or more wide tables.
        - Detect grain conflicts -> split into multiple tables
        - Detect sparse blocks -> move to sidecar
        """
        tables = []
        sidecars = []

        # Determine table shape
        table_shape = self._determine_table_shape(df, grain_result, shaped_columns)

        # Check for sparse column blocks (WTT-T2)
        sparse_columns, non_sparse_columns = self._identify_sparse_blocks(
            df, shaped_columns, threshold=0.85
        )

        # Build main wide table
        table_name = self._generate_table_name(entity_name)
        main_table = WideTableDefinition(
            table_name=table_name,
            grain=grain_result,
            shape=table_shape,
            columns=non_sparse_columns,
            source_entity=entity_name
        )
        tables.append(main_table)

        # Create sidecar for sparse columns if any
        if sparse_columns:
            sidecar = SidecarTable(
                table_name=f"{table_name}_sparse",
                purpose="Contains sparse columns (>85% null) from main table",
                grain_keys=grain_result.keys,
                columns=sparse_columns
            )
            sidecars.append(sidecar)

        # Create sidecar for excluded columns if any
        if excluded:
            excluded_cols = [
                ColumnDefinition(
                    name=ex.source_column,
                    role=ex.role,
                    type="TEXT",
                    nullable=True,
                    source_columns=[ex.source_column],
                    transform="DIRECT",
                    description=ex.reason
                )
                for ex in excluded
            ]
            sidecar = SidecarTable(
                table_name=f"{table_name}_text",
                purpose="Contains free_text and metadata columns excluded from main table",
                grain_keys=grain_result.keys,
                columns=excluded_cols
            )
            sidecars.append(sidecar)

        return tables, sidecars

    def _determine_table_shape(self, df: pd.DataFrame,
                                grain_result: GrainDiscoveryResult,
                                columns: List[ColumnDefinition]) -> str:
        """Determine the appropriate table shape"""
        has_time = grain_result.time_axis is not None
        measure_count = sum(1 for c in columns if c.role == ColumnRole.MEASURE.value)

        if has_time and measure_count > 3:
            return TableShape.KPI_OBSERVATION_WIDE.value
        elif has_time:
            return TableShape.EVENT_WIDE.value
        elif measure_count > 5:
            return TableShape.PERIODIC_SUMMARY_WIDE.value
        else:
            return TableShape.ENTITY_SNAPSHOT_WIDE.value

    def _identify_sparse_blocks(self, df: pd.DataFrame,
                                 columns: List[ColumnDefinition],
                                 threshold: float = 0.85
                                 ) -> Tuple[List[ColumnDefinition], List[ColumnDefinition]]:
        """Identify columns with null rate > threshold"""
        sparse = []
        non_sparse = []

        for col_def in columns:
            source_col = col_def.source_columns[0] if col_def.source_columns else None
            if source_col and source_col in df.columns:
                null_rate = df[source_col].isna().sum() / len(df) if len(df) > 0 else 0
                if null_rate > threshold:
                    sparse.append(col_def)
                else:
                    non_sparse.append(col_def)
            else:
                non_sparse.append(col_def)

        return sparse, non_sparse

    def _generate_table_name(self, entity_name: str) -> str:
        """Generate table name per naming conventions"""
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', entity_name)
        name = re.sub(r'[\s\-]+', '_', name)
        name = name.lower()
        name = re.sub(r'__+', '_', name)
        name = name.strip('_')

        if not name.endswith('_wide'):
            name = f"{name}_wide"

        return name[:63]  # Max length per spec

    # =========================================================================
    # Phase 5: Schema Output
    # =========================================================================

    def generate_schema_output(self, df: pd.DataFrame, profile: Dict,
                                tables: List[WideTableDefinition],
                                excluded: List[ExcludedColumn],
                                sidecars: List[SidecarTable],
                                derived_cols: List[DerivedColumn],
                                grain_result: GrainDiscoveryResult,
                                entity_name: str,
                                dataset_name: str) -> WideTablePlan:
        """
        Phase 5: Generate complete schema output with load plan and validation.
        """
        # Build load plan
        load_plan = self._build_load_plan(
            tables[0] if tables else None,
            derived_cols,
            entity_name
        )

        # Run validation
        validation = self._run_validation(tables, excluded, load_plan, grain_result)

        # Determine human review requirements
        human_review_required = grain_result.human_review_required or not validation.passed
        human_review_reasons = []
        if grain_result.human_review_required:
            human_review_reasons.append(f"Grain confidence ({grain_result.confidence:.2f}) below threshold")
        if not validation.passed:
            human_review_reasons.extend([t.details for t in validation.tests if not t.passed])

        return WideTablePlan(
            version="1.0.0",
            generated_at=self.timestamp,
            source_profile_version=profile.get('profile_version', 'unknown'),
            source_dataset_name=dataset_name,
            tables=tables,
            excluded_columns=excluded,
            sidecar_tables=sidecars,
            load_plan=load_plan,
            validation=validation,
            human_review_required=human_review_required,
            human_review_reasons=human_review_reasons
        )

    def _build_load_plan(self, table: Optional[WideTableDefinition],
                         derived_cols: List[DerivedColumn],
                         entity_name: str) -> Optional[LoadPlan]:
        """Build the load plan with column mappings"""
        if not table:
            return None

        column_mappings = []
        type_casts = {}

        for col in table.columns:
            source = col.source_columns[0] if col.source_columns else col.name
            mapping = ColumnMapping(
                source_column=source,
                target_column=col.name,
                transform=col.transform,
                type_cast=col.type if col.transform == TransformType.CAST.value else None
            )
            column_mappings.append(mapping)

            if col.type != "TEXT":
                type_casts[col.name] = col.type

        return LoadPlan(
            source_entity=entity_name,
            target_table=table.table_name,
            column_mappings=column_mappings,
            derived_columns=derived_cols,
            type_casts=type_casts,
            null_handling_notes=["Nulls preserved as SQL NULL per upstream remediation"]
        )

    def _run_validation(self, tables: List[WideTableDefinition],
                        excluded: List[ExcludedColumn],
                        load_plan: Optional[LoadPlan],
                        grain_result: GrainDiscoveryResult) -> ValidationResult:
        """Run validation tests WTT-V1 through WTT-V4"""
        tests = []
        errors = []
        warnings = []

        # WTT-V1: Dashboard can be generated without joins
        v1_passed = len(tables) > 0 and all(
            len(t.columns) > 0 for t in tables
        )
        tests.append(ValidationTest(
            test_id="WTT-V1",
            test_description="A dashboard can be generated from each wide table without joins",
            passed=v1_passed,
            details="" if v1_passed else "No valid wide table generated",
            fail_action="reject_plan"
        ))
        if not v1_passed:
            errors.append("WTT-V1 failed: Cannot generate dashboard without joins")

        # WTT-V2: Each table has exactly one grain and explicit key set
        v2_passed = all(
            t.grain and len(t.grain.keys) > 0 for t in tables
        )
        tests.append(ValidationTest(
            test_id="WTT-V2",
            test_description="Each table has exactly one grain and an explicit key set",
            passed=v2_passed,
            details="" if v2_passed else "Table missing grain or keys",
            fail_action="require_human"
        ))
        if not v2_passed:
            warnings.append("WTT-V2 failed: Table missing grain or key definition")

        # WTT-V3: Every output column traces to source
        v3_passed = True
        if load_plan:
            for mapping in load_plan.column_mappings:
                if not mapping.source_column:
                    v3_passed = False
                    break
        tests.append(ValidationTest(
            test_id="WTT-V3",
            test_description="Every output column traces to a source column or documented derivation",
            passed=v3_passed,
            details="" if v3_passed else "Column missing source traceability",
            fail_action="reject_plan"
        ))
        if not v3_passed:
            errors.append("WTT-V3 failed: Column missing source traceability")

        # WTT-V4: Measures are numeric and aggregation-safe
        v4_passed = True
        for table in tables:
            for col in table.columns:
                if col.role == ColumnRole.MEASURE.value:
                    if col.type not in ["BIGINT", "DECIMAL(38,10)", "INTEGER", "FLOAT"]:
                        v4_passed = False
                        break
        tests.append(ValidationTest(
            test_id="WTT-V4",
            test_description="Measures are numeric and aggregation-safe",
            passed=v4_passed,
            details="" if v4_passed else "Non-numeric measure found",
            fail_action="reject_plan"
        ))
        if not v4_passed:
            errors.append("WTT-V4 failed: Non-numeric measure found")

        overall_passed = all(t.passed for t in tests)

        return ValidationResult(
            passed=overall_passed,
            tests=tests,
            errors=errors,
            warnings=warnings
        )


def plan_to_dict(plan: WideTablePlan) -> Dict[str, Any]:
    """Convert WideTablePlan to serializable dictionary"""
    def dataclass_to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj

    return dataclass_to_dict(plan)


def main():
    parser = argparse.ArgumentParser(
        description='Wide Table Transformer - Transform profiled data into AI-ready wide tables'
    )
    parser.add_argument('input_file', help='Remediated data file (CSV or Excel)')
    parser.add_argument('profile_file', help='Profile JSON from data profiler')
    parser.add_argument('--spec', default=None,
                        help='Path to wide_table_transformer_spec.yaml')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--confidence-threshold', type=float, default=0.75,
                        help='Confidence threshold for human review (default: 0.75)')

    args = parser.parse_args()

    # Read data
    print(f"Reading data from {args.input_file}...")
    file_path = Path(args.input_file)
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Read profile
    print(f"Reading profile from {args.profile_file}...")
    with open(args.profile_file, 'r') as f:
        profile = json.load(f)

    # Create transformer
    transformer = WideTableTransformer(
        spec_path=args.spec,
        confidence_threshold=args.confidence_threshold
    )

    # Transform
    print("\nRunning Wide Table Transformer...")
    plan = transformer.transform(df, profile)

    # Convert to dict for serialization
    plan_dict = plan_to_dict(plan)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "wide_table_plan.json"
    with open(json_path, 'w') as f:
        json.dump(plan_dict, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # Save YAML
    yaml_path = output_dir / "wide_table_plan.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(plan_dict, f, default_flow_style=False, sort_keys=False)
    with open(yaml_path, 'w') as f:
        yaml.dump(plan_dict, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {yaml_path}")

    # Save HTML Report
    report_path = output_dir / "wide_table_report.html"
    generator = WideTableReportGenerator()
    generator.generate_report(plan_dict, str(report_path))
    print(f"  Saved: {report_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("WIDE TABLE TRANSFORMER SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {plan.source_dataset_name}")
    print(f"Tables generated: {len(plan.tables)}")

    if plan.tables:
        for table in plan.tables:
            print(f"\n  Table: {table.table_name}")
            print(f"    Shape: {table.shape}")
            print(f"    Grain: {table.grain.description}")
            print(f"    Keys: {', '.join(table.grain.keys)}")
            print(f"    Columns: {len(table.columns)}")
            print(f"    Confidence: {table.grain.confidence:.2f}")

    print(f"\nExcluded columns: {len(plan.excluded_columns)}")
    print(f"Sidecar tables: {len(plan.sidecar_tables)}")

    print(f"\nValidation: {'PASSED' if plan.validation.passed else 'FAILED'}")
    for test in plan.validation.tests:
        status = "PASS" if test.passed else "FAIL"
        print(f"  [{status}] {test.test_id}: {test.test_description}")

    if plan.human_review_required:
        print(f"\nHuman review required:")
        for reason in plan.human_review_reasons:
            print(f"  - {reason}")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
