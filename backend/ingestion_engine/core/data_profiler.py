#!/usr/bin/env python3
"""
Policy-Driven Data Profiler & Remediation Tool
Dynamically follows DQ Policy Spec (YAML) for profiling and remediation.
Policy changes automatically reflected without code changes.
"""

import pandas as pd
import numpy as np
import json
import re
import yaml
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import argparse
from typing import Dict, List, Tuple, Any, Optional


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class PolicyEngine:
    """Loads and interprets the DQ Policy Spec"""
    
    def __init__(self, policy_path: str = 'config/dq_policy_spec.yaml'):
        self.policy_path = Path(policy_path)
        self.policy = self._load_policy()
        self._cache_lookups()
    
    def _load_policy(self) -> Dict:
        """Load policy from YAML file"""
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.policy_path}")
        
        with open(self.policy_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _cache_lookups(self):
        """Pre-compute common lookups for performance"""
        # Null handling
        self.null_tokens = {
            token.lower() for token in 
            self.policy.get('null_handling', {}).get('null_tokens_case_insensitive', [])
        }
        self.whitespace_tokens = set(
            self.policy.get('null_handling', {}).get('whitespace_tokens', [])
        )
        self.null_token_regexes = [
            re.compile(pattern) for pattern in 
            self.policy.get('null_handling', {}).get('null_token_regexes', [])
        ]
        self.preserve_phrases = set(
            self.policy.get('null_handling', {}).get('preserve_phrases', [])
        )
        
        # Column role criticality
        self.role_criticality = {
            role: info.get('criticality', 'low')
            for role, info in self.policy.get('classification', {}).get('column_roles', {}).items()
        }
        
        # PII detection config
        self.pii_config = self.policy.get('pii_handling', {})
        
        # Type coercion rules
        self.safe_cast_matrix = {
            (rule['from'], rule['to']): rule.get('allowed_if')
            for rule in self.policy.get('type_coercion', {}).get('safe_cast_matrix', [])
        }
        
        # Parse thresholds
        self.parse_thresholds = self.policy.get('type_coercion', {}).get('parse_thresholds', {})
        
        # Range validations
        self.range_validations = self.policy.get('range_validations', {}).get('built_in', {})
        
        # Normalization rules
        self.normalization_rules = self.policy.get('normalization', {}).get('casing', {}).get('rules', [])
        
        # Governance principles
        self.principles = {
            p['id']: p['text']
            for p in self.policy.get('governance', {}).get('principles', [])
        }
    
    def get_criticality(self, role: str) -> str:
        """Get criticality level for a column role"""
        return self.role_criticality.get(role, 'low')
    
    def is_null_token(self, value: str) -> bool:
        """Check if value is a null token per policy"""
        if pd.isna(value):
            return True
        
        str_val = str(value).strip()
        
        # Check whitespace tokens
        if str_val in self.whitespace_tokens:
            return True
        
        # Check null tokens (case insensitive)
        if str_val.lower() in self.null_tokens:
            # Check if it's a preserved phrase
            if str_val in self.preserve_phrases:
                return False
            return True
            
        # Check regex matches
        for pattern in self.null_token_regexes:
            if pattern.match(str_val):
                if str_val in self.preserve_phrases:
                    return False
                return True
        
        return False
    
    def get_casing_rule(self, column_role: str, column_name: str) -> Optional[str]:
        """Get casing normalization rule for column"""
        for rule in self.normalization_rules:
            match_conditions = rule.get('match', {})
            
            # Check role match
            if 'column_role_in' in match_conditions:
                if column_role in match_conditions['column_role_in']:
                    return rule.get('action', {}).get('casing')
            
            # Check name regex match
            if 'column_name_regex' in match_conditions:
                pattern = match_conditions['column_name_regex']
                if re.search(pattern, column_name, re.IGNORECASE):
                    return rule.get('action', {}).get('casing')
        
        return self.policy.get('normalization', {}).get('casing', {}).get('default', 'preserve')
    
    def should_mask_pii(self, pii_type: str, classification: str = 'internal') -> bool:
        """Determine if PII should be masked based on policy"""
        if not self.pii_config.get('enabled', True):
            return False
        
        default_actions = self.pii_config.get('default_action_by_classification', {})
        action = default_actions.get(classification, 'mask')
        
        return action in ['mask', 'drop_or_mask']
    
    def get_pii_masking_method(self, pii_type: str) -> str:
        """Get masking method for PII type"""
        methods = self.pii_config.get('masking', {}).get('methods', {})
        return methods.get(pii_type, 'hash')
    
    def is_required_field(self, role: str) -> bool:
        """Check if field role is required per policy"""
        for constraint in self.policy.get('constraints', {}).get('required_fields', []):
            if constraint.get('role') == role and constraint.get('required', False):
                return True
        return False
    
    def get_failure_action(self, role: str, issue_type: str) -> Tuple[str, str]:
        """Get failure action and owner for a role/issue type"""
        # Check required fields constraints
        if issue_type == 'missing_required':
            for constraint in self.policy.get('constraints', {}).get('required_fields', []):
                if constraint.get('role') == role:
                    return constraint.get('failure_action', 'flag_only'), constraint.get('owner', 'human')
        
        # Check range validations
        if issue_type == 'range_violation':
            for validation_name, validation in self.range_validations.items():
                return validation.get('invalid_action', 'flag_only'), validation.get('owner', 'human')
        
        return 'flag_only', 'human'
    
    def can_apply_programmatic_fix(self, issue_type: str, owner: str) -> bool:
        """Determine if an issue can be fixed programmatically"""
        return owner == 'programmatic'


class DataProfiler:
    """Policy-driven data profiler with dynamic behavior"""
    
    def __init__(self, file_path, policy_path='../config/dq_policy_spec.yaml', 
                 sample_size=None, max_distinct_values=100, top_k=10, histogram_bins=10):
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.max_distinct_values = max_distinct_values
        self.top_k = top_k
        self.histogram_bins = histogram_bins
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Load policy engine
        self.policy = PolicyEngine(policy_path)
        
        # Track remediation actions
        self.remediation_log = []
        self.issues_detected = []
    
    def generate_id(self, prefix, *args):
        """Generate a unique ID based on prefix and arguments"""
        content = '_'.join(str(arg) for arg in args)
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_suffix}"
    
    def detect_file_type(self):
        """Detect if file is CSV or Excel"""
        suffix = self.file_path.suffix.lower()
        if suffix in ['.csv', '.txt']:
            return 'csv'
        elif suffix in ['.xlsx', '.xls', '.xlsm']:
            return 'excel'
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def read_data(self):
        """Read data from file, handling both CSV and Excel with multiple sheets"""
        file_type = self.detect_file_type()
        
        if file_type == 'csv':
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(self.file_path, encoding=encoding)
                    return {'Sheet1': df}
                except UnicodeDecodeError:
                    continue
            raise ValueError("Unable to read CSV with supported encodings")
        else:
            # Read all sheets from Excel
            excel_file = pd.ExcelFile(self.file_path)
            sheets = {}
            for sheet_name in excel_file.sheet_names:
                sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
            return sheets
    
    def classify_column_role(self, series: pd.Series, col_name: str, df: pd.DataFrame) -> str:
        """
        Classify column role according to policy classification.column_roles
        This follows the same logic as the DQ assessment but is policy-driven
        """
        col_lower = col_name.lower()
        dtype = series.dtype

        # Split column name into tokens for word-boundary matching
        # Handles snake_case, camelCase, and space-separated names
        import re
        col_tokens = set(re.split(r'[_\s]+|(?<=[a-z])(?=[A-Z])', col_lower))

        # Identifier detection
        if any(keyword in col_lower for keyword in ['id', 'key', 'uuid', 'guid']):
            return 'identifier'

        # Email = identifier (and PII)
        if 'email' in col_lower:
            return 'identifier'

        # Timestamp detection - use word boundary matching to avoid false positives
        # e.g., "Sentiment" contains "time" but is not a timestamp
        timestamp_keywords = {'date', 'time', 'timestamp', 'modified', 'created', 'datetime'}
        if col_tokens & timestamp_keywords:
            return 'timestamp'

        # Measure detection (numeric types)
        if pd.api.types.is_numeric_dtype(dtype):
            # But check for IDs that happen to be numeric
            if 'id' not in col_lower:
                return 'measure'
        
        # Status detection
        if 'status' in col_lower:
            return 'status'
        
        # Free text detection (long text fields)
        if dtype == 'object':
            sample = series.dropna().head(100).astype(str)
            if len(sample) > 0:
                avg_length = sample.str.len().mean()
                if avg_length > 100:  # Long text likely free text
                    return 'free_text'
        
        # Geo detection
        if any(keyword in col_lower for keyword in ['latitude', 'longitude', 'lat', 'lon', 'coord']):
            return 'geo'
        
        # Foreign key detection
        if 'fk' in col_lower or col_lower.endswith('_id'):
            return 'foreign_key'
        
        # Default to dimension
        return 'dimension'
    
    def detect_null_issues(self, series: pd.Series, col_name: str, role: str) -> List[Dict]:
        """Detect null-related issues using policy rules"""
        issues = []
        
        # Check for null tokens
        null_token_mask = series.apply(lambda x: self.policy.is_null_token(x))
        null_token_count = null_token_mask.sum()
        
        if null_token_count > 0:
            issues.append({
                'column': col_name,
                'issue_type': 'null_tokens_detected',
                'severity': 'warn',
                'count': int(null_token_count),
                'policy_section': 'null_handling.null_tokens_case_insensitive',
                'owner': 'programmatic',
                'action': 'convert_to_proper_null',
                'examples': series[null_token_mask].dropna().unique().tolist()[:5]
            })
        
        # Check for missing values in required fields
        if self.policy.is_required_field(role):
            null_count = series.isna().sum()
            if null_count > 0:
                action, owner = self.policy.get_failure_action(role, 'missing_required')
                issues.append({
                    'column': col_name,
                    'issue_type': 'missing_required_field',
                    'severity': 'error',
                    'count': int(null_count),
                    'policy_section': f'constraints.required_fields (role={role})',
                    'owner': owner,
                    'action': action,
                    'criticality': self.policy.get_criticality(role)
                })
        
        return issues
    
    def detect_whitespace_issues(self, series: pd.Series, col_name: str) -> Optional[Dict]:
        """Detect whitespace issues per policy"""
        if series.dtype != 'object':
            return None
        
        # Policy: trim_whitespace enabled?
        if not self.policy.policy.get('null_handling', {}).get('trim_whitespace', True):
            return None
        
        has_whitespace = series.apply(
            lambda x: isinstance(x, str) and (x != x.strip())
        ).sum()
        
        if has_whitespace > 0:
            return {
                'column': col_name,
                'issue_type': 'whitespace_not_trimmed',
                'severity': 'warn',
                'count': int(has_whitespace),
                'policy_section': 'null_handling.trim_whitespace',
                'owner': 'programmatic',
                'action': 'trim_whitespace'
            }
        
        return None
    
    def detect_type_issues(self, series: pd.Series, col_name: str, role: str) -> List[Dict]:
        """Detect type coercion issues per policy"""
        issues = []
        
        # Check if measure role but not numeric
        if role == 'measure' and series.dtype == 'object':
            # Try to parse as numeric
            numeric_parsed = pd.to_numeric(series, errors='coerce')
            parse_success_rate = numeric_parsed.notna().sum() / series.notna().sum() if series.notna().sum() > 0 else 0
            
            # Check against policy threshold
            min_threshold = self.parse_thresholds.get('date_time_parse_success_rate_min', 0.98)
            
            if parse_success_rate < min_threshold:
                issues.append({
                    'column': col_name,
                    'issue_type': 'measure_field_not_numeric',
                    'severity': 'error',
                    'count': int(series.notna().sum()),
                    'parse_success_rate': round(parse_success_rate, 4),
                    'threshold': min_threshold,
                    'policy_section': 'type_coercion.parse_thresholds',
                    'owner': 'human',
                    'action': 'reclassify_or_fix_data',
                    'details': f'Only {parse_success_rate*100:.1f}% parseable as numeric, below {min_threshold*100}% threshold'
                })
            else:
                issues.append({
                    'column': col_name,
                    'issue_type': 'type_coercion_available',
                    'severity': 'warn',
                    'count': int(series.notna().sum()),
                    'parse_success_rate': round(parse_success_rate, 4),
                    'policy_section': 'type_coercion.safe_cast_matrix',
                    'owner': 'programmatic',
                    'action': 'cast_to_numeric',
                    'details': f'{parse_success_rate*100:.1f}% parseable, meets threshold for safe cast'
                })
        
        # Check timestamp role
        if role == 'timestamp' and not pd.api.types.is_datetime64_any_dtype(series.dtype):
            issues.append({
                'column': col_name,
                'issue_type': 'timestamp_wrong_type',
                'severity': 'error',
                'count': len(series),
                'policy_section': 'classification.column_roles.timestamp',
                'owner': 'programmatic',
                'action': 'convert_to_datetime',
                'details': f'Current type: {series.dtype}, expected: datetime'
            })
        
        return issues
    
    def detect_pii_issues(self, series: pd.Series, col_name: str) -> Optional[Dict]:
        """Detect PII per policy rules"""
        col_lower = col_name.lower()
        pii_types = []
        
        # Email detection
        if 'email' in col_lower:
            non_null_count = series.notna().sum()
            if non_null_count > 0:
                pii_types.append('email')
                
                # Check if masking is required
                if self.policy.should_mask_pii('email'):
                    method = self.policy.get_pii_masking_method('email')
                    return {
                        'column': col_name,
                        'issue_type': 'unmasked_pii',
                        'severity': 'warn',
                        'count': int(non_null_count),
                        'pii_types': pii_types,
                        'policy_section': 'pii_handling.masking.methods.email',
                        'owner': 'programmatic',
                        'action': method,
                        'details': f'{non_null_count} email addresses detected'
                    }
        
        return None
    
    def detect_uniqueness_issues(self, series: pd.Series, col_name: str, role: str) -> Optional[Dict]:
        """Detect uniqueness issues per policy"""
        if role == 'identifier':
            total_rows = len(series)
            unique_count = series.nunique()
            
            if unique_count == total_rows and series.notna().all():
                # Potential primary key
                return {
                    'column': col_name,
                    'issue_type': 'primary_key_candidate',
                    'severity': 'info',
                    'count': total_rows,
                    'policy_section': 'constraints.uniqueness.inferred_pk_handling',
                    'owner': 'human',
                    'action': 'flag_for_human_confirmation',
                    'details': f'Column is 100% unique across {total_rows} rows'
                }
        
        return None
    
    def detect_casing_issues(self, series: pd.Series, col_name: str, role: str) -> Optional[Dict]:
        """Detect casing inconsistencies per policy normalization rules"""
        if series.dtype != 'object':
            return None
        
        casing_rule = self.policy.get_casing_rule(role, col_name)
        
        if casing_rule and casing_rule != 'preserve':
            return {
                'column': col_name,
                'issue_type': 'casing_inconsistency',
                'severity': 'info',
                'count': int(series.notna().sum()),
                'policy_section': f'normalization.casing.rules (role={role})',
                'owner': 'programmatic',
                'action': f'apply_{casing_rule}',
                'details': f'Should apply {casing_rule} per policy'
            }
        
        return None
    
    def profile_entity_with_policy(self, df: pd.DataFrame, entity_name: str, source_id: str) -> Dict:
        """Profile a single entity (sheet/table) with policy-driven analysis"""
        
        entity_id = self.generate_id('entity', entity_name)
        
        # Row summary
        row_summary = {
            'row_count': len(df),
            'sampled_rows': self.sample_size if self.sample_size and len(df) > self.sample_size else len(df)
        }
        
        # Profile each column with policy-based issue detection
        columns = []
        for col_name in df.columns:
            series = df[col_name]
            
            # Classify role per policy
            role = self.classify_column_role(series, col_name, df)
            criticality = self.policy.get_criticality(role)
            
            # Detect issues per policy
            issues = []
            
            # Null handling issues
            issues.extend(self.detect_null_issues(series, col_name, role))
            
            # Whitespace issues
            ws_issue = self.detect_whitespace_issues(series, col_name)
            if ws_issue:
                issues.append(ws_issue)
            
            # Type coercion issues
            issues.extend(self.detect_type_issues(series, col_name, role))
            
            # PII issues
            pii_issue = self.detect_pii_issues(series, col_name)
            if pii_issue:
                issues.append(pii_issue)
            
            # Uniqueness issues
            uniq_issue = self.detect_uniqueness_issues(series, col_name, role)
            if uniq_issue:
                issues.append(uniq_issue)
            
            # Casing issues
            case_issue = self.detect_casing_issues(series, col_name, role)
            if case_issue:
                issues.append(case_issue)
            
            # Track all issues
            self.issues_detected.extend(issues)
            
            # Basic statistics
            null_count = series.isna().sum()
            null_rate = null_count / len(series) if len(series) > 0 else 0
            
            column_profile = {
                'column_name': col_name,
                'column_id': self.generate_id('col', entity_name, col_name),
                'ordinal_position': df.columns.get_loc(col_name) + 1,
                'logical_type': str(series.dtype),
                'role': role,
                'criticality': criticality,
                'statistics': {
                    'sample_size': len(series),
                    'completeness': {
                        'null_count': int(null_count),
                        'null_rate': round(null_rate, 4)
                    },
                    'cardinality': {
                        'distinct_count': int(series.nunique()),
                        'distinct_rate': round(series.nunique() / len(series), 4) if len(series) > 0 else None
                    }
                },
                'policy_issues': issues,
                'programmatic_fixes_available': sum(1 for issue in issues if issue.get('owner') == 'programmatic'),
                'human_review_required': sum(1 for issue in issues if issue.get('owner') == 'human')
            }
            
            columns.append(column_profile)
        
        entity_profile = {
            'entity_id': entity_id,
            'entity_name': entity_name,
            'entity_type': 'table',
            'source_id': source_id,
            'row_summary': row_summary,
            'columns': columns,
            'total_issues': len([issue for col in columns for issue in col.get('policy_issues', [])]),
            'programmatic_fixes': sum(col.get('programmatic_fixes_available', 0) for col in columns),
            'human_reviews': sum(col.get('human_review_required', 0) for col in columns)
        }
        
        return entity_profile
    
    def generate_profile(self) -> Dict:
        """Generate complete policy-driven data profile"""
        sheets = self.read_data()
        
        source_id = self.generate_id('source', self.file_path.name)
        dataset_id = self.generate_id('dataset', self.file_path.stem)
        
        # Source information
        source = {
            'source_id': source_id,
            'source_type': self.detect_file_type(),
            'source_name': self.file_path.name,
            'connection_or_path': str(self.file_path),
            'ingested_at_utc': self.timestamp
        }
        
        # Profile each sheet/entity
        entities = []
        for sheet_name, df in sheets.items():
            entity = self.profile_entity_with_policy(df, sheet_name, source_id)
            entities.append(entity)
        
        # Build complete profile
        profile = {
            'profile_version': '2.0.0',
            'policy_version': self.policy.policy.get('policy_spec_version', 'unknown'),
            'policy_name': self.policy.policy.get('name', 'unknown'),
            'run': {
                'run_id': self.generate_id('run', self.timestamp),
                'timestamp_utc': self.timestamp,
                'engine': {
                    'name': 'PolicyDrivenDataProfiler',
                    'version': '2.0.0',
                    'policy_path': str(self.policy.policy_path)
                }
            },
            'dataset': {
                'dataset_id': dataset_id,
                'dataset_name': self.file_path.stem
            },
            'sources': [source],
            'entities': entities,
            'issues_summary': {
                'total_issues': len(self.issues_detected),
                'by_severity': {
                    'error': sum(1 for i in self.issues_detected if i.get('severity') == 'error'),
                    'warn': sum(1 for i in self.issues_detected if i.get('severity') == 'warn'),
                    'info': sum(1 for i in self.issues_detected if i.get('severity') == 'info')
                },
                'by_owner': {
                    'programmatic': sum(1 for i in self.issues_detected if i.get('owner') == 'programmatic'),
                    'human': sum(1 for i in self.issues_detected if i.get('owner') == 'human')
                }
            }
        }
        
        return profile


def main():
    parser = argparse.ArgumentParser(
        description='Policy-driven data profiler - automatically adapts to DQ Policy changes'
    )
    parser.add_argument('input_file', help='Path to input CSV or Excel file')
    parser.add_argument('-p', '--policy', default='config/dq_policy_spec.yaml',
                       help='Path to DQ Policy Spec YAML (default: config/dq_policy_spec.yaml)')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: <input>_profile.json)')
    parser.add_argument('-s', '--sample', type=int, help='Sample size (rows) for large datasets')
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = DataProfiler(
        args.input_file,
        policy_path=args.policy,
        sample_size=args.sample
    )
    
    print(f"Profiling {args.input_file}...")
    print(f"Using policy: {args.policy}")
    profile = profiler.generate_profile()
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.input_file).with_suffix('').with_suffix('.profile.json')
    
    # Write profile to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"\n✓ Profile saved to {output_path}")
    print(f"\nSummary:")
    print(f"  Policy: {profile['policy_name']} v{profile['policy_version']}")
    print(f"  Entities: {len(profile['entities'])}")
    print(f"  Total columns: {sum(len(e['columns']) for e in profile['entities'])}")
    print(f"  Total rows: {sum(e['row_summary']['row_count'] for e in profile['entities'])}")
    print(f"\nIssues Detected:")
    print(f"  Total: {profile['issues_summary']['total_issues']}")
    print(f"  ├─ Errors: {profile['issues_summary']['by_severity']['error']}")
    print(f"  ├─ Warnings: {profile['issues_summary']['by_severity']['warn']}")
    print(f"  └─ Info: {profile['issues_summary']['by_severity']['info']}")
    print(f"\nRemediation:")
    print(f"  Programmatic fixes available: {profile['issues_summary']['by_owner']['programmatic']}")
    print(f"  Human review required: {profile['issues_summary']['by_owner']['human']}")


if __name__ == '__main__':
    main()
