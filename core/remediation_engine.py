#!/usr/bin/env python3
"""
DQ Remediation Engine
Applies programmatic fixes identified by the policy-driven profiler.
Generates before/after comparison and human review report.
"""

import pandas as pd
import numpy as np
import json
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse
import re


# Import StandardOutputManager
try:
    from core.output_manager import StandardOutputManager
except ImportError:
    # Handle running as script from core directory or root
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from core.output_manager import StandardOutputManager


class RemediationEngine:
    """Applies programmatic data quality fixes per policy"""
    
    def __init__(self, policy_path: str = '../config/dq_policy_spec.yaml'):
        self.policy_path = Path(policy_path)
        with open(policy_path, 'r') as f:
            self.policy = yaml.safe_load(f)
        
        self.remediation_log = []
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
    
    def apply_trim_whitespace(self, df: pd.DataFrame, issues: List[Dict]) -> pd.DataFrame:
        """Trim whitespace from string columns per policy"""
        df_remediated = df.copy()
        
        for issue in issues:
            if issue['issue_type'] == 'whitespace_not_trimmed':
                col = issue['column']
                before_sample = df[col].dropna().head(5).tolist()
                
                # Apply trim
                df_remediated[col] = df_remediated[col].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )
                
                after_sample = df_remediated[col].dropna().head(5).tolist()
                
                self.remediation_log.append({
                    'column': col,
                    'action': 'trim_whitespace',
                    'affected_count': issue['count'],
                    'before_examples': before_sample,
                    'after_examples': after_sample,
                    'policy_section': issue['policy_section']
                })
        
        return df_remediated
    
    def apply_null_token_conversion(self, df: pd.DataFrame, issues: List[Dict]) -> pd.DataFrame:
        """Convert null tokens to proper NULL values per policy"""
        df_remediated = df.copy()

        null_tokens = set(
            self.policy.get('null_handling', {}).get('null_tokens_case_insensitive', [])
        )
        null_tokens = {token.lower() for token in null_tokens}

        # Also get whitespace tokens per policy (includes empty string "")
        whitespace_tokens = set(
            self.policy.get('null_handling', {}).get('whitespace_tokens', [])
        )

        null_token_regexes = [
            re.compile(pattern) for pattern in 
            self.policy.get('null_handling', {}).get('null_token_regexes', [])
        ]

        def is_null_value(x):
            """Check if value should be converted to NULL per policy"""
            if not isinstance(x, str):
                return False
            # Check if value is a whitespace token (before stripping)
            if x in whitespace_tokens:
                return True
            stripped = x.strip()
            # Check if stripped value is empty or a whitespace token
            if stripped == '' or stripped in whitespace_tokens:
                return True
            # Check if stripped value is a null token (case insensitive)
            if stripped.lower() in null_tokens:
                return True
                
            # Check regex matches
            for pattern in null_token_regexes:
                if pattern.match(stripped):
                    return True
            return False

        for issue in issues:
            if issue['issue_type'] == 'null_tokens_detected':
                col = issue['column']
                before_sample = issue.get('examples', [])

                # Convert null tokens and whitespace tokens to NaN
                df_remediated[col] = df_remediated[col].apply(
                    lambda x: np.nan if is_null_value(x) else x
                )
                
                self.remediation_log.append({
                    'column': col,
                    'action': 'convert_null_tokens',
                    'affected_count': issue['count'],
                    'before_examples': before_sample,
                    'after_examples': ['NULL'] * min(len(before_sample), 3),
                    'policy_section': issue['policy_section']
                })
        
        return df_remediated
    
    def apply_type_coercion(self, df: pd.DataFrame, issues: List[Dict]) -> pd.DataFrame:
        """Apply safe type coercions per policy"""
        df_remediated = df.copy()
        
        for issue in issues:
            if issue['issue_type'] == 'type_coercion_available':
                col = issue['column']
                before_dtype = df[col].dtype
                before_sample = df[col].dropna().head(3).tolist()
                
                # Apply numeric coercion
                df_remediated[col] = pd.to_numeric(df_remediated[col], errors='coerce')
                
                after_sample = df_remediated[col].dropna().head(3).tolist()
                
                self.remediation_log.append({
                    'column': col,
                    'action': 'cast_to_numeric',
                    'affected_count': issue['count'],
                    'before_dtype': str(before_dtype),
                    'after_dtype': str(df_remediated[col].dtype),
                    'before_examples': before_sample,
                    'after_examples': after_sample,
                    'policy_section': issue['policy_section']
                })
            
            elif issue['issue_type'] == 'timestamp_wrong_type':
                col = issue['column']
                before_dtype = df[col].dtype
                
                # Try to convert to datetime
                # Handle Excel serial dates (float)
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    df_remediated[col] = pd.to_datetime(df_remediated[col], unit='D', origin='1899-12-30', errors='coerce')
                else:
                    df_remediated[col] = pd.to_datetime(df_remediated[col], errors='coerce')
                
                self.remediation_log.append({
                    'column': col,
                    'action': 'convert_to_datetime',
                    'affected_count': issue['count'],
                    'before_dtype': str(before_dtype),
                    'after_dtype': str(df_remediated[col].dtype),
                    'policy_section': issue['policy_section']
                })
        
        return df_remediated
    
    def apply_pii_masking(self, df: pd.DataFrame, issues: List[Dict]) -> pd.DataFrame:
        """Mask PII per policy"""
        df_remediated = df.copy()

        for issue in issues:
            if issue['issue_type'] == 'unmasked_pii':
                col = issue['column']
                method = issue['action']
                pii_types = issue.get('pii_types', [])
                before_sample = df[col].dropna().head(3).tolist()
                affected_count = 0

                if method == 'hash':
                    # Determine if this is an email column (by pii_types OR column name)
                    col_lower = col.lower()
                    is_email_column = 'email' in pii_types or 'email' in col_lower

                    if is_email_column:
                        # For email columns, only hash values that actually look like emails (contain @)
                        def hash_if_email(x):
                            if pd.notna(x) and isinstance(x, str) and '@' in x:
                                return hashlib.sha256(x.encode()).hexdigest()[:16]
                            return x
                        df_remediated[col] = df_remediated[col].apply(hash_if_email)
                        # Count how many were actually hashed
                        affected_count = df[col].apply(
                            lambda x: pd.notna(x) and isinstance(x, str) and '@' in x
                        ).sum()
                    else:
                        # For other PII types, hash all non-null values
                        df_remediated[col] = df_remediated[col].apply(
                            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] if pd.notna(x) else x
                        )
                        affected_count = df[col].notna().sum()
                elif method == 'drop':
                    # Remove the column
                    df_remediated = df_remediated.drop(columns=[col])
                    affected_count = issue['count']

                after_sample = df_remediated[col].dropna().head(3).tolist() if method != 'drop' else ['DROPPED']

                self.remediation_log.append({
                    'column': col,
                    'action': f'mask_pii_{method}',
                    'affected_count': int(affected_count),
                    'before_examples': before_sample,
                    'after_examples': after_sample,
                    'policy_section': issue['policy_section'],
                    'pii_types': pii_types
                })
        
        return df_remediated
    
    def apply_casing_normalization(self, df: pd.DataFrame, issues: List[Dict]) -> pd.DataFrame:
        """Apply casing normalization per policy"""
        df_remediated = df.copy()
        
        for issue in issues:
            if issue['issue_type'] == 'casing_inconsistency':
                col = issue['column']
                action = issue['action']
                before_sample = df[col].dropna().head(3).tolist()
                
                if 'titlecase' in action:
                    df_remediated[col] = df_remediated[col].apply(
                        lambda x: x.title() if isinstance(x, str) else x
                    )
                elif 'lowercase' in action:
                    df_remediated[col] = df_remediated[col].apply(
                        lambda x: x.lower() if isinstance(x, str) else x
                    )
                elif 'uppercase' in action:
                    df_remediated[col] = df_remediated[col].apply(
                        lambda x: x.upper() if isinstance(x, str) else x
                    )
                
                after_sample = df_remediated[col].dropna().head(3).tolist()
                
                self.remediation_log.append({
                    'column': col,
                    'action': action,
                    'affected_count': issue['count'],
                    'before_examples': before_sample,
                    'after_examples': after_sample,
                    'policy_section': issue['policy_section']
                })
        
        return df_remediated

    def apply_global_email_masking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safety net: Scan ALL columns for any values containing @ and mask them.
        This catches email addresses in unexpected columns (e.g., free text, comments).
        """
        df_remediated = df.copy()

        for col in df_remediated.columns:
            # Only process string/object columns (check for object, string, or StringDtype)
            col_dtype = str(df_remediated[col].dtype)
            if col_dtype == 'object' or col_dtype == 'string' or 'str' in col_dtype.lower():
                # Find values containing @
                mask = df_remediated[col].apply(
                    lambda x: isinstance(x, str) and '@' in x
                )

                if mask.any():
                    before_sample = df_remediated.loc[mask, col].head(3).tolist()
                    affected_count = mask.sum()

                    # Hash values containing @
                    df_remediated.loc[mask, col] = df_remediated.loc[mask, col].apply(
                        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
                    )

                    after_sample = df_remediated.loc[mask, col].head(3).tolist()

                    self.remediation_log.append({
                        'column': col,
                        'action': 'global_email_mask',
                        'affected_count': int(affected_count),
                        'before_examples': before_sample,
                        'after_examples': after_sample,
                        'policy_section': 'pii_handling.global_scan'
                    })

        return df_remediated

    def remediate_dataframe(self, df: pd.DataFrame, profile: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Apply all programmatic fixes to dataframe based on profile
        Returns: (remediated_df, human_review_issues)
        """
        df_remediated = df.copy()
        human_review_issues = []
        
        # Extract all issues from profile
        all_issues = []
        for entity in profile.get('entities', []):
            for col in entity.get('columns', []):
                all_issues.extend(col.get('policy_issues', []))
        
        # Separate programmatic vs human issues
        programmatic_issues = [i for i in all_issues if i.get('owner') == 'programmatic']
        human_review_issues = [i for i in all_issues if i.get('owner') == 'human']
        
        # Apply fixes in order per policy
        # 1. Trim whitespace (must be first to clean data)
        df_remediated = self.apply_trim_whitespace(df_remediated, programmatic_issues)
        
        # 2. Convert null tokens
        df_remediated = self.apply_null_token_conversion(df_remediated, programmatic_issues)
        
        # 3. Type coercion
        df_remediated = self.apply_type_coercion(df_remediated, programmatic_issues)
        
        # 4. PII masking
        df_remediated = self.apply_pii_masking(df_remediated, programmatic_issues)
        
        # 5. Casing normalization
        df_remediated = self.apply_casing_normalization(df_remediated, programmatic_issues)

        # 6. Global email masking (safety net for emails in any column)
        df_remediated = self.apply_global_email_masking(df_remediated)

        return df_remediated, human_review_issues
    
    def generate_remediation_summary(self, original_df: pd.DataFrame, 
                                    remediated_df: pd.DataFrame,
                                    human_review_issues: List[Dict]) -> Dict:
        """Generate summary of remediation actions"""
        return {
            'timestamp_utc': self.timestamp,
            'policy_version': self.policy.get('policy_spec_version', 'unknown'),
            'original_shape': original_df.shape,
            'remediated_shape': remediated_df.shape,
            'actions_applied': len(self.remediation_log),
            'remediation_log': self.remediation_log,
            'human_review_required': {
                'count': len(human_review_issues),
                'issues': human_review_issues
            },
            'governance': {
                'audit_enabled': self.policy.get('governance', {}).get('audit', {}).get('enabled', True),
                'principles_applied': list(self.policy.get('governance', {}).get('principles', []))
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description='Apply programmatic DQ fixes based on policy-driven profile'
    )
    parser.add_argument('input_file', help='Original data file (CSV or Excel)')
    parser.add_argument('profile_file', help='Profile JSON from data_profiler_v2.py')
    parser.add_argument('-p', '--policy', default='config/dq_policy_spec.yaml',
                       help='Path to DQ Policy Spec YAML')
    parser.add_argument('-o', '--output-dir', default='output',
                       help='Output directory for remediated files (default: output)')
    
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    # Initialize message
    print(f"Reading data from {args.input_file}...")
    
    # Read original data
    file_path = Path(args.input_file)
    if file_path.suffix.lower() == '.csv':
        df_original = pd.read_csv(file_path)
    else:
        df_original = pd.read_excel(file_path)
    
    # Read profile
    print(f"Reading profile from {args.profile_file}...")
    with open(args.profile_file, 'r') as f:
        profile = json.load(f)
    
    # Create remediation engine
    engine = RemediationEngine(policy_path=args.policy)
    
    print(f"\nApplying programmatic fixes...")
    df_remediated, human_review_issues = engine.remediate_dataframe(df_original, profile)
    
    # Generate summary
    summary = engine.generate_remediation_summary(df_original, df_remediated, human_review_issues)
    
    # --- OUTPUT MANAGEMENT ---
    output_manager = StandardOutputManager(base_output_dir=args.output_dir)
    run_dir = output_manager.create_run_directory(file_path.name)
    
    print(f"\nSaving outputs to {run_dir}...")
    
    # Save artifacts using manager
    original_path = output_manager.save_original(file_path, df_original)
    remediated_path = output_manager.save_remediated(df_remediated, file_path.name)
    summary_path = output_manager.save_summary(summary)
    # Also save the profile for completeness
    output_manager.save_profile(profile)
    
    print(f"✓ Original saved: {original_path}")
    print(f"✓ Remediated saved: {remediated_path}")
    print(f"✓ Summary saved: {summary_path}")
    
    print(f"\n{'='*60}")
    print("REMEDIATION SUMMARY")
    print(f"{'='*60}")
    print(f"Actions applied: {summary['actions_applied']}")
    print(f"Human review required: {summary['human_review_required']['count']}")
    
    if summary['remediation_log']:
        print(f"\nActions taken:")
        for log_entry in summary['remediation_log']:
            print(f"  • {log_entry['column']}: {log_entry['action']} ({log_entry['affected_count']} values)")
    
    if human_review_issues:
        print(f"\n⚠ Issues requiring human review:")
        for issue in human_review_issues[:10]:  # Show first 10
            print(f"  • {issue['column']}: {issue['issue_type']} ({issue['severity']})")
        if len(human_review_issues) > 10:
            print(f"  ... and {len(human_review_issues) - 10} more (see summary file)")
    
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
