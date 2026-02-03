# Policy-Driven Data Quality System

## Overview

This system provides **policy-driven data profiling and remediation** that automatically adapts to changes in your Data Quality Policy Specification without requiring code changes.

## Key Innovation: Dynamic Policy Interpretation

**The Problem We Solved:**
Traditional data quality tools hardcode rules in application logic. When policies change, code must be rewritten and redeployed.

**Our Solution:**
The `PolicyEngine` class dynamically reads and interprets the `dq_policy_spec.yaml` file at runtime. When you update the policy YAML, the profiler automatically enforces the new rules without any code changes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DQ Policy Spec (YAML)                     │
│  Single source of truth for all data quality rules          │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ reads dynamically
                              │
┌─────────────────────────────────────────────────────────────┐
│                       PolicyEngine                           │
│  Interprets policy rules and provides them to profiler      │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
┌───────────────────┐                    ┌──────────────────────┐
│  DataProfiler v2  │                    │ RemediationEngine    │
│  - Classifies     │                    │ - Applies fixes      │
│  - Detects issues │                    │ - Logs changes       │
│  - Reports        │                    │ - Audits             │
└───────────────────┘                    └──────────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                    ┌──────────────────────┐
│  Profile JSON     │                    │ Remediated CSV       │
│  + Issue Manifest │                    │ + Change Log         │
└───────────────────┘                    └──────────────────────┘
```

## Components

### 1. **PolicyEngine** (in `data_profiler_v2.py`)
- Loads and parses `dq_policy_spec.yaml`
- Provides methods to query policy rules
- Caches lookups for performance
- **Zero code changes needed when policy updates**

### 2. **DataProfiler v2** (in `data_profiler_v2.py`)
- Profiles data according to policy specifications
- Classifies columns by role (identifier, measure, dimension, etc.)
- Detects policy violations dynamically
- Determines which issues are programmatic vs. human-required
- Outputs detailed JSON profile with issue manifest

### 3. **RemediationEngine** (in `dq_remediation_engine.py`)
- Applies programmatic fixes identified in profile
- Follows policy-defined remediation order
- Logs all changes with before/after examples
- Flags issues requiring human review
- Outputs remediated data + audit log

## How Policy Changes Propagate

### Example: Changing Null Token Handling

**Before (in policy YAML):**
```yaml
null_handling:
  null_tokens_case_insensitive:
    - "null"
    - "n/a"
    - "na"
```

**After (updated policy):**
```yaml
null_handling:
  null_tokens_case_insensitive:
    - "null"
    - "n/a"
    - "na"
    - "none"      # ADDED
    - "missing"   # ADDED
```

**Result:**
- No code changes needed
- Next profiler run automatically detects "none" and "missing" as null tokens
- Remediation engine automatically converts them to proper NULLs
- All according to the updated policy

### Example: Changing Field Criticality

**Before:**
```yaml
classification:
  column_roles:
    dimension: {criticality: "medium"}
```

**After:**
```yaml
classification:
  column_roles:
    dimension: {criticality: "high"}  # CHANGED
```

**Result:**
- Profiler now treats dimension fields as high criticality
- Missing dimension values trigger different actions
- All without code changes

## Usage

### Step 1: Profile Data Against Policy

```bash
python data_profiler_v2.py data.csv \
    --policy config/dq_policy_spec.yaml \
    --output data_profile.json
```

**Output:**
```json
{
  "policy_version": "0.1.0",
  "policy_name": "AI-Ready Data Quality Remediation Policy",
  "entities": [{
    "columns": [{
      "column_name": "Email",
      "role": "identifier",
      "criticality": "critical",
      "policy_issues": [{
        "issue_type": "unmasked_pii",
        "owner": "programmatic",
        "action": "hash",
        "count": 5849
      }]
    }]
  }],
  "issues_summary": {
    "by_owner": {
      "programmatic": 6,
      "human": 4
    }
  }
}
```

### Step 2: Apply Programmatic Fixes

```bash
python dq_remediation_engine.py data.csv data_profile.json \
    --policy config/dq_policy_spec.yaml \
    --output-dir output/
```

**Output:**
```
output/
└── 2025-02-03_14-30-00/
    ├── original_data.csv
    ├── remediated_data.csv
    └── remediation_summary.json
```

### Step 3: Review Human-Required Issues

Check `remediation_summary.json` for issues flagged for human review:

```json
{
  "human_review_required": {
    "count": 4,
    "issues": [
      {
        "column": "Numerator",
        "issue_type": "measure_field_not_numeric",
        "severity": "error",
        "owner": "human",
        "action": "reclassify_or_fix_data"
      }
    ]
  }
}
```

## Policy-Driven Features

### What Automatically Updates Without Code Changes:

✅ **Null token definitions** - Add/remove tokens considered as NULL
✅ **Whitespace handling** - Enable/disable trimming
✅ **Column role classification** - Change what roles exist and their criticality
✅ **Type coercion thresholds** - Adjust parse success rates
✅ **PII masking methods** - Change hash/tokenize/drop per PII type
✅ **Casing normalization** - Define rules per role or column pattern
✅ **Required field enforcement** - Mark roles as required/optional
✅ **Failure actions** - Define quarantine vs. flag vs. remediate
✅ **Range validations** - Add custom range rules for measures
✅ **Ownership assignment** - Decide what's programmatic vs. human

### Key Policy Sections Used:

| Policy Section | Profiler Use | Remediation Use |
|----------------|--------------|-----------------|
| `null_handling` | Detects null tokens, whitespace | Converts to proper NULL |
| `classification.column_roles` | Classifies columns, assigns criticality | Determines required fields |
| `type_coercion` | Identifies type mismatches | Applies safe casts |
| `pii_handling` | Detects PII | Masks/hashes/drops PII |
| `normalization.casing` | Detects inconsistencies | Applies titlecase/lowercase |
| `constraints` | Flags missing required fields | Quarantines invalid rows |
| `governance.principles` | Follows "prefer missing over misleading" | Logs all changes |

## Comparison: Old vs. New

### Old Approach (`data_profiler.py`)

```python
# HARDCODED - requires code changes to modify
def detect_pii(self, series, col_name):
    if 'email' in col_name.lower():
        pii_types.append('email')
    if 'ssn' in col_lower or 'social' in col_lower:
        pii_types.append('ssn')
```

**Problem:** To add new PII detection, you must modify code and redeploy.

### New Approach (`data_profiler_v2.py`)

```python
# POLICY-DRIVEN - reads from YAML
def detect_pii_issues(self, series: pd.Series, col_name: str):
    if self.policy.should_mask_pii('email'):
        method = self.policy.get_pii_masking_method('email')
        # ...apply per policy
```

**Benefit:** Update YAML policy, no code changes needed.

## Configuration Files

### Required Structure:

```
your-project/
├── config/
│   └── dq_policy_spec.yaml          # Policy source of truth
├── data_profiler_v2.py               # Policy-driven profiler
├── dq_remediation_engine.py          # Policy-driven remediation
└── output/                           # Remediated data + logs
    └── YYYY-MM-DD_HH-MM-SS/
        ├── original_data.csv
        ├── remediated_data.csv
        └── remediation_summary.json
```

## Advanced Usage

### Custom Policy Validation

```python
from data_profiler_v2 import PolicyEngine

# Validate policy before using
policy = PolicyEngine('config/dq_policy_spec.yaml')

# Check what null tokens are defined
print(policy.null_tokens)
# {'null', 'n/a', 'na', 'none', 'nan'}

# Check criticality of a role
print(policy.get_criticality('identifier'))
# 'critical'

# Check if field is required
print(policy.is_required_field('timestamp'))
# True
```

### Extending with Custom Rules

To add new policy-driven behavior:

1. **Update policy YAML** with new section
2. **Add reader method in PolicyEngine**
3. **Add detector in DataProfiler**
4. **Add fixer in RemediationEngine**

Example - Adding custom business rules:

```yaml
# In dq_policy_spec.yaml
business_rules:
  order_validation:
    enabled: true
    rules:
      - if_column: "OrderAmount"
        must_be: "positive"
      - if_column: "OrderDate"
        must_be: "not_future"
```

```python
# In PolicyEngine
def get_business_rules(self):
    return self.policy.get('business_rules', {})

# In DataProfiler
def detect_business_rule_violations(self, series, col_name):
    rules = self.policy.get_business_rules()
    # Check violations...
```

## Migration Guide

### From `data_profiler.py` to `data_profiler_v2.py`

1. **Copy your policy YAML** to `config/dq_policy_spec.yaml`
2. **Run new profiler** on existing data:
   ```bash
   python data_profiler_v2.py your_data.csv
   ```
3. **Compare outputs** - new profiler includes `policy_issues` section
4. **Apply remediation**:
   ```bash
   python dq_remediation_engine.py your_data.csv your_data_profile.json
   ```

### What Changed:

| Feature | Old | New |
|---------|-----|-----|
| Policy location | Hardcoded in Python | External YAML |
| Rule updates | Code changes | YAML edits |
| Issue detection | Manual coding | Policy-driven |
| Remediation | Not included | Automated engine |
| Audit logging | Minimal | Comprehensive |
| Human review | Unclear | Explicit flagging |

## Benefits

✅ **Maintainability** - Update policies without touching code
✅ **Consistency** - Single source of truth for DQ rules
✅ **Auditability** - All changes logged with policy references
✅ **Scalability** - Add new rules in minutes, not hours
✅ **Collaboration** - Business users can update policies
✅ **Version Control** - Policy changes tracked in Git
✅ **Testability** - Test different policies easily
✅ **Documentation** - Policy YAML serves as documentation

## Roadmap

- [ ] Add policy validation schema
- [ ] Support multiple data sources (Google Drive, S3, databases)
- [ ] Generate Word/PDF reports from remediation summary
- [ ] Add web UI for policy editing
- [ ] Integration with dbt for data pipelines
- [ ] Real-time policy compliance monitoring

## Contributing

When adding new policy-driven features:

1. Add policy definition to YAML
2. Add reader method to `PolicyEngine`
3. Add detection to `DataProfiler`
4. Add remediation to `RemediationEngine`
5. Update this README
6. Add tests

## License

[Your License]

## Questions?

The key principle: **If a rule can change, it belongs in the policy YAML, not in the code.**
