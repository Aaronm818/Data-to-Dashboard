#!/usr/bin/env python3
"""
Generic Data Profiler
Analyzes CSV/Excel files and generates comprehensive data profiles following the Data Profile Template.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import argparse


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


class DataProfiler:
    def __init__(self, file_path, sample_size=None, max_distinct_values=100, top_k=10, histogram_bins=10):
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.max_distinct_values = max_distinct_values
        self.top_k = top_k
        self.histogram_bins = histogram_bins
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
        
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
    
    def infer_logical_type(self, series, col_name):
        """Infer logical data type from pandas series"""
        dtype = series.dtype
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return 'datetime'
        
        # Check for numeric types
        if pd.api.types.is_integer_dtype(dtype):
            return 'integer'
        if pd.api.types.is_float_dtype(dtype):
            return 'decimal'
        if pd.api.types.is_bool_dtype(dtype):
            return 'boolean'
        
        # For object/string types, try to infer more specific types
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            # Try to parse as datetime
            non_null = series.dropna()
            if len(non_null) > 0:
                try:
                    pd.to_datetime(non_null.head(100), errors='raise')
                    return 'datetime'
                except:
                    pass
            
            return 'string'
        
        return 'unknown'
    
    def detect_semantic_patterns(self, series):
        """Detect semantic patterns like email, phone, currency, etc."""
        patterns = {}
        non_null = series.dropna().astype(str)
        
        if len(non_null) == 0:
            return patterns
        
        sample = non_null.head(1000)
        
        # Email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        email_matches = sample.str.match(email_pattern).sum()
        if email_matches / len(sample) > 0.8:
            patterns['is_email'] = True
        
        # Phone pattern (various formats)
        phone_pattern = r'^[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]$'
        phone_matches = sample.str.match(phone_pattern).sum()
        if phone_matches / len(sample) > 0.8:
            patterns['is_phone'] = True
        
        # Currency pattern
        currency_pattern = r'^[\$£€¥]?[\d,]+\.?\d{0,2}$'
        currency_matches = sample.str.match(currency_pattern).sum()
        if currency_matches / len(sample) > 0.8:
            patterns['is_currency'] = True
        
        # Percentage pattern
        pct_pattern = r'^\d+\.?\d*%$'
        pct_matches = sample.str.match(pct_pattern).sum()
        if pct_matches / len(sample) > 0.8:
            patterns['is_percentage'] = True
        
        # Latitude/Longitude
        try:
            numeric_values = pd.to_numeric(sample, errors='coerce').dropna()
            if len(numeric_values) > len(sample) * 0.8:
                if numeric_values.between(-90, 90).all():
                    patterns['is_latitude'] = True
                elif numeric_values.between(-180, 180).all():
                    patterns['is_longitude'] = True
        except:
            pass
        
        return patterns
    
    def detect_pii(self, series, col_name):
        """Detect potential PII in column"""
        pii_types = []
        confidence = 0.0
        
        col_lower = col_name.lower()
        
        # Name patterns
        if any(keyword in col_lower for keyword in ['name', 'first', 'last', 'full']):
            pii_types.append('name')
            confidence = max(confidence, 0.8)
        
        # Email detection
        non_null = series.dropna().astype(str).head(1000)
        if len(non_null) > 0:
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            if non_null.str.contains(email_pattern, regex=True).sum() / len(non_null) > 0.5:
                pii_types.append('email')
                confidence = max(confidence, 0.9)
        
        # Phone detection
        if any(keyword in col_lower for keyword in ['phone', 'tel', 'mobile', 'cell']):
            pii_types.append('phone')
            confidence = max(confidence, 0.7)
        
        # SSN pattern
        if 'ssn' in col_lower or 'social' in col_lower:
            pii_types.append('ssn')
            confidence = max(confidence, 0.9)
        
        # IP address
        if len(non_null) > 0:
            ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            if non_null.str.match(ip_pattern).sum() / len(non_null) > 0.5:
                pii_types.append('ip')
                confidence = max(confidence, 0.9)
        
        contains_pii = len(pii_types) > 0
        
        return {
            'contains_pii': contains_pii,
            'pii_types': pii_types,
            'confidence': confidence if contains_pii else None
        }
    
    def infer_column_role(self, series, col_name, df):
        """Infer the role/purpose of a column"""
        roles = []
        col_lower = col_name.lower()
        
        # Identifier hints
        if any(keyword in col_lower for keyword in ['id', 'key', 'uuid', 'guid']):
            roles.append('identifier')
        
        # Check for uniqueness
        if series.nunique() == len(series):
            if 'identifier' not in roles:
                roles.append('identifier')
        
        # Timestamp hints
        if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'created', 'updated', 'modified']):
            roles.append('timestamp')
        
        # Status/category hints
        if series.nunique() < 20 and series.dtype == 'object':
            if any(keyword in col_lower for keyword in ['status', 'state', 'type', 'category', 'flag']):
                roles.append('status')
        
        # Measure hints (numeric)
        if pd.api.types.is_numeric_dtype(series):
            if any(keyword in col_lower for keyword in ['amount', 'total', 'sum', 'count', 'price', 'cost', 'revenue']):
                roles.append('measure')
            elif 'identifier' not in roles:
                roles.append('measure')
        
        # Dimension hints
        if series.dtype == 'object' and series.nunique() < len(series) * 0.5:
            roles.append('dimension')
        
        # Free text hints
        if series.dtype == 'object':
            avg_length = series.astype(str).str.len().mean()
            if avg_length > 100:
                roles.append('free_text')
        
        # Geo hints
        if any(keyword in col_lower for keyword in ['lat', 'lon', 'longitude', 'latitude', 'country', 'city', 'state', 'zip', 'postal']):
            roles.append('geo')
        
        return roles if roles else ['dimension']
    
    def calculate_column_stats(self, series, col_name, df):
        """Calculate comprehensive statistics for a column"""
        stats = {
            'completeness': {},
            'distinctness': {},
            'frequency': {},
            'numeric': {},
            'datetime': {},
            'string': {}
        }
        
        total_count = len(series)
        null_count = series.isna().sum()
        non_null_series = series.dropna()
        
        # Completeness
        stats['completeness'] = {
            'null_count': int(null_count),
            'null_rate': round(null_count / total_count, 4) if total_count > 0 else None,
            'empty_string_count': int((series == '').sum()) if series.dtype == 'object' else 0
        }
        
        # Distinctness
        distinct_count = series.nunique()
        stats['distinctness'] = {
            'distinct_count': int(distinct_count),
            'distinct_rate': round(distinct_count / total_count, 4) if total_count > 0 else None,
            'is_unique': distinct_count == total_count,
            'cardinality_category': self.categorize_cardinality(distinct_count, total_count)
        }
        
        # Frequency (top K values)
        if distinct_count <= self.max_distinct_values and len(non_null_series) > 0:
            value_counts = non_null_series.value_counts().head(self.top_k)
            stats['frequency']['top_k'] = [
                {
                    'value': str(val),
                    'count': int(count),
                    'rate': round(count / total_count, 4)
                }
                for val, count in value_counts.items()
            ]
        else:
            stats['frequency']['top_k'] = []
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) > 0:
                q1 = numeric_series.quantile(0.25)
                q3 = numeric_series.quantile(0.75)
                iqr = q3 - q1
                outlier_count = ((numeric_series < (q1 - 1.5 * iqr)) | (numeric_series > (q3 + 1.5 * iqr))).sum()
                
                stats['numeric'] = {
                    'min': float(numeric_series.min()),
                    'max': float(numeric_series.max()),
                    'mean': float(numeric_series.mean()),
                    'median': float(numeric_series.median()),
                    'stddev': float(numeric_series.std()),
                    'q1': float(q1),
                    'q3': float(q3),
                    'outlier_count': int(outlier_count),
                    'histogram': self.create_histogram(numeric_series)
                }
        
        # Datetime statistics
        datetime_series = None
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_series = series.dropna()
        else:
            try:
                datetime_series = pd.to_datetime(non_null_series, errors='coerce').dropna()
                if len(datetime_series) < len(non_null_series) * 0.5:
                    datetime_series = None
            except:
                pass
        
        if datetime_series is not None and len(datetime_series) > 0:
            is_monotonic = datetime_series.is_monotonic_increasing or datetime_series.is_monotonic_decreasing
            stats['datetime'] = {
                'min': datetime_series.min().isoformat() if pd.notna(datetime_series.min()) else None,
                'max': datetime_series.max().isoformat() if pd.notna(datetime_series.max()) else None,
                'most_common_granularity': self.detect_datetime_granularity(datetime_series),
                'is_monotonic': bool(is_monotonic)
            }
        
        # String statistics
        if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
            string_series = non_null_series.astype(str)
            if len(string_series) > 0:
                lengths = string_series.str.len()
                stats['string'] = {
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'avg_length': round(lengths.mean(), 2),
                    'case_consistency': self.detect_case_consistency(string_series),
                    'regex_signatures': self.detect_regex_patterns(string_series)
                }
        
        return stats
    
    def categorize_cardinality(self, distinct_count, total_count):
        """Categorize cardinality of a column"""
        if distinct_count == total_count:
            return 'unique'
        elif distinct_count == 1:
            return 'constant'
        
        ratio = distinct_count / total_count if total_count > 0 else 0
        if ratio > 0.95:
            return 'high'
        elif ratio > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def create_histogram(self, numeric_series):
        """Create histogram bins for numeric data"""
        if len(numeric_series) == 0:
            return []
        
        try:
            counts, bin_edges = np.histogram(numeric_series, bins=self.histogram_bins)
            histogram = []
            for i in range(len(counts)):
                histogram.append({
                    'bin_start': float(bin_edges[i]),
                    'bin_end': float(bin_edges[i + 1]),
                    'count': int(counts[i])
                })
            return histogram
        except:
            return []
    
    def detect_datetime_granularity(self, datetime_series):
        """Detect the most common granularity in datetime data"""
        if len(datetime_series) < 2:
            return None
        
        # Check differences between consecutive timestamps
        sorted_dt = datetime_series.sort_values()
        diffs = sorted_dt.diff().dropna()
        
        if len(diffs) == 0:
            return None
        
        median_diff = diffs.median()
        
        if median_diff < pd.Timedelta(seconds=1):
            return 'second'
        elif median_diff < pd.Timedelta(minutes=1):
            return 'second'
        elif median_diff < pd.Timedelta(hours=1):
            return 'minute'
        elif median_diff < pd.Timedelta(days=1):
            return 'hour'
        elif median_diff < pd.Timedelta(days=7):
            return 'day'
        elif median_diff < pd.Timedelta(days=31):
            return 'week'
        elif median_diff < pd.Timedelta(days=92):
            return 'month'
        elif median_diff < pd.Timedelta(days=366):
            return 'quarter'
        else:
            return 'year'
    
    def detect_case_consistency(self, string_series):
        """Detect case consistency in string data"""
        sample = string_series.head(1000)
        
        lower_count = sample.str.islower().sum()
        upper_count = sample.str.isupper().sum()
        title_count = sample.str.istitle().sum()
        
        total = len(sample)
        if lower_count / total > 0.9:
            return 'lowercase'
        elif upper_count / total > 0.9:
            return 'uppercase'
        elif title_count / total > 0.9:
            return 'titlecase'
        elif lower_count / total > 0.5 or upper_count / total > 0.5:
            return 'consistent'
        else:
            return 'mixed'
    
    def detect_regex_patterns(self, string_series):
        """Detect common regex patterns in string data"""
        patterns = []
        sample = string_series.head(1000)
        
        # Email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        email_match = sample.str.match(email_pattern).sum()
        if email_match / len(sample) > 0.5:
            patterns.append({'pattern_name': 'email', 'match_rate': round(email_match / len(sample), 4)})
        
        # Phone pattern
        phone_pattern = r'^[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]$'
        phone_match = sample.str.match(phone_pattern).sum()
        if phone_match / len(sample) > 0.5:
            patterns.append({'pattern_name': 'phone', 'match_rate': round(phone_match / len(sample), 4)})
        
        # Alphanumeric ID
        id_pattern = r'^[A-Z0-9\-_]+$'
        id_match = sample.str.match(id_pattern, case=False).sum()
        if id_match / len(sample) > 0.8:
            patterns.append({'pattern_name': 'alphanumeric_id', 'match_rate': round(id_match / len(sample), 4)})
        
        return patterns
    
    def profile_entity(self, df, entity_name, source_id):
        """Profile a single entity (sheet/table)"""
        entity_id = self.generate_id('entity', entity_name)
        
        # Sample if needed
        is_sampled = False
        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            is_sampled = True
        
        # Row summary
        row_summary = {
            'row_count': len(df),
            'is_sampled': is_sampled,
            'column_count': len(df.columns),
            'duplicate_row_count': int(df.duplicated().sum()),
            'empty_row_count': int((df.isna().all(axis=1)).sum())
        }
        
        # Profile each column
        columns = []
        for idx, col_name in enumerate(df.columns):
            series = df[col_name]
            
            logical_type = self.infer_logical_type(series, col_name)
            roles = self.infer_column_role(series, col_name, df)
            stats = self.calculate_column_stats(series, col_name, df)
            pii_info = self.detect_pii(series, col_name)
            semantic_patterns = self.detect_semantic_patterns(series)
            
            column_profile = {
                'column_id': self.generate_id('col', entity_name, col_name),
                'name': col_name,
                'ordinal_position': idx,
                'description': '',
                'role_hints': roles,
                'data_type': {
                    'native_type': str(series.dtype),
                    'logical_type': logical_type,
                    'nullable': bool(series.isna().any())
                },
                'statistics': stats,
                'quality_checks': [],
                'constraints': {
                    'declared': {
                        'primary_key': False,
                        'foreign_keys': []
                    },
                    'inferred': {
                        'primary_key_candidate': series.nunique() == len(series) and series.notna().all(),
                        'primary_key_confidence': 0.9 if series.nunique() == len(series) and series.notna().all() else None
                    }
                },
                'semantic_inference': {
                    'business_term_candidates': [],
                    'pii': pii_info,
                    'domain_specific': semantic_patterns
                }
            }
            
            columns.append(column_profile)
        
        # Detect record shape
        wide_threshold = 50
        record_shape = {
            'wide_or_long_hint': 'wide' if len(df.columns) > wide_threshold else 'long' if len(df.columns) < 10 else 'unknown',
            'grain_hypothesis': '',
            'event_time_column_candidates': [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'time', 'timestamp'])],
            'entity_id_column_candidates': [col for col in df.columns if 'id' in col.lower() and df[col].nunique() > len(df) * 0.5]
        }
        
        entity_profile = {
            'entity_id': entity_id,
            'entity_name': entity_name,
            'entity_type': 'table',
            'source_id': source_id,
            'physical': {
                'format': self.detect_file_type(),
                'storage_size_bytes': None,
                'compression': 'none'
            },
            'row_summary': row_summary,
            'record_shape': record_shape,
            'columns': columns,
            'relationships': [],
            'hierarchies': []
        }
        
        return entity_profile
    
    def calculate_quality_scores(self, entities):
        """Calculate overall data quality scores"""
        if not entities:
            return {
                'overall_score': None,
                'dimensions': {
                    'completeness': {'score': None},
                    'validity': {'score': None},
                    'uniqueness': {'score': None},
                    'timeliness': {'score': None}
                }
            }
        
        # Aggregate completeness across all columns
        total_completeness = []
        for entity in entities:
            for col in entity['columns']:
                null_rate = col['statistics']['completeness']['null_rate']
                if null_rate is not None:
                    total_completeness.append(1 - null_rate)
        
        completeness_score = sum(total_completeness) / len(total_completeness) if total_completeness else None
        
        return {
            'overall_score': round(completeness_score, 4) if completeness_score else None,
            'dimensions': {
                'completeness': {'score': round(completeness_score, 4) if completeness_score else None},
                'validity': {'score': None},
                'uniqueness': {'score': None},
                'timeliness': {'score': None}
            }
        }
    
    def generate_profile(self):
        """Generate complete data profile"""
        sheets = self.read_data()
        
        source_id = self.generate_id('source', self.file_path.name)
        dataset_id = self.generate_id('dataset', self.file_path.stem)
        
        # Source information
        source = {
            'source_id': source_id,
            'source_type': self.detect_file_type(),
            'source_name': self.file_path.name,
            'connection_or_path': str(self.file_path),
            'ingestion': {
                'ingested_at_utc': self.timestamp,
                'raw_row_count': sum(len(df) for df in sheets.values()),
                'raw_byte_size': self.file_path.stat().st_size,
                'parsing': {
                    'encoding': 'utf-8',
                    'delimiter': ',' if self.detect_file_type() == 'csv' else None,
                    'quote_char': '"' if self.detect_file_type() == 'csv' else None,
                    'header_row': True,
                    'null_tokens': ['', 'NULL', 'null', 'N/A', 'NA', 'None', 'nan']
                }
            }
        }
        
        # Profile each sheet/entity
        entities = []
        for sheet_name, df in sheets.items():
            entity = self.profile_entity(df, sheet_name, source_id)
            entities.append(entity)
        
        # Build complete profile
        profile = {
            'profile_version': '1.1.0',
            'run': {
                'run_id': self.generate_id('run', self.timestamp),
                'timestamp_utc': self.timestamp,
                'engine': {
                    'name': 'GenericDataProfiler',
                    'version': '1.0.0',
                    'config': {
                        'sample_method': 'random' if self.sample_size else 'full',
                        'sample_size_rows': self.sample_size,
                        'max_distinct_values_tracked': self.max_distinct_values,
                        'top_k_values': self.top_k,
                        'histogram_bins': self.histogram_bins
                    }
                }
            },
            'dataset': {
                'dataset_id': dataset_id,
                'dataset_name': self.file_path.stem,
                'domain': '',
                'description': '',
                'owner': '',
                'tags': [],
                'time_zone': '',
                'locale': '',
                'units_conventions': {
                    'currency': '',
                    'distance': '',
                    'weight': '',
                    'temperature': ''
                },
                'compliance': {
                    'contains_pii': any(
                        col['semantic_inference']['pii']['contains_pii']
                        for entity in entities
                        for col in entity['columns']
                        if col['semantic_inference']['pii']['contains_pii']
                    ),
                    'classification': 'internal'
                }
            },
            'sources': [source],
            'entities': entities,
            'dataset_level_quality': self.calculate_quality_scores(entities),
            'lineage': {
                'upstream': []
            },
            'metadata': {
                'profiling_completeness': round(len(entities) / len(sheets), 4) if sheets else None,
                'manual_review_required': False,
                'reviewed_at_utc': ''
            }
        }
        
        return profile


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive data profile from CSV/Excel files')
    parser.add_argument('input_file', help='Path to input CSV or Excel file')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: <input>_profile.json)')
    parser.add_argument('-s', '--sample', type=int, help='Sample size (rows) for large datasets')
    parser.add_argument('--max-distinct', type=int, default=100, help='Max distinct values to track')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top values to report')
    parser.add_argument('--bins', type=int, default=10, help='Number of histogram bins')
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = DataProfiler(
        args.input_file,
        sample_size=args.sample,
        max_distinct_values=args.max_distinct,
        top_k=args.top_k,
        histogram_bins=args.bins
    )
    
    print(f"Profiling {args.input_file}...")
    profile = profiler.generate_profile()
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.input_file).with_suffix('').with_suffix('.profile.json')
    
    # Write profile to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"Profile saved to {output_path}")
    print(f"\nSummary:")
    print(f"  Entities: {len(profile['entities'])}")
    print(f"  Total columns: {sum(len(e['columns']) for e in profile['entities'])}")
    print(f"  Total rows: {sum(e['row_summary']['row_count'] for e in profile['entities'])}")
    print(f"  Contains PII: {profile['dataset']['compliance']['contains_pii']}")


if __name__ == '__main__':
    main()
