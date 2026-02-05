"""
Profile Report Generator

Generates a styled HTML report from a data profile JSON file.
The report provides a human-readable view of data quality issues,
separating items requiring human review from auto-fixable issues.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class ProfileReportGenerator:
    """
    Generates HTML reports from data profile JSON files.
    Uses a template-based approach with simple string substitution.
    """

    # Icon mappings for different issue types
    ISSUE_ICONS = {
        'missing_required_field': '!',
        'null_tokens_detected': '&#128465;',  # wastebasket
        'unmasked_pii': '&#128274;',  # lock
        'timestamp_wrong_type': '&#128336;',  # clock
        'type_mismatch': '&#128295;',  # wrench
        'whitespace_detected': '&#128465;',  # wastebasket
        'casing_inconsistency': 'Aa',
        'default': '&#9888;'  # warning
    }

    # Human-readable titles for issue types
    ISSUE_TITLES = {
        'missing_required_field': 'Missing Required Field',
        'null_tokens_detected': 'Dirty Null Tokens',
        'unmasked_pii': 'PII Masking Required',
        'timestamp_wrong_type': 'Timestamp Type Repair',
        'type_mismatch': 'Type Conversion Needed',
        'whitespace_detected': 'Whitespace Cleanup',
        'casing_inconsistency': 'Casing Normalization',
    }

    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize the report generator.

        Args:
            template_path: Path to HTML template file. If None, uses default location.
        """
        if template_path is None:
            # Default template location relative to this file
            config_dir = Path(__file__).parent.parent / 'config'
            template_path = config_dir / 'profile_report_template.html'

        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")

        with open(self.template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()

    def generate_report(self, profile_path: str, output_path: str) -> str:
        """
        Generate an HTML report from a profile JSON file.

        Args:
            profile_path: Path to the profile JSON file
            output_path: Path where the HTML report will be saved

        Returns:
            Path to the generated report
        """
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)

        html = self._render_template(profile)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return output_path

    def generate_report_from_dict(self, profile: Dict, output_path: str) -> str:
        """
        Generate an HTML report from a profile dictionary.

        Args:
            profile: Profile data as a dictionary
            output_path: Path where the HTML report will be saved

        Returns:
            Path to the generated report
        """
        html = self._render_template(profile)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return output_path

    def _render_template(self, profile: Dict) -> str:
        """
        Render the HTML template with profile data.

        Args:
            profile: Profile data dictionary

        Returns:
            Rendered HTML string
        """
        # Extract metadata
        dataset_name = profile.get('dataset', {}).get('dataset_name', 'Unknown')
        run_id = profile.get('run', {}).get('run_id', 'N/A')
        timestamp = profile.get('run', {}).get('timestamp_utc', datetime.utcnow().isoformat())
        policy_name = profile.get('policy_name', 'Unknown Policy')
        policy_version = profile.get('policy_version', '0.0.0')

        # Get source info
        sources = profile.get('sources', [])
        source_name = sources[0].get('source_name', 'Unknown') if sources else 'Unknown'

        # Get entity info (first entity)
        entities = profile.get('entities', [])
        entity = entities[0] if entities else {}
        row_count = entity.get('row_summary', {}).get('row_count', 0)
        columns = entity.get('columns', [])
        column_count = len(columns)

        # Get issue summary
        issues_summary = profile.get('issues_summary', {})
        total_issues = issues_summary.get('total_issues', 0)
        by_severity = issues_summary.get('by_severity', {})
        by_owner = issues_summary.get('by_owner', {})

        error_count = by_severity.get('error', 0)
        warn_count = by_severity.get('warn', 0)
        human_review_count = by_owner.get('human', 0)
        auto_fix_count = by_owner.get('programmatic', 0)

        # Collect all issues from columns
        human_issues = []
        auto_issues = []

        for col in columns:
            col_name = col.get('column_name', 'Unknown')
            for issue in col.get('policy_issues', []):
                issue_data = {
                    'column': col_name,
                    'issue_type': issue.get('issue_type', 'unknown'),
                    'severity': issue.get('severity', 'info'),
                    'count': issue.get('count', 0),
                    'action': issue.get('action', 'review'),
                    'details': issue.get('details', ''),
                    'policy_section': issue.get('policy_section', ''),
                }

                if issue.get('owner') == 'human':
                    human_issues.append(issue_data)
                else:
                    auto_issues.append(issue_data)

        # Group issues by type for cleaner display
        human_items_html = self._render_issue_list(human_issues, is_human=True)
        auto_items_html = self._render_issue_list(auto_issues, is_human=False)

        # Render column table rows
        column_rows_html = self._render_column_rows(columns)

        # Determine CSS classes for KPI values
        human_review_class = 'text-danger' if human_review_count > 0 else ''
        auto_fix_class = 'text-success' if auto_fix_count > 0 else ''
        errors_class = 'text-warning' if error_count > 0 else ''

        # Format timestamp for display
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            formatted_timestamp = timestamp

        # Substitute placeholders
        html = self.template
        replacements = {
            '{{dataset_name}}': self._escape_html(dataset_name),
            '{{source_name}}': self._escape_html(source_name),
            '{{row_count:,}}': f'{row_count:,}',
            '{{column_count}}': str(column_count),
            '{{run_id}}': self._escape_html(run_id),
            '{{human_review_count}}': str(human_review_count),
            '{{auto_fix_count}}': str(auto_fix_count),
            '{{total_issues}}': str(total_issues),
            '{{error_count}}': str(error_count),
            '{{human_review_class}}': human_review_class,
            '{{auto_fix_class}}': auto_fix_class,
            '{{errors_class}}': errors_class,
            '{{human_review_items}}': human_items_html,
            '{{auto_fix_items}}': auto_items_html,
            '{{column_rows}}': column_rows_html,
            '{{policy_name}}': self._escape_html(policy_name),
            '{{policy_version}}': self._escape_html(policy_version),
            '{{timestamp}}': formatted_timestamp,
        }

        for placeholder, value in replacements.items():
            html = html.replace(placeholder, value)

        return html

    def _render_issue_list(self, issues: List[Dict], is_human: bool) -> str:
        """
        Render a list of issues as HTML.

        Args:
            issues: List of issue dictionaries
            is_human: Whether these are human-review issues

        Returns:
            HTML string for the issue list
        """
        if not issues:
            icon = '&#10004;' if not is_human else '&#128269;'
            message = 'No auto-fixes needed' if not is_human else 'No human review required'
            return f'''
            <div class="empty-state">
                <div class="empty-state-icon">{icon}</div>
                <p>{message}</p>
            </div>
            '''

        # Group issues by type
        grouped: Dict[str, List[Dict]] = {}
        for issue in issues:
            issue_type = issue['issue_type']
            if issue_type not in grouped:
                grouped[issue_type] = []
            grouped[issue_type].append(issue)

        icon_class = 'icon-human' if is_human else 'icon-auto'

        items_html = '<ul class="list-group">'
        for issue_type, issue_list in grouped.items():
            icon = self.ISSUE_ICONS.get(issue_type, self.ISSUE_ICONS['default'])
            title = self.ISSUE_TITLES.get(issue_type, issue_type.replace('_', ' ').title())

            # Collect affected columns
            affected_columns = list(set(i['column'] for i in issue_list))
            total_count = sum(i['count'] for i in issue_list)

            # Get action and severity from first issue
            action = issue_list[0].get('action', 'review')
            severity = issue_list[0].get('severity', 'info')

            # Build description
            if len(affected_columns) <= 3:
                columns_str = ', '.join(affected_columns)
            else:
                columns_str = f"{', '.join(affected_columns[:3])} (+{len(affected_columns) - 3} more)"

            action_text = self._get_action_description(issue_type, action, total_count)

            severity_class = f'tag-{severity}'

            items_html += f'''
                <li class="list-item">
                    <div class="icon-box {icon_class}">{icon}</div>
                    <div class="item-content">
                        <h4>{self._escape_html(title)}</h4>
                        <p>Affected: <b>{self._escape_html(columns_str)}</b></p>
                        <p style="margin-top:4px;">{action_text}</p>
                        <span class="tag {severity_class}">Severity: {severity.title()}</span>
                        <span class="tag">Count: {total_count:,}</span>
                    </div>
                </li>
            '''

        items_html += '</ul>'
        return items_html

    def _get_action_description(self, issue_type: str, action: str, count: int) -> str:
        """
        Get a human-readable description of the remediation action.

        Args:
            issue_type: Type of issue
            action: Action code
            count: Number of affected values

        Returns:
            Description string
        """
        descriptions = {
            ('unmasked_pii', 'hash'): f'System will <b>hash</b> {count:,} exposed values.',
            ('unmasked_pii', 'tokenize'): f'System will <b>tokenize</b> {count:,} PII values.',
            ('unmasked_pii', 'drop'): f'System will <b>drop</b> this column (contains sensitive PII).',
            ('timestamp_wrong_type', 'convert_to_datetime'): f'Convert {count:,} values to proper <code>datetime</code> format.',
            ('null_tokens_detected', 'convert_to_proper_null'): f'Convert {count:,} dirty tokens (e.g., "na", "none", "-") to proper nulls.',
            ('whitespace_detected', 'trim'): f'Trim whitespace from {count:,} values.',
            ('casing_inconsistency', 'normalize'): f'Normalize casing for {count:,} values.',
            ('type_mismatch', 'coerce'): f'Safely coerce {count:,} values to expected type.',
            ('missing_required_field', 'fill_default'): f'Fill {count:,} missing values with default.',
            ('missing_required_field', 'require_human'): f'<b>Action required:</b> {count:,} rows missing critical data. Review and decide how to handle.',
        }

        key = (issue_type, action)
        if key in descriptions:
            return descriptions[key]

        # Generic fallback
        return f'Action: <b>{action}</b> ({count:,} values affected)'

    def _render_column_rows(self, columns: List[Dict]) -> str:
        """
        Render table rows for column details.

        Args:
            columns: List of column dictionaries

        Returns:
            HTML string for table rows
        """
        rows_html = ''
        for col in columns:
            col_name = col.get('column_name', 'Unknown')
            ordinal = col.get('ordinal_position', 0)
            logical_type = col.get('logical_type', 'unknown')
            role = col.get('role', 'unknown')
            stats = col.get('statistics', {})

            null_rate = stats.get('completeness', {}).get('null_rate', 0)
            null_rate_pct = f'{null_rate * 100:.1f}%'

            distinct_count = stats.get('cardinality', {}).get('distinct_count', 0)
            distinct_rate = stats.get('cardinality', {}).get('distinct_rate', 0)

            issues = col.get('policy_issues', [])
            issue_count = len(issues)
            issue_class = 'text-danger' if issue_count > 0 else 'text-success'

            rows_html += f'''
                <tr>
                    <td>{ordinal}</td>
                    <td><b>{self._escape_html(col_name)}</b></td>
                    <td><code>{self._escape_html(logical_type)}</code></td>
                    <td><span class="role-tag">{self._escape_html(role)}</span></td>
                    <td>{null_rate_pct}</td>
                    <td>{distinct_count:,} <span style="color: var(--text-muted);">({distinct_rate:.1%})</span></td>
                    <td class="{issue_class}">{issue_count}</td>
                </tr>
            '''

        return rows_html

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        if not isinstance(text, str):
            text = str(text)
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))


def generate_profile_report(profile_path: str, output_path: str, template_path: Optional[str] = None) -> str:
    """
    Convenience function to generate a profile report.

    Args:
        profile_path: Path to the profile JSON file
        output_path: Path where the HTML report will be saved
        template_path: Optional path to custom template

    Returns:
        Path to the generated report
    """
    generator = ProfileReportGenerator(template_path=template_path)
    return generator.generate_report(profile_path, output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate HTML report from data profile JSON')
    parser.add_argument('profile', help='Path to profile JSON file')
    parser.add_argument('--output', '-o', help='Output HTML file path', default=None)
    parser.add_argument('--template', '-t', help='Custom template path', default=None)

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        profile_path = Path(args.profile)
        args.output = str(profile_path.with_suffix('.html'))

    output = generate_profile_report(args.profile, args.output, args.template)
    print(f"Report generated: {output}")
