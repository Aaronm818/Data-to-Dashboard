"""
Wide Table Report Generator

Generates a styled HTML report from a WideTablePlan object/dict.
Does not rely on Jinja2; uses simple string replacement and HTML construction.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

class WideTableReportGenerator:
    """
    Generates HTML reports from WideTablePlan data.
    """

    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize the report generator.
        """
        if template_path is None:
            config_dir = Path(__file__).parent.parent / 'config'
            template_path = config_dir / 'wide_table_report_template.html'

        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")

        with open(self.template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()

    def generate_report(self, plan: Dict, output_path: str) -> str:
        """
        Generate and save HTML report.
        """
        html = self._render_template(plan)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        return output_path

    def _render_template(self, plan: Dict) -> str:
        """
        Render the template with plan data.
        """
        # Metadata
        dataset_name = plan.get('source_dataset_name', 'Unknown')
        generated_at = plan.get('generated_at', datetime.utcnow().isoformat())
        version = plan.get('version', '1.0.0')
        profile_version = plan.get('source_profile_version', 'unknown')

        # Tables
        tables = plan.get('tables', [])
        table_count = len(tables)
        
        # Calculate total mapped columns
        column_count = sum(len(t.get('columns', [])) for t in tables)
        
        # Excluded
        excluded = plan.get('excluded_columns', [])
        excluded_count = len(excluded)
        
        # Confidence logic (take min confidence of tables or from grain)
        # Assuming single grain confidence is representative or taking from first table
        confidence = 1.0
        if tables:
            # Try to get from grain object
            grain = tables[0].get('grain', {})
            confidence = grain.get('confidence', 1.0)
            
        confidence_score = f"{confidence:.2f}"
        confidence_class = 'text-success'
        if confidence < 0.75:
            confidence_class = 'text-danger'
        elif confidence < 0.9:
            confidence_class = 'text-warning'

        # Render Sections
        human_review_html = self._render_human_review(plan)
        tables_html = self._render_tables(tables)
        sidecar_html = self._render_sidecars(plan.get('sidecar_tables', []))
        validation_html = self._render_validation(plan.get('validation', {}))
        preview_html = self._render_preview(plan.get('load_plan', {}))

        # Replace
        html = self.template
        replacements = {
            '{{dataset_name}}': self._escape(dataset_name),
            '{{source_dataset}}': self._escape(dataset_name),
            '{{generated_at}}': self._escape(generated_at),
            '{{version}}': self._escape(version),
            '{{profile_version}}': self._escape(profile_version),
            '{{confidence_score}}': confidence_score,
            '{{confidence_class}}': confidence_class,
            '{{table_count}}': str(table_count),
            '{{column_count}}': str(column_count),
            '{{excluded_count}}': str(excluded_count),
            '{{human_review_section}}': human_review_html,
            '{{tables_section}}': tables_html,
            '{{sidecar_section}}': sidecar_html,
            '{{validation_items}}': validation_html,
            '{{load_plan_preview}}': preview_html
        }

        for placeholder, value in replacements.items():
            html = html.replace(placeholder, value)
            
        return html

    def _render_human_review(self, plan: Dict) -> str:
        if not plan.get('human_review_required'):
            return ""
        
        reasons = plan.get('human_review_reasons', [])
        items_html = "\n".join([f"<li>{self._escape(r)}</li>" for r in reasons])
        
        return f"""
        <div class="section alert-review">
            <h3>&#9888;&#65039; Human Review Required</h3>
            <ul>
                {items_html}
            </ul>
        </div>
        """

    def _render_tables(self, tables: List[Dict]) -> str:
        if not tables:
            return "<div class='empty-state'>No tables generated</div>"
            
        html_parts = []
        for table in tables:
            name = table.get('table_name', 'Unknown')
            grain_desc = table.get('grain', {}).get('description', 'Unknown Grain')
            
            rows = []
            columns = table.get('columns', [])
            for col in columns:
                c_name = col.get('name')
                c_role = col.get('role')
                c_type = col.get('type')
                c_source = ", ".join(col.get('source_columns', []))
                c_transform = col.get('transform')
                
                # Role class
                r_cls = 'role-dim'
                if 'key' in c_role: r_cls = 'role-key'
                elif 'measure' in c_role: r_cls = 'role-measure'
                elif 'time' in c_role: r_cls = 'role-time'
                
                rows.append(f"""
                <tr>
                    <td style="font-weight: 500;">{self._escape(c_name)}</td>
                    <td><span class="role-tag {r_cls}">{self._escape(c_role)}</span></td>
                    <td style="font-family: monospace; color: var(--text-muted);">{self._escape(c_type)}</td>
                    <td style="color: var(--text-muted);">{self._escape(c_source)}</td>
                    <td style="font-size: 11px;">{self._escape(c_transform)}</td>
                </tr>
                """)
            
            tbody = "".join(rows)
            
            card = f"""
            <div class="table-card">
                <div class="table-header">
                    <div class="table-title">{self._escape(name)}</div>
                    <div class="table-grain">
                        <span style="color: var(--accent);">GRAIN:</span> {self._escape(grain_desc)}
                    </div>
                </div>
                <div style="padding: 2px;">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Column Name</th>
                                <th>Role</th>
                                <th>Type</th>
                                <th>Source</th>
                                <th>Transform</th>
                            </tr>
                        </thead>
                        <tbody>
                            {tbody}
                        </tbody>
                    </table>
                </div>
            </div>
            """
            html_parts.append(card)
            
        return "\n".join(html_parts)

    def _render_sidecars(self, sidecars: List[Dict]) -> str:
        if not sidecars:
            return ""
            
        html_parts = []
        html_parts.append('<div class="section"><h2>Sidecar Tables</h2>')
        
        for table in sidecars:
            name = table.get('table_name')
            purpose = table.get('purpose')
            
            rows = []
            for col in table.get('columns', []):
                c_name = col.get('name')
                c_desc = col.get('description', '')
                rows.append(f"""
                <tr>
                    <td>{self._escape(c_name)}</td>
                    <td style="color: var(--text-muted);">{self._escape(c_desc)}</td>
                </tr>
                """)
                
            tbody = "".join(rows)
            
            card = f"""
            <div class="table-card" style="border-style: dashed;">
                <div class="table-header">
                    <div class="table-title" style="color: var(--text-muted);">{self._escape(name)}</div>
                    <div class="table-grain">{self._escape(purpose)}</div>
                </div>
                <div style="padding: 2px;">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Column Name</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            {tbody}
                        </tbody>
                    </table>
                </div>
            </div>
            """
            html_parts.append(card)
            
        html_parts.append('</div>')
        return "\n".join(html_parts)

    def _render_validation(self, validation: Dict) -> str:
        if not validation:
            return "No validation results"
            
        tests = validation.get('tests', [])
        lines = []
        
        for test in tests:
            passed = test.get('passed', False)
            icon = "✅" if passed else "❌"
            name = test.get('test_id')
            desc = test.get('test_description')
            details = test.get('details')
            
            detail_div = f'<div class="test-details">{self._escape(details)}</div>' if details else ''
            
            lines.append(f"""
            <div class="test-item">
                <div class="test-status">{icon}</div>
                <div class="test-desc">
                    {self._escape(name)}: {self._escape(desc)}
                    {detail_div}
                </div>
            </div>
            """)
            
        return "\n".join(lines)

    def _render_preview(self, load_plan: Dict) -> str:
        if not load_plan:
            return ""
            
        # Generate dummy SQL preview for visual
        target = load_plan.get('target_table', 'target')
        source = load_plan.get('source_entity', 'source')
        
        sql_lines = [f"INSERT INTO {target} SELECT"]
        mappings = load_plan.get('column_mappings', [])
        
        for i, m in enumerate(mappings[:10]): # Limit to 10
            comma = "," if i < len(mappings) - 1 else ""
            expr = m.get('source_column')
            if m.get('transform') == 'CAST':
                expr = f"CAST({expr} AS {m.get('type_cast')})"
            
            sql_lines.append(f"    {expr} AS {m.get('target_column')}{comma}")
            
        if len(mappings) > 10:
            sql_lines.append(f"    -- ... and {len(mappings)-10} more columns")
            
        sql_lines.append(f"FROM {source}")
        
        return "\n".join([f"<div>{self._escape(l)}</div>" for l in sql_lines])

    @staticmethod
    def _escape(text: Any) -> str:
        if text is None: return ""
        s = str(text)
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;")
                 .replace("'", "&#39;"))
