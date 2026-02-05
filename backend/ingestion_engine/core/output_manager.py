import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union
import yaml
from dataclasses import is_dataclass, asdict
from enum import Enum

class StandardOutputManager:
    """
    Manages consistent output generation for DQ processes.
    Ensures identical directory structure and naming conventions across CLI and Web App.
    """
    
    def __init__(self, base_output_dir: str = 'output'):
        self.base_output_dir = Path(base_output_dir)
        self.run_id = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_dir = None
        
    def create_run_directory(self, original_filename: str = None) -> Path:
        """
        Create a standardized run directory: output/YYYY-MM-DD_HH-MM-SS
        (Ignores filename in directory name per spec)
        """
        dir_name = f"{self.run_id}"
        self.run_dir = self.base_output_dir / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save_original(self, file_path: Union[str, Path], df: pd.DataFrame = None):
        """Save the original file copy."""
        if not self.run_dir:
            raise RuntimeError("Run directory not created. Call create_run_directory first.")
            
        file_path = Path(file_path)
        ext = file_path.suffix
        if not ext:
             # Fallback if no extension in filename
            ext = '.csv' # default to csv or handle based on content
            
        target_path = self.run_dir / f"original_data{ext}"
        
        if df is not None:
            # If DF provided, write it (ensures we save what we read)
            if ext == '.csv':
                df.to_csv(target_path, index=False)
            else:
                df.to_excel(target_path, index=False)
        else:
            # Otherwise copy the file directly
            import shutil
            shutil.copy2(file_path, target_path)
            
        return target_path

    def save_remediated(self, df: pd.DataFrame, original_filename: str):
        """Save the remediated dataframe."""
        if not self.run_dir:
            raise RuntimeError("Run directory not created. Call create_run_directory first.")
            
        ext = Path(original_filename).suffix
        if not ext:
            ext = '.csv'

        target_path = self.run_dir / f"remediated_data{ext}"
        
        if ext == '.csv':
            df.to_csv(target_path, index=False)
        else:
            df.to_excel(target_path, index=False)
            
        return target_path

    def save_summary(self, summary_data: Dict[str, Any]):
        """Save the remediation summary JSON."""
        if not self.run_dir:
            raise RuntimeError("Run directory not created. Call create_run_directory first.")
            
        target_path = self.run_dir / "remediation_summary.json"
        with open(target_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
            
        return target_path

    def save_profile(self, profile_data: Dict[str, Any]):
        """Save the profile JSON."""
        if not self.run_dir:
             raise RuntimeError("Run directory not created. Call create_run_directory first.")

        target_path = self.run_dir / "profile.json"
        with open(target_path, 'w') as f:
             json.dump(profile_data, f, indent=2, default=str)

        return target_path

    def save_profile_report(self, profile_data: Dict[str, Any], template_path: str = None):
        """
        Generate and save an HTML profile report.

        Args:
            profile_data: Profile dictionary
            template_path: Optional custom template path

        Returns:
            Path to the generated HTML report
        """
        if not self.run_dir:
            raise RuntimeError("Run directory not created. Call create_run_directory first.")

        # Import here to avoid circular imports
        from core.profile_report_generator import ProfileReportGenerator

        target_path = self.run_dir / "profile_report.html"

        generator = ProfileReportGenerator(template_path=template_path)
        generator.generate_report_from_dict(profile_data, str(target_path))

        return target_path

    def save_wide_table_plan(self, plan: Any, format: str = 'both'):
        """
        Save the wide table plan to JSON and/or YAML.
        
        Args:
            plan: WideTablePlan object
            format: 'json', 'yaml', or 'both'
        """
        if not self.run_dir:
            raise RuntimeError("Run directory not created. Call create_run_directory first.")

        # Convert to dict
        plan_dict = self._plan_to_dict(plan)

        saved_paths = []

        if format in ['json', 'both']:
            json_path = self.run_dir / "wide_table_plan.json"
            with open(json_path, 'w') as f:
                json.dump(plan_dict, f, indent=2, default=str)
            saved_paths.append(json_path)

        if format in ['yaml', 'both']:
            yaml_path = self.run_dir / "wide_table_plan.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(plan_dict, f, default_flow_style=False, sort_keys=False)
            saved_paths.append(yaml_path)

        return saved_paths

    def save_wide_table_report(self, plan: Any, template_path: str = None):
        """
        Generate and save HTML report for wide table plan.
        """
        if not self.run_dir:
            raise RuntimeError("Run directory not created. Call create_run_directory first.")

        from core.wtt_report_generator import WideTableReportGenerator

        target_path = self.run_dir / "wide_table_report.html"
        plan_dict = self._plan_to_dict(plan)

        generator = WideTableReportGenerator(template_path=template_path)
        generator.generate_report(plan_dict, str(target_path))

        return target_path

    def _plan_to_dict(self, obj: Any) -> Any:
        """Helper to convert dataclasses to dict handling Enums"""
        if is_dataclass(obj):
            return {k: self._plan_to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [self._plan_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._plan_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj

