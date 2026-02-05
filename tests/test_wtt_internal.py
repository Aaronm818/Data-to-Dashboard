
import sys
import os
import json
import pandas as pd
from pathlib import Path
import shutil

# Add backend to path
sys.path.append(os.path.abspath("backend/ingestion_engine"))
sys.path.append(os.path.abspath("backend/ingestion_engine/core"))

try:
    from core.wide_table_transformer import WideTableTransformer
    from core.output_manager import StandardOutputManager
    from wtt_data_classes import WideTablePlan
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_integration():
    print("Testing WTT Integration...")
    
    # 1. Setup Data
    df = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C1'],
        'order_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'amount': [100.5, 200.0, 150.75],
        'status': ['Completed', 'Pending', 'Completed']
    })
    
    profile = {
        'dataset': {'dataset_name': 'Test Dataset'},
        'entities': [{
            'entity_name': 'orders',
            'columns': [
                {'column_name': 'customer_id', 'role': 'identifier', 'statistics': {'completeness': {'null_rate': 0.0}, 'cardinality': {'distinct_rate': 0.6}}},
                {'column_name': 'order_date', 'role': 'timestamp', 'statistics': {'completeness': {'null_rate': 0.0}, 'cardinality': {'distinct_rate': 1.0}}},
                {'column_name': 'amount', 'role': 'measure', 'logical_type': 'float64', 'statistics': {'completeness': {'null_rate': 0.0}, 'cardinality': {'distinct_rate': 1.0}}},
                {'column_name': 'status', 'role': 'dimension', 'statistics': {'completeness': {'null_rate': 0.0}, 'cardinality': {'distinct_rate': 1.0}}}
            ]
        }],
        'run': {'run_id': 'test_run'}
    }
    
    # 2. Run Transform
    print("Running transformation...")
    transformer = WideTableTransformer()
    plan = transformer.transform(df, profile)
    
    if not isinstance(plan, WideTablePlan):
        print("FAILED: Result is not a WideTablePlan object")
        sys.exit(1)
        
    print(f"Transformation successful. Generated {len(plan.tables)} tables.")
    
    # 3. Save Output
    output_dir = Path("tests/test_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    manager = StandardOutputManager(base_output_dir=str(output_dir))
    run_dir = manager.create_run_directory()
    print(f"Created run directory: {run_dir}")
    
    print("Saving plans...")
    paths = manager.save_wide_table_plan(plan)
    print(f"Saved plan paths: {paths}")
    
    print("Saving report...")
    report_path = manager.save_wide_table_report(plan)
    print(f"Saved report to: {report_path}")
    
    # 4. Verify Files
    failures = []
    if not (run_dir / "wide_table_plan.json").exists():
        failures.append("JSON plan missing")
    if not (run_dir / "wide_table_plan.yaml").exists():
        failures.append("YAML plan missing")
    if not (run_dir / "wide_table_report.html").exists():
        failures.append("HTML report missing")
        
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
        
    print("VERIFICATION PASSED: All files generated successfully.")

if __name__ == "__main__":
    test_integration()
