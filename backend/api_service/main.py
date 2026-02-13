"""
Data-to-Dashboard Pipeline Manager API
FastAPI backend for the data profiling, remediation, and visualization pipeline.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import pandas as pd

# Add ingestion_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion_engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion_engine" / "core"))

# Import core engines
from core.data_profiler import DataProfiler, NumpyEncoder
from core.remediation_engine import RemediationEngine
from core.output_manager import StandardOutputManager

# Try to import wide table transformer (may need additional setup)
try:
    from core.wide_table_transformer import WideTableTransformer
    WTT_AVAILABLE = True
except ImportError as e:
    print(f"Wide Table Transformer not available: {e}")
    WTT_AVAILABLE = False

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
OUTPUT_FOLDER = BASE_DIR / "data" / "outputs"
CONFIG_PATH = BASE_DIR / "backend" / "ingestion_engine" / "config" / "dq_policy_spec.yaml"

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# In-memory session storage (for demo - use Redis/DB in production)
sessions: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("🚀 Pipeline Manager API starting...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print(f"📋 Policy config: {CONFIG_PATH}")
    yield
    print("👋 Pipeline Manager API shutting down...")


app = FastAPI(
    title="Data-to-Dashboard Pipeline Manager",
    description="API for data profiling, remediation, and visualization pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class SessionResponse(BaseModel):
    session_id: str
    message: str
    
class ProfileResponse(BaseModel):
    session_id: str
    status: str
    profile: Optional[Dict] = None
    issues_summary: Optional[Dict] = None
    
class RemediationRequest(BaseModel):
    session_id: str
    apply_fixes: bool = True
    
class RemediationResponse(BaseModel):
    session_id: str
    status: str
    summary: Optional[Dict] = None
    human_review_required: Optional[List[Dict]] = None
    
class WidePlanResponse(BaseModel):
    session_id: str
    status: str
    plan: Optional[Dict] = None
    
class BuildRequest(BaseModel):
    session_id: str
    approved: bool = True
    
class VisualizationResponse(BaseModel):
    session_id: str
    status: str
    data: Optional[Dict] = None
    chart_configs: Optional[List[Dict]] = None


# ============== Helper Functions ==============

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"


def get_session(session_id: str) -> Dict:
    """Get session or raise error"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return sessions[session_id]


def save_session_state(session_id: str, state: Dict):
    """Save session state"""
    sessions[session_id] = {**sessions.get(session_id, {}), **state}


# ============== API Endpoints ==============

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Data-to-Dashboard Pipeline Manager",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "profile": "/api/profile/{session_id}",
            "remediate": "/api/remediate",
            "wide_plan": "/api/wide-plan/{session_id}",
            "build": "/api/build",
            "visualize": "/api/visualize/{session_id}",
            "sessions": "/api/sessions"
        }
    }


@app.get("/api/sessions")
def list_sessions():
    """List all active sessions"""
    return {
        "sessions": [
            {
                "session_id": sid,
                "filename": data.get("filename"),
                "status": data.get("status"),
                "created_at": data.get("created_at")
            }
            for sid, data in sessions.items()
        ]
    }


@app.post("/api/upload", response_model=SessionResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Step 1: Upload a CSV or Excel file
    Returns a session_id to track the pipeline progress
    """
    # Validate file type
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate session
    session_id = generate_session_id()
    
    # Save file
    file_path = UPLOAD_FOLDER / f"{session_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Initialize session
    sessions[session_id] = {
        "session_id": session_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "status": "uploaded",
        "created_at": datetime.utcnow().isoformat(),
        "steps_completed": ["upload"]
    }
    
    return SessionResponse(
        session_id=session_id,
        message=f"File '{file.filename}' uploaded successfully"
    )


@app.get("/api/profile/{session_id}")
async def profile_data(session_id: str):
    """
    Step 2: Profile the uploaded data
    Runs data quality analysis and returns issues
    """
    session = get_session(session_id)
    
    if session["status"] not in ["uploaded", "profiled"]:
        raise HTTPException(status_code=400, detail="File must be uploaded first")
    
    file_path = session["file_path"]
    
    try:
        # Run profiler
        profiler = DataProfiler(
            file_path,
            policy_path=str(CONFIG_PATH)
        )
        profile_data = profiler.generate_profile()
        
        # Save profile
        profile_path = UPLOAD_FOLDER / f"{session_id}_profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile_data, f, indent=2, cls=NumpyEncoder)
        
        # Calculate scores for frontend
        issues_summary = profile_data.get("issues_summary", {})
        total_issues = issues_summary.get("total_issues", 0)
        
        # Get total columns and calculate completeness
        entities = profile_data.get("entities", [])
        total_columns = sum(len(e.get("columns", [])) for e in entities)
        total_rows = sum(e.get("row_summary", {}).get("row_count", 0) for e in entities)
        
        # Calculate completeness score (simplified)
        completeness_scores = []
        for entity in entities:
            for col in entity.get("columns", []):
                null_rate = col.get("statistics", {}).get("completeness", {}).get("null_rate", 0)
                completeness_scores.append(1 - null_rate)
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 1.0
        
        # Quality score based on issues
        quality_score = max(0, 1 - (total_issues / max(total_columns * 2, 1)))
        
        # Update session
        save_session_state(session_id, {
            "status": "profiled",
            "profile_path": str(profile_path),
            "profile_data": profile_data,
            "steps_completed": session.get("steps_completed", []) + ["profile"]
        })
        
        return {
            "session_id": session_id,
            "status": "profiled",
            "scores": {
                "completeness": round(avg_completeness * 100, 1),
                "quality": round(quality_score * 100, 1),
                "total_issues": total_issues,
                "programmatic_fixes": issues_summary.get("by_owner", {}).get("programmatic", 0),
                "human_review": issues_summary.get("by_owner", {}).get("human", 0)
            },
            "summary": {
                "total_columns": total_columns,
                "total_rows": total_rows,
                "entities": len(entities),
                "issues_by_severity": issues_summary.get("by_severity", {})
            },
            "profile": profile_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")


@app.post("/api/remediate")
async def remediate_data(request: RemediationRequest):
    """
    Step 3: Apply remediation (AI auto-fixes)
    Returns summary of fixes applied and issues requiring human review
    """
    session = get_session(request.session_id)
    
    if "profiled" not in session.get("steps_completed", []):
        raise HTTPException(status_code=400, detail="Data must be profiled first")
    
    file_path = session["file_path"]
    profile_data = session.get("profile_data")
    
    if not profile_data:
        profile_path = session.get("profile_path")
        with open(profile_path, "r") as f:
            profile_data = json.load(f)
    
    try:
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Run remediation engine
        engine = RemediationEngine(policy_path=str(CONFIG_PATH))
        remediated_df, human_review = engine.remediate_dataframe(df, profile_data)
        summary = engine.generate_remediation_summary(df, remediated_df, human_review)
        
        # Save remediated data
        remediated_path = OUTPUT_FOLDER / f"{request.session_id}_remediated.csv"
        remediated_df.to_csv(remediated_path, index=False)
        
        # Update session
        save_session_state(request.session_id, {
            "status": "remediated",
            "remediated_path": str(remediated_path),
            "remediated_df": remediated_df,
            "remediation_summary": summary,
            "human_review_items": human_review,
            "steps_completed": session.get("steps_completed", []) + ["remediate"]
        })
        
        return {
            "session_id": request.session_id,
            "status": "remediated",
            "summary": {
                "fixes_applied": summary.get("total_fixes_applied", 0),
                "rows_affected": summary.get("rows_affected", 0),
                "human_review_required": len(human_review),
                "actions": summary.get("actions", [])
            },
            "human_review_items": human_review[:10]  # Return first 10 for preview
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Remediation failed: {str(e)}")


@app.get("/api/wide-plan/{session_id}")
async def get_wide_plan(session_id: str):
    """
    Step 4: Generate wide table plan
    Returns the schema for the Postgres database
    """
    session = get_session(session_id)
    
    if "remediate" not in session.get("steps_completed", []):
        raise HTTPException(status_code=400, detail="Data must be remediated first")
    
    if not WTT_AVAILABLE:
        # Return mock plan if transformer not available
        return {
            "session_id": session_id,
            "status": "plan_generated",
            "plan": {
                "table_name": f"{session.get('filename', 'data').replace('.', '_')}_wide",
                "columns": [
                    {"name": "id", "type": "SERIAL PRIMARY KEY"},
                    {"name": "data_column", "type": "TEXT"},
                ],
                "message": "Wide Table Transformer not available - using simplified plan"
            }
        }
    
    try:
        # Get remediated data
        remediated_df = session.get("remediated_df")
        if remediated_df is None:
            remediated_path = session.get("remediated_path")
            remediated_df = pd.read_csv(remediated_path)
        
        profile_data = session.get("profile_data")
        
        # Generate wide table plan
        spec_path = BASE_DIR / "backend" / "ingestion_engine" / "config" / "wide_table_transformer_spec.yaml"
        transformer = WideTableTransformer(spec_path=str(spec_path))
        plan = transformer.transform(remediated_df, profile_data)
        
        # Convert to dict for JSON response
        plan_dict = plan.to_dict() if hasattr(plan, 'to_dict') else {"raw": str(plan)}
        
        # Update session
        save_session_state(session_id, {
            "status": "plan_generated",
            "wide_plan": plan_dict,
            "steps_completed": session.get("steps_completed", []) + ["wide_plan"]
        })
        
        return {
            "session_id": session_id,
            "status": "plan_generated",
            "plan": plan_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wide plan generation failed: {str(e)}")


@app.post("/api/build")
async def build_database(request: BuildRequest):
    """
    Step 5: Build Postgres database tables
    Requires user approval of the wide table plan
    """
    session = get_session(request.session_id)
    
    if not request.approved:
        return {
            "session_id": request.session_id,
            "status": "pending_approval",
            "message": "Plan must be approved before building"
        }
    
    # For now, simulate the build process
    # In production, this would connect to Postgres and create tables
    
    database_url = os.environ.get("DATABASE_URL", "postgresql://localhost/data_dashboard")
    
    try:
        # Simulate build steps
        build_steps = [
            {"step": "connect", "status": "complete", "message": "Connected to Postgres"},
            {"step": "create_schema", "status": "complete", "message": "Schema created"},
            {"step": "create_tables", "status": "complete", "message": "Tables created"},
            {"step": "load_data", "status": "complete", "message": "Data loaded"},
            {"step": "create_indexes", "status": "complete", "message": "Indexes created"},
        ]
        
        # Update session
        save_session_state(request.session_id, {
            "status": "built",
            "build_steps": build_steps,
            "database_url": database_url,
            "steps_completed": session.get("steps_completed", []) + ["build"]
        })
        
        return {
            "session_id": request.session_id,
            "status": "built",
            "build_steps": build_steps,
            "connection_info": {
                "host": "postgres.railway.internal",
                "database": "railway",
                "table": f"{session.get('filename', 'data').replace('.', '_')}_wide"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Build failed: {str(e)}")


@app.get("/api/visualize/{session_id}")
async def get_visualization_data(session_id: str):
    """
    Step 6: Get data for visualization
    Returns chart-ready data from the loaded database
    """
    session = get_session(session_id)
    
    # Get the remediated data for visualization
    remediated_df = session.get("remediated_df")
    if remediated_df is None:
        remediated_path = session.get("remediated_path")
        if remediated_path and Path(remediated_path).exists():
            remediated_df = pd.read_csv(remediated_path)
        else:
            raise HTTPException(status_code=400, detail="No data available for visualization")
    
    try:
        # Generate visualization data
        # Detect numeric columns for charts
        numeric_cols = remediated_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = remediated_df.select_dtypes(include=['object']).columns.tolist()
        date_cols = [col for col in remediated_df.columns if 'date' in col.lower()]
        
        # Build chart configurations
        chart_configs = []
        
        # Summary statistics
        summary_stats = {}
        for col in numeric_cols[:5]:  # First 5 numeric columns
            summary_stats[col] = {
                "sum": float(remediated_df[col].sum()),
                "mean": float(remediated_df[col].mean()),
                "min": float(remediated_df[col].min()),
                "max": float(remediated_df[col].max()),
            }
        
        # Bar chart config (first categorical vs first numeric)
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            grouped = remediated_df.groupby(cat_col)[num_col].sum().head(10)
            chart_configs.append({
                "type": "bar",
                "title": f"{num_col} by {cat_col}",
                "data": {
                    "labels": grouped.index.tolist(),
                    "values": grouped.values.tolist()
                }
            })
        
        # Pie chart config (categorical distribution)
        if categorical_cols:
            cat_col = categorical_cols[0]
            value_counts = remediated_df[cat_col].value_counts().head(5)
            chart_configs.append({
                "type": "pie",
                "title": f"Distribution of {cat_col}",
                "data": {
                    "labels": value_counts.index.tolist(),
                    "values": value_counts.values.tolist()
                }
            })
        
        # Line chart config (if date column exists)
        if date_cols and numeric_cols:
            chart_configs.append({
                "type": "line",
                "title": f"{numeric_cols[0]} over time",
                "data": {
                    "x_column": date_cols[0],
                    "y_column": numeric_cols[0],
                    "sample_data": remediated_df[[date_cols[0], numeric_cols[0]]].head(20).to_dict('records')
                }
            })
        
        # KPI cards
        kpis = []
        for i, col in enumerate(numeric_cols[:4]):
            kpis.append({
                "label": col.replace("_", " ").title(),
                "value": f"{remediated_df[col].sum():,.0f}",
                "change": f"+{(i+1) * 5.2}%",  # Mock change
                "trend": "up"
            })
        
        # Update session
        save_session_state(session_id, {
            "status": "visualized",
            "steps_completed": session.get("steps_completed", []) + ["visualize"]
        })
        
        return {
            "session_id": session_id,
            "status": "visualized",
            "data": {
                "row_count": len(remediated_df),
                "columns": remediated_df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "summary_stats": summary_stats,
                "sample_data": remediated_df.head(10).to_dict('records')
            },
            "chart_configs": chart_configs,
            "kpis": kpis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")


@app.get("/api/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str):
    """Download output files (profile, remediated data, etc.)"""
    session = get_session(session_id)
    
    file_map = {
        "profile": session.get("profile_path"),
        "remediated": session.get("remediated_path"),
    }
    
    file_path = file_map.get(file_type)
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_type}")
    
    return FileResponse(
        file_path,
        filename=Path(file_path).name,
        media_type="application/octet-stream"
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up session and associated files"""
    if session_id in sessions:
        session = sessions[session_id]
        
        # Clean up files
        for key in ["file_path", "profile_path", "remediated_path"]:
            file_path = session.get(key)
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
        
        del sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
