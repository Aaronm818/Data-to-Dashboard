import os
import json
import yaml
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd
from datetime import datetime

# Import core engines
from core.data_profiler import DataProfiler
from core.remediation_engine import RemediationEngine

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-demo'  # In production, use a secure key

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
CONFIG_PATH = 'config/dq_policy_spec.yaml'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_policy():
    """Load the current policy from disk"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return f.read()
    return ""

def save_policy(content):
    """Save the policy to disk"""
    with open(CONFIG_PATH, 'w') as f:
        f.write(content)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/profile', methods=['POST'])
def profile():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Instantiate profiler (ALWAYS re-reads config for dynamic behavior)
        # Note: We pass the absolute path to the config to ensure it's found
        abs_config_path = os.path.abspath(CONFIG_PATH)
        profiler = DataProfiler(file_path, policy_path=abs_config_path)
        
        # Generate profile
        profile_data = profiler.generate_profile()
        
        # Save profile for remediation step
        profile_filename = f"{filename}.profile.json"
        profile_path = os.path.join(app.config['UPLOAD_FOLDER'], profile_filename)
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
            
        return render_template('profile.html', 
                             profile=profile_data, 
                             filename=filename,
                             profile_filename=profile_filename)

@app.route('/remediate', methods=['POST'])
def remediate():
    filename = request.form.get('filename')
    profile_filename = request.form.get('profile_filename')
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    profile_path = os.path.join(app.config['UPLOAD_FOLDER'], profile_filename)
    
    # Load profile
    with open(profile_path, 'r') as f:
        profile_data = json.load(f)
    
    # Instantiate remediation engine (ALWAYS re-reads config)
    abs_config_path = os.path.abspath(CONFIG_PATH)
    engine = RemediationEngine(policy_path=abs_config_path)
    
    # Load data
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
        
    # Remediate
    remediated_df, human_review = engine.remediate_dataframe(df, profile_data)
    summary = engine.generate_remediation_summary(df, remediated_df, human_review)
    
    # Save outputs
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_subdir = os.path.join(app.config['OUTPUT_FOLDER'], f"{timestamp}_{filename}")
    os.makedirs(output_subdir, exist_ok=True)
    
    remediated_filename = f"remediated_{filename}"
    remediated_path = os.path.join(output_subdir, remediated_filename)
    summary_path = os.path.join(output_subdir, "summary.json")
    
    if filename.endswith('.csv'):
        remediated_df.to_csv(remediated_path, index=False)
    else:
        remediated_df.to_excel(remediated_path, index=False)
        
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
        
    return render_template('results.html', 
                         summary=summary, 
                         remediated_file=remediated_filename, 
                         output_dir=f"{timestamp}_{filename}")

@app.route('/download/<path:dirname>/<path:filename>')
def download_file(dirname, filename):
    directory = os.path.join(app.config['OUTPUT_FOLDER'], dirname)
    return send_file(os.path.join(directory, filename), as_attachment=True)

@app.route('/policy', methods=['GET', 'POST'])
def policy():
    if request.method == 'POST':
        content = request.form.get('policy_content')
        save_policy(content)
        flash('Policy updated successfully! Run a profile to see changes.')
        return redirect(url_for('policy'))
    
    content = load_policy()
    return render_template('policy.html', content=content)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
