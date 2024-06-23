from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Load trained model
model = xgb.XGBClassifier()
model.load_model('xgb_model.json')

# Load the scaler
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

# Path to the Excel file
EXCEL_FILE = 'leads.xlsx'

# Ensure the Excel file exists and has the correct columns
if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=['name', 'interaction_count', 'page_views', 'time_on_site', 'lead_score', 'ip_address'])
    df.to_excel(EXCEL_FILE, index=False)

@app.route('/score_lead', methods=['POST'])
def score_lead():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    df_scaled = scaler.transform(df[['interaction_count', 'page_views', 'time_on_site']])
    lead_score = model.predict_proba(df_scaled)[:, 1][0]

    # Get the user's IP address
    ip_address = request.remote_addr

    # Load the existing Excel file
    leads_df = pd.read_excel(EXCEL_FILE)

    # Add the new lead
    new_lead = {
        'name': data['name'],
        'interaction_count': data['interaction_count'],
        'page_views': data['page_views'],
        'time_on_site': data['time_on_site'],
        'lead_score': lead_score,
        'ip_address': ip_address
    }
    leads_df = leads_df.append(new_lead, ignore_index=True)

    # Save back to the Excel file
    leads_df.to_excel(EXCEL_FILE, index=False)

    response = {
        'name': data['name'],
        'lead_score': lead_score,
        'ip_address': ip_address
    }
    return jsonify(response)

@app.route('/leads', methods=['GET'])
def get_leads():
    leads_df = pd.read_excel(EXCEL_FILE)
    lead_list = leads_df.to_dict(orient='records')
    return jsonify(lead_list)

if __name__ == '__main__':
    app.run(debug=True)
