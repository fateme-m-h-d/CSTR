# scaler_utils.py
import pickle, os

scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)



