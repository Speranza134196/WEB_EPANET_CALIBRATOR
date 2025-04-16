# 💧 EPANET Calibration + INP Update Tool (Web Edition)

This is a web-based calibration tool for EPANET models using the [Streamlit](https://streamlit.io) framework. It allows users to:

- Upload an EPANET `.inp` file and observed data `.csv`
- Run automated calibration of pipe roughness
- View simulated vs. observed results (with RMSE)
- Download calibrated results and updated `.inp` files

🚀 **Live Demo:** Will be available after deployment on [Render](https://render.com)

---

## 🧠 How It Works

The app uses the `WNTR` library to simulate the network and perform calibration by optimizing roughness coefficients using:

- `scipy.optimize.differential_evolution`
- `L-BFGS-B` refinement

---

## 📁 File Structure

```
.
├── web_app2.py            # Main Streamlit app
├── requirements.txt       # Python dependencies
└── render.yaml            # Render deployment config
```

---

## 📦 Requirements

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

Or manually install the key libraries:

```bash
pip install streamlit pandas numpy matplotlib scipy wntr
```

---

## 🚀 Deployment (on Render)

### 1. Push to GitHub

Upload all files to a public GitHub repo:
- `web_app2.py`
- `requirements.txt`
- `render.yaml`

### 2. Create a Render Web Service

1. Go to [https://render.com](https://render.com)
2. Click **New > Web Service**
3. Connect your GitHub repo
4. Render will auto-detect the `render.yaml` and configure deployment
5. Wait for build → Your app will be live!

---

## 📄 Inputs

- `.inp`: EPANET model input file
- `.csv`: Observed data, indexed by timestamp, with columns matching node/pipe names

---

## 📤 Output

- Calibrated roughness CSV
- Updated `.inp` file with new roughness values

---

## 🛠️ Features

- Roughness calibration using observed data
- RMSE evaluation
- Pressure and flow comparison plots
- Download updated models

---

## 🧪 Example `.csv` Format

```csv
Time,Junction1,Pipe12
00:00,30.5,5.0
01:00,31.2,5.1
...
```

---

## 👨‍💻 Author

Built with 💙 using Python, Streamlit, and EPANET (via WNTR)

prof Francesco De Paola
---

## 📃 License

MIT License
