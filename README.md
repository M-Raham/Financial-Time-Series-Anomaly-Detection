# Financial Time-Series Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Stock market anomaly detection using Prophet, Isolation Forest, and technical indicators.

---

## 📦 Requirements

### Core Dependencies
```text
yfinance==0.2.55
prophet==1.1.6
scikit-learn==1.6.1
pandas==2.2.3
matplotlib==3.10.1
numpy==2.2.4
```

### Full Environment
For complete reproducibility, see [requirements.txt](requirements.txt) (generated via `pip freeze`).

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-anomaly-detection.git
   cd stock-anomaly-detection
   ```

2. Install either:
   - **Minimal setup** (core functionality):
     ```bash
     pip install yfinance prophet scikit-learn pandas matplotlib numpy
     ```
   - **Full environment** (exact versions):
     ```bash
     pip install -r requirements.txt
     ```

---

## 🚀 Usage
```bash
python stock_anomaly_detection.py
```
**Outputs:**
- Interactive plots with anomaly markers
- Terminal report of detected anomalies

---

## 🧹 Clean Dependency Management
To keep your environment lean:
1. Create a minimal `requirements.in`:
   ```text
   yfinance
   prophet
   scikit-learn
   pandas
   matplotlib
   ```
2. Generate optimized requirements:
   ```bash
   pip-compile requirements.in  # Requires pip-tools
   ```

---

## 📜 License
MIT License - See [LICENSE](LICENSE).

```

Key improvements:
1. **Dual Installation Options** - Users can choose between minimal or full setup
2. **Dependency Hygiene** - Added guidance for maintaining clean requirements
3. **Reduced Clutter** - Shows only core packages in the main README
4. **pip-tools Suggestion** - Best practice for dependency management

For your `requirements.txt`, I recommend:
1. Either keep the full file for reproducibility
2. Or regenerate with just the essentials:
   ```bash
   pip freeze | grep -E 'yfinance|prophet|scikit-learn|pandas|matplotlib|numpy' > requirements.txt
   ```
