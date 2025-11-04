# DeepSight

DeepSight is a Flask-based web application for medical image analysis, specifically designed to diagnose Diabetic Macular Edema (DME) using a pre-trained DenseNet model. The application allows doctors to log in, upload OCT (Optical Coherence Tomography) images, predict diagnoses, generate Grad-CAM heatmaps, and store patient records in a MySQL database. The system provides a user-friendly interface for viewing results and downloading reports.

## Features
- **Doctor Authentication**: Secure login system for doctors with unique IDs and passwords.
- **Image Upload and Analysis**: Upload OCT images (PNG/JPG) for DME diagnosis using a DenseNet model.
- **Grad-CAM Heatmaps**: Visualize model predictions with heatmaps, with reduced red intensity for "Normal" results.
- **Patient Management**: Store and update patient information and diagnosis records.
- **Diagnosis History**: View up to 500 recent diagnoses and download reports as HTML files.
- **MySQL Database**: Store doctor, patient, and diagnosis data securely.

## Prerequisites
- **Python 3.10**: Required for compatibility with TensorFlow 2.20.0.
- **MySQL Server**: A running MySQL server on `localhost:3306` with a database named `deepsight_db`.
- **Sufficient Disk Space**: Ensure at least 2 GB of free space on the drive hosting the temporary directory and virtual environment (e.g., D: drive).
- **Git Bash or Command Line Tool**: For running setup commands.

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/shadenMhmd/deep-sight.git
cd DeepSight.bepSight.b
```

### Step 2: Set Up Python 3.10 Virtual Environment

```bach
py -3.10 -m venv .venv
source .venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate  # On Linux/Mac
python --version
```

Note: Ensure Python 3.10 is installed. The command `python --version` should output Python 3.10.x. TensorFlow 2.20.0 requires Python 3.10 for compatibility.

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Set Up MySQL Database

```bash
mysql -u root -p -e "CREATE DATABASE deepsight_db;"
```

Note: Ensure the MySQL server is running on localhost:3306 with user root and no password, or update the SQLALCHEMY_DATABASE_URI in app.py if your credentials differ:

```bash
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root@localhost:3306/deepsight_db"
```

### Step 5: Prepare the Model

```bash
mkdir -p models
```

Note: Place the pre-trained DenseNet model file (best_model_fold_3.keras) in the models directory within the project root (D:\DeepSight.b\DeepSight.b\models). Ensure the model is compatible with TensorFlow 2.20.0.

### Step 6: Set Up Upload Directory

```bash
mkdir -p static/Uploads
```

Note: The application automatically creates a static/Uploads directory for image uploads. Ensure write permissions are available.

## Usage 

1. Run the Application

```bash
python app.py
```

2. Access the Application

- Open a web browser and navigate to http://localhost:5000/login.
- Log in with a doctor’s credentials (stored in the Doctors table).
- Use the interface to:
  - Upload OCT images for diagnosis.
  - View results with Grad-CAM heatmaps.
  - Access patient history and download reports.

## Database Setup

```bash
from app import db
db.create_all()
```

Note: Add doctor records to the Doctors table manually or via a script.

## Project Structure

```bash
DeepSight.b/
├── app.py                # Main Flask application
├── models/               # Directory for the DenseNet model
├── static/               # Static files (CSS, JS, images)
│   └── Uploads/          # Directory for uploaded images and heatmaps
├── templates/            # HTML templates
│   ├── Login.html
│   ├── Starting.html
│   ├── Upload.html
│   ├── Results.html
│   ├── ViewReport.html
│   ├── History.html
│   └── Support.html
└── requirements.txt       # Python dependencies
```

---

> Done by an amzaing team as part of a graduation project requirements.

