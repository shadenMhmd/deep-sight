# app.py
import os
import io
from datetime import datetime, date

from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, session, Response, abort
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import cv2
import tensorflow as tf

# -----------------------------------
# Flask & DB config
# -----------------------------------
app = Flask(__name__)
app.secret_key = "replace-with-a-strong-secret"   # غيّريه لقيمة قوية

app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root@localhost:3306/deepsight_db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# مسارات رفع الملفات
STATIC_DIR = os.path.join(app.root_path, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}

def allowed_ext(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# -----------------------------------
# تحميل نموذج DenseNet
# -----------------------------------
MODEL_PATH = os.path.join(app.root_path, "models", "best_model_fold_3.keras")
model = load_model(MODEL_PATH)
CLASSES = ["Normal", "DME"]  # ترتيب الفئات في التقرير النهائي

# -----------------------------------
# ORM Models (الأسماء كما هي)
# -----------------------------------
class Doctor(db.Model):
    __tablename__ = "Doctors"
    ID             = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Doctor_ID      = db.Column(db.String(20), unique=True, nullable=False)
    Doctor_Name    = db.Column(db.String(100), nullable=False)
    Password       = db.Column(db.String(100), nullable=False)
    Specialization = db.Column(db.String(50), nullable=False)
    Phone_Num      = db.Column(db.String(15))

class Patient(db.Model):
    __tablename__ = "Patients"
    ID            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Patient_ID    = db.Column(db.String(20), unique=True, nullable=False)
    Patient_Name  = db.Column(db.String(100), nullable=False)
    Gender        = db.Column(db.String(10), nullable=False)
    Date_Of_Birth = db.Column(db.Date, nullable=False)

class Diagnosis(db.Model):
    __tablename__ = "Diagnoses"
    ID               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Patient_Name     = db.Column(db.String(100), nullable=False)
    Patient_ID       = db.Column(db.String(20), db.ForeignKey("Patients.Patient_ID"), nullable=False)
    Doctor_Name      = db.Column(db.String(100), nullable=False)
    Date_Of_Scan     = db.Column(db.Date, nullable=False, default=date.today)
    Diagnosis_Result = db.Column(db.String(200))

# -----------------------------------
# Auth & basic pages
# -----------------------------------
@app.get("/login")
def login():
    return render_template("Login.html")

@app.post("/login")
def login_post():
    # POST
    doctor_id = (request.form.get("id") or "").strip()
    password  = request.form.get("password") or ""

    if not doctor_id or not password:
        return "Missing ID or password", 400

    doc = Doctor.query.filter_by(Doctor_ID=doctor_id).first()
    if not doc or doc.Password != password:
        return "Invalid ID or password", 401

    # نجاح تسجيل الدخول
    session["doctor_id"] = doc.Doctor_ID
    session["doctor_name"] = doc.Doctor_Name
    session["doctor_specialty"] = doc.Specialization
    return jsonify(ok=True, redirect=url_for("starting"))

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.get("/starting")
def starting():
    if "doctor_id" not in session:
        return redirect(url_for("login"))
    return render_template("Starting.html")

# -----------------------------------
# Grad-CAM helper (مع تخفيف الأحمر للـ Normal فقط)
# -----------------------------------
def build_heatmap(saved_path: str, x_batch: np.ndarray, cls_idx: int, label_text: str) -> str | None:
    """
    يبني Heatmap فوق الصورة المحفوظة saved_path
    ويعيد اسم الملف الناتج داخل static/uploads أو None عند الفشل.
    """
    try:
        # حددي طبقة كونف مناسبة لـ DenseNet. غيّري الاسم إذا اختلف في موديلك.
        last_conv = model.get_layer("conv5_block16_concat")
        heatmap_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = heatmap_model(x_batch)
            loss = preds[:, cls_idx]

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heat = tf.reduce_mean(conv_out * pooled, axis=-1).numpy()[0]

        heat = np.maximum(heat, 0)
        if heat.max() > 0:
            heat /= heat.max()

        # اقرأ الصورة الأصلية من القرص (BGR)
        orig = cv2.imread(saved_path)
        if orig is None:
            return None

        heat = cv2.resize(heat, (orig.shape[1], orig.shape[0]))
        heat = np.uint8(255 * heat)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # BGR

        # لو النتيجة Normal: خفّضي القناة الحمراء بشكل كبير (بـ BGR القناة 2 = R)
        if (label_text or "").lower() == "normal":
            colored[:, :, 2] = (colored[:, :, 2] * 0.15).astype(np.uint8)  # تقليل الأحمر فقط

        overlay = cv2.addWeighted(orig, 0.6, colored, 0.4, 0)

        heatmap_name = "heatmap_" + os.path.basename(saved_path)
        heatmap_path = os.path.join(UPLOAD_DIR, heatmap_name)
        ok = cv2.imwrite(heatmap_path, overlay)
        return heatmap_name if ok and os.path.exists(heatmap_path) else None
    except Exception as e:
        print("Grad-CAM error:", repr(e))
        return None

# -----------------------------------
# Upload -> predict -> save diagnosis -> results
# -----------------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if "doctor_id" not in session:
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("Upload.html")

    # 1) بيانات المريض
    patient_name = (request.form.get("fname") or "").strip()
    patient_id   = (request.form.get("id") or "").strip()
    gender       = (request.form.get("gender") or "").strip()
    dob_raw      = (request.form.get("dateOfBirth") or "").strip()

    # 2) الصورة
    f = request.files.get("uploadImage")
    if not f or f.filename == "":
        return "OCT image is required", 400
    if not allowed_ext(f.filename):
        return "Only PNG/JPG images are allowed", 415

    # 3) احفظي الصورة داخل static/uploads
    safe_name = secure_filename(f.filename)
    saved_path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(saved_path)

    # 4) حضّري الصورة للمودل
    pil = Image.open(saved_path).convert("RGB").resize((224, 224))
    x = np.asarray(pil, dtype="float32")
    x = np.expand_dims(x, 0)
    x = preprocess_input(x)

    # 5) التوقع
    y = model.predict(x, verbose=0)[0]
    cls_idx = int(np.argmax(y))
    # ترتيب التقارير الذي تستخدمينه في الواجهة
    # لو كانت مصفوفة المودل عكسه غيّري هنا فقط
    classes_report = ["DME", "Normal"]
    label = classes_report[cls_idx] if cls_idx < len(classes_report) else "None"

    score = float(np.max(y))
    conf_percent = round(score * 100, 2)

    # 6) Grad-CAM مع تخفيف الأحمر لو Normal
    heatmap_name = build_heatmap(saved_path, x, cls_idx, label_text=label)

    # 7) حفظ/تحديث المريض
    try:
        dob = datetime.strptime(dob_raw, "%Y-%m-%d").date()
    except ValueError:
        return "Invalid date format", 400

    patient = Patient.query.filter_by(Patient_ID=patient_id).first()
    if patient:
        patient.Patient_Name = patient_name
        patient.Gender = gender
        patient.Date_Of_Birth = dob
    else:
        patient = Patient(
            Patient_ID=patient_id,
            Patient_Name=patient_name,
            Gender=gender,
            Date_Of_Birth=dob
        )
        db.session.add(patient)
    db.session.commit()

    # 8) سجلّ التشخيص
    diag = Diagnosis(
        Patient_Name=patient_name,
        Patient_ID=patient_id,
        Doctor_Name=session.get("doctor_name", ""),
        Date_Of_Scan=date.today(),
        Diagnosis_Result=label
    )
    db.session.add(diag)
    db.session.commit()

    # 9) خزّني في السيشن للعرض في /report
    session["last_patient_id"]     = patient_id
    session["last_patient_name"]   = patient_name
    session["last_patient_gender"] = gender
    session["last_patient_dob"]    = dob_raw
    session["last_scan_time"]      = datetime.now().isoformat(timespec="minutes")
    session["last_image_name"]     = os.path.basename(saved_path)
    session["last_label"]          = label
    session["last_confidence"]     = conf_percent
    if heatmap_name:
        session["last_heatmap_name"] = heatmap_name
    else:
        session["last_heatmap_name"] = ""

    return redirect(url_for("results"))

# -----------------------------------
# Results & Report
# -----------------------------------
@app.get("/results")
def results():
    if "doctor_id" not in session:
        return redirect(url_for("login"))
    return render_template(
        "Results.html",
        patient_id=session.get("last_patient_id"),
        patient_name=session.get("last_patient_name"),
        image_name=session.get("last_image_name"),
        label=session.get("last_label")
    )

@app.get("/report")
def report():
    if "doctor_id" not in session:
        return redirect(url_for("login"))

    report_ctx = {
        "patientName": session.get("last_patient_name", "—"),
        "patientID":   session.get("last_patient_id", "—"),
        "gender":      session.get("last_patient_gender", "—"),
        "dob":         session.get("last_patient_dob", ""),
        "scan_time":   session.get("last_scan_time", "—"),
        "result":      session.get("last_label", "—"),
        "confidence":  session.get("last_confidence", "—"),
        "heatmapUrl":  url_for("static", filename=f"uploads/{session.get('last_heatmap_name','')}") if session.get("last_heatmap_name") else "",
        "octImage":    url_for("static", filename=f"uploads/{session.get('last_image_name','')}") if session.get("last_image_name") else ""
    }
    return render_template("ViewReport.html", report=report_ctx)

# -----------------------------------
# History API + Page
# -----------------------------------
@app.get("/history")
def history():
    if "doctor_id" not in session:
        return redirect(url_for("login"))
    return render_template("History.html")

@app.get("/get_history")
def get_history():
    if "doctor_id" not in session:
        return jsonify([])

    rows = (
        db.session.query(Diagnosis)
        .order_by(Diagnosis.ID.desc())
        .limit(500)
        .all()
    )
    data = []
    for r in rows:
        # Date_Of_Scan عندك نوع Date؛ نخليه نص ISO
        scan_str = r.Date_Of_Scan.isoformat() if r.Date_Of_Scan else ""
        item = {
            "id": r.ID,
            "patient_name": r.Patient_Name or "",
            "doctor_name":  r.Doctor_Name  or "",
            "result":       r.Diagnosis_Result or "",
            # مفاتيح متوافقة مع السكربت القديم والجديد:
            "patient_id":   r.Patient_ID or "",
            "Patient_ID":   r.Patient_ID or "",
            "scan_time":    scan_str,
            "scan_datetime": scan_str,
        }
        data.append(item)
    return jsonify(data)



# تنزيل تقرير HTML مبني لحظياً من الداتابيس
@app.get("/download_report/<int:diag_id>")
def download_report(diag_id: int):
    if "doctor_id" not in session:
        return redirect(url_for("login"))

    d = Diagnosis.query.filter_by(ID=diag_id).first()
    if d is None:
        abort(404)

    p = Patient.query.filter_by(Patient_ID=d.Patient_ID).first()
    # بناء سياق التقرير (صور الرفع الأخيرة غير محفوظة بالـ DB، لذا هنا نُبقيها فارغة)
    report_ctx = {
        "patientName": d.Patient_Name or (p.Patient_Name if p else "—"),
        "patientID":   d.Patient_ID,
        "gender":      (p.Gender if p else "—"),
        "dob":         (p.Date_Of_Birth.isoformat() if p else ""),
        "scan_time":   d.Date_Of_Scan.isoformat(),
        "result":      d.Diagnosis_Result or "—",
        "confidence":  "—",
        "heatmapUrl":  "",  # لا نعرف اسم الملف التاريخي؛ نتركه فارغًا
        "octImage":    ""
    }

    html = render_template("ViewReport.html", report=report_ctx)
    filename = f"DeepSight_Report_{d.Patient_ID}_{d.ID}.html"
    return Response(
        html,
        headers={
            "Content-Type": "text/html; charset=utf-8",
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

# -----------------------------------
# Support page (رجّعناه عشان الروابط تشتغل)
# -----------------------------------
@app.get("/support")
def support():
    if "doctor_id" not in session:
        return redirect(url_for("login"))
    return render_template("Support.html")

# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
