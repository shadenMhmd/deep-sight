from flask_sqlalchemy import SQLAlchemy
from flask import Flask

# نعرّف التطبيق Flask
app = Flask(__name__)

# إعدادات الاتصال بقاعدة البيانات MySQL
# 🔸 غيّري كلمة المرور واسم قاعدة البيانات حسب إعداداتك في MySQL
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root@localhost:3306/deepsight_db"

# 🔸 اختيارياً: تعطيل الرسائل التحذيرية
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# إنشاء كائن قاعدة البيانات
db = SQLAlchemy(app)




class Prediction(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    filename  = db.Column(db.String(255))
    predicted = db.Column(db.String(128))
    score     = db.Column(db.Float)
