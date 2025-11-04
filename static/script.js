/// scripts.js

document.addEventListener('DOMContentLoaded', () => {
    const userIcon = document.getElementById('userIcon');
    const userDropdown = document.getElementById('userDropdown');

    if (userIcon && userDropdown) {
        userIcon.addEventListener('click', (event) => {
            // تبديل حالة الفئة 'show' لإظهار أو إخفاء القائمة
            userDropdown.classList.toggle('show');
            event.stopPropagation(); // منع إغلاق القائمة فوراً عند النقر
        });

        // إغلاق القائمة إذا نقر المستخدم في أي مكان خارجها
        document.addEventListener('click', (event) => {
            // التحقق من وجود الفئة 'show' ومن أن النقرة ليست على الأيقونة أو داخل القائمة
            if (userDropdown.classList.contains('show') && !userDropdown.contains(event.target) && event.target !== userIcon) {
                userDropdown.classList.remove('show');
            }
        });
    }
});


const userN = localStorage.getItem('userID');
document.getElementById('nameN').textContent = userN || "—";

const icon = document.getElementById('userIcon');
  if (userN && userN.trim() !== "") {
    icon.textContent = userN.trim().charAt(0).toUpperCase();
  } else {
    icon.textContent = "—"; // لو ما فيه اسم
  }



