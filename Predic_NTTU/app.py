import streamlit as st
import numpy as np
import pandas as pd
import pickle
import pdfplumber
import csv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cấu hình trang
st.set_page_config(page_title="Dự đoán điểm sinh viên", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center;'>🎓 Hệ thống dự đoán Điểm Cuối Kỳ & Khả Năng Qua Môn</h1>
    """,
    unsafe_allow_html=True
)

def reset_tab3_state():
    st.session_state.pop('uploaded_file', None)
    st.session_state.pop('df_filtered', None)
    st.session_state.pop('df_result', None)
    st.session_state.pop('result_file', None)

# --- Giao diện upload file PDF ---
st.header("📂 Tải lên file PDF chứa dữ liệu điểm")
uploaded_files = st.file_uploader("Chọn một hoặc nhiều file PDF", accept_multiple_files=True, type=["pdf"])

csv_path = "output.csv"
data = []
headers = ["STT", "Mã SV", "Họ đệm", "Tên", "Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ", "Điểm tổng kết", "Điểm chữ", "Xếp loại", "Rớt môn"]

if uploaded_files:
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    for row in table:
                        if row and row[0] and row[0].isdigit():
                            try:
                                stt = row[0]
                                ma_sv = row[1]
                                ho_ten = row[2].split()
                                ho_dem = " ".join(ho_ten[:-1])
                                ten = ho_ten[-1]
                                giua_ky = row[3] or ""
                                thuong_ky = row[4] or ""
                                thuc_hanh = row[7] or ""
                                diem_cuoi_ky = row[10] or ""
                                diem_tong_ket = row[12] or ""
                                diem_chu = row[14] or ""
                                xep_loai = row[15] or ""
                                rot_mon = 1 if diem_chu.strip().upper() == "F" else 0
                                data.append([stt, ma_sv, ho_dem, ten, giua_ky, thuong_ky, thuc_hanh, diem_cuoi_ky, diem_tong_ket, diem_chu, xep_loai, rot_mon])
                            except IndexError:
                                with open("error_log.txt", "a", encoding="utf-8") as log_file:
                                    log_file.write(f"Bỏ qua dòng bị lỗi: {row}\n")

    with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

    st.success("✅ Dữ liệu đã được lưu vào output.csv")

# --- Hàm tải và xử lý dữ liệu ---
@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    cols_to_convert = ["Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ"]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
    df = df.dropna()
    return df

try:
    df_clean = load_data()
except FileNotFoundError:
    df_clean = pd.DataFrame()

# --- Hàm huấn luyện mô hình ---
def train_models(df):
    X_reg = df[["Giữa kỳ", "Thường kỳ", "Thực hành"]]
    y_reg = df["Điểm cuối kỳ"]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    X_clf = df[["Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ"]]
    y_clf = df["Rớt môn"]
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_reg, y_train_reg)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_clf, y_train_clf)
    
    return rf_regressor, rf_classifier

if not df_clean.empty:
    rf_regressor, rf_classifier = train_models(df_clean)
else:
    rf_regressor, rf_classifier = None, None

# --- Giao diện Streamlit ---


# Tabs
tab2, tab1, tab3 = st.tabs(["📂 Dữ liệu", "📊 Dự đoán", "🧠 Dự đoán bằng file Excel"])

with tab1:
    st.header("🔍 Nhập điểm để dự đoán")
    giua_ky = st.number_input("Nhập điểm giữa kỳ", min_value=0.0, max_value=10.0, step=0.1)
    thuong_ky = st.number_input("Nhập điểm thường kỳ", min_value=0.0, max_value=10.0, step=0.1)
    thuc_hanh = st.number_input("Nhập điểm thực hành", min_value=0.0, max_value=10.0, step=0.1)
    
    if st.button("📌 Dự đoán") and rf_regressor and rf_classifier:
        # Tạo DataFrame cho dự đoán với tên cột
        input_reg = pd.DataFrame([[giua_ky, thuong_ky, thuc_hanh]], 
                                 columns=["Giữa kỳ", "Thường kỳ", "Thực hành"])
        diem_cuoi_ky = rf_regressor.predict(input_reg)[0]
        
        input_clf = pd.DataFrame([[giua_ky, thuong_ky, thuc_hanh, diem_cuoi_ky]], 
                                 columns=["Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ"])
        rot_mon = rf_classifier.predict(input_clf)[0]
        
        st.write(f"### 📈 Điểm cuối kỳ dự đoán: {diem_cuoi_ky:.2f}")
        if rot_mon == 0:
            st.success("✅ Sinh viên có khả năng qua môn!")
        else:
            st.error("❌ Sinh viên có nguy cơ rớt môn!")
        
        df_clean["Khoảng cách"] = euclidean_distances(df_clean[["Giữa kỳ", "Thường kỳ", "Thực hành"]], input_reg).flatten()
        similar_students = df_clean.nsmallest(5, "Khoảng cách")[["Mã SV", "Họ đệm", "Tên", "Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ"]]
        
        st.write("### 🏫 Sinh viên có điểm tương đồng:")
        st.dataframe(similar_students)

with tab2:
    st.header("📂 Dữ liệu gốc")
    if not df_clean.empty:
        st.dataframe(df_clean)
    else:
        st.warning("Chưa có dữ liệu để hiển thị. Vui lòng tải file PDF trước.")

    st.subheader("📊 Trực quan hóa dữ liệu")
    if not df_clean.empty and st.button("📌 Hiển thị tất cả biểu đồ"):
        # Biểu đồ hộp (Box Plot)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(data=df_clean[["Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ"]], ax=ax)
        ax.set_title("Biểu đồ hộp của các điểm số")
        st.pyplot(fig)

        # Biểu đồ phân tán (Scatter Plot)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(data=df_clean, x="Điểm cuối kỳ", y="Giữa kỳ", label="Giữa kỳ", alpha=0.7)
        sns.scatterplot(data=df_clean, x="Điểm cuối kỳ", y="Thường kỳ", label="Thường kỳ", alpha=0.7)
        sns.scatterplot(data=df_clean, x="Điểm cuối kỳ", y="Thực hành", label="Thực hành", alpha=0.7)
        ax.set_title("Mối quan hệ giữa điểm thành phần và điểm cuối kỳ")
        st.pyplot(fig)

        # Biểu đồ Pie Chart Đậu/Rớt
        pass_fail_counts = df_clean["Rớt môn"].value_counts()
        labels = ["Đậu", "Rớt"]
        colors = ["#4CAF50", "#FF5722"]

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie(pass_fail_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
        ax.set_title("Tỷ lệ sinh viên Đậu/Rớt")
        st.pyplot(fig)


with tab3:
    st.title("Dự đoán Điểm cuối kỳ và Rủi ro Rớt môn")
    uploaded_file = st.file_uploader("Tải lên file Excel", type=["xlsx"])

    if uploaded_file:
        # Lưu file tạm thời để đọc
        file_path = "uploaded_file.xlsx"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Đọc file Excel
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

        # Xử lý dữ liệu trực tiếp không lưu file trung gian
        df_cleaned = df.dropna(how="all").reset_index(drop=True)
        stt_column = df_cleaned.iloc[:, 0]
        stt_start_index = stt_column[stt_column == 1].index[0]
        df_filtered = df_cleaned.iloc[stt_start_index:].reset_index(drop=True)
        df_filtered.columns = [
            "STT", "MSSV", "Họ", "Tên", "Giới tính", "Ngày sinh", "Lớp",
            "Giữa kỳ", "Thường kỳ", "Điểm 3", "Điểm 4", "Thực hành",
            "Điểm 6", "Điểm 7", "Điểm 8", "Điểm 9", "Điểm 10"
        ]
        df_filtered = df_filtered.dropna(axis=1, how="all")
        df_filtered = df_filtered[~df_filtered.astype(str).apply(
            lambda row: row.str.contains(
                "Tổng cộng|Các bộ giảng dạy|Cán bộ Giảng dạy|Trưởng khoa - Trưởng bộ môn",
                na=False, case=False
            )
        ).any(axis=1)]

        # Chuyển đổi điểm số về kiểu số
        df_filtered[["Giữa kỳ", "Thường kỳ", "Thực hành"]] = df_filtered[["Giữa kỳ", "Thường kỳ", "Thực hành"]].apply(pd.to_numeric, errors="coerce")

        # Dự đoán hàng loạt sinh viên
        def predict_students(df):
            try:
                if not os.path.exists("du_doan_diem_cuoiky.pkl"):
                    st.error("❌ Không tìm thấy file mô hình 'du_doan_diem_cuoiky.pkl'. Vui lòng kiểm tra đường dẫn hoặc huấn luyện mô hình trước!")
                    return None, None
                
                rf_regressor = joblib.load("du_doan_diem_cuoiky.pkl")
                df["Dự đoán Cuối kỳ"] = rf_regressor.predict(df[["Giữa kỳ", "Thường kỳ", "Thực hành"]])
                df["Dự đoán qua môn"] = np.where(df["Dự đoán Cuối kỳ"] >= 4, "Qua môn", "Rớt môn")

                output_file = "du_doan_ketqua.xlsx"
                df.to_excel(output_file, index=False)
                return df, output_file
            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi khi dự đoán: {str(e)}")
                return None, None

        if st.button("Dự đoán Điểm Cuối kỳ và Rủi ro Rớt môn"):
            df_result, result_file = predict_students(df_filtered)
            if df_result is not None:  # Kiểm tra xem hàm có trả về kết quả hay không
                st.success("✅ Dự đoán hoàn tất!")
                st.dataframe(df_result)

                # Hiển thị danh sách sinh viên dự đoán rớt môn
                st.subheader("Danh sách sinh viên dự đoán rớt môn")
                df_failed = df_result[df_result["Dự đoán qua môn"] == "Rớt môn"]
                st.dataframe(df_failed)

                st.download_button(
                    label="📥 Tải về kết quả dự đoán",
                    data=open(result_file, "rb").read(),
                    file_name="du_doan_ketqua.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Trực quan hóa kết quả
                st.subheader("Phân bố điểm cuối kỳ dự đoán")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.histplot(df_result["Dự đoán Cuối kỳ"], bins=10, kde=True, ax=ax)
                ax.set_xlabel("Điểm Cuối kỳ Dự đoán")
                ax.set_ylabel("Số lượng sinh viên")
                st.pyplot(fig)
                
                st.subheader("Tỷ lệ sinh viên qua môn vs rớt môn")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.countplot(x="Dự đoán qua môn", data=df_result, hue="Dự đoán qua môn", palette="coolwarm", legend=False, ax=ax)
                ax.set_xlabel("Kết quả Dự đoán")
                ax.set_ylabel("Số lượng sinh viên")
                st.pyplot(fig)
