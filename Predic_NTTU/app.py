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
import os

# Cấu hình trang
st.set_page_config(page_title="Dự đoán điểm sinh viên", layout="wide")

# Ẩn thanh công cụ Streamlit và các biểu tượng "Running", "Share"
st.markdown(
    """
    <style>
    [data-testid="stToolbar"] {
        visibility: hidden;
    }
    [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    [data-testid="stDecoration"] {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hàm kiểm tra đăng nhập
def check_login(username, password):
    users = {
        "gv": {"password": "gv123", "role": "giangvien"},
        "sv": {"password": "sv123", "role": "sinhvien"}
    }
    if username in users and users[username]["password"] == password:
        return users[username]["role"]
    return None

# Khởi tạo trạng thái phiên
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None

# Giao diện đăng nhập
if not st.session_state.logged_in:
    st.markdown(
        """
        <h1 style='text-align: center;'>🎓 Đăng nhập Hệ thống Dự đoán Điểm</h1>
        """,
        unsafe_allow_html=True
    )
    
    with st.form(key="login_form"):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submit_button = st.form_submit_button(label="Đăng nhập")

        if submit_button:
            role = check_login(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.success(f"Đăng nhập thành công với vai trò: {role}")
                st.rerun()
            else:
                st.error("Tên đăng nhập hoặc mật khẩu không đúng!")
else:
    # Giao diện sau khi đăng nhập
    st.markdown(
        """
        <h1 style='text-align: center;'>🎓 Hệ thống dự đoán Điểm Cuối Kỳ & Khả Năng Qua Môn</h1>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("Đăng xuất", key="logout_button"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.rerun()

    # --- Các hàm xử lý dữ liệu ---
    def reset_tab3_state():
        st.session_state.pop('uploaded_file', None)
        st.session_state.pop('df_filtered', None)
        st.session_state.pop('df_result', None)
        st.session_state.pop('result_file', None)

    # Xử lý file PDF (chỉ dành cho GV)
    if st.session_state.role == "giangvien":
        st.header("📂 Tải lên file PDF chứa dữ liệu điểm")
        uploaded_files = st.file_uploader("Chọn một hoặc nhiều file PDF", accept_multiple_files=True, type=["pdf"], key="pdf_uploader")

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
        df = pd.read_csv("output.csv")
        cols_to_convert = ["Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ"]
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
        df = df.dropna()
        return df

    # Kiểm tra và tải dữ liệu nếu file output.csv tồn tại
    df_clean = pd.DataFrame()
    if os.path.exists("output.csv"):
        try:
            df_clean = load_data()
        except Exception as e:
            st.error(f"❌ Lỗi khi tải dữ liệu từ output.csv: {str(e)}")

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

    # Huấn luyện mô hình nếu dữ liệu tồn tại
    rf_regressor, rf_classifier = None, None
    if not df_clean.empty:
        rf_regressor, rf_classifier = train_models(df_clean)

    # --- Tabs dựa trên vai trò ---
    if st.session_state.role == "giangvien":
        tab2, tab1, tab3 = st.tabs(["📂 Dữ liệu", "📊 Dự đoán", "🧠 Dự đoán bằng file Excel"])
    else:  # Sinh viên
        tab1 = st.tabs(["📊 Dự đoán"])[0]

    # Tab Dự đoán (cả GV và SV đều thấy)
    with tab1:
        st.header("🔍 Nhập điểm để dự đoán")
        giua_ky = st.number_input("Nhập điểm giữa kỳ", min_value=0.0, max_value=10.0, step=0.1, key="giua_ky_input")
        thuong_ky = st.number_input("Nhập điểm thường kỳ", min_value=0.0, max_value=10.0, step=0.1, key="thuong_ky_input")
        thuc_hanh = st.number_input("Nhập điểm thực hành", min_value=0.0, max_value=10.0, step=0.1, key="thuc_hanh_input")
        
        if st.button("📌 Dự đoán", key="predict_button_tab1"):
            if rf_regressor and rf_classifier:
                input_reg = pd.DataFrame([[giua_ky, thuong_ky, thuc_hanh]], 
                                        columns=["Giữa kỳ", "Thường kỳ", "Thực hành"])
                diem_cuoi_ky = rf_regressor.predict(input_reg)[0]
                
                input_clf = pd.DataFrame([[giua_ky, thuong_ky, thuc_hanh, diem_cuoi_ky]], 
                                        columns=["Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ"])
                rot_mon = rf_classifier.predict(input_clf)[0]
                
                st.write(f"### 📈 Điểm cuối kỳ dự đoán: {diem_cuoi_ky:.2f}")
                
                # Kiểm tra vai trò để hiển thị thông báo phù hợp
                if st.session_state.role == "sinhvien":
                    if rot_mon == 0:
                        st.success("✅ Sinh viên có khả năng qua môn!")
                        st.warning("⚠️ Bạn hiện tại dự đoán là qua môn, nhưng hãy đừng chủ quan nhé!")
                    else:
                        st.error("❌ Sinh viên có nguy cơ rớt môn!")
                        st.warning("⚠️ Bạn nên học bài thật tốt hơn để không bị nguy cơ rớt môn.")
                else:  # Vai trò giảng viên
                    if rot_mon == 0:
                        st.success("✅ Sinh viên có khả năng qua môn!")
                    else:
                        st.error("❌ Sinh viên có nguy cơ rớt môn!")
                    
                    # Hiển thị sinh viên tương đồng chỉ cho giảng viên
                    df_clean["Khoảng cách"] = euclidean_distances(df_clean[["Giữa kỳ", "Thường kỳ", "Thực hành"]], input_reg).flatten()
                    similar_students = df_clean.nsmallest(5, "Khoảng cách")[["Mã SV", "Họ đệm", "Tên", "Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ"]]
                    st.write("### 🏫 Sinh viên có điểm tương đồng:")
                    st.dataframe(similar_students)
            else:
                st.error("❌ Mô hình chưa được huấn luyện. Vui lòng kiểm tra file output.csv hoặc liên hệ giảng viên để tải dữ liệu!")

    # Các tab chỉ dành cho GV
    if st.session_state.role == "giangvien":
        with tab2:
            st.header("📂 Dữ liệu gốc")
            if not df_clean.empty:
                st.dataframe(df_clean)
            else:
                st.warning("Chưa có dữ liệu để hiển thị. Vui lòng tải file PDF trước.")

            st.subheader("📊 Trực quan hóa dữ liệu")
            if not df_clean.empty and st.button("📌 Hiển thị tất cả biểu đồ", key="visualize_button_tab2"):
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.boxplot(data=df_clean[["Giữa kỳ", "Thường kỳ", "Thực hành", "Điểm cuối kỳ"]], ax=ax)
                ax.set_title("Biểu đồ hộp của các điểm số")
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(5, 3))
                sns.scatterplot(data=df_clean, x="Điểm cuối kỳ", y="Giữa kỳ", label="Giữa kỳ", alpha=0.7)
                sns.scatterplot(data=df_clean, x="Điểm cuối kỳ", y="Thường kỳ", label="Thường kỳ", alpha=0.7)
                sns.scatterplot(data=df_clean, x="Điểm cuối kỳ", y="Thực hành", label="Thực hành", alpha=0.7)
                ax.set_title("Mối quan hệ giữa điểm thành phần và điểm cuối kỳ")
                st.pyplot(fig)

                pass_fail_counts = df_clean["Rớt môn"].value_counts()
                labels = ["Đậu", "Rớt"]
                colors = ["#4CAF50", "#FF5722"]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.pie(pass_fail_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
                ax.set_title("Tỷ lệ sinh viên Đậu/Rớt")
                st.pyplot(fig)


        with tab3:
            st.title("Dự đoán Điểm cuối kỳ và Rủi ro Rớt môn")
            uploaded_file = st.file_uploader("Tải lên file Excel", type=["xls", "xlsx"], key="excel_uploader_tab3")
        
            if uploaded_file:
                # Đọc file Excel
                xls = pd.ExcelFile(uploaded_file)
                df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None)
        
                # Tìm dòng bắt đầu dữ liệu (dòng có "STT" ở cột đầu tiên)
                stt_row_idx = None
                for idx, row in df.iterrows():
                    if row[0] == "STT":
                        stt_row_idx = idx
                        break
                
                if stt_row_idx is None:
                    st.error("❌ Không tìm thấy cột 'STT' trong file Excel. Vui lòng kiểm tra lại định dạng file!")
                else:
                    # Lấy hai dòng tiêu đề
                    header_row_1 = df.iloc[stt_row_idx].fillna("").tolist()  # Dòng tiêu đề chính
                    header_row_2 = df.iloc[stt_row_idx + 1].fillna("").tolist()  # Dòng tiêu đề con
        
                    # Kết hợp hai dòng tiêu đề để tạo tên cột
                    combined_headers = []
                    for h1, h2 in zip(header_row_1, header_row_2):
                        if h1 == "STT":
                            combined_headers.append("STT")
                        elif h1 == "Thông tin sinh viên" and h2:
                            combined_headers.append(h2)  # Lấy tiêu đề con (Mã sinh viên, Họ đệm, Tên, ...)
                        elif h1 == "Giữa kỳ\n20%":
                            combined_headers.append("Giữa kỳ")
                        elif h1 == "Thường kỳ 20%":
                            combined_headers.append("Thường kỳ")
                        elif h1 == "Thực hành" and h2 in ["1", "Thực hành 1", "Thực hành"]:
                            combined_headers.append("Thực hành")
                        else:
                            combined_headers.append(h1 if h1 else h2)
        
                    # Lấy dữ liệu từ dòng sau tiêu đề
                    df_cleaned = df.iloc[stt_row_idx + 2:].reset_index(drop=True)
                    df_cleaned.columns = combined_headers
        
                    # Loại bỏ các dòng không phải dữ liệu sinh viên (dòng có "STT" không phải số)
                    df_cleaned = df_cleaned[
                        df_cleaned["STT"].notna() & 
                        df_cleaned["STT"].str.match(r'^\d+$')
                    ]
        
                    # Kiểm tra dữ liệu trong df_cleaned
                    st.write("### Dữ liệu thô (df_cleaned) sau khi lọc dòng không hợp lệ:")
                    # Chuyển đổi kiểu dữ liệu trước khi hiển thị
                    for col in ["Giữa kỳ", "Thường kỳ", "Thực hành"]:
                        if col in df_cleaned.columns:
                            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
                    st.dataframe(df_cleaned)
        
                    # Gán df_filtered = df_cleaned (vì đã lọc ở bước trên)
                    df_filtered = df_cleaned
        
                    # Kiểm tra dữ liệu trong df_filtered
                    st.write("### Dữ liệu sau khi lọc (df_filtered):")
                    st.dataframe(df_filtered)
        
                    # Kiểm tra xem các cột cần thiết có tồn tại không
                    required_columns = ["Giữa kỳ", "Thường kỳ", "Thực hành"]
                    missing_columns = [col for col in required_columns if col not in df_filtered.columns]
                    if missing_columns:
                        st.error(f"❌ File Excel thiếu các cột cần thiết: {', '.join(missing_columns)}")
                    else:
                        # Kiểm tra dữ liệu trước khi dự đoán
                        st.write("### Dữ liệu sau khi xử lý:")
                        # Kiểm tra các cột có trong df_filtered
                        available_columns = df_filtered.columns.tolist()
                        st.write("Các cột có trong df_filtered:", available_columns)
        
                        # Chỉ chọn các cột có trong df_filtered để hiển thị
                        display_columns = ["STT", "Mã sinh viên", "Họ đệm", "Tên", "Giữa kỳ", "Thường kỳ", "Thực hành"]
                        columns_to_display = [col for col in display_columns if col in available_columns]
                        if columns_to_display:
                            st.dataframe(df_filtered[columns_to_display])
                        else:
                            st.error("❌ Không có cột nào trong số ['STT', 'Mã sinh viên', 'Họ đệm', 'Tên', 'Giữa kỳ', 'Thường kỳ', 'Thực hành'] tồn tại trong file Excel!")
                        
                        if df_filtered.empty:
                            st.error("❌ Không có dữ liệu hợp lệ để dự đoán. Vui lòng kiểm tra file Excel!")
                        elif df_filtered[["Giữa kỳ", "Thường kỳ", "Thực hành"]].isna().all().any():
                            st.error("❌ Một hoặc nhiều cột điểm ('Giữa kỳ', 'Thường kỳ', 'Thực hành') chứa toàn giá trị NaN!")
                        else:
                            # Loại bỏ các hàng có giá trị NaN trong các cột điểm
                            df_filtered = df_filtered.dropna(subset=["Giữa kỳ", "Thường kỳ", "Thực hành"])
                            st.write(f"Số lượng mẫu dữ liệu sau khi loại NaN: {len(df_filtered)}")
        
                            def predict_students(df):
                                try:
                                    if not rf_regressor:
                                        st.error("❌ Không tìm thấy mô hình đã huấn luyện. Vui lòng kiểm tra file output.csv hoặc tải file PDF trước!")
                                        return None, None
                                    
                                    if df[["Giữa kỳ", "Thường kỳ", "Thực hành"]].empty:
                                        st.error("❌ Dữ liệu đầu vào rỗng. Không thể dự đoán!")
                                        return None, None
        
                                    df["Dự đoán Cuối kỳ"] = rf_regressor.predict(df[["Giữa kỳ", "Thường kỳ", "Thực hành"]])
                                    df["Dự đoán qua môn"] = np.where(df["Dự đoán Cuối kỳ"] >= 4, "Qua môn", "Rớt môn")
        
                                    output_file = "du_doan_ketqua.xlsx"
                                    df.to_excel(output_file, index=False)
                                    return df, output_file
                                except Exception as e:
                                    st.error(f"❌ Đã xảy ra lỗi khi dự đoán: {str(e)}")
                                    return None, None
        
                            if st.button("Dự đoán Điểm Cuối kỳ và Rủi ro Rớt môn", key="predict_button_tab3"):
                                df_result, result_file = predict_students(df_filtered)
                                if df_result is not None:
                                    st.success("✅ Dự đoán hoàn tất!")
                                    st.dataframe(df_result)
        
                                    st.subheader("Danh sách sinh viên dự đoán rớt môn")
                                    df_failed = df_result[df_result["Dự đoán qua môn"] == "Rớt môn"]
                                    st.dataframe(df_failed)
        
                                    st.download_button(
                                        label="📥 Tải về kết quả dự đoán",
                                        data=open(result_file, "rb").read(),
                                        file_name="du_doan_ketqua.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="download_button_tab3"
                                    )
        
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
