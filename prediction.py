import streamlit as st
import pickle
import pandas as pd

def load_model(model_name):
    if model_name == 'Logistic Regression':
        model = pickle.load(open('models/no_resampling_svm_model.pkl', 'rb'))

       #melakukan prediksi
def predict_attrition(model, data):
    predictions = model.predict(data)
    return predictions

# Mewarnai prediction
def color_predictions(val):
    color = 'red' if val == 'Dropout' else 'green'
    return f'color: {color}'

def main():
    st.title('Prediksi Student Menggunakan Machine Learning')

    # Sidebar
    st.sidebar.title("About Me")
    st.sidebar.write("Name  : Tarish Risqan Karima Hariyono")


     # model ML
    model_name = st.sidebar.selectbox("Pilih Model Machine Learning", ("Logistic Regression", "SVM"))

      # Upload File
    uploaded_file = st.sidebar.file_uploader("Unggah file CSV untuk melanjutkan ke tahap prediksi", type=["csv"])

      if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("Data test:")
        preview_rows = st.slider(" Untuk Melihat Data Geser Slider ini.", 1, len(data), 5)
        st.write(data.head(preview_rows))

        # Extract StudentId dan StudentName
        student_id = data['StudentId']
        student_name = data['StudentName']
        data = data.drop(columns=['StudentId', 'StudentName'])

        # Load model yang dipilih
        model = load_model(model_name)

        # Button untuk trigger
        if st.button('Prediksi'):
            # Melakukan prediksi
            predictions = predict_attrition(model, data)

            # Mengubah value agar mudah dipahami
            prediction_labels = ['Graduate' if pred == 1 else 'Dropout' for pred in predictions]

            # Menampilkan hasilnya
            result_df = pd.DataFrame({
                'StudentId': student_id,
                'Student Name': student_name,
                'Status Prediction': prediction_labels
            })

            # Menampilkan hasil prediksi dengan styling
            st.write("Hasil Prediksi:")
            st.dataframe(result_df.style.applymap(color_predictions, subset=['Status Prediction']))

            # Download hasil prediksi
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download (.csv)",
                data=csv,
                file_name='hasil-prediksi-student.csv',
                mime='text/csv'
            )


            if __name__ == '__main__':
    main()