from flask import Flask, request, send_file
from flask_cors import CORS
import pandas as pd
import io
from mlpredict import predict, x_y_split
import joblib
from mlutils import *

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_route():
    model_name = request.form.get('model')
    file = request.files['file']
    # 保存文件在flask文件夹下
    file_path = 'flask_file/' + file.filename
    file.save(file_path)
    # 读取文件路径
    test_x, test_y = x_y_split(file_path, scaler=joblib.load(scaler_model_path))
    y_pred = predict(model_name, test_x)

    # 将预测结果写入到原文件中
    original_data = pd.read_excel(file_path, skiprows=0)
    original_data['YS_Pred'] = y_pred[:, 0]
    original_data['TS_Pred'] = y_pred[:, 1]
    original_data['EL_Pred'] = y_pred[:, 2]
    out_path = 'flask_file/prediction' + f'-{model_name}.xlsx'
    original_data.to_excel(out_path, index=False)

    # 读取预测结果
    output = io.BytesIO()
    with open(out_path, 'rb') as f:
        output.write(f.read())
    output.seek(0)
    
    return send_file(output, as_attachment=True, download_name='prediction_result.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
