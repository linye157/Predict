from mlutils import *

#确保88%以上的屈服强度YS误差在6%以内，90%以上的抗拉强度TS在6%以内，90%的延伸率在4%EL以内
def evaluate(y_true, y_pred,):
    #计算屈服强度YS误差在6%以内的比例
    YS=0
    for i in range(len(y_true)):
        if abs(y_true[i][0]-y_pred[i][0])<=0.06*y_true[i][0]:
            YS+=1
    YS=YS/len(y_true)

    #计算抗拉强度TS误差在6%以内的比例
    TS=0
    for i in range(len(y_true)):
        if abs(y_true[i][1]-y_pred[i][1])<=0.06*y_true[i][1]:
            TS+=1
    TS=TS/len(y_true)

    #计算延伸率EL误差在4%以内的比例
    EL=0
    for i in range(len(y_true)):
        if abs(y_true[i][2]-y_pred[i][2])<=0.04*y_true[i][2]:
            EL+=1
    EL=EL/len(y_true)

    return {'YS_Eva':YS*100,'TS_Eva':TS*100,'EL_Eva':EL*100}

    

#计算屈服强度YS，抗拉强度TS，延伸率EL的均方根误差，返回字典
def rmsle(y_true, y_pred):
    rmsle_YS = np.sqrt(mean_squared_error(y_true[:,0], y_pred[:,0]))
    rmsle_TS = np.sqrt(mean_squared_error(y_true[:,1], y_pred[:,1]))
    rmsle_EL = np.sqrt(mean_squared_error(y_true[:,2], y_pred[:,2]))
    return {'YS_RMSE':rmsle_YS,'TS_RMSE':rmsle_TS,'EL_RMSE':rmsle_EL}


#计算预测的目标值列表和真实值列表的误差平均百分比
def mape(y_true, y_pred):
    mape_YS = np.mean(np.abs((y_true[:,0] - y_pred[:,0]) / y_true[:,0])) * 100
    mape_TS = np.mean(np.abs((y_true[:,1] - y_pred[:,1]) / y_true[:,1])) * 100
    mape_EL = np.mean(np.abs((y_true[:,2] - y_pred[:,2]) / y_true[:,2])) * 100
    return {'YSmape':mape_YS,'TSmape':mape_TS,'ELmape':mape_EL}
    
def predict(model_name, X_test ):
    model = models[model_name]
    model = joblib.load(pre_model_path + model_name + '.pkl')
    y_pred = model.predict(X_test)
    return y_pred

if __name__ == '__main__':
    # 读取测试数据
    test_x, test_y = x_y_split(test_data_path, scaler=joblib.load(scaler_model_path))
    # 预测
    for model_name in models.keys():
        y_pred = predict(model_name, test_x)
        print(model_name)
        print(evaluate(test_y.values, y_pred))
        print(rmsle(test_y.values, y_pred))
        print(mape(test_y.values, y_pred))
        print('---------------------------------')
    # y_pred = predict('LightGBM', test_x_scaled)
    # #打印出y_pred和test_y的类型
    # print(type(y_pred))
    # print(type(test_y.values))

    # print(y_pred.shape)
    # print(test_y.values.shape)


