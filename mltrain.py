from mlutils import *


def mltrain_model(train_data_dir,model_name):
    train_x,train_y=x_y_split(train_data_dir,scaler=joblib.load(scaler_model_path))
    model=models[model_name]
    model.fit(train_x,train_y)
    joblib.dump(model, pre_model_path + model_name + '.pkl')
    print(model_name + '模型训练完成')
    return model

if __name__ == '__main__':
#训练模型
    for model_name in models.keys():
        model=mltrain_model(train_data_path,model_name)
        print('-----------------------------------')

