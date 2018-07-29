# -*- coding: utf-8 -*-

# Run in Python 3.6
# Author: Gu Quan <https://github.com/ZJUguquan>
# All rights reserved.

import numpy as np
import pandas as pd
import os
import lightgbm as lgb


# -----------------------------------
# 1. 批量读取数据
def readRoundFiles(round_dir):
    file_list = os.listdir(round_dir)
    def readDayFile(filename):
        data = pd.read_csv(round_dir + filename)
        data["day"] = int(filename.split(".")[0])
        return data
    data = pd.concat([readDayFile(f) for f in file_list]).reset_index(drop=True)
    data = data.sort_values(["code","day"]).reset_index(drop=True)
    data["yesterday"] = data.groupby("code")["close"].shift(1)
    data["ex_right"] = (data.close / data.yesterday - 1) <= -0.11
    data["ex_prod"] = np.where(data.ex_right == True, data.yesterday / data.close, 1)
    data.ex_prod = data.groupby("code")["ex_prod"].cumprod()
    data["close_ori"] = data.close * data.ex_prod
    return data


# -----------------------------------
# 2. 根据训练集的起止时间生成rawX
def getRawX(data, start_day, end_day, min_days_cnt, target_day=None):
    '''
    target_day: 后面的函数会用到，对每只股票，根据target_day和end_day的收盘价计算收益率
    '''
    stockX1 = data.code[data.day == end_day].unique()
    if target_day:
        stockX2 = data.code[data.day == target_day].unique()
        stockX = np.intersect1d(stockX1, stockX2)
    else:
        stockX = stockX1
    dataX = data[(data.code.isin(stockX)) & (data.day >= start_day) & (data.day <= end_day)]
    dataX.loc[:, "day_cnt"] = dataX.groupby("code")["day"].transform("count")
    dataX = dataX[dataX.day_cnt >= min_days_cnt]
    stocks = pd.DataFrame({"key":1, "code": dataX.code.unique()})
    days = pd.DataFrame({"key":1, "day": np.array(range(start_day, end_day+1))})
    stock_day = pd.merge(stocks, days, how="outer").drop(columns="key")
    dataX = pd.merge(dataX, stock_day, how="right", on=["code","day"])
    dataX = dataX[["code","day","close","close_ori","yesterday","ex_right","day_cnt"] + 
                  ["f"+str(i) for i in range(1, 87+1)]]
    dataX = dataX.sort_values(["code","day"]).reset_index(drop=True)
    dataX = dataX.groupby("code").apply(lambda g: g.interpolate(method='linear'))
    dataX = dataX.fillna(method="backfill")
    def replaceInf(col):
        repl = col[col!=np.inf].max()
        col[col==np.inf] = repl
        return col
    inf_cols = (dataX==np.inf).sum()
    inf_cols = list(inf_cols[inf_cols>0].index)
    inf_rows = False
    for col in inf_cols:
        inf_rows = (dataX[col]==np.inf) | inf_rows
    inf_code = dataX.code[inf_rows].unique()
    dataX.loc[dataX.code.isin(inf_code), inf_cols] = \
        dataX.loc[dataX.code.isin(inf_code), ["code"] + inf_cols].\
        groupby("code").apply(lambda g: g[inf_cols].apply(replaceInf))
    return dataX


# 每一个round的数据生成两份训练集，增加样本量
def getTrainData(round_dir):
    data = readRoundFiles(round_dir)
    rawX1 = getRawX(data, start_day=1, end_day=244, min_days_cnt=135, target_day=366)
    rawX2 = getRawX(data, start_day=123, end_day=366, min_days_cnt=135, target_day=488)
    feas_X1 = quickFeaX(rawX1, end_day=244)
    feas_X2 = quickFeaX(rawX2, end_day=366)
    y1_df = getY(data, rawX1.code.unique(), X_end_day=244, target_day=366)
    y2_df = getY(data, rawX1.code.unique(), X_end_day=366, target_day=488)
    train1 = y1_df.merge(feas_X1, on="code").rename(columns={"rate": "y"})
    train2 = y2_df.merge(feas_X2, on="code").rename(columns={"rate": "y"})
    return train1, train2
    

# -----------------------------------
# 3. 特征工程
def joinFeas(*args):
    assert len(args)>=2
    features = args[0]
    for fea in args[1:]:
        features = features.merge(fea, how="left", on="code")
    return features

# 以下是每类特征的构造函数
def closeMean(rawX, end_day, last_days_num):
    rawX = rawX.loc[(rawX.day >= end_day-last_days_num+1) & (rawX.day <= end_day), ["code","day","close_ori"]]
    mean = rawX.groupby("code")["close_ori"].mean().to_frame()
    mean.loc[:, "code"] = mean.index.values
    mean.columns.values[0] = "mean_close_last" + str(last_days_num)
    res = mean.iloc[:,[1,0]].reset_index(drop=True)
    return res

def RSV(rawX, end_day, last_days_num, days_in_compute=244):
    rawX = rawX[["code","day","close_ori"]]
    fea = pd.DataFrame({"code": rawX.code.unique()})
    fea.loc[:, "min_close"] = rawX.loc[(rawX.day >= end_day-days_in_compute+1) & (rawX.day <= end_day), ].\
        groupby("code")["close_ori"].min().values
    fea.loc[:, "max_close"] = rawX.loc[(rawX.day >= end_day-days_in_compute+1) & (rawX.day <= end_day), ].\
        groupby("code")["close_ori"].max().values
    fea.loc[:, "mean_close"] = rawX.loc[(rawX.day >= end_day-last_days_num+1) & (rawX.day <= end_day), ].\
        groupby("code")["close_ori"].mean().values
    fea.loc[:, "rsv_last"+str(last_days_num)+"_in"+str(days_in_compute)] = \
        (fea.mean_close - fea.min_close) / (fea.max_close - fea.min_close)
    return fea[["code", "rsv_last"+str(last_days_num)+"_in"+str(days_in_compute)]]

def rateStat(rawX, end_day, last_days_num, method="mean", days_before=122):
    rawX = rawX[["code","day","close_ori"]]
    rawX.loc[:, "close_bf"] = rawX.groupby("code")["close_ori"].shift(days_before)
    rawX.loc[:, "rate"] = rawX.close_ori / rawX.close_bf - 1
    if method == "mean":
        res = rawX.loc[(rawX.day >= end_day-last_days_num+1) & (rawX.day <= end_day), :].\
            groupby("code")["rate"].mean().to_frame()
    if method == "std":
        res = rawX.loc[(rawX.day >= end_day-last_days_num+1) & (rawX.day <= end_day), :].\
            groupby("code")["rate"].std().to_frame()
    res.loc[:, "code"] = res.index.values
    res.columns.values[0] = method + "_rate_last" + str(last_days_num)
    res = res.iloc[:,[1,0]].reset_index(drop=True)
    return res

def rateRise(rawX, end_day, last_days_num):
    rawX = rawX[["code","day","close_ori"]]
    start = rawX[rawX.day == end_day-last_days_num+1].rename(columns={"close_ori":"start"}).reset_index(drop=True)
    end = rawX[rawX.day == end_day].drop(columns="day").rename(columns={"close_ori":"end"}).reset_index(drop=True)
    res = start.merge(end, on="code")
    res.loc[:, "rise_last"+str(last_days_num)] = res.end / res.start - 1
    res = res[["code", "rise_last"+str(last_days_num)]]
    return res

def RSI(rawX, end_day, last_days_num):
    rawX = rawX[["code","day","close_ori"]]
    rawX.loc[:, "ystday"] = rawX.groupby("code")["close_ori"].shift(1)
    rawX = rawX[(rawX.day >= end_day-last_days_num+1) & (rawX.day <= end_day)]
    rawX.loc[:, "is_up"] = np.where(rawX.close_ori - rawX.ystday > 0, 1, 0)
    rawX.loc[:, "is_down"] = np.where(rawX.close_ori - rawX.ystday < 0, 1, 0)
    res = rawX.groupby("code")[["is_up","is_down"]].sum()
    res.loc[:, "code"] = res.index.values
    res.loc[:, "rsi_last"+str(last_days_num)] = res.is_up / (res.is_down + 1e-2)
    res = res[["code", "rsi_last"+str(last_days_num)]].reset_index(drop=True)
    return res

def exrightStat(rawX, end_day, days_in_compute=244):
    rawX = rawX[["code","day","close","yesterday","ex_right"]]
    rawX["ex_prod"] = np.where(rawX.ex_right == True, rawX.yesterday / rawX.close, 1)
    rawX = rawX[(rawX.day >= end_day-days_in_compute+1) & (rawX.day <= end_day)]
    fea = pd.DataFrame({"code": rawX.code.unique()})
    fea.loc[:, "cnt_ex_in"+str(days_in_compute)] = rawX.groupby("code")["ex_right"].sum().values
    fea.loc[:, "mean_exprod_in"+str(days_in_compute)] = rawX.groupby("code")["ex_prod"].mean().values
    return fea

def feaFi(rawX, end_day, last_days_num=244):
    rawX = rawX.loc[(rawX.day >= end_day-last_days_num+1) & (rawX.day <= end_day), 
                    ["code","day"] + ["f"+str(i) for i in range(1,87+1)]]
    res = rawX.groupby("code").mean().drop(columns="day")
    res.loc[:, "code"] = res.index.values
    return res.reset_index(drop=True)

# 整合全部特征
def quickFeaX(rawXi, end_day):
    fea_Xi_closeMean = joinFeas(closeMean(rawXi, end_day, 5), closeMean(rawXi, end_day, 20), closeMean(rawXi, end_day, 50),
                                closeMean(rawXi, end_day, 80), closeMean(rawXi, end_day, 122), closeMean(rawXi, end_day, 244))
    fea_Xi_RSV = joinFeas(RSV(rawXi, end_day, 1), RSV(rawXi, end_day, 5), RSV(rawXi, end_day, 10))
    fea_Xi_rateStat = joinFeas(rateStat(rawXi, end_day, 10, "mean"), rateStat(rawXi, end_day, 30, "mean"),
                               rateStat(rawXi, end_day, 60, "mean"), rateStat(rawXi, end_day, 122, "mean"),
                               rateStat(rawXi, end_day, 10, "std"), rateStat(rawXi, end_day, 30, "std"),
                               rateStat(rawXi, end_day, 60, "std"), rateStat(rawXi, end_day, 122, "std"))
    fea_Xi_rateRise = joinFeas(rateRise(rawXi, end_day, 20), rateRise(rawXi, end_day, 50), rateRise(rawXi, end_day, 80),
                               rateRise(rawXi, end_day, 122), rateRise(rawXi, end_day, 244))
    fea_Xi_RSI = joinFeas(RSI(rawXi, end_day, 20), RSI(rawXi, end_day, 50), RSI(rawXi, end_day, 80),
                          RSI(rawXi, end_day, 122), RSI(rawXi, end_day, 244))
    fea_Xi_ex = exrightStat(rawXi, end_day)
    fea_Xi_fi = feaFi(rawXi, end_day)
    feas_Xi = joinFeas(fea_Xi_closeMean, fea_Xi_RSV, fea_Xi_rateStat, fea_Xi_rateRise, fea_Xi_RSI, fea_Xi_ex, fea_Xi_fi)
    return feas_Xi

    
# -----------------------------------
# 5. 计算标签
def getY(data, codes, X_end_day, target_day):
    dataY = data.loc[data.code.isin(codes), ["code","day","close_ori"]]
    dataY = dataY[dataY.day.isin([X_end_day, target_day])].sort_values(["code","day"]).reset_index(drop=True)
    dataY["close_tg"] = dataY.groupby("code")["close_ori"].shift(-1)
    dataY["rate"] = dataY.close_tg / dataY.close_ori - 1
    res = dataY.loc[dataY.day==X_end_day, ["code","rate"]].reset_index(drop=True)
    return res


# -----------------------------------
# 6. 模型线下评估
def myEval(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    min20 = y_true[np.argsort(y_true)[:20]]
    max20 = y_true[np.argsort(y_true)[-20:][::-1]]
    pred20 = y_true[np.argsort(y_pred)[-20:][::-1]]
    print(min20.mean(), max20.mean(), pred20.mean())
    res = (pred20.mean() - min20.mean()) / (max20.mean() - min20.mean()) 
    return res

    
def myLgbEval(y_pred, train_data):
    y_true = train_data.get_label()
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    min20 = y_true[np.argsort(y_true)[:20]]
    max20 = y_true[np.argsort(y_true)[-20:][::-1]]
    pred20 = y_true[np.argsort(y_pred)[-20:][::-1]]
    res = (pred20.mean() - min20.mean()) / (max20.mean() - min20.mean()) 
    return 'eval', res, True

def lgbImp(model):
    assert "train1_rB" in globals()
    res = pd.DataFrame({"feature": train1_rB.columns[2:], "imp": model.feature_importance("split")})
    res = res.sort_values("imp", ascending=False).reset_index(drop=True)
    return res


# 基本上代码运行还是很快的
if __name__ == '__main__':
    
    # Make sure the current directory is the parent directory of "data" !!!
    # os.chdir(r"xx/xx/xx/凤凰金融")
    train1_r4, train2_r4 = getTrainData(r"./data/round4/")
    train1_rB, train2_rB = getTrainData(r"./data/roundB/")
    
    train_final = pd.concat([train1_r4, train1_rB, train2_r4, train2_rB]).reset_index(drop=True)
    weight_final = np.array([1]*len(train1_r4) + [1]*len(train1_rB) + [2]*len(train2_r4) + [2]*len(train2_rB))
    lgb_dat = lgb.Dataset(train_final.iloc[:,2:], train_final.y, weight=weight_final)
    
    lgb_params = {
        'boosting_type': 'gbdt',
        'application': 'regression',
        'metric': 'mae',
        'num_leaves': 61,
        'max_depth': 12,
        'learning_rate': 0.01,
        'verbose': 1,
        'seed': 2018,
    }
    # LightGBM建模
    lgb_model = lgb.train(lgb_params, lgb_dat, num_boost_round=100, verbose_eval=10, 
                          valid_sets=lgb_dat, early_stopping_rounds=None, feval=myLgbEval)
    
    data_r4 = readRoundFiles("./data/round4/")
    rawX3_r4 = getRawX(data_r4, start_day=245, end_day=488, min_days_cnt=135, target_day=None)
    test_r4 = quickFeaX(rawX3_r4, end_day=488)
    
    lgb_pred = lgb_model.predict(test_r4.iloc[:,1:])
    lgb_pred = pd.DataFrame({"code": test_r4.code, "rate": lgb_pred}).sort_values("rate", ascending=False)
    submit = lgb_pred.code[:20] # To Submit for Round 4, maybe a little difference because of the random "seed".
    print("--------- Submit Results ---------")
    print(submit)
































    