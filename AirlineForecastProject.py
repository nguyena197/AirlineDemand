# Team 7: Anh Nguyen, Miguel Gonzalez, Yunjung Ham
# 12/6/2019

import pandas as pd
import numpy as np


# Formating date columns
def datetime(file,columns):

    df = pd.read_csv(file)
    # Convert to datetime
    for c in columns:
        df[c] = pd.to_datetime(df[c])

    # Add days prior and day of the week
    df['Days_Prior'] = (df[columns[0]] - df[columns[1]]).dt.days
    df['Dep_DoW'] = df[columns[0]].dt.weekday_name

    return df


# Clean and add columns to training data
def trainD(trainingData):

    columns = ['departure_date', 'booking_date']

    # add days prior and DoW for trainingData
    trainingData = datetime(trainingData, columns)

    # Add final booking amount column to trainingData
    finalbook = pd.DataFrame(trainingData[['departure_date','cum_bookings']][trainingData['Days_Prior'] == 0])
    finalbook = finalbook.rename(columns = {'cum_bookings': 'final_bookings'})

    trainD = trainingData.merge(finalbook, left_on = 'departure_date', right_on = 'departure_date')

    # Calculate remain bookings and percentage booking
    trainD['remain'] = trainD['final_bookings'] - trainD['cum_bookings']
    trainD['percent'] = trainD['cum_bookings']/trainD['final_bookings']

    return trainD

# transform and add columns to validation data
def valiD(validationData):

    columns = ['departure_date', 'booking_date']
    # Load and add priordate and DoW for validationData
    validationData = datetime(validationData, columns)
    # Remove day prior = 0 in validationData
    valiD = validationData[validationData['Days_Prior'] !=0]

    # sort booking date ascending order so rolling median can work properly
    return valiD.sort_values(['departure_date','booking_date'], ascending = [True,True])


# create dataframe to looko up value by days prior and day of week from training data
def lookuptable(trainingData):

    train = trainD(trainingData)
    lookuptable = pd.DataFrame(train.groupby(['Days_Prior','Dep_DoW'])
                    .agg({'remain':'median', 'percent':'mean'})).reset_index()

    return lookuptable

# this function will be use later to calculate the rolling median forecast for each departure date
# using 21 days period because we see a spike in booking on the last 7 days before departure date
def rollMed(df):
    return df.rolling(window = 21, min_periods =1, center= True).median()


def AddModel(trainingData,validationData):

    val = valiD(validationData)
    lookup = lookuptable(trainingData)
    newval = val.merge(lookup, on=['Days_Prior','Dep_DoW'])
    newval['final_forecast'] = newval['cum_bookings'] + newval['remain']

    return newval


def MultiModel(trainingData,validationData):

    val = valiD(validationData)
    lookup = lookuptable(trainingData)
    newval = val.merge(lookup, on=['Days_Prior','Dep_DoW'])
    newval['final_forecast'] = newval['cum_bookings']/newval['percent']

    return newval

# Extending our AddModel to calculate forecast value based on rolling median of 'final_forecast' column
def ExtAddModel(trainingData,validationData):

    data = AddModel(trainingData,validationData)
    data['final_forecast'] = data.groupby(['departure_date'])['final_forecast'].apply(rollMed)

    return data

# Extending our MultiAdd to calculate forecast value based on rolling median of 'final_forecast' column
def ExtMultiModel(trainingData,validationData):

    data = MultiModel(trainingData,validationData)
    data['final_forecast'] = data.groupby(['departure_date'])['final_forecast'].apply(rollMed)

    return data


def MASE(validationData):
    return round((sum(abs(validationData['final_forecast'] - validationData['final_demand']))/sum(abs(validationData['naive_forecast']-validationData['final_demand'])))*100,2)


def AirlineForecast(trainingData,validationData):

    # List of developed models
    models = [AddModel, MultiModel,ExtMultiModel,ExtAddModel]
    thedict = {}
    for model in models:
        thedict[model] = MASE(model(trainingData,validationData))

    mod = min(thedict,key = thedict.get)

    return "\nThe lowest MASE is: " + str(min(thedict.values())) + '%\n\n'+ str(mod(trainingData,validationData)[['departure_date','booking_date','final_forecast']]) + "\nfrom " + str(mod)

def main():
    print(AirlineForecast('airline_booking_trainingData.csv','airline_booking_validationData.csv'))
main()
