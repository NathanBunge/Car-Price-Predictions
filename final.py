#Nathan Bugne
#12/12/2022
#Cpts 315 Final Project

#-----import libraries-----
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import copy


#ML libraries
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


def process_data():
    """
    Reads in train and test data from files, processes it, and returns it as two dataframes.
    
    Returns:
        train_data (pandas.DataFrame): Processed training data.
        test_data (pandas.DataFrame): Processed test data.
    """
    # Read in data from files
    train_data = pd.read_csv('Data/train-data.csv')
    test_data = pd.read_csv('Data/test-data.csv')

    # Drop unnecessary columns
    train_data = train_data.drop(['New_Price', 'Unnamed: 0'], axis=1)
    test_data = test_data.drop(['New_Price', 'Unnamed: 0'], axis=1)

    # Drop rows with missing values
    train_data = train_data.dropna(how='any')
    test_data = test_data.dropna(how='any')

    # Shorten names of cars
    train_data['Cars'] = train_data['Name'].str.split(" ").str[0] + ' ' + train_data['Name'].str.split(" ").str[1]
    test_data['Cars'] = test_data['Name'].str.split(" ").str[0] + ' ' + test_data['Name'].str.split(" ").str[1]

    # Drop remaining missing rows
    test_data.drop(test_data[test_data['Cars'].isin(['Toyota Land', 'Hindustan Motors', 'Fiat Abarth', 
                                                     'Nissan 370Z', 'Isuzu MU', 'Bentley Flying', 'OpelCorsa 1.4Gsi'])].index, inplace=True)

    # Clean up columns
    train_data['Mileage'] = train_data['Mileage'].str.replace(' kmpl', '')
    train_data['Mileage'] = train_data['Mileage'].str.replace(' km/kg', '')
    train_data['Engine'] = train_data['Engine'].str.replace(' CC', '')
    train_data['Power'] = train_data['Power'].str.replace('null bhp', '112')
    train_data['Power'] = train_data['Power'].str.replace(' bhp', '')

    test_data['Mileage'] = test_data['Mileage'].str.replace(' kmpl', '')
    test_data['Mileage'] = test_data['Mileage'].str.replace(' km/kg', '')
    test_data['Engine'] = test_data['Engine'].str.replace(' CC', '')
    test_data['Power'] = test_data['Power'].str.replace('null bhp', '112')
    test_data['Power'] = test_data['Power'].str.replace(' bhp', '')

    # Convert to numerical values
    train_data['Mileage'] = train_data['Mileage'].astype(float)
    train_data['Engine'] = train_data['Engine'].astype(float)
    train_data['Power'] = train_data['Power'].astype(float)

    test_data['Mileage'] = test_data['Mileage'].astype(float)
    test_data['Engine'] = test_data['Engine'].astype(float)
    train_data['Power'] = train_data['Power'].astype(float)

    test_data['Mileage'] = test_data['Mileage'].astype(float)
    test_data['Engine'] = test_data['Engine'].astype(float)
    test_data['Power'] = test_data['Power'].astype(float)
    
    return train_data, test_data


def make_dataframes(df_train, df_test, choice):
    f_train = ['Cars', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats','Price']
    
    # features (one without price for test data)
    feature = ['Cars', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats','Price']
    feature_no_price = ['Cars', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']

    print("\n\nTest data:\n", df_test)
    
    # create dataframes from data
    df_train = pd.DataFrame(df_train, columns=feature)
    df_test = pd.DataFrame(df_test, columns=feature_no_price)
    
    print("\n\nDataframe test data:\n", df_test)
    
    # make copy
    df_train_copy = copy.deepcopy(df_train)
    df_test_copy = copy.deepcopy(df_test)
    
    # drop irrelevant features
    cols = np.array(df_train.columns[df_train.dtypes != object])
    for i in df_train_copy.columns:
        if i not in cols:
            df_train_copy[i] = df_train_copy[i].map(str)
            df_test_copy[i] = df_test_copy[i].map(str)
    df_train_copy.drop(columns=cols, inplace=True)
    df_test_copy.drop(columns=np.delete(cols, len(cols) - 1), inplace=True)

    # make dictionary so we can encode labels
    cols = np.array(df_train.columns[df_train.dtypes != object])
    d = defaultdict(LabelEncoder)
    
    print("Categories for dictionary:\n", df_test_copy)

    # encode categorical labels with the dictionary
    df_train_copy = df_train_copy.apply(lambda x: d[x.name].fit_transform(x))
    df_test_copy = df_test_copy.apply(lambda x: d[x.name].transform(x))
    df_train_copy[cols] = df_train[cols]
    df_test_copy[np.delete(cols, len(cols) - 1)] = df_test[np.delete(cols, len(cols) - 1)]
    
    print("After encoding:\n", df_test_copy)

    # if getting training
    if choice == 0:
        data2 = df_train_copy[f_train]
        X = data2.drop(columns=['Price']).values
        y0 = data2['Price'].values
        lab_enc = preprocessing.LabelEncoder()
        y = lab_enc.fit_transform(y0)
        return X, y             
        
    # if just making a prediction
    elif choice == 1:
        data2 = df_test_copy[feature_no_price]
        return data2
    
#makes a GBR model with given feature and labled data. data needs to be in a Dataframe and already pre-procressed
def makeModel(X, y):
    
    model = GradientBoostingRegressor(random_state=21, n_estimators=5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 25)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_total = model.predict(X)
    
    #evaluate the model created
    print("\t\tError Table")
    print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Accuracy on Traing set   : ', model.score(X_train,y_train))
    print('Accuracy on Testing set  : ', model.score(X_test,y_test))
    
    return model
    
#makes a prediction for a price given predictor model and test feature data
def makePrediction(model, x):

    print("predicting: ", x)
    y_pred = model.predict(x)
    
    #convert to usd from Rubies
    y_pred = (y_pred*100000)/82./64
    return y_pred


#-----create model as a global variable-----

#get train and test data
Data, test = process_data()

#turn training/testing into X featrues and y lables
X, y = make_dataframes(Data,test, 0)
    #   ['Cars', 'Location', 'Year', 'KM', 'Fuel', 'Trans', 'Owner_#  MPG, 'Engine CC', 'HP', 'Seats','Price']

#create model from training data
model = makeModel(X, y)

#make prediction 
y_pred = makePrediction(model, X)

testing = Data['Cars'].drop_duplicates()

print(testing)
    





#-----create window for UI----
from tkinter import *
from tkinter.ttk import Combobox
class MyWindow:
    def __init__(self, win):
    
        #create lables for window
        self.lbl1=Label(win, text='Car name')
        self.lbl2=Label(win, text='Location')
        self.lbl3=Label(win, text='Year')
        self.lbl4=Label(win, text='Odometer (KM)')
        self.lbl5=Label(win, text='Fuel Type')
        self.lbl6=Label(win, text='Transmission')
        self.lbl7=Label(win, text='# of Owners')
        self.lbl8=Label(win, text='MPG')
        self.lbl9=Label(win, text='Engine')
        self.lbl10=Label(win, text='Horse Power')
        self.lbl11=Label(win, text='Seats')
        self.lbl12=Label(win, text='Predicted Price')
        
        #freate combo and input boxs for user input
        self.car=Combobox(window, values=(sorted(Data['Cars'].drop_duplicates().tolist())))
        self.location=Combobox(window, values=(sorted(Data['Location'].drop_duplicates().tolist())))
        self.gas=Combobox(window, values=(sorted(Data['Fuel_Type'].drop_duplicates().tolist())))
        self.trans=Combobox(window, values=(sorted(Data['Transmission'].drop_duplicates().tolist())))
        self.owner=Combobox(window, values=(sorted(Data['Owner_Type'].drop_duplicates().tolist())))
        self.year=Entry()
        self.km=Entry()
        self.mpg=Entry()
        self.engine=Entry()
        self.hp=Entry()
        self.seats=Combobox(window, values=(sorted(Data['Seats'].drop_duplicates().tolist())))
        self.prediction=Entry()
                
        #place lables in window
        self.lbl1.place(x=100, y=50)
        self.lbl2.place(x=100, y=100)
        self.lbl3.place(x=100, y=150)
        self.lbl4.place(x=100, y=200)
        self.lbl5.place(x=100, y=250)
        self.lbl6.place(x=100, y=300)
        self.lbl7.place(x=100, y=350)
        self.lbl8.place(x=100, y=400)
        self.lbl9.place(x=100, y=450)
        self.lbl10.place(x=100, y=500)
        self.lbl11.place(x=100, y=550)
        self.lbl12.place(x=100, y=700)
        
        #place input and combo boxes in window
        self.car.place(x=250, y=50)
        self.location.place(x=250, y=100)
        self.year.place(x=250, y=150)
        self.km.place(x=250, y=200)
        self.gas.place(x=250, y=250)
        self.trans.place(x=250, y=300)
        self.owner.place(x=250, y=350)
        self.mpg.place(x=250, y=400)
        self.engine.place(x=250, y=450)
        self.hp.place(x=250, y=500)
        self.seats.place(x=250, y=550)
        self.prediction.place(x=250, y=700)
        
        #create button to predict
        self.b1=Button(win, text='Predict', command=self.predict)
        self.b1.place(x=100, y=600)

    #function that is called when predict button is clicked. Gets preice predidction
    def predict(self):
        
        #parse user input
        userInput = [[self.car.get(), self.location.get(), int(self.year.get()), int(self.km.get()), self.gas.get(), self.trans.get(), self.owner.get(), float(self.mpg.get()),  
        float(self.engine.get()), float(self.hp.get()), float(self.seats.get())]]
        
        print("got: ", userInput)
        
        #make dataframes from input
        user_dataframe = make_dataframes(Data, userInput,1)
        
        #make prediction
        y_prediction = makePrediction(model, user_dataframe)

        print("answer price: ",y_prediction[0])
        
        #add prediction to window
        self.prediction.delete(0, 'end')
        self.prediction.insert(END, ("$" + str(round(y_prediction[0] ,2))))
        
        
        
#init window and run till closed
window=Tk()
mywin=MyWindow(window)
window.title('Hello Python')
window.geometry("500x800+10+10")
window.mainloop()





