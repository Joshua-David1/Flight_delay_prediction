from flask import Flask, render_template, request, url_for,jsonify,redirect
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

app = Flask(__name__)

# print(data.columns)

class FlightDataSetTraining(object):
    def __init__(self, file_name) -> None:
        self.clf = None
        self.otcome = 0
        self.flight_data = pd.read_csv(file_name)[:1000]
        self.initial_column_drop()
        self.preprocess_output()
        self.extra_drop()
        self.replace_NaN_values()
        self.train_model()

    def initial_column_drop(self):
        self.flight_data.drop(['YEAR','FLIGHT_NUMBER','AIRLINE','DISTANCE','TAIL_NUMBER','TAXI_OUT',
                                              'SCHEDULED_TIME','DEPARTURE_TIME','WHEELS_OFF','ELAPSED_TIME',
                                              'AIR_TIME','WHEELS_ON','DAY_OF_WEEK','TAXI_IN','CANCELLATION_REASON'],
                                             axis=1,inplace=True)
    
    def replace_NaN_values(self):
        self.flight_data.fillna(self.flight_data.mean(),inplace=True)
    
    def extra_drop(self):
        self.flight_data.drop(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'],axis=1,inplace=True)

    def change_dtype(self, column):
        pass

    def preprocess_output(self):
        result = []
        for row in self.flight_data['ARRIVAL_DELAY']:
            if row > 15:
                result.append(1)
            else:
                result.append(0)
        self.flight_data['result'] = result
    
    def train_model(self):
        data = self.flight_data.values
        X = data[:,:-1]
        Y = data[:,-1]

        x_train, x_test, y_train,y_test = train_test_split(X,Y,train_size=0.9)
        self.clf = DecisionTreeClassifier()
        self.clf.fit(x_train,y_train)
    
    def predict_outcome(self, data):
        pred_prob = self.clf.predict_proba(data)
        return pred_prob[0,1]

fdst = FlightDataSetTraining("flights.csv")

@app.route("/",methods=["GET","POST"])
def home():
    try:
        if request.method == "POST":
            month = int(request.form.get("month"))
            date = int(request.form.get("date"))
            weather = request.form.get("weather")
            if weather == "Sunny":
                weather = 1.758434
            elif weather == "Rainy":
                weather = 12.00
            elif weather == "Stormy":
                weather = 213.00
            elif weather == "Cloudy":
                weather = 28.00
            else:
                weather = 22.00
            diverted = float(request.form.get("diverted"))
            cancelled = float(request.form.get("cancelled"))
            scheduled_departure = int("".join(request.form.get("scheduled_departure").split(":")))
            scheduled_arrival = int("".join(request.form.get("scheduled_arrival").split(":")))
            departure_delay = int(request.form.get("departure_delay"))
            data = {
                "month":month,
                "date":date,
                "weather":weather,
                "diverted":diverted,
                "cacelled":cancelled,
                "scheduled_departure":scheduled_departure,
                "scheduled_arrival":scheduled_arrival,
                "departure_delay":departure_delay
            }
            given_data = np.array([month, date, scheduled_departure,departure_delay,scheduled_arrival,diverted,cancelled,0,0,0,0,weather]).reshape(1,-1)
            outcome = fdst.predict_outcome(given_data)
            data['outcome'] = int(outcome)
            fdst.otcome = int(outcome)
            print(fdst.otcome)
            return redirect(url_for('result_page'))
    except Exception as e:
        print(e)
        return jsonify({"Error":"[!]Enter Data Correctly to predict the outcome!"}) 
    return render_template("index.html")


@app.route("/result",methods = ["GET"])
def result_page():
    return render_template("result.html",outcome=fdst.otcome)

if __name__ == "__main__":
    app.run(debug=True,port=8081)