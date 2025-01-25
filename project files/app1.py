import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
longitudes = {
    -120.610829: 0,
    -80.603956: 1,
    -80.577366: 2
}
latitudes = {
    28.561857: 0,
    28.608058: 1,
    34.632093: 2
}
with open("project.pkl",'rb') as m:
  model=pickle.load(m)
with open("scaler.pkl",'rb') as s:
  scaler=pickle.load(s)
with open("orbit_le.pkl",'rb') as o:
  orbit_le=pickle.load(o)
with open("launchsite_le.pkl",'rb') as ls:
  launchsite_le=pickle.load(ls)
#create flask app
flask_app=Flask(__name__)
@flask_app.route("/")
def spacex():
  return render_template('spacex.html')
@flask_app.route("/about")
def about():
  return render_template('home.html')
@flask_app.route("/details")
def details():
  return render_template('details.html')
@flask_app.route("/contact")
def contact():
  return render_template('contact.html')
@flask_app.route("/predict",methods=['POST',"GET"])
def predict():
  return render_template('predict1.html')
@flask_app.route('/submit', methods=['POST', 'GET'])
def submit():
    payload_mass = float(request.form['PayloadMass'])
    orbit = orbit_le.transform([str(request.form.get('orbit'))])[0]
    launch_site = launchsite_le.transform([str(request.form.get('launch_site'))])[0]
    longitude = longitudes[float(request.form.get('longitude'))]
    latitude = latitudes[float(request.form.get('latitude'))]
    grindfins=0 if request.form.get("Grindfins") =="False" else 1
    legs=0 if request.form.get("Legs") =="False" else 1
    core_block_version = float(request.form.get('core_block_version'))
    flights_with_core = float(request.form['Flights_With_That_Core'])
    core_reused_count = float(request.form['Core_Reused_Count'])
    input_data = [payload_mass, orbit, launch_site, longitude, latitude, grindfins, legs, core_block_version, flights_with_core, core_reused_count]
    arr = np.array(input_data)
    print(arr)
    print(input_data)
    scaled_arr = scaler.transform([arr])
    print(scaled_arr)
    prediction = model.predict(scaled_arr)
    if prediction[0] == 1:
      return render_template('success.html') 
    else:
      return render_template('unsuccessful.html')
   
if __name__=="__main__":
  flask_app.run(debug=True)