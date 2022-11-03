from Model.utils import FlowerDetection
from flask import Flask,jsonify,render_template,request
import config


app = Flask(__name__)

@app.route("/")
def hello_flask():
    print("We are in Flask API")
    return render_template("index.html")

@app.route("/Iris_Flower",methods=['POST'])
def disease():
    
    SepalLengthCm=float(request.form.get('SepalLengthCm'))
    SepalWidthCm=float(request.form.get('SepalWidthCm'))
    PetalLengthCm=float(request.form.get('PetalLengthCm'))
    PetalWidthCm=float(request.form.get('PetalWidthCm'))
    
    Obj = FlowerDetection(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    result= Obj.get_flower_classification()
    return render_template("index.html",prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=config.PORT_NUMBER,debug=True)