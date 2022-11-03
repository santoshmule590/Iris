import numpy as np
import pandas as pd
import json
import pickle

import config


class FlowerDetection():
    def __init__(self,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
        self.SepalLengthCm = SepalLengthCm
        self.SepalWidthCm = SepalWidthCm
        self.PetalLengthCm =PetalLengthCm
        self.PetalWidthCm = PetalWidthCm
        

    def load_model(self):
        with open (config.JSON_FILE_PATH,"r") as f:
            self.json_dict = json.load(f)
        with open (config.MODEL_FILE_PATH,"rb") as f:
            self.model = pickle.load(f)



        # with open (r"C:\Users\ADMIN\Desktop\Classification Algorithm\37_Santosh_Mule_Logistic_iris_dataset -\Model\Iris.json","r") as f:
        #     self.json_dict=json.load(f)
        
        # with open (r"C:\Users\ADMIN\Desktop\Classification Algorithm\37_Santosh_Mule_Logistic_iris_dataset -\Model\Iris.pkl","rb") as f:
        #     self.model=pickle.load(f)

    def get_flower_classification(self):
        self.load_model()
        array = np.zeros(len(self.json_dict["column"]))

        array[0]=self.SepalLengthCm
        array[1]=self.SepalWidthCm
        array[2]=self.PetalLengthCm
        array[3]=self.PetalWidthCm
        
        print("Array is ::\n",array)


        result = self.model.predict([array])[0]

        return result

if __name__ == "__main__":
    SepalLengthCm = 6.5
    SepalWidthCm = 2.8
    PetalLengthCm = 8.9
    PetalWidthCm = 2.3
   
    Obj = FlowerDetection(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    result= Obj.get_flower_classification()
    print(result)



