import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age (yrs)':28,'BMI':19,'Cycle(R/I)':0,'Cycle length(days)':5,'Blood Group':15,'Pregnant(Y/N)':0,'No. of aborptions':0,
          'Weight gain(Y/N)':0, 'hair growth(Y/N)':0, 'Skin darkening (Y/N)':0,
          'Hair loss(Y/N)':0, 'Pimples(Y/N)':0, 'Fast food (Y/N)':1,'Reg.Exercise(Y/N)':0})

print(r.json())