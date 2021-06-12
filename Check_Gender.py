import pickle
import string
import numpy as np

def oneHot25(data):
    n_letters = len(string.ascii_lowercase)+len(string.ascii_uppercase)
    vector = np.zeros((len(data), 25, n_letters))
    for j in range(len(data)):
        for i in range(len(data[j])):
            if (data[j][i].isupper()):
                vector[j][i][ord(data[j][i])-65] = 1
            else:
                vector[j][i][26 + ord(data[j][i])-97] = 1
    return(vector)

pkl_filename = "randaam_forest.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
pkl_filename = "scaler.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_sc_X = pickle.load(file)
    
def find_gender(a):
    target_names = ['boy', 'girl']
    a_new = oneHot25([a])
    nsamples,nx,ny = a_new.shape
    a_new = a_new.reshape((nsamples,nx*ny))
    input_name = pickle_sc_X.transform(a_new)#a_new.reshape((nsamples,nx*ny)))
    Y_Pred = pickle_model.predict(input_name)
    return(target_names[Y_Pred[0]])

name = input("Enter the name: ")
if not name.isalpha():
    raise Exception("Improper name entered: Only first name with alphabets is allowed")
name = name.capitalize()
print('{} is a {} name'.format(name,find_gender(name)))
