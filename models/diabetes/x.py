import pickle
with open('models/diabetes/diabetes.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/diabetes/diabetes.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=2)
