def early_stoping(min_delta = 0.001, patience=20):
    from tensorflow.keras import callbacks

    early_stopping = callbacks.EarlyStopping(
        min_delta=min_delta, # minimium amount of change to count as an improvement
        patience=patience, # how many epochs to wait before stopping
        restore_best_weights=True,
)

def save_model(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{model_name}.h5")
    print("Saved model to disk")

def load_model(model_name):
    from tensorflow.keras.models import  model_from_json
    # load json and create model
    json_file = open(f'{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f"{model_name}.h5")
    print("Loaded model from disk")
