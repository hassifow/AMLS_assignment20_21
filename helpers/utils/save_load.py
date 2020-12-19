
def save(checkpoint_path, filename, model):
    if model is None:
        raise Exception("You have to build the model first.")

    print("Saving model...")
    model_json = model.to_json()

    with open(checkpoint_path + "model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(checkpoint_path + filename)
    print("Model saved")


def load(checkpoint_path, model):
    if model is None:
        raise Exception("You have to build the model first.")

    print("Loading model checkpoint {} ...\n".format(checkpoint_path))
    model.load_weights(checkpoint_path)
    print("Model loaded")