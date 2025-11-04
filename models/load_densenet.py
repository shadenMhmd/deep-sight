from tensorflow.keras.applications.densenet import DenseNet121

def load_model(weights_path=None, num_classes=2):
    """
    لو عندك وزن مُدرَّب ( .keras / .h5 ) حطه في weights_path.
    """
    model = DenseNet121(
        weights=None if weights_path else "imagenet",
        include_top=True,
        classes=num_classes
    )
    if weights_path:
        model.load_weights(weights_path)
    return model