from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input


def keras_ner(input_shape=512, hidden_size=128, num_classes=20):
    # Here goes actual model definition
    text_input = Input(shape=(input_shape,), name="Words")
    projected = Dense(hidden_size, activation="relu", name="Linear")(text_input)
    projected2 = Dropout(0.1)(projected)
    output = Dense(num_classes, activation="softmax", name="Output")(projected2)
    model = Model(inputs=[text_input], outputs=[output])
    return model
