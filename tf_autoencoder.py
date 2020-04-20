import mnist 
import numpy as np 
import tensorflow as tf

tf.enable_eager_execution()

def get_data():
    images = mnist.train_images()
    return np.expand_dims(images, axis=1)

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

    def build(self, input_shape):
        self.l1 = tf.keras.layers.Conv2D(32, (5, 5), input_shape=(28, 28), activation=tf.keras.activations.relu)
        self.l2 = tf.keras.layers.Conv2D(16, (5, 5), activation=tf.keras.activations.relu)
        self.l3 = tf.keras.layers.Conv2D(8, (5, 5), activation=tf.keras.activations.relu)
        self.l4 = tf.keras.layers.Conv2D(1, (5, 5), activation=tf.keras.activations.relu)
    
    def call(self, input):
        x = tf.reshape(input, [-1, 28, 28, 1])
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

    def build(self, input_shape):
        self.l1 = tf.keras.layers.Conv2DTranspose(8, (5, 5), activation=tf.keras.activations.relu)
        self.l2 = tf.keras.layers.Conv2DTranspose(16, (5, 5), activation=tf.keras.activations.relu)
        self.l3 = tf.keras.layers.Conv2DTranspose(32, (5, 5), activation=tf.keras.activations.relu)
        self.l4 = tf.keras.layers.Conv2DTranspose(1, (5, 5), activation=tf.keras.activations.relu)
    
    def call(self, input):
        x = self.l1(input)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

def predict(model, original):
    return model(original)

def loss_fn(original, reconstructed):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(original, reconstructed)))
    return reconstruction_error

def train(model, opt, original):
    with tf.GradientTape() as t:
        reconstructed = predict(model, original)
        loss = loss_fn(original, reconstructed)
    gradients = t.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

imgs = get_data()
imgs = imgs / tf.reduce_max(imgs)
autoencoder = Autoencoder()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

training_dataset = tf.data.Dataset.from_tensor_slices(imgs)
training_dataset = training_dataset.batch(64)
training_dataset = training_dataset.shuffle(imgs.shape[0])

for epoch in range(15):
    for step, batch_features in enumerate(training_dataset):
        train(autoencoder, opt, batch_features)