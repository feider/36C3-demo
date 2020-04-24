from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Softmax, ReLU, Conv2D, GlobalAveragePooling2D
import sys

if __name__ == "__main__":

    ## PART I: CREATE THE MODEL

    m_in = Input(shape=(28, 28, 3))
    l = Conv2D(64, (3, 3), activation="relu")(m_in)
    l = Conv2D(64, (3, 3), activation="relu")(l)
    l = GlobalAveragePooling2D()(l)
    l = Dense(128, activation="relu")(l)
    m_out = Dense(10, activation="softmax")(l)
    model = Model(inputs=m_in, outputs=m_out)
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
   
    ## PART II: TRAIN THE MODEL

    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        img_generator = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                rotation_range=30)

        train_gen = img_generator.flow_from_directory(
                'mnist_png/training',
                target_size=(28, 28),
                batch_size=16,
                class_mode = "sparse"
                )

        val_generator = ImageDataGenerator(
                rescale=1./255)
        val_gen = val_generator.flow_from_directory(
                'mnist_png/testing',
                target_size=(28, 28),
                batch_size=16,
                class_mode = "sparse"
                )
        model.fit_generator(
                train_gen,
                steps_per_epoch=60000/16,
                validation_data = val_gen,
                validation_steps = 10000/16,
                epochs=2)
    model.save("demo_model")
        

