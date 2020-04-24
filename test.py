from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
import matplotlib.pyplot as plt

if __name__=="__main__":
    model = load_model(sys.argv[1])

    
    val_generator = ImageDataGenerator(
            rescale=1./255)
    val_gen = val_generator.flow_from_directory(
            'mnist_png/testing',
            target_size=(28, 28),
            batch_size=16,
            class_mode = "sparse"
            )

    for v in val_gen:
    #    for i in v:
    #        print(i.shape)
    #    exit()
        results = model.predict(v[0])
        for i in range(16):
            img = v[0][i, :, :, :]
            print("#####")
            print("...")
            print("...")
            print("...")
            print("...")
            print("#####")
            r = results[i, :]
            m = np.argmax(r)
            print(f'Detected: {m}, p: {int(r[m]*100)}%, truth: {int(v[1][i])}')
            r[np.argmax(r)] = 0
            _m = np.argmax(r)
            print(f'Second most probable: {_m}, p: {int(r[_m]*100)}%')
            plt.imshow(img)
            plt.show()
