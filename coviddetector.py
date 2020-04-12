import keras
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Activation
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.callbacks import EarlyStopping
import numpy as np



training_folder = 'path to training data'
test_folder = 'path to test data'

## Defining CNN Model with 1 input layer and multiple hidden layers

def DefineModel():

        model = Sequential()

        model.add(Conv2D(1024,kernel_size=(3,3),input_shape=(64,64,3),activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(512,kernel_size=(3,3),activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(512,kernel_size=(3,3),activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(512,kernel_size=(3,3),padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(256,kernel_size=(3,3),padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(256,kernel_size=(3,3),padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(units=2048))
        model.add(Activation('relu'))
        model.add(Dense(units=4))
        model.add(Activation('softmax'))

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
        return model

## Training the model with images(Using Image data generator to generate images of various augmentation)

def trainModel(model):

        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                training_folder,
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
                test_folder,
                target_size=(64, 64),
                batch_size=32,
        class_mode='categorical')



        model.fit(train_generator,steps_per_epoch=400,epochs=10,validation_data=validation_generator,callbacks=[EarlyStopping(monitor='accuracy',restore_best_weights=True,patience=3)])

        model.save_weights('covid19detector.h5')

## Predicting model with real time images

def predict(model):

        image = load_img('path of the image to be predicted',target_size=(64,64))
        image  = img_to_array(image)
        image = np.expand_dims(image,axis=0)
        image /= 255.
        model.load_weights('covid19detector.h5')
        model.predict_classes(image)


if __name__ == "__main__":
        model = DefineModel()
        trainModel(model)
        predict(model)
    



