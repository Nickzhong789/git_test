from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

from keras.applications import inception_v3
from keras.applications import resnet50
from ocelot.resnet import ResnetBuilder


def iv3avg(input_shape=(512,512,3),num_classes=11):
    base_model = inception_v3.InceptionV3(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape, 
        pooling="avg",
    )

    model = Sequential()
    model.add(base_model)
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation="softmax"))
    return model


def iv3avg_linear(input_shape=(512,512,3)):
    base_model = inception_v3.InceptionV3(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape, 
        pooling="avg",
    )

    model = Sequential()
    model.add(base_model)
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


def resnet(input_shape=(512,512,3),num_classes=11):
    base_model = resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling="avg",
    )

    model = Sequential()
    model.add(base_model)
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return model


def resnet101_linear(input_shape=(512, 512, 3)):
    base_model = ResnetBuilder.build_resnet_101(
        input_shape=input_shape,
        num_outputs=1
    )

    model = Sequential()
    model.add(base_model)
    return model


def resnet50_linear(input_shape=(512,512,3)):
    base_model = resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling="avg",
    )

    model = Sequential()
    model.add(base_model)
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


get_model = resnet
