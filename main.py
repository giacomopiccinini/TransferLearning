import tensorflow as tf
import tensorflow_datasets as tfds
import logging
from tensorflow import keras
from tensorflow.keras import layers


if __name__ == '__main__':
    
    # Set logger
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    # Load CelebA dataset, including labels. Download if not present
    train_set, validation_set, test_set = tfds.load(
        "celeb_a",
        as_supervised=True
    )

    # Count images for the three datasets
    n_train = tf.data.experimental.cardinality(train_set)
    n_validation = tf.data.experimental.cardinality(validation_set)
    n_test = tf.data.experimental.cardinality(test_set)
    
    # Log info on images
    log.info(f"Train images: {n_train}")
    log.info(f"Validation images: {n_validation}")
    log.info(f"Test images: {n_test}")
    
    # Resize images to a fixed dimension
    size = (150, 150)

    train_set = train_set.map(lambda x, y: (tf.image.resize(x, size), y))
    validation_set = validation_set.map(lambda x, y: (tf.image.resize(x, size), y))
    test_set = test_set.map(lambda x, y: (tf.image.resize(x, size), y))
    
    # Batch data and use caching & prefetching to optimize loading speed
    batch_size = 32
    buffer_size = 10

    train_set = train_set.cache().batch(batch_size).prefetch(buffer_size=buffer_size)
    validation_set = validation_set.cache().batch(batch_size).prefetch(buffer_size=buffer_size)
    test_set = test_set.cache().batch(batch_size).prefetch(buffer_size=buffer_size)
    
    # Define data augmentation layer
    data_augmentation = keras.Sequential(
        [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
    )
        

    # First, load a base model. In this case InceptionResNetV2
    base_model = keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=None)

    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = keras.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.summary()