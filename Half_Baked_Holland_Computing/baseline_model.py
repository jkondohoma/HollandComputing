

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD

# =============================================================================
# import os
# 
# 
# num_skipped = 0
# for folder_name in ('bomber', 'drone', 'fighter', 'utility'):
#     folder_path = os.path.join("plane_type", folder_name)
#     for fname in os.listdir(folder_path):
#         fpath = os.path.join(folder_path, fname)
#         try:
#             fobj = open(fpath, "rb")
#             is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
#         finally:
#             fobj.close()
# 
#         if not is_jfif:
#             num_skipped += 1
#             # Delete corrupted image
#             os.remove(fpath)
# 
# print("Deleted %d images" % num_skipped)
# =============================================================================


image_size = (150, 150)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "output/train",
    labels="inferred",
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "output/val",
    image_size=image_size,
    batch_size=batch_size,
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "output/test",
    image_size=image_size,
    batch_size=batch_size,
)



import matplotlib.pyplot as plt



print(train_ds.take(1))

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")        
        

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

input_shape=(150,150,3)
num_classes=3





# =============================================================================
# hidden_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_1')
# hidden_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_2')
# pool_1 = tf.keras.layers.MaxPool2D(padding='same')
# hidden_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_3')
# hidden_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_4')
# pool_2 = tf.keras.layers.MaxPool2D(padding='same')
# flatten = tf.keras.layers.Flatten()
# output = tf.keras.layers.Dense(4)
# conv_classifier = tf.keras.Sequential([hidden_1, hidden_2, pool_1, hidden_3, hidden_4, pool_2, flatten, output])
# =============================================================================

def make_model(input_shape, num_classes):

    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=3)

# Run some data through the network to initialize it
# =============================================================================
# for batch in train_ds:
#     # data is uint8 by default, so we have to cast it
#     model(tf.cast(batch['image'], tf.float32))
#     break
# =============================================================================
#model.summary()


#keras.utils.plot_model(model, show_shapes=True)

opt = SGD(lr=1e-2) 

epochs = 10

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
#class_weight = {0: 1/443, 1: 1/1244, 2: 1/765}
model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds) #, class_weight=class_weight,



#train_ds.class_names
























