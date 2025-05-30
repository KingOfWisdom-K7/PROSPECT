import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras import backend as K
import config

@tf.keras.utils.register_keras_serializable()
def f1_score(y_true, y_pred):
    y_pred_binary = K.round(y_pred)
    y_true = K.cast(y_true, 'float')
    tp = K.sum(K.cast(y_true * y_pred_binary, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_binary, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_binary), 'float'), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

def build_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(*config.IMG_SIZE, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'), f1_score])
    return model
