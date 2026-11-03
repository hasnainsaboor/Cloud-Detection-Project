import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Calculates Intersection-over-Union (IoU) metric for cloud detection evaluation.
    Target threshold: >= 85%
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    total = K.sum(y_true_f) + K.sum(y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

def build_unet(input_shape=(256, 256, 3)):
    """
    U-Net CNN Architecture for Satellite Image Cloud Segmentation.
    Includes Skip Connections to preserve spatial resolution.
    """
    inputs = layers.Input(shape=input_shape)

    # --- ENCODER (Feature Extraction) ---
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # --- BOTTLENECK ---
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # --- DECODER (Reconstruction + Skip Connections) ---
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.concatenate([u4, c2])  # Skip connection from Encoder Level 2
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c1])  # Skip connection from Encoder Level 1
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    # --- OUTPUT LAYER (Binary Cloud Mask: 1=Cloud, 0=Clear) ---
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', iou_metric])
    
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()
    print("\nFull U-Net model with skip connections & IoU metric compiled successfully!")