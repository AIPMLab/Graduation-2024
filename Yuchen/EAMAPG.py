import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt
# 加载模型
model_path = "C:\\Users\DELL\Desktop\XAI code\Miccai COVID Densenet121.h5"
model = load_model(model_path)

# 类别标签
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def pgd_attack(model, input_image, epsilon=0.1, epsilon_step=0.01, num_steps=100):
    perturbed_image = input_image
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(perturbed_image)
            prediction = model(perturbed_image, training=False)
            loss = -tf.reduce_mean(tf.keras.losses.categorical_crossentropy(prediction, prediction))

        gradient = tape.gradient(loss, perturbed_image)
        perturbation = epsilon_step * tf.sign(gradient)
        perturbed_image = perturbed_image + perturbation
        perturbed_image = tf.clip_by_value(perturbed_image, input_image - epsilon, input_image + epsilon)
        perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)

    return perturbed_image

def reverse_preprocess_input(img_array):
    # 将数据从[-1, 1]范围转换回[0, 255]
    img_array += 1  # 将范围从[-1, 1]变换到[0, 2]
    img_array *= 127.5  # 将范围从[0, 2]变换到[0, 255]
    return np.clip(img_array, 0, 255).astype('uint8')  # 确保值在0到255之间，并转换为整数


def generate_adversarial_example(model, input_image_path, epsilon=0.1, epsilon_step=0.01, num_steps=100):
    img = image.load_img(input_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    input_image = tf.convert_to_tensor(img_array, dtype=tf.float32)
    adversarial_image = pgd_attack(model, input_image, epsilon, epsilon_step, num_steps)
    return img_array, adversarial_image

def process_predictions(predictions, class_labels):
    predicted_class_indices = np.argmax(predictions, axis=-1)
    predicted_labels = [class_labels[i] for i in predicted_class_indices]
    predicted_probabilities = np.max(predictions, axis=-1)
    return predicted_labels, predicted_probabilities

def compare_original_and_adversarial(model, original_image, adversarial_image):
    original_image_rev = reverse_preprocess_input(original_image)
    adversarial_image_rev = reverse_preprocess_input(adversarial_image.numpy())

    original_pred = model.predict(original_image)
    adversarial_pred = model.predict(adversarial_image)

    original_labels, original_confidence = process_predictions(original_pred, class_labels)
    adversarial_labels, adversarial_confidence = process_predictions(adversarial_pred, class_labels)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(original_image_rev[0].astype('uint8'))
    axs[0].set_title(f'Original: {original_labels[0]}\nConfidence: {original_confidence[0]:.2f}')
    axs[0].axis('off')

    axs[1].imshow(adversarial_image_rev[0].astype('uint8'))
    axs[1].set_title(f'Adversarial: {adversarial_labels[0]}\nConfidence: {adversarial_confidence[0]:.2f}')
    axs[1].axis('off')

    plt.show()

input_image_path ="C:\\Users\DELL\Desktop\code\dataset\covid-xray\Data\\test\COVID19\COVID19(564).jpg"# Update to your actual image path

original_image, adversarial_image = generate_adversarial_example(model, input_image_path, epsilon=0.1, epsilon_step=0.01, num_steps=100)
compare_original_and_adversarial(model, original_image, adversarial_image)
img = image.load_img(input_image_path, target_size=(224, 224))
