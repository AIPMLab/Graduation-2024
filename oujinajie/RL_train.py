from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf
from absl import app, flags
import cleverhans
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import MobileNet
import sys
import gym
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import layers, models
FLAGS = flags.FLAGS

warnings.filterwarnings("ignore")
# Confirm TensorFlow can see the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# sess = tf.Session()
# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# 自定义回调函数，跟踪最小的验证损失
class MinValLossTracker(Callback):
    def __init__(self):
        super(MinValLossTracker, self).__init__()
        self.min_val_loss = float('inf')
        self.best_epoch_stats = {}

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            val_loss = logs.get('val_loss')
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.best_epoch_stats = logs.copy()
                self.best_epoch_stats['epoch'] = epoch

    def on_train_end(self, logs=None):
        best_epoch = self.best_epoch_stats.get('epoch')
        print(f"\nBest Epoch: {best_epoch}")
        for key, value in self.best_epoch_stats.items():
            if key != 'epoch':
                print(f"{key}: {value:.4f}")



def load_data(datasetfolder, image_data_generator):
    dataflowtraining = image_data_generator.flow_from_directory(
        directory=datasetfolder,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=16,
        shuffle=True,
        subset='training')

    dataflowvalidation = image_data_generator.flow_from_directory(
        directory=datasetfolder,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=16,
        shuffle=True,
        subset='validation')


    return dataflowtraining, dataflowvalidation

def plot_history(hist):
    save_path = r"C:\Users\PS\Desktop\ojjRL\brain tumor"
    plt.figure(figsize=(12, 6))
    metrics = ['loss', 'precision', 'recall', 'accuracy']
    for i in range(4):
        plt.subplot(2, 2, (i + 1))
        plt.plot(hist.history[metrics[i]], label=metrics[i])
        plt.plot(hist.history['val_{}'.format(metrics[i])], label='val_{}'.format(metrics[i]))
        plt.legend()
    if save_path:
        plt.savefig(save_path + "VGG16_metrics.png")


    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path + "VGG16_accuracy_loss.png")
    plt.tight_layout()


def build_model():
    # Load models with pre-trained ImageNet weights
    # basemodel = LeNet5(input_shape=(224, 224, 3), num_classes=4)
    basemodel = Xception(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3), pooling=None)
    x = tf.keras.layers.GlobalAveragePooling2D()(basemodel.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    m = tf.keras.models.Model(inputs=basemodel.input, outputs=x)

    m.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])
    return m

def add_random_noise(image, noise_factor):
    # noise_factor=np.clip(noise_factor, 0.0, 1.0)
    noise_factor=abs(noise_factor)
    # 生成噪声数组（与图像大小相同）
    noise = np.random.normal(loc=0, scale=noise_factor, size=image.shape)
    # 将噪声添加到图像上
    noisy_image = image + noise*20
    # 将像素值限制在 [0, 1] 范围内

    return noisy_image  # 返回噪声图像和原始图像
# def add_random_noise(images, noise_factor):
#     # 初始化一个与输入 images 相同形状的噪声数组
#     noisy_images = np.zeros_like(images)
#     noise_factor = np.clip(noise_factor, 0.0, 1.0)
#     # 遍历每个图像并添加噪声
#     for i in range(images.shape[0]):
#         noise = np.random.normal(loc=0.0, scale=noise_factor, size=images[i].shape)
#         noisy_images[i] = np.clip(images[i] + noise, 0.0, 1.0)
#
#     return noisy_images


# def calculate_similarity_reward(model, data, noisedata, eps=0.3):
#     # 创建 FastGradientMethod 攻击实例，使用给定的模型和 epsilon 参数
#     fgsm = fast_gradient_method(model, data, eps, 1)
#
#     # 生成对抗样本，其中 eps 参数表示在每个特征上允许的最大扰动大小
#     adv_data = fgsm
#     # 获取批次大小
#     batch_size = data.shape[0]
#     # 获取模型对原始样本和对抗样本的输出
#     predictions_original = model.predict(data, steps=batch_size)
#     predictions_adversarial = model.predict(adv_data,steps=batch_size)
#     predictions_noise = model.predict(noisedata,steps=batch_size)
#     # 使用余弦相似度计算两个输出的相似性
#     similarity_matrix_1 = cosine_similarity(predictions_original, predictions_adversarial)
#     similarity_matrix_2 = cosine_similarity(predictions_original, predictions_noise)
#     # 计算平均相似性，可以使用其他统计量
#     average_similarity_1 = np.mean(similarity_matrix_1)
#     average_similarity_2 = np.mean(similarity_matrix_2)
#     # 将相似性映射到奖励值
#     similarity_reward_1= -average_similarity_1
#     similarity_reward_2= average_similarity_2
#     return similarity_reward_1,similarity_reward_2

# def calculate_similarity_reward(model, data, noisedata, Adv_Data, batch_size=16):
#     print(1)
#
#     # 将数据调整为批次大小的整数倍
#     data_adjusted = data[:len(data) // batch_size * batch_size]
#     noisedata_adjusted = noisedata[:len(noisedata) // batch_size * batch_size]
#     num_samples = len(data_adjusted)
#     similarity_reward_1_total = 0.0
#     similarity_reward_2_total = 0.0
#
#     for i in range(0, num_samples, batch_size):
#         batch_data = data_adjusted[i:i+batch_size]
#         batch_noisedata = noisedata_adjusted[i:i+batch_size]
#         batch_adv = Adv_Data[i:i+batch_size]
#
#         # 获取模型对原始样本和对抗样本的输出
#         predictions_original = model.predict(batch_data)
#         predictions_adversarial = model.predict(batch_adv,steps=batch_adv.shape[0])
#         predictions_noise = model.predict(batch_noisedata)
#         # 使用余弦相似度计算两个输出的相似性
#         similarity_matrix_1 = cosine_similarity(predictions_original, predictions_adversarial)
#         similarity_matrix_2 = cosine_similarity(predictions_original, predictions_noise)
#         # 计算平均相似性，可以使用其他统计量
#         average_similarity_1 = np.mean(similarity_matrix_1)
#         average_similarity_2 = np.mean(similarity_matrix_2)
#         # 将相似性映射到奖励值
#         similarity_reward_1_total += -average_similarity_1
#         similarity_reward_2_total += average_similarity_2
#     print(1)
#     # 计算平均奖励
#     similarity_reward_1 = similarity_reward_1_total / (num_samples / batch_size)
#     similarity_reward_2 = similarity_reward_2_total / (num_samples / batch_size)
#     print(similarity_reward_1)
#     print(similarity_reward_2)
#     del predictions_original
#     del predictions_adversarial
#     del predictions_noise
#     del similarity_matrix_1
#     del similarity_matrix_2
#     del average_similarity_1
#     del average_similarity_2
#     return similarity_reward_1, similarity_reward_2
#
def calculate_similarity_reward(model, data, datalabel,noisedata, batch_size=16, eps=0.6):
    print(1)

    # 将数据调整为批次大小的整数倍
    data_adjusted = data[:len(data) // batch_size * batch_size]
    datalabel_adjusted = datalabel[:len(datalabel) // batch_size * batch_size]
    noisedata_adjusted = noisedata[:len(noisedata) // batch_size * batch_size]

    # 只取前三个批次的数据
    data_adjusted = data_adjusted[:batch_size * 3]
    datalabel_adjusted = datalabel_adjusted[:batch_size * 3]
    noisedata_adjusted = noisedata_adjusted[:batch_size * 3]

    print(1)
    adversarial_samples = []
    # # 对每个数据批次进行攻击
    for i in range(0, len(data_adjusted), batch_size):
        data_batch = data_adjusted[i:i + batch_size]
        # 创建 FastGradientMethod 攻击实例，使用给定的模型和 epsilon 参数
        fgsm = fast_gradient_method(model, data_batch, eps, np.inf)
        adversarial_samples.append(fgsm)  # 将每个对抗样本转换为张量

        # 将对抗样本连接成一个单一张量
    adversarial_samples_combined = tf.concat(adversarial_samples, axis=0)


    evaluation1 = model.evaluate(adversarial_samples_combined, datalabel_adjusted, steps=len(datalabel_adjusted)/16)
    evaluation2 = model.evaluate(noisedata_adjusted, datalabel_adjusted)

    acc_adversarial = -evaluation1[1]
    acc_noise = evaluation2[1]
    print(2)
    del fgsm
    del adversarial_samples
    del evaluation1
    del evaluation2

    return acc_adversarial,acc_noise



class DataAugmentationEnv(gym.Env):
    def __init__(self, datasetfolder):
        self.best_val_loss = 99
        self.mixed = 0

        super(DataAugmentationEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.datasetfolder = datasetfolder

        ge = ImageDataGenerator(rescale=1 / 255,
                            rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            fill_mode='constant',
                            validation_split=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            zoom_range=0.2,)

        self.dataflowtraining, self.dataflowvalidation = load_data(self.datasetfolder, ge)



    def reset(self):
        return np.array([0.5])

    def step(self, action):
        updated_generator = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.2,
            rescale=1 / 255,
            fill_mode='constant',
            validation_split=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=lambda x: add_random_noise(x, noise_factor=(action[0]+action[1])/4)  # 调整 noise_intensity 的值
        )

        # 使用更新后的生成器加载数据
        self.dataflowtraining_noi, self.dataflowvalidation_noi = load_data(self.datasetfolder, updated_generator)


        # 初始化空列表，用于存储所有数据和标签,用于training
        all_images_validation = []
        all_labels_validation = []

        # 获取迭代器的步数（每个 epoch 的迭代次数）
        steps = len(self.dataflowvalidation)

        # 循环迭代器，将数据和标签存储在列表中
        for _ in range(steps):
            batch_images, batch_labels = next(self.dataflowvalidation)
            all_images_validation.append(batch_images)
            all_labels_validation.append(batch_labels)

        # 将列表中的数据和标签合并成一个大的数组
        all_images_validation = np.concatenate(all_images_validation, axis=0)
        all_labels_validation = np.concatenate(all_labels_validation, axis=0)

        # 初始化空列表，用于存储所有数据和标签,用于training
        noise_images_validation = []
        # noise_labels_validation = []

        # 获取迭代器的步数（每个 epoch 的迭代次数）
        steps = len(self.dataflowvalidation_noi)

        # 循环迭代器，将数据和标签存储在列表中
        for _ in range(steps):
            batch_images, batch_labels = next(self.dataflowvalidation_noi)
            noise_images_validation.append(batch_images)
            # noise_labels_validation.append(batch_labels)

        # 将列表中的数据和标签合并成一个大的数组
        noise_images_validation = np.concatenate(noise_images_validation, axis=0)
        # noise_labels_validation = np.concatenate(noise_labels_validation, axis=0)

        m = build_model()

        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss', mode='min', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=6, monitor='val_loss', mode='min', factor=0.1),
            MinValLossTracker()
        ]

        hist = m.fit(
            self.dataflowtraining_noi,
            epochs=100,
            validation_data=self.dataflowvalidation,
            verbose=0,
            callbacks=callbacks_list
        )
        # hist = m.fit(
        #     self.dataflowtraining,
        #     epochs=100,
        #     validation_data=self.dataflowvalidation,
        #     verbose=0,
        #     callbacks=callbacks_list
        # )
        # 其余代码保持不变

        print(
            "loss: {:.4f} - accuracy: {:.4f} - precision: {:.4f} - recall: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f} - val_precision: {:.4f} - val_recall: {:.4f} - lr: {:.1e}".format(
                hist.history['loss'][-1], hist.history['accuracy'][-1], hist.history['precision'][-1],
                hist.history['recall'][-1],
                hist.history['val_loss'][-1], hist.history['val_accuracy'][-1], hist.history['val_precision'][-1],
                hist.history['val_recall'][-1],
                float(tf.keras.backend.get_value(m.optimizer.learning_rate))
            ))

        # val_accuracy = np.max(hist.history['val_accuracy'])
        val_loss = np.min(hist.history['val_loss'])
        # #合并训练和验证数据
        # all_images_combined = np.concatenate([all_images_training, all_images_validation], axis=0)
        # all_noiseimages_combined = np.concatenate([noisy_train_data, noisy_validation_data], axis=0)

        adv_acc,nio_acc = calculate_similarity_reward(m, all_images_validation,all_labels_validation, noise_images_validation,batch_size=16,eps=0.6 )
        if val_loss < self.best_val_loss:
            print("当前V_loss:", val_loss)
            print("当前Best_loss:", self.best_val_loss)
            print("当前的mix准确率：", adv_acc + nio_acc)
            print("当前最大mix准确率：", self.mixed)
            self.best_val_loss = val_loss
            self.mixed = adv_acc + nio_acc
            m.save('new_chest Xception_1.h5')
            print('save the model')
            # plot_history(hist)

        reward = adv_acc + nio_acc

        del all_images_validation
        del all_labels_validation

        del noise_images_validation



        done = True
        return np.array([0.5]), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def build_actor(env):
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16, activation="relu"))
    actor.add(Dense(16, activation="relu"))
    actor.add(Dense(16, activation="relu"))
    actor.add(Dense(env.action_space.shape[0], activation="linear"))
    return actor

def build_critic(env):
    # 创建Critic模型
    action_input = tf.keras.layers.Input(shape=(env.action_space.shape[0],))
    observation_input = tf.keras.layers.Input(shape=(1,) + env.observation_space.shape)
    flattened_observation = Flatten()(observation_input)
    x = tf.keras.layers.Concatenate()([action_input, flattened_observation])
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    critic = tf.keras.models.Model(inputs=[action_input, observation_input], outputs=x)
    return action_input, critic

def main():
    # 保存当前的标准输出流
    original_stdout = sys.stdout

    # 打开文件，如果文件不存在则创建，如果存在则覆盖
    with open('new_chest result_Xception_1 .txt', 'w') as f:
        sys.stdout = f
        datasetfolder = r"C:\Users\PS\PycharmProjects\pythonProject5\Data\train"
        env = DataAugmentationEnv(datasetfolder)
        actor = build_actor(env)
        action_input, critic = build_critic(env)

        memory = SequentialMemory(limit=1000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=.15, mu=0., sigma=.3)
        agent = DDPGAgent(nb_actions=env.action_space.shape[0], actor=actor, critic=critic,
                          critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=.99, target_model_update=1e-3)
        agent.compile(Adam(lr=.00001, clipnorm=1.), metrics=["mae"])
        agent.fit(env, nb_steps=20, visualize=False, verbose=1)
        sys.stdout = original_stdout

if __name__ == "__main__":
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")

    main()
