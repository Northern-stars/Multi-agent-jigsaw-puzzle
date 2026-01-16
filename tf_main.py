import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
# import cv2
import os
from efficientnet_like import model as efficientnet_model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_virtual_device_configuration(
    tf.config.experimental.list_physical_devices('GPU')[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14000)]
)
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
#
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95)

# img1path, img2path, label_list = loadDataset.load_dir(os.path.join("/home/notting/Desktop/PuzzleSolving", 'fragment/train'))
# [main_dataset1, main_dataset2], main_label = loadDataset.read_img(img1path, img2path, label_list)
# train_dataset = [main_dataset1[:72000], main_dataset2[:72000]]
# train_label = main_label[:72000]
# val_dataset = [main_dataset1[720007327:], main_dataset2[72000:]]
# val_label = main_label[72000:]
#
# img1path, img2path, label_list = loadDataset.load_dir(os.path.join("/home/notting/Desktop/PuzzleSolving", 'fragment/test'))
# test_dataset, test_label = loadDataset.read_img(img1path, img2path, label_list)

train_data_x = np.load('MET_Dataset/select_image/train_img_no_gap.npy')
# train_data_x = np.load('./PACS/train_img_48gap.npy')
# train_data_x = np.load('./MET_Dataset/select_image/train_img_0gap.npy')
# train_data_x_2 = np.load('MET_Dataset/select_image/text_train_img_no_gap.npy')
# train_data_x = np.concatenate((train_data_x_1, train_data_x_2), axis=0)
train_data_y = np.load('MET_Dataset/select_image/train_label_no_gap.npy')
# train_data_y = np.load('./PACS/train_label_48gap.npy')
# train_data_y = np.load('./MET_Dataset/select_image/train_label_0gap.npy')
# train_data_y_2 = np.load('MET_Dataset/select_image/text_train_label_no_gap.npy')
# train_data_y = np.concatenate((train_data_y_1, train_data_y_2), axis=0)
train_x_1 = np.concatenate((train_data_x[:, :96, :96, :], train_data_x[:, :96, 96:192, :],
                            train_data_x[:, :96, 192:, :], train_data_x[:, 96:192, :96, :],
                            train_data_x[:, 96:192, 192:, :], train_data_x[:, 192:, :96, :],
                            train_data_x[:, 192:, 96:192, :], train_data_x[:, 192:, 192:, :]))
train_x_2 = np.tile(train_data_x[:, 96:192, 96:192, :], (8, 1, 1, 1))
train_y = np.concatenate((train_data_y[:, 0, :], train_data_y[:, 1, :], train_data_y[:, 2, :], train_data_y[:, 3, :],
                          train_data_y[:, 4, :], train_data_y[:, 5, :], train_data_y[:, 6, :], train_data_y[:, 7, :]))
train_x = [train_x_1[:], train_x_2[:]]

valid_data_x = np.load('MET_Dataset/select_image/valid_img_no_gap.npy')
# valid_data_x = np.load('./PACS/valid_img_48gap.npy')
# valid_data_x = np.load('./MET_Dataset/select_image/valid_img_0gap.npy')
# valid_data_x_2 = np.load('MET_Dataset/select_image/text_valid_img_no_gap.npy')
# valid_data_x = np.concatenate((valid_data_x_1, valid_data_x_2), axis=0)
valid_data_y = np.load('MET_Dataset/select_image/valid_label_no_gap.npy')
# valid_data_y = np.load('./PACS/valid_label_48gap.npy')
# valid_data_y = np.load('./MET_Dataset/select_image/valid_label_0gap.npy')
# valid_data_y_2 = np.load('MET_Dataset/select_image/text_valid_label_no_gap.npy')
# valid_data_y = np.concatenate((valid_data_y_1, valid_data_y_2), axis=0)
valid_x_1 = np.concatenate((valid_data_x[:, :96, :96, :], valid_data_x[:, :96, 96:192, :],
                            valid_data_x[:, :96, 192:, :], valid_data_x[:, 96:192, :96, :],
                            valid_data_x[:, 96:192, 192:, :], valid_data_x[:, 192:, :96, :],
                            valid_data_x[:, 192:, 96:192, :], valid_data_x[:, 192:, 192:, :]))
valid_x_2 = np.tile(valid_data_x[:, 96:192, 96:192, :], (8, 1, 1, 1))
valid_y = np.concatenate((valid_data_y[:, 0, :], valid_data_y[:, 1, :], valid_data_y[:, 2, :], valid_data_y[:, 3, :],
                          valid_data_y[:, 4, :], valid_data_y[:, 5, :], valid_data_y[:, 6, :], valid_data_y[:, 7, :]))
valid_x = [valid_x_1[:], valid_x_2[:]]

test_data_x = np.load('MET_Dataset/select_image/test_img_no_gap.npy')
# test_data_x = np.load('./PACS/test_img_48gap.npy')
# test_data_x = np.load('./MET_Dataset/select_image/test_img_0gap.npy')
# test_data_x_2 = np.load('MET_Dataset/select_image/text_test_img_no_gap.npy')
# test_data_x = np.concatenate((test_data_x_1, test_data_x_2), axis=0)
test_data_y = np.load('MET_Dataset/select_image/test_label_no_gap.npy')
# test_data_y = np.load('./PACS/test_label_48gap.npy')
# test_data_y = np.load('./MET_Dataset/select_image/test_label_0gap.npy')
# test_data_y_2 = np.load('MET_Dataset/select_image/text_test_label_no_gap.npy')
# test_data_y = np.concatenate((test_data_y_1, test_data_y_2), axis=0)
test_data_x_1 = np.concatenate((test_data_x[:, :96, :96, :], test_data_x[:, :96, 96:192, :],
                                test_data_x[:, :96, 192:, :], test_data_x[:, 96:192, :96, :],
                                test_data_x[:, 96:192, 192:, :], test_data_x[:, 192:, :96, :],
                                test_data_x[:, 192:, 96:192, :], test_data_x[:, 192:, 192:, :]))

test_data_x_2 = np.tile(test_data_x[:, 96:192, 96:192, :], (8, 1, 1, 1))

# test_data_x = np.load('pure_test_img_33.npy')
# test_data_y = np.load('test_33.npy')
#
# cv2.imwrite('1.png', test_data_x[0, 0])
# cv2.imwrite('2.png', test_data_x[0, 1])
# cv2.imwrite('3.png', test_data_x[0, 2])
# cv2.imwrite('4.png', test_data_x[0, 3])
# cv2.imwrite('5.png', test_data_x[0, 4])
# cv2.imwrite('6.png', test_data_x[0, 5])
# cv2.imwrite('7.png', test_data_x[0, 6])
# cv2.imwrite('8.png', test_data_x[0, 7])
# cv2.imwrite('9.png', test_data_x[0, 8])
#
# test_data_x_1 = np.concatenate((test_data_x[:, 0], test_data_x[:, 1], test_data_x[:, 2], test_data_x[:, 3],
#                                 test_data_x[:, 5], test_data_x[:, 6], test_data_x[:, 7], test_data_x[:, 8]))
# test_data_x_2 = np.tile(test_data_x[:, 4], (8, 1, 1, 1))

test_y = np.concatenate((test_data_y[:, 0, :], test_data_y[:, 1, :], test_data_y[:, 2, :], test_data_y[:, 3, :],
                         test_data_y[:, 4, :], test_data_y[:, 5, :], test_data_y[:, 6, :], test_data_y[:, 7, :]))
test_x = [test_data_x_1[:], test_data_x_2[:]]

# painting_test_data_x_1 = np.concatenate((test_data_x[:667, :96, :96, :], test_data_x[:667, :96, 96:192, :],
#                                          test_data_x[:667, :96, 192:, :], test_data_x[:667, 96:192, :96, :],
#                                          test_data_x[:667, 96:192, 192:, :], test_data_x[:667, 192:, :96, :],
#                                          test_data_x[:667, 192:, 96:192, :], test_data_x[:667, 192:, 192:, :]))
# painting_test_data_x_2 = np.tile(test_data_x[:667, 96:192, 96:192, :], (8, 1, 1, 1))
# painting_test_y = np.concatenate((test_data_y[:667, 0, :], test_data_y[:667, 1, :], test_data_y[:667, 2, :],
#                                   test_data_y[:667, 3, :], test_data_y[:667, 4, :], test_data_y[:667, 5, :],
#                                   test_data_y[:667, 6, :], test_data_y[:667, 7, :]))
# painting_test_x = [painting_test_data_x_1[:], painting_test_data_x_2[:]]
#
# engraving_test_data_x_1 = np.concatenate((test_data_x[667:1334, :96, :96, :], test_data_x[667:1334, :96, 96:192, :],
#                                           test_data_x[667:1334, :96, 192:, :], test_data_x[667:1334, 96:192, :96, :],
#                                           test_data_x[667:1334, 96:192, 192:, :], test_data_x[667:1334, 192:, :96, :],
#                                           test_data_x[667:1334, 192:, 96:192, :], test_data_x[667:1334, 192:, 192:, :]))
# engraving_test_data_x_2 = np.tile(test_data_x[667:1334, 96:192, 96:192, :], (8, 1, 1, 1))
# engraving_test_y = np.concatenate((test_data_y[667:1334, 0, :], test_data_y[667:1334, 1, :],
#                                    test_data_y[667:1334, 2, :], test_data_y[667:1334, 3, :],
#                                    test_data_y[667:1334, 4, :], test_data_y[667:1334, 5, :],
#                                    test_data_y[667:1334, 6, :], test_data_y[667:1334, 7, :]))
# engraving_test_x = [engraving_test_data_x_1[:], engraving_test_data_x_2[:]]
#
# artifact_test_data_x_1 = np.concatenate((test_data_x[1334:, :96, :96, :], test_data_x[1334:, :96, 96:192, :],
#                                          test_data_x[1334:, :96, 192:, :], test_data_x[1334:, 96:192, :96, :],
#                                          test_data_x[1334:, 96:192, 192:, :], test_data_x[1334:, 192:, :96, :],
#                                          test_data_x[1334:, 192:, 96:192, :], test_data_x[1334:, 192:, 192:, :]))
# artifact_test_data_x_2 = np.tile(test_data_x[1334:, 96:192, 96:192, :], (8, 1, 1, 1))
# artifact_test_y = np.concatenate((test_data_y[1334:, 0, :], test_data_y[1334:, 1, :],
#                                   test_data_y[1334:, 2, :], test_data_y[1334:, 3, :],
#                                   test_data_y[1334:, 4, :], test_data_y[1334:, 5, :],
#                                   test_data_y[1334:, 6, :], test_data_y[1334:, 7, :]))
# artifact_test_x = [artifact_test_data_x_1[:], artifact_test_data_x_2[:]]


def combo_metrics(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred) * 0.8 + \
           tf.keras.metrics.binary_accuracy(y_true, y_pred) * 0.2


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, feature_num):
        super(ResNet, self).__init__()
        self.stem = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2)),
                                         tf.keras.layers.BatchNormalization(),
                                         tf.keras.layers.Activation('relu'),
                                         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                         ])
        self.layer1 = self.build_block(64, layer_dims[0])
        self.layer2 = self.build_block(128, layer_dims[1], strides=2)
        self.layer3 = self.build_block(256, layer_dims[2], strides=2)
        self.layer4 = self.build_block(512, layer_dims[3], strides=2)
        # output.shape [b,512,h,w]->[b,512]

        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(feature_num)

    def call(self, inputs, training=None):
        '''
        调用函数：
        输入：inputs表示输入，training表示训练状态
        '''
        x = inputs
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_block(self, filter_num, blocks, strides=1):
        res_blocks = tf.keras.models.Sequential()
        res_blocks.add(ResBlock(filter_num, strides))
        # 第一个basicblock可以存在下采样，后面的basicblock stride=1
        for _ in range(1, blocks):
            res_blocks.add(ResBlock(filter_num, strides=1))

        return res_blocks

    @staticmethod
    def resnet18(feature_num):
        return ResNet([2, 2, 2, 2], feature_num)

    @staticmethod
    def resnet34(feature_num):
        return ResNet([3, 4, 6, 3], feature_num)


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, strides=1):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=strides, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        # conv+bn，与identity x相加后再relu
        self.conv2 = tf.keras.layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        # 第一层的stride不为1时，输出维度与x不同，需要进行处理
        if strides != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filter_num, (1, 1), strides=strides))
        else:
            self.downsample = lambda x: x

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        idenity = self.downsample(inputs)
        x = tf.keras.layers.add([x, idenity])
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1': self.conv1,
            'bn1': self.bn1,
            'relu': self.relu,
            'conv2': self.conv2,
            'bn2': self.bn2,
            'downsample': self.downsample
        })
        return config


class FEN (tf.keras.Model):
    def __init__(self, feature_num=512):
        super(FEN, self).__init__()

        tf.keras.backend.clear_session()

        self.block1_conv = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='block1_conv')
        self.block1_BN = tf.keras.layers.BatchNormalization(name='block1_BN')
        self.block1_relu = tf.keras.layers.ReLU(name='block1_ReLU')
        self.block1_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

        self.block2_conv = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='block2_conv')
        self.block2_BN = tf.keras.layers.BatchNormalization(name='block2_BN')
        self.block2_relu = tf.keras.layers.ReLU(name='block2_ReLU')
        self.block2_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

        self.block3_conv = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='block3_conv')
        self.block3_BN = tf.keras.layers.BatchNormalization(name='block3_BN')
        self.block3_relu = tf.keras.layers.ReLU(name='block3_ReLU')
        self.block3_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

        self.block4_conv = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='block4_conv')
        self.block4_BN = tf.keras.layers.BatchNormalization(name='block4_BN')
        self.block4_relu = tf.keras.layers.ReLU(name='block4_ReLU')
        self.block4_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

        self.block5_conv = tf.keras.layers.Conv2D(512, (3, 3), padding='same', name='block5_conv')
        self.block5_BN = tf.keras.layers.BatchNormalization(name='block5_BN')
        self.block5_relu = tf.keras.layers.ReLU(name='block5_ReLU')
        self.block5_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dense = tf.keras.layers.Dense(feature_num)
        self.flatten_BN = tf.keras.layers.BatchNormalization(name='flatten_BN')
        self.flatten_relu = tf.keras.layers.ReLU(name='flatten_ReLU')

    def call(self, inputs):
        x = inputs
        x = self.block1_conv(x)
        x = self.block1_BN(x)
        x = self.block1_relu(x)
        x = self.block1_pool(x)

        x = self.block2_conv(x)
        x = self.block2_BN(x)
        x = self.block2_relu(x)
        x = self.block2_pool(x)

        x = self.block3_conv(x)
        x = self.block3_BN(x)
        x = self.block3_relu(x)
        x = self.block3_pool(x)

        x = self.block4_conv(x)
        x = self.block4_BN(x)
        x = self.block4_relu(x)
        x = self.block4_pool(x)

        x = self.block5_conv(x)
        x = self.block5_BN(x)
        x = self.block5_relu(x)
        x = self.block5_pool(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.flatten_BN(x)
        x = self.flatten_relu(x)
        return x


def combo_net(input_shape):
    # fen_model = FEN(feature_num=512)
    # fen_model = ResNet.resnet34(feature_num=512)
    # base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights=None, input_shape=input_shape)
    # # base_model = tf.keras.applications.resnet.ResNet101(include_top=False, input_shape=input_shape)
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,
                                                                   weights=None,
                                                                   input_shape=input_shape)
    # # base_model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, input_shape=input_shape)
    # # base_model = efficientnet_model.efficientnet_like(include_top=False, input_shape=input_shape)
    # base_model_weights = tf.keras.models.load_model('./efficientnet_like/weights/resnet50_9class.h5').get_weights()[:-4]
    base_model_weights = tf.keras.models.load_model('./efficientnet_like/weights/efficientnet_b0_9class.h5').get_weights()[:-4]
    base_model.set_weights(base_model_weights)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.3, name="fen_dropout")(x)
    out = tf.keras.layers.Dense(512, activation='relu')(x)
    fen_model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    fragment1 = tf.keras.layers.Input(shape=input_shape, name='img1_input')
    fragment2 = tf.keras.layers.Input(shape=input_shape, name='img2_input')

    f1_feature = fen_model.call(fragment1)
    f2_feature = fen_model.call(fragment2)

    concatted_feature = tf.keras.layers.Concatenate()([f1_feature, f2_feature])
    x = tf.keras.layers.Dropout(0.2)(concatted_feature)
    # print(concatted_feature.shape)
    fc512 = tf.keras.layers.Dense(512)(x)
    bn = tf.keras.layers.BatchNormalization()(fc512)
    relu = tf.keras.layers.ReLU()(bn)
    fc64 = tf.keras.layers.Dense(512)(relu)
    bn = tf.keras.layers.BatchNormalization()(fc64)
    relu = tf.keras.layers.ReLU()(bn)
    # fc16 = tf.keras.layers.Dense(16)(relu)
    # bn = tf.keras.layers.BatchNormalization()(fc16)
    # relu = tf.keras.layers.ReLU()(bn)
    out8 = tf.keras.layers.Dense(8, activation='softmax', name='class_output')(relu)

    combo_net_model = tf.keras.models.Model([fragment1, fragment2], out8)
    # tf.compat.v1.Session().graph.finalize()
    return combo_net_model


def threshold_evaluate(x, y, threshold_rate=0.05):
    predict_y = model.predict(x)
    acc = 0
    for i in range(len(y)):
        cur_id = list(y[i]).index(True)
        if predict_y[i][cur_id] + threshold_rate > max(predict_y[i]):
            acc += 1
    return acc/len(y)


# if not os.path.exists('./cate_3_vgg16.h5'):
# if not os.path.exists('./PACS/cate_3_vgg_48gap.h5'):
if not os.path.exists('./cate_3_EfficientNetB0_9class_ft.h5'):
    model = combo_net([96, 96, 3])
    lr = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                 decay_steps=720,
                                                                 decay_rate=0.97,
                                                                 staircase=True)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, momentum=0.9, epsilon=1e-8)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=[['categorical_accuracy'], ['binary_accuracy']])

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max',
                                                   verbose=1, patience=10, restore_best_weights=True)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=cur_path + "VerificationCode/checkpoint/")

    model.fit(x=train_x, y=train_y, batch_size=200, epochs=500,
              callbacks=[es_callback],
              validation_data=(valid_x, valid_y))

    results = model.evaluate(test_x, test_y)
    print("test loss, test acc:", results)

    # model.save('./PACS/cate_3_vgg_48gap.h5')
    # model.save('./cate_3_vgg16.h5')
    model.save('./cate_3_EfficientNetB0_9class_ft.h5')
    

else:
    # model = tf.keras.models.load_model('./PACS/cate_3_vgg_48gap.h5')
    # model = tf.keras.models.load_model('./cate_3_vgg16.h5')
    # model = tf.keras.models.load_model('./cate_3_res50.h5')
    model = tf.keras.models.load_model('./cate_3_EfficientNetB0_9class_ft.h5')
    model.summary()
    results = model.evaluate(test_x, test_y)
    print("test loss, test acc:", results)
    # results = model.evaluate(painting_test_x, painting_test_y)
    # print("test loss, test acc:", results)
    # test_acc = threshold_evaluate(painting_test_x, painting_test_y)
    # print("test threshold acc:", test_acc)

    # results = model.evaluate(engraving_test_x, engraving_test_y)
    # print("test loss, test acc:", results)
    # test_acc = threshold_evaluate(engraving_test_x, engraving_test_y)
    # print("test threshold acc:", test_acc)

    # results = model.evaluate(artifact_test_x, artifact_test_y)
    # print("test loss, test acc:", results)
    # test_acc = threshold_evaluate(artifact_test_x, artifact_test_y)
    # print("test threshold acc:", test_acc)
