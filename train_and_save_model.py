import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.corpus import DataReader
from kashgari.tasks.labeling import BiLSTM_CRF_Model
import matplotlib.pyplot as plt

# train_x, train_y = DataReader.read_conll_format_file('/openbayes/input/input1/train_label_replaced.txt')
# valid_x, valid_y = DataReader.read_conll_format_file('/openbayes/input/input1/validate_label_replaced.txt')
# test_x, test_y = DataReader.read_conll_format_file('/openbayes/input/input1/test_label_replaced.txt')

train_x, train_y = DataReader.read_conll_format_file('./corpus/train_label_replaced.txt')
valid_x, valid_y = DataReader.read_conll_format_file('./corpus/validate_label_replaced.txt')
test_x, test_y = DataReader.read_conll_format_file('./corpus/test_label_replaced.txt')

bert_embed = BERTEmbedding('./chinese_L-12_H-768_A-12',
                           task=kashgari.LABELING,
                           sequence_length=128)

# 还可以选择 `CNN_LSTM_Model`, `BiLSTM_Model`, `BiGRU_Model` 或 `BiGRU_CRF_Model`
model = BiLSTM_CRF_Model(bert_embed)
history = model.fit(train_x,
                    train_y,
                    x_validate=valid_x,
                    y_validate=valid_y,
                    epochs=30,
                    batch_size=512)

model.save('saved_ner_model')

# 绘制loss和acc图像
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history['crf_accuracy'])
plt.plot(range(epochs), history.history['crf_accuracy'], label='crf_accuracy')
plt.plot(range(epochs), history.history['val_crf_accuracy'], label='val_crf_accuracy')
plt.legend()
plt.savefig("loss_acc.png")
