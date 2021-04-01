import kashgari
from kashgari.corpus import DataReader

test_x, test_y = DataReader.read_conll_format_file('./openbayes/input/input1/test_label_replaced.txt')
model = kashgari.utils.load_model('saved_ner_model')
print(test_x[:10])
print(model.predict(test_x[:10]))
