from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl


def tokenize(text: str):
    """
    Tokenize text into token sequence
    Args:
        text: target text sample

    Returns:
        List of tokens in this sample
    """
    result = []
    for s in text:
        result.append(s)
    return result


# transform train
dataset = read_jsonl(filepath='corpus/train.jsonl', dataset=NERDataset, encoding='utf-8')
conll = dataset.to_conll2003(tokenizer=tokenize)

# persist
with open('./corpus/train.txt', 'w') as f:
    for item in conll:
        f.write(item['data'])

# transform validate
dataset = read_jsonl(filepath='./corpus/validate.jsonl', dataset=NERDataset, encoding='utf-8')
conll = dataset.to_conll2003(tokenizer=tokenize)

# persist
with open('./corpus/validate.txt', 'w') as f:
    for item in conll:
        f.write(item['data'])


# transform validate
dataset = read_jsonl(filepath='./corpus/test.jsonl', dataset=NERDataset, encoding='utf-8')
conll = dataset.to_conll2003(tokenizer=tokenize)

# persist
with open('./corpus/test.txt', 'w') as f:
    for item in conll:
        f.write(item['data'])

# label mapping replace
"""
招标单位 -->  ZBD
项目名称 -->  XMM
项目金额 -->  XMJ
联系人  -->   LXR
联系电话 -->  LXD
所在地区 -->  SZD
招标时间 -->  ZBS
代理机构 -->  DLJ
项目编号 -->  XMB
"""

infile = open("./corpus/train.txt", "r", encoding='utf-8')
outfile = open("./corpus/train_label_replaced.txt", "w", encoding='utf-8')
for line in infile:
    line = line.replace('招标单位', 'ZBD')
    line = line.replace('项目名称', 'XMM')
    line = line.replace('项目金额', 'XMJ')
    line = line.replace('联系人', 'LXR')
    line = line.replace('联系电话', 'LXD')
    line = line.replace('所在地区', 'SZD')
    line = line.replace('招标时间', 'ZBS')
    line = line.replace('代理机构', 'DLJ')
    line = line.replace('项目编号', 'XMB')
    outfile.write(line)
infile.close()
outfile.close()
infile = open("./corpus/validate.txt", "r", encoding='utf-8')
outfile = open("./corpus/validate_label_replaced.txt", "w", encoding='utf-8')
for line in infile:
    line = line.replace('招标单位', 'ZBD')
    line = line.replace('项目名称', 'XMM')
    line = line.replace('项目金额', 'XMJ')
    line = line.replace('联系人', 'LXR')
    line = line.replace('联系电话', 'LXD')
    line = line.replace('所在地区', 'SZD')
    line = line.replace('招标时间', 'ZBS')
    line = line.replace('代理机构', 'DLJ')
    line = line.replace('项目编号', 'XMB')
    outfile.write(line)
infile.close()
outfile.close()
infile = open("./corpus/test.txt", "r", encoding='utf-8')
outfile = open("./corpus/test_label_replaced.txt", "w", encoding='utf-8')
for line in infile:
    line = line.replace('招标单位', 'ZBD')
    line = line.replace('项目名称', 'XMM')
    line = line.replace('项目金额', 'XMJ')
    line = line.replace('联系人', 'LXR')
    line = line.replace('联系电话', 'LXD')
    line = line.replace('所在地区', 'SZD')
    line = line.replace('招标时间', 'ZBS')
    line = line.replace('代理机构', 'DLJ')
    line = line.replace('项目编号', 'XMB')
    outfile.write(line)
infile.close()
outfile.close()
