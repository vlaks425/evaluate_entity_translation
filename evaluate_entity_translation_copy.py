import re
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("de_core_news_lg",disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
with open("test.txt", "r", encoding="utf-8") as f:
    doc=f.read()
entity_list = []
reference_entities=[]
entity_count = 0
# 读取文本并匹配实体和对应的翻译
for text in doc.split("\n"):
    pattern = r"\[Entity-SRC\](.*?)\[Entity-MID\](.*?)\[Entity-TGT\]"
    matches = re.findall(pattern, text)
    if matches:
        entity_list.append(matches)
        entity_count += 1
        reference_entities.append([translation[1] for translation in matches])
    else:
        entity_list.append(None)
        reference_entities.append(None)
total_correct = 0
total_entities = 0

# 读取hypothesis文件并计算refercen_entities中不为None的每个句子中的entity的准确率
with open("/raid/lyu/en_de/wmt2017/results/pred.trg.tok", "r") as f:
    hypothesis_lines = f.readlines()
for i, line in enumerate(hypothesis_lines):
    doc = nlp(line)
    terms = reference_entities[i]
    if terms is None:
        continue
    patterns = [nlp.make_doc(text) for text in terms]
    if patterns:
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        matcher.add("EntityMatcher", patterns)
        matches = matcher(doc)
        correct = len(matches)
        matcher.remove("EntityMatcher")
    else:
        correct = 0
    num_entities = len(reference_entities[i])
    accuracy = correct / num_entities if num_entities else 0
    total_correct += correct
    total_entities += num_entities

total_accuracy = total_correct / total_entities if total_entities else 0
print(f"Total Accuracy: {total_accuracy}")