import spacy
from spacy.matcher import Matcher
import json

nlp = spacy.load("de_core_news_lg",disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

# Define terms to match
entity_class = ['ORG', 'CARDINAL',  'GPE', 'PERSON', 'MONEY', 'PRODUCT',  'WORK_OF_ART', 'NORP', 'LOC', 'EVENT', 'FAC', 'LAW', 'LANGUAGE', 'MISC']

with open("/raid/lyu/OpenNMT-py/docs/source/examples/wmt17_en_de/test.trg", "r") as f:
    reference_lines = f.readlines()

reference_entities = []
for line in reference_lines:
    doc = nlp(line)
    entities = [ent.text for ent in doc.ents if ent.label_ in entity_class]
    reference_entities.append(entities)

# 读取hypothesis文件并计算每个句子中的entity的准确率
with open("/raid/lyu/en_de/wmt2017/results/pred.trg.tok", "r") as f:
    hypothesis_lines = f.readlines()

total_correct = 0
total_entities = 0
for i, line in enumerate(hypothesis_lines):
    doc = nlp(line)
    patterns = [{"TEXT": {"FUZZY": {"IN":reference_entities[i]}}}]
    if patterns:
        matcher = Matcher(nlp.vocab)
        matcher.add("EntityMatcher",[patterns])
        matches = matcher(doc)
        correct = len(matches)
    else:
        correct = 0
    num_entities = len(reference_entities[i])
    accuracy = correct / num_entities if num_entities else 0
    total_correct += correct
    total_entities += num_entities

total_accuracy = total_correct / total_entities if total_entities else 0
print(f"Total Accuracy: {total_accuracy}")

