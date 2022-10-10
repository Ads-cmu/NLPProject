import spacy
from spacy.tokens import DocBin
import os
import sys

# Preprocessing First Step:
# get corpus of data
# for each document:
### Remove Section headers
### Remove "See also", "Notes", "References", "External Links", "Sources" and any other sections after "See Also"

path_dataset = "../HW01/documents"
folders = os.listdir(path_dataset)
removed_sections=[]
dataset=[]
for folder in folders:
    if(folder[0]=='.'):
        continue
    files = os.listdir(path_dataset+"/"+folder)
    for file in files:
        text = open(path_dataset+"/"+folder+"/"+file).read()
        #print(text)
        sections = text.split("==")  # break text into sections
        if(" See also " in sections):
            del_point = sections.index(" See also ")
        else:
            try:
                del_point = sections.index(" References ")
            except:
                print(text)
                sys.exit(0)

        for i in range(del_point, len(sections)):
            if((i-del_point)%2==0):
                removed_sections.append(sections[i])
        sections = sections[:del_point] #Remove all sections after references / See also
        del sections[1::2] #Remove Headers
        dataset = dataset+sections
for k in removed_sections:
    print(k.replace(" ","_").replace("=","").lower())

#for each line of each document
#get the lemma
#get the POS
#Create some displaCy visualisations of your preprocessing

nlp = spacy.load("en_core_web_sm")
'''
doc_bin = DocBin(store_user_data = True)
for para in nlp.pipe(dataset):
    doc_bin.add(para)
doc_bin.to_disk(path="dataset.spacy")
'''

doc_bin = DocBin().from_disk("dataset.spacy")
num_tokens=0
lemmas = set()
pos = dict()
ner = dict()
for doc in doc_bin.get_docs(nlp.vocab):
    num_tokens+=len(doc)
    for token in doc:
        lemmas.add(token.lemma_)
        if(token.pos_ in pos.keys()):
            pos[token.pos_]+=1
        else:
            pos[token.pos_]=1
        if(token.ent_type_ in ner.keys()):
            ner[token.ent_type_]+=1
        else:
            ner[token.ent_type_]=1
print("Number of tokens is ",num_tokens)
print("Number of lemmas is ",len(lemmas))
#for k,v in pos.items():
#    print(k,v)

#print("NER")
#for k,v in ner.items():
#    print(k,v)





