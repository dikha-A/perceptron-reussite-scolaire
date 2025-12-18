# -*- coding: utf-8 -*-


#  GPT-2 Fine-Tuning en Wolof avec deux jeux de données Hugging Face

# Étape 1 : Installation des librairies
!pip install -q transformers datasets accelerate
!pip install --upgrade datasets fsspec

#  Étape 2 : Importations
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

# Étape 3 : Charger les deux datasets
# Charger le dataset entier


galsen = load_dataset("galsenai/french-wolof-translation", split="train")

#  Étape 4 : Fusionner les phrases en wolof
all_wolof_sentences = []

# Si aya a cette structure, on fait ça :
for row in aya:
    if row.get("language") == "wol":
        text = row.get("inputs", "").strip()
        if text:
            all_wolof_sentences.append(text)

# Pour galsen, extraction comme avant
for row in galsen:
    if "translation" in row and "wolof" in row["translation"]:
        text = row["translation"]["wolof"].strip()
        if text:
            all_wolof_sentences.append(text)

print("Nombre total de phrases collectées :", len(all_wolof_sentences))
print("Exemples :", all_wolof_sentences[:5])

print (galsen[2546])

print(f"Nombre d'éléments :", len(galsen), len(aya))
print("Exemple de texte :", galsen[0])

#  Étape 5 : Sauvegarder dans un fichier corpus.txt
with open("corpus.txt", "w", encoding="utf-8") as f:
    for line in all_wolof_sentences:
        f.write(line + "\n")

with open("corpus.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(f"Nombre de lignes : {len(lines)}")
    print("Aperçu :", lines[:5])

#  Étape 6 : Charger le fichier texte comme dataset Hugging Face
from datasets import load_dataset
custom_dataset = load_dataset("text", data_files={"train": "corpus.txt"}, split="train[:90%]")
test_dataset = load_dataset("text", data_files={"train": "corpus.txt"}, split="train[90%:]")

#  Étape 7 : Préparer le modèle et le tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

#  Étape 8 : Tokenisation
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train = custom_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

#  Étape 9 : Préparation des arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./gpt2-wolof",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=1,
    logging_steps=100,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available()
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator
)

#  Étape 10 : Entraîner le modèle
trainer.train()

#  Étape 11 : Génération de texte
prompt = "wolof: louy bataxaal?"
"français: qu’est-ce que l’explication?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, top_k=50)
print(tokenizer.decode(outputs[0]))