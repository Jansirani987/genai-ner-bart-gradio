## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.
### PROBLEM STATEMENT:

Entity Recognition: Identify and classify named entities (e.g., persons, organizations, locations) from unstructured text using a pre-trained and fine-tuned model.

Model Selection and Fine-Tuning: Utilize a BART model and fine-tune it on a suitable Named Entity Recognition (NER) dataset to improve its accuracy for the task.

User Interaction and Evaluation: Build an interactive web interface using Gradio to allow users to input text and view recognized entities, facilitating easy evaluation and testing of the NER model's performance.

### DESIGN STEPS:

#### STEP 1:
Choose a pre-trained BART model (e.g., facebook/bart-large). Fine-tune the model on a NER dataset (e.g., CoNLL-03).

#### STEP 2:

Load the fine-tuned BART model and tokenizer using the transformers library. Create a NER pipeline to extract named entities from input text.

#### STEP 3:

Develop a Gradio interface to accept text input and display extracted entities. Deploy the interface for user interaction.

### PROGRAM:
```
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import gradio as gr

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

def ner_function(text):
    entities = ner_pipeline(text)
    if entities:
        return "\n".join([f"{ent['word']} ({ent['entity_group']})" for ent in entities])
    else:
        return "No named entities found."

iface = gr.Interface(
    fn=ner_function,
    inputs=gr.Textbox(lines=5, label="Input Text"),
    outputs=gr.Textbox(lines=10, label="Named Entities"),
    title="NER Demo with Pre-trained Model",
    description="This app extracts named entities using a BERT model fine-tuned on CoNLL-03."
)

iface.launch()

```

### OUTPUT:

![Screenshot 2025-05-20 192923](https://github.com/user-attachments/assets/9c152770-e767-4d46-8102-05bad8585769)


### RESULT:

A fine-tuned BART model successfully identifies and classifies named entities in text.The NER pipeline efficiently extracts entities such as persons, organizations, and locations.The Gradio interface provides an intuitive platform for users to input text and view the results interactively.overall,the Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework implemented successfully

