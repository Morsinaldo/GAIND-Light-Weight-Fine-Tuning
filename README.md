# Light Weight Fine Tuning

Light fine-tuning is one of the most important techniques for adapting foundation models, as it allows you to modify foundation models according to your needs without the need for substantial computational resources. In this project, parameter-efficient fine-tuning was applied using the Hugging Face peft library.

In general, brought together all the essential components of a PyTorch + Hugging Face training and inference process
1. Load a pre-trained model and evaluate its performance
2. Perform parameter-efficient fine tuning using the pre-trained model
3. Perform inference using the fine-tuned model and compare its performance to the original model

## Loading and Evaluating a Foundation Model

Nesta etapa, foi escolhido o SST-2 dataset para classificação de sentimentos e o Foundation Model selecionado foi o GPT-2. Na avaliação, obteve-se a seguintes métricas:

|           | Train       | Validation  |
|-----------|-------------|-------------|
| Label 0   | 44.22 %     | 49.08 %     |
| Label 1   | 55.78 %     | 50.92 %     |

O desempenho pode ser comparado à jogar uma moeda aleatoriamente.

## Performing Parameter-Efficient Fine-Tuning

Neste passo, primero carregou-se o modelo, criou-se a configuração para o LoRA e realizou-se o treinamento, de fato, do modelo por duas épocas. O código utilizado para isso encontra-se abaixo.

```python
# Load the pre-trained model 'gpt-2' for sequence classification.
model = AutoModelForSequenceClassification.from_pretrained('gpt2',
                                                      num_labels=2,
                                                      id2label={0: "NEGATIVE", 1: "POSITIVE"},
                                                      label2id={"NEGATIVE": 0, "POSITIVE": 1})

# Create a PEFT Config for LoRA
config = LoraConfig(
                    r=8, # Rank
                    lora_alpha=32,
                    target_modules=['c_attn', 'c_proj'],
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_CLS
                )

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

# Initialize the Trainer
trainer = Trainer(
    model=peft_model,  # Make sure to pass the PEFT model here
    args=TrainingArguments(
        output_dir="./lora_model_output",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',  # If you want to log metrics and/or losses during training
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512),
    compute_metrics=compute_metrics,
)

# Start the training process
trainer.train()
```

Ao executar esse código, obtiveram-se o seguinte resultado

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|-----------------|----------|
| 1     | 0.370800      | 0.299321        | 0.879587 |
| 2     | 0.348300      | 0.290560        | 0.891055 |

## Performing Inference with a PEFT Model

Ao calcular as métricas para o split de validação, obtiveram-se os seguintes resultados:

| Metric           | Value                 |
|------------------|-----------------------|
| eval_loss        | 0.29056042432785034  |
| eval_accuracy    | 0.8910550458715596   |
| eval_runtime     | 3.6751                |
| eval_samples_per_second | 237.272        |
| eval_steps_per_second   | 14.966         |

## How to execute

1 - Clone this repositoy

```bash
git clone https://github.com/Morsinaldo/GAIND-Light-Weight-Fine-Tuning.git
cd GAIND-Light-Weight-Fine-Tuning
```

2 - Create the virtual environment

```bash
conda env create -f environment.yml
conda activate peft
```

This step uses [Anaconda](https://www.anaconda.com/) as the environment manager, but feel free to use another one of your choice. You also can use `requirements.txt` file to install the necessary libraries.

3 - Run the [notebook](./LightweightFineTuning.ipynb) file.

**Important**: It is deeply recommended to use GPU to execute the code.

## References

[Generative AI NanoDegree](https://www.udacity.com/enrollment/nd608/1.0.14)