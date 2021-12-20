"""
End part of:
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb
"""

## Hyperparameter search
#
# pip install optuna OR pip install ray[tune] is needed to do this!
#
# így kell, mert többször kell :)
#def model_init():
#    return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
#
#trainer = Trainer(
#    model_init=model_init,
#    args=args,
#    train_dataset=tokenized_datasets["train"],
#    eval_dataset=tokenized_datasets["validation"],
#    tokenizer=tokenizer,
#    compute_metrics=compute_metrics
#)
#
## can take a long time!!! -- try with smaller data
#best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
#
#print(best_run)
#
## To reproduce the best training,
## just set the hyperparameters in your `TrainingArgument`
## before creating a `Trainer`
#for n, v in best_run.hyperparameters.items():
#    setattr(trainer.args, n, v)
#
#trainer.train()

