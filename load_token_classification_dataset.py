from task import Task

from datasets import load_dataset

def tokenize_token_classification_dataset(raw_datasets,
tokenizer, task_id, label_list, text_column_name,
label_column_name, data_args, training_args):
    label_to_id = {i: i for i in range(len(label_list))}

    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if(label.startswith("B-") and label.replace("B-","I-") in label_list):
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples[text_column_name],
            padding=padding,
            truncation = True,
            max_length = data_args.max_seq_length,
            is_split_into_words = True)
        labels = []
        for i,label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["task_ids"] = [task_id] * len(tokenized_inputs["labels"])
        return tokenized_inputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        col_to_remove = ["chunk_tags", "id", "ner_tags", "pos_tags", "tokens"]

        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
        )

    return tokenized_datasets

def load_token_classification_dataset(task_id, tokenizer, data_args, training_args):

    dataset_name = "conll2003"
    raw_datasets = load_dataset(dataset_name)

    text_column_name = "tokens"
    label_column_name = "ner_tags"

    label_list = raw_datasets["train"].features[label_column_name].feature.names
    num_labels = len(label_list)

    tokenized_datasets = tokenize_token_classification_dataset(
        raw_datasets,
        tokenizer,
        task_id,
        label_list,
        text_column_name,
        label_column_name,
        data_args,
        training_args,
    )

    task_info = Task(
        id=task_id,
        name=dataset_name,
        num_labels=num_labels,
        type="token_classification",
    )

    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        task_info,
    )