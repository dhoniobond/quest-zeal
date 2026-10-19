import os
import toolz
import time
import string
import re
import pytorch_lightning as pl
import tqdm.notebook as tq
import json
import pandas as pd
import numpy as np
import torch
import textwrap
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize
from tqdm.notebook import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sense2vec import Sense2Vec
from collections import OrderedDict
from typing import List
from logging import NullHandler
# Fine-tuned checkpoints and sense2vec vectors (not committed); see README
MODEL_DIR = os.environ.get("QUESTZEAL_MODEL_DIR", "ml_models")

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
)


def remove_duplicates(items: List[str]) -> List[str]:
    unique_items = []
    normalized_unique_items = []

    for item in items:
        normalized_item = _normalize_item(item)

        if normalized_item not in normalized_unique_items:
            unique_items.append(item)
            normalized_unique_items.append(normalized_item)

    return unique_items


def remove_distractors_duplicate_with_correct_answer(
    correct: str, distractors: List[str]
) -> List[str]:
    for distractor in distractors:
        if _normalize_item(correct) == _normalize_item(distractor):
            distractors.remove(distractor)

    return distractors


def _get_most_distinct_from_key(key: str, items: List[str]) -> List[str]:
    # TODO: This seems not as useful. For example "the family Phascolarctidae" and "the family Vombatidae" are close, but good distractors.

    return items


def _get_most_distinct_from_each_other():
    # TODO
    # calculate bleu for each with each.
    # find the most similar pair
    # remove the second in the original list (assuming the list comes ordered by better)
    # run until you get the desired amount
    pass


def _normalize_item(item) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(item))))


def _calculate_nltk_bleu(references: List[str], hypothesis: str, bleu_n: int = 1):
    if hypothesis == "":
        return 0, 0, 0, 0

    # Word tokenize
    refs_tokenized = list(map(lambda x: word_tokenize(x), references))
    hyp_tokenized = word_tokenize(hypothesis)

    # Smoothing function to avoid the cases where it resuts 1.0 in the cases when // Corpus/Sentence contains 0 counts of 2-gram overlaps. BLEU scores might be undesirable; use SmoothingFunction() //
    chencherry = SmoothingFunction()
    bleu = 0

    if bleu_n == 1:
        bleu = sentence_bleu(
            refs_tokenized,
            hyp_tokenized,
            weights=(1, 0, 0, 0),
            smoothing_function=chencherry.method2,
        )
    elif bleu_n == 2:
        bleu = sentence_bleu(
            refs_tokenized,
            hyp_tokenized,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=chencherry.method2,
        )
    elif bleu_n == 3:
        bleu = sentence_bleu(
            refs_tokenized,
            hyp_tokenized,
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=chencherry.method2,
        )
    elif bleu_n == 4:
        bleu = sentence_bleu(
            refs_tokenized,
            hyp_tokenized,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=chencherry.method2,
        )

    return bleu


def clean_text(text: str) -> str:

    cleaned_text = _remove_brackets(text)
    cleaned_text = _remove_square_brackets(cleaned_text)
    cleaned_text = _remove_multiple_spaces(cleaned_text)
    cleaned_text = _replace_weird_hyphen(cleaned_text)

    return cleaned_text


def _remove_brackets(text: str) -> str:

    return re.sub(r"\((.*?)\)", lambda L: "", text)


def _remove_square_brackets(text: str) -> str:

    return re.sub(r"\[(.*?)\]", lambda L: "", text)


def _remove_multiple_spaces(text: str) -> str:

    return re.sub(" +", " ", text)


def _replace_weird_hyphen(text: str) -> str:

    return text.replace("–", "-")


# ANSWER GENERATION

MODEL_NAME = "t5-small"
SOURCE_MAX_TOKEN_LEN = 64
TARGET_MAX_TOKEN_LEN = 24
LEARNING_RATE = 0.0001


class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME, return_dict=True
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)


# QUESTION GENERATION

# Constants
MODEL_NAME = "t5-small"
LEARNING_RATE = 0.0001
SOURCE_MAX_TOKEN_LEN = 512
TARGET_MAX_TOKEN_LEN = 64
SEP_TOKEN = "<sep>"
TOKENIZER_LEN = 32101  # after adding the new <sep> token

# Model
class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME, return_dict=True
        )
        self.model.resize_token_embeddings(
            TOKENIZER_LEN
        )  # resizing after adding new tokens to the tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)


class DistractorGenerator:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

        self.tokenizer.add_tokens(SEP_TOKEN)

        self.tokenizer_len = len(self.tokenizer)

        checkpoint_path = os.path.join(MODEL_DIR, "race-distractors.ckpt")
        self.dg_model = QGModel.load_from_checkpoint(checkpoint_path)
        self.dg_model.freeze()
        self.dg_model.eval()

    def generate(
        self, generate_count: int, correct: str, question: str, context: str
    ) -> List[str]:

        generate_triples_count = (
            int(generate_count / 3) + 1
        )  # since this model generates 3 distractors per generation

        model_output = self._model_predict(
            generate_triples_count, correct, question, context
        )

        cleaned_result = model_output.replace("<pad>", "").replace("</s>", "<sep>")
        cleaned_result = self._replace_all_extra_id(cleaned_result)
        distractors = cleaned_result.split("<sep>")[:-1]
        distractors = [
            x.translate(str.maketrans("", "", string.punctuation)) for x in distractors
        ]
        distractors = list(map(lambda x: x.strip(), distractors))

        return distractors

    def _model_predict(
        self, generate_count: int, correct: str, question: str, context: str
    ) -> str:
        source_encoding = self.tokenizer(
            "{} {} {} {} {}".format(correct, SEP_TOKEN, question, SEP_TOKEN, context),
            max_length=SOURCE_MAX_TOKEN_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        generated_ids = self.dg_model.model.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            num_beams=generate_count,
            num_return_sequences=generate_count,
            max_length=TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )

        preds = {
            self.tokenizer.decode(
                generated_id,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            for generated_id in generated_ids
        }

        return "".join(preds)

    def _correct_index_of(self, text: str, substring: str, start_index: int = 0):
        try:
            index = text.index(substring, start_index)
        except ValueError:
            index = -1

        return index

    def _replace_all_extra_id(self, text: str):
        new_text = text
        start_index_of_extra_id = 0

        while self._correct_index_of(new_text, "<extra_id_") >= 0:
            start_index_of_extra_id = self._correct_index_of(
                new_text, "<extra_id_", start_index_of_extra_id
            )
            end_index_of_extra_id = self._correct_index_of(
                new_text, ">", start_index_of_extra_id
            )

            new_text = (
                new_text[:start_index_of_extra_id]
                + "<sep>"
                + new_text[end_index_of_extra_id + 1 :]
            )

        return new_text


# DISTRACTOR GENERATION

# Constants
MODEL_NAME = "t5-small"
LEARNING_RATE = 0.0001
SOURCE_MAX_TOKEN_LEN = 300
TARGET_MAX_TOKEN_LEN = 80
SEP_TOKEN = "<sep>"
TOKENIZER_LEN = 32101  # after adding the new <sep> token

# QG Model
class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME, return_dict=True
        )
        self.model.resize_token_embeddings(
            TOKENIZER_LEN
        )  # resizing after adding new tokens to the tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)


class QuestionGenerator:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        # print('tokenizer len before: ', len(self.tokenizer))
        self.tokenizer.add_tokens(SEP_TOKEN)
        # print('tokenizer len after: ', len(self.tokenizer))
        self.tokenizer_len = len(self.tokenizer)

        checkpoint_path = os.path.join(MODEL_DIR, "multitask-qg-ag.ckpt")
        self.qg_model = QGModel.load_from_checkpoint(checkpoint_path)
        self.qg_model.freeze()
        self.qg_model.eval()

    def generate(self, answer: str, context: str) -> str:
        model_output = self._model_predict(answer, context)

        generated_answer, generated_question = model_output.split("<sep>")

        return generated_question

    def generate_qna(self, context: str) -> Tuple[str, str]:
        answer_mask = "[MASK]"
        model_output = self._model_predict(answer_mask, context)

        qna_pair = model_output.split("<sep>")

        if len(qna_pair) < 2:
            generated_answer = ""
            generated_question = qna_pair[0]
        else:
            generated_answer = qna_pair[0]
            generated_question = qna_pair[1]

        return generated_answer, generated_question

    def _model_predict(self, answer: str, context: str) -> str:
        source_encoding = self.tokenizer(
            "{} {} {}".format(answer, SEP_TOKEN, context),
            max_length=SOURCE_MAX_TOKEN_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        generated_ids = self.qg_model.model.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            num_beams=16,
            max_length=TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )

        preds = {
            self.tokenizer.decode(
                generated_id,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for generated_id in generated_ids
        }

        return "".join(preds)


# SENSE2VEC DISTRACTOR GENERATION


class Sense2VecDistractorGeneration:
    def __init__(self):
        self.s2v = Sense2Vec().from_disk(os.path.join(MODEL_DIR, "s2v_old"))

    def generate(self, answer: str, desired_count: int) -> List[str]:
        distractors = []
        answer = answer.lower()
        answer = answer.replace(" ", "_")

        sense = self.s2v.get_best_sense(answer)

        if not sense:
            return []

        most_similar = self.s2v.most_similar(sense, n=desired_count)

        for phrase in most_similar:
            normalized_phrase = phrase[0].split("|")[0].replace("_", " ").lower()

            if (
                normalized_phrase.lower() != answer
            ):  # TODO: compare the stem of the words (e.g. wrote, writing)
                distractors.append(normalized_phrase.capitalize())

        return list(OrderedDict.fromkeys(distractors))


class Question:
    def __init__(
        self, answerText: str, questionText: str = "", distractors: List[str] = []
    ):
        self.answerText = answerText
        self.questionText = questionText
        self.distractors = distractors


# MCQ GENERATOR


class MCQGenerator:
    def __init__(self, is_verbose=False):
        start_time = time.perf_counter()
        print("Loading ML Models...")

        self.question_generator = QuestionGenerator()
        print(
            "Loaded QuestionGenerator in",
            round(time.perf_counter() - start_time, 2),
            "seconds.",
        ) if is_verbose else ""

        self.distractor_generator = DistractorGenerator()
        print(
            "Loaded DistractorGenerator in",
            round(time.perf_counter() - start_time, 2),
            "seconds.",
        ) if is_verbose else ""

        self.sense2vec_distractor_generator = Sense2VecDistractorGeneration()
        print(
            "Loaded Sense2VecDistractorGenerator in",
            round(time.perf_counter() - start_time, 2),
            "seconds.",
        ) if is_verbose else ""

    # Main function
    def generate_mcq_questions(
        self, context: str, desired_count: int
    ) -> List[Question]:
        cleaned_text = clean_text(context)

        questions = self._generate_question_answer_pairs(cleaned_text, desired_count)
        questions = self._generate_distractors(cleaned_text, questions)

        # for question in questions:
        #     print("-------------------")
        #     print("Question: ")
        #     print(question.questionText)
        #     print("Answer: ")
        #     print(question.answerText)
        #     print()
        #     print("Distractors: ")
        #     for distractor in question.distractors:
        #         print(distractor)

        return questions

    def _generate_answers(self, context: str, desired_count: int) -> List[Question]:

        answers = self._generate_multiple_answers_according_to_desired_count(
            context, desired_count
        )

        print(answers)
        unique_answers = remove_duplicates(answers)

        questions = []
        for answer in unique_answers:
            questions.append(Question(answer))

        return questions

    def _generate_questions(
        self, context: str, questions: List[Question]
    ) -> List[Question]:
        for question in questions:
            question.questionText = self.question_generator.generate(
                question.answerText, context
            )

        return questions

    def _generate_question_answer_pairs(
        self, context: str, desired_count: int
    ) -> List[Question]:
        context_splits = self._split_context_according_to_desired_count(
            context, desired_count
        )

        questions = []

        for split in context_splits:
            answer, question = self.question_generator.generate_qna(split)
            questions.append(Question(answer.capitalize(), question))

        questions = list(toolz.unique(questions, key=lambda x: x.answerText))

        return questions

    def _generate_distractors(
        self, context: str, questions: List[Question]
    ) -> List[Question]:
        for question in questions:
            t5_distractors = self.distractor_generator.generate(
                5, question.answerText, question.questionText, context
            )

            if len(t5_distractors) < 3:
                s2v_distractors = self.sense2vec_distractor_generator.generate(
                    question.answerText, 3
                )
                distractors = t5_distractors + s2v_distractors
            else:
                distractors = t5_distractors

            distractors = remove_duplicates(distractors)
            distractors = remove_distractors_duplicate_with_correct_answer(
                question.answerText, distractors
            )
            # TODO - filter distractors having a similar bleu score with another distractor

            question.distractors = distractors

        return questions

    def _generate_answer_for_each_sentence(self, context: str) -> List[str]:
        sents = sent_tokenize(context)

        answers = []
        for sent in sents:
            answers.append(self.answer_generator.generate(sent, 1)[0])

        return answers

    def _split_context_according_to_desired_count(
        self, context: str, desired_count: int
    ) -> List[str]:
        sents = sent_tokenize(context)
        sent_ratio = len(sents) / desired_count  # Number of tokens /  = 10/2 = 5

        context_splits = []

        if sent_ratio < 1:  # When tokens are less compared to desired questions
            return sents
        else:
            take_sents_count = int(sent_ratio + 1)

            start_sent_index = 0

            while start_sent_index < len(sents):
                context_split = " ".join(
                    sents[start_sent_index : start_sent_index + take_sents_count]
                )
                context_splits.append(context_split)
                start_sent_index += take_sents_count - 1

        return context_splits


MCQ_Generator = MCQGenerator(True)

# Questions = MCQ_Generator.generate_mcq_questions(content, 10)