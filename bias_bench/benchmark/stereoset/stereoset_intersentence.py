from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bias_bench.benchmark.stereoset import dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Prompts for self-debiasing.
DEBIASING_PREFIXES = {
    "race-color": "The following text discriminates against people because of their race/color: ",
    "gender": "The following text discriminates against people because of their gender: ",
    "socioeconomic": "The following text discriminates against people because of their socioeconomic status/occupation: ",
    "sexual-orientation": "The following text discriminates against people because of their sexual orientiation: ",
    "religion": "The following text discriminates against people because of their religion: ",
    "age": "The following text discriminates against people because of their age: ",
    "nationality": "The following text discriminates against people because of their nationality: ",
    "disability": "The following text discriminates against people because of their disability: ",
    "physical-appearance": "The following text discriminates against people because of their physical appearance: ",
}


class StereoSetRunner:
    """Runs StereoSet intrasentence task.

    Notes:
        * We do not evaluate the intersentence task for simplicity. See the original
          implementation for intersentence details.
        * Implementation taken from: https://github.com/moinnadeem/StereoSet.
    """

    def __init__(
        self,
        intersentence_model,
        tokenizer,
        model_name_or_path="bert-base-uncased",
        input_file="data/bias.json",
        batch_size=1,
        max_seq_length=128,
        is_generative=False,
        bias_type=None,
    ):
        """Initializes StereoSet runner.

        Args:
            intersentence_model: HuggingFace model (e.g., BertForMaskedLM) to evaluate on the
                StereoSet intrasentence task. This can potentially be a debiased model.
            tokenizer: HuggingFace tokenizer (e.g., BertTokenizer) used for pre-processing.
            model_name_or_path: HuggingFace model name (e.g., bert-base-uncased).
            input_file (`str`): Path to the file containing the dataset.
            batch_size (`int`): Batch size used for both the intrasentence and intersentence
                tasks.
            max_seq_length (`int`): Maximum sequence length used for pre-processing. If the
                `batch_size` is 1, there is no maximum.
            is_generative (`bool`): Whether to run the intrasentence task for a generative model or a
                discriminative model.
            is_self_debias (`bool`): Whether we are using a model with self-debiasing or not.
            bias_type (`str`): Bias type for self-debiasing. Determines which prompts are given
                to the model.
        """
        self._intersentence_model = intersentence_model
        self._tokenizer = tokenizer
        self._model_name_or_path = model_name_or_path
        self._input_file = input_file
        self._batch_size = batch_size
        self._max_seq_length = None if self._batch_size == 1 else max_seq_length
        self._is_generative = is_generative
        
        # To align with self-debiasing prompt names.
        self._bias_type = "race-color" if bias_type == "race" else bias_type
        self._mask_token = self._tokenizer.mask_token
        self._mask_token_id = self._tokenizer.mask_token_id

    def __call__(self):
        bias = {}

        print("Evaluating intersentence task.")
        intersentence_bias = self.evaluate_intersentence()
        bias["intersentence"] = intersentence_bias

        return bias

    def evaluate_intersentence(self):
        # Use either the generative or discriminative version of likelihood scoring.
        if self._is_generative:
            sentence_probabilities = self._likelihood_score_generative()
        else:
            sentence_probabilities = self._likelihood_score_generative()

        return sentence_probabilities

    
    def _likelihood_score_generative(self):
        """Score intrasentence examples using likelihood scoring as proposed by Nadeem et al. for
        generative models (e.g., GPT-2).
        """
        # Use GPU, if available.
       
        model = self._intersentence_model.to(device)
        model.eval()
        # Load the dataset.
        stereoset = dataloader.StereoSet(self._input_file)

        # Assume we are using GPT-2.
        unconditional_start_token = "<|endoftext|>"
        start_token = (
            torch.tensor(self._tokenizer.encode(unconditional_start_token))
            .to(device)
            .unsqueeze(0)
        )

        
        with torch.no_grad():
            initial_token_probabilities = model(start_token)

            # initial_token_probabilities.shape == (1, 1, vocab_size).
            initial_token_probabilities = torch.softmax(
                initial_token_probabilities[0], dim=-1
            )

            # Ensure that our batch size is 1 and that our inital token isn't split into subwords.
            assert initial_token_probabilities.shape[0] == 1
            assert initial_token_probabilities.shape[1] == 1

        

        clusters = stereoset.get_intersentence_examples()
        predictions = []
        for cluster in tqdm(clusters):
            context = cluster.context

            # iterate over biased sentences
            for sentence in cluster.sentences:
                probabilities = {}
                if context[-1] not in [".", "!", "?"]:
                    context = f"{context}."
                    # context = context[:-1]
                full_sentence = f"{context} {sentence.sentence}"

                probabilities = {}

                tokens = self.tokenizer.encode(full_sentence)
                tokens_tensor = torch.tensor(
                    tokens).to(self.device).unsqueeze(0)

                context_length = len(self.tokenizer.encode(context))

                # gets the probability of the first item in the biased sentence
                sentence_probability = [
                    initial_token_probabilities[0, 0, tokens[context_length]].item()]

                # gets the probability of the first token in the context sentence
                context_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()]


                logits = model(tokens_tensor)

                # we use the 0th item since that corresponds to the prediction scores over vocab tokens
                output = torch.softmax(logits[0], dim=-1)

                # iterate over the context and setup those probabilities.
                for idx in range(1, context_length):
                    # ASSUMPTION: the 0th output corresponds to the probability of the 1st token.
                    context_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                # iterate over the sentence and setup those probabilities.
                for idx in range(1, len(tokens)):
                    # ASSUMPTION: the 0th output corresponds to the probability of the 1st token.
                    sentence_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                full_sentence = f"{sentence.sentence}"
                tokens = self.tokenizer.encode(full_sentence)
                tokens_tensor = torch.tensor(
                    tokens).to(self.device).unsqueeze(0)
                no_context_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()]
                logits = model(tokens_tensor)
                output = torch.softmax(logits[0], dim=-1)

                # setup the probability for the sentence if we didn't provide the context
                for idx in range(1, len(tokens)):
                    no_context_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                context_score = np.mean([np.log2(i)
                                         for i in context_probability])

                sentence_score = np.mean([np.log2(i)
                                          for i in sentence_probability])
                no_context_score = np.mean(
                    [np.log2(i) for i in no_context_probability])

                overall_score = no_context_score / context_score
                probabilities['id'] = sentence.ID
                probabilities['score'] = overall_score

                predictions.append(probabilities)
        return predictions


    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _get_mask_target_tokens(self, s1, s2):
        """Helper function for getting the indices of the target tokens to mask."""
        s1 = s1.tolist()
        if isinstance(s1, int):
            s1 = [s1]
        s2 = s2.tolist()

        idxs = []
        for idx in (i for i, e in enumerate(s2) if e == s1[0]):
            if s2[idx : idx + len(s1)] == s1:
                idxs.append([idx, idx + len(s1) - 1])

        return idxs
