# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/6/10
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Commented out for M1 Mac compatibility
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm


class UAG:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # Initialize main model
        # self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Requires authorization
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.stop_token_id = self.tokenizer.eos_token_id

        # Load model with appropriate settings for different devices
        if self.device == "mps":
            # MPS works better with float16 for Apple Silicon
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        else:
            # CUDA or CPU
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

        # Initialize embedding model
        self.embedding_model_name = "nvidia/NV-Embed-v2"
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name, trust_remote_code=True)
        # Move embedding model to appropriate device
        if self.device != "cpu":
            self.embedding_model = self.embedding_model.to(self.device)

        # Set hyperparameters
        self.theta = args.theta
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.k = args.k
        self.temperature = args.temperature
        self.max_length = args.max_length
        self.max_loop = args.max_loop

        # Other initializations
        self.task = args.task
        self.data_path = args.data_path
        self.record_path = args.record_path
        self.demonstration_path = args.demonstration_path
        self.demonstration_number = args.demonstration_number

        # Load data and demonstrations
        self.load_data()
        self.load_demonstrations()
        self.load_metric()
        # Perform clustering on demonstrations
        self.demonstration_clustering()

    def load_data(self):
        # Load input data
        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def load_metric(self):
        from metric import GSM8K_Metric, AQuA_Metric, CSQA_Metric, StrategyQA_Metric

        if self.task == "GSM8K":
            self.metric = GSM8K_Metric()
        elif self.task == "AQuA":
            self.metric = AQuA_Metric()
        elif self.task == "CSQA":
            self.metric = CSQA_Metric()
        elif self.task == "StrategyQA":
            self.metric = StrategyQA_Metric()

    def load_demonstrations(self):
        # Load demonstrations specific to the task
        # For simplicity, we use a predefined prompt from prompt.py
        from prompt import GSM8K_Prompt, AQuA_Prompt, CSQA_Prompt, StrategyQA_Prompt
        from prompt import GSM8K_Prompt_list, AQuA_Prompt_list, CSQA_Prompt_list, StrategyQA_Prompt_list

        self.demonstrations = []
        if self.task == "GSM8K":
            self.demonstration = GSM8K_Prompt
            for item in GSM8K_Prompt_list:
                demo = {"question": "Q: " + item["question"], "reasoning": "A: " + item["reasoning"] + "\nThus, the answer is " + item["answer"], "answer": item["answer"][:-1]}
                self.demonstrations.append(demo)
        elif self.task == "AQuA":
            self.demonstration = AQuA_Prompt
            for item in AQuA_Prompt_list:
                demo = {"question": "Q: " + item["question"], "reasoning": "A: " + item["reasoning"] + "\nThus, the answer is " + item["answer"], "answer": item["answer"][:-1]}
                self.demonstrations.append(demo)
        elif self.task == "CSQA":
            self.demonstration = CSQA_Prompt
            for item in CSQA_Prompt_list:
                demo = {"question": "Q: " + item["question"], "reasoning": "A: " + item["reasoning"] + "\nThus, the answer is " + item["answer"], "answer": item["answer"][:-1]}
                self.demonstrations.append(demo)
        elif self.task == "StrategyQA":
            self.demonstration = StrategyQA_Prompt
            for item in StrategyQA_Prompt_list:
                demo = {"question": "Q: " + item["question"], "reasoning": "A: " + item["reasoning"] + "\nThus, the answer is " + item["answer"], "answer": item["answer"][:-1]}
                self.demonstrations.append(demo)

        if self.demonstration_path is not None:
            # Read demonstrations from the file at self.demonstration_path
            with open(self.demonstration_path, "r", encoding="utf-8") as f:
                for line in f:
                    demo = json.loads(line.strip())
                    self.demonstrations.append(demo)
                    demo = {"question": "Q: " + item["question"], "reasoning": "A: " + item["reasoning"], "answer": item["answer"]}
                    self.demonstrations.append(demo)


    def demonstration_clustering(self):
        # Compute embeddings for each demonstration
        demonstration_texts = []
        for demo in self.demonstrations:
            text = demo["question"] + "\n" + demo["reasoning"]
            demonstration_texts.append(text)

        embeddings = self.embedding_model.encode(demonstration_texts)

        # Cluster into k clusters
        k = min(self.k, len(embeddings))  # Ensure k does not exceed number of demonstrations
        if k > 0:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
            labels = kmeans.labels_

            # Group demonstrations into clusters
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[label].append((self.demonstrations[i], embeddings[i]))

            # For each cluster, sort demonstrations according to proximity to cluster centroid
            self.clustered_demonstrations = {}
            for i in range(k):
                cluster_demos = clusters[i]
                centroid = kmeans.cluster_centers_[i]
                # Compute distances to centroid
                distances = [(np.linalg.norm(embedding - centroid), (demo, embedding)) for demo, embedding in cluster_demos]
                # Sort demonstrations by distance
                sorted_demos_with_embeddings = [(demo, embedding) for dist, (demo, embedding) in sorted(distances, key=lambda x: x[0])]
                self.clustered_demonstrations[i] = sorted_demos_with_embeddings
        else:
            self.clustered_demonstrations = {0: self.demonstrations}

    def uncertainty_identification(self, logits, next_token_id):
        # Compute uncertainty for the next token
        probs = F.softmax(logits, dim=-1)
        token_prob = probs[0, next_token_id].item()
        uncertainty = -np.log(token_prob + 1e-8)
        return uncertainty

    def compute_loss(self, context, target_text):
        """
        Compute the total negative log likelihood (loss) of target_text given context.
        """
        # Tokenize context and target text separately
        context_inputs = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        target_inputs = self.tokenizer(target_text, return_tensors="pt").to(self.model.device)

        # Concatenate context and target input_ids
        input_ids = torch.cat([context_inputs["input_ids"], target_inputs["input_ids"]], dim=1)

        # Create labels: -100 for context tokens, token ids for target tokens
        labels = input_ids.clone()
        labels[:, : context_inputs["input_ids"].size(1)] = -100  # Ignore context tokens in loss computation

        # Get model output
        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
        # Total loss over target tokens
        loss = outputs.loss.item() * target_inputs["input_ids"].size(1)
        return loss

    def adaptive_reasoning_adjustment(self, question, reasoning_chain, m):
        self.step_token = "\n"  # Define the step token as newline
        adjusted_reasoning_chain = reasoning_chain
        loop = True
        loop_counter = 0  # Initialize loop counter

        while loop:
            # Increment loop counter
            loop_counter += 1
            # Backtrack to the last occurrence of self.step_token in the reasoning chain
            last_step_index = adjusted_reasoning_chain.rfind(self.step_token)
            if last_step_index != -1:
                truncated_reasoning = adjusted_reasoning_chain[:last_step_index]
            else:
                truncated_reasoning = ""

            # Prepare query embedding using the truncated reasoning chain
            query_text = "Q: " + question + "\nA: Let's think step by step." + truncated_reasoning
            query_embedding = self.embedding_model.encode([query_text])[0]

            # Initialize list to store best demonstrations per cluster
            best_demos_per_cluster = []

            # For each cluster, select top demonstrations based on cosine similarity
            for cluster_id, cluster_demos in self.clustered_demonstrations.items():
                similarities = []
                for demo, demo_embedding in cluster_demos:
                    # Compute cosine similarity between query_embedding and demo_embedding
                    similarity = np.dot(query_embedding, demo_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(demo_embedding) + 1e-8)
                    similarities.append((similarity, demo))

                # Sort demos in the cluster by similarity
                similarities.sort(key=lambda x: x[0], reverse=True)
                num_to_select = max(1, int(self.demonstration_number / self.k))
                selected = similarities[:num_to_select]

                # Compute selection scores for selected demonstrations
                demo_scores = []
                for similarity, demo in selected:
                    demo_text = demo["question"] + "\n" + demo["reasoning"]

                    # Relevance score: log P(D | Q, r_{<=m})
                    context_relevance = "Q: " + question + "\nA: Let's think step by step." + truncated_reasoning
                    relevance_loss = self.compute_loss(context_relevance, demo_text)
                    relevance_score = -relevance_loss  # Higher probability corresponds to higher score

                    # Originality score: -log P(D | Q)
                    context_originality = "Q: " + question
                    originality_loss = self.compute_loss(context_originality, demo_text)
                    originality_score = originality_loss

                    # Selection score as weighted sum
                    selection_score = self.lambda1 * relevance_score + self.lambda2 * originality_score

                    demo_scores.append((selection_score, demo))

                # Select the demonstration with the highest selection score in this cluster
                if demo_scores:
                    best_demo = max(demo_scores, key=lambda x: x[0])
                    best_demos_per_cluster.append(best_demo)

            # Sort the best demonstrations from all clusters by selection score
            best_demos_per_cluster.sort(key=lambda x: x[0], reverse=True)

            # Try using each demonstration to generate continuation
            reasoning_extended = False
            for selection_score, demo in best_demos_per_cluster:
                # Construct new prompt with demonstration and truncated reasoning chain
                prompt = demo["question"] + "\n" + demo["reasoning"] + "\n" + "Q: " + question + "\nA: Let's think step by step." + truncated_reasoning

                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                # Generate continuation from the model
                outputs = self.model.generate(
                    input_ids,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                generated_ids = outputs.sequences[0][input_ids.shape[1] :]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Combine truncated reasoning and new generated text
                new_reasoning_chain = truncated_reasoning + generated_text

                # Compute uncertainties without explicit loops
                logits_stack = torch.stack(outputs.scores, dim=1).squeeze(0)
                logits_stack = logits_stack / self.temperature
                probs = F.softmax(logits_stack, dim=-1)
                token_probs = probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
                uncertainties = -torch.log(token_probs + 1e-8)
                delta_uncertainties = uncertainties[1:] - uncertainties[:-1]
                delta_uncertainties = torch.cat([uncertainties[:1], delta_uncertainties], dim=0)

                # Check if uncertainties satisfy the threshold condition
                if all(delta <= self.theta for delta in delta_uncertainties.cpu().numpy()):
                    # Return the new reasoning chain if condition is satisfied
                    adjusted_reasoning_chain = new_reasoning_chain
                    reasoning_extended = True
                    break
                else:
                    # Remove steps with high uncertainty (back to last self.step_token)
                    high_uncertainty_indices = torch.where(delta_uncertainties > self.theta)[0]
                    if len(high_uncertainty_indices) > 0:
                        high_uncertainty_index = high_uncertainty_indices[0].item()
                        # Find the position of the last self.step_token before the high uncertainty token
                        generated_text_until_high_uncertainty = self.tokenizer.decode(generated_ids[:high_uncertainty_index], skip_special_tokens=True)
                        last_step_index = generated_text_until_high_uncertainty.rfind(self.step_token)
                        if last_step_index != -1:
                            generated_text_truncated = generated_text_until_high_uncertainty[:last_step_index]
                            adjusted_reasoning_chain = truncated_reasoning + generated_text_truncated
                        else:
                            adjusted_reasoning_chain = truncated_reasoning
                    else:
                        adjusted_reasoning_chain = truncated_reasoning

            if reasoning_extended:
                # Successfully extended the reasoning chain
                loop = False
            else:
                # If none of the demonstrations can help, exit the loop after max_loop times
                if not truncated_reasoning.strip() or loop_counter >= self.max_loop:
                    # No more reasoning steps to backtrack
                    loop = False
                    # Use initial input to generate
                    input_text = self.demonstration.format(question) if self.demonstration else question
                    input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
                    outputs = self.model.generate(
                        input_ids,
                        max_length=self.max_length,
                        temperature=self.temperature,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    adjusted_reasoning_chain = self.tokenizer.decode(outputs[0][input_ids.shape[1] :], skip_special_tokens=True)
                else:
                    # Remove the last reasoning step and try again
                    adjusted_reasoning_chain = truncated_reasoning

        return adjusted_reasoning_chain

    def compute_uncertainties(self, input_ids, generated_text):
        uncertainties = []
        delta_uncertainties = []
        tokens = self.tokenizer.encode(generated_text, add_special_tokens=False)
        past_key_values = None

        for idx, token_id in enumerate(tokens):
            outputs = self.model(input_ids, past_key_values=past_key_values, return_dict=True, use_cache=False)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            uncertainty = self.uncertainty_identification(logits, token_id)
            uncertainties.append(uncertainty)

            if idx > 0:
                delta_uncertainty = uncertainties[idx] - uncertainties[idx - 1]
            else:
                delta_uncertainty = uncertainties[0]

            delta_uncertainties.append(delta_uncertainty)
            input_ids = torch.tensor([[token_id]]).to(self.device)

        return uncertainties, delta_uncertainties

    def process(self):
        outputs = []
        correct_list = []
        for item in tqdm(self.data):
            question = item["question"]
            answer = item.get("answer", None)

            # Prepare initial input
            input_text = self.demonstration.format(question) if self.demonstration else question
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

            # Generate the reasoning chain using model.generate
            outputs_model = self.model.generate(
                input_ids,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

            generated_ids = outputs_model.sequences[0][input_ids.shape[1] :]  # Exclude the prompt
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            reasoning_chain = generated_text

            # Compute uncertainties and delta uncertainties without explicit loop
            # Stack logits into a tensor
            logits_stack = torch.stack(outputs_model.scores, dim=1).squeeze(0)  # Shape: [sequence_length, vocab_size]

            # Adjust logits for temperature
            logits_stack = logits_stack / self.temperature

            # Compute probabilities
            probs = F.softmax(logits_stack, dim=-1)  # Shape: [sequence_length, vocab_size]

            # Get the probabilities of the generated tokens
            token_probs = probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)  # Shape: [sequence_length]

            # Compute uncertainties
            uncertainties = -torch.log(token_probs + 1e-8)

            # Compute delta uncertainties
            delta_uncertainties = uncertainties[1:] - uncertainties[:-1]
            delta_uncertainties = torch.cat([uncertainties[:1], delta_uncertainties], dim=0)

            # Convert to lists
            uncertainties = uncertainties.cpu().numpy().tolist()
            delta_uncertainties = delta_uncertainties.cpu().numpy().tolist()

            # Check if uncertainties satisfy the condition
            if not all(delta <= self.theta for delta in delta_uncertainties):
                # Perform adaptive reasoning adjustment
                # Find the position m where delta_uncertainty exceeds theta
                for idx, delta in enumerate(delta_uncertainties):
                    if delta > self.theta:
                        m = idx
                        break
                else:
                    m = len(delta_uncertainties) - 1  # All deltas are below theta
                reasoning_chain = self.adaptive_reasoning_adjustment(question, reasoning_chain, m)

            # Remove any content after "Q: " if it exists
            if "Q: " in reasoning_chain:
                reasoning_chain = reasoning_chain[: reasoning_chain.index("Q: ")]

            # Ensure "the answer is" is present in the reasoning chain
            if "the answer is" not in reasoning_chain.lower():
                reasoning_chain += "\nThus, the answer is "
                # Generate the answer
                new_input_text = input_text + reasoning_chain
                new_input_ids = self.tokenizer.encode(new_input_text, return_tensors="pt").to(self.device)
                answer_output = self.model.generate(
                    new_input_ids,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                answer_generated = self.tokenizer.decode(answer_output[0][new_input_ids.shape[1] :], skip_special_tokens=True)
                # Append the generated answer to reasoning_chain
                reasoning_chain += answer_generated.split("\n")[0]  # Take only the first line

            pred = self.metric.process_pred(reasoning_chain).strip()
            correct = self.metric.cal_acc(pred, answer)
            correct_list.append(correct)

            # Save output
            output_item = {
                "question": question,
                "answer": answer,
                "reasoning_chain": reasoning_chain,
                "pred": pred,
                "correct": correct,
                "uncertainties": uncertainties,
                "delta_uncertainties": delta_uncertainties,
            }
            outputs.append(output_item)

        print("Task:{} Accuracy: {}".format(self.task, sum(correct_list) / len(correct_list)))

        # Write outputs to record path
        output_dir = os.path.dirname(self.record_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.record_path, "w", encoding="utf-8") as f:
            for output_item in outputs:
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Uncertainty-aware Adaptive Guidance (UAG)")
    parser.add_argument("--task", type=str, default="GSM8K", help="Task name")
    parser.add_argument(
        "--data-path",
        type=str,
        default="UAG_input.jsonl",
        help="Path to input data",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default="UAG_output.jsonl",
        help="Path to output record",
    )
    parser.add_argument(
        "--demonstration-path",
        type=str,
        default=None,
        help="Path to demonstration",
    )
    parser.add_argument(
        "--demonstration-number",
        type=int,
        default=16,
        help="Number of demonstrations (default: 16)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=16,
        help="Uncertainty threshold (default: 16)",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.5,
        help="Weight for relevance score (default: 0.5)",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.5,
        help="Weight for originality score (default: 0.5)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of clusters for demonstration clustering (default: 8)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Generation temperature (default: 0.5)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--max-loop",
        type=int,
        default=10,
        help="Maximum loop times (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to use (default: mps if available, cuda if available, else cpu)",
    )

    args = parser.parse_args()

    uag = UAG(args)
    uag.process()


if __name__ == "__main__":
    main()
