# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/6/10
# Fixed version for UAG with alternative embedding model
import argparse
import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

# 使用sentence-transformers作为替代嵌入模型
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False
    print("Warning: sentence-transformers not available, using dummy embeddings")


class UAGFixed:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # Initialize main model
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            cache_dir=os.environ.get('HF_HOME', None)
        )
        self.stop_token_id = self.tokenizer.eos_token_id

        # Load model with GPU optimization
        print(f"Loading model: {self.model_name}")
        if self.device == "cuda":
            # GPU模式：使用device_map自动分配
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=os.environ.get('HF_HOME', None),
                low_cpu_mem_usage=True
            )
        else:
            # CPU模式
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=None,
                cache_dir=os.environ.get('HF_HOME', None),
                low_cpu_mem_usage=True
            ).to(self.device)
        
        self.model.eval()

        # Initialize embedding model (使用替代方案)
        print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        if EMBEDDING_MODEL_AVAILABLE:
            try:
                # 设置缓存目录
                cache_folder = os.environ.get('HF_HOME', '/rds/general/user/js3623/home/.cache/huggingface')
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_folder)
            except Exception as e:
                print(f"Warning: Failed to load sentence-transformers model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            print("Warning: Using dummy embedding model")

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
        """Load input data"""
        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            # 检查文件格式：JSONL vs JSON
            first_line = f.readline().strip()
            f.seek(0)  # 重置文件指针
            
            if first_line.startswith('['):
                # JSON数组格式
                data = json.load(f)
                self.data = data
            else:
                # JSONL格式
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        self.data.append(item)
        print(f"Loaded {len(self.data)} samples from {self.data_path}")

    def load_metric(self):
        """Load evaluation metric"""
        from .metric import GSM8K_Metric, AQuA_Metric, CSQA_Metric, StrategyQA_Metric
        from .comprehensive_metrics import ComprehensiveMetrics

        if self.task == "GSM8K":
            self.metric = GSM8K_Metric()
        elif self.task == "AQuA":
            self.metric = AQuA_Metric()
        elif self.task == "CSQA":
            self.metric = CSQA_Metric()
        elif self.task == "StrategyQA":
            self.metric = StrategyQA_Metric()
        
        # Load comprehensive metrics
        self.comprehensive_metrics = ComprehensiveMetrics()

    def load_demonstrations(self):
        """Load demonstrations specific to the task"""
        from .prompt import GSM8K_Prompt, AQuA_Prompt, CSQA_Prompt, StrategyQA_Prompt
        from .prompt import GSM8K_Prompt_list, AQuA_Prompt_list, CSQA_Prompt_list, StrategyQA_Prompt_list

        self.demonstrations = []
        if self.task == "GSM8K":
            self.demonstration = GSM8K_Prompt
            for item in GSM8K_Prompt_list:
                demo = {"question": item["question"], "answer": item["answer"]}
                self.demonstrations.append(demo)
        elif self.task == "AQuA":
            self.demonstration = AQuA_Prompt
            for item in AQuA_Prompt_list:
                demo = {"question": item["question"], "answer": item["answer"]}
                self.demonstrations.append(demo)
        elif self.task == "CSQA":
            self.demonstration = CSQA_Prompt
            for item in CSQA_Prompt_list:
                demo = {"question": item["question"], "answer": item["answer"]}
                self.demonstrations.append(demo)
        elif self.task == "StrategyQA":
            self.demonstration = StrategyQA_Prompt
            for item in StrategyQA_Prompt_list:
                demo = {"question": item["question"], "answer": item["answer"]}
                self.demonstrations.append(demo)

        if self.demonstration_path is not None:
            with open(self.demonstration_path, "r", encoding="utf-8") as f:
                # 检查文件格式：JSONL vs JSON
                first_line = f.readline().strip()
                f.seek(0)  # 重置文件指针
                
                if first_line.startswith('['):
                    # JSON数组格式
                    data = json.load(f)
                    for item in data:
                        # 转换StrategyQA格式到演示格式
                        if self.task == "StrategyQA":
                            demo = {
                                "question": item["question"],
                                "answer": "yes" if item["answer"] else "no"
                            }
                        else:
                            demo = item
                        self.demonstrations.append(demo)
                else:
                    # JSONL格式
                    for line in f:
                        if line.strip():
                            demo = json.loads(line.strip())
                            self.demonstrations.append(demo)

        print(f"Loaded {len(self.demonstrations)} demonstrations for {self.task}")

    def demonstration_clustering(self):
        """Cluster demonstrations using K-means"""
        print("Clustering demonstrations...")
        
        # Compute embeddings for each demonstration
        demonstration_texts = []
        for demo in self.demonstrations:
            # 使用question和answer构建文本
            if "reasoning" in demo:
                text = demo["question"] + "\n" + demo["reasoning"]
            else:
                text = demo["question"] + "\n" + demo["answer"]
            demonstration_texts.append(text)

        # Use the embedding model
        if self.embedding_model is not None:
            embeddings = self.embedding_model.encode(demonstration_texts)
        else:
            # 使用随机嵌入作为fallback
            embeddings = np.random.randn(len(demonstration_texts), 384)
            print("Warning: Using random embeddings")

        # Cluster into k clusters
        k = min(self.k, len(embeddings))
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
            self.clustered_demonstrations = {0: [(demo, np.zeros(384)) for demo in self.demonstrations]}

        print(f"Created {len(self.clustered_demonstrations)} clusters")

    def uncertainty_identification(self, logits, next_token_id):
        """Compute uncertainty for the next token"""
        probs = F.softmax(logits, dim=-1)
        token_prob = probs[0, next_token_id].item()
        uncertainty = -np.log(token_prob + 1e-8)
        return uncertainty

    def compute_loss(self, context, target_text):
        """Compute the total negative log likelihood (loss) of target_text given context."""
        # Tokenize context and target text separately
        context_inputs = self.tokenizer(context, return_tensors="pt")
        target_inputs = self.tokenizer(target_text, return_tensors="pt")

        # Move to device
        if self.device == "cuda":
            context_inputs = {k: v.to(self.device) for k, v in context_inputs.items()}
            target_inputs = {k: v.to(self.device) for k, v in target_inputs.items()}

        # Concatenate context and target input_ids
        input_ids = torch.cat([context_inputs["input_ids"], target_inputs["input_ids"]], dim=1)

        # Create labels: -100 for context tokens, token ids for target tokens
        labels = input_ids.clone()
        labels[:, : context_inputs["input_ids"].size(1)] = -100

        # Get model output
        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
        # Total loss over target tokens
        loss = outputs.loss.item() * target_inputs["input_ids"].size(1)
        return loss

    def adaptive_reasoning_adjustment(self, question, reasoning_chain, m):
        """Adaptive reasoning adjustment"""
        self.step_token = "\n"
        adjusted_reasoning_chain = reasoning_chain
        loop = True
        loop_counter = 0

        while loop and loop_counter < self.max_loop:
            loop_counter += 1
            
            # Backtrack to the last occurrence of self.step_token
            last_step_index = adjusted_reasoning_chain.rfind(self.step_token)
            if last_step_index != -1:
                truncated_reasoning = adjusted_reasoning_chain[:last_step_index]
            else:
                truncated_reasoning = ""

            # Prepare query embedding using the truncated reasoning chain
            query_text = "Q: " + question + "\nA: Let's think step by step." + truncated_reasoning
            if self.embedding_model is not None:
                query_embedding = self.embedding_model.encode([query_text])[0]
            else:
                query_embedding = np.random.randn(384)

            # Initialize list to store best demonstrations per cluster
            best_demos_per_cluster = []

            # For each cluster, select top demonstrations based on cosine similarity
            for cluster_id, cluster_demos in self.clustered_demonstrations.items():
                similarities = []
                for demo, demo_embedding in cluster_demos:
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, demo_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(demo_embedding) + 1e-8)
                    similarities.append((similarity, demo))

                # Sort demos in the cluster by similarity
                similarities.sort(key=lambda x: x[0], reverse=True)
                num_to_select = max(1, int(self.demonstration_number / self.k))
                selected = similarities[:num_to_select]

                # Compute selection scores for selected demonstrations
                demo_scores = []
                for similarity, demo in selected:
                    # 构建演示文本，使用answer作为reasoning
                    if "reasoning" in demo:
                        demo_text = demo["question"] + "\n" + demo["reasoning"]
                    else:
                        demo_text = demo["question"] + "\n" + demo["answer"]

                    # Relevance score: log P(D | Q, r_{<=m})
                    context_relevance = "Q: " + question + "\nA: Let's think step by step." + truncated_reasoning
                    relevance_loss = self.compute_loss(context_relevance, demo_text)
                    relevance_score = -relevance_loss

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
                # 使用answer作为reasoning，如果没有reasoning字段
                if "reasoning" in demo:
                    demo_reasoning = demo["reasoning"]
                else:
                    demo_reasoning = demo["answer"]
                prompt = demo["question"] + "\n" + demo_reasoning + "\n" + "Q: " + question + "\nA: Let's think step by step." + truncated_reasoning

                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                if self.device == "cuda":
                    input_ids = input_ids.to(self.device)
                
                # Generate continuation from the model
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=getattr(self.args, 'max_new_tokens', 512),
                        temperature=self.temperature,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_ids = outputs.sequences[0][input_ids.shape[1] :]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Combine truncated reasoning and new generated text
                new_reasoning_chain = truncated_reasoning + generated_text

                # Compute uncertainties
                logits_stack = torch.stack(outputs.scores, dim=1).squeeze(0)
                logits_stack = logits_stack / self.temperature
                probs = F.softmax(logits_stack, dim=-1)
                token_probs = probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
                uncertainties = -torch.log(token_probs + 1e-8)
                delta_uncertainties = uncertainties[1:] - uncertainties[:-1]
                delta_uncertainties = torch.cat([uncertainties[:1], delta_uncertainties], dim=0)

                # Check if uncertainties satisfy the threshold condition
                if all(delta <= self.theta for delta in delta_uncertainties.cpu().numpy()):
                    adjusted_reasoning_chain = new_reasoning_chain
                    reasoning_extended = True
                    break
                else:
                    # Remove steps with high uncertainty
                    high_uncertainty_indices = torch.where(delta_uncertainties > self.theta)[0]
                    if len(high_uncertainty_indices) > 0:
                        high_uncertainty_index = high_uncertainty_indices[0].item()
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
                loop = False
            else:
                if not truncated_reasoning.strip() or loop_counter >= self.max_loop:
                    loop = False
                    # Use initial input to generate
                    input_text = self.demonstration.format(question) if self.demonstration else question
                    input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
                    if self.device == "cuda":
                        input_ids = input_ids.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            max_new_tokens=getattr(self.args, 'max_new_tokens', 512),
                            temperature=self.temperature,
                            do_sample=True,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    adjusted_reasoning_chain = self.tokenizer.decode(outputs[0][input_ids.shape[1] :], skip_special_tokens=True)
                else:
                    adjusted_reasoning_chain = truncated_reasoning

        return adjusted_reasoning_chain

    def process(self):
        """Process all data samples"""
        print(f"Processing {len(self.data)} samples...")
        outputs = []
        correct_list = []
        
        for i, item in enumerate(tqdm(self.data, desc="Processing")):
            question = item["question"]
            answer = item.get("answer", None)

            print(f"\n--- Sample {i+1}/{len(self.data)} ---")
            print(f"Question: {question[:100]}...")

            # Prepare initial input
            if self.demonstration:
                input_text = self.demonstration.get_prompt(question)
            else:
                input_text = question
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            if self.device == "cuda":
                input_ids = input_ids.to(self.device)

            # Generate the reasoning chain using model.generate
            with torch.no_grad():
                outputs_model = self.model.generate(
                    input_ids,
                    max_new_tokens=getattr(self.args, 'max_new_tokens', 512),
                    temperature=self.temperature,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_ids = outputs_model.sequences[0][input_ids.shape[1] :]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            reasoning_chain = generated_text

            print(f"Initial reasoning: {reasoning_chain[:200]}...")

            # Compute uncertainties
            logits_stack = torch.stack(outputs_model.scores, dim=1).squeeze(0)
            logits_stack = logits_stack / self.temperature
            probs = F.softmax(logits_stack, dim=-1)
            token_probs = probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
            uncertainties = -torch.log(token_probs + 1e-8)
            delta_uncertainties = uncertainties[1:] - uncertainties[:-1]
            delta_uncertainties = torch.cat([uncertainties[:1], delta_uncertainties], dim=0)

            # Check if uncertainties satisfy the condition
            if not all(delta <= self.theta for delta in delta_uncertainties.cpu().numpy()):
                print("High uncertainty detected, applying UAG...")
                # Find the position m where delta_uncertainty exceeds theta
                for idx, delta in enumerate(delta_uncertainties):
                    if delta > self.theta:
                        m = idx
                        break
                else:
                    m = len(delta_uncertainties) - 1
                reasoning_chain = self.adaptive_reasoning_adjustment(question, reasoning_chain, m)
                print(f"Adjusted reasoning: {reasoning_chain[:200]}...")

            # Remove any content after "Q: " if it exists
            if "Q: " in reasoning_chain:
                reasoning_chain = reasoning_chain[: reasoning_chain.index("Q: ")]

            # Ensure "the answer is" is present in the reasoning chain
            if "the answer is" not in reasoning_chain.lower():
                reasoning_chain += "\nThus, the answer is "
                new_input_text = input_text + reasoning_chain
                new_input_ids = self.tokenizer.encode(new_input_text, return_tensors="pt")
                if self.device == "cuda":
                    new_input_ids = new_input_ids.to(self.device)
                
                with torch.no_grad():
                    answer_output = self.model.generate(
                        new_input_ids,
                        max_new_tokens=getattr(self.args, 'max_new_tokens', 512),
                        temperature=self.temperature,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                answer_generated = self.tokenizer.decode(answer_output[0][new_input_ids.shape[1] :], skip_special_tokens=True)
                reasoning_chain += answer_generated.split("\n")[0]

            pred = self.metric.process_pred(reasoning_chain).strip()
            correct = self.metric.cal_acc(pred, answer)
            correct_list.append(correct)

            # Calculate comprehensive metrics
            comprehensive_result = self.comprehensive_metrics.evaluate_sample(
                prediction=pred,
                ground_truth=answer,
                reasoning=reasoning_chain,
                uncertainties=uncertainties.cpu().numpy().tolist()
            )

            print(f"Predicted: {pred}")
            print(f"Correct: {correct}")
            print(f"F1 Score: {comprehensive_result['f1_score']:.4f}")
            print(f"BLEU Score: {comprehensive_result['bleu_score']:.4f}")

            # Save output with comprehensive metrics
            output_item = {
                "question": question,
                "answer": answer,
                "reasoning_chain": reasoning_chain,
                "pred": pred,
                "correct": correct,
                "uncertainties": uncertainties.cpu().numpy().tolist(),
                "delta_uncertainties": delta_uncertainties.cpu().numpy().tolist(),
                "comprehensive_metrics": comprehensive_result
            }
            outputs.append(output_item)

        accuracy = sum(correct_list) / len(correct_list)
        print(f"\nTask: {self.task} Accuracy: {accuracy:.4f} ({sum(correct_list)}/{len(correct_list)})")

        # Calculate comprehensive evaluation metrics
        comprehensive_results = [output["comprehensive_metrics"] for output in outputs]
        aggregate_metrics = self.comprehensive_metrics.evaluate_batch(comprehensive_results)
        
        print(f"\n=== Comprehensive Evaluation Results ===")
        print(f"Exact Match Accuracy: {aggregate_metrics.get('exact_match_accuracy', 0):.4f}")
        print(f"F1 Score: {aggregate_metrics.get('f1_score_mean', 0):.4f} ± {aggregate_metrics.get('f1_score_std', 0):.4f}")
        print(f"BLEU Score: {aggregate_metrics.get('bleu_score_mean', 0):.4f} ± {aggregate_metrics.get('bleu_score_std', 0):.4f}")
        print(f"ROUGE-L: {aggregate_metrics.get('rouge_l_mean', 0):.4f} ± {aggregate_metrics.get('rouge_l_std', 0):.4f}")
        
        # Print reasoning quality metrics
        if 'reasoning_step_indicator_ratio_mean' in aggregate_metrics:
            print(f"\n=== Reasoning Quality Metrics ===")
            print(f"Step Indicator Ratio: {aggregate_metrics.get('reasoning_step_indicator_ratio_mean', 0):.4f}")
            print(f"Math Indicator Ratio: {aggregate_metrics.get('reasoning_math_indicator_ratio_mean', 0):.4f}")
            print(f"Logic Indicator Ratio: {aggregate_metrics.get('reasoning_logic_indicator_ratio_mean', 0):.4f}")
            print(f"Avg Sentence Length: {aggregate_metrics.get('reasoning_avg_sentence_length_mean', 0):.2f}")
        
        # Print uncertainty metrics
        if 'uncertainty_mean_uncertainty_mean' in aggregate_metrics:
            print(f"\n=== Uncertainty Metrics ===")
            print(f"Mean Uncertainty: {aggregate_metrics.get('uncertainty_mean_uncertainty_mean', 0):.4f}")
            print(f"Uncertainty Std: {aggregate_metrics.get('uncertainty_std_uncertainty_mean', 0):.4f}")
            print(f"Max Uncertainty: {aggregate_metrics.get('uncertainty_max_uncertainty_mean', 0):.4f}")

        # Write outputs to record path
        output_dir = os.path.dirname(self.record_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.record_path, "w", encoding="utf-8") as f:
            for output_item in outputs:
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
        
        # Save comprehensive metrics summary
        metrics_summary_path = self.record_path.replace('.jsonl', '_metrics_summary.json')
        with open(metrics_summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "task": self.task,
                "total_samples": len(outputs),
                "exact_match_accuracy": accuracy,
                "comprehensive_metrics": aggregate_metrics
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {self.record_path}")
        print(f"Metrics summary saved to: {metrics_summary_path}")

        return accuracy


def main():
    parser = argparse.ArgumentParser(description="Uncertainty-aware Adaptive Guidance (UAG) - Fixed Version")
    parser.add_argument("--task", type=str, default="GSM8K", help="Task name")
    parser.add_argument("--data-path", type=str, default="UAG_input.jsonl", help="Path to input data")
    parser.add_argument("--record-path", type=str, default="UAG_output.jsonl", help="Path to output record")
    parser.add_argument("--demonstration-path", type=str, default=None, help="Path to demonstration")
    parser.add_argument("--demonstration-number", type=int, default=16, help="Number of demonstrations")
    parser.add_argument("--theta", type=float, default=16, help="Uncertainty threshold")
    parser.add_argument("--lambda1", type=float, default=0.5, help="Weight for relevance score")
    parser.add_argument("--lambda2", type=float, default=0.5, help="Weight for originality score")
    parser.add_argument("--k", type=int, default=8, help="Number of clusters")
    parser.add_argument("--temperature", type=float, default=0.5, help="Generation temperature")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max-loop", type=int, default=10, help="Maximum loop times")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    uag = UAGFixed(args)
    uag.process()


if __name__ == "__main__":
    main()

