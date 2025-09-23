# -*- coding: utf-8 -*-
# Comprehensive evaluation metrics for UAG experiments

import re
import json
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Union
import math

class ComprehensiveMetrics:
    """Comprehensive evaluation metrics for UAG experiments"""
    
    def __init__(self):
        self.metrics = {}
    
    def exact_match(self, prediction: str, ground_truth: Union[str, bool]) -> int:
        """Exact match accuracy"""
        pred_clean = self._clean_answer(prediction)
        gt_clean = self._clean_answer(ground_truth)
        return 1 if pred_clean == gt_clean else 0
    
    def f1_score(self, prediction: str, ground_truth: Union[str, bool]) -> float:
        """F1 score based on token overlap"""
        pred_tokens = self._tokenize(prediction)
        gt_tokens = self._tokenize(str(ground_truth))
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        pred_counter = Counter(pred_tokens)
        gt_counter = Counter(gt_tokens)
        
        common = sum((pred_counter & gt_counter).values())
        precision = common / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = common / len(gt_tokens) if len(gt_tokens) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def bleu_score(self, prediction: str, ground_truth: Union[str, bool], n_gram: int = 4) -> float:
        """BLEU score for n-gram overlap"""
        pred_tokens = self._tokenize(prediction)
        gt_tokens = self._tokenize(str(ground_truth))
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        # Calculate precision for each n-gram
        precisions = []
        for n in range(1, n_gram + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            gt_ngrams = self._get_ngrams(gt_tokens, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            overlap = 0
            for ngram in pred_ngrams:
                if ngram in gt_ngrams:
                    overlap += 1
            
            precision = overlap / len(pred_ngrams)
            precisions.append(precision)
        
        # Calculate brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(gt_tokens):
            bp = math.exp(1 - len(gt_tokens) / len(pred_tokens))
        
        # Calculate BLEU score
        if all(p > 0 for p in precisions):
            bleu = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            bleu = 0.0
        
        return bleu
    
    def rouge_l(self, prediction: str, ground_truth: Union[str, bool]) -> float:
        """ROUGE-L score based on longest common subsequence"""
        pred_tokens = self._tokenize(prediction)
        gt_tokens = self._tokenize(str(ground_truth))
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        lcs_length = self._lcs_length(pred_tokens, gt_tokens)
        
        precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = lcs_length / len(gt_tokens) if len(gt_tokens) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def reasoning_quality_score(self, reasoning: str) -> Dict[str, float]:
        """Evaluate reasoning quality metrics"""
        reasoning_lower = reasoning.lower()
        
        # Step-by-step reasoning indicators
        step_indicators = ['step', 'first', 'second', 'third', 'next', 'then', 'finally', 'therefore', 'thus']
        step_count = sum(1 for indicator in step_indicators if indicator in reasoning_lower)
        
        # Mathematical reasoning indicators
        math_indicators = ['+', '-', '*', '/', '=', 'calculate', 'compute', 'solve', 'equation']
        math_count = sum(1 for indicator in math_indicators if indicator in reasoning)
        
        # Logical reasoning indicators
        logic_indicators = ['because', 'since', 'given', 'if', 'then', 'therefore', 'thus', 'so']
        logic_count = sum(1 for indicator in logic_indicators if indicator in reasoning_lower)
        
        # Length and complexity
        word_count = len(reasoning.split())
        sentence_count = len([s for s in reasoning.split('.') if s.strip()])
        
        return {
            'step_indicator_ratio': step_count / max(word_count, 1),
            'math_indicator_ratio': math_count / max(word_count, 1),
            'logic_indicator_ratio': logic_count / max(word_count, 1),
            'avg_sentence_length': word_count / max(sentence_count, 1),
            'reasoning_length': word_count
        }
    
    def uncertainty_metrics(self, uncertainties: List[float]) -> Dict[str, float]:
        """Calculate uncertainty-related metrics"""
        if not uncertainties:
            return {}
        
        uncertainties = np.array(uncertainties)
        
        return {
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties)),
            'min_uncertainty': float(np.min(uncertainties)),
            'uncertainty_entropy': float(-np.sum(uncertainties * np.log(uncertainties + 1e-8)))
        }
    
    def adaptive_adjustment_metrics(self, adjustment_count: int, total_samples: int) -> Dict[str, float]:
        """Calculate adaptive adjustment metrics"""
        return {
            'adjustment_rate': adjustment_count / max(total_samples, 1),
            'total_adjustments': adjustment_count
        }
    
    def _clean_answer(self, answer: Union[str, bool]) -> str:
        """Clean answer for comparison"""
        # Handle boolean answers (StrategyQA)
        if isinstance(answer, bool):
            return "yes" if answer else "no"
        
        # Convert to string if not already
        answer = str(answer)
        
        # Extract numerical answer from GSM8K format
        if '####' in answer:
            match = re.search(r'####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)', answer)
            if match:
                return match.group(1).replace(',', '')
        
        # Clean and normalize
        answer = answer.strip().lower()
        answer = re.sub(r'[^\w\s]', '', answer)
        return answer
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Get n-grams from tokens"""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def evaluate_sample(self, prediction: str, ground_truth: Union[str, bool], reasoning: str = "", 
                       uncertainties: List[float] = None) -> Dict[str, Any]:
        """Evaluate a single sample with comprehensive metrics"""
        result = {
            'exact_match': self.exact_match(prediction, ground_truth),
            'f1_score': self.f1_score(prediction, ground_truth),
            'bleu_score': self.bleu_score(prediction, ground_truth),
            'rouge_l': self.rouge_l(prediction, ground_truth)
        }
        
        if reasoning:
            result['reasoning_quality'] = self.reasoning_quality_score(reasoning)
        
        if uncertainties:
            result['uncertainty_metrics'] = self.uncertainty_metrics(uncertainties)
        
        return result
    
    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate a batch of results and return aggregate metrics"""
        if not results:
            return {}
        
        # Aggregate metrics
        exact_matches = [r.get('exact_match', 0) for r in results]
        f1_scores = [r.get('f1_score', 0) for r in results]
        bleu_scores = [r.get('bleu_score', 0) for r in results]
        rouge_scores = [r.get('rouge_l', 0) for r in results]
        
        # Reasoning quality metrics
        reasoning_qualities = [r.get('reasoning_quality', {}) for r in results if 'reasoning_quality' in r]
        
        # Uncertainty metrics
        uncertainty_metrics = [r.get('uncertainty_metrics', {}) for r in results if 'uncertainty_metrics' in r]
        
        aggregate = {
            'exact_match_accuracy': np.mean(exact_matches),
            'f1_score_mean': np.mean(f1_scores),
            'f1_score_std': np.std(f1_scores),
            'bleu_score_mean': np.mean(bleu_scores),
            'bleu_score_std': np.std(bleu_scores),
            'rouge_l_mean': np.mean(rouge_scores),
            'rouge_l_std': np.std(rouge_scores)
        }
        
        # Add reasoning quality aggregates
        if reasoning_qualities:
            for key in reasoning_qualities[0].keys():
                values = [rq[key] for rq in reasoning_qualities if key in rq]
                if values:
                    aggregate[f'reasoning_{key}_mean'] = np.mean(values)
                    aggregate[f'reasoning_{key}_std'] = np.std(values)
        
        # Add uncertainty aggregates
        if uncertainty_metrics:
            for key in uncertainty_metrics[0].keys():
                values = [um[key] for um in uncertainty_metrics if key in um]
                if values:
                    aggregate[f'uncertainty_{key}_mean'] = np.mean(values)
                    aggregate[f'uncertainty_{key}_std'] = np.std(values)
        
        return aggregate
