# -*- coding: utf-8 -*-
# Prompt templates for different tasks

class GSM8K_Prompt:
    @staticmethod
    def get_prompt(question, demonstrations=None):
        if demonstrations:
            demo_text = "\n\n".join([
                f"Q: {demo['question']}\nA: {demo['answer']}"
                for demo in demonstrations
            ])
            return f"{demo_text}\n\nQ: {question}\nA: Let's think step by step."
        else:
            return f"Q: {question}\nA: Let's think step by step."

class AQuA_Prompt:
    @staticmethod
    def get_prompt(question, options, demonstrations=None):
        options_text = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        
        if demonstrations:
            demo_text = "\n\n".join([
                f"Q: {demo['question']}\nOptions:\n{demo['options']}\nA: {demo['answer']}"
                for demo in demonstrations
            ])
            return f"{demo_text}\n\nQ: {question}\nOptions:\n{options_text}\nA: Let's think step by step."
        else:
            return f"Q: {question}\nOptions:\n{options_text}\nA: Let's think step by step."

class CSQA_Prompt:
    @staticmethod
    def get_prompt(question, choices, demonstrations=None):
        choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        
        if demonstrations:
            demo_text = "\n\n".join([
                f"Q: {demo['question']}\nChoices:\n{demo['choices']}\nA: {demo['answer']}"
                for demo in demonstrations
            ])
            return f"{demo_text}\n\nQ: {question}\nChoices:\n{choices_text}\nA: Let's think step by step."
        else:
            return f"Q: {question}\nChoices:\n{choices_text}\nA: Let's think step by step."

class StrategyQA_Prompt:
    @staticmethod
    def get_prompt(question, demonstrations=None):
        if demonstrations:
            demo_text = "\n\n".join([
                f"Q: {demo['question']}\nA: {demo['answer']}"
                for demo in demonstrations
            ])
            return f"{demo_text}\n\nQ: {question}\nA: Let's think step by step."
        else:
            return f"Q: {question}\nA: Let's think step by step."

# Prompt lists for demonstrations
GSM8K_Prompt_list = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "reasoning": "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May.",
        "answer": "72"
    },
    {
        "question": "A robe takes 2 bolts of blue fabric and half that much white fabric. How many bolts of fabric does it take?",
        "reasoning": "A robe takes 2 bolts of blue fabric. It takes 2/2 = 1 bolt of white fabric. A robe takes 2+1 = 3 bolts of fabric.",
        "answer": "3"
    }
]

AQuA_Prompt_list = [
    {
        "question": "A train 150 m long is running at a speed of 90 km/h. How long will it take to cross a platform 300 m long?",
        "reasoning": "Speed = 90 km/h = 90 * 1000/3600 = 25 m/s. Distance = 150 + 300 = 450 m. Time = 450/25 = 18 seconds.",
        "answer": "18"
    }
]

CSQA_Prompt_list = [
    {
        "question": "What do people use to absorb extra ink from a fountain pen?",
        "reasoning": "People use blotting paper to absorb extra ink from a fountain pen.",
        "answer": "blotting paper"
    }
]

StrategyQA_Prompt_list = [
    {
        "question": "Do hamsters provide food for any fish?",
        "reasoning": "Hamsters are small rodents that are sometimes kept as pets. Fish are aquatic animals. Hamsters do not provide food for fish in their natural habitat or as pets.",
        "answer": "No"
    }
]
