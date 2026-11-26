# Week 4 - Day 1: Prompts as Code & Golden Dataset Evaluation

## Folder Structure
- **prompts/**: Versioned prompt templates (classifier_v1.txt and classifier_v2.txt).
- **task_2/**: Golden dataset (golden.jsonl) and automated evaluation runner (eval_prompts.py).

## Evaluation Summary
- **Prompt v1:** 1/20 (5%)
- **Prompt v2:** 19/20 (95%) - Passed >= 85% CI threshold
- **Failing Case (#19):** 'Is there a desktop version available for Linux?' (Expected: General | Got: Technical). Decision: Update prompt rules in v3 to clarify that general platform availability inquiries belong in General rather than modifying the valid dataset label.
