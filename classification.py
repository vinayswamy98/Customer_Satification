import anthropic
import pandas as pd
import json
from typing import Dict, List, Optional

class LLMClassifier:
def **init**(self, api_key: str, model: str = “claude-sonnet-4-5-20250929”):
self.client = anthropic.Anthropic(api_key=api_key)
self.model = model

```
def build_comprehensive_prompt(
    self, 
    summary: str, 
    keywords: str, 
    mapping_data: pd.DataFrame,
    few_shot_examples: Optional[List[Dict]] = None
) -> str:
    """
    Build a comprehensive classification prompt with detailed instructions.
    
    Args:
        summary: The summary text to classify
        keywords: Associated keywords
        mapping_data: DataFrame with 'name' and 'definition' columns
        few_shot_examples: Optional list of example classifications
    """
    
    # Format the categories
    categories_text = ""
    for idx, row in mapping_data.iterrows():
        categories_text += f"{idx + 1}. **{row['name']}**\n"
        categories_text += f"   Definition: {row['definition']}\n\n"
    
    # Build few-shot examples section
    examples_section = ""
    if few_shot_examples:
        examples_section = "\n## EXAMPLES OF CORRECT CLASSIFICATIONS:\n\n"
        for i, example in enumerate(few_shot_examples, 1):
            examples_section += f"Example {i}:\n"
            examples_section += f"Summary: {example['summary']}\n"
            examples_section += f"Keywords: {example['keywords']}\n"
            examples_section += f"Correct Category: {example['category']}\n"
            examples_section += f"Reasoning: {example['reasoning']}\n\n"
    
    # Build the comprehensive prompt
    prompt = f"""# CLASSIFICATION TASK
```

You are an expert classification system. Your task is to analyze a summary and its keywords, then assign it to the most appropriate category from the provided list.

## AVAILABLE CATEGORIES:

{categories_text}

## CLASSIFICATION INSTRUCTIONS:

1. **Carefully read** the summary and keywords provided
1. **Analyze the core topic** and main subject matter
1. **Compare against each category** definition systematically
1. **Select the single best match** based on semantic similarity and relevance
1. **Provide reasoning** that explains your classification decision
1. **Assign a confidence score** (0-100) based on:
- 90-100: Perfect match, unambiguous alignment
- 70-89: Strong match, clear alignment with minor ambiguity
- 50-69: Moderate match, reasonable fit but could overlap with other categories
- 30-49: Weak match, significant uncertainty
- 0-29: Poor match, forced classification

## DECISION CRITERIA:

- **Primary relevance**: Does the core subject matter align with the category definition?
- **Keyword alignment**: Do the keywords strongly indicate this category?
- **Semantic similarity**: Is the meaning and context consistent with the category?
- **Specificity**: Choose the most specific applicable category over generic ones
- **Exclusion principle**: If it clearly doesn’t fit other categories, explain why

## EDGE CASES TO HANDLE:

- **Multiple possible categories**: Choose the PRIMARY category (the strongest match)
- **No clear match**: Select the closest option and indicate low confidence
- **Ambiguous content**: Use the keywords to help disambiguate
- **Broad summaries**: Focus on the dominant theme

{examples_section}

## ITEM TO CLASSIFY:

**Summary:** {summary}

**Keywords:** {keywords}

## REQUIRED OUTPUT FORMAT:

Respond with ONLY a valid JSON object in this exact format (no additional text):

{{
“category”: “exact_category_name_from_list”,
“confidence”: 85,
“reasoning”: “Clear explanation of why this category was chosen, referencing specific aspects of the summary and keywords that align with the category definition. Mention any alternative categories considered and why they were rejected.”,
“alternative_category”: “second_best_option_or_null”,
“alternative_confidence”: 45
}}

**IMPORTANT**:

- Use the EXACT category name as listed above
- Confidence must be an integer between 0-100
- Reasoning should be 2-4 sentences
- Alternative category is optional (use null if no reasonable alternative exists)

Begin your classification now:”””

```
    return prompt

def classify_single(
    self,
    summary: str,
    keywords: str,
    mapping_data: pd.DataFrame,
    few_shot_examples: Optional[List[Dict]] = None,
    max_retries: int = 3
) -> Dict:
    """
    Classify a single summary-keywords pair.
    
    Returns:
        Dictionary with classification results
    """
    prompt = self.build_comprehensive_prompt(
        summary, keywords, mapping_data, few_shot_examples
    )
    
    for attempt in range(max_retries):
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,  # Use 0 for consistent classification
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text.strip()
            
            # Try to extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Validate the result
            if "category" not in result or "confidence" not in result:
                raise ValueError("Missing required fields in response")
            
            # Verify category exists in mapping data
            if result["category"] not in mapping_data["name"].values:
                raise ValueError(f"Invalid category: {result['category']}")
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == max_retries - 1:
                # Last attempt failed, return error result
                return {
                    "category": "CLASSIFICATION_FAILED",
                    "confidence": 0,
                    "reasoning": f"Error: {str(e)}",
                    "alternative_category": None,
                    "alternative_confidence": 0,
                    "error": str(e)
                }
            # Retry with slightly higher temperature
            continue
    
    return {"category": "CLASSIFICATION_FAILED", "confidence": 0, "reasoning": "Max retries exceeded"}

def classify_batch(
    self,
    summary_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    summary_col: str = "summary",
    keywords_col: str = "keywords",
    few_shot_examples: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Classify all rows in the summary dataframe.
    
    Args:
        summary_df: DataFrame with summaries and keywords
        mapping_df: DataFrame with category names and definitions
        summary_col: Name of the summary column
        keywords_col: Name of the keywords column
        few_shot_examples: Optional examples for few-shot learning
        
    Returns:
        Original dataframe with added classification columns
    """
    results = []
    
    for idx, row in summary_df.iterrows():
        print(f"Classifying row {idx + 1}/{len(summary_df)}...")
        
        result = self.classify_single(
            summary=row[summary_col],
            keywords=row[keywords_col],
            mapping_data=mapping_df,
            few_shot_examples=few_shot_examples
        )
        
        results.append(result)
    
    # Add classification results to dataframe
    summary_df['predicted_category'] = [r['category'] for r in results]
    summary_df['confidence'] = [r['confidence'] for r in results]
    summary_df['reasoning'] = [r['reasoning'] for r in results]
    summary_df['alternative_category'] = [r.get('alternative_category') for r in results]
    summary_df['alternative_confidence'] = [r.get('alternative_confidence', 0) for r in results]
    
    return summary_df
```

# USAGE EXAMPLE

if **name** == “**main**”:
# Initialize classifier
classifier = LLMClassifier(api_key=“your-anthropic-api-key”)

```
# Load your datasets
mapping_df = pd.DataFrame({
    'name': ['Technology', 'Healthcare', 'Finance', 'Education'],
    'definition': [
        'Topics related to software, hardware, IT infrastructure, and digital innovation',
        'Medical services, patient care, pharmaceuticals, and health systems',
        'Banking, investments, economic policy, and financial markets',
        'Learning institutions, teaching methods, and educational policy'
    ]
})

summary_df = pd.DataFrame({
    'summary': [
        'New AI model achieves breakthrough in natural language processing',
        'Hospital implements new patient care protocols for better outcomes',
        'Stock market reaches all-time high amid economic recovery'
    ],
    'keywords': [
        'AI, machine learning, NLP, technology',
        'hospital, patients, healthcare, treatment',
        'stocks, market, economy, investment'
    ]
})

# Optional: Define few-shot examples for better accuracy
few_shot_examples = [
    {
        'summary': 'University launches new online degree program',
        'keywords': 'university, degree, online, learning',
        'category': 'Education',
        'reasoning': 'The summary focuses on university education and degree programs, which directly aligns with the Education category definition.'
    }
]

# Classify all summaries
results_df = classifier.classify_batch(
    summary_df=summary_df,
    mapping_df=mapping_df,
    few_shot_examples=few_shot_examples
)

# Display results
print("\n=== CLASSIFICATION RESULTS ===\n")
print(results_df[['summary', 'predicted_category', 'confidence', 'reasoning']])

# Save results
results_df.to_csv('classification_results.csv', index=False)
print("\nResults saved to 'classification_results.csv'")
```