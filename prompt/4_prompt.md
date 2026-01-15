# Prompt guidelines
- Be clear and specific
- Analyze why result does not give desired output.
- Refine the idea and the prompt.
- Repeat

Prompt:
```Python
fact_sheet_chair = f"""
...
"""
text_1 = f"""
Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.
Write a product description based on the information provided in the technical specifications delimited by triple backticks.

Use at most 50 words.
Use at most 3 sentences.
Use at most 300 characters.

Technical specifications:'''{fact_sheet_chair}'''
"""
```

Prompt:
```python
text = f"""
....
"""
prompt = f""""
Your task is to generate a short summary of a product review from an e-commerce site.
 
Summarize the review below, delimited by triple backticks, in at most 30 words.

Review: ```{text}```
"""
```