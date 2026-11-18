"""Download 12 authentic PathVQA held-out test examples for a Gradio demo.
Run: python -m pip install datasets pillow
     python download_pathvqa_test_examples.py
"""
import csv
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).resolve().parent / 'PathVQA_Test_Examples'
OUT.mkdir(exist_ok=True)
print('Connecting to official PathVQA held-out test split...')
try:
    ds = load_dataset('flaviagiammarino/path-vqa', split='test', streaming=True)
except Exception as e:
    raise SystemExit(f'Could not open dataset: {e}')

counts = {'Yes/No': 0, 'Open-ended': 0}
rows = []
seen_questions = set()
for idx, example in enumerate(ds):
    answer = str(example['answer']).strip()
    question = str(example['question']).strip()
    category = 'Yes/No' if answer.lower() in ('yes', 'no') else 'Open-ended'
    if counts[category] >= 6 or (question.lower(), answer.lower()) in seen_questions:
        continue
    image = example['image']
    filename = f"test_{idx:04d}_{'yesno' if category == 'Yes/No' else 'open'}_{counts[category]+1}.png"
    image.convert('RGB').save(OUT / filename)
    rows.append({'test_row_index': idx, 'image_file': filename, 'category': category,
                 'question': question, 'correct_answer': answer})
    seen_questions.add((question.lower(), answer.lower()))
    counts[category] += 1
    print(f"Saved {filename} | {category} | {question} -> {answer}")
    if all(v >= 6 for v in counts.values()):
        break

with (OUT / 'questions_and_answers.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['test_row_index', 'image_file', 'category', 'question', 'correct_answer'])
    writer.writeheader()
    writer.writerows(rows)
with (OUT / 'READ_ME.txt').open('w', encoding='utf-8') as f:
    f.write('Official PathVQA held-out test examples\n\n')
    f.write('Upload the named PNG to Gradio, paste its question, select the matching answer category, and compare the prediction with the correct answer.\n\n')
    for i, row in enumerate(rows, 1):
        f.write(f"Example {i} | {row['category']}\nImage: {row['image_file']}\nQuestion: {row['question']}\nCorrect answer: {row['correct_answer']}\nTest row index: {row['test_row_index']}\n\n")
print(f'\nFinished: {len(rows)} authentic test examples in {OUT}')
if len(rows) < 12:
    print('Warning: fewer than 12 examples found. Check network/dataset access.')
