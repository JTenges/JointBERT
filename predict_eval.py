import os
from sklearn.metrics import classification_report

# Maybe unused, just commited for history
if __name__ == '__main__':
    # intent_preds, intent_labels, slot_preds, slot_labels
    data_dir = 'data'
    task = 'conda'

    slot_tokens_file = 'test/seq.out'

    intent_preds = []
    slot_preds = []
    with open('predictions.out', 'r') as f:
        line = f.readline()
        f2 = open(os.path.join(data_dir, task, slot_tokens_file), 'r', encoding='utf-8')

        line_f = f2.readline()
        while line != '':
            intent, tokens = line.split(' -> ')

            intent = intent[1]
            intent_preds.append(intent)
            
            tokens = tokens.strip()
            if tokens == '':
                slot_preds.append('O')
                print(f'{line}|||{line_f}')
            else:
                tokens = tokens.split(' ')
                if len(tokens) != len(line_f.split(' ')):
                    print(f'{line}|||{line_f}|||{len(tokens)} {len(line_f.split(" "))}')
                    
                tokens = [t[-2] if t[0] == '[' and t[-1] == ']' else 'O' for t in tokens]
                slot_preds += tokens

            line = f.readline()
            line_f = f2.readline()
    

    data_dir = 'data'
    task = 'conda'

    intent_label_file = 'test/label'
    intent_labels = [label.strip() for label in open(os.path.join(data_dir, task, intent_label_file), 'r', encoding='utf-8')]
    
    slot_tokens_file = 'test/seq.out'
    slot_tokens = [label.strip().split(' ') for label in open(os.path.join(data_dir, task, slot_tokens_file), 'r', encoding='utf-8')]
    slot_labels = []
    for slots in slot_tokens:
        slot_labels += slots

    print(classification_report(slot_labels, slot_preds))
