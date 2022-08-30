import csv
import os

RAW_PATH = 'conda_raw'
OUT_PATH = 'conda'
INTENT_LABELS = ['E', 'I', 'A', 'O']
SLOT_LABELS = ['T', 'C', 'D', 'S', 'P', 'O']

def generate_examples(out_path, csv_fname):
    if not os.path.exists(out_path):
      os.makedirs(out_path)

    label_file = open(f'{out_path}/label', 'w', encoding='UTF-8')
    seq_in_file = open(f'{out_path}/seq.in', 'w', encoding='UTF-8')
    seq_out_file = open(f'{out_path}/seq.out', 'w', encoding='UTF-8')

    with open(csv_fname, 'r', encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        # skip header row
        csv_reader.__next__()
        for row in csv_reader:
            label = row[7]

            slot_tokens = row[-1]
            slot_tokens = slot_tokens.split(', ')[:-1]

            text = []
            labels = []
            for s_tok in slot_tokens:
                s_tok_split = s_tok.split(' ')
                text.append(s_tok_split[0])
                labels.append(s_tok_split[1][1:-1])
            
            text = ' '.join(text)
            labels = ' '.join(labels)
            seq_in_file.write(f'{text}\n')
            seq_out_file.write(f'{labels}\n')
            label_file.write(f'{label}\n')

    
    label_file.close()
    seq_in_file.close()
    seq_out_file.close()

def generate_data():
    # Generate files containing all intent and slot labels
    intent_list = sorted(INTENT_LABELS)
    with open(f'{OUT_PATH}/intent_label.txt', 'w') as f:
        f.write('\n'.join(intent_list))
        f.write('\n')
    
    slot_list = sorted(SLOT_LABELS + ['SEPA'])
    with open(f'{OUT_PATH}/slot_label.txt', 'w') as f:
        f.write('\n'.join(slot_list))
        f.write('\n')
    
    # Generate data files
    generate_examples(f'{OUT_PATH}/train', f'{RAW_PATH}/CONDA_train.csv')
    generate_examples(f'{OUT_PATH}/dev', f'{RAW_PATH}/CONDA_valid.csv')
    generate_examples(f'{OUT_PATH}/test', f'{RAW_PATH}/CONDA_test_original.csv')

if __name__ == '__main__':
    generate_data()