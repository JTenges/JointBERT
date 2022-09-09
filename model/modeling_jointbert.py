import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from model.modelling_entity_bert import EntityBertModel


from .module import POOLING_AVG, POOLING_MAX, POOLING_MIN, IntentClassifier, SlotClassifier

import os

import pickle as pkl

class EntityBertConfig(BertConfig):
    def __init__(
        self,
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02,
        layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False,
        trainable_entity=False, pooler=POOLING_AVG,
        **kwargs
    ):
        super().__init__(
            vocab_size, hidden_size, num_hidden_layers,
            num_attention_heads, intermediate_size,
            hidden_act, hidden_dropout_prob,
            attention_probs_dropout_prob, max_position_embeddings,
            type_vocab_size, initializer_range, layer_norm_eps,
            pad_token_id, gradient_checkpointing,
            **kwargs
        )
        self.trainable_entity = trainable_entity
        self.pooler = pooler

class EntityPooler(nn.Module):
    def __init__(self, hidden_size, pooling=POOLING_AVG):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

        if pooling == POOLING_AVG:
            self.pooler = lambda t: torch.mean(t, 1)
        elif pooling == POOLING_MAX:
            self.pooler = lambda t: torch.max(t, 1)[0]
        elif pooling == POOLING_MIN:
            self.pooler = lambda t: torch.min(t, 1)[0]
        else:
            raise ValueError(f'{pooling} is not a valid pooling method')

    def forward(self, hidden_states):
        pooled_token_tensor = self.pooler(hidden_states)
        pooled_output = self.dense(pooled_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)

        # Pretrained entity embeddings
        entity_pretrained_embeddings_path = os.path.join(args.data_dir, args.task, args.entity_embeddings)
        with open(entity_pretrained_embeddings_path, 'rb') as f:
            entity_pretrained_embeddings = pkl.load(f)
                
        self.entity_dim = args.entity_dim
        self.classifier_input_dim = config.hidden_size + self.entity_dim
        self.pooler = EntityPooler(args.entity_dim, args.pooling)
        
        self.trainable_entity = config.trainable_entity
        # include none entity
        num_entities = len(entity_pretrained_embeddings) + 1
        entity_dim = self.entity_dim
        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.entity_embeddings.weight.data = torch.tensor(
            list(entity_pretrained_embeddings.values())
        )
        self.entity_embeddings.weight.requires_grad = self.trainable_entity
        
        self.bert = EntityBertModel(
            config=config, # Load pretrained bert
        )

        self.intent_classifier = IntentClassifier(self.classifier_input_dim, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(self.classifier_input_dim, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, entity_ids):
        # Get entity embeddings
        entity_embeddings = self.entity_embeddings(entity_ids)

        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_embeddings=entity_embeddings,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        pooled_entity_embeddings = self.pooler(entity_embeddings)

        sequence_output = torch.cat(
            (sequence_output, entity_embeddings),
            2
        )

        pooled_output = torch.cat(
            (pooled_output, pooled_entity_embeddings),
            1
        )

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
