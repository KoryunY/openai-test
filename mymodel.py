import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        max_epochs,
        device,
        batch_size,
        max_sequence_length,
        random_sequence_length,
        data_dir,
        real_dataset,
        fake_dataset,
        large,
        learning_rate,
        weight_decay,
        command,
        num_encoder_layers,
        **kwargs
    ):
        super(MyModel, self).__init__()
        # Initialize the model attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_epochs = max_epochs
        self.device = device
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.random_sequence_length = random_sequence_length
        self.data_dir = data_dir
        self.real_dataset = real_dataset
        self.fake_dataset = fake_dataset
        self.large = large
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.command = command

        # Define the layers and other components of the model
        self.roberta = nn.ModuleDict({
            'embeddings': nn.ModuleDict({
                'word_embeddings': nn.Embedding(input_size, hidden_size),
                'position_embeddings': nn.Embedding(max_sequence_length, hidden_size),
                'token_type_embeddings': nn.Embedding(2, hidden_size),
                'LayerNorm': nn.LayerNorm(hidden_size),
            }),
            'encoder': nn.ModuleList([
                nn.ModuleDict({
                    'attention': nn.ModuleDict({
                        'self': nn.ModuleDict({
                            'query': nn.Linear(hidden_size, hidden_size),
                            'key': nn.Linear(hidden_size, hidden_size),
                            'value': nn.Linear(hidden_size, hidden_size),
                        }),
                        'output': nn.ModuleDict({
                            'dense': nn.Linear(hidden_size, hidden_size),
                            'LayerNorm': nn.LayerNorm(hidden_size),
                        }),
                    }),
                    'intermediate': nn.ModuleDict({
                        'dense': nn.Linear(hidden_size, hidden_size),
                    }),
                    'output': nn.ModuleDict({
                        'dense': nn.Linear(hidden_size, hidden_size),
                        'LayerNorm': nn.LayerNorm(hidden_size),
                    }),
                })
                for _ in range(num_encoder_layers)
            ]),
            'pooler': nn.ModuleDict({
                'dense': nn.Linear(hidden_size, hidden_size),
            }),
            'classifier': nn.Linear(hidden_size, output_size),
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        logits = self.roberta['classifier'](pooled_output)
        return logits
