from transformers import BertModel
import torch 

class NewsClassifier(torch.nn.Module):
    def __init__(self, pretrained_model_name: str) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(768, 256)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(256, 12)
        self.dropout3 = torch.nn.Dropout(0.25)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.normalize = torch.nn.functional.normalize

    def forward(self, bert_input: dict[list[int]], tabular_input: list[int]):
        _, pooled_output = self.bert(**bert_input)
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.normalize(x, p=2, dim=1)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.normalize(x, p=2, dim=1)

        y = self.normalize(tabular_input, p=2, dim=1)
        z = torch.cat([x, y], dim=1)
        z = self.dropout3(z)
        z = self.fc3(z)
        return self.sigmoid(z)


