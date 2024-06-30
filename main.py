from model import NewsClassifier
from preprocess import TextDataset
import settings as S

from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

from torch.distributed import init_process_group

tokenizer = AutoTokenizer.from_pretrained(S.PRETRAINED_MODEL_NAME)

def ddp_setup():
    backend = 'nccl' if S.DEVICE == 'cuda' else 'gloo'
    init_process_group(backend=backend)

def train_model(
        local_rank: int,
        rank: int,
        pretrained_model_name: str,
        train_data_s3_url: str,
        test_data_s3_url: str,
        train_len: int,
        test_len: int,
        train_files: int,
        test_files: int,
        epochs: int,
        lr: float) -> NewsClassifier:

    device = S.DEVICE
    model = NewsClassifier(pretrained_model_name).to(device)
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.BCELoss()

    train_df = TextDataset(tokenizer, train_data_s3_url, train_files)
    test_df = TextDataset(tokenizer, test_data_s3_url, test_files)
    train_loader = DataLoader(train_df, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_df, batch_size=1, shuffle=False)

    print(f'[Node:{rank} - {local_rank}] Training model...')
    for epoch in range(epochs):
        training_loss = 0
        training_acc = 0
        model.train()

        for bert_input, tabular_input, label in train_loader:
            bert_input = {
                'input_ids': bert_input[0].squeeze().to(device),
                'attention_mask': bert_input[1].squeeze().to(device),
                'return_dict': False
            }
            tabular_input = torch.cat(tabular_input).T.to(device)
            label = label.T.to(device)
            
            output = model(bert_input, tabular_input)

            loss = loss_function(output, label.float())
            training_loss += loss.item()

            # get acc of signmoid output
            acc = (output[0].round() == label).sum().item()
            training_acc += acc

            model.zero_grad()
            loss.backward()
            optimizer.step()

        validation_loss = 0.0
        validation_acc = 0.0

        # Calculate validation loss and acc for each epoch\
        model.eval()
        with torch.no_grad():
            for bert_input, tabular_input, label in test_loader:
                bert_input = {
                    'input_ids': bert_input[0].squeeze().to(device),
                    'attention_mask': bert_input[1].squeeze().to(device),
                    'return_dict': False
                }
                tabular_input = torch.cat(tabular_input).T.to(device)
                label = label.T.to(device)

                output = model(bert_input, tabular_input)


                loss = loss_function(output, label.float())
                validation_loss += loss.item()

                # get acc of signmoid output
                acc = (output[0].round() == label).sum().item()
                validation_acc += acc
        print(f'[Node:{rank} - {local_rank}] Epoch: {epoch+1}/{epochs} | Training loss: {training_loss/train_len:.3f} | Training acc: {training_acc/train_len:.3f} | Validation loss: {validation_loss/test_len:.3f} | Validation acc: {validation_acc/test_len:.3f}')
    return model
    
def save_model_to_s3(model, s3, model_output_s3_url: str):
    torch.save(model.state_dict(), 'tmp/model.pth')
    s3.upload_file('tmp/model.pth', S.BUCKET_NAME, model_output_s3_url)
if __name__ == '__main__':
    ddp_setup()
    model = train_model(
        S.LOCAL_RANK,
        S.RANK,
        S.PRETRAINED_MODEL_NAME,
        S.TRAIN_DATA_S3_URL,
        S.TEST_DATA_S3_URL,
        S.TRAIN_DATASET_SIZE,
        S.TEST_DATASET_SIZE,
        S.TRAIN_FILES,
        S.TEST_FILES,
        S.EPOCHS,
        S.LR
    )
    save_model_to_s3(model, S.s3, S.MODEL_OUTPUT_S3_URL)
