
import torch
from tqdm import tqdm

from Modules.metrics import evaluate_captioning
from Modules.utils import add_dict
from Modules.config import VLM_CHECKPOINT_DIR
from Modules.FusionVLM import save_FusionVLM


def evaluate_epoch(model, loader, tokenizer):
    model.eval()

    loss = 0
    preds = []
    refs = []

    with torch.no_grad():
        for batch in loader:
            gt_captions = batch["all_captions"]  # List[List[str]]
            
            outputs = model(
            query_pixel_values=batch["query_pixel_values"],
            retrieved_pixel_values=batch["retrieved_pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
            )
            loss += outputs.loss.item()
            
            generated_ids = model.generate(
                    query_pixel_values=batch["query_pixel_values"],
                    retrieved_pixel_values=batch["retrieved_pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=64,
                    # num_beams=3
            )

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            preds.extend(decoded)
            refs.extend(gt_captions)
            
    loss = loss / len(loader)
    print(f"Val Loss: {loss:.8f}")
    return loss, evaluate_captioning(preds, refs)


def train_epoch(model, loader, optimizer, epoch):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True)

    for batch in progress_bar:
        optimizer.zero_grad()

        outputs = model(
            query_pixel_values=batch["query_pixel_values"],
            retrieved_pixel_values=batch["retrieved_pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    progress_bar.set_postfix(loss=epoch_loss)
    print(f"Avg Loss: {epoch_loss:.8f}")
    
    return epoch_loss


def train_and_evaluate_model(model, train_loader, optimizer, num_epochs, val_loader=None, tokenizer=None, save_interval=5):    
    history = {'train_loss': [],
               'val_loss': []
               }
    
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, train_loader, optimizer, epoch)
        history['train_loss'].append(epoch_loss)

        # Evaluate
        if val_loader:
            test_loss, metrics = evaluate_epoch(model, val_loader, tokenizer)
            history['val_loss'].append(test_loss)
            add_dict(history, metrics)
            
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

        # ---- Backup every 5 epochs ----
        if (epoch + 1) % save_interval == 0:
            save_FusionVLM(model, f'epoch{epoch+1}', VLM_CHECKPOINT_DIR)
    
    return history


if __name__ == "__main__":
    pass