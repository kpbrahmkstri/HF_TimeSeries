import torch
from src.config import TSConfig

def train_model(model, train_loader, cfg: TSConfig) -> None:
    model.train()
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        total_loss, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n += 1

        print(f"[epoch {epoch+1}/{cfg.epochs}] loss={total_loss/max(n,1):.4f}")
