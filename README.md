# PyTorch-lightning-pick-up
Pick up PyTorch-Lightning 

# Table of content 

1. [Installation](#install)
2. [Example](#example)
    - [From PyTorch to Lightning](#light)
    - [TensorBoard](#tensorboard)

## Installation <a name="install"><\a>

-pip
```{bash}
pip install pytorch-lightning
```

- conda
```{bash}
conda install pytorch-lightning -c conda-forge
```
## Example for an Auto-Encoder <a name="example"><\a>

### From PyTorch to PyTorch lightning. <a name="ligth"><\a>

From [Lightning in 15 minutes](https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html)

- The model architecture goes into  ```__init__```.

__PyTorch__ : 
``` {python}
self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64), 
            nn.ReLU(),
            nn.Linear(64, 3)
            )
self.decoder = nn.Sequential(
    nn.Linear(3, 64), 
    nn.ReLU(), 
    nn.Linear(64, 28 * 28)
    )
encoder.cuda(0)
decoder.cuda(0)
```

__Lightning__ : 
``` {python}
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64), 
            nn.ReLU(),
            nn.Linear(64, 3)
            )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64), 
            nn.ReLU(), 
            nn.Linear(64, 28 * 28)
            )
```

- Optimizers go to ```configure_optimizers```.

__PyTorch__ : 
```{python}
# parameters
params = [encoder.parameters(), decoder.parameters()]
optimizer = torch.optim.Adam(params, lr=1e-3)
```

__Lightning__ : 
```{python}
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3) 
        # self.parameters will contain parameters from encoder & decoder
        return optimizer 
```

- Training goes into ```training_step``` :

__PyTorch__ :
```{python}
# train loop
model.train()
num_epochs = 1
for epoch in range(num_epochs):
    for train_batch in mnist_train : 
        x,y = train_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        z = encoder(x)
        x_hat = decoder(z)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        optimizer.step()
        optimize.zero_grad()
```

__Lightning__ : 
```{python}
  def training_step(self, batch, batch_idx):
  # training_step defines the train loop.
    # it is independent of forward
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = nn.functional.mse_loss(x_hat, x)
    # Logging to TensorBoard by default
    self.log("train_loss", loss) # send any metric to tensor board.
    # add on_epoch = True, to calculate epoch-level metric. 
    return loss
```

- Validation goes into ```validation_step```

__PyTorch__ :
```{python}
model.eval()
with torch.no_grad():
    val_loss = []
    for val_batch in mnist_val : 
        x, y = val_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        z = encoder(x)
        x_hat = decoder(z)
        loss = F.mse_loss(x_hat, x)
        val_loss.append(loss)
        val_loss = torch.mean(torch.tensor(val_loss))
```

__Lightning__ :
```{python}
    def validation_step(self, val_batch, batch_idx):    
    x, y = val_batch
    x = x.cuda(0)
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_error(x_hat, x)
    # calling self.log from validation step will automatically
    # accumulate and log at the end of the epoch 
    self.log('val_loss', loss)
```

- .cuda()
Remove any .cuda() or device calls. 
LightningModules are hardware agnosotic. 

- Override LightningModule Hooks

__PyTorch__ : 
```{python}
loss.backward()
```

__Lightning__ : 
```{python}
    def backward(self, loss, optimizer, optimizer_idx): 
        loss.backward(retain_graph=True)
```

- Init 
```{python}
model = LitAutoEncoder(encoder, decoder)
```
- Training loop
The lightning trainer automates all the engineering : loops, hardware calls, model.train(), zero_grad ..
```{python}
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader) # pass data to the trainer
```
Nb : train on gpus
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, gpus=4)

### TensorBoard <a name="tensorboard"><\a>
```{bash}
tensorboard --logdir .
```