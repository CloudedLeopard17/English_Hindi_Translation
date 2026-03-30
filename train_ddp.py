import os
import torch
import math
import time
import inspect
import tokenizers
import sentencepiece
import numpy as np
import torch.nn as nn
import sentencepiece as spm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.text import BLEUScore
from tokenizers import Tokenizer
from datasets import load_dataset, load_from_disk

#tensorboard
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#defualt is highest
torch.set_float32_matmul_precision('high')

# ====================================================
# CFG
# ====================================================

@dataclass
class CFG:
    ddp = True
    #defaults for ddp 
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    d_model = 512
    h=8
    d_ff=2048
    dropout=0.1
    tokenizer_src_loc = './sentencepiece_tok/en_hi_shared.model'
    tokenizer_tgt_loc = './sentencepiece_tok/en_hi_shared.model'
    dataset = 'cfilt/iitb-english-hindi'
    prepared_dataset_dir = './tokenized_dataset/'
    prepared_train_dataset_name = 'train_dataset.pt'
    prepared_valid_dataset_name = 'valid_dataset.pt'
    src_seq_len = 100
    tgt_seq_len = 120
    src_lang = 'en'
    tgt_lang = 'hi'
    batch_size_train = 128
    batch_size_valid = 1
    valid_interval = 2
    num_workers = 8
    init_learning_rate = .0002
    weight_decay=0.1
    betas = (0.9, 0.95)
    grad_clip = 1.0
    grad_accumulation_steps = 2
    seed=42
    n_epochs=10
    train=True
    experiment_name = 'logs_tensorboard_bpe_tokenizer_ddp'
    state_dir = 'models_ddp'
    best_state_file_name = 'best_state_ddp'
    last_state_file_name = 'last_state_ddp'
    last_state =  None #'last_state_ddp_9.pth'
    max_token_gen_len = 100 #total tokens to generate for validation
    print_examples = 3 #total examples to print in validation

class InputEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        "In paper embed_dim is d_model, which is 512"
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
    def forward(self,x):
        return self.embeddings(x)*math.sqrt(self.embed_dim)
    
class PositionalEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int, dropout: float):
        super().__init__()        
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        #log exp for numerical stability
        div = torch.exp(-2*(torch.arange(0, embed_dim))/embed_dim*math.log(10000))
        pe = pos*div
        pe[:, 0::2] = torch.sin(pe[:,0::2])
        pe[:, 1::2] = torch.cos(pe[:,1::2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        
        x = x+(self.pe[:, torch.arange(0, x.shape[1]),:]).requires_grad_(False)
        return self.dropout(x)
    
class FFN(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self,x):
        return self.feed_forward(x)

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model%h ==0, "d_model should be divisible by h"
        
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model, bias= False)
        self.w_k = nn.Linear(d_model,d_model, bias= False)        
        self.w_v = nn.Linear(d_model, d_model, bias= False)  
        
        self.w_o = nn.Linear(d_model, d_model, bias= False)  
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        scores = query@key.transpose(-2,-1)/math.sqrt(d_k)
        #print(query.shape,key.shape, value.shape, scores.shape, mask.shape)
        if mask is not None:
            scores.masked_fill_(mask==0, -1e9)
        scores = scores.softmax(dim=-1)
        
        ### do we need ?
        if dropout is not None:
            scores = dropout(scores)
            
        return scores@value,scores 
        
    def forward(self, q, k, v, mask):
        # batch, seq len, embed dim
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # batch, h, seq, d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        #x, scores = self.attention(query, key, value, mask, self.dropout)
        
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=self.dropout.p)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)
        
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    
    def __init__(self, d_model:int, dropout:float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        #in paper layer norm is applied after the sum 
        return x + self.dropout(sublayer(self.layer_norm(x)))
        

class EncoderBlock(nn.Module):
    def __init__(self, d_model:int, h:int, d_ff:int, dropout:float):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x_ln = self.layer_norm_1(x)
        attention = self.attention(x_ln, x_ln, x_ln, mask)
        x = x + self.dropout(attention)
        
        ffn_out = self.ffn(self.layer_norm_2(x))
        x = x+ self.dropout(ffn_out)
        
        return x
        
class DecoderBlock(nn.Module):
    def __init__(self, d_model:int, h:int, d_ff:int, dropout:float):
        super().__init__()
        self.attention_1 = MultiHeadAttention(d_model, h, dropout)
        self.attention_2 = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_out, src_mask, tgt_mask):
        x_ln = self.layer_norm_1(x)
        attention = self.attention_1(x_ln, x_ln, x_ln, tgt_mask)
        x = self.layer_norm_2(x + self.dropout(attention))
        #key, value from encoder and query from decoder
        attention = self.attention_2(x, encoder_out, encoder_out, src_mask)
        x = x + self.dropout(attention)
        
        ffn_out = self.ffn(self.layer_norm_3(x))
        x = x+ self.dropout(ffn_out)
        
        return x
        
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self,x):
        #for numerical stability!!!
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model:int, h:int, d_ff:int, dropout:float, N=6):
        super().__init__()
        
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(d_model, h, d_ff, dropout)
                for _ in range(N)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(d_model, h, d_ff, dropout)
                for _ in range(N)
            ]
        )
        
        self.src_embed = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        self.src_pos_embed = PositionalEmbeddings(d_model, src_seq_len, dropout)
        self.tgt_pos_embed = PositionalEmbeddings(d_model, tgt_seq_len, dropout)
        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
        
    
    def encode(self,x, src_mask):
        x = self.src_embed(x)
        x = self.src_pos_embed(x)
        for layer in  self.encoder:
            x = layer(x, src_mask)
        return x
        
    def decode(self, x, encoder_out,src_mask, tgt_mask):
        x = self.tgt_embed(x)
        x = self.tgt_pos_embed(x)
        for layer in  self.decoder:
            x = layer(x, encoder_out,src_mask,tgt_mask)
        return x
        
    def project(self,x):
        x = self.projection_layer(x)
        return x
    
    def forward(self, encoder_inp, decoder_inp, encoder_mask, decoder_mask):
        enoder_out = self.encode(encoder_inp, encoder_mask)
        decoder_out = self.decode(decoder_inp, enoder_out, encoder_mask, decoder_mask)
        projection_out = self.project(decoder_out)
        return projection_out
            

def initialize_weight(model: Transformer):
    for name,p in model.named_parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
        else:
            if 'layer' in name:
                continue
            else:
                nn.init.zeros_(p)


class BilingualDataset(Dataset):
    def __init__(self, dataset_loc, dataset_name, master_process):
        self.dataset_loc = dataset_loc
        self.dataset_name = dataset_name
        self.master_process = master_process
        
        if master_process:
            print("Loading prepared dataset...")

        data = torch.load(os.path.join(dataset_loc, dataset_name))
        self.encoder_inps = data['encoder_inps']
        self.decoder_inps = data['decoder_inps']
        self.encoder_masks = data['encoder_masks']
        self.decoder_masks = data['decoder_masks']
        self.labels = data['labels']
        if master_process: 
            print("Dataset loaded successfully!")
    
    def __getitem__(self, idx):
        
        return {
            "encoder_inp": self.encoder_inps[idx],
            "decoder_inp": self.decoder_inps[idx],
            "encoder_mask": self.encoder_masks[idx], 
            "decoder_mask": self.decoder_masks[idx],
            "labels": self.labels[idx]
        }
    
    def __len__(self):
        return len(self.encoder_inps)

    
def save_model(model, optimizer, epoch, device, state_file_name, state_dir, bleu_score, ddp = False):
    
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)

    if 'best' in state_file_name:
        state_path = os.path.join(state_dir, state_file_name + '.pth')
    else:   
        state_path = os.path.join(state_dir, state_file_name + '_' + str(epoch)+ '.pth')

    
    # make sure you transfer the model to cpu.
    if device == 'cpu':
        model.to('cpu')
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    # save the state_dict
    torch.save({
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': torch.tensor(epoch),
        'bleu_score': bleu_score
    }, state_path)
    
    #transfer model to gpu again
    if device == 'cuda':
        model.to('cuda')
    
    return

def load_model(model, state_file_name ,state_dir=CFG.state_dir, ddp=False):
    state_path = os.path.join(state_dir, state_file_name)

    # loading the model and getting model parameters by using load_state_dict
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    state = torch.load(state_path)
    raw_model.load_state_dict(state['model_state_dict'])
    return model

def train(
    train_config: CFG,
    model: torch.nn.parallel.distributed.DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    loss_fn: torch.nn.modules.loss.CrossEntropyLoss,
    vocab_size: int,
    device: str, 
    device_type: str,
    )-> float:

    batch_loss = np.array([])
    batch_norms = np.array([])
    model.train()
    batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch_idx}") if not config.ddp else train_loader
    t0 = time.time()
    for step,batch in enumerate(batch_iterator):
        encoder_inp = batch['encoder_inp'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_inp = batch['decoder_inp'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        labels = batch['labels'].to(device)
        #as we are doing gradient accumulation, we don't want to synchronise gradients every step
        #we accumulate the gradients and synchronise at the last step
        if config.ddp:
            model.require_backward_grad_sync = ((step+1)%train_config.grad_accumulation_steps == 0)
        
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # enoder_out = model.module.encode(encoder_inp, encoder_mask)
            # decoder_out = model.module.decode(decoder_inp, enoder_out, encoder_mask, decoder_mask)
            # project = model.module.project(decoder_out)
            project = model(encoder_inp, decoder_inp, encoder_mask, decoder_mask)
            loss = loss_fn(project.view(-1, vocab_size),labels.view(-1))

        batch_loss = np.append(batch_loss, [loss.item()])
        loss = loss / train_config.grad_accumulation_steps
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        batch_norms = np.append(batch_norms, [norm.item()])
        if (step+1)%train_config.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if not config.ddp:
            batch_iterator.set_postfix({"loss": f"{loss.item()*train_config.grad_accumulation_steps:.4f}"})

    epoch_loss = torch.tensor(batch_loss, device=device).mean()
    epoch_norm = torch.tensor(batch_norms, device=device).mean()
    if config.ddp:
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(epoch_norm, op=dist.ReduceOp.AVG)
    
    dt = time.time() - t0
    tokens_processed = len(train_loader)*(train_config.src_seq_len + train_config.tgt_seq_len)
    tokens_per_sec = (tokens_processed/dt)*config.ddp_world_size
    if config.master_process:
        print('Epoch: {} Train Loss: {:.3f} Average Norm: {:.3f} Tokens/sec: {:.2f} Time: {:.2f}s'.format(epoch_idx, epoch_loss, epoch_norm, tokens_per_sec, dt))
    return epoch_loss.item()


def validation(
    config: CFG,
    model: torch.nn.parallel.distributed.DistributedDataParallel,
    valid_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    max_token_gen_len: int,
    tokenizer_tgt: sentencepiece.SentencePieceProcessor,
    device: str
    )-> torch.Tensor:
    
    '''
        calculate bleu score for translation and display 
        some outputs
    '''
    scores = []
    examples = 0
    eos_token = tokenizer_tgt.eos_id()
    model.eval()
    t0 = time.time()
    raw_model = model.module if config.ddp else model
    with torch.no_grad():
        #batch_iterator = tqdm(valid_loader, desc=f"Processing Epoch {epoch_idx}")
        for batch in valid_loader:
            decoder_inp = torch.empty(1,1).fill_(tokenizer_tgt.bos_id()).type(torch.int32).to(device)
            encoder_inp = batch['encoder_inp'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            labels = batch['labels'].to(device)
            enoder_out = raw_model.encode(encoder_inp, encoder_mask)
            for _ in range(max_token_gen_len):
                decoder_mask = torch.tril(torch.ones(decoder_inp.shape[1],decoder_inp.shape[1])).type(torch.bool).to(device)
                decoder_out = raw_model.decode(decoder_inp, enoder_out, encoder_mask, decoder_mask)
                #only last word
                project = raw_model.project(decoder_out[:,-1,:])
                probs = project.softmax(dim=-1)
                next_idx = probs.argmax(dim=-1).item()
                next_idx_tensor = torch.empty(1,1).fill_(next_idx).type(torch.int32).to(device)
                decoder_inp = torch.cat((decoder_inp, next_idx_tensor), dim=1)
                if next_idx == eos_token:
                    break
            tgt_txt = tokenizer_tgt.decode(labels.squeeze().cpu().tolist())
            pred_txt = tokenizer_tgt.decode(decoder_inp.squeeze().cpu().tolist())

            #the bleu score works above 4 gram, modifying the score function for smaller sentences
            sent_len = decoder_inp.shape[1]-2
            if sent_len<4 and sent_len>0:
                bleu_metric = BLEUScore(n_gram=sent_len)
            else:
                bleu_metric = BLEUScore()
            score = bleu_metric([pred_txt], [[tgt_txt]])
            scores.append(score.item())
        
            if (np.random.random_sample()<.001) and examples<config.print_examples:
                print(f"Source txt: {tokenizer_tgt.decode(encoder_inp.squeeze().cpu().tolist())}")
                print(f'Predicted txt: {pred_txt}')
                print(f'Label: {tgt_txt}\n')
                examples +=1
            #batch_iterator.set_postfix({"bleu": f"{score.item():.4f}"})
        
    print(f"Validation Time: {time.time() - t0:.2f} seconds")
    
    return torch.tensor(scores).mean().to(device)


def main(
    config: CFG,
    model: torch.nn.parallel.distributed.DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    dataset_train: torch.utils.data.Dataset,
    dataset_valid: torch.utils.data.Dataset,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    tokenizer_src: sentencepiece.SentencePieceProcessor,
    tokenizer_tgt: sentencepiece.SentencePieceProcessor,
    device: str,
    device_type: str,
    ):
    
    if config.master_process:
        summary_writer = SummaryWriter(log_dir=config.experiment_name)

    
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size_train,\
                              num_workers=config.num_workers,
                              sampler=DistributedSampler(dataset_train, shuffle=True) if config.ddp else None,
                              shuffle= not config.ddp
                            )
    valid_loader = DataLoader(dataset_valid, batch_size=config.batch_size_valid)
    loss_fn =  nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.pad_id(), label_smoothing=0.1)
    vocab_size = tokenizer_tgt.vocab_size()
    start_epoch = 0
    model.to(device)

    best_bleu = -torch.tensor(np.inf).to(device)
    bleu_score = -torch.tensor(np.inf).to(device)
    if config.last_state is not None:
        model = load_model(model, config.last_state, config.state_dir, ddp=config.ddp)
        state_path = os.path.join(config.state_dir, config.last_state)
        state = torch.load(state_path)
        best_bleu = state['bleu_score'].to(device)
        start_epoch = state['epoch'].item() + 1
        if config.master_process:
            print(f"Loaded model from {config.last_state}")
            print(f"Best Bleu Score: {best_bleu.item():.4f}, Resuming training from epoch {start_epoch}")


    t_begin = time.time()
    if config.master_process:
        print(f"Starting training for {config.n_epochs-start_epoch} epochs...")
    for epoch in range(start_epoch,config.n_epochs):
        
        # Train and visualize training images
        
        train_loss= train(config, model, optimizer, train_loader, epoch, loss_fn, vocab_size, device, device_type)
        if config.master_process:
            summary_writer.add_scalar('Loss/Train',train_loss, epoch) 
            summary_writer.add_scalar('Learning Rate',optimizer.param_groups[0]['lr'] , epoch)

            if ((epoch)%config.valid_interval==0) or epoch == config.n_epochs-1:
            
                bleu_score = validation(config, model, valid_loader, epoch, config.max_token_gen_len,tokenizer_tgt, device)
                print(f'Bleu Score: {bleu_score.item():.4f}')

                summary_writer.add_scalar('BleuScore/Valid',bleu_score.item(), epoch)
                if bleu_score > best_bleu:
                    best_bleu = bleu_score
                    print('...Model Improved. Saving the Model...\n')
                    save_model(model, optimizer, epoch,device=device,state_file_name= config.best_state_file_name,
                        state_dir= config.state_dir, bleu_score=bleu_score.cpu().detach().clone(), ddp=config.ddp)
                
                if epoch==config.n_epochs-1:
                    save_model(model, optimizer, epoch,device=device,state_file_name= config.last_state_file_name,
                        state_dir= config.state_dir, bleu_score=bleu_score.cpu().detach().clone(), ddp=config.ddp)
                    
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_begin
            speed_epoch = elapsed_time / (epoch + 1)
            eta = speed_epoch *config.n_epochs - elapsed_time

            summary_writer.add_scalar('Time Elapsed',elapsed_time, epoch)
            summary_writer.add_scalar('ETA',eta, epoch)

        if config.ddp:  
            dist.broadcast(best_bleu, src=0)
            dist.broadcast(bleu_score, src=0)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(bleu_score.item())
            else:
                scheduler.step()

    if config.master_process:    
        print("Total time: {:.2f}, Best Score: {:.3f}".format(time.time() - t_begin, best_bleu.item()))

    if config.ddp:
        destroy_process_group()


config = CFG()
if config.ddp:
    config.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if config.ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    config.ddp_rank = ddp_rank
    config.ddp_local_rank = ddp_local_rank
    config.ddp_world_size = ddp_world_size
    config.master_process = master_process
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

tokenizer_src = spm.SentencePieceProcessor()
tokenizer_src.load(config.tokenizer_src_loc)

model = Transformer(
    src_vocab_size= tokenizer_src.vocab_size(),
    tgt_vocab_size= tokenizer_src.vocab_size(),
    src_seq_len=config.src_seq_len,
    tgt_seq_len=config.tgt_seq_len,
    d_model=config.d_model,
    h=config.h,
    d_ff=config.d_ff,
    dropout=config.dropout
    )
initialize_weight(model)
if config.master_process:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")


model.to(device)
model = torch.compile(model)

if config.ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if config.ddp else model # always contains the "raw" unwrapped model

def get_optimizer_params(model, lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == "cuda" 
optimizer_params = get_optimizer_params(raw_model, config.init_learning_rate, config.weight_decay)

optimizer = torch.optim.AdamW(
    optimizer_params,
    lr = config.init_learning_rate,
    betas=config.betas,
    fused=use_fused
)

schedular = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.3, patience = 2, min_lr= 1e-6)
#schedular = None


train_dataset = BilingualDataset(dataset_loc= config.prepared_dataset_dir, 
                                dataset_name= config.prepared_train_dataset_name,
                                master_process= config.master_process
                )   

valid_dataset = BilingualDataset(dataset_loc= config.prepared_dataset_dir, 
                                dataset_name= config.prepared_valid_dataset_name,
                                master_process= config.master_process
                )

if config.train:
    main(config, model, optimizer, train_dataset, valid_dataset,schedular, tokenizer_src,
        tokenizer_src, device, device_type)



#python -m torch.distributed.run --nproc_per_node=2   train_ddp.py 




