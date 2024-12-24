import torch
from torch.utils.tensorboard import SummaryWriter
import accelerate
from accelerate import Accelerator
from transformers import LlamaConfig, LlamaForCausalLM
import os, time, argparse



from dataset import BIOPreTrain, BIOVal
from utils import add_weight_decay, adjust_learning_rate

def get_args_parser():
    parser = argparse.ArgumentParser('Pretraining', add_help=False)
    # Model Details
    parser.add_argument('--config_path', type=str, default="config.json")
    parser.add_argument('--tokenizer_path', type=str, default="tokenizer.model")
    # Dataset Details
    parser.add_argument('--size', type=str, default="tiny", choices=["tiny", "small", "medium", "large"])
    parser.add_argument('--train_mode', type=str, default="mixed", choices=["mixed", "pure"])
    parser.add_argument('--use_augment', type=bool, default=False)
    parser.add_argument('--qa_ratio', type=float, default=4.0)
    # Training Details
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accum_iter', type=int, default=12)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--lr_decay_iters', type=int, default=8000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=50)
    # Logging Details
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--valid_freq', type=int, default=1)
    return parser.parse_args()

accelerator = Accelerator()


def write_log(log_writer, mode, step, loss, max_equal=None, lr=None):
    if log_writer is None:
        return
    if loss is not None:
        log_writer.add_scalar('{}_loss'.format(mode), loss, step)
    if max_equal is not None:
        log_writer.add_scalar('{}_max_equal'.format(mode), max_equal, step)
    if lr is not None:
        log_writer.add_scalar('lr', lr, step)


def train(model, data_loader_train, data_loader_valid, optimizer, log_writer, args):
    batch_num_one_epoch = data_loader_train.dataset.data_size_one_epoch // 2 // args.batch_size
    model.train(True)
    optimizer.zero_grad(set_to_none=True)

    loss_list, acc_list = [], []
    for data_iter_step, examples in enumerate(data_loader_train):
        epoch = data_iter_step // batch_num_one_epoch
        epoch_data_iter_step = data_iter_step % batch_num_one_epoch
        if epoch < args.start_epoch and args.load_success:
            lr = adjust_learning_rate(optimizer, data_iter_step//args.accum_iter, 
                    args.warmup_iters, args.lr_decay_iters, args.lr, args.min_lr)
            continue
        if epoch_data_iter_step == 0 and epoch > 0: # a new epoch starts
            if len(loss_list) > 0 and len(acc_list) > 0:
                loss_value = sum(loss_list) / len(loss_list)
                acc_value = sum(acc_list) / len(acc_list)
                accelerator.print(f"Epoch[{epoch-1}] finished. Average Loss: {loss_value:.6f}; MaxEqual: {acc_value:.6f}")
                loss_list, acc_list = [], []
            unwrapped_model = accelerator.unwrap_model(model)
            output_dir = os.path.join(args.output_dir, "Epoch{}".format(epoch))
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            # if epoch % args.valid_freq == 0:
                # eval_one_epoch(model, data_loader_valid, epoch, log_writer, args)
        # Training the model
        labels = examples.clone().detach()
        output = model(input_ids=examples, labels=labels,
            output_attentions=False, output_hidden_states=False) 
        loss = output.loss
        loss = loss / args.accum_iter
        if data_iter_step % args.accum_iter == 0:
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, data_iter_step//args.accum_iter, 
                    args.warmup_iters, args.lr_decay_iters, args.lr, args.min_lr)
        # Log the loss value
        closs = loss.detach().item()
        logits = output.logits
        preds = torch.argmax(logits[:,:-1,:], dim=-1)
        max_equal = (preds == labels[:,1:]).float().mean().item()
        loss_list.append(closs)
        acc_list.append(max_equal)
        
        # write_log(log_writer, 'train', data_iter_step, closs, max_equal, lr)
        if (data_iter_step+1) % args.print_freq == 0:
            accelerator.print('Train Epoch[{}], Iter[{}/{}] Loss: {:.4f} MaxEqual: {:.4f} LR: {:.6f}'.format(
                        epoch, epoch_data_iter_step, batch_num_one_epoch, closs, max_equal, lr))


def eval_one_epoch(model, data_loader, epoch, log_writer, args):
    model.eval()
    for data_iter_step, (input_ids, attention_mask) in enumerate(data_loader):
        with accelerator.accumulate(model):
            labels = input_ids.clone().detach()
            output = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                output_attentions=False, output_hidden_states=False) 
            loss = output.loss
            closs = loss.detach().item()
            logits = output.logits
            preds = torch.argmax(logits[:,:-1,:], dim=-1)
            max_equal = (preds == labels[:,1:]).float().mean().item()
            write_log(log_writer, 'valid', epoch*len(data_loader)+data_iter_step, closs, max_equal)
            if (data_iter_step+1) % args.print_freq == 0:
                accelerator.print('Valid Epoch[{}], Iter[{}/{}]\t Loss: {:.4f}\t MaxEqual: {:.4f}'.format(
                          epoch, data_iter_step, len(data_loader), closs, max_equal))
            


def main(args):
    """Set the settings"""
    accelerate.utils.set_seed(args.seed, device_specific=True) 
    
    """Define the model, optimizer, dataset, and log_writer"""
    # [NOTE] Define a 12-layer, 768-hidden, 12-heads, 110M parameters model
    config = LlamaConfig.from_json_file(args.config_path)
    # tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    # config.vocab_size = tokenizer.vocab_size
    model = LlamaForCausalLM(config)
    args.load_success = False
    if args.start_epoch > 0:
        model_path = os.path.join(args.output_dir, "Epoch{}".format(args.start_epoch))
        if os.path.exists(model_path):
            args.load_success = True
            model = model.from_pretrained(model_path)
            accelerator.print(f"Loaded model from Epoch{args.start_epoch}")
    
    accelerator.print(model)
    # [NOTE] Define the optimizer
    param_groups = add_weight_decay(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=1e-3, eps=1e-6) # [TODO] Check whether weight_decay is right
    accelerator.print(optimizer)
    # [NOTE] Define train, valid, test dataset
    train_dataset = BIOPreTrain(args)
    accelerator.print(f"Loaded training examples {train_dataset.data.shape}")
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    # valid_dataset = BIOVal(args)
    # accelerator.print(f"Loaded validation examples {valid_dataset.data['input_ids'].shape}")
    # valid_dataloader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=4, 
    #     pin_memory=True,
    #     drop_last=False,
    # )
    model, optimizer, train_dataloader = \
        accelerator.prepare( model, optimizer, train_dataloader)
    # [NOTE] Define log writer
    # log_writer = SummaryWriter(log_dir=args.output_dir)
    
    """ Train the model """
    accelerator.print(f"Start training")
    start_time = time.time()
    train(model, train_dataloader, None, optimizer, None, args)
    # if accelerator.is_main_process:
    #     log_writer.flush()
    end_time = time.time()
    accelerator.print(f"Training finished in {end_time-start_time:.2f} seconds.")


if __name__ == "__main__":
    args = get_args_parser()
    
    accelerator.print(args)
    accelerator.gradient_accumulation_steps = args.accum_iter
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)


