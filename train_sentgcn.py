from data import Biview_MultiSent, sent_collate_fn
from sentgcn import SentGCN
from evaluate import evaluate
from my_build_vocab import Vocabulary
import os
import math
import argparse
import logging
import pickle
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import time
import statistics
from utils import *
from process_emb import WordEmbeddings, read_word_embeddings, relativize, relativize_data, text_to_id
from constants import FOLDER

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default= FOLDER + 'models')
    parser.add_argument('--output-dir', type=str, default= FOLDER + 'output/openi_top30')
    parser.add_argument('--pretrained', type=str, default= FOLDER + 'models/gcnclassifier_30keywords_train234val0test1_e15.pth')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset-dir', type=str, default= FOLDER + 'data/openi')
    parser.add_argument('--train-folds', type=str, default='012')
    parser.add_argument('--val-folds', type=str, default='3')
    parser.add_argument('--test-folds', type=str, default='4')
    parser.add_argument('--report-path', type=str, default= FOLDER + 'data/reports.json')
    parser.add_argument('--vocab-path', type=str, default= FOLDER + 'data/vocab.pkl')
    parser.add_argument('--label-path', type=str, default= FOLDER + 'data/label_dict.json')
    parser.add_argument('--log-dir', type=str, default= FOLDER + 'logs/openi_top30')
    parser.add_argument('--log-freq', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--encoder-lr', type=float, default=1e-6)
    parser.add_argument('--decoder-lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--num_classes', type=int, default=30)
    parser.add_argument('--clip-value', type=float, default=5.0)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.log_dir, args.name + '.log'), level=logging.INFO)
    print('------------------------Model and Training Details--------------------------')
    print(args)
    for k, v in vars(args).items():
        logging.info('{}: {}'.format(k, v))

    writer = SummaryWriter(log_dir=os.path.join( FOLDER + 'runs/openi_top30', args.name))

    gpus = [int(_) for _ in list(args.gpus)]
    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    print(device) 
    torch.manual_seed(args.seed)

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    embs = read_word_embeddings( FOLDER + "data/embedding/glove.6B.200d-relativized.txt")
    
    train_set = Biview_MultiSent('train', args.dataset_dir, args.train_folds, args.report_path, args.vocab_path, args.label_path, embs)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=sent_collate_fn)
    val_set = Biview_MultiSent('val', args.dataset_dir, args.val_folds, args.report_path, args.vocab_path, args.label_path, embs)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=sent_collate_fn)
    test_set = Biview_MultiSent('test', args.dataset_dir, args.test_folds, args.report_path, args.vocab_path, args.label_path, embs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=sent_collate_fn)

    with open( FOLDER + 'data/openi_top30/auxillary_openi_matrix_30nodes.txt','r') as matrix_file:
        adjacency_matrix = [[int(num) for num in line.split(',')] for line in matrix_file]

    fw_adj = torch.tensor(adjacency_matrix, dtype=torch.float, device=device)
    identity_matrix = torch.eye(args.num_classes+1, device=device)
    bw_adj = fw_adj.t()
    fw_adj = fw_adj.add(identity_matrix)
    bw_adj = bw_adj.add(identity_matrix)

    model = SentGCN(args.num_classes, fw_adj, bw_adj, len(vocab), embs).to(device)

    for param in model.densenet121.parameters():
        param.requires_grad = False
    for param in model.cls_atten.parameters():
        param.requires_grad = False
    for param in model.gcn.parameters():
        param.requires_grad = False
    model.densenet121.eval()
    model.gcn.eval()
    decoder_params = \
        list(model.atten.parameters()) + \
        list(model.embed.parameters()) + \
        list(model.init_sent_h.parameters()) + \
        list(model.init_sent_c.parameters()) + \
        list(model.sent_lstm.parameters()) + \
        list(model.word_lstm.parameters()) + \
        list(model.fc.parameters())

    CELoss = nn.CrossEntropyLoss(reduction='none')

    optimizer = torch.optim.Adam(decoder_params, lr=args.decoder_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150], gamma=0.1)
    
    if args.pretrained:
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['model_state_dict']
        state_dict = model.state_dict()
        state_dict.update({k: v for k, v in pretrained_state_dict.items() if k in state_dict and 'fc' not in k})
        model.load_state_dict(state_dict)
    
    start_epoch = 1
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)

    num_steps = math.ceil(len(train_set) / args.batch_size)

    val_gts = {}
    test_gts = {}

    start_time = time.time()

    current_best_val_epoch = -1
    current_best_val_average = -1
    best_test_b1, best_test_b2, best_test_b3, best_test_b4, best_test_rouge, best_test_cider = -1, -1, -1, -1, -1, -1
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.dropout.train()
        epoch_loss = 0.0
        print('------------------------Training for Epoch {}---------------------------'.format(epoch))
        print('Learning rate {:.7f}'.format(optimizer.param_groups[0]['lr']))

        for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(train_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            captions = captions.to(device)
            loss_masks = loss_masks.to(device)
            update_masks = update_masks.to(device)
            optimizer.zero_grad()
            logits = model(images1, images2, captions[:, :, :-1], update_masks)
            logits = logits.permute(0, 3, 1, 2).contiguous()
            captions = captions[:, :, 1:].contiguous()
            loss_masks = loss_masks[:, :,1 :].contiguous()
            loss = CELoss(logits, captions)
            loss = loss.masked_select(loss_masks).mean()
            loss.backward()
            epoch_loss += loss.item()
            clip_grad_value_(model.parameters(), args.clip_value)
            optimizer.step()

        epoch_loss /= num_steps
        print('Epoch {}/{}, Loss {:.4f}'.format(epoch, args.num_epochs, epoch_loss))
        writer.add_scalar('loss', epoch_loss, epoch)

        scheduler.step(epoch)
        print('Epoch {}, using {} mins.'.format(epoch, (time.time()-start_time)/60))
        if epoch % args.log_freq == 0:

            save_fname = os.path.join(args.model_dir, '{}_e{}.pth'.format(args.name, epoch))
            if len(gpus) > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_fname)

            model.dropout.eval()
            val_res = {}
            for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(val_loader):
                images1 = images1.to(device)
                images2 = images2.to(device)
                #preds = model(images1, images2, stop_id=vocab('.'))
                preds = model(images1, images2, stop_id=embs.word_indexer.index_of('.'))

                caseid = caseids[0]
                val_res[caseid] = ['']
                pred = preds[0].detach().cpu()
                for isent in range(pred.size(0)):
                    words = []
                    for wid in pred[isent].tolist():
                        #w = vocab.idx2word[wid]
                        w = embs.word_indexer.get_object(wid)
                        if w == '<start>' or w == '<pad>':
                            continue
                        if w == '<end>':
                            break
                        words.append(w)
                    val_res[caseid][0] += ' '.join(words)
                    val_res[caseid][0] += ' '

                if epoch == start_epoch:
                    val_gts[caseid] = ['']
                    cap = captions[0]
                    for isent in range(cap.size(0)):
                        words = []
                        for wid in cap[isent, 1:].tolist():
                            #w = vocab.idx2word[wid]
                            w = embs.word_indexer.get_object(wid)
                            if w == '<start>' or w == '<pad>':
                                continue
                            if w == '<end>':
                                break
                            words.append(w)
                        val_gts[caseid][0] += ' '.join(words)
                        val_gts[caseid][0] += ' '

            scores = evaluate(val_gts, val_res)
            writer.add_scalar('VAL BLEU 1', scores['Bleu_1'], epoch)
            writer.add_scalar('VAL BLEU 2', scores['Bleu_2'], epoch)
            writer.add_scalar('VAL BLEU 3', scores['Bleu_3'], epoch)
            writer.add_scalar('VAL BLEU 4', scores['Bleu_4'], epoch)
            writer.add_scalar('VAL ROUGE_L', scores['ROUGE_L'], epoch)
            writer.add_scalar('VAL CIDEr', scores['CIDEr'], epoch)
            # writer.add_scalar('VAL Meteor', scores['METEOR'], epoch)

            temp_val_average = statistics.mean([scores['Bleu_1'], scores['Bleu_2'], scores['Bleu_3'], scores['Bleu_4'], scores['ROUGE_L'], scores['CIDEr']])
            current_best_val_epoch = epoch if temp_val_average>current_best_val_average else current_best_val_epoch
            current_best_val_average = temp_val_average if temp_val_average>current_best_val_average else current_best_val_average
            print('Current best epoch: {}, average score on val set: {}'.format(current_best_val_epoch, current_best_val_average))

            test_res = {}
            for i, (images1, images2, labels, captions, loss_masks, update_masks, caseids) in enumerate(test_loader):
                images1 = images1.to(device)
                images2 = images2.to(device)
                #preds = model(images1, images2, stop_id=vocab('.'))
                preds = model(images1, images2, stop_id=embs.word_indexer.index_of('.'))

                
                caseid = caseids[0]
                test_res[caseid] = ['']
                pred = preds[0].detach().cpu()
                for isent in range(pred.size(0)):
                    words = []
                    for wid in pred[isent].tolist():
                        #w = vocab.idx2word[wid]
                        w = embs.word_indexer.get_object(wid)
                        if w == '<start>' or w == '<pad>':
                            continue
                        if w == '<end>':
                            break
                        words.append(w)
                    test_res[caseid][0] += ' '.join(words)
                    test_res[caseid][0] += ' '

                if epoch == start_epoch:
                    test_gts[caseid] = ['']
                    cap = captions[0]
                    for isent in range(cap.size(0)):
                        words = []
                        for wid in cap[isent, 1:].tolist():
                            #w = vocab.idx2word[wid]
                            w = embs.word_indexer.get_object(wid)
                            if w == '<start>' or w == '<pad>':
                                continue
                            if w == '<end>':                           
                                break
                            words.append(w)
                        test_gts[caseid][0] += ' '.join(words)
                        test_gts[caseid][0] += ' '

            scores = evaluate(test_gts, test_res)
            writer.add_scalar('TEST BLEU 1', scores['Bleu_1'], epoch)
            writer.add_scalar('TEST BLEU 2', scores['Bleu_2'], epoch)
            writer.add_scalar('TEST BLEU 3', scores['Bleu_3'], epoch)
            writer.add_scalar('TEST BLEU 4', scores['Bleu_4'], epoch)
            writer.add_scalar('TEST ROUGE_L', scores['ROUGE_L'], epoch)
            writer.add_scalar('TEST CIDEr', scores['CIDEr'], epoch)
            # writer.add_scalar('TEST Meteor', scores['METEOR'], epoch)

            if current_best_val_epoch == epoch:
                best_test_b1, best_test_b2, best_test_b3, best_test_b4, best_test_rouge, best_test_cider = scores['Bleu_1'], scores['Bleu_2'], scores['Bleu_3'], scores['Bleu_4'], scores['ROUGE_L'], scores['CIDEr']
            print('Current best epoch: {}, current best test scores:'.format(current_best_val_epoch))
            print('BLEU_1: %.7f' % best_test_b1)
            print('BLEU_2: %.7f' % best_test_b2)
            print('BLEU_3: %.7f' % best_test_b3)
            print('BLEU_4: %.7f' % best_test_b4)            
            print('ROUGE_L: %.7f' % best_test_rouge)
            print('CIDEr: %.7f' % best_test_cider)

            with open(os.path.join(args.output_dir, '{}_test_e{}.csv'.format(args.name, epoch)), 'w') as f1:
                with open(os.path.join(args.output_dir, '{}_test_gts.csv'.format(args.name)), 'w') as f2:
                    for caseid in test_res.keys():
                        f1.write(test_res[caseid][0] + '\n')
                        f2.write(test_gts[caseid][0] + '\n')
    writer.close()
