import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import *
from process_emb import *
from torch.nn.parameter import Parameter

class Encoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.densenet121.features(x)
        x = F.relu(x)
        return x

class Attention(nn.Module):

    def __init__(self, k_size, v_size, affine_size=512):
        super().__init__()
        self.affine_k = nn.Linear(k_size, affine_size, bias=False)
        self.affine_v = nn.Linear(v_size, affine_size, bias=False)
        self.affine = nn.Linear(affine_size, 1, bias=False)

    def forward(self, k, v):
        # k: batch size x hidden size
        # v: batch size x spatial size x hidden size
        # z: batch size x spatial size
        content_v = self.affine_k(k).unsqueeze(1) + self.affine_v(v)
        z = self.affine(torch.tanh(content_v)).squeeze(2)
        alpha = torch.softmax(z, dim=1)
        context = (v * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class ClsAttention(nn.Module):

    def __init__(self, feat_size, num_classes):
        super().__init__()
        self.feat_size = feat_size
        self.num_classes = num_classes
        self.channel_w = nn.Conv2d(feat_size, num_classes, 1, bias=False)

    def forward(self, feats):
        # feats: batch size x feat size x H x W
        batch_size, feat_size, H, W = feats.size()
        att_maps = self.channel_w(feats)
        att_maps = torch.softmax(att_maps.view(batch_size, self.num_classes, -1), dim=2)
        feats_t = feats.view(batch_size, feat_size, H * W).permute(0, 2, 1)
        cls_feats = torch.bmm(att_maps, feats_t)
        return cls_feats

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        x = x.permute(0,2,1)
        self.weight = Parameter(torch.FloatTensor(x.size()[0], self.in_features, self.out_features).to('cuda'))
        self.reset_parameters_xavier()
        support = torch.bmm(x, self.weight)
        output = torch.bmm(adj, support)
        
        if self.bias is not None:
            return  (output + self.bias).permute(0,2,1)
        else:
            return  output.permute(0,2,1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionLayer(nn.Module):

    def __init__(self,in_size, state_size):
        super(GraphConvolutionLayer, self).__init__()
        self.in_size = in_size
        self.state_size = state_size

        self.condense = nn.Conv1d(in_size, state_size, 1)
        self.condense_norm = nn.BatchNorm1d(state_size)

        self.gcn_forward = GraphConvolution(in_size, state_size)
        self.gcn_backward = GraphConvolution(in_size, state_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.conv1d = nn.Conv1d(3*state_size, in_size, 1, bias=False)
        self.norm = nn.BatchNorm1d(in_size)

        self.test_conv = nn.Conv1d(state_size, in_size, 1, bias=False)
    def forward(self, x, fw_A, bw_A):
        
        states = x
        condensed_message = self.relu(self.condense_norm(self.condense(x)))
        fw_message = self.relu(self.gcn_forward(x, fw_A))
        bw_message = self.relu(self.gcn_backward(x, bw_A))
        update = torch.cat((condensed_message, fw_message, bw_message),dim=1)
        x = self.norm(self.conv1d(update))
        x = self.relu(x+states)

        return x


class GCN(nn.Module):

    def __init__(self, in_size, state_size):
        super(GCN_new, self).__init__()

        self.gcn1 = GraphConvolutionLayer(in_size, state_size)
        self.gcn2 = GraphConvolutionLayer(in_size, state_size)
        self.gcn3 = GraphConvolutionLayer(in_size, state_size)

    def forward(self, states, fw_A, bw_A):
        states = states.permute(0,2,1)
        states = self.gcn1(states, fw_A, bw_A)
        states = self.gcn2(states, fw_A, bw_A)
        states = self.gcn3(states, fw_A, bw_A)
        
        return states.permute(0,2,1)

class GCLayer(nn.Module):

    def __init__(self, in_size, state_size):
        super().__init__()
        self.condense = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.condense_norm = nn.BatchNorm1d(state_size)
        self.fw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.fw_norm = nn.BatchNorm1d(state_size)
        self.bw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.bw_norm = nn.BatchNorm1d(state_size)
        self.update = nn.Conv1d(3 * state_size, in_size, 1, bias=False)
        self.update_norm = nn.BatchNorm1d(in_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, states, fw_A, bw_A):
        # states: batch size x feat size x nodes
        condensed = self.relu(self.condense_norm(self.condense(states)))
        fw_msg = self.relu(self.fw_norm(self.fw_trans(states).bmm(fw_A)))
        bw_msg = self.relu(self.bw_norm(self.bw_trans(states).bmm(bw_A)))
        updated = self.update_norm(self.update(torch.cat((condensed, fw_msg, bw_msg), dim=1)))
        updated = self.relu(updated + states)
        return updated


class SentGCN(nn.Module):

    # random embeddings with dimension 256
    #def __init__(self, num_classes, fw_adj, bw_adj, vocab_size, embds, feat_size=1024, embed_size=256, hidden_size=512):

    # pretrained embeddings with dimension 200
    def __init__(self, num_classes, fw_adj, bw_adj, vocab_size, embds, feat_size=1024, embed_size=200, hidden_size=512):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.densenet121 = models.densenet121(pretrained=True)
        self.densenet121.classifier = nn.Linear(feat_size, num_classes)
        self.cls_atten = ClsAttention(feat_size, num_classes)
        
        self.gcn = GCN(feat_size, feat_size//4)
        self.atten = Attention(hidden_size, feat_size)

        # random embeddings
        #self.embed = nn.Embedding(vocab_size, embed_size)
        # pretrained embeddings
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embds.vectors, dtype=torch.float64))

        self.init_sent_h = nn.Linear(2 * feat_size, hidden_size)
        self.init_sent_c = nn.Linear(2 * feat_size, hidden_size)
        self.sent_lstm = nn.LSTMCell(2 * feat_size, hidden_size)
        self.word_lstm = nn.LSTMCell(embed_size + hidden_size + 2 * feat_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

        # Added layers
        size_of_node_feature = num_classes + 1
        self.conv1d = nn.Conv1d(2*size_of_node_feature, size_of_node_feature, 1, bias=False)
        self.norm = nn.BatchNorm1d(size_of_node_feature)
        self.relu = nn.ReLU(inplace=True)

        fw_D = torch.diag_embed(fw_adj.sum(dim=1))
        bw_D = torch.diag_embed(bw_adj.sum(dim=1))
        inv_sqrt_fw_D = fw_D.pow(-0.5)
        inv_sqrt_fw_D[torch.isinf(inv_sqrt_fw_D)] = 0
        inv_sqrt_bw_D = bw_D.pow(-0.5)
        inv_sqrt_bw_D[torch.isinf(inv_sqrt_bw_D)] = 0
        
        self.fw_A = inv_sqrt_fw_D.mm(fw_adj).mm(inv_sqrt_fw_D)
        self.bw_A = inv_sqrt_bw_D.mm(bw_adj).mm(inv_sqrt_bw_D)

    def forward(self, images1, images2, captions=None, update_masks=None, stop_id=None, max_sents=10, max_len=30):
        cnn_feats1 = self.densenet121.features(images1)
        cnn_feats2 = self.densenet121.features(images2)
        batch_size, _, h, w = cnn_feats1.size()
        fw_A = self.fw_A.repeat(batch_size, 1, 1)
        bw_A = self.bw_A.repeat(batch_size, 1, 1)
        
        if captions is not None:
            num_sents = captions.size(1)
            seq_len = captions.size(2)
        else:
            num_sents = max_sents
            seq_len = max_len

        global_feats1 = cnn_feats1.mean(dim=(2, 3))
        global_feats2 = cnn_feats2.mean(dim=(2, 3))
        cls_feats1 = self.cls_atten(cnn_feats1)
        cls_feats2 = self.cls_atten(cnn_feats2)
        node_feats1 = torch.cat((global_feats1.unsqueeze(1), cls_feats1), dim=1)
        node_feats2 = torch.cat((global_feats2.unsqueeze(1), cls_feats2), dim=1)
        
        node_states1 = self.gcn(node_feats1, fw_A, bw_A)
        node_states2 = self.gcn(node_feats2, fw_A, bw_A)

        sent_h = self.init_sent_h(torch.cat((global_feats1, global_feats2), dim=1))
        sent_c = self.init_sent_c(torch.cat((global_feats1, global_feats2), dim=1))
        word_h = cnn_feats1.new_zeros((batch_size, self.hidden_size), dtype=torch.float)
        word_c = cnn_feats1.new_zeros((batch_size, self.hidden_size), dtype=torch.float)

        logits = cnn_feats1.new_zeros((batch_size, num_sents, seq_len, self.vocab_size), dtype=torch.float)
        if captions is not None:
            #embeddings = self.embed(captions)
            embeddings = self.embed(captions).float()

            for k in range(num_sents):
                context1, alpha1 = self.atten(sent_h, node_states1)
                context2, alpha2 = self.atten(sent_h, node_states2)
                context = torch.cat((context1, context2), dim=1)
                
                sent_h, sent_c = self.sent_lstm(context, (sent_h, sent_c))
                seq_len_k = update_masks[:, k].sum(dim=1).max().item()

                for t in range(seq_len_k):
                    batch_mask = update_masks[:, k, t]
                    word_h_, word_c_ = self.word_lstm(torch.cat((embeddings[batch_mask, k, t], sent_h[batch_mask], context[batch_mask]), dim=1), 
                            (word_h[batch_mask], word_c[batch_mask]))
                    indices = [*batch_mask.unsqueeze(1).repeat(1, self.hidden_size).nonzero().t()]
                    word_h = word_h.index_put(indices, word_h_.view(-1))
                    word_c = word_c.index_put(indices, word_c_.view(-1))
                    logits[batch_mask, k, t] = self.fc(self.dropout(word_h[batch_mask]))

            return logits

        else:
            x_t = cnn_feats1.new_full((batch_size,), 1, dtype=torch.long)

            for k in range(num_sents):
                context1, alpha1 = self.atten(sent_h, node_states1)
                context2, alpha2 = self.atten(sent_h, node_states2)
                context = torch.cat((context1, context2), dim=1)
                sent_h, sent_c = self.sent_lstm(context, (sent_h, sent_c))
                
                for t in range(seq_len):
                    #embedding = self.embed(x_t)
                    embedding = self.embed(x_t).float()
                    word_h, word_c = self.word_lstm(torch.cat((embedding, sent_h, context), dim=1), (word_h, word_c))

                    logit = self.fc(word_h)
                    x_t = logit.argmax(dim=1)
                    logits[:, k, t] = logit

                    if x_t[0] == stop_id:
                        break
                    
            return logits.argmax(dim=3)
