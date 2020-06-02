import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from dgl.nn.pytorch import GATConv
from utils import create_batched_graphs
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden, mask=None):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(image_features)  # (batch_size, N, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, N)
        if mask is not None:
            # where the mask == 1, fill with value,
            # The mask we receive has ones where an object is, so we inverse it.
            att.masked_fill_(~mask, float('-inf'))
        alpha = self.softmax(att)  # (batch_size, N)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)
        return attention_weighted_encoding


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, gat_out_dim, vocab_size, features_dim=2048,
                 graph_features_dim=512, dropout=0.5, gat_num_heads=1):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(Decoder, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.gat = GATConv(graph_features_dim, gat_out_dim, gat_num_heads, feat_drop=dropout, attn_drop=dropout)

        # cascade attention network
        self.cascade1_attention = Attention(gat_out_dim, decoder_dim, attention_dim)
        self.cascade2_attention = Attention(features_dim, decoder_dim + gat_out_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = nn.LSTMCell(embed_dim + features_dim + gat_out_dim + decoder_dim,
                                              decoder_dim, bias=True)  # top down attention LSTMCell
        self.language_model = nn.LSTMCell(features_dim + gat_out_dim + decoder_dim, decoder_dim, bias=True)  # language model LSTMCell
        self.fc1 = weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.fc = weight_norm(nn.Linear(decoder_dim, vocab_size))  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param batch_size: size of the batch
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).to(device)
        return h, c

    def forward(self, image_features, object_features, relation_features, object_mask, relation_mask, pair_ids,
                encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param graph_features: encoded images as graphs, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param graph_mask: mask for the graph_features, shows were non empty features are
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size

        # Flatten image
        image_features_mean = image_features.mean(1).to(device)  # (batch_size, num_pixels, encoder_dim)
        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        object_features = object_features[sort_ind]
        relation_features = relation_features[sort_ind]
        object_mask = object_mask[sort_ind]
        relation_mask = relation_mask[sort_ind]
        pair_ids = pair_ids[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        graphs = create_batched_graphs(object_features, object_mask, relation_features, relation_mask, pair_ids)
        graphs = graphs.to(device)

        graph_features = self.gat(graphs, graphs.ndata['x'])
        graph_features = torch.split(graph_features, graphs.batch_num_nodes)
        graph_features = pad_sequence(graph_features, batch_first=True).squeeze(2)
        graph_mask = graph_features.sum(dim=-1) != 0

        graph_features_mean = graph_features.sum(dim=1) / graph_mask.sum(dim=1, keepdim=True)
        graph_features_mean = graph_features_mean.to(device)
        print(torch.any((graph_mask.sum(dim=1, keepdim=True) == 0)))
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        predictions1 = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        
        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up 
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model 
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h1, c1 = self.top_down_attention(torch.cat([h2[:batch_size_t],
                                                        image_features_mean[:batch_size_t],
                                                        graph_features_mean[:batch_size_t],
                                                        embeddings[:batch_size_t, t, :]], dim=1),
                                             (h1[:batch_size_t], c1[:batch_size_t]))
            graph_weighted_enc = self.cascade1_attention(graph_features[:batch_size_t], h1[:batch_size_t],
                                                         mask=graph_mask[:batch_size_t])
            img_weighted_enc = self.cascade2_attention(image_features[:batch_size_t],
                                                       torch.cat([h1[:batch_size_t], graph_weighted_enc[:batch_size_t]],
                                                                 dim=1))
            preds1 = self.fc1(self.dropout(h1))
            h2, c2 = self.language_model(
                torch.cat([graph_weighted_enc[:batch_size_t], img_weighted_enc[:batch_size_t], h1[:batch_size_t]], dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]))
            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            predictions1[:batch_size_t, t, :] = preds1

        return predictions, predictions1, encoded_captions, decode_lengths, sort_ind

