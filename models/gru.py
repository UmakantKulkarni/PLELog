from module.Attention import *
from module.CPUEmbedding import *
from module.Common import *


class AttGRUModel(nn.Module):
    # Dispose Loggers.
    _logger = logging.getLogger('AttGRU')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'AttGRU.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for Attention-Based GRU succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return AttGRUModel._logger

    def __init__(self, vocab, lstm_layers, lstm_hiddens, dropout=0):
        super(AttGRUModel, self).__init__()
        self.dropout = dropout
        self.logger.info('==== Model Parameters ====')
        vocab_size, word_dims = vocab.vocab_size, vocab.word_dim
        self.word_embed = CPUEmbedding(vocab_size, word_dims, padding_idx=vocab_size - 1)
        self.word_embed.weight.data.copy_(torch.from_numpy(vocab.embeddings))
        self.word_embed.weight.requires_grad = False
        self.logger.info('Input Dimension: %d' % word_dims)
        self.logger.info('Hidden Size: %d' % lstm_hiddens)
        self.logger.info('Num Layers: %d' % lstm_layers)
        self.logger.info('Dropout %.3f' % dropout)
        self.rnn = nn.GRU(input_size=word_dims, hidden_size=lstm_hiddens, num_layers=lstm_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.sent_dim = 2 * lstm_hiddens
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)
        self.proj = NonLinear(self.sent_dim, 2)

    def reset_word_embed_weight(self, vocab, pretrained_embedding):
        vocab_size, word_dims = pretrained_embedding.shape
        self.word_embed = CPUEmbedding(vocab.vocab_size, word_dims, padding_idx=vocab.PAD)
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

    def forward(self, inputs):
        words, masks, word_len = inputs
        embed = self.word_embed(words)
        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        if device.type == 'cuda':
            embed = embed.cuda(device)
        else:
            embed = embed.to(device)
        batch_size = embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        hiddens, state = self.rnn(embed)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()

        sent_probs = sent_probs.contiguous()
        if sent_probs.dim() == 1:
            total = sent_probs.numel()
            if total % batch_size == 0:
                sent_probs = sent_probs.view(batch_size, total // batch_size)
            elif batch_size == 1:
                sent_probs = sent_probs.view(1, total)
            else:
                raise RuntimeError(
                    f"Cannot reshape 1D attention weights of length {total} for batch {batch_size}")
        else:
            if sent_probs.size(0) != batch_size:
                transposed = False
                for axis in range(1, sent_probs.dim()):
                    if sent_probs.size(axis) == batch_size:
                        sent_probs = sent_probs.transpose(0, axis).contiguous()
                        transposed = True
                        break
                if not transposed and sent_probs.numel() % batch_size != 0:
                    raise RuntimeError(
                        f"Cannot align attention weights of shape {sent_probs.shape} with batch {batch_size}")
            total = sent_probs.numel()
            if total % batch_size != 0:
                raise RuntimeError(
                    f"Cannot reshape attention weights of shape {sent_probs.shape} for batch {batch_size}")
            sent_probs = sent_probs.view(batch_size, total // batch_size)

        if sent_probs.size(1) != srclen:
            if srclen > 0 and sent_probs.size(1) % srclen == 0:
                sent_probs = sent_probs.view(batch_size, srclen, -1).mean(dim=-1)
            elif sent_probs.size(1) > srclen:
                sent_probs = sent_probs[:, :srclen]
            elif srclen > sent_probs.size(1):
                pad = srclen - sent_probs.size(1)
                sent_probs = torch.cat(
                    [sent_probs, sent_probs.new_zeros(batch_size, pad)], dim=1)

        if masks.dim() > 2:
            masks = masks.view(batch_size, -1)
        if masks.size(1) != srclen:
            if masks.size(1) > srclen:
                masks = masks[:, :srclen]
            else:
                pad = srclen - masks.size(1)
                if pad > 0:
                    masks = torch.cat([masks, masks.new_zeros(batch_size, pad)], dim=1)

        sent_probs = sent_probs * masks
        denom = sent_probs.sum(dim=1, keepdim=True)
        zero_mask = denom <= 0
        denom = denom.clamp_min(1e-13)
        sent_probs = sent_probs / denom
        if zero_mask.any() and srclen > 0:
            zero_mask = zero_mask.expand(-1, srclen)
            fallback = masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-13)
            sent_probs = torch.where(zero_mask, fallback, sent_probs)

        sent_probs = sent_probs.unsqueeze(-1)
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)
        outputs = self.proj(represents)
        return outputs  # , represents
