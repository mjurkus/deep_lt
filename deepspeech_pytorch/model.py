import math
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from warpctc_pytorch import CTCLoss


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(pl.LightningModule):
    def __init__(
            self,
            hparams: dict,
            decoder=None,
    ):
        super(DeepSpeech, self).__init__()

        self.hparams = hparams
        self.decoder = decoder
        self.criterion = CTCLoss()

        model_hparams = self.hparams['model']
        num_classes = self.hparams['num_classes']
        self.hidden_size = model_hparams['hidden_size']
        self.hidden_layers = model_hparams['hidden_layers']

        self.audio_conf = self.hparams['audio_conf']
        sample_rate = self.audio_conf["sample_rate"]
        window_size = self.audio_conf["window_size"]

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=self.hidden_size,
                       bidirectional=True, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(self.hidden_layers - 1):
            rnn = BatchRNN(input_size=self.hidden_size, hidden_size=self.hidden_size,
                           bidirectional=True)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def configure_optimizers(self):
        o = self.hparams['optimizer']
        self.optimizer = optim.AdamW(
            self.parameters(), lr=float(o['learning_rate']),
            betas=eval(o['betas']),
            eps=float(o['eps']),
            weight_decay=float(o['weight_decay']),
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.50,
            patience=6,
        )

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        out, output_sizes = self.forward(inputs, input_sizes)
        out = out.transpose(0, 1)  # TxNxH
        out = out.float()

        # https://github.com/SeanNaren/warp-ctc/issues/62
        device = torch.device('cpu')
        loss = self.criterion(
            out,
            targets.to(device),
            output_sizes.to(device),
            target_sizes.to(device),
        )
        loss = loss / inputs.size(0)

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        out, output_sizes = self.forward(inputs, input_sizes)
        out = out.transpose(0, 1)  # TxNxH
        out = out.float()

        # https://github.com/SeanNaren/warp-ctc/issues/62
        device = torch.device('cpu')
        val_loss = self.criterion(
            out,
            targets.to(device),
            output_sizes.to(device),
            target_sizes.to(device)
        )
        val_loss = val_loss / inputs.size(0)

        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        decoded_output, _ = self.decoder.decode(out, output_sizes)
        target_strings = self.decoder.convert_to_strings(split_targets)

        verbose_counter, total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]

            wer_inst = self.decoder.wer(transcript, reference)
            cer_inst = self.decoder.cer(transcript, reference)

            if self.hparams['verbose'] and verbose_counter <= 2:
                verbose_counter += 1
                log = f"Ref: {reference.lower()}\n" \
                      f"Hyp: {transcript.lower()}"

                metadata = {
                    "wer": float(wer_inst) / len(reference.split() * 100),
                    "cer": float(cer_inst) / len(reference.replace(' ', '')) * 100
                }

                self.logger.experiment.log_text(text=log, metadata=metadata)

            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))

        wer = float(total_wer) / num_tokens * 100
        cer = float(total_cer) / num_chars * 100

        self.log_dict(
            {
                "val_loss": val_loss,
                "wer": wer,
                "cer": cer
            }
        )

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_wer = torch.stack([torch.tensor(x['wer']) for x in outputs]).mean()
        avg_cer = torch.stack([torch.tensor(x['cer']) for x in outputs]).mean()

        self.log_dict(
            {
                'val_loss': avg_loss,
                'wer': avg_wer,
                'cer': avg_cer
            }
        )

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
