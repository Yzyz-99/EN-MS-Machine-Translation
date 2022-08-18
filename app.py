import socket

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
from argparse import Namespace
from fairseq import utils

from flask import Flask, render_template, request
from corpus_processor import CorpusProcessor
from model import Seq2Seq
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from fairseq.models.transformer import (
    TransformerEncoder, 
    TransformerDecoder,
)

app = Flask(__name__)
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        english = request.form['english']
        english = english.split('\n')

        try:
            cp = CorpusProcessor()
            english, counter = cp.preprocess(english, IPAddr)

            config = Namespace(
            datadir = "data-bin/en-ms",
            source_lang = "en",
            target_lang = "ms",

            # cpu threads when fetching & processing data.
            num_workers=2,  
            # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
            max_tokens=4096,
            accum_steps=4,

            # beam size for beam search
            beam=5, 
            # generate sequences of maximum length ax + b, where x is the source length
            max_len_a=1.2, 
            max_len_b=10,
            # when decoding, post process sentence by removing sentencepiece symbols.
            post_process = "sentencepiece",
            )

            task_cfg = TranslationConfig(
            data=config.datadir,
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            train_subset="train",
            required_seq_len_multiple=8,
            dataset_impl="mmap",
            upsample_primary=1,
            )

            task = TranslationTask.setup_task(task_cfg)

            def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
                batch_iterator = task.get_batch_iterator(
                dataset=task.dataset(split),
                max_tokens=max_tokens,
                max_sentences=None,
                max_positions=utils.resolve_max_positions(
                    task.max_positions(),
                    max_tokens,
                ),
                ignore_invalid_inputs=True,
                seed=73,
                num_workers=num_workers,
                epoch=epoch,
                disable_iterator_cache=not cached,
                )
                return batch_iterator

            def build_model(args, task):
                # build a model instance based on hyperparameters
                src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

                # token embeddings
                encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
                decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())

                encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens) # transformer encoder
                decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens) # transformer decode

                # sequence to sequence model
                model = Seq2Seq(args, encoder, decoder)

                def init_params(module):
                    from fairseq.modules import MultiheadAttention
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(mean=0.0, std=0.02)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    if isinstance(module, nn.Embedding):
                        module.weight.data.normal_(mean=0.0, std=0.02)
                        if module.padding_idx is not None:
                            module.weight.data[module.padding_idx].zero_()
                    if isinstance(module, MultiheadAttention):
                        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
                        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
                        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
                    if isinstance(module, nn.RNNBase):
                        for name, param in module.named_parameters():
                            if "weight" in name or "bias" in name:
                                param.data.uniform_(-0.1, 0.1)

                # weight initialization
                model.apply(init_params)

                # load trained model
                check = torch.load("model/avg_last_5_checkpoint.pt")
                model.load_state_dict(check["model"])
                return model

            arch_args = Namespace(
            encoder_embed_dim=512,
            encoder_ffn_embed_dim=2048,
            encoder_layers=6,
            decoder_embed_dim=512,
            decoder_ffn_embed_dim=2048,
            decoder_layers=6,
            share_decoder_input_output_embed=True,
            dropout=0.1,
            )

            # parameters for transformer only
            def add_transformer_args(args):
                args.encoder_attention_heads=8
                args.encoder_normalize_before=True

                args.decoder_attention_heads=8
                args.decoder_normalize_before=True

                args.activation_fn="relu"
                args.max_source_positions=1024
                args.max_target_positions=1024

                from fairseq.models.transformer import base_architecture 
                base_architecture(arch_args)
            
            add_transformer_args(arch_args)
            model = build_model(arch_args, task)

            
            # fairseq's beam search generator
            # given model and input seqeunce, produce translation hypotheses by beam search
            sequence_generator = task.build_generator([model], config)    

            def decode(toks, dictionary):
                # convert from Tensor to human readable sentence
                s = dictionary.string(
                    toks.int().cpu(),
                    config.post_process,
                )
                return s if s else "<unk>"

            def inference_step(sample, model, task, sequence_generator):
                gen_out = sequence_generator.generate([model], sample)
                srcs = []
                hyps = []
                refs = []
                for i in range(len(gen_out)):
                    # for each sample, collect the input, hypothesis and reference, later be used to calculate BLEU
                    srcs.append(decode(
                        utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()), 
                        task.source_dictionary,
                    ))
                    hyps.append(decode(
                        gen_out[i][0]["tokens"], # 0 indicates using the top hypothesis in beam
                        task.target_dictionary,
                    ))
                return srcs, hyps

            def generate_prediction(task, split=f'{IPAddr}/test({counter-1})'):
                task.load_dataset(split=split, epoch=1)
                itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)

                idxs = []
                hyps = []

                model.eval()
                progress = tqdm.tqdm(itr, desc=f"prediction")
                with torch.no_grad():
                    for i, sample in enumerate(progress):

                        # do inference
                        s, h= inference_step(sample, model, task, sequence_generator)

                        hyps.extend(h)
                        idxs.extend(list(sample['id']))

                # sort based on the order before preprocess
                result = [x for _,x in sorted(zip(idxs,hyps))]
                return result

            def merge(list1, list2):
                merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
                return merged_list
      
            result = generate_prediction(task)
            result = merge(result, english)
            print(result)
            return render_template('result.html', result=result)

        except Exception as e:
            print(e)
            return render_template('main.html', error=e)

    return render_template('main.html', error=None)

if __name__ == '__main__':
    app.run()