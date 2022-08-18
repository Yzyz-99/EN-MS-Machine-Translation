import os
import shutil
import re
import sentencepiece as spm
import subprocess

class CorpusProcessor:
    counter = 1
    
    def __init__(self):
        self.src_lang = 'en'
        self.tgt_lang = 'ms'
        self.spm_model = spm.SentencePieceProcessor(model_file='model/spm8000.model')
        self.binpath = 'data-bin/en-ms'
        self.srcdict = 'data-bin/en-ms/dict.en.txt'
        self.tgtdict = 'data-bin/en-ms/dict.ms.txt'

    @staticmethod
    def getCounter():
        return CorpusProcessor.counter

    def preprocess(self, s, IPAddr):
        lines = []
        for line in s:
            line = line.strip()
            line = re.sub(r"\([^()]*\)", "", line) # remove ([text])
            line = line.replace('-', '') # remove '-'
            line = re.sub('([.,;!?()\"])', r' \1 ', line) # keep punctuation
            line = line.strip()
            line = self.spm_model.encode(line, out_type=str)
            lines.append(line)

        print(lines)
        with open(f'data/{IPAddr}.en', 'w',  encoding="utf-8") as outfile:
            for line in lines:
                outfile.write(' '.join(line))
                outfile.write('\n')
            
        self.binpath = f'{self.binpath}/{IPAddr}'

        if (CorpusProcessor.counter == 1 and os.path.exists(self.binpath)):
            shutil.rmtree(self.binpath)

        if not os.path.exists(self.binpath):
            os.makedirs(self.binpath)
                
        subprocess.call(
            ['python', '-m', 'fairseq_cli.preprocess',
            '--source-lang', self.src_lang,
            '--target-lang', self.tgt_lang,
            '--testpref', f'data/{IPAddr}',
            '--destdir', self.binpath,
            '--srcdict', self.srcdict,
            '--tgtdict', self.tgtdict,
            '--only-source',
            '--workers', '2'], shell=True)
        
        if(os.path.exists(f'{self.binpath}/test.en-ms.en.bin')):
            os.rename(f'{self.binpath}/test.en-ms.en.bin', f'{self.binpath}/test({CorpusProcessor.counter}).en-ms.en.bin')
            os.rename(f'{self.binpath}/test.en-ms.en.idx', f'{self.binpath}/test({CorpusProcessor.counter}).en-ms.en.idx')
        CorpusProcessor.counter += 1
        return s, CorpusProcessor.counter
        