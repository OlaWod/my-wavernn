import os
import pickle
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

from hparams import hp
from utils.spec import melspectrogram
from utils.display import progbar, stream
from utils.tools import load_wav, float_2_label


def convert_file(path: Path):
    y = load_wav(path)
    
    mel = melspectrogram(y)
    quant = float_2_label(y, hp.bits)

    return mel.astype(np.float32), quant.astype(np.int64)


def process_wav(path: Path):
    basename = path.stem
    mel, quant = convert_file(path)

    np.save(os.path.join(hp.data_dir,'mel',f'{basename}.npy'), mel, allow_pickle=False)
    np.save(os.path.join(hp.data_dir,'quant',f'{basename}.npy'), quant, allow_pickle=False)

    return basename


if __name__ == '__main__':
    wav_paths = list(Path(hp.wav_dir).rglob('*.wav'))
    print(f'\n{len(wav_paths)} wav files found in "{hp.wav_dir}"\n')

    if len(wav_paths) == 0:
        print('Please point wav_dir in hparams.py to your dataset.')
    
    else:
        os.makedirs(hp.data_dir, exist_ok=True)
        os.makedirs(os.path.join(hp.data_dir,'mel'), exist_ok=True)
        os.makedirs(os.path.join(hp.data_dir,'quant'), exist_ok=True)
        
        pool = Pool(processes=cpu_count()-1)
        basenames = []

        for i, basename in enumerate(pool.imap_unordered(process_wav, wav_paths), 1):
            basenames.append(basename)
            bar = progbar(i, len(wav_paths))
            message = f'{bar} {i}/{len(wav_paths)} '
            stream(message)

        with open(hp.data_dir+'/basenames.pkl', 'wb') as f:
            pickle.dump(basenames, f)

        print('\n\nCompleted.\n')
