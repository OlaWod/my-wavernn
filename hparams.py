
class Hparams():
    def __init__(self):
        # Paths
        self.wav_dir = '/home/jingyi/code/my-tacotron2/corpus/LJSpeech-1.1/wavs'
        self.data_dir = './data'
        self.ckpt_dir = "./output/ckpt"
        self.log_dir = "./output/log"
        self.result_dir = "./output/result"

        # Steps
        self.total_step = 1_000_000         # Total number of training steps
        self.restore_step = 0
        self.log_step = 10
        self.save_step = 25_000
        self.synth_step = 10

        # Melspectrogram settings
        self.sample_rate = 22050
        self.n_fft = 2048
        self.num_mels = 80
        self.hop_length = 275                    # 12.5ms - in line with Tacotron 2 paper
        self.win_length = 1100                   # 50ms - same reason as above
        self.fmin = 40
        self.min_level_db = -100

        self.bits = 9                            # bit depth of signal
        self.pad_len = 2                         # this will pad the input so that the resnet can 'see' wider than input length
        self.seq_len = self.hop_length * 5        # must be a multiple of hop_length

        # Model Hparams
        self.upsample_scales = (5, 5, 11)   # NB - this needs to correctly factorise hop_length
        self.n_res_block = 10
        self.rnn_dim = 512
        self.fc_dim = 512
        self.hidden_dim = 128
        self.res_out_dim = 128
        
        # Training
        self.val_size = 10               # How many unseen samples to put aside for testing
        self.batch_size = 32
        self.lr = 1e-4
        self.clip_grad_norm = 4              # set to None if no gradient clipping needed



        self.voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
        
        # Generating / Synthesizing
        self.voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
        self.voc_target = 11_000                 # target number of samples to be generated in each batch entry
        self.voc_overlap = 550                   # number of samples for crossfading between batches

hp = Hparams()
