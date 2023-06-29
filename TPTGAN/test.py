# 计算测试集数据
import torch

# from scipy.io import wavfile
# import numpy as np
from torch.utils.data import DataLoader

from eval_composite import eval_composite
from helper_funcs import numParams

# import librosa
import datetime

from AudioData import EvalDataset, EvalCollate
from new_model import Net
import h5py
import os


# test function
def evaluate(net, eval_loader):
    net.eval()

    print('********Starting metrics evaluation on test dataset**********')
    total_stoi = 0.0
    total_ssnr = 0.0
    total_pesq = 0.0
    total_csig = 0.0
    total_cbak = 0.0
    total_covl = 0.0

    with torch.no_grad():
        count, total_eval_loss = 0, 0.0
        for k, (features, labels) in enumerate(eval_loader):
            print('\rprocess = [{}/{}]'.format(k + 1, len(eval_loader)), end="")
            features = features.cuda()  # [1, 1, num_frames,frame_size]
            labels = labels.cuda()  # [signal_len, ]

            output = net(features)  # [1, 1, sig_len_recover]
            output = output.squeeze()  # [sig_len_recover, ]

            # keep length same (output label)
            output = output[:labels.shape[-1]]

            eval_loss = torch.mean((output - labels) ** 2)
            total_eval_loss += eval_loss.data.item()

            est_sp = output.cpu().numpy()
            cln_raw = labels.cpu().numpy()

            eval_metric = eval_composite(cln_raw, est_sp, sr)

            # st = get_stoi(cln_raw, est_sp, sr)
            # pe = get_pesq(cln_raw, est_sp, sr)
            # sn = snr(cln_raw, est_sp)
            total_pesq += eval_metric['pesq']
            total_ssnr += eval_metric['ssnr']
            total_stoi += eval_metric['stoi']
            total_cbak += eval_metric['cbak']
            total_csig += eval_metric['csig']
            total_covl += eval_metric['covl']

            # wavfile.write(os.path.join(audio_file_save, os.path.basename(file_list[k])), sr, est_sp.astype(np.float32)) 原项目
            # wavfile.write(os.path.join(audio_file_save, str(k) + '.wav'), sr, est_sp.astype(np.float32)) 本项目
            count += 1
        avg_eval_loss = total_eval_loss / count

    return avg_eval_loss, total_stoi / count, total_pesq / count, total_ssnr / count, total_csig / count, total_cbak / count, total_covl / count

    # wavfile.write(os.path.join(audio_file_save, str(k) + '.wav'), sr, est_sp.astype(np.float32))


# evaluate(model, test_loader)
# print('Audio process Finished!')


def eva_noisy(file_path):
    print('********Starting metrics evaluation on raw noisy data**********')
    total_stoi = 0.0
    total_ssnr = 0.0
    total_pesq = 0.0
    total_csig = 0.0
    total_cbak = 0.0
    total_covl = 0.0
    count = 0

    with open(file_path, 'r') as eva_file_list:
        file_list = [line.strip() for line in eva_file_list.readlines()]

    for i in range(len(file_list)):
        filename = file_list[i]
        reader = h5py.File(filename, 'r')

        noisy_raw = reader['noisy_raw'][:]
        cln_raw = reader['clean_raw'][:]

        eval_metric = eval_composite(cln_raw, noisy_raw, sr)

        total_pesq += eval_metric['pesq']
        total_ssnr += eval_metric['ssnr']
        total_stoi += eval_metric['stoi']
        total_cbak += eval_metric['cbak']
        total_csig += eval_metric['csig']
        total_covl += eval_metric['covl']

        count += 1

    return total_stoi / count, total_pesq / count, total_ssnr / count, total_cbak / count, total_csig / count, total_covl / count


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sr = 16000
root = os.getcwd()
# file_name = 'psquare_17.5'
test_file_list_path = os.path.join(root, 'data_bank_voice/test_file_list_bank.txt')
test_results_file = "test_results_conformer{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# test_file_mix_path = os.path.join(root, 'test_audio/mix')
# audio_file_save = os.path.join(root, 'save_audio/enhancedAudio')
# if not os.path.isdir(audio_file_save):
#     os.makedirs(audio_file_save)
#
# file_list = sorted(os.listdir(test_file_list_path))
# with open(test_file_list_path, 'r') as test_file_list:
#     file_list = [line.strip() for line in test_file_list.readlines()]
# audio_name = os.path.basename(file_list[0])
# print('The audio to be enhanced:')
# print(file_list)
# count = 0
#
# for audio in file_list:
#     count = count + 1
#     # clean_name = audio.split(".")[0] + '.wav'
#     file_name = '%s_%d' % ('val_mix', count)
#     train_writer = h5py.File(test_file_mix_path + '/' + file_name, 'w')
#
#     noisy_audio, sr = librosa.load(os.path.join(test_file_list_path, audio), sr=16000)
#
#     train_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
#
#     train_writer.close()
#
# # save .txt file
# print('sleep for 1 secs...')
# time.sleep(1)
# print('begin save the .txt index file for audio...')
# test_file_list = sorted(os.listdir(test_file_mix_path))
# read_test = open("test_file_list.txt", "w+")
#
# for i in test_file_list:
#     read_test.write(os.path.join(test_file_mix_path, i) + '\n')
#
# read_test.close()
# print('making data prepared!')
# test_file_mix_list_path = os.path.join(root, 'test_file_list.txt')

test_data = EvalDataset(test_file_list_path, frame_size=512, frame_shift=256)
test_loader = DataLoader(test_data,
                         batch_size=1,
                         shuffle=False,
                         num_workers=0,
                         collate_fn=EvalCollate())

weights_path = os.path.join(root, 'save_modeule/weights_50.pth')

model = Net()
model.cuda()
# model = nn.DataParallel(model, device_ids=[0, 1])
# checkpoint = Checkpoint()
# checkpoint.load(ckpt_path)
state_dict = torch.load(weights_path)
model.load_state_dict(state_dict)


print('Total Params of the model is:' + str(numParams(model)))


# avg_eval, avg_stoi, avg_pesq, avg_ssnr, avg_csig, avg_cbak, avg_covl = evaluate(model, test_loader)

avg_stoi, avg_pesq, avg_ssnr, avg_cbak, avg_csig, avg_covl = eva_noisy(test_file_list_path)

# print('Avg_loss: {:.4f}'.format(avg_eval))
print('STOI: {:.4f}'.format(avg_stoi))
print('SSNR: {:.4f}'.format(avg_ssnr))
print('PESQ: {:.4f}'.format(avg_pesq))
print('CSIG: {:.4f}'.format(avg_csig))
print('CBAK: {:.4f}'.format(avg_cbak))
print('COVL: {:.4f}'.format(avg_covl))

with open(test_results_file, "a") as f:
    # 记录验证集各指标
    val_info = f"COVL: {avg_covl:.6f}\n" \
               f"CBAK: {avg_cbak:.6f}\n" \
               f"CSIG: {avg_csig:.6f}\n" \
               f"STOI: {avg_stoi:.6f}\n" \
               f"SSNR: {avg_ssnr:.2f}\n" \
               f"PESQ: {avg_pesq:.3f}\n"
    f.write(val_info + "\n\n")


