# 计算noisy语音的原始数据
from eval_composite import eval_composite


import datetime

import h5py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sr = 16000
root = os.getcwd()
# file_name = 'psquare_17.5'
test_file_list_path = os.path.join(root, 'data_bank_voice/test_file_list_bank_linux.txt')
test_results_file = "test_results_noisy{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


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
        print('\rprocess = [{}/{}]'.format(i + 1, len(file_list)), end="")
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
    val_info = f"COVL: {avg_covl:.4f}\n" \
               f"CBAK: {avg_cbak:.4f}\n" \
               f"CSIG: {avg_csig:.4f}\n" \
               f"STOI: {avg_stoi:.4f}\n" \
               f"SSNR: {avg_ssnr:.4f}\n" \
               f"PESQ: {avg_pesq:.4f}\n"
    f.write(val_info + "\n\n")
