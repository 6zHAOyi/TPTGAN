import torch
from AudioData import TrainingDataset, EvalDataset
from torch.utils.data import DataLoader
from improved_model2 import Net
# from STOI import stoi
# from PESQ import get_pesq
# from metric import get_pesq, get_stoi
from helper_funcs import numParamsAll
from criteria import mse_loss, stftm_loss
from checkpoints import Checkpoint
import os
import datetime
from eval_composite import eval_composite
from utils import power_compress, power_uncompress
import torch.nn.functional as F
from discriminator.Discriminator_ln import Discriminator, batch_pesq

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root = os.getcwd()
# hyperparameter
frame_size = 400
overlap = 0.75
frame_shift = int(frame_size * (1 - overlap))
max_epochs = 100
batch_size = 4
lr_init = 64 ** (-0.5)
lr_init_disc = 2 * lr_init
eval_steps = 500
weight_delay = 1e-7
weight = [0.1, 0.9, 0.2, 0.05]
# lr scheduling
step_num = 0

warm_ups = 0

sr = 16000
# resume_model = os.path.join(root, 'save_modeule_4trans+complex_magnitude/latest.model-68.model')  # 不是None的话 就是 相应的存model的路径
resume_model = None
resume_disc = None

model_save_path = os.path.join(root, 'save_modeule_gan')
disc_save_path = os.path.join(root, 'save_modeule_disc')
audio_file_save = os.path.join(root, 'save_audio')

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

early_stop = False
val_results_file = "val_results_4trans_gan{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
train_results_file1 = "train_iter__total_new{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
train_results_file2 = "train_step__total_new{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

train_file_list_path = os.path.join(root, 'data_bank_voice/train_file_list_bank_linux.txt')
validation_file_list_path = os.path.join(root, 'data_bank_voice/test_file_list_bank_linux.txt')
# train_file_list_path = os.path.join(root, 'train_file_list_tiaoshi.txt')
# validation_file_list_path = os.path.join(root, 'val_file_list_tiaoshi.txt')

# data and data_loader
train_data = TrainingDataset(train_file_list_path, frame_size=frame_size, frame_shift=frame_shift)
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)

validation_data = TrainingDataset(validation_file_list_path, frame_size=frame_size, frame_shift=frame_shift)
validation_loader = DataLoader(validation_data,
                               batch_size=4,
                               shuffle=False,
                               num_workers=0)

# define model
model = Net()
discriminator = Discriminator(ndf=16)
# model = torch.nn.DataParallel(model)
model = model.cuda()
discriminator = discriminator.cuda()
# print('\n')
# print('Number of learnable parameters: %d' % numParams(model))
print('\n')
print('Number of all parameters: %d' % numParamsAll(model))

optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_delay)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=lr_init_disc, weight_decay=weight_delay)
# lr_list = [0.0002] * 3 + [0.0001] * 6 + [0.00005] * 3 + [0.00001] * 3

time_loss = mse_loss()
freq_loss = stftm_loss()


# def validate(net, eval_loader, test_metric=False):
#     net.eval()
#     if test_metric:
#         print('********Starting metrics evaluation on val dataset**********')
#         total_stoi = 0.0
#         total_snr = 0.0
#         total_pesq = 0.0
#
#     with torch.no_grad():
#         count, total_eval_loss = 0, 0.0
#         for k, (features, labels) in enumerate(eval_loader):
#             features = features.cuda()  # [1, 1, num_frames, frame_size]
#             labels = labels.cuda()  # [signal_len, ]
#
#             output = net(features)  # [1, 1, sig_len_recover] for ola [1, sig_len_recover] for ISTFT
#             output = output.squeeze()  # [sig_len_recover,]
#
#             # shape = labels.shape[-1]
#             output = output[:labels.shape[-1]]  # keep length same (output label)
#
#             eval_loss = torch.mean((output - labels) ** 2)
#             # print(str(k))
#             total_eval_loss += eval_loss.data.item()
#
#             est_sp = output.cpu().numpy()
#             cln_raw = labels.cpu().numpy()
#             # wavfile.write(os.path.join(audio_file_save, str(k)), sr, est_sp.astype(np.float32))
#             if test_metric:
#                 st = get_stoi(cln_raw, est_sp, sr)
#                 pe = get_pesq(cln_raw, est_sp, sr)
#                 sn = snr(cln_raw, est_sp)
#                 total_pesq += pe
#                 total_snr += sn
#                 total_stoi += st
#
#             count += 1
#         avg_eval_loss = total_eval_loss / count
#     net.train()
#     if test_metric:
#         return avg_eval_loss, total_stoi / count, total_pesq / count, total_snr / count
#     else:
#         return avg_eval_loss


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
    nums = len(eval_loader)

    with torch.no_grad():
        count, total_eval_loss = 0, 0.0
        for k, (noisy, clean) in enumerate(eval_loader):
            # print('\rprocess = {}|{}'.format(k + 1, nums), end="")
            print('\r', "{}/{}".format(k + 1, nums), end='', flush=True)
            # noisy[1, sig_len] clean[1, sig_len]
            # features = features.unsqueeze(0).cuda()  # [1, frame_size, num_frames]-->[1, 1, frame_size, num_frames]
            # labels = labels.cuda()  # [signal_len, ]
            #
            # output = net(features)  # [1, sig_len]
            # del features
            # torch.cuda.empty_cache()
            # output = output.squeeze()  # [sig_len_recover, ]
            #
            # # keep length same (output label)
            # output = output[:labels.shape[-1]]
            #
            # eval_loss = torch.mean((output - labels) ** 2)
            # total_eval_loss += eval_loss.data.item()
            #
            # est_sp = output.cpu().numpy()
            # cln_raw = labels.cpu().numpy()
            # del output, labels
            # torch.cuda.empty_cache()
            # noisy -- [batch_size, sig_len]
            # clean -- [batch_size,sig_len]
            clean = clean.cuda()  # [batch, cut_len]
            noisy = noisy.cuda()  # [b, cut_len]
            # Normalization
            c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
            noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)  # [cut_len, batch]
            noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)  # [batch, cut_len]
            # sig_Len = [int(clean.shape[-1])]*batch_size

            noisy_spec = torch.stft(noisy, frame_size, frame_shift, window=torch.hamming_window(frame_size).cuda(),
                                    onesided=True)  # [b, f_size, num_f, 2]
            clean_spec = torch.stft(clean, frame_size, frame_shift, window=torch.hamming_window(frame_size).cuda(),
                                    onesided=True)  # [b, f_size, num_f, 2]
            del noisy, c
            torch.cuda.empty_cache()

            noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)  # [b, 2, num, f_size]
            clean_spec = power_compress(clean_spec)
            clean_real = clean_spec[:, 0, :, :].unsqueeze(1)  # [b, 1, f_size, num]
            clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)  # [b, 1, f_size, num]
            est_real, est_imag = net(noisy_spec)
            est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
            est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)
            # LOSS
            loss_mag = F.mse_loss(est_mag, clean_mag)
            loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_audio = torch.istft(est_spec_uncompress, frame_size, frame_shift,
                                    window=torch.hamming_window(frame_size).cuda(),
                                    onesided=True)  # recoveraudio[batch_size, sig_len]

            time_loss = torch.mean(torch.abs(est_audio - clean))
            loss = weight[0] * loss_ri + weight[1] * loss_mag + weight[2] * time_loss
            del noisy_spec, clean_spec, clean_real, clean_imag, est_real, est_imag, est_spec_uncompress, est_mag, clean_mag, time_loss, loss_ri, loss_mag
            torch.cuda.empty_cache()
            total_eval_loss += loss.data.item()

            del loss
            torch.cuda.empty_cache()
            est_audio = est_audio.squeeze(0).cpu().numpy()
            clean = clean.squeeze(0).cpu().numpy()

            eval_metric = eval_composite(clean, est_audio, sr)
            del est_audio, clean

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
    net.train()
    return avg_eval_loss, total_stoi / count, total_pesq / count, total_ssnr / count, total_csig / count, total_cbak / count, total_covl / count


# train model
if resume_model:
    print('Resume model from "%s"' % resume_model)
    checkpoint = Checkpoint()
    checkpoint.load(resume_model)

    start_epoch = checkpoint.start_epoch + 1
    best_val_loss = checkpoint.best_val_loss
    prev_val_loss = checkpoint.prev_val_loss
    num_no_improv = checkpoint.num_no_improv
    half_lr = checkpoint.half_lr
    model.load_state_dict(checkpoint.state_dict)
    optimizer.load_state_dict(checkpoint.optimizer)

    print('Resume discriminator from "%s"' % resume_disc)
    checkpoint.load(resume_disc)
    discriminator.load_state_dict(checkpoint.state_dict)
    optimizer_disc.load_state_dict(checkpoint.optimizer)

else:
    print('Training from scratch.')
    start_epoch = 0
    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    num_no_improv = 0
    half_lr = False

for epoch in range(start_epoch, max_epochs):
    model.train()
    # total_train_loss, ave_train_loss = 0.0, 0.0

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr_list[epoch]

    # if half_lr:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr'] / 2
    #         print('Learning rate adjusted to  %5f' % (param_group['lr']))
    #     half_lr = False

    # avg_eval, avg_stoi, avg_pesq, avg_ssnr, avg_csig, avg_cbak, avg_covl, avg_batch_pesq = evaluate(model, validation_loader)
    for index, (noisy, clean) in enumerate(train_loader):
        torch.cuda.empty_cache()
        one_labels = torch.ones(batch_size).cuda()
        step_num += 1
        if step_num <= warm_ups:
            lr = 0.2 * lr_init * min(step_num ** (-0.5),
                                     step_num * (warm_ups ** (-1.5)))
            lr_disc = 0.2 * lr_init_disc * min(step_num ** (-0.5),
                                               step_num * (warm_ups ** (-1.5)))
        else:
            lr = 0.0004 * (0.98 ** ((epoch - 1) // 2))
            lr_disc = 0.0004 * (0.98 ** ((epoch - 1) // 2))

        for param_group, param_group_disc in zip(optimizer.param_groups, optimizer_disc.param_groups):
            param_group['lr'] = lr
            param_group_disc['lr'] = lr_disc
            print('generator Learning rate adjusted to  %6f' % (param_group['lr']))
            print('discriminator Learning rate adjusted to  %6f' % (param_group_disc['lr']))

        # noisy -- [batch_size, sig_len]
        # clean -- [batch_size,sig_len]
        clean = clean.cuda()  # [batch, cut_len]
        noisy = noisy.cuda()  # [b, cut_len]
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)  # [cut_len, batch]
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)  # [batch, cut_len]
        # sig_Len = [int(clean.shape[-1])]*batch_size

        noisy_spec = torch.stft(noisy, frame_size, frame_shift, window=torch.hamming_window(frame_size).cuda(),
                                onesided=True)  # [b, f_size, num_f, 2]
        clean_spec = torch.stft(clean, frame_size, frame_shift, window=torch.hamming_window(frame_size).cuda(),
                                onesided=True)  # [b, f_size, num_f, 2]
        del c
        torch.cuda.empty_cache()

        noisy_spec = power_compress(noisy_spec)
        clean_spec = power_compress(clean_spec)
        noisy_real = noisy_spec[:, 0, :, :].unsqueeze(1)
        noisy_imag = noisy_spec[:, 1, :, :].unsqueeze(1)
        noisy_spec = noisy_spec.permute(0, 1, 3, 2)  # [b, 2, num, f_size]
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)  # [b, 1, f_size, num]
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)  # [b, 1, f_size, num]

        optimizer.zero_grad()

        est_real, est_imag = model(noisy_spec)  # [batch_size, 1, num, f_size]
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)  # [batch_size, 1, f_size, num]
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)  # [batch_size, f_size, num, 2]
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)
        noisy_mag = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2)
        
        est_audio = torch.istft(est_spec_uncompress, frame_size, frame_shift,
                                window=torch.hamming_window(frame_size).cuda(), onesided=True)  # [batch_size, sig_len]
        # LOSS
        predict_fake_metric = discriminator(clean_mag, est_mag)
        gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
        time_loss = torch.mean(torch.abs(est_audio - clean))
        loss = weight[0] * loss_ri + weight[1] * loss_mag + weight[2] * time_loss + weight[3] * gen_loss_GAN
        del noisy_spec, clean_spec, clean_real, clean_imag, est_real, est_imag, est_spec_uncompress, time_loss, loss_ri, loss_mag, gen_loss_GAN, noisy_imag, noisy_real
        torch.cuda.empty_cache()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)

        optimizer.step()

        train_loss = loss.data.item()
        # total_train_loss += float(train_loss)

        del loss
        torch.cuda.empty_cache()

        length = est_audio.size(-1)
        noisy_audio_list = list(noisy.cpu().numpy()[:, :length])
        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        pesq_score_noisy = batch_pesq(clean_audio_list, noisy_audio_list)

        if pesq_score is not None:
            optimizer_disc.zero_grad()
            predict_enhance_metric = discriminator(clean_mag, est_mag.detach())  # D(G(x),y)
            predict_max_metric = discriminator(clean_mag, clean_mag)  # D(y,y)
            predict_noisy_metric = discriminator(noisy_mag, clean_mag)  # D(x,y)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score) + \
                                  F.mse_loss((predict_noisy_metric.flatten()), pesq_score_noisy)
            discrim_loss_metric.backward()
            optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        disc_loss = discrim_loss_metric.item()

        del discrim_loss_metric, clean, est_audio, est_audio_list, clean_audio_list, clean_mag, est_mag

        print(
            'iter = {}/{}, epoch = {}/{}, train_loss = {:.5f}, disc_loss = {:.5f}'.format(index + 1, len(train_loader),
                                                                                          epoch + 1, max_epochs,
                                                                                          train_loss, disc_loss))

        # with open(train_results_file1, "a") as f:
        #     train_info1 = f"[epoch: {epoch + 1}]\n" \
        #                   f"[iter: {index + 1}]\n" \
        #                   f"train_loss: {train_loss:.5f}\n"
        #     f.write(train_info1 + "\n\n")

        # if (index + 1) % eval_steps == 0:
        #     ave_train_loss = total_train_loss / count
        #
        #     # validation
        #     avg_eval_loss = validate(model, validation_loader)
        #     model.train()
        #
        #     print('Epoch [%d/%d], Iter [%d/%d],  ( TrainLoss: %.4f | EvalLoss: %.4f )' % (
        #     epoch + 1, max_epochs, index + 1, len(train_loader), ave_train_loss, avg_eval_loss))
        #     with open(train_results_file2, "a") as f:
        #         train_info1 = f"[epoch: {epoch + 1}]\n" \
        #                       f"[iter: {index + 1}]\n" \
        #                       f"avg_eval_loss: {avg_eval_loss:.4f}]\n" \
        #                       f"avg_train_loss: {train_loss:.4f}\n"
        #         f.write(train_info1 + "\n\n")
        #
        #     count = 0
        #     total_train_loss = 0.0

        # if (index + 1) % len(train_loader) == 0:
        #     break

    # validate metric
    # avg_eval, avg_stoi, avg_pesq, avg_snr = validate(model, validation_loader, test_metric=True)
    avg_eval, avg_stoi, avg_pesq, avg_ssnr, avg_csig, avg_cbak, avg_covl = evaluate(model, validation_loader)
    model.train()
    print('#' * 50)
    print('')
    print('After {} epoch the performance on validation score is a s follows:'.format(epoch + 1))
    print('')
    print('Avg_loss: {:.4f}'.format(avg_eval))
    print('STOI: {:.4f}'.format(avg_stoi))
    print('SSNR: {:.4f}'.format(avg_ssnr))
    print('PESQ: {:.4f}'.format(avg_pesq))
    print('CSIG: {:.4f}'.format(avg_csig))
    print('CBAK: {:.4f}'.format(avg_cbak))
    print('COVL: {:.4f}'.format(avg_covl))
    with open(val_results_file, "a") as f:
        # 记录每个epoch对应的train_loss、lr以及验证集各指标
        val_info = f"[epoch: {epoch}]\n" \
                   f"train_loss: {avg_eval:.6f}\n" \
                   f"lr: {lr:.6f}\n" \
                   f"STOI: {avg_stoi:.4f}\n" \
                   f"COVL: {avg_covl:.4f}\n" \
                   f"CBAK: {avg_cbak:.4f}\n" \
                   f"CSIG: {avg_csig:.4f}\n" \
                   f"SSNR: {avg_ssnr:.4f}\n" \
                   f"PESQ: {avg_pesq:.4f}\n"
        f.write(val_info + "\n\n")

    # adjust learning rate and early stop
    if avg_eval >= prev_val_loss:
        num_no_improv += 1
        # if num_no_improv == 2:
        # half_lr = True
        if num_no_improv >= 10 and early_stop is True:
            print("No improvement and apply early stop")
            break
    else:
        num_no_improv = 0

    prev_val_loss = avg_eval

    if avg_eval < best_val_loss:
        best_val_loss = avg_eval
        is_best_model = True
    else:
        is_best_model = False

    # save model
    latest_model = 'latest.model'
    best_model = 'best.model'
    # torch.save(model.state_dict(), model_state_dict)

    checkpoint = Checkpoint(start_epoch=epoch,
                            best_val_loss=best_val_loss,
                            prev_val_loss=prev_val_loss,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            num_no_improv=num_no_improv,
                            half_lr=half_lr)
    checkpoint_disc = Checkpoint(start_epoch=epoch,
                                 state_dict=discriminator.state_dict(),
                                 optimizer=optimizer_disc.state_dict()
                                 )
    checkpoint.save(is_best=is_best_model,
                    filename=os.path.join(model_save_path, latest_model + '-{}.model'.format(epoch + 1)),
                    best_model=os.path.join(model_save_path, best_model))
    checkpoint_disc.save(is_best=False,
                         filename=os.path.join(disc_save_path, latest_model + '-{}.model'.format(epoch + 1)),
                         best_model=os.path.join(disc_save_path, best_model))
