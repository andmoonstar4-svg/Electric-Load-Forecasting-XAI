from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
def EMEX (pred,true,X=4):
    return torch.mean(2**(X*abs(pred - true))-1)
def MSEX(pred,true,X=3):
    return torch.mean(X*abs(pred - true))
def MAE(pred, true):
    return torch.mean(abs(true - pred))
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = torch.MSELoss()
        elif self.args.loss == 'MAE':
            criterion = MAE
        elif self.args.loss == 'EMEX':
            criterion = EMEX
        elif self.args.loss == 'MESX':
            criterion = MSEX
        return criterion


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe  = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def shap_analysis(self, setting, n_background=100):
        """
        专为时序模型打造的黑盒采样 SHAP 可解释性分析函数 (完全绕过 Autograd 梯度与 In-place 报错)
        """
        print(f'⚡ [SHAP] 开始为 {self.args.model} 构建可解释性分析矩阵 (KernelExplainer 模式)...')
        import os
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import shap

        # ==========================================
        # 核心修改点 1：强制在文件夹前缀加入真实运行的模型名称
        # 防止 run.py 中 --model_id 忘记改导致不同模型的 SHAP 文件夹覆盖
        # ==========================================
        folder_path = './shap_results/' + self.args.model + '_' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 1. 加载测试集 DataLoader
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        # 2. 从 DataLoader 中提取数据
        data_iter = iter(test_loader)
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(data_iter)

        batch_x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # 构建 Decoder 输入
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # 采样 5 个样本作为背景，3 个样本作为测试预测，保证速度
        background_x_np = batch_x[:5].cpu().numpy()
        test_x_np = batch_x[5:8].cpu().numpy()

        # 3. 构造纯 NumPy 输入/输出的预测函数
        def model_predict(x_enc_numpy):
            with torch.no_grad():  # ⚠️ 关键：完全关闭 PyTorch 梯度计算，杜绝任何 In-place 报错！
                x_enc_tensor = torch.tensor(x_enc_numpy, dtype=torch.float32).to(self.device)
                current_bs = x_enc_tensor.shape[0]

                mark_enc = batch_x_mark[:1].repeat(current_bs, 1, 1)
                dec_in = dec_inp[:1].repeat(current_bs, 1, 1)
                mark_dec = batch_y_mark[:1].repeat(current_bs, 1, 1)

                outputs = self.model(x_enc_tensor, mark_enc, dec_in, mark_dec)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # 返回未来第 1 个小时的目标变量预测值 (Batch, 1)
                return outputs[:, 0, 0].cpu().numpy()

        # 4. 使用 KernelExplainer 进行黑盒 SHAP 计算
        print('⏳ [SHAP] 正在进行黑盒采样计算 (绕过 PyTorch 反向图，预计耗时 30~60 秒)...')
        try:
            # 将 3D 张量展平为 2D 喂给 KernelExplainer，计算完再还原
            orig_shape = test_x_np.shape  # (3, seq_len, features)
            bg_flat = background_x_np.reshape(background_x_np.shape[0], -1)
            test_flat = test_x_np.reshape(test_x_np.shape[0], -1)

            def flat_predict(x_flat):
                x_3d = x_flat.reshape(-1, orig_shape[1], orig_shape[2])
                return model_predict(x_3d)

            explainer = shap.KernelExplainer(flat_predict, bg_flat)
            shap_values_flat = explainer.shap_values(test_flat, nsamples=100)

            if isinstance(shap_values_flat, list):
                shap_values_flat = shap_values_flat[0]

            # 还原为 (Batch, seq_len, features)
            shap_values = shap_values_flat.reshape(orig_shape)

            # 5. 计算各特征的平均贡献度 (对 Batch 和时间轴求平均)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))

            # 获取特征列名
            feature_names = getattr(test_data, 'cols', [f'Feature_{i}' for i in range(self.args.enc_in)])

            # 6. 绘图与保存
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(feature_names))
            plt.barh(y_pos, mean_abs_shap, align='center', color='#2b5c8f')
            plt.yticks(y_pos, labels=feature_names)
            plt.xlabel('Mean |SHAP Value| (Impact on Model Output)')

            # ==========================================
            # 核心修改点 2：动态图表标题与输出文件名清理
            # ==========================================
            plt.title(f"SHAP Feature Importance for {self.args.model}")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            # 保存图片和数组时，文件名均带上当前模型名字，杜绝任何覆盖可能
            save_img_path = os.path.join(folder_path, f'shap_importance_{self.args.model}.png')
            plt.savefig(save_img_path, dpi=300)
            plt.close()

            save_npy_path = os.path.join(folder_path, f'shap_values_{self.args.model}.npy')
            np.save(save_npy_path, shap_values)

            print(f'🎉 [SHAP] 分析成功！结果已安全保存至专属文件夹: {folder_path}')
            print(f'📊 专属图表已生成: {save_img_path}')

        except Exception as e:
            print(f'❌ [SHAP] {self.args.model} 计算过程中发生错误: {e}')
            
        def model_predict(x_enc_numpy):
            with torch.no_grad(): # ⚠️ 关键：完全关闭 PyTorch 梯度计算，杜绝任何 In-place 报错！
                x_enc_tensor = torch.tensor(x_enc_numpy, dtype=torch.float32).to(self.device)
                current_bs = x_enc_tensor.shape[0]

                mark_enc = batch_x_mark[:1].repeat(current_bs, 1, 1)
                dec_in = dec_inp[:1].repeat(current_bs, 1, 1)
                mark_dec = batch_y_mark[:1].repeat(current_bs, 1, 1)

                outputs = self.model(x_enc_tensor, mark_enc, dec_in, mark_dec)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # 返回未来第 1 个小时的目标变量预测值 (Batch, 1)
                return outputs[:, 0, 0].cpu().numpy()

        # 4. 使用 KernelExplainer 进行黑盒 SHAP 计算
        print('⏳ [SHAP] 正在进行黑盒采样计算 (绕过 PyTorch 反向图，预计耗时 30~60 秒)...')
        try:
            # 将 3D 张量展平为 2D 喂给 KernelExplainer，计算完再还原
            orig_shape = test_x_np.shape # (3, seq_len, features)
            bg_flat = background_x_np.reshape(background_x_np.shape[0], -1)
            test_flat = test_x_np.reshape(test_x_np.shape[0], -1)

            def flat_predict(x_flat):
                x_3d = x_flat.reshape(-1, orig_shape[1], orig_shape[2])
                return model_predict(x_3d)

            explainer = shap.KernelExplainer(flat_predict, bg_flat)
            shap_values_flat = explainer.shap_values(test_flat, nsamples=100)

            if isinstance(shap_values_flat, list):
                shap_values_flat = shap_values_flat[0]

            # 还原为 (Batch, seq_len, features)
            shap_values = shap_values_flat.reshape(orig_shape)

            # 5. 计算各特征的平均贡献度 (对 Batch 和时间轴求平均)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))

            # 获取特征列名
            feature_names = getattr(test_data, 'cols', [f'Feature_{i}' for i in range(self.args.enc_in)])

            # 6. 绘图与保存
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(feature_names))
            plt.barh(y_pos, mean_abs_shap, align='center', color='#2b5c8f')
            plt.yticks(y_pos, labels=feature_names)
            plt.xlabel('Mean |SHAP Value| (Impact on Model Output)')
            # 动态获取模型名称，如果是 Autoformer 就会显示 Autoformer
            plt.title(f"SHAP Feature Importance for {self.args.model}")

            # 如果保存图片的文件名也被写死了，也可以顺便改掉：
            plt.savefig(os.path.join(folder_path, f'shap_summary_{self.args.model}.png'))
            plt.gca().invert_yaxis()
            plt.tight_layout()

            save_path = os.path.join(folder_path, 'shap_feature_importance.png')
            plt.savefig(save_path, dpi=300)
            plt.close()

            np.save(os.path.join(folder_path, 'shap_values.npy'), shap_values)

            print(f'🎉 [SHAP] 分析成功！结果已保存至文件夹: {folder_path}')
            print(f'📊 特征重要性图表已生成: {save_path}')

        except Exception as e:
            print(f'❌ [SHAP] 计算过程中发生错误: {e}')