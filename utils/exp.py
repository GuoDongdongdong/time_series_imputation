import os

import torch
import numpy as np

from utils.dataset import data_provider
from utils.tools import logger, EarlyStopping
# from models import SAITS, BRITS, Transformer, USGAN, LOCF, CSDI
from models import CSDI, BRITS

class Experiment:
    def __init__(self, args):
        self.model_dict = {
            # 'SAITS'       : SAITS,
            'BRITS'       : BRITS,
            # 'Transformer' : Transformer,
            # 'USGAN'       : USGAN,
            # 'LOCF'        : LOCF,
            'CSDI'        : CSDI
        }
        self.args = args
        self.args.device = self._get_device()
        self.model = self._build_model()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.multi_gpu:
            device_ids = list(self.args.gpu_id.split(','))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        if isinstance(model, torch.nn.Module) :
            model = model.float().to(self.args.device)

        return model

    def _get_device(self):
        if self.args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_id
            device = torch.device(f"cuda:{self.args.gpu_id}")
            logger.debug(f"use gpu : {device}")
        else:
            device = torch.device('cpu')
            logger.debug(f"use cpu")

        return device

    def _get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self) :
        dataset, dataloader = self._get_data('train')
        early_stop = EarlyStopping(patience=self.args.patience)
        optimizer = self._get_optimizer()
        if self.args.use_amp :
            scaler = torch.cuda.amp.grad_scaler.GradScaler()

        for epoch in range(self.args.epochs):
            train_loss = []
            self.model.train()
            for batch in dataloader:
                optimizer.zero_grad()
                loss = self.model(batch=batch, is_train=True)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            validation_loss = self.validate()
            logger.info("Epoch: {0} Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(epoch + 1, train_loss, validation_loss))

            early_stop(validation_loss, self.model, self.args.checkpoints_path)
            if early_stop.stop:
                logger.info("Early stopping")
                break

    def validate(self):
        dataset, dataloader = self._get_data('validation')
        validation_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                loss = self.model(batch=batch, is_train=False)
                validation_loss.append(loss.item())

        validation_loss = np.average(validation_loss)
        return validation_loss

    def impute(self):
        dataset, dataloader = self._get_data('test')
        with torch.no_grad():
            self.model.eval()
            mse_total = 0
            mae_total = 0
            target_mask_sum = 0

            all_gt_mask = []
            all_observed_data = []
            all_observed_mask = []
            all_generated_samples = []
            all_generated_samples_median = []
            for batch in dataloader:
                # [B, n_samples, D, L]
                output = self.model.impute(batch=batch, n_samples=self.args.n_samples)
                B, n_samples, D, L = output.shape
                # [n_samples, B * L, D]
                output = output.reshape(n_samples, B * L, D)
                # denormlization
                for i in range(n_samples):
                    output[i, :, :] = dataset.inverse(output[i, :, :])
                # [B * L, n_samples, D]
                all_generated_samples.append(output.permute(1, 0, 2))

                # [B * L, D]
                samples_median = output.median(dim=0).values.cpu()
                all_generated_samples_median.append(samples_median)

                # [B * L, D]
                observed_data = batch['observed_data']
                observed_mask = batch['observed_mask']
                gt_mask       = batch['gt_mask']
                observed_data = dataset.inverse(observed_data.reshape(B * L, D))
                all_observed_mask.append(observed_mask)
                all_gt_mask.append(gt_mask)
                all_observed_data.append(observed_data)
                target_mask = observed_mask - gt_mask
                target_mask = target_mask.reshape(B * L, D)
                mse_current = ((samples_median - observed_data) * target_mask) ** 2
                mae_current = torch.abs((samples_median - observed_data) * target_mask)

                mse_total       += mse_current.sum().item()
                mae_total       += mae_current.sum().item()
                target_mask_sum += target_mask.sum().item()

            logger.info(f"RMSE: {np.sqrt(mse_total / target_mask_sum)}")
            logger.info(f"MAE: {mae_total / target_mask_sum}")

            # [B * L, D]
            all_gt_mask = torch.cat(all_gt_mask, dim=0).cpu().reshape(-1, D)
            # [B * L, D]
            all_observed_data = torch.cat(all_observed_data, dim=0).cpu().reshape(-1, D)
            # [B * L, n_samples, D]
            all_generated_samples = torch.cat(all_generated_samples, dim=0).cpu()
            # [B * L, D]
            all_generated_samples_median = torch.cat(all_generated_samples_median, dim=0).cpu()
            dataset.save_result(observed_data=all_observed_data,
                                observed_mask=all_observed_mask,
                                gt_mask=all_gt_mask,
                                samples_data=all_generated_samples,
                                impute_data=all_generated_samples_median)