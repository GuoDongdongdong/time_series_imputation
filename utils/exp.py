import os

import pandas as pd
import torch
import numpy as np
import torch.nn as nn

from utils.dataset import data_provider
from utils.tools import logger, EarlyStopping, DISCRIMINATIVE_MODEL_LIST, GENERATIVE_MODEL_LIST, STATISTICAL_MODEL_LIST
from models import Mean, Median, Interpolate
from models import SAITS, ImputeFormer
from models import BRITS, GRUD, MRNN
from models import TimesNet
from models import CSDI, GPVAE
from models import USGAN

class Experiment:
    def __init__(self, args):
        self.model_dict = {
            'Interpolate'       : Interpolate,
            'Mean'              : Mean,
            'Median'            : Median,
            'SAITS'             : SAITS,
            'ImputeFormer'      : ImputeFormer,
            'BRITS'             : BRITS,
            'GRUD'              : GRUD,
            'MRNN'              : MRNN,
            'TimesNet'          : TimesNet,
            'USGAN'             : USGAN,
            'CSDI'              : CSDI,
            'GPVAE'             : GPVAE
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

    def _get_GAN_optimizer(self):
        G_optimizer = torch.optim.Adam(self.model.get_generator().parameters())
        D_optimizer = torch.optim.Adam(self.model.get_discriminator().parameters())
        return G_optimizer, D_optimizer

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _move_cuda(self, obj : dict | torch.Tensor):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.args.device, dtype=torch.float32)
        elif isinstance(obj, dict):
            return {key : self._move_cuda(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return list(self._move_cuda(item) for item in obj)
        elif isinstance(obj, tuple):
            return tuple(self._move_cuda(item) for item in obj)
        else:
            return obj

    def _impute_generative_model(self):
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
                batch = self._move_cuda(batch)
                output = self.model.impute(batch, self.args.n_samples)
                B, n_samples, D, L = output.shape
                # [n_samples, B * L, D]
                output = output.reshape(n_samples, B * L, D)
                # denormlization
                for i in range(n_samples):
                    output[i, :, :] = dataset.inverse(output[i, :, :])
                # [B * L, n_samples, D]
                all_generated_samples.append(output.permute(1, 0, 2))

                # [B * L, D]
                samples_median = output.median(dim=0).values
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

            logger.info(f"MAE: {mae_total / target_mask_sum}")
            logger.info(f"RMSE: {np.sqrt(mse_total / target_mask_sum)}")

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

    def _impute_discriminative_model(self):
        dataset, dataloader = self._get_data('test')
        with torch.no_grad():
            self.model.eval()
            mse_total         = 0
            mae_total         = 0
            target_mask_sum   = 0
            all_gt_mask       = []
            all_observed_data = []
            all_observed_mask = []
            all_generate_data = []
            for batch in dataloader:
                batch = self._move_cuda(batch)
                # [B, L, D]
                output = self.model.impute(batch)
                B, L, D = output.shape
                # denormlization
                output = dataset.inverse(output.reshape(B * L, D))
                all_generate_data.append(output)
                # [B, L, D]
                observed_data = batch['observed_data']
                observed_data = dataset.inverse(observed_data.reshape(B * L, D))
                all_observed_data.append(observed_data)
                
                gt_mask = batch['gt_mask']
                all_gt_mask.append(gt_mask)
                
                observed_mask = batch['observed_mask']
                all_observed_mask.append(observed_mask)
                
                target_mask = observed_mask - gt_mask
                target_mask = target_mask.reshape(B * L, D)
                mse_current = ((output - observed_data) * target_mask) ** 2
                mae_current = torch.abs((output - observed_data) * target_mask)

                mse_total       += mse_current.sum().item()
                mae_total       += mae_current.sum().item()
                target_mask_sum += target_mask.sum().item()

            logger.info(f"MAE: {mae_total / target_mask_sum}")
            logger.info(f"RMSE: {np.sqrt(mse_total / target_mask_sum)}")

            # [B * L, D]
            all_gt_mask       = torch.cat(all_gt_mask, dim=0).cpu().reshape(-1, D)
            all_observed_data = torch.cat(all_observed_data, dim=0).cpu().reshape(-1, D)
            all_generate_data = torch.cat(all_generate_data, dim=0).cpu().reshape(-1, D)
            
            df = pd.DataFrame()
            BL, D = all_observed_data.shape
            df['date'] = dataset.test_date[0 : BL]
            all_observed_data[all_gt_mask == 0] = np.nan
            df[self.args.target] = all_observed_data
            df[[target + '_imputation' for target in self.args.target]] = all_generate_data
            df.to_csv(os.path.join(self.args.checkpoints_path, 'result.csv'), index=False, float_format='%.2f', na_rep='NaN')
    
    def _impute_statistical_model(self):
        dataset, dataloader = self._get_data('test')
        self.model.impute(dataset)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

    def params(self):
        if isinstance(self.model, nn.Module):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return 0

    def train_GAN(self) :
        dataset, dataloader = self._get_data('train')
        early_stop = EarlyStopping(patience=self.args.patience)
        G_optimizer, D_optimizer = self._get_GAN_optimizer()
        if self.args.use_amp :
            scaler = torch.cuda.amp.grad_scaler.GradScaler()

        for epoch in range(self.args.epochs):
            self.model.train()
            G_train_loss = []
            D_train_loss = []
            for idx, batch in enumerate(dataloader):
                batch = self._move_cuda(batch)
                if idx % self.args.G_steps == 0:
                    G_optimizer.zero_grad()
                    loss = self.model.evaluate(batch, True, "generator")
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(G_optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        G_optimizer.step()
                    G_train_loss.append(loss.item())
                if idx % self.args.D_steps == 0:
                    D_optimizer.zero_grad()
                    loss = self.model.evaluate(batch, True, "discriminator")
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(D_optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        D_optimizer.step()
                    D_train_loss.append(loss.item())
                    
            G_train_loss = np.average(G_train_loss)
            D_train_loss = np.average(D_train_loss)
            validation_loss = self.validate()
            logger.info("Epoch: {0} G_Train Loss: {1:.7f} D_Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, G_train_loss, D_train_loss, validation_loss))

            early_stop(validation_loss, self.model, self.args.checkpoints_path)
            if early_stop.stop:
                logger.info("Early stopping")
                break

    def train(self):
        if self.args.model == 'USGAN':
            self.train_GAN()
            return 
        
        dataset, dataloader = self._get_data('train')
        early_stop = EarlyStopping(patience=self.args.patience)
        optimizer = self._get_optimizer()
        if self.args.use_amp :
            scaler = torch.cuda.amp.grad_scaler.GradScaler()

        for epoch in range(self.args.epochs):
            train_loss = []
            self.model.train()
            for batch in dataloader:
                batch = self._move_cuda(batch)
                optimizer.zero_grad()
                loss = self.model.evaluate(batch, True)

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
                batch = self._move_cuda(batch)
                loss = self.model.evaluate(batch, False)
                validation_loss.append(loss.item())

        validation_loss = np.average(validation_loss)
        return validation_loss

    def impute(self):
        if self.args.model in STATISTICAL_MODEL_LIST:
            self._impute_statistical_model()
        elif self.args.model in GENERATIVE_MODEL_LIST:
            self._impute_generative_model()
        elif self.args.model in DISCRIMINATIVE_MODEL_LIST:
            self._impute_discriminative_model()
        else:
            raise KeyError(f"model should in {STATISTICAL_MODEL_LIST + GENERATIVE_MODEL_LIST + DISCRIMINATIVE_MODEL_LIST}")