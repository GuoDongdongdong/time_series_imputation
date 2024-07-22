import os

import torch
import numpy as np

from utils.dataset import data_provider
from utils.tools import logger, EarlyStopping
# from models import SAITS, BRITS, Transformer, USGAN, LOCF, CSDI
from models import CSDI

class Experiment:
    def __init__(self, args):
        self.model_dict = {
            # 'SAITS'       : SAITS,
            # 'BRITS'       : BRITS,
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
            for i, x in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.model(x)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    logger.info("iters: {0}, epoch: {1} loss: {2:.7f}"
                                 .format(i + 1, epoch + 1, loss.item()))

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
            for x in dataloader:
                loss = self.model(batch=x, is_train=0)
                validation_loss.append(loss.item())

        validation_loss = np.average(validation_loss)
        return validation_loss

    def impute(self):
        dataset, dataloader = self._get_data('test')
        with torch.no_grad():
            self.model.eval()
            mse_total = 0
            mae_total = 0
            evalpoints_total = 0

            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            all_generated_samples_median = []
            for batch in dataloader:
                output = self.model.evaluate(batch, self.args.n_samples)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1).values
                B, L, D = samples_median.shape
                samples_median = dataset.inverse(samples_median.reshape(-1, D))
                samples_median = samples_median.reshape(B, L, D)
                samples_median = torch.from_numpy(samples_median).to(self.args.device)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                all_generated_samples_median.append(samples_median)

                mse_current = ((samples_median - c_target) * eval_points) ** 2
                mae_current = torch.abs((samples_median - c_target) * eval_points) 

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

            logger.info(f"RMSE: {np.sqrt(mse_total / evalpoints_total)}")
            logger.info(f"MAE: {mae_total / evalpoints_total}")
            all_generated_samples_median = torch.cat(all_generated_samples_median, dim=0).cpu()
            all_generated_samples_median = all_generated_samples_median.view(-1)
            dataset.result_to_csv(all_generated_samples_median)