from copy import deepcopy

import torch
import numpy as np
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from sklearn.metrics import f1_score

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class EngineForBert(Engine):

    def __init__(self, func, model, crit, optimizer, scheduler, config):

        # Ignite Engine does not have objects in below lines.
        # Thus, we assign class variables to access these object, during the procedure.
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        super().__init__(func)  # Ignite Engine only needs function to run.

        self.best_loss = np.inf
        self.best_accuracy = 0
        self.best_f1 = 0
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()  # Because we assign model as class variable, we can easily access to it.
        engine.optimizer.zero_grad()

        x, y = mini_batch['input_ids'], mini_batch['labels']
        x, y = x.to(engine.device), y.to(engine.device)

        x = x[:, :engine.config.max_length]

        # Take feed-forward
        y_hat = engine.model(x)

        loss = engine.crit(y_hat, y)

        loss.backward()

        # Calculate accuracy only if 'y' is LongTensor,
        # which means that 'y' is one-hot representation.

        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            f1 = f1_score(y_pred=torch.argmax(y_hat, dim=-1).cpu(), y_true=y.cpu(), average='macro')
        else:
            accuracy = 0
            f1 = 0
        engine.accuracy = accuracy

        # Take a step of gradient descent.
        engine.optimizer.step()
        engine.scheduler.step()
        lossn = float(loss)
        return {
            'loss': (lossn),
            'accuracy': float(accuracy),
            'f1': f1,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch['input_ids'], mini_batch['labels']
            x, y = x.to(engine.device), y.to(engine.device)

            x = x[:, :engine.config.max_length]

            # Take feed-forward
            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
                f1 = f1_score(y_pred=torch.argmax(y_hat, dim=-1).cpu(), y_true=y.cpu(), average='macro')
            else:
                accuracy = 0
                f1 = 0
            engine.accuracy = accuracy

            lossn = float(loss)


        return {
            'loss': lossn,
            'accuracy': float(accuracy),
            'f1': float(f1),
        }

    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.

        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        training_metric_names = ['loss', 'accuracy', 'f1']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |f1_score|={:.2e} loss={:.4e} accuracy={:.4f} '.format(
                    engine.state.epoch,
                    engine.state.metrics['f1'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metric_names = ['loss', 'accuracy', 'f1']

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print(
                    'Validation - loss={:.4e} accuracy={:.4f} f1-score={:.4f} F1_best={:.4f} best_accuracy={:.4f} '
                    'best_loss={:.4f}'.format(
                        engine.state.metrics['loss'],
                        engine.state.metrics['accuracy'],
                        engine.state.metrics['f1'],
                        engine.best_f1,
                        engine.best_accuracy,
                        engine.best_loss,
                    ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        accuracy = float(engine.state.metrics['accuracy'])
        f1 = float(engine.state.metrics['f1'])
        if accuracy >= engine.best_accuracy:  # If current epoch returns lower validation loss,
            engine.best_loss = loss  # Update lowest validation loss.
            engine.best_accuracy = accuracy
            engine.best_f1 = f1  # Update lowest validation loss.
            engine.best_model = deepcopy(engine.model.state_dict())  # Update best model weights.

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )


class BertTrainer():

    def __init__(self, config):
        self.config = config

    def train(
            self,
            model, crit, optimizer, scheduler,
            train_loader, valid_loader,
    ):
        train_engine = EngineForBert(
            EngineForBert.train,
            model, crit, optimizer, scheduler, self.config
        )
        validation_engine = EngineForBert(
            EngineForBert.validate,
            model, crit, optimizer, scheduler, self.config
        )

        EngineForBert.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            run_validation,  # function
            validation_engine, valid_loader,  # arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            EngineForBert.check_best,  # function
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model
