from os import path as osp
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import forge
from forge import flags
import forge.experiment_tools as fet

from train_tools import (
	log_tensorboard,
	parse_reports,
	parse_reports_cpu,
	print_reports,
	load_checkpoint,
	save_checkpoint,
	nested_to,
	param_count,
)

from copy import deepcopy
from attrdict import AttrDict
import deepdish as dd
from tqdm import tqdm

# For reproducibility while researching, but might affect speed!
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


# Job config
flags.DEFINE_string('data_config', 'data_configs/penn_dataset.py',
					'Path to a data config file.')
flags.DEFINE_string('model_config', 'model_configs/convNet.py',
					'Path to a model config file.')
flags.DEFINE_string('results_dir', 'checkpoints',
					'Top directory for all experimental results.')
flags.DEFINE_string('model_name', 'conv1d',
					'shorthand name for model.')
flags.DEFINE_string('run_name', 'demo',
					'Name of this job and name of results folder.')
flags.DEFINE_boolean('resume', False, 'Tries to resume a job if True.')

# Logging config
flags.DEFINE_integer('report_loss_every', 50,
					 'Number of iterations between reporting minibatch loss.')
flags.DEFINE_integer('train_epochs', 25, 'Maximum number of training epochs.')
flags.DEFINE_integer(
	"save_check_points",
	50,
	"frequency with which to save checkpoints, in number of epoches.",
)
flags.DEFINE_boolean("log_train_values", True, "Logs train values if True.")
flags.DEFINE_integer(
	"total_evaluations",
	100,
	"Maximum number of evaluations on test and validation data during training.",
)
flags.DEFINE_boolean(
	"save_test_predictions",
	False,
	"Makes and saves test predictions on one or more test sets (e.g. 5-step and 100-step predictions) at the end of training.",
)
flags.DEFINE_boolean(
    "log_val_test", True, "Turns off computation of validation and test errors."
)

flags.DEFINE_boolean('use_mps',False,"Use Metal Performance Shaders (M1) for training.")

# Experiment config
flags.DEFINE_integer('batch_size', 100, 'Mini-batch size.')
flags.DEFINE_float("learning_rate", 1e-3, "Adam learning rate.")
flags.DEFINE_float("beta1", 0.9, "Adam Beta 1 parameter")
flags.DEFINE_float("beta2", 0.999, "Adam Beta 2 parameter")

# Specific config options
# flags.DEFINE_integer("input_size",39,"Input shape to model")

#########################################################################################

def evaluate(model, loader, device,fold = 0,headings = [],loss_func = nn.CrossEntropyLoss()):
	# this function does not account for class weights when evaluating cross-entropy loss.
	# however, they are accounted for when evaluating training loss

	# MODIFY THIS FUNCTION TO ACCOUNT FOR COORDINATES BEING INCLUDED
	reports = None
	for data in loader:
		labels = data['label']
		inputs = nested_to(data['data'],device,torch.float32)
		
		labels_tensor = torch.zeros((len(labels), 2))
		for i, label in enumerate(labels):
			labels_tensor[i, label] = 1
		labels_tensor = labels_tensor.to(device)

		model = model.to(device)
		model_output = model(data) # evaluate model

		loss = loss_func(model_output, labels_tensor) # compute loss
		outputs = AttrDict({"loss": loss, "prediction": model_output,'data_fold':fold})
		outputs.reports = AttrDict({"cross_entropy": loss})

		if reports is None:
			reports = {k: v.detach().clone().cpu() for k, v in outputs.reports.items()}
		else:
			for k, v in outputs.reports.items():
				reports[k] += v.detach().clone().cpu()

	for k, v in reports.items():
		reports[k] = v / len(
			loader
		)  # SZ: note this can be slightly incorrect if mini-batch sizes vary (if batch_size doesn't divide train_size), but approximately correct.

	# reports['test_headings'] = np.array(headings)

	return reports

def main():

	config = forge.config()

	# Set device
	if torch.cuda.is_available():
		device = f"cuda:{config.device}"
		torch.cuda.set_device(device)
	else:
		device = "cpu" # can change this to MPS on M1 Macs

	if config.use_mps:
		device = "mps"

	# Load data
	data_name = "penn_data"
	kfold_loaders,L_inds = fet.load(config.data_config, config)

	if L_inds == None: # in this case we train on all wavelengths
		Ldata_size = 39
	else:
		Ldata_size = len(L_inds)

	config.input_size = Ldata_size

	#############################################################
	# ENVIRONMENT/DIRECTORY MANAGEMENT

	# Prepare environment
	params_in_run_name = [
		("batch_size", "bs"),
		("learning_rate", "lr"),
		("n_train", "ntrain"),
		("include_coords",'inclx'),
		("input_size",'inpsize'),
		("cross_validate",'crval'),
		("train_epochs",'n_epochs'),
		("n_repeats",'nrep')
	]

	run_name = ""  # config.run_name
	for config_param in params_in_run_name:
		attr = config_param[0]
		abbrev = config_param[1]

		if hasattr(config, attr):
			run_name += abbrev
			run_name += str(getattr(config, attr))
			run_name += "_"

	results_folder_name = osp.join(
		data_name,
		config.model_name,
		config.run_name,
		run_name,
	)

	# Prepare environment
	logdir = osp.join(config.results_dir, results_folder_name.replace(".", "_"))
	logdir, resume_checkpoint = fet.init_checkpoint(
		logdir, config.data_config, config.model_config, config.resume)

	print("Checkpoint directory:",logdir)

	# Print flags
	fet.print_flags()

	n_folds = len(kfold_loaders)
	print("Doing {}-fold cross-validation.".format(n_folds))

	#############################################################
	# ENVIRONMENT/DIRECTORY MANAGEMENT
	# (KFOLD TRAINING LOOP)
	for k,loader_k in enumerate(kfold_loaders): # loop over data splits

		print("Training fold {}...".format(k+1))

		# create folder for this data split
		fold_dir = osp.join(logdir,"data_fold{}".format(k+1))
		fold_dir_exists = osp.exists(fold_dir)
		if not fold_dir_exists:
			if config.resume:
				raise ValueError("Can't resume when the checkpoint/fold dir '{}' doesn't exist.".format(fold_dir))
			else:
				os.makedirs(fold_dir)
		checkpoint_name = osp.join(fold_dir, 'model_fold{}.ckpt'.format(k+1))

		# keep track of which data is test data for a given Kfold data split
		test_headings = loader_k['test_headings']
		headingsFile = osp.join(fold_dir,"test_headings.txt")
		with open(headingsFile,'w') as headfile:
			for heading in test_headings:
				headfile.write(heading+'\n')

		# Load model
		model,model_name = fet.load(config.model_config, config)
		model = model.to(device)

		# Print model info
		print(model)

		# Setup optimizer
		model_params = model.parameters()
		opt_learning_rate = config.learning_rate
		model_opt = torch.optim.Adam(
			model_params, lr=opt_learning_rate, betas=(config.beta1, config.beta2)
		)
		scheduler = torch.optim.lr_scheduler.StepLR(model_opt, step_size=10, gamma=0.5)
		print("Model Optimizer:")
		print(model_opt)

		# Try to restore model and optimizer from checkpoint
		if resume_checkpoint is not None:
			start_epoch = load_checkpoint(resume_checkpoint, model, model_opt)
		else:
			start_epoch = 1

		# load the training data
		train_loader = loader_k["train"]
		n_train = len(train_loader.dataset)
		train_batch_size = train_loader.batch_size

		# load testing, validation
		test_loader = loader_k["test"]
		n_test = len(test_loader.dataset)
		val_loader = loader_k["val"]
		n_val = len(val_loader.dataset)

		train_class_counts = np.bincount(train_loader.dataset.labels)
		train_total_counts = sum(train_class_counts)
		class_weight = torch.tensor(1.0 / train_class_counts) * train_total_counts / 2.0

		class_weight = nested_to(class_weight,device,torch.float32)

		loss_func = nn.CrossEntropyLoss(weight = class_weight) # define reweighted loss function

		# print(checkpoint_name)

		train_iter = (start_epoch - 1) * (
		len(train_loader.dataset) // train_batch_size
		) + 1

		print("Starting training at epoch = {}, iter = {}".format(start_epoch, train_iter))

		# Setup tensorboard writing
		summary_writer = SummaryWriter(logdir)

		train_reports = []
		report_all = {}
		report_all_val = {}

		# Saving model at epoch 0 before training
		print("saving model at epoch 0 before training ... ")
		save_checkpoint(checkpoint_name, 0, model, model_opt, loss=0.0)
		print("finished saving model at epoch 0 before training")

		num_params = param_count(model)
		print(f"Number of model parameters: {num_params}")

		# Training
		start_t = time.time()

		total_train_iters = len(train_loader) * config.train_epochs
		iters_per_eval = max(1, int(total_train_iters / config.total_evaluations))
		# print('batch',train_batch_size)
		# print('ntrain',n_train)
		# assert (
		# 	n_train % min(train_batch_size, n_train) == 0
		# ), "Batch size doesn't divide dataset size. Can be inaccurate for loss computation (see below)."

		training_failed = False
		best_val_loss_so_far = 1e7

		for epoch in tqdm(range(start_epoch, config.train_epochs + 1)):
			model.train()

			runsum = 0
			for batch_idx, data_dict in enumerate(train_loader): # Loop over batched data

				labels = data_dict['label']
				data = data_dict['data']

				# send data to device
				data = nested_to(data,device,torch.float32)
				# One-hot encode the data labels
				labels_tensor = torch.zeros((len(labels), 2))
				for i, label in enumerate(labels):
					labels_tensor[i, label] = 1

				labels_tensor = labels_tensor.to(device)

				model_output = model(data_dict) # evaluate model

				loss = loss_func(model_output, labels_tensor) # compute loss
				runsum += loss.item() # track loss
				outputs = AttrDict({"loss": loss, "prediction": model_output})
				outputs.reports = AttrDict({"cross_entropy": loss})

				if torch.isnan(outputs.loss): # check for failure in training
					if not training_failed:
						epoch_of_nan = epoch
					if (epoch > epoch_of_nan + 1) and training_failed:
						raise ValueError("Loss Nan-ed.")
					training_failed = True

				model_opt.zero_grad()
				outputs.loss.backward(retain_graph=False)

				model_opt.step() # move optimizer forward

				if config.log_train_values:
					reports = parse_reports(outputs.reports)
					if batch_idx % config.report_loss_every == 0:
						train_reports.append([train_iter,runsum/config.report_loss_every])
						runsum = 0
						log_tensorboard(summary_writer, train_iter, reports, "train/")
						print_reports(
							reports,
							start_t,
							epoch,
							batch_idx,
							len(train_loader.dataset) // train_batch_size,
							prefix="train",
						)
						log_tensorboard(
							summary_writer,
							train_iter,
							{"lr": model_opt.param_groups[0]["lr"]},
							"hyperparams/",
						)
				# Logging and evaluation
				if (
					train_iter % iters_per_eval == 0 or (train_iter == total_train_iters)
				) and config.log_val_test:  # batch_idx % config.evaluate_every == 0:
					model.eval()
					
					with torch.no_grad():
						reports = evaluate(model, test_loader, device,fold = k,headings = loader_k['test_headings'])
						# print("REPORTS",reports)
						reports = parse_reports(reports)
						reports["time"] = time.time() - start_t
						if report_all == {}:
							report_all = deepcopy(reports)

							for d in reports.keys():
								report_all[d] = [report_all[d]]
						else:
							for d in reports.keys():
								report_all[d].append(reports[d])

						log_tensorboard(summary_writer, train_iter, reports, "test/")
						print_reports(
							reports,
							start_t,
							epoch,
							batch_idx,
							len(train_loader.dataset) // config.batch_size,
							prefix="test",
						)

						# repeat for validation data
						reports = evaluate(model, val_loader, device,fold = k,headings = loader_k['test_headings'])
						reports = parse_reports(reports)
						reports["time"] = time.time() - start_t
						if report_all_val == {}:
							report_all_val = deepcopy(reports)

							for d in reports.keys():
								report_all_val[d] = [report_all_val[d]]
						else:
							for d in reports.keys():
								report_all_val[d].append(reports[d])

						log_tensorboard(summary_writer, train_iter, reports, "val/")
						print_reports(
							reports,
							start_t,
							epoch,
							batch_idx,
							len(train_loader.dataset) // config.batch_size,
							prefix="val",
						)

						if report_all_val["cross_entropy"][-1] < best_val_loss_so_far:
							save_checkpoint(
								checkpoint_name,
								f"early_stop",
								model,
								model_opt,
								loss=outputs.loss,
							)
							best_val_loss_so_far = report_all_val["cross_entropy"][-1]

					model.train()

				train_iter += 1

			scheduler.step()

			

			if epoch % config.save_check_points == 0:
				save_checkpoint(
					checkpoint_name, train_iter, model, model_opt, loss=outputs.loss
				)
		
		dd.io.save(fold_dir + "/results_dict_train.h5", train_reports)
		dd.io.save(fold_dir + "/results_dict.h5", report_all)
		dd.io.save(fold_dir + "/results_dict_val.h5", report_all_val)

	# always save final model
	save_checkpoint(checkpoint_name, train_iter, model, model_opt, loss=outputs.loss)

	
	if config.save_test_predictions:
		pass
		# print("Starting to make model predictions on test sets for *final model*.")
		# for chunk_len in [5, 100]:
		#     start_t_preds = time.time()
		#     data_config = SimpleNamespace(
		#         **{
		#             **config.__dict__["__flags"],
		#             **{"chunk_len": chunk_len, "batch_size": 500},
		#         }
		#     )
		#     dataloaders, data_name = fet.load(config.data_config, config=data_config)
		#     test_loader_preds = dataloaders["test"]

		#     torch.cuda.empty_cache()
		#     with torch.no_grad():
		#         preds = []
		#         true = []
		#         num_datapoints = 0
		#         for idx, d in enumerate(test_loader_preds):
		#             true.append(d[-1])
		#             d = nested_to(d, device, torch.float32)
		#             outputs = model(d)

		#             pred_zs = outputs.prediction
		#             preds.append(pred_zs)

		#             num_datapoints += len(pred_zs)

		#             if num_datapoints >= 2000:
		#                 break

		#         preds = torch.cat(preds, dim=0).cpu()
		#         true = torch.cat(true, dim=0).cpu()

		#         save_dir = osp.join(logdir, f"traj_preds_{chunk_len}_steps_2k_test.pt")
		#         torch.save(preds, save_dir)

		#         save_dir = osp.join(logdir, f"traj_true_{chunk_len}_steps_2k_test.pt")
		#         torch.save(true, save_dir)

		#         print(
		#             f"Completed making test predictions for chunk_len = {chunk_len} in {time.time() - start_t_preds:.2f} seconds."
		#         )	
		

if __name__ == "__main__":
	main()

