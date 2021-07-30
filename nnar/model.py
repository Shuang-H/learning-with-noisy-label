import time
import copy
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable, grad
from torchvision import models
from loss import LossFunction
import torch.nn.functional as func



class PretrainedModel(object):
	def __init__(self, model_name, num_classes, feature_extract):
		super(PretrainedModel, self).__init__()
		self.model_name = model_name
		self.num_classes = num_classes
		self.feature_extract = feature_extract


	def initialize_model(self, use_pretrained=True):
		model_ft = None
		input_size = 0

		if self.model_name == "resnet":
			model_ft = models.resnet50(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, self.feature_extract)
			num_ftrs = model_ft.fc.in_features
			model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
			input_size = 224

		elif self.model_name == "alexnet":
			model_ft = models.alexnet(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, self.feature_extract)
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
			input_size = 224

		elif self.model_name == "vgg":
			model_ft = models.vgg11_bn(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, self.feature_extract)
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
			input_size = 224

		elif self.model_name == "squeezenet":
			model_ft = models.squeezenet1_0(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, self.feature_extract)
			model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
			model_ft.num_classes = self.num_classes
			input_size = 224

		elif self.model_name == "densenet":
			model_ft = models.densenet121(pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, self.feature_extract)
			num_ftrs = model_ft.classifier.in_features
			model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
			input_size = 224

		elif self.model_name == "wide_resnet":
			model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2', pretrained=use_pretrained)
			self.set_parameter_requires_grad(model_ft, self.feature_extract)
			num_ftrs = model_ft.fc.in_features
			model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
			input_size = 224

		else:
			print("Invalid model name, exiting...")
			exit()

		# data parallel
		if torch.cuda.device_count() > 1:
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			model_ft = nn.DataParallel(model_ft)
		model_ft = model_ft.cuda()
		self.model = model_ft
		return input_size


	def set_parameter_requires_grad(self, model, feature_extracting):
		if feature_extracting:
			for param in model.parameters():
				param.requires_grad = False


	def train_model(self, dataloaders, criterion, lr, w_lr, num_epochs=25, reweight=False):
		self.lr = lr
		if reweight:
			self.w_lr = w_lr
			self.w_dict = torch.tensor(np.zeros(shape=(len(dataloaders['train_w'].dataset),self.num_classes)))
			for _, labels, indexes in dataloaders['train_w']:
				for k in range(len(indexes)):
					self.w_dict[indexes[k]][labels[k]] = 1

		since = time.time()
		params_name, _ = self.get_params()
		best_val_acc = float("-inf")
		best_test_acc = float("-inf")
		final_test_acc = float("-inf")

		for epoch in range(num_epochs):
			print('**********Epoch {}/{}**********'.format(epoch+1, num_epochs))
			if (epoch+1)%10 == 0:
				self.lr = self.lr * 0.5
				if reweight: self.w_lr = self.w_lr * 0.5

			for phase in ['train_w', 'train_c', 'val', 'test']:
				if 'train' in phase:
					self.model.train()
				else:
					self.model.eval()

				running_loss = 0.0
				running_corrects = 0
				for inputs, labels, indexes in dataloaders[phase]:
					inputs, labels = inputs.cuda(), labels.cuda()
					# forward
					outputs = self.model(inputs)

					with torch.set_grad_enabled('train' in phase):
						if reweight and phase == 'train_w':
							#obtain the weights of this batch
							weights = Variable(torch.tensor(np.zeros(shape=(len(inputs),self.num_classes))), requires_grad=True)
							for k in range(len(indexes)):
								weights[k] = weights[k] + self.w_dict[indexes[k]]
							weights = weights.cuda()

							loss = criterion(outputs, labels, weights)
						else:
							loss = criterion(outputs, labels)

						_, preds = torch.max(outputs, 1)
						# backward + optimize only if in training phase
						if 'train' in phase:
							self.model.zero_grad()
							param_grads = torch.autograd.grad(outputs=loss, inputs=self.model.parameters(), create_graph=True)
							param_list = self.update_params(params_name, self.model.parameters(), param_grads, phase == 'train_w')
							if reweight and phase == 'train_w':
								X_val, y_val, _ = iter(dataloaders['val']).next()
								X_val, y_val = X_val.cuda(), y_val.cuda()
								self.compute_w(inputs, labels, indexes, X_val, y_val, param_list, weights)

					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)

				epoch_loss = running_loss / len(dataloaders[phase].dataset)
				epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
				print('Total {} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))
				if phase == 'val' and epoch_acc > best_val_acc:
					best_val_acc = epoch_acc
					best_test_acc = self.score(dataloaders['test'])
				if phase == 'test':
					final_test_acc = epoch_acc

		time_elapsed = time.time() - since
		print('-' * 10)
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		print('Best test accuracy is {:.4f}'.format(best_test_acc.data))
		print('Final test accuracy is {:.4f}'.format(final_test_acc.data))
		print('\n')
		return self.model, best_test_acc


	def get_params(self):
		params_name = []
		params_to_update = []
		for name, param in self.model.named_parameters():
			if param.requires_grad == True:
				params_name.append(name)
				params_to_update.append(param)
		return params_name, params_to_update


	def update_params(self, params_name, params_to_update, grads, get_list=False):
		param_list = []
		for name, param, grad in zip(params_name, params_to_update, grads):
			param_tmp = param - self.lr * grad
			if get_list: param_list.append(param_tmp)
			self.set_param(self.model, name, param_tmp)
		return param_list


	def set_param(self, model, name, param):
		if '.' in name:
			n = name.split('.')
			module_name = n[0]
			rest = '.'.join(n[1:])
			for name, mod in model.named_children():
				if module_name == name:
					self.set_param(mod, rest, param)
					break
		else:
			setattr(model, name, torch.nn.Parameter(param))


	def compute_w(self, X_train, y_train, indexes, X_val, y_val, param_list, weights):
		w_criterion = nn.CrossEntropyLoss()
		val_loss = w_criterion(self.model(X_val), y_val)
		param_grads_val = torch.autograd.grad(outputs=val_loss, inputs=self.model.parameters(), only_inputs=True)
		w_grad = torch.autograd.grad(outputs=param_list, inputs=weights, grad_outputs=param_grads_val, only_inputs=True)

		weights = weights - self.w_lr * w_grad[0]
		weights = self.normalize(weights).cpu()
		for k in range(len(indexes)):
			self.w_dict[indexes[k]] = weights.data[k]


	def normalize(self, T):
		T = torch.clamp(T, min=0)
		T = T / (torch.sum(T, axis=1, keepdims=True)+1e-9)
		return T


	def score(self, dataloaders):
		self.model.eval()
		running_corrects = 0
		with torch.no_grad():
			for inputs, labels, _ in dataloaders:
				inputs, labels = inputs.cuda(), labels.cuda()
				outputs = self.model(inputs)
				_, preds = torch.max(outputs, 1)
				running_corrects += torch.sum(preds == labels.data)
		return running_corrects.double() / len(dataloaders.dataset)
