from __future__ import print_function, division
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as func
from dataset import DatasetWithIndex
from model import PretrainedModel
from loss import LossFunction
import argparse

parser = argparse.ArgumentParser(description='NNAR Training Model')
parser.add_argument('--data', type=str, default='CIFAR10', metavar='DIR', help='name of the dataset (default: CIFAR10)')
parser.add_argument('--num_classes', type=int, default=10, metavar='N', help='the number of classes in the dataset (default: 10)')
parser.add_argument('--val_size', type=int, default=500, metavar='N', help='size of the clean validation set (default: 500)')

parser.add_argument('--model_name', type=str, default='resnet', metavar='ARCH',
					help='name of the model (default: resnet), we support [resnet, alexnet, vgg, squeezenet, densenet, wide_resnet]')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to run (default: 50)')

parser.add_argument('--error_rate', type=float, default=0, metavar='N', help='rate of label error injected into the dataset (default: 0)')
parser.add_argument('--error_type', type=str, default='nar', metavar='TYPE', help='type of label error injected into the dataset (default: nar)')

parser.add_argument('--pretrained_model', type=str, default=None, metavar='DIR', help='path of the pretrained model')
args = parser.parse_args()


def load_data(input_size, model=None):
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(input_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
			transforms.Resize(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}

	print("Initializing Datasets and Dataloaders...")
	torch.manual_seed(10)
	data_dir = "../data/" + args.data

	if args.data == 'Clothing_1M':
		image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
		dataloaders_dict = {x: torch.utils.data.DataLoader(DatasetWithIndex(image_datasets[x]), batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['val', 'test']}
		dataloaders_dict = create_dataloader(image_datasets['train'], model, dataloaders_dict)
	elif args.data == 'Food_101N':
		image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
		dataloaders_dict = {x: torch.utils.data.DataLoader(DatasetWithIndex(image_datasets[x]), batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['val', 'test']}
		dataloaders_dict = create_dataloader(image_datasets['train'], model, dataloaders_dict)

	elif args.data == 'CIFAR10' or args.data == 'CIFAR100':
		if args.data == 'CIFAR10':
			cifar_train = torchvision.datasets.CIFAR10(data_dir, train=True, transform=data_transforms['train'], download=True)
			cifar_test = torchvision.datasets.CIFAR10(data_dir, train=False, transform=data_transforms['test'], download=True)
		else:
			cifar_train = torchvision.datasets.CIFAR100(data_dir, train=True, transform=data_transforms['train'], download=True)
			cifar_test = torchvision.datasets.CIFAR100(data_dir, train=False, transform=data_transforms['test'], download=True)

		cifar_train, cifar_val = torch.utils.data.random_split(cifar_train, [len(cifar_train)-args.val_size, args.val_size])

		if args.error_rate > 0 and args.error_type == 'nar':
			print('Injecting NAR Noise...', args.error_rate)
			cifar_train = inject_nar_noise(cifar_train, args.num_classes, args.error_rate)
		if args.error_rate > 0 and args.error_type == 'nnar':
			print('Injecting NNAR Noise...', args.error_rate)
			model_path = '../model/best_resnet_'+args.data.lower()+'_unweight.pth.tar'
			cifar_train = inject_nnar_noise(cifar_train, args.num_classes, args.error_rate, model_path)

		val_loader = torch.utils.data.DataLoader(DatasetWithIndex(cifar_val), batch_size=args.batch_size, shuffle=True, num_workers=4)
		test_loader = torch.utils.data.DataLoader(DatasetWithIndex(cifar_test), batch_size=args.batch_size, shuffle=True, num_workers=4)
		dataloaders_dict = {'val': val_loader, 'test': test_loader}
		dataloaders_dict = create_dataloader(cifar_train, model, dataloaders_dict)

	return dataloaders_dict


def create_dataloader(image_datasets, model, dataloaders_dict):
	train_loader = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size, shuffle=False, num_workers=4)
	preds_correct = []
	preds_incorrect = []
	batch_id = 0
	for inputs, labels in train_loader:
		inputs, labels = inputs.cuda(), labels.cuda()
		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)

		for k in range(len(labels)):
			idx = batch_id * args.batch_size + k
			if preds[k] == labels[k]:
				preds_correct.append(image_datasets[idx])
			else:
				preds_incorrect.append(image_datasets[idx])
		batch_id = batch_id + 1
	print(len(preds_correct), len(preds_incorrect))

	dataloaders_dict['train_c'] = torch.utils.data.DataLoader(DatasetWithIndex(preds_correct), batch_size=args.batch_size, shuffle=True, num_workers=4)
	dataloaders_dict['train_w'] = torch.utils.data.DataLoader(DatasetWithIndex(preds_incorrect), batch_size=args.batch_size, shuffle=True, num_workers=4)
	return dataloaders_dict


def get_trans_matrix(num_classes, error_rate, error_type = 'random', random_state=10):
	T = np.eye(num_classes)*(1-error_rate)
	random.seed(random_state)

	if error_type == 'flip_to_one':
		for i in range(num_classes):
			j = random.randint(0,num_classes-1)
			if j != i:
				T[i][j] = error_rate
			else:
				j = random.randint(0,num_classes-1)
				T[i][j] = error_rate

	elif error_type == 'uniform':
		error_rate = error_rate/(num_classes-1)
		for i in range(num_classes):
			for j in range(num_classes):
				if j != i:
					T[i][j] = error_rate

	else: #error_type == 'random'
		for i in range(num_classes):
			sum = 1-error_rate
			appear = []
			while sum < 1:
				j = random.randint(0,num_classes-1)
				if j != i and j not in appear:
					appear.append(j)
					value = round(random.uniform(0, error_rate), 2)
					if sum + value <= 1:
						T[i][j] = value
						sum = sum + value
					else:
						T[i][j] = 1 - sum
						sum = 1
	print(T)
	return T


def inject_nar_noise(dataset, num_classes, error_rate, random_state=10):
	T = get_trans_matrix(num_classes, error_rate)
	random.seed(random_state)
	datalist = list(dataset)
	for i in range(len(dataset)):
		datalist[i] = list(dataset[i])

	total = np.zeros(num_classes)
	for data in datalist: total[data[1]] = total[data[1]] + 1
	count = np.zeros(shape=(num_classes,num_classes))
	for i in range(num_classes):
		for j in range(num_classes):
			if i != j:
				count[i][j] = round(total[i] * T[i][j])

	while not np.all(count == 0):
		idx = random.randint(0,len(datalist)-1)
		cur_label = datalist[idx][1]
		for label in range(num_classes):
			if count[cur_label][label] > 0:
				count[cur_label][label] = count[cur_label][label] - 1
				datalist[idx][1] = label
				break

	return datalist


def takeSecond(elem): return elem[1]
def inject_nnar_noise(dataset, num_classes, error_rate, model_path, random_state=10):
	random.seed(random_state)
	datalist = list(dataset)
	for i in range(len(dataset)):
		datalist[i] = list(dataset[i])

	model_ft = torch.load(model_path).cuda()
	loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
	predict_proba = []
	idx = 0
	prob_list = []
	for inputs, labels in loader:
		inputs, labels = inputs.cuda(), labels.cuda()
		outputs = model_ft(inputs)
		outputs = func.softmax(outputs, dim=1)
		for i in range(len(outputs)):
			predict_proba.append(outputs[i].cpu().detach().numpy())
			prob_list.append([idx, predict_proba[idx][labels[i]]])
			idx = idx + 1
	prob_list.sort(key=takeSecond)

	error_num = int(len(datalist)*error_rate)
	print('# noisy label =', error_num)
	has_appear = []
	while len(has_appear) != error_num:
		i = random.randint(0, int(error_num*2))
		if i not in has_appear:
			has_appear.append(i)
			idx = prob_list[i][0]
			cur_label = datalist[idx][1]
			sort = np.argsort(predict_proba[idx])
			if sort[-1] != cur_label:
				datalist[idx][1] = sort[-1]
			else:
				datalist[idx][1] = sort[-2]

	return datalist


if __name__ == "__main__":
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	feature_extract = False # When False, we finetune the whole model
	pretrained_model = PretrainedModel(args.model_name, args.num_classes, feature_extract)
	input_size = pretrained_model.initialize_model(use_pretrained=True)

	model_ft = torch.load(args.pretrained_model).cuda()
	dataloaders_dict = load_data(input_size, model=model_ft)
	
	criterion = LossFunction()
	best_val_acc = float("-inf")
	best_test_acc = float("-inf")
	best_lr = float("-inf")
	best_w_lr = float("-inf")
	for lr in 10**np.linspace(-3, -1, 3):
		for w_lr in [1]:
			print('lr =', lr, 'w_lr =', w_lr)
			model_ft, best_acc = pretrained_model.train_model(dataloaders_dict, criterion, lr, w_lr, num_epochs=args.epochs, reweight=True)
			val_acc = pretrained_model.score(dataloaders_dict['val'])
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				best_lr = lr
				best_w_lr = w_lr
			if best_acc > best_test_acc:
				best_test_acc = best_acc
			pretrained_model.initialize_model(use_pretrained=True)
	print('-' * 10)
	print('best_converge_lr =', best_lr, 'best_w_lr =', best_w_lr)
	print('best_test_acc_with_early_stop = {:.4f}'.format(best_test_acc.data))
	print('\n')
	print('\n')
	
