"""
training script for classification

"""

# Imports
from __future__ import print_function

import os
import time
import copy
import pprint
import argparse

import numpy as np

import torch
import torchvision
import torch.backends.cudnn as cudnn

from config.config import config, update_config, check_config
from utils.logging.logger import initialize_logger
from utils.checkpointing import save_checkpoint, load_checkpoint

from model_definitions.initialize import initialize_model
from data_loading.initialize import initialize_dataset, initialize_sampler
from losses.initialize import initialize_loss
from callbacks.initialize import initialize_callbacks


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Classification Network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()

    return args

args = parse_args()

def train():
          # set_name,
          # model_name,
          # loss_type,
          # m, d, k, alpha,
          # n_iterations=1000,
          # net_learning_rate=0.0001,
          # cluster_learning_rate=0.001,
          # chunk_size=32,
          # refresh_clusters=50,
          # norm_clusters=False,

    # setup seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Setup the paths and dirs
    save_path = os.path.join(config.model.root_dir, config.model.type, config.model.id, config.run_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert os.path.exists(save_path), '{} does not exist'.format(save_path)

    # Setup the logger
    logger = initialize_logger(save_path=save_path, run_id=config.run_id)
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    #################### MODEL ########################
    # Load the model definition
    model, input_size, output_size = initialize_model(config=config,
                                                      model_name=config.model.type,
                                                      model_id=config.model.id)
    model = model.to(device)

    # Load model params

    # Use the GPU and Parallel it
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    cudnn.benchmark = True

    #################### DATA ########################
    # Load set and get train and test labels from datasets

    datasets = dict()
    datasets['train'] = initialize_dataset(config=config,
                                           dataset_name=config.dataset.name,
                                           dataset_id=config.dataset.id,
                                           split='train')
    datasets['val'] = initialize_dataset(config=config,
                                         dataset_name=config.dataset.name,
                                         dataset_id=config.dataset.id,
                                         split='val')

    samplers = dict()
    samplers['train'] = initialize_sampler(config=config,
                                           sampler_name=config.train.sampler,
                                           dataset=datasets['train'],
                                           split='train')
    samplers['val'] = initialize_sampler(config=config,
                                         sampler_name=config.val.sampler,
                                         dataset=datasets['val'],
                                         split='val')

    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_sampler=samplers['train'])
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_sampler=samplers['val'])

    #################### LOSSES + METRICS ######################
    # Setup losses
    losses = dict()
    losses['train'] = initialize_loss(config=config,
                                      loss_name=config.train.loss,
                                      split='train',
                                      n_classes=datasets['train'].n_categories)
    losses['val'] = initialize_loss(config=config,
                                    loss_name=config.val.loss,
                                    split='val',
                                    n_classes=datasets['val'].n_categories)

    # Setup Optimizer
    optimizer = torch.optim.Adam(params=(list(filter(lambda p: p.requires_grad, model.parameters())) + list(losses['train'].parameters())),
                                 lr=config.train.learning_rate)

    if config.run_type == 'protonets':  # TODO consider putting in a callback on epoch_end, but then need to pass lr_sch
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   gamma=config.train.lr_scheduler_gamma,
                                                   step_size=config.train.lr_scheduler_step)
    else:
        lr_scheduler = None

    ################### CALLBACKS #####################
    # Setup Callbacks
    callbacks = initialize_callbacks(config=config,
                                     model=model,
                                     datasets=datasets,
                                     samplers=samplers,
                                     dataloaders=dataloaders,
                                     losses=losses,
                                     optimizer=optimizer)
    #
    # if config.TRAIN.kmeans != 0:
    #     callbacks[0] = [callback.RepsKMeans(data=train_data_classwise, k=config.TRAIN.k, n_classes=config.dataset.NUM_CLASSES,
    #                                                 emb_size=config.EMBEDDING_SIZE, max_per_class=5,
    #                                                 frequent=config.TRAIN.kmeans, model_background=config.MODEL_BACKGROUND)]
    #
    # callbacks[1] = [callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True)]


    fit(config=config,
        model=model,
        dataloaders=dataloaders,
        losses=losses,
        optimizer=optimizer,
        callbacks=callbacks,
        lr_scheduler=lr_scheduler,
        is_inception=False)
#
#     # make list of cluster refresh if given an interval int
#     if isinstance(refresh_clusters, int):
#         refresh_clusters = list(range(0, n_iterations, refresh_clusters))
#
#     # Get initial embedding using all samples in training set
#     initial_reps = compute_all_reps(net, train_dataset, chunk_size)
#
#     # Create loss object (this stores the cluster centroids)
#     if loss_type == "magnet":
#         the_loss = MagnetLoss(train_y, k, m, d, alpha=alpha)
#
#         # Initialise the embeddings/representations/clusters
#         print("Initialising the clusters")
#         the_loss.update_clusters(initial_reps)
#
#         # Setup the optimizer
#         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=net_learning_rate)
#         optimizerb = None
#     elif loss_type == "repmet" or loss_type == "repmet2" or loss_type == "repmet3" or loss_type == "myloss1":
#         if loss_type == "repmet":
#             the_loss = RepMetLoss(train_y, k, m, d, alpha=alpha)
#         elif loss_type == "repmet2":
#             the_loss = RepMetLoss2(train_y, k, m, d, alpha=alpha)
#         elif loss_type == "repmet3":
#             the_loss = RepMetLoss3(train_y, k, m, d, alpha=alpha)
#         elif loss_type == "myloss1":
#             the_loss = MyLoss1(train_y, k, m, d, alpha=alpha)
#
#         # Initialise the embeddings/representations/clusters
#         print("Initialising the clusters")
#         the_loss.update_clusters(initial_reps)
#
#         # Setup the optimizer
#         if cluster_learning_rate < 0:
#             optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters())) + [the_loss.centroids], lr=net_learning_rate)
#             optimizerb = None
#         else:
#             optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=net_learning_rate)
#             optimizerb = torch.optim.Adam([the_loss.centroids], lr=cluster_learning_rate)
#
#     l = os.listdir(save_path)
#     if load_latest and len(l) > 1:
#         l.sort(reverse=True)
#         state = torch.load("%s/%s" % (save_path, l[1])) # ignore log.txt
#
#         print("Loading model: %s/%s" % (save_path, l[1]))
#
#         net.load_state_dict(state['state_dict'])
#         optimizer.load_state_dict(state['optimizer'])
#         if optimizerb:
#             optimizerb.load_state_dict(state['optimizerb'])
#
#         start_iteration = state['iteration']+1
#         best_acc = state['best_acc']
#         the_loss = state['the_loss'] # overwrite the loss
#         plot_sample_indexs = state['plot_sample_indexs']
#         plot_classes = state['plot_classes']
#         plot_test_sample_indexs = state['plot_test_sample_indexs']
#         plot_test_classes = state['plot_test_classes']
#         batch_losses = state['batch_losses']
#         train_accs = state['train_accs']
#         test_accs = state['test_accs']
#
#         test_acc = test_accs[0][-1]
#         train_acc = train_accs[0][-1]
#         test_acc_b = test_accs[1][-1]
#         train_acc_b = train_accs[1][-1]
#         test_acc_c = test_accs[2][-1]
#         train_acc_c = train_accs[2][-1]
#         test_acc_d = test_accs[3][-1]
#         train_acc_d = train_accs[3][-1]
#     else:
#
#         # Randomly sample the classes then the samples from each class to plot
#         plot_sample_indexs, plot_classes = get_indexs(train_y, n_plot_classes, n_plot_samples)
#         plot_test_sample_indexs, plot_test_classes = get_indexs(test_y, n_plot_classes, n_plot_samples, class_ids=plot_classes)
#
#         batch_losses = []
#         train_accs = [[], [], [], []]
#         test_accs = [[], [], [], []]
#         start_iteration = 0
#         best_acc = 0
#         test_acc = 0
#         train_acc = 0
#         test_acc_b = 0
#         train_acc_b = 0
#         test_acc_c = 0
#         train_acc_c = 0
#         test_acc_d = 0
#         train_acc_d = 0
#
#     # lets plot the initial embeddings
#     cluster_classes = the_loss.cluster_classes
#
#     # use this to get indexs (indx to match cluster classes) for class ids (plot_classes) that we are plotting
#     for i in range(len(cluster_classes)):
#         cluster_classes[i] = the_loss.unique_y[cluster_classes[i]]
#
#     cluster_indexs = []
#     for ci in range(len(the_loss.cluster_classes)):
#         if the_loss.cluster_classes[ci] in plot_classes:
#             cluster_indexs.append(ci)
#
#     if not load_latest or len(l) < 2:
#         # plot it
#         graph(initial_reps[plot_sample_indexs], train_y[plot_sample_indexs],
#               cluster_centers=ensure_numpy(the_loss.centroids)[cluster_indexs],
#               cluster_classes=the_loss.cluster_classes[cluster_indexs],
#               savepath="%s/emb-initial%s" % (plots_path, plots_ext))
#
#     # Get some sample indxs to do acc test on... compare these to the acc coming out of the batch calc
#     test_train_inds,_ = get_indexs(train_y, len(set(train_y)), 10)
#
#     # Lets setup the training loop
#     iteration = None
#     for iteration in range(start_iteration, n_iterations):
#         # Sample batch and do forward-backward
#         batch_example_inds, batch_class_inds = the_loss.gen_batch()
#
#         # Get inputs and and labels from the dataset
#         batch_x = get_inputs(train_dataset, batch_example_inds).cuda()
#         batch_y = torch.from_numpy(batch_class_inds).cuda()
#
#         # Calc the outputs (embs) and then the loss + accs
#         outputs = net(batch_x)
#         batch_loss, batch_example_losses, batch_acc = the_loss.loss(outputs, batch_y)
#
#         # Pass the gradient and update
#         optimizer.zero_grad()
#         if optimizerb:
#             optimizerb.zero_grad()
#         batch_loss.backward()
#         optimizer.step()
#         if optimizerb:
#             optimizerb.step()
#
#             if norm_clusters:
#                 # Let's also normalise those centroids [because repmet pushes them away from unit sphere] to:
#                 # Option 1: sit on the hypersphere (use norm)
#                 # g = the_loss.centroids.norm(p=2,dim=0,keepdim=True)
#                 import torch.nn.functional as F
#                 the_loss.centroids.data = F.normalize(the_loss.centroids)
#
#                 # Option 2: sit on OR within the hypersphere (divide by max [scales all evenly]))
#                 # mx, _ = the_loss.centroids.max(0)
#                 # mx, _ = mx.max(0)
#                 # the_loss.centroids.data_loading = the_loss.centroids/mx
#                 # What you wrote here doesn't work as scales axes independently...
#
#         # Just changing some types
#         batch_loss = float(ensure_numpy(batch_loss))
#         batch_example_losses = ensure_numpy(batch_example_losses)
#
#         # Update loss index
#         the_loss.update_losses(batch_example_inds, batch_example_losses)
#
#         if iteration > 0 and not iteration % calc_acc_every:
#             # calc all the accs
#             train_reps = compute_reps(net, train_dataset, test_train_inds, chunk_size)
#             test_test_inds, _ = get_indexs(test_y, len(set(test_y)), 10)
#             test_reps = compute_reps(net, test_dataset, test_test_inds, chunk_size)
#
#             test_acc = the_loss.calc_accuracy(test_reps, test_y[test_test_inds], method='simple')
#             train_acc = the_loss.calc_accuracy(train_reps, train_y[test_train_inds], method='simple')
#
#             test_acc_b = the_loss.calc_accuracy(test_reps, test_y[test_test_inds], method='magnet')
#             train_acc_b = the_loss.calc_accuracy(train_reps, train_y[test_train_inds], method='magnet')
#
#             test_acc_c = the_loss.calc_accuracy(test_reps, test_y[test_test_inds], method='repmet')
#             train_acc_c = the_loss.calc_accuracy(train_reps, train_y[test_train_inds], method='repmet')
#
#             # removed because of failed runs with out of mem errors
#             # test_acc_d = the_loss.calc_accuracy(test_reps, test_y[test_test_inds], method='unsupervised')
#             # train_acc_d = the_loss.calc_accuracy(train_reps, train_y[test_train_inds], method='unsupervised')
#
#             test_acc_d = test_acc_c
#             train_acc_d = train_acc_c
#
#             with open(save_path+'/log.txt', 'a') as f:
#                 f.write("Iteration %06d/%06d: Tr. L: %0.3f :: Batch. A: %0.3f :::: Tr. A - simple: %0.3f -- magnet: %0.3f -- repmet: %0.3f -- unsupervised: %0.3f :::: Te. A - simple: %0.3f -- magnet: %0.3f -- repmet: %0.3f -- unsupervised: %0.3f\n" % (iteration, n_iterations, batch_loss, batch_acc, train_acc, train_acc_b, train_acc_c, train_acc_d, test_acc, test_acc_b, test_acc_c, test_acc_d))
#             print("Iteration %06d/%06d: Tr. L: %0.3f :: Batch. A: %0.3f :::: Tr. A - simple: %0.3f -- magnet: %0.3f -- repmet: %0.3f -- unsupervised: %0.3f :::: Te. A - simple: %0.3f -- magnet: %0.3f -- repmet: %0.3f -- unsupervised: %0.3f" % (iteration, n_iterations, batch_loss, batch_acc, train_acc, train_acc_b, train_acc_c, train_acc_d, test_acc, test_acc_b, test_acc_c, test_acc_d))
#
#             batch_ass_ids = np.unique(the_loss.assignments[batch_example_inds])
#
#             os.makedirs("%s/batch-emb/" % plots_path, exist_ok=True)
#             os.makedirs("%s/batch-emb-all/" % plots_path, exist_ok=True)
#             os.makedirs("%s/batch-clusters/" % plots_path, exist_ok=True)
#
#             graph(ensure_numpy(outputs),
#                   train_y[batch_example_inds],
#                   cluster_centers=ensure_numpy(the_loss.centroids)[batch_ass_ids],
#                   cluster_classes=the_loss.cluster_classes[batch_ass_ids],
#                   savepath="%s/batch-emb/i%06d%s" % (plots_path, iteration, plots_ext))
#
#             graph(ensure_numpy(outputs),
#                   train_y[batch_example_inds],
#                   cluster_centers=ensure_numpy(the_loss.centroids),
#                   cluster_classes=the_loss.cluster_classes,
#                   savepath="%s/batch-emb-all/i%06d%s" % (plots_path, iteration, plots_ext))
#
#             graph(np.zeros_like(ensure_numpy(outputs)),
#                   np.zeros_like(train_y[batch_example_inds]),
#                   cluster_centers=ensure_numpy(the_loss.centroids),
#                   cluster_classes=the_loss.cluster_classes,
#                   savepath="%s/batch-clusters/i%06d%s" % (plots_path, iteration, plots_ext))
#
#         train_reps_this_iter = False
#         if iteration in refresh_clusters:
#             with open(save_path+'/log.txt', 'a') as f:
#                 f.write('Refreshing clusters')
#             print('Refreshing clusters')
#             train_reps = compute_all_reps(net, train_dataset, chunk_size=chunk_size)
#             the_loss.update_clusters(train_reps)
#
#             cluster_classes = the_loss.cluster_classes
#             train_reps_this_iter = True
#
#         # store the stats to graph at end
#         batch_losses.append(batch_loss)
#         # batch_accs.append(batch_acc)
#         train_accs[0].append(train_acc)
#         test_accs[0].append(test_acc)
#         train_accs[1].append(train_acc_b)
#         test_accs[1].append(test_acc_b)
#         train_accs[2].append(train_acc_c)
#         test_accs[2].append(test_acc_c)
#         train_accs[3].append(train_acc_d)
#         test_accs[3].append(test_acc_d)
#
#         if iteration > 0 and not iteration % plot_every:
#             #use this to get indexs (indx to match cluster classes) for class ids (plot_classes) that we are plotting
#             for i in range(len(cluster_classes)):
#                 cluster_classes[i] = the_loss.unique_y[cluster_classes[i]]
#
#             # so 1. we don't have to recalc, 2. the kmeans update occured on these reps, better graphing ...
#             # if we were to re-get with compute_reps(), batch norm and transforms could give different embeddings
#             if train_reps_this_iter:
#                 plot_train_emb = train_reps[test_train_inds]
#             else:
#                 plot_train_emb = compute_reps(net, train_dataset, test_train_inds, chunk_size=chunk_size)
#
#             plot_test_emb = compute_reps(net, test_dataset, plot_test_sample_indexs, chunk_size=chunk_size)
#
#             os.makedirs("%s/train-emb/" % plots_path, exist_ok=True)
#             os.makedirs("%s/test-emb/" % plots_path, exist_ok=True)
#             os.makedirs("%s/train-emb-all/" % plots_path, exist_ok=True)
#             os.makedirs("%s/test-emb-all/" % plots_path, exist_ok=True)
#             os.makedirs("%s/cluster-losses/" % plots_path, exist_ok=True)
#             os.makedirs("%s/cluster-counts/" % plots_path, exist_ok=True)
#
#             graph(plot_train_emb,
#                   train_y[plot_sample_indexs],
#                   cluster_centers=ensure_numpy(the_loss.centroids)[cluster_indexs],
#                   cluster_classes=the_loss.cluster_classes[cluster_indexs],
#                   savepath="%s/train-emb/i%06d%s" % (plots_path, iteration, plots_ext))
#
#             graph(plot_test_emb,
#                   test_y[plot_test_sample_indexs],
#                   cluster_centers=ensure_numpy(the_loss.centroids)[cluster_indexs],
#                   cluster_classes=the_loss.cluster_classes[cluster_indexs],
#                   savepath="%s/test-emb/i%06d%s" % (plots_path, iteration, plots_ext))
#
#             graph(plot_train_emb,
#                   # train_y[plot_sample_indexs],
#                   train_y[test_train_inds],
#                   cluster_centers=ensure_numpy(the_loss.centroids),
#                   cluster_classes=the_loss.cluster_classes,
#                   savepath="%s/train-emb-all/i%06d%s" % (plots_path, iteration, plots_ext))
#
#             graph(plot_test_emb,
#                   test_y[plot_test_sample_indexs],
#                   cluster_centers=ensure_numpy(the_loss.centroids),
#                   cluster_classes=the_loss.cluster_classes,
#                   savepath="%s/test-emb-all/i%06d%s" % (plots_path, iteration, plots_ext))
#
#             plot_smooth({'loss': batch_losses,
#                          'train acc': train_accs[0],
#                          'test acc': test_accs[0]},
#                         savepath="%s/loss_simple%s" % (plots_path, plots_ext))
#             plot_smooth({'loss': batch_losses,
#                          'train acc': train_accs[1],
#                          'test acc': test_accs[1]},
#                         savepath="%s/loss_magnet%s" % (plots_path, plots_ext))
#             plot_smooth({'loss': batch_losses,
#                          'train acc': train_accs[2],
#                          'test acc': test_accs[2]},
#                         savepath="%s/loss_repmet%s" % (plots_path, plots_ext))
#             # plot_smooth({'loss': batch_losses,
#             #              'train acc': train_accs[3],
#             #              'test acc': test_accs[3]},
#             #             savepath="%s/loss_unsupervised%s" % (plots_path, plots_ext))
#
#             plot_cluster_data(the_loss.cluster_losses,
#                               the_loss.cluster_classes,
#                               title="cluster losses",
#                               savepath="%s/cluster-losses/i%06d%s" % (plots_path, iteration, plots_ext))
#
#             cluster_counts = []
#             for c in range(len(the_loss.cluster_assignments)):
#                 cluster_counts.append(len(the_loss.cluster_assignments[c]))
#
#             plot_cluster_data(cluster_counts,
#                               the_loss.cluster_classes,
#                               title="cluster counts",
#                               savepath="%s/cluster-counts/i%06d%s" % (plots_path, iteration, plots_ext))
#
#         if iteration > 0 and not iteration % save_every:
#             if save_path:
#                 if test_acc_d > best_acc:
#                     print("Saving model (is best): %s/i%06d%s" % (save_path, iteration, '.pth'))
#                     best_acc = test_acc_d
#                 else:
#                     print("Saving model: %s/i%06d%s" % (save_path, iteration, '.pth'))
#
#                 state = {
#                     'iteration': iteration,
#                     'state_dict': net.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'acc': test_acc_d,
#                     'best_acc': best_acc,
#                     'the_loss': the_loss,
#                     'plot_sample_indexs': plot_sample_indexs,
#                     'plot_classes': plot_classes,
#                     'plot_test_sample_indexs': plot_test_sample_indexs,
#                     'plot_test_classes': plot_test_classes,
#                     'batch_losses': batch_losses,
#                     'train_accs': train_accs,
#                     'test_accs': test_accs,
#                 }
#                 if optimizerb:
#                     state['optimizerb'] = optimizerb.state_dict()
#                 torch.save(state, "%s/i%06d%s" % (save_path, iteration, '.pth'))
#
#     # END TRAINING LOOP
#
#     # Plot curves and graphs
#     plot_smooth({'loss': batch_losses,
#                  'train acc': train_accs[0],
#                  'test acc': test_accs[0]},
#                 savepath="%s/loss_simple%s" % (plots_path, plots_ext))
#     plot_smooth({'loss': batch_losses,
#                  'train acc': train_accs[1],
#                  'test acc': test_accs[1]},
#                 savepath="%s/loss_magnet%s" % (plots_path, plots_ext))
#     plot_smooth({'loss': batch_losses,
#                  'train acc': train_accs[2],
#                  'test acc': test_accs[2]},
#                 savepath="%s/loss_repmet%s" % (plots_path, plots_ext))
#     plot_smooth({'loss': batch_losses,
#                  'train acc': train_accs[3],
#                  'test acc': test_accs[3]},
#                 savepath="%s/loss_unsupervised%s" % (plots_path, plots_ext))
#
#     # Calculate and graph the final
#     final_reps = compute_reps(net, train_dataset, plot_sample_indexs, chunk_size=chunk_size)
#     graph(final_reps, train_y[plot_sample_indexs], savepath="%s/emb-final%s" % (plots_path, plots_ext))
#
#     if save_path and iteration:
#         if test_acc_d > best_acc:
#             print("Saving model (is best): %s/i%06d%s" % (save_path, iteration+1, '.pth'))
#             best_acc = test_acc_d
#         else:
#             print("Saving model: %s/i%06d%s" % (save_path, iteration+1, '.pth'))
#
#         state = {
#             'iteration': iteration,
#             'state_dict': net.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'acc': test_acc_d,
#             'best_acc': best_acc,
#             'the_loss': the_loss,
#             'plot_sample_indexs': plot_sample_indexs,
#             'plot_classes': plot_classes,
#             'plot_test_sample_indexs': plot_test_sample_indexs,
#             'plot_test_classes': plot_test_classes,
#             'batch_losses': batch_losses,
#             'train_accs': train_accs,
#             'test_accs': test_accs,
#         }
#         if optimizerb:
#             state['optimizerb'] = optimizerb.state_dict()
#         torch.save(state, "%s/i%06d%s" % (save_path, iteration+1, '.pth'))
#


def fit(config,
        model,
        dataloaders,
        losses,
        optimizer,
        callbacks,
        lr_scheduler=None,
        is_inception=False,
        resume_from='L'):

    since = time.time()

    train_loss = []
    train_acc = []

    best_state = copy.deepcopy(model.state_dict())
    best_state = model.state_dict()

    # Load Checkpoint?
    start_epoch, best_acc, model, optimizer, reps = load_checkpoint(config=config,
                                                                    resume_from=resume_from,
                                                                    model=model,
                                                                    optimizer=optimizer)

    if hasattr(losses['train'], 'reps') and reps is not None:
        losses['train'].set_reps(reps)

    for callback in callbacks['training_start']:
        callback(0, 0, 0, model, dataloaders, losses, optimizer,
                 data={},
                 stats={})

    step = start_epoch*len(dataloaders['train'])
    for epoch in range(start_epoch, config.train.epochs):
        print('Epoch {}/{}'.format(epoch, config.train.epochs - 1))
        print('-' * 10)

        for callback in callbacks['epoch_start']:
            callback(epoch, 0, step, model, dataloaders, losses, optimizer,
                     data={},
                     stats={})

        # Iterate over data.
        model.train()
        batch = 0
        for inputs, labels in dataloaders['train']:  # this gets a batch (or an episode)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # Get model outputs and calculate loss
            # backward + optimize
            # if is_inception:
            #     # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            #     outputs, aux_outputs = model(inputs)
            #     loss1 = losses(outputs, labels)
            #     loss2 = losses(aux_outputs, labels)
            #     loss = loss1 + 0.4 * loss2
            # else:
            #     outputs = model(inputs)
            #     loss = losses(input=outputs, target=labels)
            outputs = model(inputs)
            loss, sample_losses, pred, acc = losses['train'](input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            # statistics
            train_loss.append(loss.item())
            train_acc.append(acc.item())

            for callback in callbacks['batch_end']:
                callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                         data={'inputs': inputs, 'outputs': outputs, 'labels': labels},
                         stats={'Training Loss': train_loss[-1], 'Training Acc': train_acc[-1],
                                'sample_losses': sample_losses})

            batch += 1
            step += 1

        avg_loss = np.mean(train_loss[-batch:])
        avg_acc = np.mean(train_acc[-batch:])

        print('Avg Training Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
        if lr_scheduler:
            lr_scheduler.step()

        # Validation?
        if config.val.every > 0 and (epoch+1) % config.val.every == 0:

            for callback in callbacks['validation_start']:
                callback(epoch, 0, step, model, dataloaders, losses, optimizer,
                         data={},
                         stats={})

            model.eval()
            v_batch = 0
            val_loss = []
            val_acc = []
            for v_inputs, v_labels in dataloaders['val']:
                v_inputs = v_inputs.to(device)
                v_labels = v_labels.to(device)

                # with torch.set_grad_enabled(False):  # todo do we need the set grad? or does the zero handle this before the next backward call?
                # Get model outputs and calculate loss
                v_outputs = model(v_inputs)
                loss, sample_losses, pred, acc = losses['val'](input=v_outputs, target=v_labels)

                # statistics
                val_loss.append(loss.item())
                val_acc.append(acc.item())

                for callback in callbacks['validation_batch_end']:
                    callback(epoch, batch, step, model, dataloaders, losses, optimizer,  # todo should we make this v_batch?
                             data={'inputs': v_inputs, 'outputs': v_outputs, 'labels': v_labels},
                             stats={'Validation Loss': val_loss[-1], 'Validation Acc': val_acc[-1]})

                v_batch += 1

            avg_v_loss = np.mean(val_loss)
            avg_v_acc = np.mean(val_acc)

            print('Avg Validation Loss: {:.4f} Acc: {:.4f}'.format(avg_v_loss, avg_v_acc))

            # Best validation accuracy yet?
            if avg_v_acc > best_acc:
                best_acc = avg_v_acc
                # best_state = copy.deepcopy(model.state_dict())
                best_state = model.state_dict()
                save_checkpoint(config, epoch, model, optimizer, best_acc, is_best=True)

            # End of validation callbacks
            for callback in callbacks['validation_end']:
                callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                         data={'inputs': v_inputs, 'outputs': v_outputs, 'labels': v_labels},
                         stats={'Avg Validation Loss': avg_v_loss, 'Avg Validation Acc': avg_v_acc})

        # End of epoch callbacks
        for callback in callbacks['epoch_end']:
            callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                     data={'inputs': inputs, 'outputs': outputs, 'labels': labels},
                     stats={'Avg Training Loss': avg_loss, 'Avg Training Acc': avg_acc})

        # Checkpoint?
        if config.train.checkpoint_every > 0 and epoch % config.train.checkpoint_every == 0:
            if hasattr(losses['train'], 'reps'):
                reps = losses['train'].get_reps()
            else:
                reps = None
            save_checkpoint(config, epoch, model, optimizer, best_acc, reps=reps, is_best=False)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    for callback in callbacks['training_end']:
        callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                 data={},
                 stats={})

    return model, best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def main():
    from utils.debug import set_working_dir
    # set the working directory as appropriate
    set_working_dir()

    print('Called with argument:', args)
    # update config
    update_config(args.cfg)
    check_config(config)
    train()


if __name__ == '__main__':
    main()
