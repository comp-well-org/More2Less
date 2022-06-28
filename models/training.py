from os import setpgid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
from configs.opts import MODEL_PATH, NUM_PRETRAINING
from utils.utils import focal_regularization
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


from torch.utils.tensorboard import SummaryWriter


def logits_2_binary(logits):
    #print(logits)
    pred = 1/( 1 + np.exp(-logits))
    threshold = 0.505
    #print(pred)
    pred[pred >= threshold] = 1;
    pred[pred < threshold] = 0;

    return pred


def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, device, num_epoches=25):


    best_acc = -0.1
    f1 = 0
    acc = 0
    lambda_ = 0.05
    #num_pretrain = 20;   # number of pre-training for a more stable training.

    curr_patience = patience = 8
    num_trials = 3


    writer = SummaryWriter()

    cos_distance = nn.CosineSimilarity(dim=1, eps=1e-6)


    step_cout =0

    for epoch in range(num_epoches):
        print("Epoch {}/{}".format(epoch, num_epoches-1));
        print('+' * 80)

        train_losses = []
        train_true_labels = []
        train_pred_labels = []

        val_losses = []
        val_true_labels = []
        val_pred_labels = []

        model.train()
        for m1, m2, labels in dataloaders['train']:
            # move data to GPU
            m1 = m1.to(device, dtype=torch.float32)
            m2 = m2.to(device, dtype=torch.float32)
            labels = labels.to(device)

            # reset optimizer.
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()

            logits_1, feats_1, logits_2, feats_2 = model(m1, m2)
            feats_1 = torch.squeeze(feats_1)   # [B, hidden_dim]
            feats_2 = torch.squeeze(feats_2)   # [B, hidden_dim]
            
            loss_1 = criterion(logits_1, labels)
            loss_1_mean = torch.mean(loss_1)
            loss_2 = criterion(logits_2, labels)
            loss_2_mean = torch.mean(loss_2)

            # if m2 perform better than m1, then transfer knowledge.
            focal_reg_param_2to1 = focal_regularization(loss_1.detach().cpu(), loss_2.detach().cpu());
            focal_reg_param_2to1 = torch.tensor(focal_reg_param_2to1).to(device)

            focal_reg_param_1to2 = focal_regularization(loss_2.detach().cpu(), loss_1.detach().cpu());
            focal_reg_param_1to2 = torch.tensor(focal_reg_param_1to2).to(device)
            # calculate the correlation among individual modalities.
            # flowing the same equation as in Eq(1)
            #corr_1 = torch.mul(feats_1,  torch.transpose(feats_1, 0, 1))
            #corr_2 = torch.mul(feats_2,  torch.transpose(feats_2, 0, 1))
            #feats_similarity = torch.sqrt(torch.sum(torch.sub(feats_1.detach(), feats_2.detach()) ** 2))
            #feats_similarity =  torch.mean(1 - cos_distance(feats_1.detach(), feats_2.detach()))

            #feats_similarity = torch.sqrt(torch.sum((feats_1.detach() - feats_2.detach()) ** 2, axis=1))/(torch.norm(feats_1.detach(), dim=1) * torch.norm(feats_2.detach(), dim=1))
            feats_similarity =  1 - cos_distance(feats_1.detach(), feats_2.detach())

            torch.clamp(feats_similarity, min=0, max=1)

            if epoch < NUM_PRETRAINING:
                loss_1_total = loss_1_mean;
            else:
                loss_1_total = loss_1_mean + lambda_ * torch.mean(focal_reg_param_2to1 * feats_similarity)


            step_cout += 1
            writer.add_scalar("Loss/loss_1", loss_1_mean, step_cout)
            writer.add_scalar("Loss/loss_1_total", loss_1_total, step_cout)
            writer.add_scalar("Loss/regularization_2to1",  torch.mean(focal_reg_param_2to1* feats_similarity), step_cout)
            # back-propagate loss
            #loss_1_total.backward(retain_graph=True)
            loss_1_total.backward()
            # update model's parameters based on loss.
            optimizer[0].step()
            
            if epoch < NUM_PRETRAINING:
                loss_2_total = loss_2_mean;
            else:
                loss_2_total = loss_2_mean + lambda_ * torch.mean(focal_reg_param_1to2 * feats_similarity)

            writer.add_scalar("Loss/loss_2", loss_2_mean, step_cout)
            writer.add_scalar("Loss/loss_2_total", loss_2_total, step_cout)
            writer.add_scalar("Loss/regularization_1to2", torch.mean(focal_reg_param_1to2 * feats_similarity), step_cout)
            if epoch < NUM_PRETRAINING:
                loss_2_total.backward()
                optimizer[1].step()


            # obtain necessary information for displaying.
            train_losses.append(loss_1_total.item())
            train_pred_labels.append(logits_1.detach().cpu())
            train_true_labels.append(labels.detach().cpu())

        lr_scheduler[0].step()
        lr_scheduler[1].step()
        all_pred = np.vstack(train_pred_labels)
        all_true = np.vstack(train_true_labels)
        # convert from one-hot coding to binary label.
        #all_pred_binary = np.argmax(all_pred, axis=1)
        #all_true_binary = np.argmax(all_true, axis=1)
        all_pred_binary = logits_2_binary(all_pred)
        all_true_binary = all_true
        # output training information after each epoch.
        print("                         Training:")
        print("Loss: %.4f" %(np.mean(np.array(train_losses))))
        F1 = f1_score(all_true_binary, all_pred_binary)
        print("F1 score: %.4f" %(F1))
        ACC = accuracy_score(all_true_binary, all_pred_binary)
        print("Accuracy: %.4f " %(ACC))
        print(confusion_matrix(all_true_binary, all_pred_binary))

        ################################################################
        ############### Val  ###########################################
        # if epoch >= num_pretrain:
        #     model.eval()
             
        #     for m1, m2, labels in dataloaders['val']:
        #         m1 = m1.to(device, dtype=torch.float32)
        #         m2 = m2.to(device, dtype=torch.float32)
        #         labels = labels.to(device)
        #         optimizer[0].zero_grad()
        #         optimizer[1].zero_grad()
        #         logits_1, feats_1, logits_2, feats_2 = model(m1, m2)

        #         loss = criterion(logits_1,labels)
        #         loss_mean = torch.mean(loss)

        #         val_losses.append(loss_mean.item())
        #         val_true_labels.append(labels.detach().cpu())
        #         val_pred_labels.append(logits_1.detach().cpu())

        #     all_pred = np.vstack(val_pred_labels)
        #     all_true = np.vstack(val_true_labels)

        #     #all_pred_binary = np.argmax(all_pred, axis=1)
        #     #all_true_binary = np.argmax(all_true, axis=1)
        #     all_pred_binary = logits_2_binary(all_pred)
        #     all_true_binary = all_true

        #     f1 = f1_score(all_true_binary, all_pred_binary)
        #     acc = accuracy_score(all_true_binary, all_pred_binary)
        #     print("#" * 40)
        #     print("                         Val:")
        #     print("Loss: %.4f" %(np.mean(np.array(val_losses))))
        #     print("F1 score: %.4f" %(f1))
        #     print("Accuracy: %.4f " %(acc))

        #     if acc > best_acc:
        #         best_acc = acc;
        #         print("Save new best model")
        #         torch.save(model.state_dict(), MODEL_PATH+'model.std')
        #         torch.save(optimizer[0].state_dict(), MODEL_PATH+'optim_0.std')
        #         torch.save(optimizer[1].state_dict(), MODEL_PATH+'optim_1.std')
        #         curr_patience = patience;
        #     else:
        #         # if the accuracy score doesnot increase for patience number
        #         # reload from previous best model, reduce learning rate
        #         # and re-train
        #         curr_patience -= 1;
        #         if curr_patience <=-1:
        #             print("Running out of patience, loading previous best model")
        #             num_trials -= 1;
        #             curr_patience = patience;
        #             model.load_state_dict(torch.load(MODEL_PATH + 'model.std'))
        #             optimizer[0].load_state_dict(torch.load(MODEL_PATH + 'optim_0.std'))
        #             optimizer[1].load_state_dict(torch.load(MODEL_PATH + 'optim_1.std'))
            
        #     if num_trials <= 0:
        #         print("Running out of patience, early stopping")
        #         break

    
    torch.save(model.state_dict(), MODEL_PATH+'model.std')

    writer.flush()
    writer.close()

    print("#"*50)
    # Test the model
    model.load_state_dict(torch.load(MODEL_PATH+'model.std'))

    test_losses = []
    test_pred_labels = []
    test_true_labels = []
    model.eval()
    for m1, m2, labels in dataloaders['test']:
        m1 = m1.to(device, dtype=torch.float32)
        m2 = m2.to(device, dtype=torch.float32)
        labels = labels.to(device)
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        # two networks (m1 and m2) will run independently, therefore  the model still can be ran with only one modality.
        # Here we use two modalities as it is easier to report performance for m1 and m2.
        # and it also equivalent to:
        #  logits_1, feats_1, _, _ = model(m1, _)
        #  _, _,  logits_2, feats_2 = model(_, m2)
        logits_1, feats_1, logits_2, feats_2 = model(m1, m2)

        loss = criterion(logits_1,labels)
        loss_mean = torch.mean(loss)

        test_losses.append(loss_mean.item())
        test_true_labels.append(labels.detach().cpu())
        test_pred_labels.append(logits_1.detach().cpu())

    all_pred = np.vstack(test_pred_labels)
    all_true = np.vstack(test_true_labels)

    #all_pred_binary = np.argmax(all_pred, axis=1)
    #all_true_binary = np.argmax(all_true, axis=1)
    all_pred_binary = logits_2_binary(all_pred)
    all_true_binary = all_true
    print("                         Testing:")
    print("Loss: %.4f" %(np.mean(np.array(test_losses))))
    print("F1 score: %.4f" %(f1_score(all_true_binary, all_pred_binary)))
    print("Accuracy: %.4f " %(accuracy_score(all_true_binary, all_pred_binary)))
    print(confusion_matrix(all_true_binary, all_pred_binary))
