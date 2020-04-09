import util
import shutil
import time
import datetime
import torch
import random

from model import TransformSentence
from torch.utils.tensorboard import SummaryWriter


def train_epoch(model, train_loader, criterion, optimizer, use_gpu=False):
    model.train()

    running_loss = []
    running_accuracy = []
    for i, pos in enumerate(train_loader):
        sentence = pos['sentence']  # [1, words, 4001] to [words, 4001]
        tags = pos['tags']  # [1, words] to [words]
        if use_gpu:
            sentence, tags = sentence.cuda(), tags.cuda()
        optimizer.zero_grad()
        predictions = model(sentence)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        loss = criterion(predictions, tags)
        running_loss.append(loss.item())
        running_accuracy.append(util.categorical_accuracy(predictions, tags))
        loss.backward()
        optimizer.step()

    loss = sum(running_loss) / len(train_loader)
    accuracy = sum(running_accuracy) / len(train_loader)
    return loss, accuracy


def validate(model, val_loader, optimizer, criterion):
    current_loss = []
    current_acc = []
    model.eval()

    with torch.no_grad():
        for item in val_loader:
            sentence = item['sentence']
            tags = item['tags']
            optimizer.zero_grad()

            predictions = model(sentence)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)
            accuracy = util.categorical_accuracy(predictions, tags)
            current_loss.append(loss.item())
            current_acc.append(accuracy)
    model.train()
    loss = sum(current_loss) / len(current_loss)
    accuracy = sum(current_acc) / len(current_acc)
    return loss, accuracy


def train_model(model, data_loaders, criterion, optimizer, scheduler, save_dir, num_epochs=5, use_gpu=False):
    print('Training model...')
    start = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    writer = SummaryWriter(f'logs/{datetime.datetime.now()}')
    for epoch in range(num_epochs):
        print('-' * 15)
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_start = time.time()
        train_loss, train_acc = train_epoch(model, data_loaders['train'], criterion, optimizer, use_gpu)
        train_time = round(time.time() - train_start, 2)
        print(f'Epoch Train Time: {int(train_time // 60)}m {train_time % 60}s')
        writer.add_scalar('Train loss', train_loss, epoch)
        writer.add_scalar('Train accuracy', train_acc, epoch)

        val_start = time.time()
        val_loss, val_acc = validate(model, data_loaders['validation'], optimizer, criterion)
        val_time = round(time.time() - val_start, 2)
        print(f'Epoch Validation Time: {int(val_time // 60)}m {val_time % 60}s')
        writer.add_scalar('Validation loss', val_loss, epoch)
        writer.add_scalar('Validation accuracy', val_acc, epoch)

        # save model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        save_checkpoint(save_dir, {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
        }, is_best)

        scheduler.step()

    elapsed = time.time() - start
    print(f'Training complete in {int(elapsed // 60)}m {elapsed // 60}s')
    print(f'Best val Acc: {best_acc}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()

    return model


def save_checkpoint(folder, state, is_best):
    path = f'{folder}/{datetime.datetime.now()}_checkpoint.pth'
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, f'{folder}/model_best.pth')


def test_model(model, test_loader, use_gpu):
    model.eval()
    test_start = time.time()
    all_accuracies = []
    for i, pos in enumerate(test_loader):
        sentence = pos['sentence']
        tags = pos['tags']
        if use_gpu:
            sentence, tags = sentence.cuda(), tags.cuda()

        predictions = model(sentence)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        all_accuracies.append(util.categorical_accuracy(predictions, tags))
    accuracy_total = round(sum(all_accuracies) / len(all_accuracies), 2)
    print(f'Test acc: {accuracy_total*100}%')
    test_time = time.time() - test_start
    print(f'Testing took: {test_time//60}m {test_time%60}s')


def classify_example(model, example, use_gpu):
    model.eval()
    sentence = example['sentence'].unsqueeze(0)
    if use_gpu:
        sentence = sentence.cuda()

    predictions = model(sentence)
    predictions = predictions.view(-1, predictions.shape[-1]).argmax(dim=1, keepdim=True)
    return predictions


def demonstrate(model, raw_test, hanzi, pos_map, use_gpu):
    random_sen = raw_test[random.randint(0, len(raw_test))]
    random_tokens = [w['form'] for w in list(random_sen)]
    random_pos = [w['upostag'] for w in list(random_sen)]
    pos_tagged_sen = [f'{random_tokens[i]}_{random_pos[i]}' for i in range(len(random_tokens))]
    transform = TransformSentence(hanzi, pos_map)
    item = transform({'sentence': random_tokens, 'tags': random_pos})
    prediction = classify_example(model, item, use_gpu)
    reverse_dict = {v: k for k, v in pos_map.items()}
    true_tags = item['tags']
    correct_count = 0
    for i in range(len(true_tags)):
        if true_tags[i] == prediction[i]:
            correct_count += 1
    ratio = correct_count / len(true_tags)
    tail = ''
    if ratio <= 0.2:
        tail = 'Disgusting.'
    elif 0.2 < ratio <= 0.4:
        tail = 'Eh, good enough.'
    elif 0.4 < ratio <= 0.6:
        tail = 'Not bad at all!'
    elif 0.6 < ratio <= 0.8:
        tail = "Whoa, didn't expect that to happen."
    elif 0.8 < ratio < 1:
        tail = "That was totally intentional."

    predicted_tags = [reverse_dict[idx.item()] for idx in prediction]
    predicted_sen = [f'{random_tokens[i]}_{predicted_tags[i]}' for i in range(len(random_tokens))]
    print(f"Sentence: {''.join(random_tokens)}. Wanna know what it means? Google it.\n")
    print(f"POS-tagged: {' '.join(pos_tagged_sen)}.\n")
    print(f"Tagged by our model: {' '.join(predicted_sen)}\n")
    print(f"Predicted correctly: {correct_count}/{len(true_tags)}. {tail}")
