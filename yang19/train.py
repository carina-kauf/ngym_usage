"""Train networks for reproducing multi-cognitive-tasks from

Task representations in neural networks trained to perform many cognitive tasks
https://www.nature.com/articles/s41593-018-0310-2
"""

import os
import time
import torch
import torch.nn as nn

from models import get_performance
from make_environments import set_seed

def main(args, model, device, env, dataset, act_size):
    set_seed(args.seed, args.cuda)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print_step = 200
    running_loss = 0.0
    running_task_time = 0
    running_train_time = 0

    for i in range(40000):
        task_time_start = time.time()
        inputs, labels = dataset()
        running_task_time += time.time() - task_time_start
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

        train_time_start = time.time()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, _ = model(inputs)

        loss = criterion(outputs.view(-1, act_size), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_train_time += time.time() - train_time_start
        # print statistics
        running_loss += loss.item()
        if i % print_step == (print_step - 1):
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / print_step))
            running_loss = 0.0
            if True:
                print('Task/Train time {:0.1f}/{:0.1f} ms/step'.format(
                        running_task_time / print_step * 1e3,
                        running_train_time / print_step * 1e3))
                running_task_time, running_train_time = 0, 0

            perf = get_performance(model, env, device=device, num_trial=200)
            print('{:d} perf: {:0.2f}'.format(i + 1, perf))

            fname = os.path.join('files',f'seed={args.seed}_model.pt')
            torch.save(model.state_dict(), fname)

    print('Finished Training')

if __name__ == '__main__':
    main(args)
