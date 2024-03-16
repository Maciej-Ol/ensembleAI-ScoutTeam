def main():
    args = parse_args()
    stealing_model = build_stealing_model(args)
    train_loader = build_train_loader(args)
    optimizer = build_optimizer(args)
    loss_func = build_loss_func(args)

    train

def parse_args():
    pass

def build_stealing_model(args):
    pass

def build_train_loader(args):
    pass

def build_optimizer(args):
    pass

def build_loss_func(args):
    pass

def train(args, train_loader, stealing_model, optimizer, loss_func):
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = stealing_model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


if __name__ == "__main__":
    main()
