import torch
import os


def save_checkpoint(config, epoch, model, optimizer, best_acc, is_best, reps=None, save_path='', tag='', ext='.pth.tar'):

    # If specific save path not specified lets make it the default
    if len(save_path) < 1:
        save_path = os.path.join(config.model.root_dir, config.model.type, config.model.id, config.run_id, 'checkpoints')

    # Make the checkpoints directory
    os.makedirs(save_path, exist_ok=True)

    # Create the save dictionary
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'is_best': is_best,
        'reps': reps
        }

    # replace slash if in tag so we don't get path errors, still though the user should play nice
    tag = tag.replace('/', '-')

    # Save the dictionary, potentially with a user defined tag at the start of the filename
    if len(tag) > 0:
        if is_best:
            torch.save(save_dict, os.path.join(save_path, "best_" + tag + ext))
        else:
            torch.save(save_dict, os.path.join(save_path, tag + "_e%04d%s" % (epoch, ext)))
    else:
        if is_best:
            torch.save(save_dict, os.path.join(save_path, "best" + ext))
        else:
            torch.save(save_dict, os.path.join(save_path, "e%04d%s" % (epoch, ext)))


def load_checkpoint(config, resume_from, model, optimizer):

    # Set some default variables
    save_path = os.path.join(config.model.root_dir, config.model.type, config.model.id, config.run_id, 'checkpoints')
    start_epoch = 0
    best_acc = 0
    file_path = -1
    reps = None

    # Check we have been given a resume_from string
    if len(resume_from) > 0:
        if resume_from == 'L':  # Load Latest - from the default save path
            if os.path.isdir(save_path):
                checkpoint_files = os.listdir(save_path)
                checkpoint_files.sort()
                file_path = os.path.join(save_path, checkpoint_files[-1])
        elif resume_from == 'B':  # Load Best - from the default save path
            if os.path.isdir(save_path):
                checkpoint_files = os.listdir(save_path)
                checkpoint_files.sort()
                for checkpoint_file in checkpoint_files:
                    if checkpoint_file[:4] == "best":
                        file_path = os.path.join(save_path, checkpoint_file)
                        break

        else:  # Load Specific - from given (non-default) path
            file_path = resume_from

        if file_path == -1:
            # Tried latest or best but checkpoints dir doesn't exist yet
            print("\nLearning model from scratch\n")

        elif os.path.isfile(file_path):  # Check that it is a file that exists, if so load it
            checkpoint = torch.load(file_path)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer: # is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            is_best = checkpoint['is_best']
            if is_best:
                print("\nLoaded the best checkpoint '{}' (epoch {})\n".format(file_path, start_epoch))
            else:
                print("\nLoaded checkpoint '{}' (epoch {})\n".format(file_path, start_epoch))
            reps = checkpoint['reps']
        else:
            # if can't find file raise error
            raise FileNotFoundError("Can't find %s, maybe try latest (L) or best (B)" % file_path)

    else:
        # if we aren't given anything in resume_from then start training from scratch
        print("\nLearning model from scratch\n")

    return start_epoch, best_acc, model, optimizer, reps
