import pdb
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from models.models import get_models
from models.convolutional_attention import ConfidenceModel
from models.utils import forward, store_models, load_models
from data.utils import load_cdp_data
from utils.anomaly_functions import *
from utils.utils import *
from visualizations.utils import dump_intermediate_images
from os import makedirs
from torchinfo import summary


def train(mode, train_loader, val_loader, lr, device, epochs, result_dir="./"):
    """Training loop which trains models according to the mode, train and validation loaders, learning rate, device and
    epochs. The model/s is/are stored in the provided path."""
    # Creating the model
    models = get_models(mode, device=device)
    optims = [Adam(model.parameters(), lr=lr) for model in models]

    # Training loop
    for model in models:
        model.train()
    makedirs(f"{result_dir}/train", exist_ok=True)
    makedirs(f"{result_dir}/val", exist_ok=True)
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss, val_loss = 0.0, 0.0
        with torch.cuda.amp.autocast(enabled=False):
            is_dump_done = False
            batch_anomaly_score = {
                "train": {"original": 0.0, "fake": 0.0},
                "val": {"original": 0.0, "fake": 0.0},
            }
            for batch in train_loader:
                t = batch["template"].to(device)
                x = batch["originals"][0].to(device)
                f = batch["fakes"][0].to(device)
                model_output = forward(mode, models, t, x)
                batch_loss = model_output[0]
                anomaly_map_original = get_anomaly_map(
                    t.cpu().detach(),
                    model_output[1].cpu().detach(),
                    x.cpu().detach(),
                )
                anomaly_map_fake = get_anomaly_map(
                    t.cpu().detach(),
                    model_output[1].cpu().detach(),
                    f.cpu().detach(),
                )
                batch_anomaly_score["train"]["original"] += float(
                    torch.sum(anomaly_map_original, dim=[1, 2, 3]).mean().item()
                ) / len(train_loader)
                batch_anomaly_score["train"]["fake"] += float(
                    torch.sum(anomaly_map_fake, dim=[1, 2, 3]).mean().item()
                ) / len(train_loader)

                if not is_dump_done:
                    dump_intermediate_images(
                        t.cpu().detach(),
                        x.cpu().detach(),
                        model_output[1].cpu().detach(),
                        f.cpu().detach(),
                        anomaly_map_original,
                        anomaly_map_fake,
                        epoch,
                        f"{result_dir}/train/dump_{epoch}.pdf",
                    )
                    is_dump_done = True

                for optim in optims:
                    optim.zero_grad()

                batch_loss.backward()

                for optim in optims:
                    optim.step()

                epoch_loss += batch_loss.item() / len(train_loader)
                del x, t, f, model_output, anomaly_map_fake, anomaly_map_original

            is_dump_done = False
            for batch in val_loader:
                t = batch["template"].to(device)
                x = batch["originals"][0].to(device)
                f = batch["fakes"][0].to(device)

                model_output = forward(mode, models, t, x)
                batch_loss = model_output[0]
                anomaly_map_original = get_anomaly_map(
                    t.cpu().detach(),
                    model_output[1].cpu().detach(),
                    x.cpu().detach(),
                )
                anomaly_map_fake = get_anomaly_map(
                    t.cpu().detach(),
                    model_output[1].cpu().detach(),
                    f.cpu().detach(),
                )
                batch_anomaly_score["val"]["original"] += float(
                    torch.sum(anomaly_map_original, dim=[1, 2, 3]).mean().item()
                ) / len(val_loader)
                batch_anomaly_score["val"]["fake"] += float(
                    torch.sum(anomaly_map_fake, dim=[1, 2, 3]).mean().item()
                ) / len(val_loader)
                if not is_dump_done:
                    dump_intermediate_images(
                        t.cpu().detach(),
                        x.cpu().detach(),
                        model_output[1].cpu().detach(),
                        f.cpu().detach(),
                        anomaly_map_original,
                        anomaly_map_fake,
                        epoch,
                        f"{result_dir}/val/dump_{epoch}.pdf",
                    )
                    is_dump_done = True

                val_loss += batch_loss.item() / len(val_loader)
                del x, t, f, model_output, anomaly_map_fake, anomaly_map_original

        epoch_str = (
            f"Epoch {epoch + 1}/{epochs}\tTrain loss: {epoch_loss:.5f}\tVal loss: {val_loss:.5f}"
            f"\tTrain original anomaly {batch_anomaly_score['train']['original']:.0f}"
            f"\tTrain fake anomaly {batch_anomaly_score['train']['fake']:.0f}"
            f"\tVal original anomaly {batch_anomaly_score['val']['original']:.0f}"
            f"\tVal fake anomaly {batch_anomaly_score['val']['fake']:.0f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            store_models(mode, models, result_dir)
            epoch_str += " --> Stored best model(s)"
        print(epoch_str)


def train_self_attention(
    mode,
    train_loader,
    val_loader,
    device,
    batch_size,
    epochs,
    title=None,
    result_dir="./",
    o_names=None,
    f_names=None,
):
    models = load_models(mode, result_dir, "cpu")
    for m in models:
        m.eval()
    train_samples = []
    val_samples = []
    for split, samples in zip([train_loader, val_loader], [train_samples, val_samples]):
        for batch in split:
            t = batch["template"]
            x = batch["originals"][0]
            f = batch["fakes"][0]
            model_output = forward(mode, models, t, x)
            x_hat = model_output[1]
            samples.append(torch.abs(x_hat - x).detach())
    train_samples = torch.concat(train_samples)
    val_samples = torch.concat(val_samples)

    del models, val_loader, train_loader
    attention_train_loader = DataLoader(
        TensorDataset(train_samples), batch_size=batch_size, shuffle=False
    )
    attention_val_loader = DataLoader(
        TensorDataset(train_samples), batch_size=batch_size, shuffle=False
    )

    attention_model = ConfidenceModel(kernel_size=7).to(device)
    optim = torch.optim.Adam(attention_model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        epoch_loss = {"train": 0.0, "val": 0.0}
        for split, phase in zip(
            [attention_train_loader, attention_val_loader], ["train", "val"]
        ):
            if phase == "train":
                attention_model.train()
            else:
                attention_model.eval()
            optim.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                for (anomaly_map,) in split:
                    anomaly_map = anomaly_map.to(device)
                    anomaly_score = attention_model(anomaly_map)
                    anomaly_score = anomaly_score.mean()
                    if phase == "train":
                        anomaly_score.backward()
                        optim.step()
                    epoch_loss[phase] += anomaly_score.item() / len(split)
                    del anomaly_map
        epoch_str = f"Epoch {epoch + 1}/{epochs}\tTrain loss: {epoch_loss['train']:.5f}\tVal loss: {epoch_loss['val']:.5f}"
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     store_models(mode, models, result_dir)
        #     epoch_str += " --> Stored best model(s)"
        print(epoch_str)


def test(
    mode, test_loader, device, title=None, result_dir="./", o_names=None, f_names=None
):
    """Testing loop which"""
    models = load_models(mode, result_dir, device)

    for model in models:
        model.eval()
    o_scores = [[] for _ in range(len(test_loader.dataset.x_dirs))]
    f_scores = [[] for _ in range(len(test_loader.dataset.f_dirs))]

    with torch.no_grad():
        for batch in test_loader:
            t = batch["template"].to(device)
            x = batch["originals"][0].to(device)
            for idx, x in enumerate(batch["originals"]):
                x = x.to(device)
                o_scores[idx].extend(get_anomaly_score(mode, models, t, x))

            for idx, f in enumerate(batch["fakes"]):
                f = f.to(device)
                f_scores[idx].extend(get_anomaly_score(mode, models, t, f))

        store_scores(o_scores, f_scores, result_dir)
        store_hist_picture(o_scores, f_scores, result_dir, title, o_names, f_names)


def main():
    # Parameters
    args = parse_args()
    mode = args[MODE]
    n_epochs = args[EPOCHS]
    result_dir = args[RESULT_DIR]
    lr = args[LR]
    bs = args[BS]
    tp = args[TP]
    vp = args[VP]
    no_train = args[NO_TRAIN]
    seed = args[SEED]
    o_names = args[ORIG_NAMES]
    f_names = args[FAKE_NAMES]
    is_mobile_dataset = args.get(IS_MOBILE_DATASET, False)
    train_attention = args.get(TRAIN_ATTENTION, False)
    n_epochs_self_attention = args[EPOCHS_SELF_ATTENTION]
    bs_self_attention = args[BS_SELF_ATTENTION]
    print(args)

    # Setting reproducibility
    set_reproducibility(seed)

    # Creating result directory
    create_dir(result_dir)

    # Getting program device
    device = get_device()

    # Loading data
    train_loader, val_loader, test_loader, _ = load_cdp_data(
        args, tp, vp, bs, is_mobile_dataset=is_mobile_dataset
    )

    # Training new model(s) is result directory does not exist
    if not no_train:
        print(f"Training new models.")

        # Training loop
        train(mode, train_loader, val_loader, lr, device, n_epochs, result_dir)
    if train_attention:
        train_self_attention(
            mode,
            train_loader,
            val_loader,
            device,
            bs_self_attention,
            n_epochs_self_attention,
            result_dir=result_dir,
        )

    # Testing loop
    print(f"\n\nTesting trained model(s)")
    test(
        mode,
        test_loader,
        device,
        f"Results with mode ({mode})",
        result_dir,
        o_names,
        f_names,
    )

    # Notifying program has finished
    print(f"\nProgram completed successfully. Results are available at {result_dir}")


if __name__ == "__main__":
    main()
